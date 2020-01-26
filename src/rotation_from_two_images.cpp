/*

Rotation From Two Images - calculate only the rotation matrix from matched points in two images
- this uses NED and camera coordinates
- A Closed-Form Solution to Rotation Estimation for Structure from Small Motion - Ha, Oh, Kweon
- Direct Optimization of Frame-to-Frame Rotation - Kneip, Lynen
- NOTE: This solution degrades as the ratio of camera translation to average feature depth increases!

*/
#include <chrono>
#include "common_cpp/common.h"
#include "common_cpp/quaternion.h"
#include "common_cpp/transform.h"

using namespace std;
using namespace chrono;
using namespace Eigen;

static const common::Quaterniond q_cb2c = [] {
  common::Quaterniond q(M_PI/2, 0, M_PI/2);
  return q;
}();

struct Point
{
  Point(const double& _x, const double& _y, const double& _z, const unsigned& _id)
    : x(_x), y(_y), z(_z), id(_id) {}
  
  Vector3d vec3() const { return Vector3d(x, y, z); }

  double x;
  double y;
  double z;
  unsigned id;
};

struct Match
{
  Match(const Point& _p1, const Point& _p2)
    : p1(_p1), p2(_p2) {}
  
  Point p1, p2;
};




/*================================= FUNCTIONS =================================*/

vector<Point> createInertialPoints(const unsigned& N, const double& bound)
{
  vector<Point> pts_I;
  vector<double> pts_east, pts_down;
  for (size_t i = 0; i < N; ++i)
  {

    double pt = bound * (2 * i / double(N-1) - 1);
    pts_east.push_back(pt);
    pts_down.push_back(pt);
  }
  unsigned id = 0;
  for (const auto& pe : pts_east)
  {
    for (const auto& pd : pts_down)
    {
      pts_I.push_back(Point(0, pe, pd, id));
      ++id;
    }
  }
  
  return pts_I;
}


void projectAndMatchImagePoints(const Matrix3d& K, const vector<Point>& pts_I,
                                const common::Transformd& x1, const common::Transformd& x2,
                                const double& pix_noise_bound, const double& matched_pts_inlier_ratio,
                                vector<Match>& matches)
{
  matches.clear();
  double image_size_x = 2*K(0,2);
  double image_size_y = 2*K(1,2);
  for (const Point& pt_I : pts_I)
  {
    // Transform points into camera frame
    Vector3d pt_c1 = q_cb2c.rot(x1.transform(pt_I.vec3()));
    Vector3d pt_c2 = q_cb2c.rot(x2.transform(pt_I.vec3()));

    // Project points into image
    Vector2d pt_img1, pt_img2;
    common::projToImg(pt_img1, pt_c1, K);
    common::projToImg(pt_img2, pt_c2, K);
    pt_img1 += pix_noise_bound*Vector2d::Random();
    pt_img2 += pix_noise_bound*Vector2d::Random();

    // Save image points inside image bounds
    if (pt_img1(0) >= 0 && pt_img1(1) >= 0 && pt_img1(0) <= image_size_x && pt_img1(1) <= image_size_y &&
        pt_img2(0) >= 0 && pt_img2(1) >= 0 && pt_img2(0) <= image_size_x && pt_img2(1) <= image_size_y)
    {
      matches.push_back(Match(Point(pt_img1(0), pt_img1(1), 1, pt_I.id),
                              Point(pt_img2(0), pt_img2(1), 1, pt_I.id)));
    }
  }

  // Add outliers to matched points
  double num_inliers = matches.size();
  while (num_inliers/matches.size() > matched_pts_inlier_ratio)
  {
    Vector2d pt_img1, pt_img2;
    pt_img1(0) = image_size_x*double(rand())/RAND_MAX;
    pt_img1(1) = image_size_y*double(rand())/RAND_MAX;
    pt_img2(0) = image_size_x*double(rand())/RAND_MAX;
    pt_img2(1) = image_size_y*double(rand())/RAND_MAX;
    matches.push_back(Match(Point(pt_img1(0), pt_img1(1), 1, 999999999),
                            Point(pt_img2(0), pt_img2(1), 1, 999999999)));
  }
}


void linearSolutionHa(const Matrix3d& K, const vector<Match>& matches, common::Quaterniond& q)
{
  // Unpack camera intrinsic parameters
  double fx = K(0,0);
  double fy = K(1,1);
  double u0 = K(0,2);
  double v0 = K(1,2);

  // Build linear system with point matches
  unsigned N = 2 * matches.size();
  MatrixXd A(N,3);
  MatrixXd b(N,1);
  A.setZero();
  b.setZero();
  for (size_t i = 0; i < matches.size(); ++i)
  {
    // Unpack points for readability
    Point p1 = matches[i].p1;
    Point p2 = matches[i].p2;

    // Make sure points are not aligned with camera axis
    if (abs(p1.x - u0) < 1.0 && abs(p1.y - v0) < 1.0) continue;
    if (abs(p2.x - u0) < 1.0 && abs(p2.y - v0) < 1.0) continue;

    // Populate matrix/vector solution
    size_t idx1 = 2 * i;
    size_t idx2 = 2 * i + 1;
    A(idx1,0) = fx*u0*(p1.y - v0);
    A(idx1,1) = fy*(fx*fx - u0*p1.x + u0*u0);
    A(idx1,2) = fx*fx*(v0 - p1.y);
    A(idx2,0) = fx*(v0*p1.y - fy*fy - v0*v0);
    A(idx2,1) = fy*v0*(u0 - p1.x);
    A(idx2,2) = fy*fy*(p1.x - u0);
    b(idx1) = fx*fy*(p2.x - p1.x);
    b(idx2) = fx*fy*(p2.y - p1.y);
  }

  // Solve for apporximate rotation vector
  Vector3d r = A.householderQr().solve(b);
  
  // -r because the algorithm is based on a rotation matrix and quaternions are opposite rotations
  q = common::Quaterniond::exp(-r);
}


common::Quaterniond rotationFromPointsHa(const Matrix3d& K, const vector<Match>& matches)
{
  common::Quaterniond q;
  linearSolutionHa(K, matches, q);
  return q;
}


common::Quaterniond rotationFromPointsHaRANSAC(const Matrix3d& K, const vector<Match>& matches)
{
  static const int RANSAC_iters = 24;
  static const double RANSAC_thresh = 40.0;
  static const double RANSAC_min_inlier_ratio = 0.8;

  Matrix3d Ax, Ay, Az, Axy, Axz, Ayz;
  vector<Match> inliers, inliers_final, matches_shuffled;
  static const Matrix3d K_inv = K.inverse();
  for (int ii = 0; ii < RANSAC_iters; ++ii)
  {
    // Randomly select two point pairs
    inliers.clear();
    matches_shuffled = matches;
    std::random_shuffle(matches_shuffled.begin(), matches_shuffled.end());
    for (int jj = 0; jj < 2; ++jj)
    {
      inliers.push_back(matches_shuffled.back());
      matches_shuffled.pop_back();
    }

    // Compute solution based on two points
    common::Quaterniond q;
    linearSolutionHa(K, inliers, q);

    // Iterate through remaining point pairs
    while (matches_shuffled.size() > 0)
    {
      // Unpack points for readability
      Point p1 = matches_shuffled.back().p1;
      Point p2 = matches_shuffled.back().p2;

      // Check eprojection error
      Vector2d pixel_diff = (p2.vec3() - K*q.rot(K_inv*p1.vec3())).topRows<2>();
      if (pixel_diff.norm() < RANSAC_thresh)
      {
        // cout << endl;
        // cout << "pixel_diff.norm(): " << pixel_diff.norm() << ", ID: " << matches_shuffled.back().p1.id << endl;
        inliers.push_back(matches_shuffled.back());
      }

      // Remove point from set
      matches_shuffled.pop_back();
    }

    // Keep track of set with the most inliers
    if (inliers_final.size() < inliers.size())
      inliers_final = inliers;

    // End if minimum inlier threshold is met
    if (double(inliers.size())/matches.size() > RANSAC_min_inlier_ratio)
      break;
  }

  // Smooth final solution over all inliers
  common::Quaterniond q;
  if (inliers_final.size() > 1)
    linearSolutionHa(K, inliers_final, q);

  return q;
}


void matrixMconstants(const Matrix3d& K, const vector<Match>& matches,
                      Matrix3d& Ax,  Matrix3d& Ay,  Matrix3d& Az,
                      Matrix3d& Axy, Matrix3d& Axz, Matrix3d& Ayz)
{
  Ax.setZero();
  Ay.setZero();
  Az.setZero();
  Axy.setZero();
  Axz.setZero();
  Ayz.setZero();
  Matrix3d K_inv = K.inverse();
  for (size_t i = 0; i < matches.size(); ++i)
  {
    Vector3d a = (K_inv*matches[i].p1.vec3()).normalized();
    Vector3d b = (K_inv*matches[i].p2.vec3()).normalized();
    Matrix3d bbT = b*b.transpose();
    Ax += a(0)*a(0)*bbT;
    Ay += a(1)*a(1)*bbT;
    Az += a(2)*a(2)*bbT;
    Axy += a(0)*a(1)*bbT;
    Axz += a(0)*a(2)*bbT;
    Ayz += a(1)*a(2)*bbT;
  }
}


double minEigenvalueOfM(const common::Quaterniond& q,
                        const Matrix3d& Ax,  const Matrix3d& Ay,  const Matrix3d& Az,
                        const Matrix3d& Axy, const Matrix3d& Axz, const Matrix3d& Ayz)
{
  // Initial guess of rotation matrix from Cayley parameters
  Matrix3d R = q.R();
  Vector3d r1 = R.row(0);
  Vector3d r2 = R.row(1);
  Vector3d r3 = R.row(2);

  // Compute components of matrix M
  double m11 = r2.dot(Az*r2) + r3.dot(Ay*r3) - 2.0*r3.dot(Ayz*r2);
  double m22 = r1.dot(Az*r1) + r3.dot(Ax*r3) - 2.0*r1.dot(Axz*r3);
  double m33 = r1.dot(Ay*r1) + r2.dot(Ax*r2) - 2.0*r1.dot(Axy*r2);
  double m12 = r1.dot(Ayz*r3) - r3.dot(Axy*r3) - r1.dot(Az*r2) + r3.dot(Axz*r2);
  double m13 = r2.dot(Axy*r3) - r2.dot(Axz*r2) - r1.dot(Ay*r3) + r1.dot(Ayz*r2);
  double m23 = r1.dot(Axz*r2) - r1.dot(Ayz*r1) - r3.dot(Ax*r2) + r3.dot(Axy*r1);

  double a = 1.0;
  double b = -(m11 + m22 + m33);
  double c = m11*m22 + m11*m33 + m22*m33 - m12*m12 - m13*m13 - m23*m23;
  double d = m11*m23*m23 + m22*m13*m13 + m33*m12*m12 - 2.0*m12*m13*m23 - m11*m22*m33;

  complex<double> Delta0 = b*b - 3.0*a*c;
  complex<double> Delta1 = 2.0*b*b*b - 9.0*a*b*c + 27.0*a*a*d;

  complex<double> root2 = sqrt(pow(Delta1,2.0) - 4.0*pow(Delta0,3.0));
  complex<double> C = pow((Delta1 + root2)/2.0, 1.0/3.0);
  if (abs(C.real()) < 1e-6)
    C = pow((Delta1 - root2)/2.0, 1.0/3.0);

  return (-1.0/3.0*(b + C + Delta0/C)).real();
}


Vector3d derivativeOfMinEigenvalueOfM(const common::Quaterniond& q,
                                      const Matrix3d& Ax, const Matrix3d& Ay, const Matrix3d& Az,
                                      const Matrix3d& Axy, const Matrix3d& Axz, const Matrix3d& Ayz)
{
  // Initial guess of rotation matrix from Cayley parameters
  Matrix3d R = q.R();
  Vector3d r1 = R.col(0);
  Vector3d r2 = R.col(1);
  Vector3d r3 = R.col(2);

  // Components of minimum eigenvalue of M
  double m11 = r2.dot(Az*r2) + r3.dot(Ay*r3) - 2.0*r3.dot(Ayz*r2);
  double m22 = r1.dot(Az*r1) + r3.dot(Ax*r3) - 2.0*r1.dot(Axz*r3);
  double m33 = r1.dot(Ay*r1) + r2.dot(Ax*r2) - 2.0*r1.dot(Axy*r2);
  double m12 = r1.dot(Ayz*r3) - r3.dot(Axy*r3) - r1.dot(Az*r2) + r3.dot(Axz*r2);
  double m13 = r2.dot(Axy*r3) - r2.dot(Axz*r2) - r1.dot(Ay*r3) + r1.dot(Ayz*r2);
  double m23 = r1.dot(Axz*r2) - r1.dot(Ayz*r1) - r3.dot(Ax*r2) + r3.dot(Axy*r1);

  double a = 1.0;
  double b = -(m11 + m22 + m33);
  double c = m11*m22 + m11*m33 + m22*m33 - m12*m12 - m13*m13 - m23*m23;
  double d = m11*m23*m23 + m22*m13*m13 + m33*m12*m12 - 2.0*m12*m13*m23 - m11*m22*m33;

  complex<double> Delta0 = b*b - 3.0*a*c;
  complex<double> Delta1 = 2.0*b*b*b - 9.0*a*b*c + 27.0*a*a*d;

  complex<double> root2 = sqrt(pow(Delta1,2.0) - 4.0*pow(Delta0,3.0));
  complex<double> C = pow((Delta1 + root2)/2.0, 1.0/3.0);
  if (abs(C.real()) < 1e-6)
    C = pow((Delta1 - root2)/2.0, 1.0/3.0);

  // Derivatives
  Matrix3d dr1_dq = -common::skew(q.rot(common::e1));
  Matrix3d dr2_dq = -common::skew(q.rot(common::e2));
  Matrix3d dr3_dq = -common::skew(q.rot(common::e3));

  RowVector3d dm11_dv = 2.0*((r2.transpose()*Az - r3.transpose()*Ayz)*dr2_dq + (r3.transpose()*Ay - r2.transpose()*Ayz)*dr3_dq);
  RowVector3d dm22_dv = 2.0*((r1.transpose()*Az - r3.transpose()*Axz)*dr1_dq + (r3.transpose()*Ax - r1.transpose()*Axz)*dr3_dq);
  RowVector3d dm33_dv = 2.0*((r1.transpose()*Ay - r2.transpose()*Axy)*dr1_dq + (r2.transpose()*Ax - r1.transpose()*Axy)*dr2_dq);
  RowVector3d dm12_dv = (r3.transpose()*Ayz - r2.transpose()*Az)*dr1_dq + (r3.transpose()*Axz - r1.transpose()*Az)*dr2_dq + (r1.transpose()*Ayz - 2.0*r3.transpose()*Axy + r2.transpose()*Axz)*dr3_dq;
  RowVector3d dm13_dv = (r2.transpose()*Ayz - r3.transpose()*Ay)*dr1_dq + (r2.transpose()*Axy - r1.transpose()*Ay)*dr3_dq + (r1.transpose()*Ayz - 2.0*r2.transpose()*Axz + r3.transpose()*Axy)*dr2_dq;
  RowVector3d dm23_dv = (r1.transpose()*Axz - r3.transpose()*Ax)*dr2_dq + (r1.transpose()*Axy - r2.transpose()*Ax)*dr3_dq + (r2.transpose()*Axz - 2.0*r1.transpose()*Ayz + r3.transpose()*Axy)*dr1_dq;

  RowVector3cd db_dv = -(dm11_dv + dm22_dv + dm33_dv);
  RowVector3d dc_dv = (m22 + m33)*dm11_dv + (m11 + m33)*dm22_dv + (m11 + m22)*dm33_dv - 2.0*(m12*dm12_dv + m13*dm13_dv + m23*dm23_dv);
  RowVector3d dd_dv = (m23*m23 - m22*m33)*dm11_dv + (m13*m13 - m11*m33)*dm22_dv + (m12*m12 - m11*m22)*dm33_dv +
                       2.0*((m33*m12 - m13*m23)*dm12_dv + (m22*m13 - m12*m23)*dm13_dv + (m11*m23 - m12*m13)*dm23_dv);

  RowVector3cd dDelta0_dv = 2.0*b*db_dv - 3.0*dc_dv;
  RowVector3cd dDelta1_dv = 3.0*(2.0*b*b - 3.0*c)*db_dv - 9.0*b*dc_dv + 27.0*dd_dv;

  RowVector3cd dC_dv = 1.0/6.0*pow((Delta1 + root2)/2.0,-2.0/3.0)*(dDelta1_dv + (Delta1*dDelta1_dv - 6.0*Delta0*Delta0*dDelta0_dv)/root2);
  if (abs(C.real() < 1e-6))
    dC_dv = 1.0/6.0*pow((Delta1 - root2)/2.0,-2.0/3.0)*(dDelta1_dv - (Delta1*dDelta1_dv - 6.0*Delta0*Delta0*dDelta0_dv)/root2);

  return (-1.0/3.0*(db_dv + dDelta0_dv/C + (1.0 - Delta0/(C*C))*dC_dv)).real();
}


Matrix3d secondDerivativeOfMinEigenvalueOfM(const common::Quaterniond& q,
                                            const Matrix3d& Ax, const Matrix3d& Ay, const Matrix3d& Az,
                                            const Matrix3d& Axy, const Matrix3d& Axz, const Matrix3d& Ayz)
{
  static const double eps = 1e-6;
  Matrix3d ddl_ddq;
  for (int i = 0; i < 3; ++i)
  {
    common::Quaterniond qp = q + eps * common::I_3x3.col(i);
    common::Quaterniond qm = q + -eps * common::I_3x3.col(i);
    Vector3d dlam_p = derivativeOfMinEigenvalueOfM(qp, Ax, Ay, Az, Axy, Axz, Ayz);
    Vector3d dlam_m = derivativeOfMinEigenvalueOfM(qm, Ax, Ay, Az, Axy, Axz, Ayz);
    ddl_ddq.col(i) = (dlam_p - dlam_m)/(2.0*eps);
  }
  return ddl_ddq;
}


void kneipLM(common::Quaterniond& q, 
             const int& max_iters, const double& exit_tol, const double& lambda0,
             const double& lambda_adjust, const double& restart_variation,
             const Matrix3d& K, const vector<Match>& matches,
             const Matrix3d& Ax,  const Matrix3d& Ay,  const Matrix3d& Az,
             const Matrix3d& Axy, const Matrix3d& Axz, const Matrix3d& Ayz)
{
  common::Quaterniond q0;// = rotationFromPointsHa(K, matches);
  double lambda = lambda0;
  q = q0;

  // Find R(v) to minimize smallest eigenvalue of M
  common::Quaterniond q_new;
  Vector3d dl_dq, dl_dq_new, b, delta;
  Matrix3d J, H, A;
  bool prev_fail = false;
  for (int i = 0; i < max_iters; ++i)
  {
    // Calculate change in Cayley parameters
    if (!prev_fail)
    {
      dl_dq = derivativeOfMinEigenvalueOfM(q, Ax, Ay, Az, Axy, Axz, Ayz);
      J = secondDerivativeOfMinEigenvalueOfM(q, Ax, Ay, Az, Axy, Axz, Ayz);
      H = J.transpose()*J;
      b = -J.transpose()*dl_dq;
    }
    // A = H + lambda*Matrix3d(H.diagonal().asDiagonal());
    A = H + lambda*Matrix3d::Identity();
    delta = A.householderQr().solve(b);

    // Compute error with new parameters
    q_new = q + delta;
    dl_dq_new = derivativeOfMinEigenvalueOfM(q_new, Ax, Ay, Az, Axy, Axz, Ayz);
    if (dl_dq_new.norm() < dl_dq.norm())
    {
      q = q_new;
      lambda /= lambda_adjust;
      prev_fail = false;
    }
    else
    {
      lambda *= lambda_adjust;
      prev_fail = true;
    }

    if (delta.norm() < exit_tol) break;

    // Try restarting at random value if rotation > some threshold
    double rot_from_q0 = common::Quaterniond::log(q.inv() * q0).norm();
    if (rot_from_q0 > restart_variation)
      q = q0 + restart_variation * Vector3d::Random();
  }
}


common::Quaterniond rotationFromPointsKneip(const Matrix3d& K, const vector<Match>& matches)
{
  static const int max_iters = 100;
  static const double exit_tol = 1e-6;
  static const double lambda_adjust = 10.0;
  static const double restart_variation = 3*M_PI/180;
  static const double lambda0 = 1.0;

  // Compute constants
  Matrix3d Ax, Ay, Az, Axy, Axz, Ayz;
  matrixMconstants(K, matches, Ax, Ay, Az, Axy, Axz, Ayz);

  // Find R(q) to minimize smallest eigenvalue of M via Levenberg-Marquardt algorithm
  common::Quaterniond q;
  kneipLM(q, max_iters, exit_tol, lambda0, lambda_adjust, restart_variation,
          K, matches, Ax, Ay, Az, Axy, Axz, Ayz);

  return q; 
}


common::Quaterniond rotationFromPointsKneipRANSAC(const Matrix3d& K, const vector<Match>& matches)
{
  static const int max_iters = 100;
  static const double exit_tol = 1e-6;
  static const double lambda_adjust = 10.0;
  static const double restart_variation = 1*M_PI/180;
  static const double lambda0 = 1.0;

  static const int RANSAC_iters = 16;
  static const double RANSAC_thresh = 1e-7;
  static const double RANSAC_min_inlier_ratio = 0.7; // if this is greater than the real thing, why do I get bad answers?

  common::Quaterniond q;
  Matrix3d Ax, Ay, Az, Axy, Axz, Ayz;
  vector<Match> inliers, inliers_final, inliers_test, matches_shuffled;
  for (int ii = 0; ii < RANSAC_iters; ++ii)
  {
    // Randomly select two point pairs
    inliers.clear();
    matches_shuffled = matches;
    std::random_shuffle(matches_shuffled.begin(), matches_shuffled.end());
    inliers.push_back(matches_shuffled.back());
    matches_shuffled.pop_back();
    inliers.push_back(matches_shuffled.back());
    matches_shuffled.pop_back();

    // Compute constants
    matrixMconstants(K, inliers, Ax, Ay, Az, Axy, Axz, Ayz);

    // Find R(q) to minimize smallest eigenvalue of M via Levenberg-Marquardt algorithm
    kneipLM(q, max_iters, exit_tol, lambda0, lambda_adjust, restart_variation,
            K, inliers, Ax, Ay, Az, Axy, Axz, Ayz);

    // Iterate through remaining point pairs
    inliers_test = inliers;
    while (matches_shuffled.size() > 0)
    {
      // Add third point to inlier set
      inliers_test.push_back(matches_shuffled.back());
      matches_shuffled.pop_back();

      // Recompute constants and check derivative
      matrixMconstants(K, inliers_test, Ax, Ay, Az, Axy, Axz, Ayz);
      Vector3d dl_dq = derivativeOfMinEigenvalueOfM(q, Ax, Ay, Az, Axy, Axz, Ayz);
      double lam_min = minEigenvalueOfM(q, Ax, Ay, Az, Axy, Axz, Ayz);
      if (lam_min < RANSAC_thresh)
      // if (inliers_test.back().p1.id < 999999)
      {
        cout << endl;
        cout << "dl_dq.norm(): " << dl_dq.norm() << endl;
        cout << "  lambda_min: " << lam_min << endl;
        cout << "          ID: " << inliers_test.back().p1.id << endl;
        inliers.push_back(inliers_test.back());
      }

      // Remove the third point
      inliers_test.pop_back();
    }

    // Keep track of set with the most inliers
    if (inliers_final.size() < inliers.size())
      inliers_final = inliers;

    // End if minimum inlier threshold is met
    if (double(inliers.size())/matches.size() > RANSAC_min_inlier_ratio)
      break;
  }

  // Smooth final solution over all inliers
  matrixMconstants(K, inliers_final, Ax, Ay, Az, Axy, Axz, Ayz);
  kneipLM(q, max_iters, exit_tol, lambda0, lambda_adjust, restart_variation,
          K, inliers_final, Ax, Ay, Az, Axy, Axz, Ayz);
  cout << "\nFINAL INLIER RATIO: " << double(inliers_final.size())/matches.size() << endl;

  return q;
}




/*================================= MAIN =================================*/

int main(int argc, char* argv[])
{
  // Random parameters
  auto t0 = high_resolution_clock::now();
  size_t seed = 0;//time(0);
  default_random_engine rng(seed);
  uniform_real_distribution<double> dist(-1.0, 1.0);
  srand(seed);

  // Camera parameters
  double cx = 320;
  double cy = 240;
  double half_fov_x = 3.0*M_PI/180.0;
  double half_fov_y = cy/cx*half_fov_x;
  double fx = cx/tan(half_fov_x);
  double fy = cy/tan(half_fov_y);
  Matrix3d K = Matrix3d::Identity();
  K(0,0) = fx;
  K(1,1) = fy;
  K(0,2) = cx;
  K(1,2) = cy;


  // solver:
  // - 0: Ha solver
  // - 1: Ha solver w/ RANSAC
  // - 2: Kneip solver
  // - 3: Kneip solver w/ RANSAC
  int solver = 1;

  double zI_offset = 1000;
  const unsigned N = 31; // number of points along single grid line
  const double pix_noise_bound = 1.0; // pixels
  const double trans_err = 5.0;
  const double rot_err = 5.0;
  const double matched_pts_inlier_ratio = 0.8;
  const double bound = zI_offset*tan(half_fov_x+rot_err*M_PI/180);

  size_t num_iters = 1000;
  size_t num_bad_iters = 0;
  double error_tol = 2.0*M_PI/180;


  double dt_calc_mean = 0.0; // seconds
  double dt_calc_var = 0.0; // seconds
  double error_mean = 0.0;
  double error_var = 0.0;
  double lambda_min_mean = 0.0;
  double lambda_min_var = 0.0;
  double match_pts_mean = 0.0;
  double match_pts_var = 0.0;
  size_t n_stats = 0;
  for (size_t iter = 0; iter < num_iters; ++iter)
  {
    cout << "Iteration: " << iter+1 << " out of " << num_iters << "\r" << flush;

    // Camera poses
    double p1_n = -zI_offset + trans_err*dist(rng);
    double p1_e = trans_err*dist(rng);
    double p1_d = trans_err*dist(rng);

    double p1_r = rot_err*M_PI/180.0*dist(rng);
    double p1_p = rot_err*M_PI/180.0*dist(rng);
    double p1_y = rot_err*M_PI/180.0*dist(rng);

    double p2_n = -zI_offset + trans_err*dist(rng);
    double p2_e = trans_err*dist(rng);
    double p2_d = trans_err*dist(rng);

    double p2_r = rot_err*M_PI/180.0*dist(rng);
    double p2_p = rot_err*M_PI/180.0*dist(rng);
    double p2_y = rot_err*M_PI/180.0*dist(rng);

    common::Transformd x1, x2;
    x1.setP(Vector3d(p1_n, p1_e, p1_d));
    x2.setP(Vector3d(p2_n, p2_e, p2_d));
    x1.setQ(common::Quaterniond(p1_r, p1_p, p1_y));
    x2.setQ(common::Quaterniond(p2_r, p2_p, p2_y));

    // Planar points (NED)
    // - N x N grid within +-bound in east and down directions
    vector<Point> pts_I = createInertialPoints(N, bound);

    // Project matching points into each camera image
    vector<Match> matches;
    projectAndMatchImagePoints(K, pts_I, x1, x2, pix_noise_bound, matched_pts_inlier_ratio, matches);
    if (matches.size() < 8)
    {
      --iter; // Try that iteration again
      continue;
    }

    common::Quaterniond q_hat;
    auto t_calc_0 = high_resolution_clock::now();
    if (solver == 0)
      q_hat = rotationFromPointsHa(K, matches);
    else if (solver == 1)
      q_hat = rotationFromPointsHaRANSAC(K, matches);
    else if (solver == 2)
      q_hat = rotationFromPointsKneip(K, matches);
    else if (solver == 3)
      q_hat = rotationFromPointsKneipRANSAC(K, matches);
    else
      throw runtime_error("Select an existing solver!");
    double dt_calc = duration_cast<microseconds>(high_resolution_clock::now() - t_calc_0).count()*1e-6;

    // Compute true rotation and translation and rotation error
    common::Quaterniond q = q_cb2c.inv() * x1.q().inv() * x2.q() * q_cb2c;
    Vector3d t = x2.p() - x1.p();
    double rot_error = common::Quaterniond::log(q.inv()*q_hat).norm();

    // Show debug output if solution is not close enough to truth
    if (rot_error > error_tol)
    {
      ++num_bad_iters;
      cout << "\n\n";
      cout << "           Calc time taken: " << dt_calc << " seconds\n";
      cout << "   True rotation magnitude: " << common::Quaterniond::log(q).norm()*180/M_PI << " degrees\n";
      cout << "True translation magnitude: " << t.norm() << " meters\n";
      cout << "                     Error: " << rot_error*180/M_PI << " degrees\n";
      cout << "   Number of point matches: " << matches.size() << "\n";
      cout << "                       q_0: " << rotationFromPointsHa(K, matches).toEigen().transpose() << "\n";
      cout << "                     q_hat: " << q_hat.toEigen().transpose() << "\n";
      cout << "                    q_true: " << q.toEigen().transpose() << "\n\n";
      continue; // Bad solutions aren't useful in the following statistics
    }

    // Recursive error and variance of things
    dt_calc_mean = (n_stats*dt_calc_mean + dt_calc)/(n_stats+1);
    error_mean = (n_stats*error_mean + rot_error)/(n_stats+1);
    match_pts_mean = (n_stats*match_pts_mean + matches.size())/(n_stats+1);
    if (n_stats > 0)
    {
      dt_calc_var = ((n_stats-1)*dt_calc_var + pow(dt_calc - dt_calc_mean, 2.0))/n_stats;
      error_var = ((n_stats-1)*error_var + pow(rot_error - error_mean, 2.0))/n_stats;
      match_pts_var = ((n_stats-1)*match_pts_var + pow(matches.size() - match_pts_mean, 2.0))/n_stats;
    }
    ++n_stats;
  }
  auto tf = duration_cast<microseconds>(high_resolution_clock::now() - t0).count()*1e-6;
  cout << "        Total time taken: " << tf << " seconds\n";
  cout << "         Error tolerance: " << error_tol*180/M_PI << " degrees\n";
  cout << "Number of bad iterations: " << num_bad_iters << " out of " << num_iters << endl;
  cout << " Calc time (mean, stdev): (" << dt_calc_mean << ", " << sqrt(dt_calc_var) << ") seconds\n";
  cout << "     Error (mean, stdev): (" << error_mean*180/M_PI << ", " << sqrt(error_var)*180/M_PI << ") degrees\n";
  cout << " match_pts (mean, stdev): (" << match_pts_mean << ", " << sqrt(match_pts_var) << ")\n\n";

  return 0;
}
