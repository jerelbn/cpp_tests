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

vector<Point> createInertialPoints(const unsigned& N, const double& bound);
vector<Point> projectPointsToImage(const Matrix3d& K, const vector<Point>& pts_I, const common::Transformd& x);
void imagePointMatches(const vector<Point>& pts_1, const vector<Point>& pts_2, vector<Point>& matches_1, vector<Point>& matches_2);
Matrix3d rotationFromPointsHa(const Matrix3d& K, const vector<Point>& matches_1, const vector<Point>& matches_2);
Matrix3d rotationFromPointsKneip(const Matrix3d& K, const vector<Point>& matches_1, const vector<Point>& matches_2);
Matrix3d Rcayley(const Vector3d& v);
double minEigenvalueOfM(const Vector3d& v,
                        const Matrix3d& Ax, const Matrix3d& Ay, const Matrix3d& Az,
                        const Matrix3d& Axy, const Matrix3d& Axz, const Matrix3d& Ayz);
Vector3d derivativeOfMinEigenvalueOfM(const Vector3d& v,
                                      const Matrix3d& Ax, const Matrix3d& Ay, const Matrix3d& Az,
                                      const Matrix3d& Axy, const Matrix3d& Axz, const Matrix3d& Ayz);
Matrix3d secondDerivativeOfMinEigenvalueOfM(const Vector3d& v,
                                            const Matrix3d& Ax, const Matrix3d& Ay, const Matrix3d& Az,
                                            const Matrix3d& Axy, const Matrix3d& Axz, const Matrix3d& Ayz);


/*================================= MAIN =================================*/

int main(int argc, char* argv[])
{
  // Random parameters
  auto t0 = chrono::high_resolution_clock::now();
  size_t seed = time(0);
  default_random_engine rng(seed);
  uniform_real_distribution<double> dist(-1.0, 1.0);

  // Camera parameters
  double fx = 600;
  double fy = 600;
  double cx = 320;
  double cy = 240;
  Matrix3d K = Matrix3d::Identity();
  K(0,0) = fx;
  K(1,1) = fy;
  K(0,2) = cx;
  K(1,2) = cy;
  double half_fov_x = atan(cx/fx);
  double half_fov_y = atan(cy/fy);
  double zI_offset = 1500;

  size_t num_iters = 1;
  size_t num_bad_iters = 0;
  double error_tol = 2.0; // degrees
  for (size_t iter = 0; iter < num_iters; ++iter)
  {
    cout << "Iteration: " << iter+1 << " out of " << num_iters << "\r" << flush;

    // Camera poses
    double p1_n = -zI_offset + 5.0*dist(rng);
    double p1_e = 5.0*dist(rng);
    double p1_d = 5.0*dist(rng);

    double p1_r = 5.0*M_PI/180.0*dist(rng);
    double p1_p = 5.0*M_PI/180.0*dist(rng);
    double p1_y = 5.0*M_PI/180.0*dist(rng);

    double p2_n = -zI_offset + 5.0*dist(rng);
    double p2_e = 5.0*dist(rng);
    double p2_d = 5.0*dist(rng);

    double p2_r = 5.0*M_PI/180.0*dist(rng);
    double p2_p = 5.0*M_PI/180.0*dist(rng);
    double p2_y = 5.0*M_PI/180.0*dist(rng);

    common::Transformd x1, x2;
    x1.setP(Vector3d(p1_n, p1_e, p1_d));
    x2.setP(Vector3d(p2_n, p2_e, p2_d));
    x1.setQ(common::Quaterniond(p1_r, p1_p, p1_y));
    x2.setQ(common::Quaterniond(p2_r, p2_p, p2_y));

    // Planar points (NED)
    // - N x N grid within +-bound in east and down directions
    const unsigned N = 101;
    const double bound = zI_offset*tan(half_fov_x);
    vector<Point> pts_I = createInertialPoints(N, bound);

    // Project points into each camera image
    vector<Point> pts_1 = projectPointsToImage(K, pts_I, x1);
    vector<Point> pts_2 = projectPointsToImage(K, pts_I, x2);

    // Compute rotation from matched image points
    vector<Point> matches_1; matches_1.reserve(N);
    vector<Point> matches_2; matches_2.reserve(N);
    imagePointMatches(pts_1, pts_2, matches_1, matches_2);
    if (matches_1.size() < 10) continue;
    // Matrix3d R_hat = rotationFromPointsHa(K, matches_1, matches_2);
    Matrix3d R_hat = rotationFromPointsKneip(K, matches_1, matches_2);

    // Compute true rotation and translation and rotation error
    Matrix3d R = (q_cb2c.inv() * x1.q().inv() * x2.q() * q_cb2c).R();
    Vector3d t = x2.p() - x1.p();
    double R_error = common::logR(Matrix3d(R.transpose()*R_hat)).norm()*180/M_PI; // degrees

    // Show debug output if solution is not close enough to truth
    if (R_error > error_tol)
    {
      ++num_bad_iters;
      cout << "\n\n";
      cout << "True rotation magnitude =    " << common::vex(common::logR(R)).norm()*180/M_PI << " degrees\n";
      cout << "True translation magnitude = " << t.norm() << " meters\n";
      cout << "Error =                      " << R_error << " degrees\n\n";
      cout << "R_hat =  \n" << R_hat << "\n\n";
      cout << "R_true = \n" << R << "\n\n";
    }
  }
  auto tf = chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - t0).count();
  cout << "Time taken: " << tf*1e-6 << " seconds" << endl;
  cout << "Error tolerance: " << error_tol << " degrees\n";
  cout << "Number of bad iterations: " << num_bad_iters << " out of " << num_iters << endl;

  return 0;
}



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


vector<Point> projectPointsToImage(const Matrix3d& K, const vector<Point>& pts_I, const common::Transformd& x)
{
  vector<Point> pts_img;
  for (const Point& pt_I : pts_I)
  {
    // Transform points into camera frame
    Vector3d pt_c = q_cb2c.rot(x.transform(pt_I.vec3()));

    // Project points into image
    Vector2d pt_img;
    common::projToImg(pt_img, pt_c, K);

    // Save image points inside image bounds
    if (pt_img(0) >= 0 && pt_img(1) >= 0 && pt_img(0) <= 2*K(0,2) && pt_img(1) <= 2*K(1,2))
      pts_img.push_back(Point(pt_img(0), pt_img(1), 1, pt_I.id));
  }

  return pts_img;
}


void imagePointMatches(const vector<Point>& pts_1, const vector<Point>& pts_2, vector<Point>& matches_1, vector<Point>& matches_2)
{
  // Collect points with matching ids
  matches_1.clear();
  matches_2.clear();
  for (const auto& p1 : pts_1)
  {
    for (const auto& p2 : pts_2)
    {
      if (p1.id == p2.id)
      {
        matches_1.push_back(p1);
        matches_2.push_back(p2);
        break;
      }
    }
  }
}


Matrix3d rotationFromPointsHa(const Matrix3d& K, const vector<Point>& matches_1, const vector<Point>& matches_2)
{
  // Unpack camera intrinsic parameters
  double fx = K(0,0);
  double fy = K(1,1);
  double u0 = K(0,2);
  double v0 = K(1,2);

  // Build linear system with point matches
  unsigned N = 2 * matches_1.size();
  MatrixXd A(N,3);
  MatrixXd b(N,1);
  A.setZero();
  b.setZero();
  for (size_t i = 0; i < matches_1.size(); ++i)
  {
    // Unpack points for readability
    Point p1 = matches_1[i];
    Point p2 = matches_2[i];

    // Make sure points are not aligned with camera axis
    if (std::abs(p1.x - u0) < 1.0 && std::abs(p1.y - v0) < 1.0) continue;
    if (std::abs(p2.x - u0) < 1.0 && std::abs(p2.y - v0) < 1.0) continue;

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

  // Solve for apporximate rotation vector and build rotation matrix
  Vector3d r = A.householderQr().solve(b);
  Matrix3d R = common::expR(common::skew(r));
  
  return R;
}


Matrix3d rotationFromPointsKneip(const Matrix3d& K, const vector<Point>& matches_1, const vector<Point>& matches_2)
{
  // Compute constants
  unsigned N = 2 * matches_1.size();
  Matrix3d K_inv = K.inverse();
  Matrix3d Ax = Matrix3d::Zero();
  Matrix3d Ay = Matrix3d::Zero();
  Matrix3d Az = Matrix3d::Zero();
  Matrix3d Axy = Matrix3d::Zero();
  Matrix3d Axz = Matrix3d::Zero();
  Matrix3d Ayz = Matrix3d::Zero();
  for (size_t i = 0; i < matches_1.size(); ++i)
  {
    Vector3d a = (K_inv*matches_1[i].vec3()).normalized();
    Vector3d b = (K_inv*matches_2[i].vec3()).normalized();
    Matrix3d bbT = b*b.transpose();
    Ax += a(0)*a(0)*bbT;
    Ay += a(1)*a(1)*bbT;
    Az += a(2)*a(2)*bbT;
    Axy += a(0)*a(1)*bbT;
    Axz += a(0)*a(2)*bbT;
    Ayz += a(1)*a(2)*bbT;
  }

  // Find R(v) to minimize smallest eigenvalue of M
  Vector3d v = Vector3d::Zero();
  for (int i = 0; i < 10; ++i)
  {
    Vector3d dl_dv = derivativeOfMinEigenvalueOfM(v, Ax, Ay, Az, Axy, Axz, Ayz);
    Matrix3d J = secondDerivativeOfMinEigenvalueOfM(v, Ax, Ay, Az, Axy, Axz, Ayz);
    Matrix3d A = J.transpose()*J;
    Vector3d b = -J.transpose()*dl_dv;
    Vector3d delta = A.householderQr().solve(b);

    v += delta;
  }
  
  return Rcayley(v);
}


Matrix3d Rcayley(const Vector3d& v)
{
  return (2.0*(v*v.transpose() - common::skew(v)) + (1.0 - v.transpose()*v)*common::I_3x3) / (1.0 + v.norm()*v.norm());
}


double minEigenvalueOfM(const Vector3d& v,
                        const Matrix3d& Ax, const Matrix3d& Ay, const Matrix3d& Az,
                        const Matrix3d& Axy, const Matrix3d& Axz, const Matrix3d& Ayz)
{
  // Initial guess of rotation matrix from Cayley parameters
  Matrix3d R = 2.0*(v*v.transpose() - common::skew(v)) + (1.0 - v.transpose()*v)*common::I_3x3;
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


Vector3d derivativeOfMinEigenvalueOfM(const Vector3d& v,
                                    const Matrix3d& Ax, const Matrix3d& Ay, const Matrix3d& Az,
                                    const Matrix3d& Axy, const Matrix3d& Axz, const Matrix3d& Ayz)
{
  // Initial guess of rotation matrix from Cayley parameters
  Matrix3d R = 2.0*(v*v.transpose() - common::skew(v)) + (1.0 - v.transpose()*v)*common::I_3x3;
  Vector3d r1 = R.row(0);
  Vector3d r2 = R.row(1);
  Vector3d r3 = R.row(2);

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
  if (std::abs(C.real()) < 1e-6)
    C = pow((Delta1 - root2)/2.0, 1.0/3.0);

  // Derivatives
  Matrix3d dr1_dv = 2.0*(v(0)*common::I_3x3 + v*common::e1.transpose() - common::skew(common::e1) - common::e1*v.transpose());
  Matrix3d dr2_dv = 2.0*(v(1)*common::I_3x3 + v*common::e2.transpose() - common::skew(common::e2) - common::e2*v.transpose());
  Matrix3d dr3_dv = 2.0*(v(2)*common::I_3x3 + v*common::e3.transpose() - common::skew(common::e3) - common::e3*v.transpose());

  RowVector3d dm11_dv = 2.0*((r2.transpose()*Az - r3.transpose()*Ayz)*dr2_dv + (r3.transpose()*Ay - r2.transpose()*Ayz)*dr3_dv);
  RowVector3d dm22_dv = 2.0*((r1.transpose()*Az - r3.transpose()*Axz)*dr1_dv + (r3.transpose()*Ax - r1.transpose()*Axz)*dr3_dv);
  RowVector3d dm33_dv = 2.0*((r1.transpose()*Ay - r2.transpose()*Axy)*dr1_dv + (r2.transpose()*Ax - r1.transpose()*Axy)*dr2_dv);
  RowVector3d dm12_dv = (r3.transpose()*Ayz - r2.transpose()*Az)*dr1_dv + (r3.transpose()*Axz - r1.transpose()*Az)*dr2_dv + (r1.transpose()*Ayz - 2.0*r3.transpose()*Axy + r2.transpose()*Axz)*dr3_dv;
  RowVector3d dm13_dv = (r2.transpose()*Ayz - r3.transpose()*Ay)*dr1_dv + (r2.transpose()*Axy - r1.transpose()*Ay)*dr3_dv + (r1.transpose()*Ayz - 2.0*r2.transpose()*Axz + r3.transpose()*Axy)*dr2_dv;
  RowVector3d dm23_dv = (r1.transpose()*Axz - r3.transpose()*Ax)*dr2_dv + (r1.transpose()*Axy - r2.transpose()*Ax)*dr3_dv + (r2.transpose()*Axz - 2.0*r1.transpose()*Ayz + r3.transpose()*Axy)*dr1_dv;

  RowVector3cd db_dv = -(dm11_dv + dm22_dv + dm33_dv);
  RowVector3d dc_dv = (m22 + m33)*dm11_dv + (m11 + m33)*dm22_dv + (m11 + m22)*dm33_dv - 2.0*(m12*dm12_dv + m13*dm13_dv + m23*dm23_dv);
  RowVector3d dd_dv = (m23*m23 - m22*m33)*dm11_dv + (m13*m13 - m11*m33)*dm22_dv + (m12*m12 - m11*m22)*dm33_dv +
                       2.0*((m33*m12 - m13*m23)*dm12_dv + (m22*m13 - m12*m23)*dm13_dv + (m11*m23 - m12*m13)*dm23_dv);

  RowVector3cd dDelta0_dv = 2.0*b*db_dv - 3.0*dc_dv;
  RowVector3cd dDelta1_dv = 3.0*(2.0*b*b - 3.0*c)*db_dv - 9.0*b*dc_dv + 27.0*dd_dv;

  RowVector3cd dC_dv = 1.0/6.0*pow((Delta1 + root2)/2.0,-2.0/3.0)*(dDelta1_dv + (Delta1*dDelta1_dv - 6.0*Delta0*Delta0*dDelta0_dv)/root2);
  if (std::abs(C.real() < 1e-6))
    dC_dv = 1.0/6.0*pow((Delta1 - root2)/2.0,-2.0/3.0)*(dDelta1_dv - (Delta1*dDelta1_dv - 6.0*Delta0*Delta0*dDelta0_dv)/root2);

  return (-1.0/3.0*(db_dv + dDelta0_dv/C + (1.0 - Delta0/(C*C))*dC_dv)).real();
}

Matrix3d secondDerivativeOfMinEigenvalueOfM(const Vector3d& v,
                                            const Matrix3d& Ax, const Matrix3d& Ay, const Matrix3d& Az,
                                            const Matrix3d& Axy, const Matrix3d& Axz, const Matrix3d& Ayz)
{
  static const double eps = 1e-5;
  Matrix3d ddl_ddv;
  for (int i = 0; i < 3; ++i)
  {
    Vector3d vp = v + eps * common::I_3x3.col(i);
    Vector3d vm = v - eps * common::I_3x3.col(i);
    Vector3d dlam_p = derivativeOfMinEigenvalueOfM(vp, Ax, Ay, Az, Axy, Axz, Ayz);
    Vector3d dlam_m = derivativeOfMinEigenvalueOfM(vm, Ax, Ay, Az, Axy, Axz, Ayz);
    ddl_ddv.col(i) = (dlam_p - dlam_m)/(2.0*eps);
  }
  return ddl_ddv;
}
