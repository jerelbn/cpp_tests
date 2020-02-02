/*

Homography decomposition testing
- this uses NED and camera coordinates
- Deeper understanding of the homography decomposition for vision-based control - Malis, Vargas
- NOTE: This does not handle special cases, such as pure rotation or translation along the plane normal
- NOTE: This solution degrades as the ratio of camera translation to average feature depth decreases!

*/
#include <chrono>
#include "common_cpp/common.h"
#include "common_cpp/quaternion.h"
#include "common_cpp/transform.h"

using namespace std;
using namespace chrono;
using namespace Eigen;

static const common::Quaterniond q_cb2c = common::Quaterniond::fromEuler(M_PI/2, 0, M_PI/2);

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
    Vector3d pt_c1 = q_cb2c.rotp(x1.transformp(pt_I.vec3()));
    Vector3d pt_c2 = q_cb2c.rotp(x2.transformp(pt_I.vec3()));

    // Project points into image
    Vector2d pt_img1, pt_img2;
    common::projectToImage(pt_img1, pt_c1, K);
    common::projectToImage(pt_img2, pt_c2, K);
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


Matrix3d homographyFromPoints(const vector<Match>& matches)
{
  // Build solution matrix with point matches
  unsigned N = 2 * matches.size();
  MatrixXd P(N,9);
  P.setZero();
  for (size_t i = 0; i < matches.size(); ++i)
  {
    size_t idx1 = 2 * i;
    size_t idx2 = 2 * i + 1;
    Point p1 = matches[i].p1;
    Point p2 = matches[i].p2;
    P(idx1,0) = -p1.x;
    P(idx1,1) = -p1.y;
    P(idx1,2) = -1;
    P(idx1,6) = p1.x * p2.x;
    P(idx1,7) = p1.y * p2.x;
    P(idx1,8) = p2.x;
    P(idx2,3) = -p1.x;
    P(idx2,4) = -p1.y;
    P(idx2,5) = -1;
    P(idx2,6) = p1.x * p2.y;
    P(idx2,7) = p1.y * p2.y;
    P(idx2,8) = p2.y;
  }

  // Solve for homography vector then reshape to matrix
  BDCSVD<MatrixXd> svd(P, ComputeFullV);
  Matrix<double,9,1> h = svd.matrixV().col(8);
  h /= h(8);
  Matrix3d H = Map<Matrix<double,3,3,RowMajor> >(h.data());

  return H;
}


Matrix3d homographyFromGeometry(const Matrix3d& K, const common::Transformd& x1, const common::Transformd& x2)
{
  Matrix3d R = (q_cb2c.inverse() * x1.q().inverse() * x2.q() * q_cb2c).R();
  Vector3d t = q_cb2c.rotp(x2.q().rotp((x2.p() - x1.p()) / x1.p()(0)));
  Vector3d n = q_cb2c.rotp(x1.q().rotp(common::e1));

  Matrix3d G = K * (R + t*n.transpose()) * K.inverse();
  G /= G(2,2);

  return G;
}


// Computes the Euclidean Homography such that H = R + n*t^T (eq. 3 of Mails paper)
Matrix3d euclideanHomography(const Matrix3d& K, const Matrix3d& G)
{
  Matrix3d H_hat = K.inverse() * G * K;
  Matrix3d M = H_hat.transpose() * H_hat;
  
  double m11 = M(0,0);
  double m22 = M(1,1);
  double m33 = M(2,2);
  double m12 = M(0,1);
  double m13 = M(0,2);
  double m23 = M(1,2);

  double b = -(m11 + m22 + m33);
  double c = m11*m22 + m11*m33 + m22*m33 - m12*m12 - m13*m13 - m23*m23;
  double d = m12*m12*m33 + m13*m13*m22 + m23*m23*m11 - m11*m22*m33 - 2*m12*m13*m23;

  vector<double> reals, imags;
  common::solveCubic(1.0, b, c, d, reals, imags);

  return H_hat / sqrt(reals[1]);
}


void decomposeEuclideanHomography(const Matrix3d& H,
                                  vector<Matrix3d, aligned_allocator<Matrix3d> >& Rs,
                                  vector<Vector3d, aligned_allocator<Vector3d> >& ts,
                                  vector<Vector3d, aligned_allocator<Vector3d> >& ns)
{
  // Build symmetric matrix
  // - extract components
  // - compile determinants of opposite minors
  // - null close to zero values
  Matrix3d S = H.transpose() * H - common::I_3x3;
  double s11 = S(0,0);
  double s12 = S(0,1);
  double s13 = S(0,2);
  double s22 = S(1,1);
  double s23 = S(1,2);
  double s33 = S(2,2);

  s11 = abs(s11) < 1e-6 ? 0 : s11;
  s12 = abs(s12) < 1e-6 ? 0 : s12;
  s13 = abs(s13) < 1e-6 ? 0 : s13;
  s22 = abs(s22) < 1e-6 ? 0 : s22;
  s23 = abs(s23) < 1e-6 ? 0 : s23;
  s33 = abs(s33) < 1e-6 ? 0 : s33;

  double M_s11 = s23*s23 - s22*s33;
  double M_s22 = s13*s13 - s11*s33;
  double M_s33 = s12*s12 - s11*s22;
  double M_s12 = s13*s23 - s12*s33;
  double M_s13 = s13*s22 - s12*s23;
  double M_s23 = s12*s13 - s11*s23;

  M_s11 = abs(M_s11) < 1e-6 ? 0 : M_s11;
  M_s22 = abs(M_s22) < 1e-6 ? 0 : M_s22;
  M_s33 = abs(M_s33) < 1e-6 ? 0 : M_s33;
  M_s12 = abs(M_s12) < 1e-6 ? 0 : M_s12;
  M_s13 = abs(M_s13) < 1e-6 ? 0 : M_s13;
  M_s23 = abs(M_s23) < 1e-6 ? 0 : M_s23;

  // Compute some common parameters
  double nu = 2.0 * sqrt(1.0 + S.trace() - M_s11 - M_s22 - M_s33);
  double rho = sqrt(2.0 + S.trace() + nu);
  double te = sqrt(2.0 + S.trace() - nu);

  // Compute possible solutions
  Vector3d na, nb, ta_star, tb_star;
  if (abs(s11) > abs(s22) && abs(s11) > abs(s33))
  {
    // Plane normal
    na = Vector3d(s11, s12+sqrt(M_s33), s13+common::sign(M_s23)*sqrt(M_s22)).normalized();
    nb = Vector3d(s11, s12-sqrt(M_s33), s13-common::sign(M_s23)*sqrt(M_s22)).normalized();

    // Translation vector in wrong frame
    ta_star = te/2.0*(common::sign(s11)*rho*nb - te*na);
    tb_star = te/2.0*(common::sign(s11)*rho*na - te*nb);
  }
  else if (abs(s22) > abs(s11) && abs(s22) > abs(s33))
  {
    // Plane normal
    na = Vector3d(s12+sqrt(M_s33), s22, s23-common::sign(M_s13)*sqrt(M_s11)).normalized();
    nb = Vector3d(s12-sqrt(M_s33), s22, s23+common::sign(M_s13)*sqrt(M_s11)).normalized();

    // Translation vector in wrong frame
    ta_star = te/2.0*(common::sign(s22)*rho*nb - te*na);
    tb_star = te/2.0*(common::sign(s22)*rho*na - te*nb);
  }
  else
  {
    // Plane normal
    na = Vector3d(s13+common::sign(M_s12)*sqrt(M_s22), s23+sqrt(M_s11), s33).normalized();
    nb = Vector3d(s13-common::sign(M_s12)*sqrt(M_s22), s23-sqrt(M_s11), s33).normalized();

    // Translation vector in wrong frame
    ta_star = te/2.0*(common::sign(s33)*rho*nb - te*na);
    tb_star = te/2.0*(common::sign(s33)*rho*na - te*nb);
  }

  // Rotation matrix
  Matrix3d Ra = H*(common::I_3x3 - 2.0/nu*ta_star*na.transpose());
  Matrix3d Rb = H*(common::I_3x3 - 2.0/nu*tb_star*nb.transpose());

  // Translation vector
  Vector3d ta = Ra*ta_star;
  Vector3d tb = Rb*tb_star;

  // Populate output with 4 possible solution sets
  ns.clear();
  ts.clear();
  Rs.clear();

  Rs.push_back(Ra);
  ns.push_back(na);
  ts.push_back(ta);

  Rs.push_back(Ra);
  ns.push_back(-na);
  ts.push_back(-ta);

  Rs.push_back(Rb);
  ns.push_back(nb);
  ts.push_back(tb);

  Rs.push_back(Rb);
  ns.push_back(-nb);
  ts.push_back(-tb);
}


// Eliminate impossible solutions to the homography decomposition with physical constraints
void eliminateInvalidSolutions(const vector<Match>& matches, const Matrix3d& K,
                               vector<Matrix3d, aligned_allocator<Matrix3d> >& Rs,
                               vector<Vector3d, aligned_allocator<Vector3d> >& ts,
                               vector<Vector3d, aligned_allocator<Vector3d> >& ns)
{
  // Copy solutions and clear containers
  vector<Matrix3d, aligned_allocator<Matrix3d> > Rs_original = Rs;
  vector<Vector3d, aligned_allocator<Vector3d> > ts_original = ts;
  vector<Vector3d, aligned_allocator<Vector3d> > ns_original = ns;
  Rs.clear();
  ts.clear();
  ns.clear();

  // Fill output containers with physically possible solutions
  for (int i = 0; i < Rs_original.size(); ++i)
  {
    // Unpack for readability
    Matrix3d R = Rs_original[i];
    Vector3d t = ts_original[i];
    Vector3d n = ns_original[i];

    // Check that each point is in front of the camera
    bool pass = true;
    for (const auto& match : matches)
    {
      Vector3d m = K.inverse() * match.p2.vec3();
      if (m.dot(R*n) <= 0)
      {
        pass = false;
        break;
      }
    }

    // Only save solutions that put points in front of camera
    if (pass)
    {
      Rs.push_back(R);
      ts.push_back(t);
      ns.push_back(n);
    }
  }
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


  double zI_offset = 150;
  const unsigned N = 51; // number of points along single grid line
  const double pix_noise_bound = 1.0; // pixels
  const double trans_err = 5.0;
  const double rot_err = 5.0; // degrees
  const double matched_pts_inlier_ratio = 1.0;
  const double bound = zI_offset*tan(half_fov_x+rot_err*M_PI/180);

  size_t num_iters = 1000;
  size_t num_bad_iters = 0;
  double error_tol = 0.1;


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
    x1.p(Vector3d(p1_n, p1_e, p1_d));
    x2.p(Vector3d(p2_n, p2_e, p2_d));
    x1.q(common::Quaterniond::fromEuler(p1_r, p1_p, p1_y));
    x2.q(common::Quaterniond::fromEuler(p2_r, p2_p, p2_y));

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

    // Compute homography
    Matrix3d G = homographyFromPoints(matches);

    // Convert to Euclidean homography
    Matrix3d H = euclideanHomography(K, G);

    // Decompose H into 4 possible solutions R, t, n
    vector<Matrix3d, aligned_allocator<Matrix3d> > Rs;
    vector<Vector3d, aligned_allocator<Vector3d> > ns, ts;
    decomposeEuclideanHomography(H, Rs, ts, ns);

    // Eliminate impossible solutions
    eliminateInvalidSolutions(matches, K, Rs, ts, ns);

    // Check that true solution is in the resulting solution set
    bool solution_obtained = false;
    Matrix3d R = (q_cb2c.inverse() * x1.q().inverse() * x2.q() * q_cb2c).R();
    Vector3d t = q_cb2c.rotp(x2.q().rotp((x2.p() - x1.p()) / x1.p()(0)));
    Vector3d n = q_cb2c.rotp(x1.q().rotp(common::e1));
    for (int i = 0; i < Rs.size(); ++i)
    {
      if ((R - Rs[i]).norm() < error_tol && (t - ts[i]).norm() < error_tol && (n - ns[i]).norm() < error_tol)
      {
        solution_obtained = true;
        break;
      }
    }

    // Show debug output if solution is not obtained
    if (!solution_obtained)
    {
      ++num_bad_iters;
      cout << "\n\nTrue solution not found!\n\n";
      cout << "Number of matches: " << matches.size() << "\n\n";
      Vector3d p2_prime = G * matches[0].p1.vec3();
      p2_prime /= p2_prime(2);
      cout << "point  = " << matches[0].p2.vec3().transpose() << endl;
      cout << "point' = " << p2_prime.transpose() << endl;

      // Compute homography from camera geometry
      cout << endl;
      Matrix3d G2 = homographyFromGeometry(K, x1, x2);
      cout << "G = \n" << G << endl;
      cout << "G2 = \n" << G2 << endl;

      cout << endl;
      cout << "H = \n" << H << endl;
      cout << "H2 = \n" << R + t*n.transpose() << endl;

      for (int i = 0; i < Rs.size(); ++i)
      {
        cout << "Solutions:\n\n";
        cout << "R" << i << " =\n" << Rs[i] << endl;
        cout << "t" << i << " = " << ts[i].transpose() << endl;
        cout << "n" << i << " = " << ns[i].transpose() << endl;
      }

      cout << "Truth: \n\n";
      cout << "R_true = \n" << R << endl;
      cout << "t_true = " << t.transpose() << endl;
      cout << "n_true = " << n.transpose() << endl;
    }
  }
  cout << "\nNumber of bad solutions found: " << num_bad_iters << " out of " << num_iters << " iterations." << "\n\n";
  auto tf = chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - t0).count();
  cout << "Time taken: " << tf*1e-6 << " seconds" << endl;

  return 0;
}
