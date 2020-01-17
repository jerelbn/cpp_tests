/*

Homography decomposition testing
- this uses NED and camera coordinates
- Deeper understanding of the homography decomposition for vision-based control - Malis, Vargas

*/
#include "common_cpp/common.h"
#include "geometry/xform.h"

using namespace std;
using namespace Eigen;

static const quat::Quatd q_cb2c = [] {
  quat::Quatd q = quat::Quatd::from_euler(M_PI/2, 0, M_PI/2);
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
vector<Point> projectPointsToImage(const Matrix3d& K, const vector<Point>& pts_I, const xform::Xformd& x);
void imagePointMatches(const vector<Point>& pts_1, const vector<Point>& pts_2, vector<Point>& matches_1, vector<Point>& matches_2);
Matrix3d homographyFromPoints(const vector<Point>& matches_1, const vector<Point>& matches_2);
Matrix3d homographyFromGeometry(const Matrix3d& K, const xform::Xformd& x1, const xform::Xformd& x2);
Matrix3d euclideanHomography(const Matrix3d& K, const Matrix3d& G);
void decomposeEuclideanHomography(const Matrix3d& H,
                                  vector<Matrix3d, aligned_allocator<Matrix3d> >& Rs,
                                  vector<Vector3d, aligned_allocator<Vector3d> >& ts,
                                  vector<Vector3d, aligned_allocator<Vector3d> >& ns);
void eliminateInvalidSolutions(const vector<Point>& pts, const Matrix3d& K,
                               vector<Matrix3d, aligned_allocator<Matrix3d> >& Rs,
                               vector<Vector3d, aligned_allocator<Vector3d> >& ts,
                               vector<Vector3d, aligned_allocator<Vector3d> >& ns);



/*================================= MAIN =================================*/

int main(int argc, char* argv[])
{
  // Random parameters
  std::default_random_engine rng(time(0));
  std::uniform_real_distribution<double> dist(-1.0, 1.0);

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

  size_t num_iters = 1000;
  size_t num_bad_solutions = 0;
  for (size_t iter = 0; iter < num_iters; ++iter)
  {
    // Camera poses
    double p1_n = -45 + 5.0*dist(rng);
    double p1_e = 5.0*dist(rng);
    double p1_d = 5.0*dist(rng);

    double p1_r = 20.0*M_PI/180.0*dist(rng);
    double p1_p = 20.0*M_PI/180.0*dist(rng);
    double p1_y = 20.0*M_PI/180.0*dist(rng);

    double p2_n = -45 + 5.0*dist(rng);
    double p2_e = 5.0*dist(rng);
    double p2_d = 5.0*dist(rng);

    double p2_r = 20.0*M_PI/180.0*dist(rng);
    double p2_p = 20.0*M_PI/180.0*dist(rng);
    double p2_y = 20.0*M_PI/180.0*dist(rng);

    xform::Xformd x1, x2;
    x1.t() = Vector3d(p1_n, p1_e, p1_d);
    x2.t() = Vector3d(p2_n, p2_e, p2_d);
    x1.q() = quat::Quatd::from_euler(p1_r, p1_p, p1_y);
    x2.q() = quat::Quatd::from_euler(p2_r, p2_p, p2_y);

    // Planar points (NED)
    // - N x N grid within +-bound in east and down directions
    const unsigned N = 101;
    const double bound = 50;
    vector<Point> pts_I = createInertialPoints(N, bound);

    // Project points into each camera image
    vector<Point> pts_1 = projectPointsToImage(K, pts_I, x1);
    vector<Point> pts_2 = projectPointsToImage(K, pts_I, x2);

    // Compute homography from matched image points
    vector<Point> matches_1, matches_2;
    imagePointMatches(pts_1, pts_2, matches_1, matches_2);
    if (matches_1.size() < 10) continue;
    Matrix3d G = homographyFromPoints(matches_1, matches_2);

    // Convert to Euclidean homography
    Matrix3d H = euclideanHomography(K, G);

    // Decompose H into 8 possible solutions R, t, n
    vector<Matrix3d, aligned_allocator<Matrix3d> > Rs;
    vector<Vector3d, aligned_allocator<Vector3d> > ns, ts;
    decomposeEuclideanHomography(H, Rs, ts, ns);

    // Eliminate impossible solutions
    eliminateInvalidSolutions(matches_2, K, Rs, ts, ns);

    // Check that true solution is in the resulting solution set
    double tol = 0.1;
    bool solution_obtained = false;
    Matrix3d R = (q_cb2c.inverse() * x1.q().inverse() * x2.q() * q_cb2c).R();
    Vector3d t = q_cb2c.rotp(x2.q().rotp((x2.t() - x1.t()) / x1.t()(0)));
    Vector3d n = q_cb2c.rotp(x1.q().rotp(common::e1));
    for (int i = 0; i < Rs.size(); ++i)
    {
      if ((R - Rs[i]).norm() < tol && (t - ts[i]).norm() < tol && (n - ns[i]).norm() < tol)
      {
        solution_obtained = true;
        break;
      }
    }

    // Show debug output if solution is not obtained
    if (!solution_obtained)
    {
      ++num_bad_solutions;
      cout << "\n\n\nTrue solution not found!\n";
      cout << endl;
      Vector3d p2_prime = G * matches_1[0].vec3();
      p2_prime /= p2_prime(2);
      cout << "point  = " << matches_2[0].vec3().transpose() << endl;
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
  cout << "\nNumber of bad solutions found: " << num_bad_solutions << " out of " << num_iters << " iterations." << "\n\n";

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


vector<Point> projectPointsToImage(const Matrix3d& K, const vector<Point>& pts_I, const xform::Xformd& x)
{
  vector<Point> pts_img;
  for (const Point& pt_I : pts_I)
  {
    // Transform points into camera frame
    Vector3d pt_c = q_cb2c.rotp(x.transformp(pt_I.vec3()));

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


Matrix3d homographyFromPoints(const vector<Point>& matches_1, const vector<Point>& matches_2)
{
  // Build solution matrix with point matches
  unsigned N = 2 * matches_1.size();
  MatrixXd P(N,9);
  P.setZero();
  for (size_t i = 0; i < matches_1.size(); ++i)
  {
    size_t idx1 = 2 * i;
    size_t idx2 = 2 * i + 1;
    Point p1 = matches_1[i];
    Point p2 = matches_2[i];
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


Matrix3d homographyFromGeometry(const Matrix3d& K, const xform::Xformd& x1, const xform::Xformd& x2)
{
  Matrix3d R = (q_cb2c.inverse() * x1.q().inverse() * x2.q() * q_cb2c).R();
  Vector3d t = q_cb2c.rotp(x2.q().rotp((x2.t() - x1.t()) / x1.t()(0)));
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

  s11 = std::abs(s11) < 1e-6 ? 0 : s11;
  s12 = std::abs(s12) < 1e-6 ? 0 : s12;
  s13 = std::abs(s13) < 1e-6 ? 0 : s13;
  s22 = std::abs(s22) < 1e-6 ? 0 : s22;
  s23 = std::abs(s23) < 1e-6 ? 0 : s23;
  s33 = std::abs(s33) < 1e-6 ? 0 : s33;

  double M_s11 = s23*s23 - s22*s33;
  double M_s22 = s13*s13 - s11*s33;
  double M_s33 = s12*s12 - s11*s22;
  double M_s12 = s13*s23 - s12*s33;
  double M_s13 = s13*s22 - s12*s23;
  double M_s23 = s12*s13 - s11*s23;

  M_s11 = std::abs(M_s11) < 1e-6 ? 0 : M_s11;
  M_s22 = std::abs(M_s22) < 1e-6 ? 0 : M_s22;
  M_s33 = std::abs(M_s33) < 1e-6 ? 0 : M_s33;
  M_s12 = std::abs(M_s12) < 1e-6 ? 0 : M_s12;
  M_s13 = std::abs(M_s13) < 1e-6 ? 0 : M_s13;
  M_s23 = std::abs(M_s23) < 1e-6 ? 0 : M_s23;

  // Compute some common parameters
  double nu = 2.0 * sqrt(1.0 + S.trace() - M_s11 - M_s22 - M_s33);
  double rho = sqrt(2.0 + S.trace() + nu);
  double te = sqrt(2.0 + S.trace() - nu);

  // Compute possible solutions
  Vector3d na, nb, ta_star, tb_star;
  if (std::abs(s11) > std::abs(s22) && std::abs(s11) > std::abs(s33))
  {
    // Plane normal
    na = Vector3d(s11, s12+sqrt(M_s33), s13+common::sign(M_s23)*sqrt(M_s22)).normalized();
    nb = Vector3d(s11, s12-sqrt(M_s33), s13-common::sign(M_s23)*sqrt(M_s22)).normalized();

    // Translation vector in wrong frame
    ta_star = te/2.0*(common::sign(s11)*rho*nb - te*na);
    tb_star = te/2.0*(common::sign(s11)*rho*na - te*nb);
  }
  else if (std::abs(s22) > std::abs(s11) && std::abs(s22) > std::abs(s33))
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
void eliminateInvalidSolutions(const vector<Point>& pts, const Matrix3d& K,
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
    for (const auto& pt : pts)
    {
      Vector3d m = K.inverse() * pt.vec3();
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
