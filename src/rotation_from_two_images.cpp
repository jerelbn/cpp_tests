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
Matrix3d rotationFromPoints(const Matrix3d& K, const vector<Point>& matches_1, const vector<Point>& matches_2);



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

  size_t num_iters = 1000;
  size_t num_bad_iters = 0;
  double error_tol = 2.0; // degrees
  for (size_t iter = 0; iter < num_iters; ++iter)
  {
    cout << "Iteration: " << iter+1 << " out of " << num_iters << "\r" << std::flush;

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
    Matrix3d R_hat = rotationFromPoints(K, matches_1, matches_2);

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


Matrix3d rotationFromPoints(const Matrix3d& K, const vector<Point>& matches_1, const vector<Point>& matches_2)
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
    size_t idx1 = 2 * i;
    size_t idx2 = 2 * i + 1;
    Point p1 = matches_1[i];
    Point p2 = matches_2[i];
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
