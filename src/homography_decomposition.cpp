/*

Homography decomposition testing
- this uses NED and camera coordinates

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
Matrix3d computeHomographyFromPoints(const vector<Point>& matches_1, const vector<Point>& matches_2);
Matrix3d computeHomographyFromRTN(const Matrix3d& R, const Vector3d& t, const Vector3d& n);

// Create points on a plane
// Define two camera poses looking at the points
// Project the points into each camera image
// Calculate homography using the image points (verify correctness by transforming image points)
// Calculate homography using camera rotation and translation
// Compare homographies
// Decompose homography into 8 solutions of R/t using Malis paper
// Reduce possible solutions from 8 to 2 using Malis paper
// Explore possible methods to choose between final two solutions, assuming they are not too similar

int main(int argc, char* argv[])
{
  // Camera parameters
  double fx = 800;
  double fy = 800;
  double cx = 320;
  double cy = 240;
  Matrix3d K = Matrix3d::Identity();
  K(0,0) = fx;
  K(1,1) = fy;
  K(0,2) = cx;
  K(1,2) = cy;

  // Camera poses
  xform::Xformd x1, x2;
  x1.t() = Vector3d(-10, 0, 0);
  x2.t() = Vector3d(-11, 4, -1);
  x1.q() = quat::Quatd::from_euler(0, 0, 0);
  x2.q() = quat::Quatd::from_euler(0.1, -0.1, -0.2);

  // Planar points (NED)
  // - N x N grid within +-bound in east and down directions
  const unsigned N = 5;
  const double bound = 5;
  vector<Point> pts_I = createInertialPoints(N, bound);

  // Project points into each camera image
  vector<Point> pts_1 = projectPointsToImage(K, pts_I, x1);
  vector<Point> pts_2 = projectPointsToImage(K, pts_I, x2);

  // Compute homography from matched image points
  vector<Point> matches_1, matches_2;
  imagePointMatches(pts_1, pts_2, matches_1, matches_2);
  Matrix3d H = computeHomographyFromPoints(matches_1, matches_2);

  // Compute homography from camera geometry
  Matrix3d R = (q_cb2c.inverse() * x1.q().inverse() * x2.q() * q_cb2c).R();
  Vector3d t = q_cb2c.rotp(x2.q().rotp((x2.t() - x1.t()) / x1.t()(0)));
  Vector3d n = q_cb2c.rotp(x1.q().rotp(common::e1));
  Matrix3d G = K * (R + t*n.transpose()) * K.inverse();
  G /= G(2,2);

  Vector3d p2_prime = H * matches_1[0].vec3();
  p2_prime /= p2_prime(2);
  cout << "p2  = " << matches_2[0].vec3().transpose() << endl;
  cout << "p2' = " << p2_prime.transpose() << endl;

  cout << "H = \n" << H << endl;
  cout << "G = \n" << G << endl;

  return 0;
}



// Functions
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


Matrix3d computeHomographyFromPoints(const vector<Point>& matches_1, const vector<Point>& matches_2)
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
