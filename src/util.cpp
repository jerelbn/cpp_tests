#include "util.h"



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
