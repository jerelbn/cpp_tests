/*

Homography decomposition testing
- this uses NED and camera coordinates
- Deeper understanding of the homography decomposition for vision-based control - Malis, Vargas
- NOTE: This does not handle special cases, such as pure rotation or translation along the plane normal
- NOTE: This solution degrades as the ratio of camera translation to average feature depth decreases!

*/
#pragma once

#include <chrono>
#include <iostream>
#include "common_cpp/common.h"
#include "common_cpp/quaternion.h"
#include "common_cpp/transform.h"

#define CLOSE_TO_ZERO 0.00001

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


vector<Point> createInertialPoints(const unsigned& N, const double& bound);

void projectAndMatchImagePoints(const Matrix3d& K, const vector<Point>& pts_I,
                                const common::Transformd& x1, const common::Transformd& x2,
                                const double& pix_noise_bound, const double& matched_pts_inlier_ratio,
                                vector<Match>& matches);