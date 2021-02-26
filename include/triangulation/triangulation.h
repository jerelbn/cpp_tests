// Triangulation test
#pragma once

#include <iostream>
#include <eigen3/Eigen/Dense>
#include <common_cpp/quaternion.h>

using namespace std;
using namespace Eigen;

struct Point
{
    float x;     // Horizontal pixel position
    float y;     // Vertical pixel position
    float depth; // Distance to the point
};

float randi() {
    return 2.0*(rand()/(float)RAND_MAX - 0.5);
}

void triangulatePoint(Point& pt1, Point& pt2, const Matrix3f& K_inv,
                       const common::Quaternionf& q, const Vector3f& t)
{
    // Find optimal correspondences (Lindstrom paper)
    
    // Get depth
    Vector3f a = (K_inv * Vector3f(pt1.x, pt1.y, 1.0)).normalized();
    Vector3f b = (K_inv * Vector3f(pt2.x, pt2.y, 1.0)).normalized();
    Matrix<float,3,2> A;
    A.col(0) = b;
    A.col(1) = -q.rotp(a);
    Vector2f depths = A.householderQr().solve(t);
    pt1.depth = depths(1);
    pt2.depth = depths(0);
}