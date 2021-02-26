// Triangulation test
#pragma once

#include <iostream>
#include <eigen3/Eigen/Dense>
#include <common_cpp/quaternion.h>

using namespace std;
using namespace Eigen;

struct Point
{
    float x; // Horizontal pixel position
    float y; // Vertical pixel position
    float z; // Distance to the point along optical axis of camera frame
};

float randi() {
    return 2.0*(rand()/(float)RAND_MAX - 0.5);
}

void triangulatePoint(Point& pt1, Point& pt2, const Matrix3f& K_inv,
                       const common::Quaternionf& q, const Vector3f& t)
{
    // Find optimal correspondences (Lindstrom paper)
    
    // Get depth
    Vector3f a = K_inv * Vector3f(pt1.x, pt1.y, 1.0);
    Vector3f b = K_inv * Vector3f(pt2.x, pt2.y, 1.0);
    Matrix<float,3,2> A;
    A.col(0) = b;
    A.col(1) = -q.rotp(a);
    Vector2f zs = A.householderQr().solve(t);
    pt1.z = zs(1);
    pt2.z = zs(0);
    float theta = acos(b.normalized().dot(q.rotp(a).normalized()));
    cout << "theta: " << theta*180/M_PI << endl;
}