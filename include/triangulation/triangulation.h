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
    static const Eigen::Matrix<float,2,3> S = (Eigen::Matrix<float,2,3>() << 1, 0, 0, 0, 1, 0).finished();
    Vector3f x1 = K_inv * Vector3f(pt1.x, pt1.y, 1);
    Vector3f x2 = K_inv * Vector3f(pt2.x, pt2.y, 1);
    Matrix3f E = common::skew(q.rota(-t)) * q.inverse().R();
    Matrix2f E_tilde = S * E * S.transpose();
    Vector2f n1 = S * E * x2;
    Vector2f n2 = S * E.transpose() * x1;
    float a = n1.dot(E_tilde * n2);
    float b = 0.5 * (n1.dot(n1) + n2.dot(n2));
    float c = x1.dot(E * x2);
    float d = sqrt(b * b - a * c);
    float lambda = c / (b + d);
    Vector2f dx1 = lambda * n1;
    Vector2f dx2 = lambda * n2;
    n1 -= E_tilde * dx2;
    n2 -= E_tilde.transpose() * dx1;
    dx1 = dx1.dot(n1) / n1.dot(n1) * n1;
    dx2 = dx2.dot(n2) / n2.dot(n2) * n2;
    x1 -= S.transpose() * dx1;
    x2 -= S.transpose() * dx2;
    
    // Get depth
    Vector3f z = x1.cross(q.rota(x2));
    Vector3f pt1_3d = z.dot(E * x2) / z.dot(z) * x1;
    pt1.depth = pt1_3d.norm();
    pt2.depth = (t + q.rotp(pt1_3d)).norm();
}