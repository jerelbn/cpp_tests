/*

Homography decomposition testing
- this uses NED and camera coordinates
- Deeper understanding of the homography decomposition for vision-based control - Malis, Vargas
- NOTE: This does not handle special cases, such as pure rotation or translation along the plane normal

*/
#pragma once

#include "rotation_from_2_images/util.h"


void homographyFromPoints(Matrix3d& G, const vector<Match>& matches);

void homographyFromPointsRANSAC(Matrix3d& G, vector<Match>& inliers, const vector<Match>& matches,
                                const int& RANSAC_iters=71, const double& RANSAC_thresh=3.0);

Matrix3d homographyFromGeometry(const Matrix3d& K, const common::Transformd& x1, const common::Transformd& x2);

// Computes the Euclidean Homography such that H = R + n*t^T (eq. 3 of Mails paper)
bool euclideanHomography(Matrix3d& H, const Matrix3d& K, const Matrix3d& G);

void decomposeEuclideanHomography(const Matrix3d& H,
                                  vector<Matrix3d, aligned_allocator<Matrix3d> >& Rs,
                                  vector<Vector3d, aligned_allocator<Vector3d> >& ts,
                                  vector<Vector3d, aligned_allocator<Vector3d> >& ns);

void eliminateInvalidSolutions(const vector<Match>& matches, const Matrix3d& K,
                               vector<Matrix3d, aligned_allocator<Matrix3d> >& Rs,
                               vector<Vector3d, aligned_allocator<Vector3d> >& ts,
                               vector<Vector3d, aligned_allocator<Vector3d> >& ns);

void decomposeHomography(vector<Matrix3d, aligned_allocator<Matrix3d> >& Rs,
                         vector<Vector3d, aligned_allocator<Vector3d> >& ts,
                         vector<Vector3d, aligned_allocator<Vector3d> >& ns,
                         const Matrix3d& G, const Matrix3d& K, const vector<Match>& matches);
