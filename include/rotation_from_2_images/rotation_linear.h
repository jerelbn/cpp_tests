/*

Rotation From Two Images - calculate only the rotation matrix from matched points in two images
- A Closed-Form Solution to Rotation Estimation for Structure from Small Motion - Ha, Oh, Kweon
- NOTE: This solution degrades as the ratio of camera translation to average feature depth increases!

*/
#pragma once

#include "rotation_from_2_images/util.h"

using namespace std;
using namespace Eigen;

void linearSolutionHa(common::Quaterniond& q, const Matrix3d& K, const vector<Match>& matches);

void refineHaLM(common::Quaterniond& q, const Matrix3d& K, const vector<Match>& matches,
                const int& max_iters=10, const double& exit_tol=1e-6,
                const double& lambda0=1, const double& lambda_adjust=10);

void rotationFromPointsHa(common::Quaterniond& q, const Matrix3d& K, const vector<Match>& matches,
                          const bool& refine_solution=false,
                          const int& max_iters=10, const double& exit_tol=1e-6,
                          const double& lambda0=1e-6, const double& lambda_adjust=10);
                          
void rotationFromPointsHaRANSAC(common::Quaterniond& q, vector<Match>& inliers,
                                const Matrix3d& K, const vector<Match>& matches,
                                const bool& refine_solution=false,
                                const int& max_iters=10, const double& exit_tol=1e-6,
                                const double& lambda0=1e-6, const double& lambda_adjust=10,
                                const int RANSAC_iters=16, const double& RANSAC_thresh=50);
