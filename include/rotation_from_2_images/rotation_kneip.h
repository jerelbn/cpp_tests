/*

Rotation From Two Images - calculate only the rotation matrix from matched points in two images
- Direct Optimization of Frame-to-Frame Rotation - Kneip, Lynen

*/
#pragma once

#include "rotation_from_2_images/util.h"


void matrixMconstants(Matrix3d& Ax,  Matrix3d& Ay,  Matrix3d& Az,
                      Matrix3d& Axy, Matrix3d& Axz, Matrix3d& Ayz,
                      const Matrix3d& K, const vector<Match>& matches);

void minEigenvalueOfM(double& lambda_min, const common::Quaterniond& q,
                      const Matrix3d& Ax,  const Matrix3d& Ay,  const Matrix3d& Az,
                      const Matrix3d& Axy, const Matrix3d& Axz, const Matrix3d& Ayz);

void derivativeOfMinEigenvalueOfM(Vector3d& dlambda_min, const common::Quaterniond& q,
                                  const Matrix3d& Ax, const Matrix3d& Ay, const Matrix3d& Az,
                                  const Matrix3d& Axy, const Matrix3d& Axz, const Matrix3d& Ayz);

void secondDerivativeOfMinEigenvalueOfM(Matrix3d& ddl_ddq, const common::Quaterniond& q,
                                        const Matrix3d& Ax, const Matrix3d& Ay, const Matrix3d& Az,
                                        const Matrix3d& Axy, const Matrix3d& Axz, const Matrix3d& Ayz);

void kneipLM(common::Quaterniond& q, const Matrix3d& K, const vector<Match>& matches,
             const Matrix3d& Ax,  const Matrix3d& Ay,  const Matrix3d& Az,
             const Matrix3d& Axy, const Matrix3d& Axz, const Matrix3d& Ayz,
             const int& max_iters=50, const double& exit_tol=1e-6, const double& lambda0=1e-6,
             const double& lambda_adjust=10, const double& restart_variation=0.05);

void rotationFromPointsKneip(common::Quaterniond& q, const Matrix3d& K, const vector<Match>& matches,
                             const int& max_iters=50, const double& exit_tol=1e-6, const double& lambda0=1e-6,
                             const double& lambda_adjust=10, const double& restart_variation=0.05);

void rotationFromPointsKneipRANSAC(common::Quaterniond& q, vector<Match>& inliers,
                                   const Matrix3d& K, const vector<Match>& matches,
                                   const int& max_iters=50, const double& exit_tol=1e-6, const double& lambda0=1e-6,
                                   const double& lambda_adjust=10, const double& restart_variation=0.05,
                                   const int& RANSAC_iters=16, const double& RANSAC_thresh=1e-7);
