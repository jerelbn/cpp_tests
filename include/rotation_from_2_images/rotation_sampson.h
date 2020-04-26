/*

Rotation From Two Images - calculate only the rotation matrix from matched points in two images
- Optimize for R/t on Sampson error

*/
#pragma once

#include "rotation_from_2_images/util.h"



void sampson(double& S,
             const Vector3d& p1, const Vector3d& p2,
             const common::Quaterniond& q, const common::Quaterniond& qt);

void sampsonDerivativeNumerical(Matrix<double,1,5>& dS,
                                const Vector3d& p1, const Vector3d& p2,
                                const common::Quaterniond& q, const common::Quaterniond& qt);

void sampsonDerivative(Matrix<double,1,5>& dS,
                       const Vector3d& p1, const Vector3d& p2,
                       const common::Quaterniond& q, const common::Quaterniond& qt);

void sampsonDerivativeR(Matrix<double,1,3>& dS,
                        const Vector3d& p1, const Vector3d& p2,
                        const common::Quaterniond& q, const common::Quaterniond& qt);

void sampsonDerivativeT(Matrix<double,1,2>& dS,
                        const Vector3d& p1, const Vector3d& p2,
                        const common::Quaterniond& q, const common::Quaterniond& qt);

void sampsonLM(common::Quaterniond& q, common::Quaterniond& qt, const vector<Match>& matches,
               const int& max_iters=10, const double& exit_tol=1e-6,
               const double& lambda0=1e-6, const double& lambda_adjust=10);

void sampsonLMR(common::Quaterniond& q, const common::Quaterniond& qt, const vector<Match>& matches,
                const int& max_iters=10, const double& exit_tol=1e-6,
                const double& lambda0=1e-6, const double& lambda_adjust=10);

void sampsonLMT(common::Quaterniond& qt, const common::Quaterniond& q, const vector<Match>& matches,
                const int& max_iters=10, const double& exit_tol=1e-6,
                const double& lambda0=1e-6, const double& lambda_adjust=10);

void sampsonInitTranslation(common::Quaterniond& qt, const common::Quaterniond& q,
                            const vector<Match>& matches, const int& num_iters=10);

void estimateTranslationDirection(common::Quaterniond& qt, const common::Quaterniond& q, const Matrix3d& K, const vector<Match>& matches);

void estimateTranslationDirectionKnownRotation(common::Quaterniond& qt, const common::Quaterniond& q, const vector<Match>& matches);

void rotationFromPointsSampson(common::Quaterniond& q, common::Quaterniond& qt,
                               const Matrix3d& K, const vector<Match>& matches,
                               const int& max_iters=10, const double& exit_tol=1e-6,
                               const double& lambda0=1e-6, const double& lambda_adjust=10);
