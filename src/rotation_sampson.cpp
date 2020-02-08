#include "rotation_sampson.h"


void sampson(double& S,
             const Vector3d& p1, const Vector3d& p2,
             const common::Quaterniond& q, const common::Quaterniond& qt)
{
  Matrix3d R = q.R();
  Vector3d et = qt.uvec();

  Matrix3d E = common::skew(et)*R;
  RowVector3d p2TE = p2.transpose()*E;
  Vector3d Ep1 = E*p1;

  S = p2.dot(Ep1)/sqrt(p2TE(0)*p2TE(0) + p2TE(1)*p2TE(1) + Ep1(0)*Ep1(0) + Ep1(1)*Ep1(1));
}


void sampsonDerivativeNumerical(Matrix<double,1,5>& dS,
                                const Vector3d& p1, const Vector3d& p2,
                                const common::Quaterniond& q, const common::Quaterniond& qt)
{
  static const double eps = 1e-6;
  double Sp, Sm;
  for (int i = 0; i < 3; ++i)
  {
    common::Quaterniond qp = q + eps * common::I_3x3.col(i);
    common::Quaterniond qm = q + -eps * common::I_3x3.col(i);
    sampson(Sp, p1, p2, qp, qt);
    sampson(Sm, p1, p2, qm, qt);
    dS(i) = (Sp - Sm)/(2.0*eps);
  }
  for (int i = 0; i < 2; ++i)
  {
    common::Quaterniond qtp = common::Quaterniond::boxPlusUnitVector(qt, eps*common::I_2x2.col(i));
    common::Quaterniond qtm = common::Quaterniond::boxPlusUnitVector(qt, -eps*common::I_2x2.col(i));
    sampson(Sp, p1, p2, q, qtp);
    sampson(Sm, p1, p2, q, qtm);
    dS(i+3) = (Sp - Sm)/(2.0*eps);
  }
}


void sampsonDerivative(Matrix<double,1,5>& dS,
                       const Vector3d& p1, const Vector3d& p2,
                       const common::Quaterniond& q, const common::Quaterniond& qt)
{
  Matrix3d R = q.R();
  Vector3d et = qt.uvec();

  Matrix3d E = common::skew(et)*R;
  RowVector3d p2TE = p2.transpose()*E;
  Vector3d Ep1 = E*p1;

  double val = p2TE(0)*p2TE(0) + p2TE(1)*p2TE(1) + Ep1(0)*Ep1(0) + Ep1(1)*Ep1(1);
  double S = p2.dot(Ep1)/sqrt(val);
  
  Matrix3d dE_dq1 = -common::skew(et)*common::skew(common::e1)*R;
  Matrix3d dE_dq2 = -common::skew(et)*common::skew(common::e2)*R;
  Matrix3d dE_dq3 = -common::skew(et)*common::skew(common::e3)*R;
  Matrix3d dE_dqt1 = common::skew((qt.proj()*Vector2d(1,0)).cross(et))*R;
  Matrix3d dE_dqt2 = common::skew((qt.proj()*Vector2d(0,1)).cross(et))*R;

  dS(0) = (p2.dot(dE_dq1*p1)*sqrt(val) - S*(p2TE(0)*p2.dot(dE_dq1.col(0)) + p2TE(1)*p2.dot(dE_dq1.col(1)) + Ep1(0)*(dE_dq1*p1)(0) + Ep1(1)*(dE_dq1*p1)(1)))/val;
  dS(1) = (p2.dot(dE_dq2*p1)*sqrt(val) - S*(p2TE(0)*p2.dot(dE_dq2.col(0)) + p2TE(1)*p2.dot(dE_dq2.col(1)) + Ep1(0)*(dE_dq2*p1)(0) + Ep1(1)*(dE_dq2*p1)(1)))/val;
  dS(2) = (p2.dot(dE_dq3*p1)*sqrt(val) - S*(p2TE(0)*p2.dot(dE_dq3.col(0)) + p2TE(1)*p2.dot(dE_dq3.col(1)) + Ep1(0)*(dE_dq3*p1)(0) + Ep1(1)*(dE_dq3*p1)(1)))/val;
  dS(3) = (p2.dot(dE_dqt1*p1)*sqrt(val) - S*(p2TE(0)*p2.dot(dE_dqt1.col(0)) + p2TE(1)*p2.dot(dE_dqt1.col(1)) + Ep1(0)*(dE_dqt1*p1)(0) + Ep1(1)*(dE_dqt1*p1)(1)))/val;
  dS(4) = (p2.dot(dE_dqt2*p1)*sqrt(val) - S*(p2TE(0)*p2.dot(dE_dqt2.col(0)) + p2TE(1)*p2.dot(dE_dqt2.col(1)) + Ep1(0)*(dE_dqt2*p1)(0) + Ep1(1)*(dE_dqt2*p1)(1)))/val;
}


void sampsonDerivativeR(Matrix<double,1,3>& dS,
                        const Vector3d& p1, const Vector3d& p2,
                        const common::Quaterniond& q, const common::Quaterniond& qt)
{
  Matrix3d R = q.R();
  Vector3d et = qt.uvec();

  Matrix3d E = common::skew(et)*R;
  RowVector3d p2TE = p2.transpose()*E;
  Vector3d Ep1 = E*p1;

  double val = p2TE(0)*p2TE(0) + p2TE(1)*p2TE(1) + Ep1(0)*Ep1(0) + Ep1(1)*Ep1(1);
  double S = p2.dot(Ep1)/sqrt(val);
  
  Matrix3d dE_dq1 = -common::skew(et)*common::skew(common::e1)*R;
  Matrix3d dE_dq2 = -common::skew(et)*common::skew(common::e2)*R;
  Matrix3d dE_dq3 = -common::skew(et)*common::skew(common::e3)*R;

  dS(0) = (p2.dot(dE_dq1*p1)*sqrt(val) - S*(p2TE(0)*p2.dot(dE_dq1.col(0)) + p2TE(1)*p2.dot(dE_dq1.col(1)) + Ep1(0)*(dE_dq1*p1)(0) + Ep1(1)*(dE_dq1*p1)(1)))/val;
  dS(1) = (p2.dot(dE_dq2*p1)*sqrt(val) - S*(p2TE(0)*p2.dot(dE_dq2.col(0)) + p2TE(1)*p2.dot(dE_dq2.col(1)) + Ep1(0)*(dE_dq2*p1)(0) + Ep1(1)*(dE_dq2*p1)(1)))/val;
  dS(2) = (p2.dot(dE_dq3*p1)*sqrt(val) - S*(p2TE(0)*p2.dot(dE_dq3.col(0)) + p2TE(1)*p2.dot(dE_dq3.col(1)) + Ep1(0)*(dE_dq3*p1)(0) + Ep1(1)*(dE_dq3*p1)(1)))/val;
}


void sampsonDerivativeT(Matrix<double,1,2>& dS,
                        const Vector3d& p1, const Vector3d& p2,
                        const common::Quaterniond& q, const common::Quaterniond& qt)
{
  Matrix3d R = q.R();
  Vector3d et = qt.uvec();

  Matrix3d E = common::skew(et)*R;
  RowVector3d p2TE = p2.transpose()*E;
  Vector3d Ep1 = E*p1;

  double val = p2TE(0)*p2TE(0) + p2TE(1)*p2TE(1) + Ep1(0)*Ep1(0) + Ep1(1)*Ep1(1);
  double S = p2.dot(Ep1)/sqrt(val);
  
  Matrix3d dE_dqt1 = common::skew((qt.proj()*Vector2d(1,0)).cross(et))*R;
  Matrix3d dE_dqt2 = common::skew((qt.proj()*Vector2d(0,1)).cross(et))*R;

  dS(0) = (p2.dot(dE_dqt1*p1)*sqrt(val) - S*(p2TE(0)*p2.dot(dE_dqt1.col(0)) + p2TE(1)*p2.dot(dE_dqt1.col(1)) + Ep1(0)*(dE_dqt1*p1)(0) + Ep1(1)*(dE_dqt1*p1)(1)))/val;
  dS(1) = (p2.dot(dE_dqt2*p1)*sqrt(val) - S*(p2TE(0)*p2.dot(dE_dqt2.col(0)) + p2TE(1)*p2.dot(dE_dqt2.col(1)) + Ep1(0)*(dE_dqt2*p1)(0) + Ep1(1)*(dE_dqt2*p1)(1)))/val;
}


void sampsonLM(common::Quaterniond& q, common::Quaterniond& qt, const vector<Match>& matches,
               const int& max_iters, const double& exit_tol,
               const double& lambda0, const double& lambda_adjust)
{
  double lambda = lambda0;

  common::Quaterniond q_new, qt_new;
  Matrix<double,1,5> dS;
  Matrix<double,5,1> b, delta;
  Matrix<double,5,5> H, H_diag, A;
  unsigned N = matches.size();
  VectorXd cost(N), cost_new(N);
  MatrixXd J(N,5);
  double cost_squared;
  bool prev_fail = false;
  for (int i = 0; i < max_iters; ++i)
  {
    if (!prev_fail)
    {
      // Build cost function and Jacobian
      for (int j = 0; j < N; ++j)
      {
        sampson(cost(j), matches[j].p1.vec3(), matches[j].p2.vec3(), q, qt);
        sampsonDerivative(dS, matches[j].p1.vec3(), matches[j].p2.vec3(), q, qt);
        // sampsonDerivativeNumerical(dS, matches[j].p1.vec3(), matches[j].p2.vec3(), q, qt);
        J.row(j) = dS;
      }
      H = J.transpose()*J;
      b = -J.transpose()*cost;
      cost_squared = cost.dot(cost);
    }
    H_diag = H.diagonal().asDiagonal();
    A = H + lambda*H_diag;
    delta = A.householderQr().solve(b);

    // Compute cost with new parameters
    q_new = q + delta.head<3>();
    qt_new = common::Quaterniond::boxPlusUnitVector(qt, delta.tail<2>());
    for (int j = 0; j < N; ++j)
      sampson(cost_new(j), matches[j].p1.vec3(), matches[j].p2.vec3(), q_new, qt_new);
    if (cost_new.dot(cost_new) < cost_squared)
    {
      q = q_new;
      qt = qt_new;
      lambda /= lambda_adjust;
      prev_fail = false;
    }
    else
    {
      lambda *= lambda_adjust;
      prev_fail = true;
    }

    if (delta.norm() < exit_tol) break;
  }
}


void sampsonLMR(common::Quaterniond& q, const common::Quaterniond& qt, const vector<Match>& matches,
                const int& max_iters, const double& exit_tol,
                const double& lambda0, const double& lambda_adjust)
{
  double lambda = lambda0;

  common::Quaterniond q_new;
  Matrix<double,1,3> dS;
  Matrix<double,3,1> b, delta;
  Matrix<double,3,3> H, H_diag, A;
  unsigned N = matches.size();
  VectorXd cost(N), cost_new(N);
  MatrixXd J(N,3);
  double cost_squared;
  bool prev_fail = false;
  for (int i = 0; i < max_iters; ++i)
  {
    if (!prev_fail)
    {
      // Build cost function and Jacobian
      for (int j = 0; j < N; ++j)
      {
        sampson(cost(j), matches[j].p1.vec3(), matches[j].p2.vec3(), q, qt);
        sampsonDerivativeR(dS, matches[j].p1.vec3(), matches[j].p2.vec3(), q, qt);
        J.row(j) = dS;
      }
      H = J.transpose()*J;
      b = -J.transpose()*cost;
      cost_squared = cost.dot(cost);
    }
    H_diag = H.diagonal().asDiagonal();
    A = H + lambda*H_diag;
    delta = A.householderQr().solve(b);

    // Compute cost with new parameters
    q_new = q + delta;
    for (int j = 0; j < N; ++j)
      sampson(cost_new(j), matches[j].p1.vec3(), matches[j].p2.vec3(), q_new, qt);
    if (cost_new.dot(cost_new) < cost_squared)
    {
      q = q_new;
      lambda /= lambda_adjust;
      prev_fail = false;
    }
    else
    {
      lambda *= lambda_adjust;
      prev_fail = true;
    }

    if (delta.norm() < exit_tol) break;
  }
}


void sampsonLMT(common::Quaterniond& qt, const common::Quaterniond& q, const vector<Match>& matches,
                const int& max_iters, const double& exit_tol,
                const double& lambda0, const double& lambda_adjust)
{
  double lambda = lambda0;

  common::Quaterniond qt_new;
  Matrix<double,1,2> dS;
  Matrix<double,2,1> b, delta;
  Matrix<double,2,2> H, H_diag, A;
  unsigned N = matches.size();
  VectorXd cost(N), cost_new(N);
  MatrixXd J(N,2);
  double cost_squared;
  bool prev_fail = false;
  for (int i = 0; i < max_iters; ++i)
  {
    if (!prev_fail)
    {
      // Build cost function and Jacobian
      for (int j = 0; j < N; ++j)
      {
        sampson(cost(j), matches[j].p1.vec3(), matches[j].p2.vec3(), q, qt);
        sampsonDerivativeT(dS, matches[j].p1.vec3(), matches[j].p2.vec3(), q, qt);
        J.row(j) = dS;
      }
      H = J.transpose()*J;
      b = -J.transpose()*cost;
      cost_squared = cost.dot(cost);
    }
    H_diag = H.diagonal().asDiagonal();
    A = H + lambda*H_diag;
    delta = A.householderQr().solve(b);

    // Compute cost with new parameters
    qt_new = common::Quaterniond::boxPlusUnitVector(qt, delta);
    for (int j = 0; j < N; ++j)
      sampson(cost_new(j), matches[j].p1.vec3(), matches[j].p2.vec3(), q, qt_new);
    if (cost_new.dot(cost_new) < cost_squared)
    {
      qt = qt_new;
      lambda /= lambda_adjust;
      prev_fail = false;
    }
    else
    {
      lambda *= lambda_adjust;
      prev_fail = true;
    }

    if (delta.norm() < exit_tol) break;
  }
}


void sampsonInitTranslation(common::Quaterniond& qt, const common::Quaterniond& q,
                            const vector<Match>& matches, const int& num_iters)
{
  // Randomly initialize translation direction and check for lowest Sampson error
  unsigned N = matches.size();
  VectorXd cost(N);
  double cost_squared = 1e9;
  common::Quaterniond qt_tmp;
  for (int i = 0; i < num_iters; ++i)
  {
    Vector3d et = Vector3d::Random().normalized();
    qt_tmp = common::Quaterniond::fromUnitVector(et);
    for (int j = 0; j < matches.size(); ++j)
      sampson(cost(j), matches[j].p1.vec3(), matches[j].p2.vec3(), q, qt_tmp);
    double cost_squared_tmp = cost.dot(cost);
    if (cost_squared_tmp < cost_squared)
    {
      cost_squared = cost_squared_tmp;
      qt = qt_tmp;
    }
  }
}


void estimateTranslationDirection(common::Quaterniond& qt, const common::Quaterniond& q, const Matrix3d& K, const vector<Match>& matches)
{
  Matrix3d K_inv = K.inverse();
  Matrix3d M = Matrix3d::Zero();
  for (size_t i = 0; i < matches.size(); ++i)
  {
    Vector3d a = (K_inv*matches[i].p1.vec3()).normalized();
    Vector3d b = (K_inv*matches[i].p2.vec3()).normalized();
    Vector3d a_x_RTb = a.cross(q.rota(b));
    M += a_x_RTb*a_x_RTb.transpose();
  }

  BDCSVD<Matrix3d> svd(M, ComputeFullV);
  Vector3d et = svd.matrixV().col(2);
  qt = common::Quaterniond::fromUnitVector(et);
}


void estimateTranslationDirectionKnownRotation(common::Quaterniond& qt, const common::Quaterniond& q, const vector<Match>& matches)
{
  // Solve homogeneous equation for translation given at least two points
  unsigned N = matches.size();
  MatrixXd A(N,3);
  for (size_t i = 0; i < N; ++i)
  {
    Vector3d a = matches[i].p1.vec3();
    Vector3d b = matches[i].p2.vec3();
    A.row(i) = b.transpose()*common::skew(q.rotp(a));
  }
  A = A.transpose()*A;
  BDCSVD<Matrix3d> svd(A, ComputeFullV);
  Vector3d et = svd.matrixV().col(2);

  // Make sure depth is positive
  Vector3d a = matches[0].p1.vec3();
  Vector3d b = matches[0].p2.vec3();
  Matrix<double,3,2> F;
  F.col(0) = -q.rotp(a);
  F.col(1) = b;
  Vector2d ds = F.householderQr().solve(et);
  if (ds(0) < 0 || ds(1) < 0)
    et *= -1;
  
  // Output
  qt = common::Quaterniond::fromUnitVector(et);
}


void rotationFromPointsSampson(common::Quaterniond& q, common::Quaterniond& qt,
                               const Matrix3d& K, const vector<Match>& matches,
                               const int& max_iters, const double& exit_tol,
                               const double& lambda0, const double& lambda_adjust)
{
  Matrix3d K_inv = K.inverse();

  // Compute direction vectors for matches
  vector<Match> match_dirs;
  for (const auto& match : matches)
  {
    Vector3d dir1 = (K_inv*match.p1.vec3()).normalized();
    Vector3d dir2 = (K_inv*match.p2.vec3()).normalized();
    Point p1(dir1(0), dir1(1), dir1(2), match.p1.id);
    Point p2(dir2(0), dir2(1), dir2(2), match.p2.id);
    match_dirs.push_back(Match(p1, p2));
  }

  // Refine using Sampson error
  // sampsonLMR(q, qt, match_dirs, max_iters, exit_tol, lambda0, lambda_adjust);
  // sampsonLMT(qt, q, match_dirs, max_iters, exit_tol, lambda0, lambda_adjust);
  // sampsonLM(q, qt, match_dirs, max_iters, exit_tol, lambda0, lambda_adjust);

  for (int i = 0; i < max_iters; ++i)
  {
    sampsonLM(q, qt, match_dirs, max_iters, exit_tol, lambda0, lambda_adjust);
    estimateTranslationDirectionKnownRotation(qt, q, match_dirs);
    sampsonLMT(qt, q, match_dirs, max_iters, exit_tol, lambda0, lambda_adjust);
  }
}
