#include "rotation_kneip.h"


void matrixMconstants(Matrix3d& Ax,  Matrix3d& Ay,  Matrix3d& Az,
                      Matrix3d& Axy, Matrix3d& Axz, Matrix3d& Ayz,
                      const Matrix3d& K, const vector<Match>& matches)
{
  Ax.setZero();
  Ay.setZero();
  Az.setZero();
  Axy.setZero();
  Axz.setZero();
  Ayz.setZero();
  Matrix3d K_inv = K.inverse();
  for (size_t i = 0; i < matches.size(); ++i)
  {
    Vector3d a = (K_inv*matches[i].p1.vec3()).normalized();
    Vector3d b = (K_inv*matches[i].p2.vec3()).normalized();
    Matrix3d bbT = b*b.transpose();
    Ax += a(0)*a(0)*bbT;
    Ay += a(1)*a(1)*bbT;
    Az += a(2)*a(2)*bbT;
    Axy += a(0)*a(1)*bbT;
    Axz += a(0)*a(2)*bbT;
    Ayz += a(1)*a(2)*bbT;
  }
}


void minEigenvalueOfM(double& lambda_min, const common::Quaterniond& q,
                      const Matrix3d& Ax,  const Matrix3d& Ay,  const Matrix3d& Az,
                      const Matrix3d& Axy, const Matrix3d& Axz, const Matrix3d& Ayz)
{
  // Initial guess of rotation matrix from Cayley parameters
  Matrix3d R = q.R();
  Vector3d r1 = R.row(0);
  Vector3d r2 = R.row(1);
  Vector3d r3 = R.row(2);

  // Compute components of matrix M
  double m11 = r2.dot(Az*r2) + r3.dot(Ay*r3) - 2.0*r3.dot(Ayz*r2);
  double m22 = r1.dot(Az*r1) + r3.dot(Ax*r3) - 2.0*r1.dot(Axz*r3);
  double m33 = r1.dot(Ay*r1) + r2.dot(Ax*r2) - 2.0*r1.dot(Axy*r2);
  double m12 = r1.dot(Ayz*r3) - r3.dot(Axy*r3) - r1.dot(Az*r2) + r3.dot(Axz*r2);
  double m13 = r2.dot(Axy*r3) - r2.dot(Axz*r2) - r1.dot(Ay*r3) + r1.dot(Ayz*r2);
  double m23 = r1.dot(Axz*r2) - r1.dot(Ayz*r1) - r3.dot(Ax*r2) + r3.dot(Axy*r1);

  double a = 1.0;
  double b = -(m11 + m22 + m33);
  double c = m11*m22 + m11*m33 + m22*m33 - m12*m12 - m13*m13 - m23*m23;
  double d = m11*m23*m23 + m22*m13*m13 + m33*m12*m12 - 2.0*m12*m13*m23 - m11*m22*m33;

  complex<double> Delta0 = b*b - 3.0*a*c;
  complex<double> Delta1 = 2.0*b*b*b - 9.0*a*b*c + 27.0*a*a*d;

  complex<double> root2 = sqrt(pow(Delta1,2.0) - 4.0*pow(Delta0,3.0));
  complex<double> C = pow((Delta1 + root2)/2.0, 1.0/3.0);
  if (abs(C.real()) < 1e-6)
    C = pow((Delta1 - root2)/2.0, 1.0/3.0);

  lambda_min = (-1.0/3.0*(b + C + Delta0/C)).real();
}


void derivativeOfMinEigenvalueOfM(Vector3d& dlambda_min, const common::Quaterniond& q,
                                  const Matrix3d& Ax, const Matrix3d& Ay, const Matrix3d& Az,
                                  const Matrix3d& Axy, const Matrix3d& Axz, const Matrix3d& Ayz)
{
  // Initial guess of rotation matrix from Cayley parameters
  Matrix3d R = q.R();
  Vector3d r1 = R.col(0);
  Vector3d r2 = R.col(1);
  Vector3d r3 = R.col(2);

  // Components of minimum eigenvalue of M
  double m11 = r2.dot(Az*r2) + r3.dot(Ay*r3) - 2.0*r3.dot(Ayz*r2);
  double m22 = r1.dot(Az*r1) + r3.dot(Ax*r3) - 2.0*r1.dot(Axz*r3);
  double m33 = r1.dot(Ay*r1) + r2.dot(Ax*r2) - 2.0*r1.dot(Axy*r2);
  double m12 = r1.dot(Ayz*r3) - r3.dot(Axy*r3) - r1.dot(Az*r2) + r3.dot(Axz*r2);
  double m13 = r2.dot(Axy*r3) - r2.dot(Axz*r2) - r1.dot(Ay*r3) + r1.dot(Ayz*r2);
  double m23 = r1.dot(Axz*r2) - r1.dot(Ayz*r1) - r3.dot(Ax*r2) + r3.dot(Axy*r1);

  double a = 1.0;
  double b = -(m11 + m22 + m33);
  double c = m11*m22 + m11*m33 + m22*m33 - m12*m12 - m13*m13 - m23*m23;
  double d = m11*m23*m23 + m22*m13*m13 + m33*m12*m12 - 2.0*m12*m13*m23 - m11*m22*m33;

  complex<double> Delta0 = b*b - 3.0*a*c;
  complex<double> Delta1 = 2.0*b*b*b - 9.0*a*b*c + 27.0*a*a*d;

  complex<double> root2 = sqrt(pow(Delta1,2.0) - 4.0*pow(Delta0,3.0));
  complex<double> C = pow((Delta1 + root2)/2.0, 1.0/3.0);
  if (abs(C.real()) < 1e-6)
    C = pow((Delta1 - root2)/2.0, 1.0/3.0);

  // Derivatives
  Matrix3d dr1_dq = -common::skew(q.rotp(common::e1));
  Matrix3d dr2_dq = -common::skew(q.rotp(common::e2));
  Matrix3d dr3_dq = -common::skew(q.rotp(common::e3));

  RowVector3d dm11_dv = 2.0*((r2.transpose()*Az - r3.transpose()*Ayz)*dr2_dq + (r3.transpose()*Ay - r2.transpose()*Ayz)*dr3_dq);
  RowVector3d dm22_dv = 2.0*((r1.transpose()*Az - r3.transpose()*Axz)*dr1_dq + (r3.transpose()*Ax - r1.transpose()*Axz)*dr3_dq);
  RowVector3d dm33_dv = 2.0*((r1.transpose()*Ay - r2.transpose()*Axy)*dr1_dq + (r2.transpose()*Ax - r1.transpose()*Axy)*dr2_dq);
  RowVector3d dm12_dv = (r3.transpose()*Ayz - r2.transpose()*Az)*dr1_dq + (r3.transpose()*Axz - r1.transpose()*Az)*dr2_dq + (r1.transpose()*Ayz - 2.0*r3.transpose()*Axy + r2.transpose()*Axz)*dr3_dq;
  RowVector3d dm13_dv = (r2.transpose()*Ayz - r3.transpose()*Ay)*dr1_dq + (r2.transpose()*Axy - r1.transpose()*Ay)*dr3_dq + (r1.transpose()*Ayz - 2.0*r2.transpose()*Axz + r3.transpose()*Axy)*dr2_dq;
  RowVector3d dm23_dv = (r1.transpose()*Axz - r3.transpose()*Ax)*dr2_dq + (r1.transpose()*Axy - r2.transpose()*Ax)*dr3_dq + (r2.transpose()*Axz - 2.0*r1.transpose()*Ayz + r3.transpose()*Axy)*dr1_dq;

  RowVector3cd db_dv = -(dm11_dv + dm22_dv + dm33_dv);
  RowVector3d dc_dv = (m22 + m33)*dm11_dv + (m11 + m33)*dm22_dv + (m11 + m22)*dm33_dv - 2.0*(m12*dm12_dv + m13*dm13_dv + m23*dm23_dv);
  RowVector3d dd_dv = (m23*m23 - m22*m33)*dm11_dv + (m13*m13 - m11*m33)*dm22_dv + (m12*m12 - m11*m22)*dm33_dv +
                       2.0*((m33*m12 - m13*m23)*dm12_dv + (m22*m13 - m12*m23)*dm13_dv + (m11*m23 - m12*m13)*dm23_dv);

  RowVector3cd dDelta0_dv = 2.0*b*db_dv - 3.0*dc_dv;
  RowVector3cd dDelta1_dv = 3.0*(2.0*b*b - 3.0*c)*db_dv - 9.0*b*dc_dv + 27.0*dd_dv;

  RowVector3cd dC_dv = 1.0/6.0*pow((Delta1 + root2)/2.0,-2.0/3.0)*(dDelta1_dv + (Delta1*dDelta1_dv - 6.0*Delta0*Delta0*dDelta0_dv)/root2);
  if (abs(C.real() < 1e-6))
    dC_dv = 1.0/6.0*pow((Delta1 - root2)/2.0,-2.0/3.0)*(dDelta1_dv - (Delta1*dDelta1_dv - 6.0*Delta0*Delta0*dDelta0_dv)/root2);

  dlambda_min = (-1.0/3.0*(db_dv + dDelta0_dv/C + (1.0 - Delta0/(C*C))*dC_dv)).real();
}


void secondDerivativeOfMinEigenvalueOfM(Matrix3d& ddl_ddq, const common::Quaterniond& q,
                                        const Matrix3d& Ax, const Matrix3d& Ay, const Matrix3d& Az,
                                        const Matrix3d& Axy, const Matrix3d& Axz, const Matrix3d& Ayz)
{
  static const double eps = 1e-6;
  Vector3d dlam_p, dlam_m;
  common::Quaterniond qp, qm;
  for (int i = 0; i < 3; ++i)
  {
    qp = q + eps * common::I_3x3.col(i);
    qm = q + -eps * common::I_3x3.col(i);
    derivativeOfMinEigenvalueOfM(dlam_p, qp, Ax, Ay, Az, Axy, Axz, Ayz);
    derivativeOfMinEigenvalueOfM(dlam_m, qm, Ax, Ay, Az, Axy, Axz, Ayz);
    ddl_ddq.col(i) = (dlam_p - dlam_m)/(2.0*eps);
  }
}


void kneipLM(common::Quaterniond& q, const Matrix3d& K, const vector<Match>& matches,
             const Matrix3d& Ax,  const Matrix3d& Ay,  const Matrix3d& Az,
             const Matrix3d& Axy, const Matrix3d& Axz, const Matrix3d& Ayz,
             const int& max_iters, const double& exit_tol, const double& lambda0,
             const double& lambda_adjust, const double& restart_variation)
{
  common::Quaterniond q0;// = rotationFromPointsHa(K, matches);
  double lambda = lambda0;
  q = q0 + restart_variation * Vector3d::Random();

  // Find R(v) to minimize smallest eigenvalue of M
  common::Quaterniond q_new;
  Vector3d dl_dq, dl_dq_new, b, delta;
  Matrix3d J, H, A;
  bool prev_fail = false;
  for (int i = 0; i < max_iters; ++i)
  {
    // Calculate change in Cayley parameters
    if (!prev_fail)
    {
      derivativeOfMinEigenvalueOfM(dl_dq, q, Ax, Ay, Az, Axy, Axz, Ayz);
      secondDerivativeOfMinEigenvalueOfM(J, q, Ax, Ay, Az, Axy, Axz, Ayz);
      H = J.transpose()*J;
      b = -J.transpose()*dl_dq;
    }
    A = H + lambda*Matrix3d(H.diagonal().asDiagonal());
    delta = A.householderQr().solve(b);

    // Compute error with new parameters
    q_new = q + delta;
    derivativeOfMinEigenvalueOfM(dl_dq_new, q_new, Ax, Ay, Az, Axy, Axz, Ayz);
    if (dl_dq_new.norm() < dl_dq.norm())
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

    // Try restarting at random value if rotation > some threshold
    double rot_from_q0 = common::Quaterniond::log(q.inverse() * q0).norm();
    if (rot_from_q0 > 0.78)
      q = q0 + restart_variation * Vector3d::Random();
  }
}


void rotationFromPointsKneip(common::Quaterniond& q, const Matrix3d& K, const vector<Match>& matches,
                             const int& max_iters, const double& exit_tol, const double& lambda0,
                             const double& lambda_adjust, const double& restart_variation)
{
  // Compute constants
  Matrix3d Ax, Ay, Az, Axy, Axz, Ayz;
  matrixMconstants(Ax, Ay, Az, Axy, Axz, Ayz, K, matches);

  // Find R(q) to minimize smallest eigenvalue of M via Levenberg-Marquardt algorithm
  kneipLM(q, K, matches, Ax, Ay, Az, Axy, Axz, Ayz, max_iters, exit_tol, lambda0, lambda_adjust, restart_variation);
}


void rotationFromPointsKneipRANSAC(common::Quaterniond& q, vector<Match>& inliers,
                                   const Matrix3d& K, const vector<Match>& matches,
                                   const int& max_iters, const double& exit_tol, const double& lambda0,
                                   const double& lambda_adjust, const double& restart_variation,
                                   const int& RANSAC_iters, const double& RANSAC_thresh)
{
  inliers.clear();
  double lam_min;
  Vector3d dl_dq;
  Matrix3d Ax, Ay, Az, Axy, Axz, Ayz;
  vector<Match> inliers_tmp, inliers_test, matches_shuffled;
  for (int ii = 0; ii < RANSAC_iters; ++ii)
  {
    // Randomly select 2 point pairs
    inliers_tmp.clear();
    matches_shuffled = matches;
    std::random_shuffle(matches_shuffled.begin(), matches_shuffled.end());
    for (int jj = 0; jj < 2; ++jj)
    {
      inliers_tmp.push_back(matches_shuffled.back());
      matches_shuffled.pop_back();
    }

    // Compute solution from the 2 points
    matrixMconstants(Ax, Ay, Az, Axy, Axz, Ayz, K, inliers_tmp);
    kneipLM(q, K, inliers_tmp, Ax, Ay, Az, Axy, Axz, Ayz, max_iters, exit_tol, lambda0, lambda_adjust, restart_variation);

    // Iterate through remaining point pairs
    while (matches_shuffled.size() > 0)
    {
      // Add another point to the set to test
      inliers_tmp.push_back(matches_shuffled.back());
      matches_shuffled.pop_back();

      // Recompute constants and check derivative
      matrixMconstants(Ax, Ay, Az, Axy, Axz, Ayz, K, inliers_tmp);
      derivativeOfMinEigenvalueOfM(dl_dq, q, Ax, Ay, Az, Axy, Axz, Ayz);
      minEigenvalueOfM(lam_min, q, Ax, Ay, Az, Axy, Axz, Ayz);
      if (lam_min < RANSAC_thresh)
      // if (inliers_tmp.back().p1.id < 999999)
      {
        // cout << endl;
        // cout << "dl_dq.norm(): " << dl_dq.norm() << endl;
        // cout << "  lambda_min: " << lam_min << endl;
        // cout << "          ID: " << inliers_tmp.back().p1.id << endl;
        inliers.push_back(matches_shuffled.back());
      }
      else
      {
        inliers_tmp.pop_back();
      }
    }

    // Keep track of set with the most inliers
    if (inliers.size() < inliers_tmp.size())
      inliers = inliers_tmp;
  }

  // Smooth final solution over all inliers
  matrixMconstants(Ax, Ay, Az, Axy, Axz, Ayz, K, inliers);
  kneipLM(q, K, inliers, Ax, Ay, Az, Axy, Axz, Ayz, max_iters, exit_tol, lambda0, lambda_adjust, restart_variation);
}
