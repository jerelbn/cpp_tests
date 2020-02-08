#include "rotation_linear.h"


void linearSolutionHa(common::Quaterniond& q, const Matrix3d& K, const vector<Match>& matches)
{
  // Unpack camera intrinsic parameters
  double fx = K(0,0);
  double fy = K(1,1);
  double u0 = K(0,2);
  double v0 = K(1,2);

  // Build linear system with point matches
  unsigned N = 2 * matches.size();
  MatrixXd A(N,3);
  MatrixXd b(N,1);
  A.setZero();
  b.setZero();
  for (size_t i = 0; i < matches.size(); ++i)
  {
    // Unpack points for readability
    Point p1 = matches[i].p1;
    Point p2 = matches[i].p2;

    // Make sure points are not aligned with camera axis
    if (abs(p1.x - u0) < 1.0 && abs(p1.y - v0) < 1.0) continue;
    if (abs(p2.x - u0) < 1.0 && abs(p2.y - v0) < 1.0) continue;

    // Populate matrix/vector solution
    size_t idx1 = 2 * i;
    size_t idx2 = 2 * i + 1;
    A(idx1,0) = fx*u0*(p1.y - v0);
    A(idx1,1) = fy*(fx*fx - u0*p1.x + u0*u0);
    A(idx1,2) = fx*fx*(v0 - p1.y);
    A(idx2,0) = fx*(v0*p1.y - fy*fy - v0*v0);
    A(idx2,1) = fy*v0*(u0 - p1.x);
    A(idx2,2) = fy*fy*(p1.x - u0);
    b(idx1) = fx*fy*(p2.x - p1.x);
    b(idx2) = fx*fy*(p2.y - p1.y);
  }

  // Solve for apporximate rotation vector
  Vector3d r = A.householderQr().solve(b);
  
  // -r because the algorithm is based on a rotation matrix and quaternions are opposite rotations
  q = common::Quaterniond::exp(-r);
}


void refineHaLM(common::Quaterniond& q, const Matrix3d& K, const vector<Match>& matches,
                const int& max_iters, const double& exit_tol,
                const double& lambda0, const double& lambda_adjust)
{
  double lambda = lambda0;
  Matrix3d K_inv = K.inverse();

  // Find R(v) to minimize smallest eigenvalue of M
  common::Quaterniond q_new;
  Vector3d b, delta;
  Matrix3d H, A;
  unsigned N = matches.size();
  VectorXd cost(N), cost_new(N);
  MatrixXd J(N,3);
  bool prev_fail = false;
  for (int i = 0; i < max_iters; ++i)
  {
    // Calculate change in Cayley parameters
    if (!prev_fail)
    {
      // Build cost function and Jacobian
      for (int j = 0; j < N; ++j)
      {
        Vector3d nua = matches[j].p1.vec3();
        Vector3d nub = matches[j].p2.vec3();
        Vector3d nub_hat = K*q.rotp(K_inv*nua);
        Vector3d err = nub - nub_hat/nub_hat(2);
        cost(j) = 0.5*err.dot(err);

        Matrix3d dnub_hat_dq = -K*common::skew(q.rotp(K_inv*nua));
        J.row(j) = err.transpose()*((Matrix3d::Identity() - nub_hat*common::e3.transpose()/nub_hat(2))/nub_hat(2)*dnub_hat_dq);
      }
      H = J.transpose()*J;
      b = -J.transpose()*cost;
    }
    Matrix3d H_diag = H.diagonal().asDiagonal();
    A = H + lambda*H_diag;
    delta = A.householderQr().solve(b);

    // Compute cost with new parameters
    q_new = q + delta;
    for (int j = 0; j < N; ++j)
    {
      Vector3d nua = matches[j].p1.vec3();
      Vector3d nub = matches[j].p2.vec3();
      Vector3d nub_hat = K*q_new.rotp(K_inv*nua);
      Vector3d err = nub - nub_hat/common::e3.dot(nub_hat);
      cost_new(j) = 0.5*err.dot(err);
    }
    if (cost_new.dot(cost_new) < cost.dot(cost))
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


void rotationFromPointsHa(common::Quaterniond& q, const Matrix3d& K, const vector<Match>& matches,
                          const bool& refine_solution,
                          const int& max_iters, const double& exit_tol,
                          const double& lambda0, const double& lambda_adjust)
{
  linearSolutionHa(q, K, matches);
  if (refine_solution)
    refineHaLM(q, K, matches, max_iters, exit_tol, lambda0, lambda_adjust);
}


void rotationFromPointsHaRANSAC(common::Quaterniond& q, vector<Match>& inliers,
                                const Matrix3d& K, const vector<Match>& matches,
                                const bool& refine_solution,
                                const int& max_iters, const double& exit_tol,
                                const double& lambda0, const double& lambda_adjust,
                                const int RANSAC_iters, const double& RANSAC_thresh)
{
  inliers.clear();

  vector<Match> inliers_tmp, matches_shuffled;
  static const Matrix3d K_inv = K.inverse();
  for (int ii = 0; ii < RANSAC_iters; ++ii)
  {
    // Randomly select two point pairs
    inliers_tmp.clear();
    matches_shuffled = matches;
    std::random_shuffle(matches_shuffled.begin(), matches_shuffled.end());
    for (int jj = 0; jj < 2; ++jj)
    {
      inliers_tmp.push_back(matches_shuffled.back());
      matches_shuffled.pop_back();
    }

    // Compute solution based on two points
    linearSolutionHa(q, K, inliers_tmp);
    if (refine_solution)
      refineHaLM(q, K, inliers_tmp, max_iters, exit_tol, lambda0, lambda_adjust);

    // Iterate through remaining point pairs
    while (matches_shuffled.size() > 0)
    {
      // Unpack points for readability
      Point p1 = matches_shuffled.back().p1;
      Point p2 = matches_shuffled.back().p2;

      // Check eprojection error
      Vector2d pixel_diff = (p2.vec3() - K*q.rotp(K_inv*p1.vec3())).topRows<2>();
      if (pixel_diff.norm() < RANSAC_thresh)
      {
        // cout << endl;
        // cout << "pixel_diff.norm(): " << pixel_diff.norm() << ", ID: " << matches_shuffled.back().p1.id << endl;
        inliers_tmp.push_back(matches_shuffled.back());
      }

      // Remove point from set
      matches_shuffled.pop_back();
    }

    // Keep track of set with the most inliers
    if (inliers.size() < inliers_tmp.size())
      inliers = inliers_tmp;
  }

  // Smooth final solution over all inliers
  if (inliers.size() > 1)
  {
    linearSolutionHa(q, K, inliers);
    if (refine_solution)
      refineHaLM(q, K, inliers, max_iters, exit_tol, lambda0, lambda_adjust);
  }
}
