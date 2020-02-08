#include "homography_decomposition.h"


void homographyFromPoints(Matrix3d& G, const vector<Match>& matches)
{
  // Build solution matrix with point matches
  unsigned N = 2 * matches.size();
  MatrixXd P(N,9);
  P.setZero();
  for (size_t i = 0; i < matches.size(); ++i)
  {
    size_t idx1 = 2 * i;
    size_t idx2 = 2 * i + 1;
    Point p1 = matches[i].p1;
    Point p2 = matches[i].p2;
    P(idx1,0) = -p1.x;
    P(idx1,1) = -p1.y;
    P(idx1,2) = -1;
    P(idx1,6) = p1.x * p2.x;
    P(idx1,7) = p1.y * p2.x;
    P(idx1,8) = p2.x;
    P(idx2,3) = -p1.x;
    P(idx2,4) = -p1.y;
    P(idx2,5) = -1;
    P(idx2,6) = p1.x * p2.y;
    P(idx2,7) = p1.y * p2.y;
    P(idx2,8) = p2.y;
  }

  // Solve for homography vector then reshape to matrix
  BDCSVD<MatrixXd> svd(P, ComputeFullV);
  Matrix<double,9,1> h = svd.matrixV().col(8);
  h /= h(8);
  G = Map<Matrix<double,3,3,RowMajor> >(h.data());
}


void homographyFromPointsRANSAC(Matrix3d& G, vector<Match>& inliers, const vector<Match>& matches,
                                const int& RANSAC_iters, const double& RANSAC_thresh)
{
  inliers.clear();
  vector<Match> inliers_tmp, matches_shuffled;
  for (int ii = 0; ii < RANSAC_iters; ++ii)
  {
    // Randomly select 4 point pairs
    inliers_tmp.clear();
    matches_shuffled = matches;
    std::random_shuffle(matches_shuffled.begin(), matches_shuffled.end());
    for (int jj = 0; jj < 4; ++jj)
    {
      inliers_tmp.push_back(matches_shuffled.back());
      matches_shuffled.pop_back();
    }

    // Compute solution based on 4 points
    homographyFromPoints(G, inliers_tmp);

    // Iterate through remaining point pairs
    while (matches_shuffled.size() > 0)
    {
      // Unpack points for readability
      Point p1 = matches_shuffled.back().p1;
      Point p2 = matches_shuffled.back().p2;

      // Check eprojection error
      Vector3d p2_prime = G * p1.vec3();
      p2_prime /= p2_prime(2);
      Vector2d pixel_diff = (p2.vec3() - p2_prime).topRows<2>();
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
    homographyFromPoints(G, inliers);
  }
}


Matrix3d homographyFromGeometry(const Matrix3d& K, const common::Transformd& x1, const common::Transformd& x2)
{
  Matrix3d R = (q_cb2c.inverse() * x1.q().inverse() * x2.q() * q_cb2c).R();
  Vector3d t = q_cb2c.rotp(x2.q().rotp((x2.p() - x1.p()) / x1.p()(0)));
  Vector3d n = q_cb2c.rotp(x1.q().rotp(common::e1));

  Matrix3d G = K * (R + t*n.transpose()) * K.inverse();
  G /= G(2,2);

  return G;
}


// Computes the Euclidean Homography such that H = R + n*t^T (eq. 3 of Mails paper)
bool euclideanHomography(Matrix3d& H, const Matrix3d& K, const Matrix3d& G)
{
  Matrix3d H_hat = K.inverse() * G * K;
  Matrix3d M = H_hat.transpose() * H_hat;

  // Return false for pure rotation case
  if (abs(M(0,0) - M(1,1)) > 0.1 || abs(M(0,0) - M(2,2)) > 0.1)
  {
    H = H_hat; // Do something else here
    return true;
  }
  
  double m11 = M(0,0);
  double m22 = M(1,1);
  double m33 = M(2,2);
  double m12 = M(0,1);
  double m13 = M(0,2);
  double m23 = M(1,2);

  double b = -(m11 + m22 + m33);
  double c = m11*m22 + m11*m33 + m22*m33 - m12*m12 - m13*m13 - m23*m23;
  double d = m12*m12*m33 + m13*m13*m22 + m23*m23*m11 - m11*m22*m33 - 2*m12*m13*m23;

  vector<double> reals, imags;
  common::solveCubic(1.0, b, c, d, reals, imags);

  H = H_hat / sqrt(reals[1]);

  return false;
}


void decomposeEuclideanHomography(const Matrix3d& H,
                                  vector<Matrix3d, aligned_allocator<Matrix3d> >& Rs,
                                  vector<Vector3d, aligned_allocator<Vector3d> >& ts,
                                  vector<Vector3d, aligned_allocator<Vector3d> >& ns)
{
  // Build symmetric matrix
  // - extract components
  // - compile determinants of opposite minors
  // - null close to zero values
  Matrix3d S = H.transpose() * H - common::I_3x3;

  double s11 = abs(S(0,0)) < CLOSE_TO_ZERO ? 0 : S(0,0);
  double s12 = abs(S(0,1)) < CLOSE_TO_ZERO ? 0 : S(0,1);
  double s13 = abs(S(0,2)) < CLOSE_TO_ZERO ? 0 : S(0,2);
  double s22 = abs(S(1,1)) < CLOSE_TO_ZERO ? 0 : S(1,1);
  double s23 = abs(S(1,2)) < CLOSE_TO_ZERO ? 0 : S(1,2);
  double s33 = abs(S(2,2)) < CLOSE_TO_ZERO ? 0 : S(2,2);

  double M_s11 = s23*s23 - s22*s33;
  double M_s22 = s13*s13 - s11*s33;
  double M_s33 = s12*s12 - s11*s22;
  double M_s12 = s13*s23 - s12*s33;
  double M_s13 = s13*s22 - s12*s23;
  double M_s23 = s12*s13 - s11*s23;

  M_s11 = abs(M_s11) < CLOSE_TO_ZERO ? 0 : M_s11;
  M_s22 = abs(M_s22) < CLOSE_TO_ZERO ? 0 : M_s22;
  M_s33 = abs(M_s33) < CLOSE_TO_ZERO ? 0 : M_s33;
  M_s12 = abs(M_s12) < CLOSE_TO_ZERO ? 0 : M_s12;
  M_s13 = abs(M_s13) < CLOSE_TO_ZERO ? 0 : M_s13;
  M_s23 = abs(M_s23) < CLOSE_TO_ZERO ? 0 : M_s23;

  // Compute some common parameters
  double nu = 2.0 * sqrt(1.0 + S.trace() - M_s11 - M_s22 - M_s33);
  double rho = sqrt(2.0 + S.trace() + nu);
  double te = sqrt(2.0 + S.trace() - nu);

  // Compute possible solutions
  Vector3d na, nb, ta_star, tb_star;
  if (abs(s11) > abs(s22) && abs(s11) > abs(s33))
  {
    // Plane normal
    na = Vector3d(s11, s12+sqrt(M_s33), s13+common::sign(M_s23)*sqrt(M_s22)).normalized();
    nb = Vector3d(s11, s12-sqrt(M_s33), s13-common::sign(M_s23)*sqrt(M_s22)).normalized();

    // Translation vector in wrong frame
    ta_star = te/2.0*(common::sign(s11)*rho*nb - te*na);
    tb_star = te/2.0*(common::sign(s11)*rho*na - te*nb);
  }
  else if (abs(s22) > abs(s11) && abs(s22) > abs(s33))
  {
    // Plane normal
    na = Vector3d(s12+sqrt(M_s33), s22, s23-common::sign(M_s13)*sqrt(M_s11)).normalized();
    nb = Vector3d(s12-sqrt(M_s33), s22, s23+common::sign(M_s13)*sqrt(M_s11)).normalized();

    // Translation vector in wrong frame
    ta_star = te/2.0*(common::sign(s22)*rho*nb - te*na);
    tb_star = te/2.0*(common::sign(s22)*rho*na - te*nb);
  }
  else
  {
    // Plane normal
    na = Vector3d(s13+common::sign(M_s12)*sqrt(M_s22), s23+sqrt(M_s11), s33).normalized();
    nb = Vector3d(s13-common::sign(M_s12)*sqrt(M_s22), s23-sqrt(M_s11), s33).normalized();

    // Translation vector in wrong frame
    ta_star = te/2.0*(common::sign(s33)*rho*nb - te*na);
    tb_star = te/2.0*(common::sign(s33)*rho*na - te*nb);
  }

  // Rotation matrix
  Matrix3d Ra = H*(common::I_3x3 - 2.0/nu*ta_star*na.transpose());
  Matrix3d Rb = H*(common::I_3x3 - 2.0/nu*tb_star*nb.transpose());

  // Translation vector
  Vector3d ta = Ra*ta_star;
  Vector3d tb = Rb*tb_star;

  // Populate output with 4 possible solution sets
  ns.clear();
  ts.clear();
  Rs.clear();

  Rs.push_back(Ra);
  ns.push_back(na);
  ts.push_back(ta);

  Rs.push_back(Ra);
  ns.push_back(-na);
  ts.push_back(-ta);

  Rs.push_back(Rb);
  ns.push_back(nb);
  ts.push_back(tb);

  Rs.push_back(Rb);
  ns.push_back(-nb);
  ts.push_back(-tb);
}


// Eliminate impossible solutions to the homography decomposition with physical constraints
void eliminateInvalidSolutions(const vector<Match>& matches, const Matrix3d& K,
                               vector<Matrix3d, aligned_allocator<Matrix3d> >& Rs,
                               vector<Vector3d, aligned_allocator<Vector3d> >& ts,
                               vector<Vector3d, aligned_allocator<Vector3d> >& ns)
{
  // Copy solutions and clear containers
  vector<Matrix3d, aligned_allocator<Matrix3d> > Rs_original = Rs;
  vector<Vector3d, aligned_allocator<Vector3d> > ts_original = ts;
  vector<Vector3d, aligned_allocator<Vector3d> > ns_original = ns;
  Rs.clear();
  ts.clear();
  ns.clear();

  // Fill output containers with physically possible solutions
  for (int i = 0; i < Rs_original.size(); ++i)
  {
    // Unpack for readability
    Matrix3d R = Rs_original[i];
    Vector3d t = ts_original[i];
    Vector3d n = ns_original[i];

    // Check that each point is in front of the camera
    bool pass = true;
    for (const auto& match : matches)
    {
      Vector3d m = K.inverse() * match.p2.vec3();
      if (m.dot(R*n) <= 0)
      {
        pass = false;
        break;
      }
    }

    // Only save solutions that put points in front of camera
    if (pass)
    {
      Rs.push_back(R);
      ts.push_back(t);
      ns.push_back(n);
    }
  }
}


// Decompose Homography computed from points and eliminate impossible solutions
void decomposeHomography(vector<Matrix3d, aligned_allocator<Matrix3d> >& Rs,
                         vector<Vector3d, aligned_allocator<Vector3d> >& ts,
                         vector<Vector3d, aligned_allocator<Vector3d> >& ns,
                         const Matrix3d& G, const Matrix3d& K, const vector<Match>& matches)
{
    // Convert to Euclidean homography
    Matrix3d H;
    bool pure_rotation = euclideanHomography(H, K, G);

    // For pure rotation, t is zero, n is undefined and R = H
    if (pure_rotation)
    {
      Rs.push_back(H);
      ts.push_back(Vector3d::Zero());
      ns.push_back(Vector3d::Constant(NAN));
      return;
    }

    // Decompose H into 4 possible solutions R, t, n
    decomposeEuclideanHomography(H, Rs, ts, ns);

    // Eliminate impossible solutions
    eliminateInvalidSolutions(matches, K, Rs, ts, ns);
}




// /*================================= MAIN =================================*/

// int main(int argc, char* argv[])
// {
//   // Random parameters
//   auto t0 = high_resolution_clock::now();
//   size_t seed = 0;//time(0);
//   default_random_engine rng(seed);
//   uniform_real_distribution<double> dist(-1.0, 1.0);
//   srand(seed);

//   // Camera parameters
//   double cx = 320;
//   double cy = 240;
//   double half_fov_x = 3.0*M_PI/180.0;
//   double half_fov_y = cy/cx*half_fov_x;
//   double fx = cx/tan(half_fov_x);
//   double fy = cy/tan(half_fov_y);
//   Matrix3d K = Matrix3d::Identity();
//   K(0,0) = fx;
//   K(1,1) = fy;
//   K(0,2) = cx;
//   K(1,2) = cy;


//   double zI_offset = 1000;
//   const unsigned N = 31; // number of points along single grid line
//   const double pix_noise_bound = 0.5; // pixels
//   const double trans_err = 5.0;
//   const double rot_err = 5.0; // degrees
//   const double matched_pts_inlier_ratio = 1.0;
//   const double bound = zI_offset*tan(half_fov_x+rot_err*M_PI/180);

//   size_t num_iters = 1000;
//   size_t num_bad_iters = 0;
//   double error_tol = 0.78;

//   double dt_calc_mean = 0.0; // seconds
//   double dt_calc_var = 0.0; // seconds
//   double rot_error_mean = 0.0;
//   double tran_error_mean = 0.0;
//   double rot_error_var = 0.0;
//   double tran_error_var = 0.0;
//   double match_pts_mean = 0.0;
//   double match_pts_var = 0.0;
//   size_t n_stats = 0;
//   for (size_t iter = 0; iter < num_iters; ++iter)
//   {
//     cout << "Iteration: " << iter+1 << " out of " << num_iters << "\r" << flush;

//     // Camera poses
//     double p1_n = -zI_offset + trans_err*dist(rng);
//     double p1_e = trans_err*dist(rng);
//     double p1_d = trans_err*dist(rng);

//     double p1_r = rot_err*M_PI/180.0*dist(rng);
//     double p1_p = rot_err*M_PI/180.0*dist(rng);
//     double p1_y = rot_err*M_PI/180.0*dist(rng);

//     double p2_n = -zI_offset + trans_err*dist(rng);
//     double p2_e = trans_err*dist(rng);
//     double p2_d = trans_err*dist(rng);

//     double p2_r = rot_err*M_PI/180.0*dist(rng);
//     double p2_p = rot_err*M_PI/180.0*dist(rng);
//     double p2_y = rot_err*M_PI/180.0*dist(rng);

//     common::Transformd x1, x2;
//     x1.p(Vector3d(p1_n, p1_e, p1_d));
//     x2.p(Vector3d(p2_n, p2_e, p2_d));
//     x1.q(common::Quaterniond::fromEuler(p1_r, p1_p, p1_y));
//     x2.q(common::Quaterniond::fromEuler(p2_r, p2_p, p2_y));

//     // Planar points (NED)
//     // - N x N grid within +-bound in east and down directions
//     vector<Point> pts_I = createInertialPoints(N, bound);

//     // Project matching points into each camera image
//     vector<Match> matches;
//     projectAndMatchImagePoints(K, pts_I, x1, x2, pix_noise_bound, matched_pts_inlier_ratio, matches);
//     if (matches.size() < 24)
//     {
//       --iter; // Try that iteration again
//       continue;
//     }

//     // Compute homography and decompose it
//     auto t_calc_0 = high_resolution_clock::now();
//     Matrix3d G;
//     vector<Match> inliers = matches;
//     vector<Matrix3d, aligned_allocator<Matrix3d> > Rs;
//     vector<Vector3d, aligned_allocator<Vector3d> > ns, ts;
//     // homographyFromPointsRANSAC(G, inliers, matches);
//     homographyFromPoints(G, inliers);
//     decomposeHomography(Rs, ts, ns, G, K, inliers);
//     double dt_calc = duration_cast<microseconds>(high_resolution_clock::now() - t_calc_0).count()*1e-6;

//     // Truth
//     Matrix3d R = (q_cb2c.inverse() * x1.q().inverse() * x2.q() * q_cb2c).R();
//     Vector3d t = q_cb2c.rotp(x2.q().rotp((x2.p() - x1.p()) / x1.p()(0)));
//     Vector3d n = q_cb2c.rotp(x1.q().rotp(common::e1));

//     // Error
//     double rot_error, tran_error;
//     bool solution_obtained = false;
//     for (int i = 0; i < Rs.size(); ++i)
//     {
//       if (common::logR(Matrix3d(R.transpose()*Rs[i])).norm() < error_tol ||
//           common::angleBetweenVectors(t, ts[i]) < error_tol || 
//           common::angleBetweenVectors(n, ns[i]) < error_tol)
//       {
//         rot_error = common::logR(Matrix3d(R.transpose()*Rs[i])).norm();
//         tran_error = common::angleBetweenVectors(t, ts[i]);
//         solution_obtained = true;
//         break;
//       }
//     }

//     // Show debug output if solution is not obtained
//     if (!solution_obtained)
//     {
//       ++num_bad_iters;
//       cout << "\n\n\n\n";
//       cout << "True solution not found!\n\n";
//       cout << "Number of inliers: " << inliers.size() << "\n\n";

//       cout << endl;
//       cout << "Errors:\n";
//       for (int i = 0; i < Rs.size(); ++i)
//       {
//         cout << endl;
//         cout << "R" << i << "_err = " << common::logR(Matrix3d(R.transpose()*Rs[i])).norm() << endl;
//         cout << "t" << i << "_err = " << common::angleBetweenVectors(t, ts[i]) << endl;
//         cout << "n" << i << "_err = " << common::angleBetweenVectors(n, ns[i]) << endl;
//       }


//       Vector3d p2_prime = G * inliers[0].p1.vec3();
//       p2_prime /= p2_prime(2);
//       cout << endl;
//       cout << "point  = " << inliers[0].p2.vec3().transpose() << endl;
//       cout << "point' = " << p2_prime.transpose() << endl;

//       // Compute homography from camera geometry
//       cout << endl;
//       cout << "G = \n" << G << endl;
//       cout << "G_from_geometry = \n" << homographyFromGeometry(K, x1, x2) << endl;

//       cout << endl;
//       Matrix3d H;
//       euclideanHomography(H, K, G);
//       cout << "H = \n" << H << endl;
//       cout << "H_from_geometry = \n" << R + t*n.transpose() << endl;

//       cout << endl;
//       cout << "Solutions:\n";
//       for (int i = 0; i < Rs.size(); ++i)
//       {
//         cout << endl;
//         cout << "R" << i << " =\n" << Rs[i] << endl;
//         cout << "t" << i << " = " << ts[i].transpose() << endl;
//         cout << "n" << i << " = " << ns[i].transpose() << endl;
//       }

//       cout << endl;
//       cout << "Truth: \n\n";
//       cout << "R_true = \n" << R << endl;
//       cout << "t_true = " << t.transpose() << endl;
//       cout << "n_true = " << n.transpose() << endl;
//       cout << "t_true.norm() = " << t.norm() << endl;

//       continue; // Bad solutions aren't useful in the following statistics
//     }

//     // Recursive error and variance of things
//     dt_calc_mean = (n_stats*dt_calc_mean + dt_calc)/(n_stats+1);
//     match_pts_mean = (n_stats*match_pts_mean + inliers.size())/(n_stats+1);
//     rot_error_mean = (n_stats*rot_error_mean + rot_error)/(n_stats+1);
//     tran_error_mean = (n_stats*tran_error_mean + tran_error)/(n_stats+1);
//     if (n_stats > 0)
//     {
//       dt_calc_var = ((n_stats-1)*dt_calc_var + pow(dt_calc - dt_calc_mean, 2.0))/n_stats;
//       match_pts_var = ((n_stats-1)*match_pts_var + pow(inliers.size() - match_pts_mean, 2.0))/n_stats;
//       rot_error_var = ((n_stats-1)*rot_error_var + pow(rot_error - rot_error_mean, 2.0))/n_stats;
//       tran_error_var = ((n_stats-1)*tran_error_var + pow(tran_error - tran_error_mean, 2.0))/n_stats;
//     }
//     ++n_stats;
//   }
//   auto tf = duration_cast<microseconds>(high_resolution_clock::now() - t0).count()*1e-6;
//   cout << "\n\n";
//   cout << "                Total time taken: " << tf << " seconds\n";
//   cout << "                 Error tolerance: " << error_tol*180/M_PI << " degrees\n";
//   cout << "        Number of bad iterations: " << num_bad_iters << " out of " << num_iters << endl;
//   cout << "         Calc time (mean, stdev): (" << dt_calc_mean << ", " << sqrt(dt_calc_var) << ") seconds\n";
//   cout << "         match_pts (mean, stdev): (" << match_pts_mean << ", " << sqrt(match_pts_var) << ")\n";
//   cout << "         Rot error (mean, stdev): (" << rot_error_mean*180/M_PI << ", " << sqrt(rot_error_var)*180/M_PI << ") degrees\n";
//   cout << "        Tran error (mean, stdev): (" << tran_error_mean*180/M_PI << ", " << sqrt(tran_error_var)*180/M_PI << ") degrees\n";

//   return 0;
// }
