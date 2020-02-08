/*

Rotation From Two Images - calculate only the rotation matrix from matched points in two images

*/
#include "util.h"
#include "rotation_linear.h"
#include "rotation_kneip.h"
#include "rotation_sampson.h"
#include "homography_decomposition.h"



int main(int argc, char* argv[])
{
  // Random parameters
  auto t0 = high_resolution_clock::now();
  size_t seed = 0;//time(0);
  default_random_engine rng(seed);
  uniform_real_distribution<double> dist(-1.0, 1.0);
  srand(seed);

  // Camera parameters
  double cx = 320;
  double cy = 240;
  double half_fov_x = 3.0*M_PI/180.0;
  double half_fov_y = cy/cx*half_fov_x;
  double fx = cx/tan(half_fov_x);
  double fy = cy/tan(half_fov_y);
  Matrix3d K = Matrix3d::Identity();
  K(0,0) = fx;
  K(1,1) = fy;
  K(0,2) = cx;
  K(1,2) = cy;


  // solver:
  // - 0: Ha solver
  // - 1: Ha solver w/ RANSAC
  // - 2: Kneip solver
  // - 3: Kneip solver w/ RANSAC
  // - 4: Sampson
  // - 5: Homography decomposition
  int solver = 4;

  double zI_offset = 1000;
  const unsigned N = 31; // number of points along single grid line
  const double pix_noise_bound = 0.5; // pixels
  const double trans_err = 5.0; // meters
  const double rot_err = 5.0; // degrees
  const double matched_pts_inlier_ratio = 1.0;
  const double bound = zI_offset*tan(half_fov_x+rot_err*M_PI/180);

  size_t num_iters = 1000;
  size_t num_bad_iters = 0;
  double R_error_tol = 30.0*M_PI/180;
  double t_error_tol = 180*M_PI/180;


  double dt_calc_mean = 0.0; // seconds
  double dt_calc_var = 0.0; // seconds
  double rot_error_mean = 0.0;
  double tran_error_mean = 0.0;
  double rot_error_var = 0.0;
  double tran_error_var = 0.0;
  double reproj_error_mean = 0.0;
  double reproj_error_var = 0.0;
  double lambda_min_mean = 0.0;
  double lambda_min_var = 0.0;
  double match_pts_mean = 0.0;
  double match_pts_var = 0.0;
  size_t n_stats = 0;
  for (size_t iter = 0; iter < num_iters; ++iter)
  {
    cout << "Iteration: " << iter+1 << " out of " << num_iters << "\r" << flush;

    // Camera poses
    double p1_n = -zI_offset + trans_err*dist(rng);
    double p1_e = trans_err*dist(rng);
    double p1_d = trans_err*dist(rng);

    double p1_r = rot_err*M_PI/180.0*dist(rng);
    double p1_p = rot_err*M_PI/180.0*dist(rng);
    double p1_y = rot_err*M_PI/180.0*dist(rng);

    double p2_n = -zI_offset + trans_err*dist(rng);
    double p2_e = trans_err*dist(rng);
    double p2_d = trans_err*dist(rng);

    double p2_r = rot_err*M_PI/180.0*dist(rng);
    double p2_p = rot_err*M_PI/180.0*dist(rng);
    double p2_y = rot_err*M_PI/180.0*dist(rng);

    common::Transformd x1, x2;
    x1.p(Vector3d(p1_n, p1_e, p1_d));
    x2.p(Vector3d(p2_n, p2_e, p2_d));
    x1.q(common::Quaterniond::fromEuler(p1_r, p1_p, p1_y));
    x2.q(common::Quaterniond::fromEuler(p2_r, p2_p, p2_y));

    // Planar points (NED)
    // - N x N grid within +-bound in east and down directions
    vector<Point> pts_I = createInertialPoints(N, bound);

    // Project matching points into each camera image
    vector<Match> matches;
    projectAndMatchImagePoints(K, pts_I, x1, x2, pix_noise_bound, matched_pts_inlier_ratio, matches);
    if (matches.size() < 24)
    {
      --iter; // Try that iteration again
      continue;
    }

    // Truth
    common::Quaterniond q = q_cb2c.inverse() * x1.q().inverse() * x2.q() * q_cb2c;
    Vector3d t = q_cb2c.rotp(x2.q().rotp(x1.p() - x2.p()));
    Vector3d n = q_cb2c.rotp(x1.q().rotp(common::e1));

    // Run estimator
    common::Quaterniond q_hat, qt_hat;
    Vector3d et_hat = Vector3d::Random().normalized();
    vector<Match> inliers = matches;
    bool solution_obtained = true; // Needed for homography decomposition
    auto t_calc_0 = high_resolution_clock::now();
    if (solver == 0)
    {
      rotationFromPointsHa(q_hat, K, inliers);
      et_hat = t.normalized();
    }
    else if (solver == 1)
    {
      rotationFromPointsHaRANSAC(q_hat, inliers, K, matches);
      et_hat = t.normalized();
    }
    else if (solver == 2)
    {
      rotationFromPointsKneip(q_hat, K, inliers, 50, 1e-6, 1e-6, 10, 0.02);
      et_hat = t.normalized();
    }
    else if (solver == 3)
    {
      rotationFromPointsKneipRANSAC(q_hat, inliers, K, matches, 50, 1e-6, 1e-6, 10, 0.02, 16, 1e-9);
      et_hat = t.normalized();
    }
    else if (solver == 4)
    {
      // et_hat = t.normalized();
      qt_hat = common::Quaterniond::fromUnitVector(et_hat);
      rotationFromPointsSampson(q_hat, qt_hat, K, matches, 5, 1e-6, 1e-6, 10);
      et_hat = qt_hat.uvec();
    }
    else if (solver == 5)
    {
      Matrix3d G;
      vector<Matrix3d, aligned_allocator<Matrix3d> > Rs;
      vector<Vector3d, aligned_allocator<Vector3d> > ns, ts;
      // homographyFromPointsRANSAC(G, inliers, matches);
      homographyFromPoints(G, inliers);
      decomposeHomography(Rs, ts, ns, G, K, inliers);
      if (Rs.size() > 0)
      {
        solution_obtained = true;

        // Keep solution closest to truth
        double error_prev = 9999;
        for (int i = 0; i < Rs.size(); ++i)
        {
          double error = common::logR(Matrix3d(Rs[i].transpose()*q.R())).norm();
          if (error < error_prev)
          {
            q_hat = common::Quaterniond::fromRotationMatrix(Rs[i]);
            et_hat = ts[i].normalized();
            error_prev = error;
          }
        }
      }
      else
      {
        Vector3d p2_prime = G * inliers[0].p1.vec3();
        p2_prime /= p2_prime(2);
        cout << endl;
        cout << "point  = " << inliers[0].p2.vec3().transpose() << endl;
        cout << "point' = " << p2_prime.transpose() << endl;

        cout << endl;
        cout << "G = \n" << G << endl;
        cout << "G_from_geometry = \n" << homographyFromGeometry(K, x1, x2) << endl;

        cout << endl;
        Matrix3d H;
        euclideanHomography(H, K, G);
        cout << "H = \n" << H << endl;
        cout << "H_from_geometry = \n" << q.R() + t*n.transpose() << endl;

        cout << endl;
        cout << "Truth: \n\n";
        cout << "R_true = \n" << q.R() << endl;
        cout << "t_true = " << t.transpose() << endl;
        cout << "n_true = " << n.transpose() << endl;
        cout << "t_true.norm() = " << t.norm() << endl;
      }
    }
    else
      throw runtime_error("Select an existing solver!");
    double dt_calc = duration_cast<microseconds>(high_resolution_clock::now() - t_calc_0).count()*1e-6;
 
    // Compute error in relative pose
    double rot_error = Vector3d(q - q_hat).norm();
    double tran_error = common::angleBetweenVectors(t, et_hat);

    // Reprojection error ignoring translation
    Matrix3d K_inv = K.inverse();
    vector<double> reproj_errs(inliers.size());
    int idx = 0;
    for (int i = 0; i < inliers.size(); ++i)
    {
      if (inliers[i].p1.id > 999999) continue;
      Vector3d k1 = K_inv*inliers[i].p1.vec3();
      Vector3d reproj_no_tran_true = K*q.rotp(k1); reproj_no_tran_true /= reproj_no_tran_true(2);
      Vector3d reproj_no_tran_hat = K*q_hat.rotp(k1); reproj_no_tran_hat /= reproj_no_tran_hat(2);
      reproj_errs.push_back((reproj_no_tran_true - reproj_no_tran_hat).norm());
    }
    double reproj_error = accumulate(reproj_errs.begin(), reproj_errs.end(), 0.0)/reproj_errs.size();

    // Show debug output if solution is not close enough to truth
    if (solution_obtained && (rot_error > R_error_tol || tran_error > t_error_tol))
    {
      ++num_bad_iters;
      cout << "\n\n";
      cout << "            Calc time taken: " << dt_calc << " seconds\n";
      cout << "    Number of point inliers: " << inliers.size() << "\n";
      cout << "             Rotation truth: " << common::Quaterniond::log(q).norm()*180/M_PI << " degrees\n";
      cout << "             Rotation error: " << rot_error*180/M_PI << " degrees\n";
      cout << "Translation direction error: " << tran_error*180/M_PI << " degrees\n";
      cout << "                      q_hat: " << q_hat.toEigen().transpose() << "\n";
      cout << "                     q_true: " << q.toEigen().transpose() << "\n";
      cout << "                     et_hat: " << et_hat.transpose() << "\n";
      cout << "                    et_true: " << t.normalized().transpose() << "\n";
      cout << "                     t_true: " << t.transpose() << "\n";
      cout << "                      t_mag: " << t.norm() << "\n\n";
      continue; // Bad solutions aren't useful in the following statistics
    }

    // Recursive error and variance of things
    dt_calc_mean = (n_stats*dt_calc_mean + dt_calc)/(n_stats+1);
    match_pts_mean = (n_stats*match_pts_mean + inliers.size())/(n_stats+1);
    rot_error_mean = (n_stats*rot_error_mean + rot_error)/(n_stats+1);
    tran_error_mean = (n_stats*tran_error_mean + tran_error)/(n_stats+1);
    reproj_error_mean = (n_stats*reproj_error_mean + reproj_error)/(n_stats+1);
    if (n_stats > 0)
    {
      dt_calc_var = ((n_stats-1)*dt_calc_var + pow(dt_calc - dt_calc_mean, 2.0))/n_stats;
      match_pts_var = ((n_stats-1)*match_pts_var + pow(inliers.size() - match_pts_mean, 2.0))/n_stats;
      rot_error_var = ((n_stats-1)*rot_error_var + pow(rot_error - rot_error_mean, 2.0))/n_stats;
      tran_error_var = ((n_stats-1)*tran_error_var + pow(tran_error - tran_error_mean, 2.0))/n_stats;
      reproj_error_var = ((n_stats-1)*reproj_error_var + pow(reproj_error - reproj_error_mean, 2.0))/n_stats;
    }
    ++n_stats;
  }
  auto tf = duration_cast<microseconds>(high_resolution_clock::now() - t0).count()*1e-6;
  cout << "\n\n";
  cout << "                Total time taken: " << tf << " seconds\n";
  cout << "        Rotation error tolerance: " << R_error_tol*180/M_PI << " degrees\n";
  cout << "     Translation error tolerance: " << t_error_tol*180/M_PI << " degrees\n";
  cout << "        Number of bad iterations: " << num_bad_iters << " out of " << num_iters << endl;
  cout << "         Calc time (mean, stdev): (" << dt_calc_mean << ", " << sqrt(dt_calc_var) << ") seconds\n";
  cout << "         match_pts (mean, stdev): (" << match_pts_mean << ", " << sqrt(match_pts_var) << ")\n";
  cout << "         Rot error (mean, stdev): (" << rot_error_mean*180/M_PI << ", " << sqrt(rot_error_var)*180/M_PI << ") degrees\n";
  cout << "        Tran error (mean, stdev): (" << tran_error_mean*180/M_PI << ", " << sqrt(tran_error_var)*180/M_PI << ") degrees\n";
  cout << "Reprojection error (mean, stdev): (" << reproj_error_mean*180/M_PI << ", " << sqrt(reproj_error_var)*180/M_PI << ")\n\n";

  return 0;
}
