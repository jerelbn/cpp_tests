#include "rotation_from_2_images/homography_decomposition.h"


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
  // NOTE: This cannot happen on real images due to image noise
  if (abs(M(0,0) - M(1,1)) > 1.0 || abs(M(0,0) - M(2,2)) > 1.0)
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
