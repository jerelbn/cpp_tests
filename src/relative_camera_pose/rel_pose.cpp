// Testing relative pose computations
#include <iostream>
#include <Eigen/Dense>
#include <common_cpp/quaternion.h>

using namespace std;
using namespace Eigen;

struct Point
{
    float x;     // Horizontal pixel position
    float y;     // Vertical pixel position
    float depth; // Distance to the Point
};

Point getUndistortedNormalizedPoint(const Point& pt, const Matrix3f& K, const Matrix<float,5,1>& D)
{
    // Unpack intrinsic parameters
    float fx = K(0,0);
    float fy = K(1,1);
    float cx = K(0,2);
    float cy = K(1,2);

    // Unpack distortion parameters
    float k1 = D(0);
    float k2 = D(1);
    float p1 = D(2);
    float p2 = D(3);
    float k3 = D(4);

    // Iterative solve for undistorted normalized coordinates
    float xd = (pt.x - cx)/fx;
    float yd = (pt.y - cy)/fy;
    float x = xd;
    float y = yd;
    float x_prev = 1e9;
    float y_prev = 1e9;
    int iters = 0;
    const float tol = 1e-6;
    const int max_iters = 1000;
    while (abs(x - x_prev) > tol && abs(y - y_prev) > tol && iters < max_iters)
    {
        x_prev = x;
        y_prev = y;
        float x2 = x * x;
        float y2 = y * y;
        float xy = x * y;
        float r2 = x2 + y2;
        float c1 = 1 + r2 * (k1 + r2 * (k2 + k3 * r2));

        // // Iterative method
        // x = (xd - 2.0 * p1 * xy - p2 * (r2 + 2.0 * x2)) / c1;
        // y = (yd - p1 * (r2 + 2.0 * y2) - 2.0 * p2 * xy) / c1;

        // Nonlinear least squares method
        float c2 = k1 + 2 * r2 * (k2 + 3 * k3 * r2);
        float c3 = 2 * xy * c2 + 2 * p1 * x + 2 * p2 * y;
        Matrix2f J;
        J(0,0) = c1 + 2 * x2 * c2 + 2 * p1 * y + 6 * p2 * x;
        J(0,1) = c3;
        J(1,0) = c3;
        J(1,1) = c1 + 2 * y2 * c2 + 6 * p1 * y + 2 * p2 * x;
        Vector2f err;
        err(0) = x * c1 + 2 * p1 * xy + p2 * (r2 + 2 * x2) - xd;
        err(1) = y * c1 + p1 * (r2 + 2 * y2) + 2 * p2 * xy - yd;
        Vector2f delta = J.householderQr().solve(err);
        x -= delta(0);
        y -= delta(1);

        iters++;
    }

    return Point{x, y, pt.depth};
}

void triangulatePoint(Point& pt1, Point& pt2, const Matrix3f& K, const Matrix<float,5,1>& D,
                      const common::Quaternionf& q, const Vector3f& t)
{
    // Undistort each point
    Point ptu1 = getUndistortedNormalizedPoint(pt1, K, D);
    Point ptu2 = getUndistortedNormalizedPoint(pt2, K, D);

    // Find optimal correspondences (Lindstrom paper)
    static const Eigen::Matrix<float,2,3> S = (Eigen::Matrix<float,2,3>() << 1, 0, 0, 0, 1, 0).finished();
    Vector3f x1(ptu1.x, ptu1.y, 1.0);
    Vector3f x2(ptu2.x, ptu2.y, 1.0);
    Matrix3f E = common::skew(q.rota(-t)) * q.inverse().R();
    Matrix2f E_tilde = S * E * S.transpose();
    Vector2f n1 = S * E * x2;
    Vector2f n2 = S * E.transpose() * x1;
    float a = n1.dot(E_tilde * n2);
    float b = 0.5 * (n1.dot(n1) + n2.dot(n2));
    float c = x1.dot(E * x2);
    float d = sqrt(b * b - a * c);
    float lambda = c / (b + d);
    Vector2f dx1 = lambda * n1;
    Vector2f dx2 = lambda * n2;
    n1 -= E_tilde * dx2;
    n2 -= E_tilde.transpose() * dx1;
    dx1 = dx1.dot(n1) / n1.dot(n1) * n1;
    dx2 = dx2.dot(n2) / n2.dot(n2) * n2;
    x1 -= S.transpose() * dx1;
    x2 -= S.transpose() * dx2;
    
    // Get depth
    Vector3f z = x1.cross(q.rota(x2));
    Vector3f pt1_3d = z.dot(E * x2) / z.dot(z) * x1;
    pt1.depth = pt1_3d.norm();
    pt2.depth = (t + q.rotp(pt1_3d)).norm();
}

Point projectToImage(const Vector3f& lm, const Matrix3f& K, const Matrix<float,5,1>& D)
{
    // Unpack intrinsic parameters
    float fx = K(0,0);
    float fy = K(1,1);
    float cx = K(0,2);
    float cy = K(1,2);

    // Unpack distortion parameters
    float k1 = D(0);
    float k2 = D(1);
    float p1 = D(2);
    float p2 = D(3);
    float k3 = D(4);

    // Get normalized coordinates
    float x = lm(0) / lm(2);
    float y = lm(1) / lm(2);

    // Apply distortion model
    float x2 = x * x;
    float y2 = y * y;
    float xy = x * y;
    float r2 = x2 + y2;
    float c1 = 1 + r2 * (k1 + r2 * (k2 + k3 * r2));
    float xd = x*c1 + 2.0 * p1 * xy + p2 * (r2 + 2.0 * x2);
    float yd = y*c1 + p1 * (r2 + 2.0 * y2) + 2.0 * p2 * xy;

    // Project into image
    Point pt;
    pt.x = fx * xd + cx;
    pt.y = fy * yd + cy;
    pt.depth = lm.norm();

    return pt;
}

Vector3f projectFromImage(const Point& pt, const Matrix3f& K, const Matrix<float,5,1>& D)
{
    // Compute undisorted point in normalized coordinates
    Point ptu = getUndistortedNormalizedPoint(pt, K, D);

    return pt.depth * Vector3f(ptu.x, ptu.y, 1.0).normalized();
}

Point undistortPoint(const Point& pt, const Matrix3f& K, const Matrix<float,5,1>& D)
{
    // Unpack intrinsic parameters
    float fx = K(0,0);
    float fy = K(1,1);
    float cx = K(0,2);
    float cy = K(1,2);

    // Compute undisorted point in normalized coordinates
    Point ptn = getUndistortedNormalizedPoint(pt, K, D);

    // Build undistorted point
    Point ptu;
    ptu.x = fx*ptn.x + cx;
    ptu.y = fy*ptn.y + cy;
    ptu.depth = pt.depth;

    return ptu;
}

Point predictPointLoc(const Point& pt, const Matrix3f& K, const Matrix<float,5,1>& D,
                      const common::Quaternionf& q, const Vector3f& t)
{
    Vector3f lm1 = projectFromImage(pt, K, D);
    Vector3f lm2 = q.rotp(lm1) + t;
    return projectToImage(lm2, K, D);
}


vector<Point> predictPointLocs(const vector<Point>& pts, const Matrix3f& K, const Matrix<float,5,1>& D,
                               const common::Quaternionf& q, const Vector3f& t)
{
    vector<Point> new_pts = pts;
    for (int i = 0; i < new_pts.size(); ++i)
        new_pts[i] = predictPointLoc(pts[i], K, D, q, t);
    return new_pts;
}


float computeError(const Point &pt_hat, const Point &pt) {
    float dx = (int)pt.x - (int)pt_hat.x;
    float dy = (int)pt.y - (int)pt_hat.y;
    return sqrt(dx * dx + dy * dy);
}


template<int N>
Matrix<float,N,1> computeErrors(const vector<Point> &pts2_hat, const vector<Point> &pts2) {
    Matrix<float,N,1> err;
    for (int i = 0; i < N; ++i)
        err(i) = computeError(pts2_hat[i], pts2[i]);
    return err;
}


template<int N>
int solveQT(int iters, float eps, const Matrix3f &K, const Matrix<float,5,1>& D, 
            const vector<Point> &pts1, const vector<Point> &pts2, common::Quaternionf &q_hat, Vector3f &t_hat) {
    // Extra parameters
    const float weight_factor = 0.1;
    
    int ii;
    const Matrix3f I = eps*Matrix3f::Identity();
    Matrix<float,N,6> J;
    Eigen::VectorXf W(N);
    W.setOnes();
    for (ii = 0; ii < iters; ++ii) {
        // Compute radiometric error of each predicted point
        vector<Point> pts2_hat = predictPointLocs(pts1, K, D, q_hat, t_hat);
        Matrix<float,N,1> err = computeErrors<N>(pts2_hat, pts2);

        // Compute Jacobian of error w.r.t. rotation and translation
        for (int i = 0; i < N; ++i)
        {
            // Shift predicted pixel by one in each direction
            Point ptxp = pts2_hat[i];
            ptxp.x += 1;
            Point ptxm = pts2_hat[i];
            ptxm.x -= 1;
            Point ptyp = pts2_hat[i];
            ptyp.y += 1;
            Point ptym = pts2_hat[i];
            ptym.y -= 1;
            
            // Compute radiometric values at shifted locations
            float errxp = computeError(ptxp, pts2[i]);
            float errxm = computeError(ptxm, pts2[i]);
            float erryp = computeError(ptyp, pts2[i]);
            float errym = computeError(ptym, pts2[i]);

            // Numerical derivatives of error w.r.t. pixel location
            Matrix<float,1,2> derr_dp;
            derr_dp(0) = (errxp - errxm) / (ptxp.x - ptxm.x);
            derr_dp(1) = (erryp - errym) / (ptyp.y - ptym.y);

            // Numerical derivative of pixel location w.r.t. rotation and translation
            Matrix<float,2,6> dp_dqt;
            for (int j = 0; j < 3; ++j)
            {
                // Rotation
                common::Quaternionf qp = q_hat +  I.col(j);
                common::Quaternionf qm = q_hat + -I.col(j);
                Point ptqp = predictPointLoc(pts1[i], K, D, qp, t_hat);
                Point ptqm = predictPointLoc(pts1[i], K, D, qm, t_hat);
                dp_dqt(0,j) = (ptqp.x - ptqm.x)/(2.0*eps);
                dp_dqt(1,j) = (ptqp.y - ptqm.y)/(2.0*eps);

                // Translation
                Vector3f tp = t_hat +  I.col(j);
                Vector3f tm = t_hat + -I.col(j);
                Point pttp = predictPointLoc(pts1[i], K, D, q_hat, tp);
                Point pttm = predictPointLoc(pts1[i], K, D, q_hat, tm);
                dp_dqt(0,j+3) = (pttp.x - pttm.x)/(2.0*eps);
                dp_dqt(1,j+3) = (pttp.y - pttm.y)/(2.0*eps);
            }

            // // Analytical derivative of pixel location w.r.t. rotation and translation
            // Matrix<float,2,6> dp_dqt;
            // static const Vector3f e3(0,0,1);
            // static const Matrix3f I3 = Matrix3f::Identity();
            // Vector3f pt1_c = pts1[i].depth * (K_inv * Vector3f(pts1[i].x, pts1[i].y, 1.0)).normalized();
            // Vector3f pt2_c = q_hat.rotp(pt1_c) + t_hat;
            // float z = (K * pt2_c)(2);
            // float zi = 1.0/z;
            // Matrix<float,2,3> M = (zi*K*(I3 - zi*pt2_c*e3.transpose())).topRows(2);
            // dp_dqt.block<2,3>(0,0) = M*common::skew(q_hat.rotp(pt1_c));
            // dp_dqt.block<2,3>(0,3) = M;

            // Chain rule the previous derivatives to form a row of the Jacobian
            J.row(i) = derr_dp * dp_dqt;
        }

        // Ensure invertible Hessian, stabilizing the solution
        Eigen::Matrix<float,6,6> H = J.transpose() * W.asDiagonal() * J;
        H.diagonal() += 0.1*Eigen::Matrix<float,6,1>::Ones();

        // Update R/t estimates
        Eigen::Matrix<float,6,1> delta = H.colPivHouseholderQr().solve(J.transpose()*W.asDiagonal()*err);
        if (delta != delta) {
            cout << delta.transpose() << endl;
            cout << "J = \n" << J << endl;
            cout << "pts1 = \n";
            for (const auto &pt : pts1)
                cout << pt.x << " " << pt.y << " " << pt.depth << endl;
            cout << "pts2 = \n";
            for (const auto &pt : pts2)
                cout << pt.x << " " << pt.y << " " << pt.depth << endl;
            cout << "pts2_hat = \n";
            for (const auto &pt : pts2_hat)
                cout << pt.x << " " << pt.y << " " << pt.depth << endl;
            break;
        }
        q_hat += -delta.segment<3>(0);
        t_hat += -delta.segment<3>(3);

        // The solution has converged when it stops changing
        if (delta.norm() < 1e-6)
            break;

        // Update weights
        for (int i = 0; i < N; ++i) {
            W(i) = 1.0 / std::max(weight_factor, std::abs(err(i)));
        }
    }

    return ii;
}


int main(int argc, char* argv[])
{
    // General parameters
    const int Nmc = 1000; // Number of Monte Carlo runs
    const int Np = 10; // number of points
    const int gn_iters = 50; // maximum number of Gauss Newton iterations
    const float gn_eps = 1e-5; // Nudge for computing derivatives

    // Random parameters
    size_t seed = 0;//time(0);
    default_random_engine rng(seed);
    uniform_real_distribution<float> dist(-1.0, 1.0);
    uniform_real_distribution<float> dist2(0.0, 1.0);
    srand(seed);

    float t_spread = 1.0;
    float r_spread = 5.0*M_PI/180.0;
    float depth_center = 20.0;
    float depth_spread = 15.0;
    float pix_err = 1.0;
    float depth0 = 1.0;
    int num_unknown_depth = 0*Np; // must be <= Np

    // Camera intrinsics and distortion
    int img_width = 1280;
    int img_height = 960;
    float fx = 1146.61; // horizontal focal length (pixels)
    float fy = 1149.34; // vertical focal length (pixels)
    float cx = 679.04;  // horizontal optical center (pixels)
    float cy = 546.44;  // vertical optical center (pixels)

    Matrix3f K = Matrix3f::Identity();
    K(0,0) = fx;
    K(1,1) = fy;
    K(0,2) = cx;
    K(1,2) = cy;

    Matrix3f K_inv = K.inverse();

    Matrix<float,5,1> D;
    D(0) =  8.655555e-02; // first radial distortion coefficient
    D(1) = -1.243856e-01; // second radial distortion coefficient
    D(2) =  1.329729e-03; // first tangential distortion coefficient
    D(3) =  6.591213e-04; // second tangential distortion coefficient
    D(4) = -4.242863e-02; // third radial distortion coefficient

    vector<int> h_iters;
    vector<float> h_qerr;
    vector<float> h_tderr;
    vector<float> h_tmerr;
    for (int jj = 0; jj < Nmc; ++jj) {
        // Camera poses in camera coordinates (right-down-forward)
        // - camera 1 is the origin at null attitude
        Vector3f p1 = Vector3f::Zero();
        common::Quaternionf q1;

        Vector3f p2 = t_spread*Vector3f::Random();
        common::Quaternionf q2 = common::Quaternionf::fromAxisAngle(Vector3f::Random().normalized(), r_spread*dist(rng));

        // Relative pose
        // - rotation 1 to 2
        // - translation 2 to 1 in 2's reference frame
        common::Quaternionf q = q1.inverse() * q2;
        Vector3f t = q2.rotp(p1 - p2);

        // Create points that project into each image
        vector<Point> pts1, pts2;
        while (pts1.size() < Np)
        {
            Point pt1;
            pt1.x = img_width*dist2(rng);
            pt1.y = img_height*dist2(rng);
            pt1.depth = depth_center + depth_spread*dist(rng);
            Point pt2 = predictPointLoc(pt1, K, D, q, t);
            if (pt2.x > 0 && pt2.x < img_width && pt2.y > 0 && pt2.y < img_height)
            {
                pt1.x += pix_err*dist(rng);
                pt1.y += pix_err*dist(rng);
                pt2.x += pix_err*dist(rng);
                pt2.y += pix_err*dist(rng);
                if (pts1.size() < num_unknown_depth) {
                    pt1.depth = depth0;//0.1 + 1000.0 * dist2(rng);
                    pt2.depth = depth0;//0.1 + 1000.0 * dist2(rng);
                }
                else {
                    triangulatePoint(pt1, pt2, K, D, q, t);
                }
                pts1.push_back(pt1);
                pts2.push_back(pt2);
            }
        }
        
        // Solve for relative camera rotation and translation via Gauss Newton optimization
        common::Quaternionf q_hat;
        Vector3f t_hat = Vector3f::Zero();
        int num_iters = solveQT<Np>(gn_iters, gn_eps, K, D, pts1, pts2, q_hat, t_hat);

        h_iters.push_back(num_iters);
        h_qerr.push_back((q - q_hat).norm()*180/M_PI);
        h_tderr.push_back(acos(t.normalized().dot(t_hat.normalized()))*180/M_PI);
        h_tmerr.push_back((t - t_hat).norm());
    }
    
    // Print results
    cout << "num iters: " << std::accumulate(h_iters.begin(), h_iters.end(), 0.)/Nmc << endl;
    cout << "q_err = " << std::accumulate(h_qerr.begin(), h_qerr.end(), 0.)/Nmc << " degrees" << endl;
    cout << "t_dir_err = " << std::accumulate(h_tderr.begin(), h_tderr.end(), 0.)/Nmc << " degrees" << endl;
    cout << "t_mag_err = " << std::accumulate(h_tmerr.begin(), h_tmerr.end(), 0.)/Nmc << " meters" << endl;

    return 0;
}
