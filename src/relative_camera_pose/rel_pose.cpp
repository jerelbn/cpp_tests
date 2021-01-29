// Testing relative pose computations
#include <relative_camera_pose/util.hpp>

struct Point
{
    float x;     // Horizontal pixel position
    float y;     // Vertical pixel position
    float depth; // Distance to the Point
};


Point predictPointLoc(const Point& pt, const Matrix3f& K, const Matrix3f& K_inv,
                      const common::Quaternionf& q, const Vector3f& t)
{
    Point new_pt = pt;
    Vector3f pt_h(pt.x, pt.y, 1.0);
    Vector3f pt_c = pt.depth * (K_inv * pt_h).normalized();
    Vector3f new_pt_c = q.rotp(pt_c) + t;
    Vector3f new_pt_h = K*new_pt_c;
    new_pt_h /= new_pt_h(2);
    new_pt.x = new_pt_h(0);
    new_pt.y = new_pt_h(1);
    new_pt.depth = new_pt_c.norm();
    return new_pt;
}


vector<Point> predictPointLocs(const vector<Point>& pts, const Matrix3f& K, const Matrix3f& K_inv,
                               const common::Quaternionf& q, const Vector3f& t)
{
    vector<Point> new_pts = pts;
    for (int i = 0; i < new_pts.size(); ++i)
        new_pts[i] = predictPointLoc(pts[i], K, K_inv, q, t);
    return new_pts;
}


float computeError(const Point &pt_hat, const Point &pt) {
    float dx = pt.x - pt_hat.x;
    float dy = pt.y - pt_hat.y;
    return dx * dx + dy * dy;
}


template<int N>
Matrix<float,N,1> computeErrors(const vector<Point> &pts2_hat, const vector<Point> &pts2) {
    Matrix<float,N,1> err;
    for (int i = 0; i < N; ++i)
        err(i) = computeError(pts2_hat[i], pts2[i]);
    return err;
}


template<int N>
int solveQT(int iters, float eps, const Matrix3f &K, const Matrix3f &K_inv, 
            const vector<Point> &pts1, const vector<Point> &pts2, common::Quaternionf &q_hat, Vector3f &t_hat) {
    int ii;
    const Matrix3f I = eps*Matrix3f::Identity();
    for (ii = 0; ii < iters; ++ii) {
        // Compute Jacobian of error w.r.t. rotation and translation
        Matrix<float,N,6> J;
        for (int i = 0; i < 3; ++i)
        {
            common::Quaternionf qp = q_hat +  I.col(i);
            common::Quaternionf qm = q_hat + -I.col(i);
            vector<Point> pts2p = predictPointLocs(pts1, K, K_inv, qp, t_hat);
            vector<Point> pts2m = predictPointLocs(pts1, K, K_inv, qm, t_hat);
            Matrix<float,N,1> errp = computeErrors<N>(pts2p , pts2);
            Matrix<float,N,1> errm = computeErrors<N>(pts2m , pts2);
            J.col(i) = (errp - errm)/(2.0*eps);
        }
        for (int i = 0; i < 3; ++i)
        {
            Vector3f tp = t_hat +  I.col(i);
            Vector3f tm = t_hat + -I.col(i);
            vector<Point> pts2p = predictPointLocs(pts1, K, K_inv, q_hat, tp);
            vector<Point> pts2m = predictPointLocs(pts1, K, K_inv, q_hat, tm);
            Matrix<float,N,1> errp = computeErrors<N>(pts2p , pts2);
            Matrix<float,N,1> errm = computeErrors<N>(pts2m , pts2);
            J.col(i+3) = (errp - errm)/(2.0*eps);
        }

        // Computer current error
        vector<Point> pts2_hat = predictPointLocs(pts1, K, K_inv, q_hat, t_hat);
        Matrix<float,N,1> err = computeErrors<N>(pts2_hat, pts2);

        // Update R/t estimates
        Matrix<float,6,1> delta = J.completeOrthogonalDecomposition().solve(err);
        q_hat += -delta.segment<3>(0);
        t_hat += -delta.segment<3>(3);

        if (delta.norm() < 1e-6)
            break;
    }

    return ii;
}


template<int N>
int solveQT2(int iters, float eps, const Matrix3f &K, const Matrix3f &K_inv, 
             const vector<Point> &pts1, const vector<Point> &pts2, common::Quaternionf &q_hat, Vector3f &t_hat) {
    int ii;
    const Matrix3f I = eps*Matrix3f::Identity();
    for (ii = 0; ii < iters; ++ii) {
        // Compute radiometric error of each predicted point
        vector<Point> pts2_hat = predictPointLocs(pts1, K, K_inv, q_hat, t_hat);
        Matrix<float,N,1> err = computeErrors<N>(pts2_hat, pts2);

        // Compute Jacobian of error w.r.t. rotation and translation
        Matrix<float,N,6> J;
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

            // // Numerical derivative of pixel location w.r.t. rotation and translation
            // Matrix<float,2,6> dp_dqt;
            // for (int j = 0; j < 3; ++j)
            // {
            //     // Rotation
            //     common::Quaternionf qp = q_hat +  I.col(j);
            //     common::Quaternionf qm = q_hat + -I.col(j);
            //     Point ptqp = predictPointLoc(pts1[i], K, K_inv, qp, t_hat);
            //     Point ptqm = predictPointLoc(pts1[i], K, K_inv, qm, t_hat);
            //     dp_dqt(0,j) = (ptqp.x - ptqm.x)/(2.0*eps);
            //     dp_dqt(1,j) = (ptqp.y - ptqm.y)/(2.0*eps);

            //     // Translation
            //     Vector3f tp = t_hat +  I.col(j);
            //     Vector3f tm = t_hat + -I.col(j);
            //     Point pttp = predictPointLoc(pts1[i], K, K_inv, q_hat, tp);
            //     Point pttm = predictPointLoc(pts1[i], K, K_inv, q_hat, tm);
            //     dp_dqt(0,j+3) = (pttp.x - pttm.x)/(2.0*eps);
            //     dp_dqt(1,j+3) = (pttp.y - pttm.y)/(2.0*eps);
            // }

            // Analytical derivative of pixel location w.r.t. rotation and translation
            Matrix<float,2,6> dp_dqt;
            static const Vector3f e3(0,0,1);
            static const Matrix3f I3 = Matrix3f::Identity();
            Vector3f pt1_c = pts1[i].depth * (K_inv * Vector3f(pts1[i].x, pts1[i].y, 1.0)).normalized();
            Vector3f pt2_c = q_hat.rotp(pt1_c) + t_hat;
            float z = (K * pt2_c)(2);
            float zi = 1.0/z;
            Matrix<float,2,3> M = (zi*(I3 - zi*K*pt2_c*e3.transpose())*K).topRows(2);
            dp_dqt.block<2,3>(0,0) = M*common::skew(q_hat.rotp(pt1_c));
            dp_dqt.block<2,3>(0,3) = M;

            // Chain rule the previous derivatives to form a row of the Jacobian
            J.row(i) = derr_dp * dp_dqt;
        }

        // Update R/t estimates
        Matrix<float,6,1> delta = J.completeOrthogonalDecomposition().solve(err);
        q_hat += -delta.segment<3>(0);
        t_hat += -delta.segment<3>(3);

        // The solution has converged when it stops changing
        if (delta.norm() < 1e-6)
            break;
    }

    return ii;
}


int main(int argc, char* argv[])
{
    // General parameters
    const int Np = 10; // number of points
    const int gn_iters = 1000; // maximum number of Gauss Newton iterations
    const float gn_eps = 1e-5; // Nudge for computing derivatives

    // Random parameters
    size_t seed = 0;//time(0);
    default_random_engine rng(seed);
    uniform_real_distribution<float> dist(-1.0, 1.0);
    srand(seed);

    float t_err = 1.0;
    float r_err = 5.0*M_PI/180.0;
    float pts_spread = 10.0;
    float pts_offset_z = 30.0;

    // Camera intrinsics and distortion
    float img_width = 640;
    float img_height = 512;
    float cx = img_width/2;
    float cy = img_height/2;
    float fov_x = 60.0*M_PI/180.0;
    float fov_y = cy/cx*fov_x;
    float fx = cx/tan(fov_x/2);
    float fy = cy/tan(fov_y/2);

    Matrix3f K = Matrix3f::Identity();
    K(0,0) = fx;
    K(1,1) = fy;
    K(0,2) = cx;
    K(1,2) = cy;

    Matrix3f K_inv = K.inverse();

    Matrix<float,5,1> D;
    D(0) = 0;
    D(1) = 0;
    D(2) = 0;
    D(3) = 0;
    D(4) = 0;

    // Camera poses in camera coordinates (right-down-forward)
    // - camera 1 is the origin at null attitude
    Vector3f p1 = Vector3f::Zero();
    common::Quaternionf q1;

    Vector3f p2 = t_err*Vector3f::Random();
    common::Quaternionf q2 = common::Quaternionf::fromAxisAngle(Vector3f::Random().normalized(), r_err*dist(rng));

    // Relative pose
    // - rotation 1 to 2
    // - translation 2 to 1 in 2's reference frame
    common::Quaternionf q = q1.inverse() * q2;
    Vector3f t = q2.rotp(p1 - p2);

    // Points in each camera frame
    Matrix<float,3,Np> lms1 = pts_spread*Matrix<float,3,Np>::Random();
    lms1.row(2) += pts_offset_z*Matrix<float,1,Np>::Ones();
    Matrix<float,3,Np> lms2;
    for (int i = 0; i < Np; ++i) {
        lms2.col(i) = q2.rotp(p1 - p2 + lms1.col(i));
    }

    // Project landmarks into each camera image
    vector<Point> pts1(Np), pts2(Np);
    for (int i = 0; i < Np; ++i) {
        pts1[i].x = fx*lms1(0,i)/lms1(2,i) + cx;
        pts1[i].y = fy*lms1(1,i)/lms1(2,i) + cy;
        pts1[i].depth = lms1.col(i).norm();
        pts2[i].x = fx*lms2(0,i)/lms2(2,i) + cx;
        pts2[i].y = fy*lms2(1,i)/lms2(2,i) + cy;
        pts2[i].depth = lms2.col(i).norm();
    }
    
    // Solve for relative camera rotation and translation via Gauss Newton optimization
    common::Quaternionf q_hat;
    Vector3f t_hat = Vector3f::Zero();
    // int num_iters = solveQT<Np>(gn_iters, gn_eps, K, K_inv, pts1, pts2, q_hat, t_hat);
    int num_iters = solveQT2<Np>(gn_iters, gn_eps, K, K_inv, pts1, pts2, q_hat, t_hat);
    
    // Print results
    cout << "num iters: " << num_iters << endl;
    cout << "q = \n" << q.toEigen() << endl;
    cout << "q_hat = \n" << q_hat.toEigen() << endl;
    cout << "t = \n" << t << endl;
    cout << "t_hat = \n" << t_hat << endl;

    return 0;
}
