// Testing relative pose computations
#include "triangulation/triangulation.h"

int main(int argc, char* argv[])
{
    // General parameters
    const int Np = 10; // number of points

    // Random parameters
    size_t seed = time(0);
    default_random_engine rng(seed);
    uniform_real_distribution<float> dist(-1.0, 1.0);
    srand(seed);

    float t_err = 5.0;
    float r_err = 5.0*M_PI/180.0;
    float pts_spread = 10.0;
    float pts_offset_z = 30.0;
    float pixel_err = 1.0;

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
        pts1[i].x = fx*lms1(0,i)/lms1(2,i) + cx + pixel_err*randi();
        pts1[i].y = fy*lms1(1,i)/lms1(2,i) + cy + pixel_err*randi();
        pts1[i].depth = lms1.col(i).norm();
        pts2[i].x = fx*lms2(0,i)/lms2(2,i) + cx + pixel_err*randi();
        pts2[i].y = fy*lms2(1,i)/lms2(2,i) + cy + pixel_err*randi();
        pts2[i].depth = lms2.col(i).norm();
    }
    
    // Solve for optical depth of each point
    vector<Point> pts1_hat = pts1;
    vector<Point> pts2_hat = pts2;
    for (int i = 0; i < Np; ++i) {
        pts1_hat[i].depth = 0;
        pts2_hat[i].depth = 0;
        triangulatePoint(pts1_hat[i], pts2_hat[i], K_inv, q, t);
    }
    
    // Print percent errors
    printf("Percent Errors (error/depth):\n");
    for (int i = 0; i < Np; ++i) {
        printf("pt%d: %10.4f %10.4f\n", i, 
            100*abs(pts1[i].depth-pts1_hat[i].depth)/pts1[i].depth, 
            100*abs(pts2[i].depth-pts2_hat[i].depth)/pts2[i].depth);
    }

    return 0;
}
