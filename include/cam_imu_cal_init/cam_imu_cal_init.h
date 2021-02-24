// Testing out camera-imu spatio-temporal initialization
#pragma once

#include <iostream>
#include <eigen3/Eigen/Dense>
#include <common_cpp/quaternion.h>

using namespace std;
using namespace Eigen;

const float Ar = 1.1;
const float Ap = 1.2;
const float Ay = 1.3;
const float Pr = 0.1;
const float Pp = 0.2;
const float Py = 0.3;
const common::Quaternionf q_bc = common::Quaternionf::fromEuler(0.1,0.2,0.3);
const uint32_t time_delay = 131;

struct Image {
    uint32_t t_ms;
    common::Quaternionf q;
    Vector3f dq;
};

struct Imu {
    uint32_t t_ms;
    Vector3f accel;
    Vector3f gyro;
};

void generateEulers(uint32_t t_ms, Vector3f &euler, Vector3f &euler_dot) {
    float t = t_ms / 1000.0;

    float r  =  Ar * sin(t - Pr);
    float dr =  Ar * cos(t - Pr);

    float p  =  Ap * cos(t - Pp);
    float dp = -Ap * sin(t - Pp);

    float y  =  Ay * sin(t - Py);
    float dy =  Ay * cos(t - Py);

    euler = Vector3f(r,p,y);
    euler_dot = Vector3f(dr,dp,dy);
}

Imu buildImu(uint32_t t_ms, const Vector3f &euler, const Vector3f &euler_dot) {
    Imu imu;
    imu.t_ms = t_ms;
    imu.gyro = common::R_euler_rate_to_body_rate(euler(0),euler(1),321) * euler_dot;
    return imu;
}

Image buildImage(uint32_t t_ms, const Vector3f &euler) {
    Image img;
    img.t_ms = t_ms + time_delay;
    img.q = common::Quaternionf::fromEuler(euler(0), euler(1), euler(2)) * q_bc;
    return img;
}