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

Image buildImage(uint32_t t_ms, const Vector3f &euler, uint32_t time_delay, const common::Quaternionf &q_bc) {
    Image img;
    img.t_ms = t_ms + time_delay;
    img.q = common::Quaternionf::fromEuler(euler(0), euler(1), euler(2)) * q_bc;
    return img;
}

uint32_t computeTimeDelay(const vector<Image> &images, vector<Imu> imus, uint32_t max_delay) {
    // Numerically differentiate poses
    vector<Image> dqs;
    for (int i = 1; i < images.size()-1; ++i) {
        Image new_img = images[i];
        new_img.dq = (images[i+1].q - images[i-1].q) / (images[i+1].t_ms - images[i-1].t_ms) * 1000.0;
        dqs.push_back(new_img);
    }

    // Drop gyro measurements with time stamps outside of new array bounds
    while (imus.begin()->t_ms < dqs.begin()->t_ms)
        imus.erase(imus.begin());
    while (imus.back().t_ms > dqs.back().t_ms)
        imus.pop_back();

    // Upsample the pose derivatives to match gyro array via linear interpolation
    vector<Image> dqs_upsampled(imus.size());
    auto it0 = dqs.begin();
    auto it1 = it0+1;
    for (int i = 0; i < imus.size(); ++i) {
        // Interpolate new image derivative
        Image new_img;
        new_img.t_ms = imus[i].t_ms;
        new_img.dq = it0->dq + (imus[i].t_ms - it0->t_ms) * (it1->dq - it0->dq) / (it1->t_ms - it0->t_ms);
        dqs_upsampled[i] = new_img;
        
        // Shift iterators when needed
        if (imus[i].t_ms > it1->t_ms) {
            ++it0;
            ++it1;
        }
    }

    // Compute magnitudes if angular rates
    vector<float> mags_imu(imus.size());
    vector<float> mags_cam(imus.size());
    vector<uint32_t> ts(imus.size());
    for (int i = 0; i < imus.size(); ++i) {
        mags_imu[i] = imus[i].gyro.norm();
        mags_cam[i] = dqs_upsampled[i].dq.norm();
        ts[i] = imus[i].t_ms;
    }

    // Perform cross correlation to find time delay
    vector<float> x = mags_imu;
    vector<float> y = mags_cam;
    int N = x.size();
    
    // Compute mean of each array
    float mx = std::accumulate(x.begin(), x.end(), 0) / N;
    float my = std::accumulate(y.begin(), y.end(), 0) / N;

    // Calculate the correlation series
    vector<float> rs;
    int delay = 0;
    float r_max = -99999;
    for (int d = 0; d < max_delay; ++d) {
        float sxy = 0;
        float sx = 0;
        float sy = 0;
        for (int i = 0; i < N; ++i) {
            int j = i + d;
            if (j < 0 || j >= N)
                continue;
            else {
                sxy += (x[i] - mx) * (y[j] - my);
                sx += (x[i] - mx) * (x[i] - mx);
                sy += (y[j] - my) * (y[j] - my);
            }
        }
        float r = sxy / sqrt(sx*sy);
        rs.push_back(r);
        if (r > r_max) {
            r_max = r;
            delay = d;
        }
    }

    return ts[delay] - ts[0];
}