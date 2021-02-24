// Testing out camera-imu spatio-temporal initialization
#include "cam_imu_cal_init/cam_imu_cal_init.h"

int main(int argc, char* argv[])
{
    // General parameters
    const int cam_rate = 25;     // Camera update rate (Hz)
    const int imu_rate = 250;    // IMU update rate (Hz)
    const uint32_t t0_ms = 0;    // Initial time stamp (ms)
    const uint32_t tf_ms = 2000; // Final time stamp (ms)
    const int max_delay = 200;   // Maximum possible time delay (ms)

    // Random parameters
    size_t seed = time(0);
    default_random_engine rng(seed);
    uniform_real_distribution<float> dist(-1.0, 1.0);
    srand(seed);

    // Fill measurement containers
    vector<Image> images;
    vector<Imu> imus;
    for (uint32_t t = t0_ms; t < tf_ms; ++t) {
        // Generate Euler angles of body
        Vector3f euler, euler_dot;
        generateEulers(t, euler, euler_dot);

        // Fill IMU
        if (t % (1000/imu_rate) == 0) {
            imus.push_back(buildImu(t, euler, euler_dot));
        }

        // Fill Image
        if (t % (1000/cam_rate) == 0) {
            images.push_back(buildImage(t, euler));
        }
    }

    // Numerically differentiate poses
    vector<Image> images2;
    for (int i = 1; i < images.size()-1; ++i) {
        Image new_img = images[i];
        new_img.dq = (images[i+1].q - images[i-1].q) / (images[i+1].t_ms - images[i-1].t_ms) * 1000.0;
        images2.push_back(new_img);
    }

    // Drop gyro measurements with time stamps outside of new array bounds
    while (imus.begin()->t_ms < images2.begin()->t_ms)
        imus.erase(imus.begin());
    while (imus.back().t_ms > images2.back().t_ms)
        imus.pop_back();

    // Upsample the pose derivatives to match gyro array via linear interpolation
    vector<Image> images3(imus.size());
    auto it0 = images2.begin();
    auto it1 = it0+1;
    for (int i = 0; i < imus.size(); ++i) {
        // Interpolate new image derivative
        Image new_img;
        new_img.t_ms = imus[i].t_ms;
        new_img.dq = it0->dq + (imus[i].t_ms - it0->t_ms) * (it1->dq - it0->dq) / (it1->t_ms - it0->t_ms);
        images3[i] = new_img;
        
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
        mags_cam[i] = images3[i].dq.norm();
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

    cout << "delay = " << ts[delay] - ts[0] << " ms" << endl;

    return 0;
}
