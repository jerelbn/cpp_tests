// Testing out camera-imu spatio-temporal initialization
#include "cam_imu_cal_init/cam_imu_cal_init.h"

int main(int argc, char* argv[])
{
    // General parameters
    const int cam_rate = 25;     // Camera update rate (Hz)
    const int imu_rate = 250;    // IMU update rate (Hz)
    const uint32_t t0_ms = 1000;    // Initial time stamp (ms)
    const uint32_t tf_ms = 3000; // Final time stamp (ms)
    const int max_delay = 200;   // Maximum possible time delay (ms)

    // Truth parameters
    size_t seed = time(0);
    srand(seed);
    const Vector3f eulers = M_PI/2*Vector3f::Random();
    const common::Quaternionf q_bc = common::Quaternionf::fromEuler(eulers(0), eulers(1), eulers(2));
    const uint32_t true_delay = rand() % max_delay;

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
            images.push_back(buildImage(t, euler, true_delay, q_bc));
        }
    }

    // Estimate time delay and adjust image time stamps accordingly
    uint32_t delay = computeTimeDelay(images, imus, max_delay);
    for (auto &img : images)
        img.t_ms -= delay;

    // Estimate camera to IMU rotation
    common::Quaternionf q_bc_hat = estimateCamToImuRotation(images, imus);

    cout << "delay_est = " << delay << " ms" << endl;
    cout << "delay_tru = " << true_delay << " ms" << endl;
    cout << "q_bc_est = " << q_bc_hat.toEigen().transpose() << endl;
    cout << "q_bc_tru = " << q_bc.toEigen().transpose() << endl;

    return 0;
}
