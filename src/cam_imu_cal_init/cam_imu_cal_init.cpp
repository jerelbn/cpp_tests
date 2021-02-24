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

    // Truth parameters
    size_t seed = time(0);
    srand(seed);
    const common::Quaternionf q_bc = common::Quaternionf::fromEuler(0.1,0.2,0.3);
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

    uint32_t delay = computeTimeDelay(images, imus, max_delay);

    cout << "est delay =  " << delay << " ms" << endl;
    cout << "true delay = " << true_delay << " ms" << endl;

    return 0;
}
