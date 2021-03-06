cmake_minimum_required(VERSION 3.10)
project(cpp_tests)

# default to release build
if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "Build type: Release")
  set(CMAKE_BUILD_TYPE Release)
endif(NOT CMAKE_BUILD_TYPE)
set(CMAKE_CXX_FLAGS "-std=c++11")

find_package(Eigen3 REQUIRED)
find_package(yaml-cpp REQUIRED)
# find_package(GTest REQUIRED)
# find_package(OpenCV REQUIRED)
# find_package(Ceres REQUIRED)

add_subdirectory(lib/common_cpp)

include_directories(include
  lib/common_cpp/include
  ${EIGEN3_INCLUDE_DIRS}
  ${YAML_CPP_INCLUDE_DIR}
  # ${OPENCV_INCLUDE_DIRS}
  # ${CERES_INCLUDE_DIRS}
  )

add_executable(rotation_from_two_images
  src/rotation_from_2_images/rotation_from_two_images.cpp
  src/rotation_from_2_images/util.cpp
  src/rotation_from_2_images/rotation_linear.cpp
  src/rotation_from_2_images/rotation_kneip.cpp
  src/rotation_from_2_images/rotation_sampson.cpp
  src/rotation_from_2_images/homography_decomposition.cpp
  )
target_link_libraries(rotation_from_two_images
  ${YAML_CPP_LIBRARIES}
  )

add_executable(memory_manipulation
  src/memory_manipulation/main.cpp
  )

add_executable(rel_pose
  src/relative_camera_pose/rel_pose.cpp
  )
target_link_libraries(rel_pose
  ${YAML_CPP_LIBRARIES}
  )

add_executable(triangulation
  src/triangulation/triangulation.cpp
  )
target_link_libraries(triangulation
  ${YAML_CPP_LIBRARIES}
  )

add_executable(cam_imu_cal_init
  src/cam_imu_cal_init/cam_imu_cal_init.cpp
  )
target_link_libraries(cam_imu_cal_init
  ${YAML_CPP_LIBRARIES}
  )