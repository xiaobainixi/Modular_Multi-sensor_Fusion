#pragma once
#include <Eigen/Core>
#include <Eigen/Geometry>

class OptimizeBasedFusion {
public:
  OptimizeBasedFusion() {}
  // imu_data : acc_x, acc_y, acc_z, gps_x, gps_y, gps_z
  virtual bool AddImuData(double timestamp,
                          const Eigen::Matrix<double, 6, 1> &imu_data) = 0;
  // gps_data : x,y,z in utm
  virtual bool AddGpsData(double timestamp,
                          const Eigen::Matrix<double, 3, 1> &gps_data) = 0;
  virtual bool IsInit() = 0;
  virtual bool Predict(double predict_time, Eigen::Vector3d& p, Eigen::Vector3d& v, Eigen::Quaterniond& q, Eigen::Vector3d& ba, Eigen::Vector3d& bg) = 0;
  virtual bool GetLatestImuTime() {return -1;}
};