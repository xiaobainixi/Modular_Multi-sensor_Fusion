#pragma once
#include "ceres_core.hpp"
#include "optimize_fusion.h"

class CeresBasedFusionInterface : public OptimizeBasedFusion {
public:
  CeresBasedFusionInterface() : OptimizeBasedFusion() {
    // TODO : 通过配置文件来initialization
    fusion_core_ptr_ = std::make_shared<CeresFusion>();
  }

  virtual bool AddImuData(double timestamp,
                          const Eigen::Matrix<double, 6, 1> &imu_data) {
    fusion_core_ptr_->AddImuData(timestamp, imu_data);
    return true;
  }

  virtual bool AddGpsData(double timestamp,
                          const Eigen::Matrix<double, 3, 1> &gps_data) {
    fusion_core_ptr_->AddGpsData(timestamp, gps_data);
    return true;
  }

  virtual bool IsInit() {
    return fusion_core_ptr_->IsInit();
  }

  virtual bool Predict(double predict_time, Eigen::Vector3d &p,
                       Eigen::Vector3d &v, Eigen::Quaterniond &q, Eigen::Vector3d &ba, Eigen::Vector3d &bg) {
    if (!IsInit()) {
      return false;
    }

    CeresFusion::NavState state;
    fusion_core_ptr_->GetLastestState(state);
    predict_time = state.timestamp;
    p = state.p;
    v = state.v;
    ba = state.ba;
    bg = state.bg;

    return true;
  }

  virtual bool GetLatestImuTime() {return fusion_core_ptr_->latest_imu_time_;}

private:
  std::shared_ptr<CeresFusion> fusion_core_ptr_;
};