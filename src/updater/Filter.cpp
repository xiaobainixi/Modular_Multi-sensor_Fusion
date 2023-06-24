#pragma once
#include "Filter.h"

void Filter::Update(const std::shared_ptr<State> & state_ptr, const GPSData & gps_data) {
    Eigen::VectorXd X;
    Eigen::MatrixXd K;
    Eigen::MatrixXd H;
    Eigen::MatrixXd Z;
    Eigen::MatrixXd R;
    Eigen::MatrixXd C_new;

    if (gps_observer_ptr_) {
        if (gps_data.time_ < 0.0)
            return;
        // 1. 计算ESKF需要的数据
        gps_observer_ptr_->ComputeHZR(gps_data, state_ptr, H, Z, R);
        // 2. 计算更新量
        ESKFUpdate(H, state_ptr->C_, R, Z, C_new, X);
        // 3. 更新
        state_ptr->Update(param_ptr_, X, C_new);
    }
    // todo add other sensor
}