#pragma once
#include "Filter.h"

void Filter::ESKFUpdate(
    const Eigen::MatrixXd & H, const Eigen::MatrixXd & C, const Eigen::MatrixXd & R,
    Eigen::MatrixXd & Z, Eigen::MatrixXd & C_new, Eigen::VectorXd & X)
{
    Eigen::MatrixXd K = C * H.transpose() * (H * C * H.transpose() + R).inverse();
    C_new = (Eigen::MatrixXd::Identity(param_ptr_->STATE_DIM, param_ptr_->STATE_DIM) - K * H) * C;
    X = K * Z;
}

void Filter::UpdateFromGPS(const std::shared_ptr<State> & state_ptr) {
    if (!gps_observer_ptr_)
        return;
    GPSData cur_gps_data;
    if (!data_manager_ptr_->GetLastGPSData(cur_gps_data, last_gps_data.time_) || std::abs(state_ptr->time_ - cur_gps_data.time_) > 0.01)
        return;

    Eigen::VectorXd X;
    Eigen::MatrixXd K;
    Eigen::MatrixXd H;
    Eigen::MatrixXd Z;
    Eigen::MatrixXd R;
    Eigen::MatrixXd C_new;
    // 1. 计算ESKF需要的数据
    gps_observer_ptr_->ComputeHZR(cur_gps_data, state_ptr, H, Z, R);
    // 2. 计算更新量
    ESKFUpdate(H, state_ptr->C_, R, Z, C_new, X);
    // 3. 更新
    state_ptr->Update(param_ptr_, X, C_new);
    last_gps_data.time_ = cur_gps_data.time_;
}

void Filter::UpdateFromWheel(const std::shared_ptr<State> & state_ptr) {
    if (!wheel_observer_ptr_)
        return;
    WheelData cur_wheel_data;
    // 默认不超过一个imu间隔
    if (!data_manager_ptr_->GetLastWheelData(cur_wheel_data, last_wheel_data.time_) || std::abs(state_ptr->time_ - cur_wheel_data.time_) > 0.01)
        return;

    Eigen::VectorXd X;
    Eigen::MatrixXd K;
    Eigen::MatrixXd H;
    Eigen::MatrixXd Z;
    Eigen::MatrixXd R;
    Eigen::MatrixXd C_new;
    // 1. 计算ESKF需要的数据
    wheel_observer_ptr_->ComputeHZR(cur_wheel_data, state_ptr, H, Z, R);
    // 2. 计算更新量
    ESKFUpdate(H, state_ptr->C_, R, Z, C_new, X);
    // 3. 更新
    state_ptr->Update(param_ptr_, X, C_new);
    last_wheel_data.time_ = cur_wheel_data.time_;
}

void Filter::UpdateFromGPSWheel(const std::shared_ptr<State> & state_ptr) {
    if (!gps_wheel_observer_ptr_)
        return;
    GPSData cur_gps_data;
    if (!data_manager_ptr_->GetLastGPSData(cur_gps_data, last_gps_data.time_) || std::abs(state_ptr->time_ - cur_gps_data.time_) > 0.01)
        return;
    WheelData cur_wheel_data;
    if (!data_manager_ptr_->GetLastWheelData(cur_wheel_data, last_wheel_data.time_) || std::abs(state_ptr->time_ - cur_wheel_data.time_) > 0.01)
        return;

    Eigen::VectorXd X;
    Eigen::MatrixXd K;
    Eigen::MatrixXd H;
    Eigen::MatrixXd Z;
    Eigen::MatrixXd R;
    Eigen::MatrixXd C_new;
    // 1. 计算ESKF需要的数据
    gps_wheel_observer_ptr_->ComputeHZR(cur_wheel_data, cur_gps_data, state_ptr, H, Z, R);
    // 2. 计算更新量
    ESKFUpdate(H, state_ptr->C_, R, Z, C_new, X);
    // 3. 更新
    state_ptr->Update(param_ptr_, X, C_new);
    last_gps_data.time_ = cur_gps_data.time_;
    last_wheel_data.time_ = cur_wheel_data.time_;
}

// todo add other sensor
void Filter::Run() {
    std::ofstream result_file;    //用ofstream类定义输入对象
	result_file.open("./result_file.txt");
    // 循环读数据
    while (1)
    {
        // todo 输入数据如果时间错乱需要进一步判断
        // 简易法
        std::shared_ptr<State> state_ptr;
        if(!state_manager_ptr_->GetNearestState(state_ptr)) {
            usleep(100);
            continue;
        }

        // 用这个或者下面的，同样参数下可以对比下区别
        // UpdateFromWheel(state_ptr);
        // UpdateFromGPS(state_ptr);

        UpdateFromGPSWheel(state_ptr);
        // result_file << state_ptr->twb_.x() << "," << state_ptr->twb_.y() << "," << state_ptr->twb_.z() << std::endl;
        usleep(100);
    }
}