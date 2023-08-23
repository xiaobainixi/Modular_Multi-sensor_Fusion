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

void Filter::Run() {
    GPSData last_gps_data;
    std::ofstream result_file;    //用ofstream类定义输入对象
	result_file.open("./result_file.txt");
    // 循环读数据
    while (1)
    {
        // todo 差值
        std::shared_ptr<State> state_ptr;
        if(!state_manager_ptr_->GetNearestState(state_ptr)) {
            usleep(100);
            continue;
        }

        GPSData cur_gps_data = data_manager_ptr_->GetLastGPSData();
        if (cur_gps_data.time_ < 0.0 || cur_gps_data.time_ <= last_gps_data.time_) {
            usleep(100);
            continue;
        }
        Update(state_ptr, cur_gps_data);
        result_file << state_ptr->twb_.x() << "," << state_ptr->twb_.y() << "," << state_ptr->twb_.z() << std::endl;
        // std::cout << std::setprecision(9) << cur_gps_data.time_ << std::endl;
        std::cout << state_ptr->twb_.x() << "," << state_ptr->twb_.y() << "," << state_ptr->twb_.z() << std::endl;
        last_gps_data = cur_gps_data;
        usleep(100);
    }
}