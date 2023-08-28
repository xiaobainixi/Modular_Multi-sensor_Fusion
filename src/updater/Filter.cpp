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
}

void Filter::Update(const std::shared_ptr<State> & state_ptr, const WheelData & wheel_data) {
    Eigen::VectorXd X;
    Eigen::MatrixXd K;
    Eigen::MatrixXd H;
    Eigen::MatrixXd Z;
    Eigen::MatrixXd R;
    Eigen::MatrixXd C_new;

    if (wheel_observer_ptr_) {
        if (wheel_data.time_ < 0.0)
            return;
        // 1. 计算ESKF需要的数据
        wheel_observer_ptr_->ComputeHZR(wheel_data, state_ptr, H, Z, R);
        // 2. 计算更新量
        ESKFUpdate(H, state_ptr->C_, R, Z, C_new, X);
        // 3. 更新
        state_ptr->Update(param_ptr_, X, C_new);
    }
}

// todo add other sensor
void Filter::Run() {
    GPSData last_gps_data;
    WheelData last_wheel_data;
    std::ofstream result_file;    //用ofstream类定义输入对象
	result_file.open("./result_file.txt");
    // 循环读数据
    while (1)
    {
        // todo 差值, 时间对齐。流程需要优化
        std::shared_ptr<State> state_ptr;
        if(!state_manager_ptr_->GetNearestState(state_ptr)) {
            usleep(100);
            continue;
        }

        WheelData cur_wheel_data;
        if (data_manager_ptr_->GetLastWheelData(cur_wheel_data, last_wheel_data.time_)) {
            Update(state_ptr, cur_wheel_data);
            std::cout << std::setprecision(9) << cur_wheel_data.time_ << std::endl;
            // result_file << state_ptr->twb_.x() << "," << state_ptr->twb_.y() << "," << state_ptr->twb_.z() << std::endl;
            last_wheel_data = cur_wheel_data;
        }

        // GPSData cur_gps_data;
        // if (data_manager_ptr_->GetLastGPSData(cur_gps_data, last_gps_data.time_)) {
        //     Update(state_ptr, cur_gps_data);
        //     result_file << state_ptr->twb_.x() << "," << state_ptr->twb_.y() << "," << state_ptr->twb_.z() << std::endl;
        //     // std::cout << std::setprecision(9) << cur_gps_data.time_ << " " << (state_ptr->Rbw_ * state_ptr->Vw_).transpose() << std::endl;
        //     // std::cout << state_ptr->time_ << " " << state_ptr->twb_.transpose() << " " << state_ptr->ba_.transpose() << " " << state_ptr->bg_.transpose() << " " << state_ptr->Vw_.transpose() << std::endl;
        //     std::cout << state_ptr->C_ << std::endl;
        //     last_gps_data = cur_gps_data;
        // }

        usleep(100);
    }
}