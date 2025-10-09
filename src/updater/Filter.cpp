#pragma once
#include "Filter.h"

FeatureIDType Feature::next_id = 0;
void Filter::ESKFUpdate(
    const Eigen::MatrixXd & H, const Eigen::MatrixXd & C, const Eigen::MatrixXd & R,
    Eigen::MatrixXd & Z, Eigen::MatrixXd & C_new, Eigen::VectorXd & X)
{
    int state_dim = param_ptr_->STATE_DIM + state_manager_ptr_->cam_states_.size() * 6;
    Eigen::MatrixXd K = C * H.transpose() * (H * C * H.transpose() + R).inverse();
    C_new = (Eigen::MatrixXd::Identity(state_dim, state_dim) - K * H) * C;
    X = K * Z;
}

void Filter::UpdateFromGNSS(const std::shared_ptr<State> & state_ptr) {
    if (!gnss_observer_ptr_)
        return;
    GNSSData cur_gnss_data;
    if (!data_manager_ptr_->GetLastGNSSData(cur_gnss_data, last_gnss_data_.time_) || std::abs(state_ptr->time_ - cur_gnss_data.time_) > 0.01)
        return;

    Eigen::VectorXd X;
    Eigen::MatrixXd K;
    Eigen::MatrixXd H;
    Eigen::MatrixXd Z;
    Eigen::MatrixXd R;
    Eigen::MatrixXd C_new;
    // 1. 计算ESKF需要的数据
    gnss_observer_ptr_->ComputeHZR(cur_gnss_data, state_ptr, H, Z, R);
    // 2. 计算更新量
    ESKFUpdate(H, state_ptr->C_, R, Z, C_new, X);
    // 3. 更新
    state_ptr->Update(param_ptr_, X, C_new, state_manager_ptr_->cam_states_);
    last_gnss_data_.time_ = cur_gnss_data.time_;
}

void Filter::UpdateFromWheel(const std::shared_ptr<State> & state_ptr) {
    if (!wheel_observer_ptr_)
        return;
    WheelData cur_wheel_data;
    // 默认不超过一个imu间隔
    if (!data_manager_ptr_->GetLastWheelData(cur_wheel_data, last_wheel_data_.time_) || std::abs(state_ptr->time_ - cur_wheel_data.time_) > 0.01)
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
    state_ptr->Update(param_ptr_, X, C_new, state_manager_ptr_->cam_states_);
    last_wheel_data_.time_ = cur_wheel_data.time_;
}

void Filter::UpdateFromGNSSWheel(const std::shared_ptr<State> & state_ptr) {
    if (!gnss_wheel_observer_ptr_)
        return;
    GNSSData cur_gnss_data;
    if (!data_manager_ptr_->GetLastGNSSData(cur_gnss_data, last_gnss_data_.time_) || std::abs(state_ptr->time_ - cur_gnss_data.time_) > 0.01)
        return;
    WheelData cur_wheel_data;
    if (!data_manager_ptr_->GetLastWheelData(cur_wheel_data, last_wheel_data_.time_) || std::abs(state_ptr->time_ - cur_wheel_data.time_) > 0.01)
        return;

    // LOG(INFO) << "UpdateFromGNSSWheel";
    Eigen::VectorXd X;
    Eigen::MatrixXd K;
    Eigen::MatrixXd H;
    Eigen::MatrixXd Z;
    Eigen::MatrixXd R;
    Eigen::MatrixXd C_new;
    // 1. 计算ESKF需要的数据
    gnss_wheel_observer_ptr_->ComputeHZR(cur_wheel_data, cur_gnss_data, state_ptr, H, Z, R);
    // 2. 计算更新量
    ESKFUpdate(H, state_ptr->C_, R, Z, C_new, X);
    // 3. 更新
    state_ptr->Update(param_ptr_, X, C_new, state_manager_ptr_->cam_states_);
    last_gnss_data_.time_ = cur_gnss_data.time_;
    last_wheel_data_.time_ = cur_wheel_data.time_;
}

void Filter::UpdateFromCamera(const std::shared_ptr<State> & state_ptr) {
    if (!camera_observer_ptr_)
        return;
    FeatureData feature_data;
    std::shared_ptr<State> feature_data_state;
    if (!data_manager_ptr_->GetNewFeatureData(feature_data, last_feature_data_.time_) ||
        !state_manager_ptr_->GetNearestState(feature_data_state, feature_data.time_))
        return;
    // LOG(INFO) << "UpdateFromCamera";
    feature_data.Rwc_ = feature_data_state->Rwb_ * param_ptr_->Rbc_;
    feature_data.twc_ = feature_data_state->Rwb_ * param_ptr_->tbc_ + feature_data_state->twb_;
    feature_data.Rwb_ = feature_data_state->Rwb_;
    feature_data.twb_ = feature_data_state->twb_;

    Eigen::VectorXd X;
    Eigen::MatrixXd K;
    Eigen::MatrixXd H;
    Eigen::MatrixXd Z;
    Eigen::MatrixXd R;
    Eigen::MatrixXd C_new;
    // 1. 计算ESKF需要的数据
    camera_observer_ptr_->ComputeHZR(feature_data, state_ptr, H, Z, R);
    // 2. 计算更新量
    // ESKFUpdate(H, state_ptr->C_, R, Z, C_new, X);
    // // 3. 更新
    // state_ptr->Update(param_ptr_, X, C_new);
    last_feature_data_.time_ = feature_data.time_;
}

// todo add other sensor
void Filter::Run() {
    std::ofstream result_file;
	result_file.open("./result_file.txt");
    // 循环读数据
    while (1)
    {
        if (!initialized_) {
            if (initializers_ptr_->Initialization())
                initialized_ = true;
            else
                continue;
        }
        if (predictor_ptr_)
            predictor_ptr_->RunOnce();
        // 简易法
        std::shared_ptr<State> state_ptr;
        if(!state_manager_ptr_->GetNearestState(state_ptr)) {
            usleep(100);
            continue;
        }

        if (param_ptr_->gnss_wheel_align_)
            UpdateFromGNSSWheel(state_ptr);
        else {
            UpdateFromWheel(state_ptr);
            UpdateFromGNSS(state_ptr);
        }
        if (param_ptr_->use_camera_)
            UpdateFromCamera(state_ptr);

        // result_file << state_ptr->twb_.x() << "," << state_ptr->twb_.y() << "," << state_ptr->twb_.z() << std::endl;
        usleep(100);
    }
}