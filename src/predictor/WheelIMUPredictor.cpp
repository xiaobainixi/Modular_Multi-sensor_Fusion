#include "WheelIMUPredictor.h"


// todo 数个一起来还是一个一个，数个一起来比较好
void WheelIMUPredictor::Run() {
    while(1) {
        RunOnce();
        usleep(100);
    }
}

void WheelIMUPredictor::RunOnce() {
    WheelIMUData curr_data;
    if (!data_manager_ptr_->GetLastWheelIMUData(curr_data, last_data_.time_)) {
        usleep(100);
        return;
    }
    
    // 第一个数据
    if (last_data_.time_ <= 0.0) {
        if (state_manager_ptr_->Empty()) {
            std::shared_ptr<State> cur_state_ptr = std::make_shared<State>();
            cur_state_ptr->time_ = curr_data.time_;
            cur_state_ptr->C_ = Eigen::MatrixXd::Identity(param_ptr_->STATE_DIM, param_ptr_->STATE_DIM);
            cur_state_ptr->C_.block<3, 3>(param_ptr_->POSI_INDEX, param_ptr_->POSI_INDEX) = Eigen::Matrix3d::Identity() * 0.25;
            cur_state_ptr->C_.block<3, 3>(param_ptr_->ORI_INDEX_STATE_, param_ptr_->ORI_INDEX_STATE_) = Eigen::Matrix3d::Identity() * 0.25;
            cur_state_ptr->C_.block<3, 3>(param_ptr_->GYRO_BIAS_INDEX_STATE_, param_ptr_->GYRO_BIAS_INDEX_STATE_) = Eigen::Matrix3d::Identity() * 0.01;
            state_manager_ptr_->PushState(cur_state_ptr);
            last_data_ = curr_data;
            usleep(100);
            return;
        } else {
            std::shared_ptr<State> last_state_ptr;
            state_manager_ptr_->GetNearestState(last_state_ptr);
            last_data_ = curr_data;
            last_data_.time_ = last_state_ptr->time_;
        }
    }
    double delta_t = curr_data.time_ - last_data_.time_;
    if (delta_t <= 0.0) {
        usleep(100);
        return;
    }

    // 计算轮速的线速度和角速度
    double wheel_v = (curr_data.lv_ + curr_data.rv_) * 0.5;
    // double wheel_w = (curr_data.rv_ - curr_data.lv_) / param_ptr_->wheel_b_;
    // note: v是线速度，在车中，只观测x轴方向的速度
    // 将速度包装成向量的形式
    Eigen::Vector3d velo_vec(wheel_v, 0, 0);   // x, y, z 三个轴的线速度
    // Eigen::Vector3d wheel_w_vec(0, 0, wheel_w); // x, y, z 三个轴的角速度

    std::shared_ptr<State> last_state_ptr;
    state_manager_ptr_->GetNearestState(last_state_ptr);
    std::shared_ptr<State> cur_state_ptr = std::make_shared<State>();
    cur_state_ptr->time_ = curr_data.time_;
    cur_state_ptr->bg_ = last_state_ptr->bg_;

    //-------------------------------------------------------------------------------------------------------------
    // 计算当前状态的角度（使用IMU的角度）
    Eigen::Vector3d cur_unbias_angular_vel = curr_data.w_ + last_state_ptr->bg_;
    Eigen::Vector3d last_unbias_angular_vel = last_data_.w_ + last_state_ptr->bg_;
    Eigen::Vector3d angular_delta = 0.5 * (cur_unbias_angular_vel + last_unbias_angular_vel) * delta_t;

    cur_state_ptr->Rwb_ =  last_state_ptr->Rwb_ * Converter::so3ToQuat(angular_delta);

    //---------------------------------------------------------------------------------------------------
    // 使用计算出的速度算位置变化
    cur_state_ptr->twb_ = last_state_ptr->twb_ + last_state_ptr->Rwb_ * velo_vec * delta_t;

    //---------------------------------------------------------------------------------------------------
    // 计算协方差矩阵
    // 定义： 理想数值（优质数值） = 估计数值 + 误差
    Eigen::MatrixXd F = Eigen::MatrixXd::Zero(param_ptr_->STATE_DIM, param_ptr_->STATE_DIM);
    F.block<3, 3>(param_ptr_->ORI_INDEX_STATE_, param_ptr_->GYRO_BIAS_INDEX_STATE_) =
        cur_state_ptr->Rwb_ * Converter::RightJacobianSO3(angular_delta);
    F.block<3, 3>(param_ptr_->POSI_INDEX, param_ptr_->ORI_INDEX_STATE_) =
        -Converter::Skew(last_state_ptr->Rwb_ * velo_vec);

    Eigen::MatrixXd G = Eigen::MatrixXd::Zero(param_ptr_->STATE_DIM, 9);
    G.block<3, 3>(param_ptr_->ORI_INDEX_STATE_, 0) = cur_state_ptr->Rwb_ * Converter::RightJacobianSO3(angular_delta) * delta_t;
    G.block<3, 3>(param_ptr_->GYRO_BIAS_INDEX_STATE_, 3) = Eigen::Matrix3d::Identity();
    G.block<3, 3>(param_ptr_->POSI_INDEX, 6) = last_state_ptr->Rwb_.toRotationMatrix() * delta_t;

    Eigen::MatrixXd Phi = Eigen::MatrixXd::Identity(param_ptr_->STATE_DIM, param_ptr_->STATE_DIM) + F * delta_t;

    cur_state_ptr->C_ = last_state_ptr->C_;
    cur_state_ptr->C_.block(0, 0, param_ptr_->STATE_DIM, param_ptr_->STATE_DIM) =
        Phi * last_state_ptr->C_.block(0, 0, param_ptr_->STATE_DIM, param_ptr_->STATE_DIM) * Phi.transpose() +
        G * param_ptr_->predict_dispersed_noise_cov_ * G.transpose();

    Eigen::MatrixXd state_cov_fixed = 
        (cur_state_ptr->C_ + cur_state_ptr->C_.transpose()) / 2.0;
    cur_state_ptr->C_ = state_cov_fixed;

    state_manager_ptr_->PushState(cur_state_ptr);
    last_data_ = curr_data;

    if (viewer_ptr_)
        viewer_ptr_->DrawWheelPose(cur_state_ptr->Rwb_.toRotationMatrix(), cur_state_ptr->twb_);
}
