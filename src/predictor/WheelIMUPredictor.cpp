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
        std::shared_ptr<State> cur_state_ptr = std::make_shared<State>();
        cur_state_ptr->time_ = curr_data.time_;
        cur_state_ptr->C_ = Eigen::MatrixXd::Identity(param_ptr_->STATE_DIM, param_ptr_->STATE_DIM);
        cur_state_ptr->C_.block<3, 3>(param_ptr_->POSI_INDEX, param_ptr_->POSI_INDEX) = Eigen::Matrix3d::Identity() * 0.25;
        cur_state_ptr->C_.block<3, 3>(param_ptr_->VEL_INDEX_STATE_, param_ptr_->VEL_INDEX_STATE_) = Eigen::Matrix3d::Identity() * 0.25;
        cur_state_ptr->C_.block<3, 3>(param_ptr_->ORI_INDEX_STATE_, param_ptr_->ORI_INDEX_STATE_) = Eigen::Matrix3d::Identity() * 0.25;
        cur_state_ptr->C_.block<3, 3>(param_ptr_->GYRO_BIAS_INDEX_STATE_, param_ptr_->GYRO_BIAS_INDEX_STATE_) = Eigen::Matrix3d::Identity() * 0.01;
        cur_state_ptr->C_.block<3, 3>(param_ptr_->ACC_BIAS_INDEX_STATE_, param_ptr_->ACC_BIAS_INDEX_STATE_) = Eigen::Matrix3d::Identity() * 0.01;
        state_manager_ptr_->PushState(cur_state_ptr);
        last_data_ = curr_data;
        usleep(100);
        return;
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
    Eigen::Vector3d wheel_v_vec(wheel_v, 0, 0);   // x, y, z 三个轴的线速度
    // Eigen::Vector3d wheel_w_vec(0, 0, wheel_w); // x, y, z 三个轴的角速度

    std::shared_ptr<State> last_state_ptr;
    state_manager_ptr_->GetNearestState(last_state_ptr);
    std::shared_ptr<State> cur_state_ptr = std::make_shared<State>();
    cur_state_ptr->time_ = curr_data.time_;
    cur_state_ptr->ba_ = last_state_ptr->ba_;
    cur_state_ptr->bg_ = last_state_ptr->bg_;

    //-------------------------------------------------------------------------------------------------------------
    // 计算当前状态的角度（使用IMU的角度）
    Eigen::Vector3d cur_unbias_angular_vel = curr_data.w_ + last_state_ptr->bg_;
    Eigen::Vector3d last_unbias_angular_vel = last_data_.w_ + last_state_ptr->bg_;
    Eigen::Vector3d angular_delta = 0.5 * (cur_unbias_angular_vel + last_unbias_angular_vel) * delta_t;

    cur_state_ptr->Rwb_ =  last_state_ptr->Rwb_ * Eigen::AngleAxisd(angular_delta.norm(), angular_delta.normalized()).toRotationMatrix();

    //-------------------------------------------------------------------------------------------------------------
    // 计算当前状态的速度（x轴使用轮速的速度，y,z轴使用IMU的速度）
    Eigen::Vector3d last_v, avg_a;
    Eigen::Vector3d cur_unbias_a = cur_state_ptr->Rwb_ * (curr_data.a_ + last_state_ptr->ba_) + gw_;
    Eigen::Vector3d last_unbias_a = last_state_ptr->Rwb_ * (last_data_.a_ + last_state_ptr->ba_) + gw_;

    last_v = last_state_ptr->Vw_;
    avg_a = 0.5 * (cur_unbias_a + last_unbias_a);
    // cur_state_ptr->Vw_ = last_v + delta_t * avg_a;
    // 这里速度使用轮速的速度
    cur_state_ptr->Vw_ = last_state_ptr->Rwb_ * wheel_v_vec;

    //---------------------------------------------------------------------------------------------------
    // 使用计算出的速度算位置变化
    cur_state_ptr->twb_ = last_state_ptr->twb_ + delta_t * last_v + 0.5 * avg_a * delta_t * delta_t;

    //---------------------------------------------------------------------------------------------------
    // 计算协方差矩阵
    // 定义： 理想数值（优质数值） = 估计数值 + 误差

    // todo: 这里有问题
    Eigen::MatrixXd F = Eigen::MatrixXd::Zero(param_ptr_->STATE_DIM, param_ptr_->STATE_DIM);
    F.block<3, 3>(param_ptr_->ORI_INDEX_STATE_, param_ptr_->GYRO_BIAS_INDEX_STATE_) = last_state_ptr->Rwb_;

    // 现在V直接观测，这些都不需要
    // F.block<3, 3>(param_ptr_->VEL_INDEX_STATE_, param_ptr_->ORI_INDEX_STATE_) =
    //     -Converter::Skew(last_state_ptr->Rwb_ * (last_data_.a_ + last_state_ptr->ba_));
    // F.block<3, 3>(param_ptr_->VEL_INDEX_STATE_, param_ptr_->ACC_BIAS_INDEX_STATE_) = last_state_ptr->Rwb_;
    F.block<3, 3>(param_ptr_->VEL_INDEX_STATE_, param_ptr_->ORI_INDEX_STATE_) =
           -Converter::Skew(last_state_ptr->Rwb_ * wheel_v_vec);

    F.block<3, 3>(param_ptr_->POSI_INDEX, param_ptr_->VEL_INDEX_STATE_) = Eigen::Matrix3d::Identity();
    F.block<3, 3>(param_ptr_->POSI_INDEX, param_ptr_->ACC_BIAS_INDEX_STATE_) = 0.5 * delta_t * last_state_ptr->Rwb_;
    F.block<3, 3>(param_ptr_->POSI_INDEX, param_ptr_->ORI_INDEX_STATE_) = -0.5 * delta_t * Converter::Skew(last_state_ptr->Rwb_ * (last_data_.a_ + last_state_ptr->ba_));

    Eigen::Matrix<double, 3, 3> odom_dispersed_noise_cov = Eigen::Matrix<double, 3, 3>::Zero();
    odom_dispersed_noise_cov(0, 0)= param_ptr_->wheel_vel_noise_ * param_ptr_->wheel_vel_noise_;  // 轮速计只观测X轴的速度

    Eigen::Matrix<double, 15, 15> imu_odom_dispersed_noise_cov = Eigen::Matrix<double, 15, 15>::Zero();
    imu_odom_dispersed_noise_cov.block<12, 12>(0, 0) = param_ptr_->imu_dispersed_noise_cov_;
    imu_odom_dispersed_noise_cov.block<3, 3>(12, 12) = odom_dispersed_noise_cov;

    Eigen::MatrixXd G = Eigen::MatrixXd::Zero(param_ptr_->STATE_DIM, 15);
    G.block<3, 3>(param_ptr_->ORI_INDEX_STATE_, 0) = last_state_ptr->Rwb_ * delta_t;
    G.block<3, 3>(param_ptr_->GYRO_BIAS_INDEX_STATE_, 3) = Eigen::Matrix3d::Identity() * delta_t;
    // G.block<3, 3>(param_ptr_->VEL_INDEX_STATE_, 6) = last_state_ptr->Rwb_ * delta_t;  // 速度的观测不再从IMU的加速得来
    G.block<3, 3>(param_ptr_->ACC_BIAS_INDEX_STATE_, 9) = Eigen::Matrix3d::Identity() * delta_t;
    G.block<3, 3>(param_ptr_->POSI_INDEX, 12) = last_state_ptr->Rwb_ * delta_t;

    Eigen::MatrixXd Phi = Eigen::MatrixXd::Identity(param_ptr_->STATE_DIM, param_ptr_->STATE_DIM) + F * delta_t;

    cur_state_ptr->C_ = last_state_ptr->C_;
    cur_state_ptr->C_.block(0, 0, param_ptr_->STATE_DIM, param_ptr_->STATE_DIM) =
        Phi * last_state_ptr->C_.block(0, 0, param_ptr_->STATE_DIM, param_ptr_->STATE_DIM) * Phi.transpose() +
        G * imu_odom_dispersed_noise_cov * G.transpose();

    Eigen::MatrixXd state_cov_fixed = 
        (cur_state_ptr->C_ + cur_state_ptr->C_.transpose()) / 2.0;
    cur_state_ptr->C_ = state_cov_fixed;

    state_manager_ptr_->PushState(cur_state_ptr);
    last_data_ = curr_data;

    if (viewer_ptr_)
        viewer_ptr_->DrawWheelPose(cur_state_ptr->Rwb_, cur_state_ptr->twb_);
}
