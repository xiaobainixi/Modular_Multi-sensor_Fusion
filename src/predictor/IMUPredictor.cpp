#include "IMUPredictor.h"

// todo 数个一起来还是一个一个，数个一起来比较好
void IMUPredictor::Run() {
    while(1) {
        IMUData cur_data;
        if (!data_manager_ptr_->GetLastIMUData(cur_data, last_data_.time_)) {
            usleep(100);
            continue;
        }
        
        // 第一个数据
        if (last_data_.time_ <= 0.0) {
            std::shared_ptr<State> cur_state_ptr = std::make_shared<State>();
            cur_state_ptr->time_ = cur_data.time_;
            cur_state_ptr->C_ = Eigen::MatrixXd::Identity(param_ptr_->STATE_DIM, param_ptr_->STATE_DIM);
            cur_state_ptr->C_.block<3, 3>(param_ptr_->POSI_INDEX, param_ptr_->POSI_INDEX) = Eigen::Matrix3d::Identity() * 0.25;
            cur_state_ptr->C_.block<3, 3>(param_ptr_->VEL_INDEX_STATE_, param_ptr_->VEL_INDEX_STATE_) = Eigen::Matrix3d::Identity() * 0.25;
            cur_state_ptr->C_.block<3, 3>(param_ptr_->ORI_INDEX_STATE_, param_ptr_->ORI_INDEX_STATE_) = Eigen::Matrix3d::Identity() * 0.25;
            cur_state_ptr->C_.block<3, 3>(param_ptr_->GYRO_BIAS_INDEX_STATE_, param_ptr_->GYRO_BIAS_INDEX_STATE_) = Eigen::Matrix3d::Identity() * 0.01;
            cur_state_ptr->C_.block<3, 3>(param_ptr_->ACC_BIAS_INDEX_STATE_, param_ptr_->ACC_BIAS_INDEX_STATE_) = Eigen::Matrix3d::Identity() * 0.01;
            state_manager_ptr_->PushState(cur_state_ptr);
            last_data_ = cur_data;
            // std::cout << std::setprecision(9) << cur_data.time_ << std::endl;
            // std::cout << cur_data.a_.transpose() << std::endl;
            // std::cout << cur_data.w_.transpose() << std::endl;
            usleep(100);
            continue;
        }
        double delta_t = cur_data.time_ - last_data_.time_;
        if (delta_t <= 0.0) {
            usleep(100);
            continue;
        }

        // std::cout << cur_data.time_ << std::endl;
        // std::cout << cur_data.a_.transpose() << std::endl;
        // std::cout << cur_data.w_.transpose() << std::endl;
        std::shared_ptr<State> last_state_ptr;
        state_manager_ptr_->GetNearestState(last_state_ptr);
        std::shared_ptr<State> cur_state_ptr = std::make_shared<State>();
        cur_state_ptr->time_ = cur_data.time_;
        cur_state_ptr->ba_ = last_state_ptr->ba_;
        cur_state_ptr->bg_ = last_state_ptr->bg_;

        //-------------------------------------------------------------------------------------------------------------
        // 计算当前状态的角度
        Eigen::Vector3d cur_unbias_angular_vel = cur_data.w_ + last_state_ptr->bg_;
        Eigen::Vector3d last_unbias_angular_vel = last_data_.w_ + last_state_ptr->bg_;
        Eigen::Vector3d angular_delta = 0.5 * (cur_unbias_angular_vel + last_unbias_angular_vel) * delta_t;

        cur_state_ptr->Rwb_ =  last_state_ptr->Rwb_ * Eigen::AngleAxisd(angular_delta.norm(), angular_delta.normalized()).toRotationMatrix();

        //-------------------------------------------------------------------------------------------------------------
        // 计算当前状态的速度
        Eigen::Vector3d last_v, avg_a;
        Eigen::Vector3d cur_unbias_a = cur_state_ptr->Rwb_ * (cur_data.a_ + last_state_ptr->ba_) + gw_;
        Eigen::Vector3d last_unbias_a = last_state_ptr->Rwb_ * (last_data_.a_ + last_state_ptr->ba_) + gw_;

        last_v = last_state_ptr->Vw_;
        avg_a = 0.5 * (cur_unbias_a + last_unbias_a);
        cur_state_ptr->Vw_ = last_v + delta_t * avg_a;

        //---------------------------------------------------------------------------------------------------
        // 使用计算出的速度算位置变化
        cur_state_ptr->twb_ = last_state_ptr->twb_ + delta_t * last_v + 0.5 * avg_a * delta_t * delta_t;

        //---------------------------------------------------------------------------------------------------
        // 计算协方差矩阵
        // 定义： 理想数值（优质数值） = 估计数值 + 误差
        Eigen::MatrixXd F = Eigen::MatrixXd::Zero(param_ptr_->STATE_DIM, param_ptr_->STATE_DIM);
        F.block<3, 3>(param_ptr_->ORI_INDEX_STATE_, param_ptr_->GYRO_BIAS_INDEX_STATE_) = last_state_ptr->Rwb_;

        F.block<3, 3>(param_ptr_->VEL_INDEX_STATE_, param_ptr_->ORI_INDEX_STATE_) =
            -Converter::Skew(last_state_ptr->Rwb_ * (last_data_.a_ + last_state_ptr->ba_));
        F.block<3, 3>(param_ptr_->VEL_INDEX_STATE_, param_ptr_->ACC_BIAS_INDEX_STATE_) = last_state_ptr->Rwb_;

        F.block<3, 3>(param_ptr_->POSI_INDEX, param_ptr_->VEL_INDEX_STATE_) = Eigen::Matrix3d::Identity();
        F.block<3, 3>(param_ptr_->POSI_INDEX, param_ptr_->ACC_BIAS_INDEX_STATE_) = 0.5 * delta_t * last_state_ptr->Rwb_;
        F.block<3, 3>(param_ptr_->POSI_INDEX, param_ptr_->ORI_INDEX_STATE_) = -0.5 * delta_t * Converter::Skew(last_state_ptr->Rwb_ * (last_data_.a_ + last_state_ptr->ba_));

        Eigen::MatrixXd G = Eigen::MatrixXd::Zero(param_ptr_->STATE_DIM, 12);
        G.block<3, 3>(param_ptr_->ORI_INDEX_STATE_, 0) = last_state_ptr->Rwb_ * delta_t;
        G.block<3, 3>(param_ptr_->GYRO_BIAS_INDEX_STATE_, 3) = Eigen::Matrix3d::Identity() * delta_t;
        G.block<3, 3>(param_ptr_->VEL_INDEX_STATE_, 6) = last_state_ptr->Rwb_ * delta_t;
        G.block<3, 3>(param_ptr_->ACC_BIAS_INDEX_STATE_, 9) = Eigen::Matrix3d::Identity() * delta_t;

        Eigen::MatrixXd Phi = Eigen::MatrixXd::Identity(param_ptr_->STATE_DIM, param_ptr_->STATE_DIM) + F * delta_t;

        cur_state_ptr->C_ = Phi * last_state_ptr->C_ * Phi.transpose() + G * param_ptr_->imu_continuous_noise_cov_ * G.transpose();

        Eigen::MatrixXd state_cov_fixed = 
            (cur_state_ptr->C_ + cur_state_ptr->C_.transpose()) / 2.0;
        cur_state_ptr->C_ = state_cov_fixed;

        state_manager_ptr_->PushState(cur_state_ptr);
        last_data_ = cur_data;

        if (viewer_ptr_)
            viewer_ptr_->DrawWheelPose(cur_state_ptr->Rwb_, cur_state_ptr->twb_);
        // std::cout << std::setprecision(9) << cur_state_ptr->time_ << " " << cur_state_ptr->twb_.transpose() << " " << cur_state_ptr->ba_.transpose() << " " << cur_state_ptr->bg_.transpose() << " " << cur_state_ptr->Vw_.transpose() << std::endl;
        usleep(100);
    }
    
}