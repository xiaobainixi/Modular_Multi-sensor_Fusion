#include "CeresPredictor.h"

// todo 数个一起来还是一个一个，数个一起来比较好
void CeresPredictor::Run() {
    while(1) {
        RunOnce();
        usleep(100);
    }
}

void CeresPredictor::RunOnce() {
        IMUData cur_data;
        if (!data_manager_ptr_->GetLastIMUData(cur_data, last_data_.time_)) {
            return;
        }

        Eigen::Matrix<double, 6, 1> imu_data;
        imu_data.head(3) = cur_data.a_;
        imu_data.tail(3) = cur_data.w_;
        ceres_fusion_ptr_->AddImuData(cur_data.time_, imu_data);

        // 等待初始化成功
        if (!ceres_fusion_ptr_->IsInit()) {
            last_data_ = cur_data;
            return;
        }

        // TODO : 利用imu数据来进行预测。获取最新的imu预测的状态。
        std::shared_ptr<State> cur_state_ptr = std::make_shared<State>();
        {
          double predict_time;
          Eigen::Vector3d p;
          Eigen::Vector3d v;
          Eigen::Quaterniond q;
          Eigen::Vector3d ba;
          Eigen::Vector3d bg;
          ceres_fusion_ptr_->Predict(predict_time, p, v, q, ba, bg);
          cur_state_ptr->time_ = predict_time;
          cur_state_ptr->Vw_ = v;
          cur_state_ptr->twb_ = p;
          cur_state_ptr->Rwb_ = Eigen::Matrix3d(q);
          cur_state_ptr->ba_ = ba;
          cur_state_ptr->bg_ = bg;
        }

#if 0
        // TODO : 这部分代码改为用imu的预积分 imu_preint_ptr_ 来进行管理
        
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
            return;
        }
        double delta_t = cur_data.time_ - last_data_.time_;
        if (delta_t <= 0.0) {
            return;
        }

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
        G.block<3, 3>(param_ptr_->POSI_INDEX, 6) = 0.5 * last_state_ptr->Rwb_ * delta_t * delta_t;

        Eigen::MatrixXd Phi = Eigen::MatrixXd::Identity(param_ptr_->STATE_DIM, param_ptr_->STATE_DIM) + F * delta_t;

        cur_state_ptr->C_ = last_state_ptr->C_;
        cur_state_ptr->C_.block(0, 0, param_ptr_->STATE_DIM, param_ptr_->STATE_DIM) =
            Phi * last_state_ptr->C_.block(0, 0, param_ptr_->STATE_DIM, param_ptr_->STATE_DIM) * Phi.transpose() +
            G * param_ptr_->predict_dispersed_noise_cov_ * G.transpose();

        if (state_manager_ptr_->cam_states_.size() > 0)
        {
            // 起点是0 param_ptr_->STATE_DIM  然后是21行 cur_state_ptr->C_.cols() - param_ptr_->STATE_DIM 列的矩阵
            // 也就是整个协方差矩阵的右上角，这部分说白了就是imu状态量与相机状态量的协方差，imu更新了，这部分也需要更新
            cur_state_ptr->C_.block(0, param_ptr_->STATE_DIM, param_ptr_->STATE_DIM, cur_state_ptr->C_.cols() - param_ptr_->STATE_DIM) =
                Phi * cur_state_ptr->C_.block(0, param_ptr_->STATE_DIM, param_ptr_->STATE_DIM, cur_state_ptr->C_.cols() - param_ptr_->STATE_DIM);

            // 同理，这个是左下角
            cur_state_ptr->C_.block(param_ptr_->STATE_DIM, 0, cur_state_ptr->C_.rows() - param_ptr_->STATE_DIM, param_ptr_->STATE_DIM) =
                cur_state_ptr->C_.block(param_ptr_->STATE_DIM, 0, cur_state_ptr->C_.rows() - param_ptr_->STATE_DIM, param_ptr_->STATE_DIM) *
                Phi.transpose();
        }

        Eigen::MatrixXd state_cov_fixed = 
            (cur_state_ptr->C_ + cur_state_ptr->C_.transpose()) / 2.0;
        cur_state_ptr->C_ = state_cov_fixed;
#endif

        state_manager_ptr_->PushState(cur_state_ptr);
        last_data_ = cur_data;

        if (viewer_ptr_){
            viewer_ptr_->DrawWheelPose(cur_state_ptr->Rwb_, cur_state_ptr->twb_);
        }
            
        // LOG(INFO) << "predict";
}