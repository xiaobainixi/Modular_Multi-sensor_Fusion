#include "IMUPredictor.h"

// todo 数个一起来还是一个一个，数个一起来比较好
void IMUPredictor::Run() {
    while(1) {
        RunOnce();
        usleep(100);
    }
}

void IMUPredictor::RunOnce() {
        IMUData cur_data;
        if (!data_manager_ptr_->GetLastIMUData(cur_data, last_data_.time_)) {
            return;
        }

        // 第一个数据
        if (last_data_.time_ <= 0.0) {
            if (state_manager_ptr_->Empty()) {
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
            } else {
                std::shared_ptr<State> last_state_ptr;
                state_manager_ptr_->GetNearestState(last_state_ptr);
                last_data_ = cur_data;
                last_data_.time_ = last_state_ptr->time_;
            }
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

        //--------------------------------------------------------------------------------
        // 1. 计算当前时刻状态的旋转
        // cur_state_ptr是当前时刻状态的智能指针，该状态包含位姿(四元数)、速度、偏置、协方差等信息
        // last_state_ptr是上一时刻状态的智能指针
        // 当前时刻数据经过偏置补偿后的角速度
        Eigen::Vector3d cur_unbias_angular_vel = cur_data.w_ + last_state_ptr->bg_;
        // 上一时刻数据经过偏置补偿后的角速度
        Eigen::Vector3d last_unbias_angular_vel = last_data_.w_ + last_state_ptr->bg_;
        // 角速度的平均值乘以时间间隔，得到旋转向量
        // 这部分虽然与书中推导的式子不同，但数学上是等价的，且更加准确
        Eigen::Vector3d angular_delta =
            0.5 * (cur_unbias_angular_vel + last_unbias_angular_vel) * delta_t;

        // 当前状态的旋转 = 上一时刻状态的旋转 * (角速度*时间间隔)
        // Converter::so3ToQuat表示将旋转向量转成四元数
        // 式4.3a
        cur_state_ptr->Rwb_ =
            last_state_ptr->Rwb_ * Converter::so3ToQuat(angular_delta);

        //--------------------------------------------------------------------------------
        // 2. 计算当前状态的速度
        // 当前时刻数据在世界坐标系下经过偏置+重力补偿后的加速度
        Eigen::Vector3d cur_unbias_a =
            cur_state_ptr->Rwb_ * (cur_data.a_ + last_state_ptr->ba_) + gw_;
        // 上一时刻数据在世界坐标系下经过偏置+重力补偿后的加速度
        Eigen::Vector3d last_unbias_a =
            last_state_ptr->Rwb_ * (last_data_.a_ + last_state_ptr->ba_) + gw_;
        // 取平均值，与角速度同理
        Eigen::Vector3d avg_a = 0.5 * (cur_unbias_a + last_unbias_a);

        // 计算当前时刻世界坐标系下的速度
        cur_state_ptr->Vw_ = last_state_ptr->Vw_ + delta_t * avg_a;  // 式4.3b

        //--------------------------------------------------------------------------------
        // 3. 使用当前时刻状态的位移，世界坐标系下
        // 式4.3c
        cur_state_ptr->twb_ =
            last_state_ptr->twb_ + delta_t * last_state_ptr->Vw_ +
            0.5 * avg_a * delta_t * delta_t;

        //--------------------------------------------------------------------------------
        // 计算协方差矩阵
        // 定义： 理想数值（优质数值） = 估计数值 + 误差
        // param_ptr_->STATE_DIM 表示状态维度，当前为15

        // 1. 计算F矩阵
        Eigen::MatrixXd F =
            Eigen::MatrixXd::Identity(param_ptr_->STATE_DIM, param_ptr_->STATE_DIM);

        // 式4.17 位移误差与速度误差的关系
        F.block<3, 3>(param_ptr_->POSI_INDEX, param_ptr_->VEL_INDEX_STATE_) =
            Eigen::Matrix3d::Identity() * delta_t;
        // 式4.18 位移误差与旋转误差的关系
        // Converter::Skew表示反对称矩阵
        F.block<3, 3>(param_ptr_->POSI_INDEX, param_ptr_->ORI_INDEX_STATE_) =
            -0.5 * delta_t * delta_t * Converter::Skew(
                last_state_ptr->Rwb_ * (last_data_.a_ + last_state_ptr->ba_));
        // 式4.19 位移误差与加速度计误差的关系
        F.block<3, 3>(param_ptr_->POSI_INDEX, param_ptr_->ACC_BIAS_INDEX_STATE_) =
            0.5 * delta_t * delta_t * last_state_ptr->Rwb_.toRotationMatrix();

        // 式4.11 速度误差与旋转误差的关系
        F.block<3, 3>(param_ptr_->VEL_INDEX_STATE_, param_ptr_->ORI_INDEX_STATE_) =
            -Converter::Skew(
                last_state_ptr->Rwb_ * (last_data_.a_ + last_state_ptr->ba_) * delta_t);
        // 式4.15b 速度误差与加速度计偏置误差的关系
        F.block<3, 3>(param_ptr_->VEL_INDEX_STATE_, param_ptr_->ACC_BIAS_INDEX_STATE_) =
            last_state_ptr->Rwb_.toRotationMatrix() * delta_t;

        // 式4.9 旋转误差与陀螺仪偏置误差的关系
        // Converter::RightJacobianSO3表示取Jr
        // angular_delta =
        //   0.5 * (cur_unbias_angular_vel + last_unbias_angular_vel) * delta_t;
        F.block<3, 3>(param_ptr_->ORI_INDEX_STATE_, param_ptr_->GYRO_BIAS_INDEX_STATE_) =
            cur_state_ptr->Rwb_ * Converter::RightJacobianSO3(angular_delta) * delta_t;

        // 1. 计算G矩阵(书中的B矩阵)
        Eigen::MatrixXd G = Eigen::MatrixXd::Zero(param_ptr_->STATE_DIM, 12);
        // 式4.21 旋转误差与陀螺仪噪声的关系
        G.block<3, 3>(param_ptr_->ORI_INDEX_STATE_, 0) =
            cur_state_ptr->Rwb_ * Converter::RightJacobianSO3(angular_delta) * delta_t;
        // 式4.22b 陀螺仪偏置的误差状态与随机游走的关系
        G.block<3, 3>(param_ptr_->GYRO_BIAS_INDEX_STATE_, 3) = Eigen::Matrix3d::Identity();
        // 式4.24 位置误差状态与加速度计的噪声关系
        G.block<3, 3>(param_ptr_->POSI_INDEX, 6) =
            0.5 * last_state_ptr->Rwb_.toRotationMatrix() * delta_t * delta_t;
        // 式4.23 速度误差状态与加速度计的噪声关系
        G.block<3, 3>(param_ptr_->VEL_INDEX_STATE_, 6) =
            last_state_ptr->Rwb_.toRotationMatrix() * delta_t;
        // 加速度计偏置的误差状态与随机游走的关系，与陀螺仪同理
        G.block<3, 3>(param_ptr_->ACC_BIAS_INDEX_STATE_, 9) = Eigen::Matrix3d::Identity();
        

        cur_state_ptr->C_ = last_state_ptr->C_;
        // 式4.28 误差状态协方差的传播
        // C_表示协方差矩阵
        // param_ptr_->predict_dispersed_noise_cov_ 表示IMU的噪声协方差矩阵
        auto last_1515_C = last_state_ptr->C_.block(
            0, 0, param_ptr_->STATE_DIM, param_ptr_->STATE_DIM);
        cur_state_ptr->C_.block(0, 0, param_ptr_->STATE_DIM, param_ptr_->STATE_DIM) =
            F * last_1515_C * F.transpose() +
            G * param_ptr_->predict_dispersed_noise_cov_ * G.transpose();

        if (state_manager_ptr_->cam_states_.size() > 0)
        {
            // 起点是0 param_ptr_->STATE_DIM  然后是21行 cur_state_ptr->C_.cols() - param_ptr_->STATE_DIM 列的矩阵
            // 也就是整个协方差矩阵的右上角，这部分说白了就是imu状态量与相机状态量的协方差，imu更新了，这部分也需要更新
            cur_state_ptr->C_.block(0, param_ptr_->STATE_DIM, param_ptr_->STATE_DIM, cur_state_ptr->C_.cols() - param_ptr_->STATE_DIM) =
                F * cur_state_ptr->C_.block(0, param_ptr_->STATE_DIM, param_ptr_->STATE_DIM, cur_state_ptr->C_.cols() - param_ptr_->STATE_DIM);

            // 同理，这个是左下角
            cur_state_ptr->C_.block(param_ptr_->STATE_DIM, 0, cur_state_ptr->C_.rows() - param_ptr_->STATE_DIM, param_ptr_->STATE_DIM) =
                cur_state_ptr->C_.block(param_ptr_->STATE_DIM, 0, cur_state_ptr->C_.rows() - param_ptr_->STATE_DIM, param_ptr_->STATE_DIM) *
                F.transpose();
        }

        Eigen::MatrixXd state_cov_fixed = 
            (cur_state_ptr->C_ + cur_state_ptr->C_.transpose()) / 2.0;
        cur_state_ptr->C_ = state_cov_fixed;

        state_manager_ptr_->PushState(cur_state_ptr);
        last_data_ = cur_data;

        if (viewer_ptr_)
            viewer_ptr_->DrawWheelPose(cur_state_ptr->Rwb_.toRotationMatrix(), cur_state_ptr->twb_);
        // LOG(INFO) << "predict";
}