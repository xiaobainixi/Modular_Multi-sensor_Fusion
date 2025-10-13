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

    //--------------------------------------------------------------------------------
    // 根据轮速数据计算线速度
    // cur_data.lv_是左轮速度，cur_data.rv_是右轮速度，单位m/s
    // 式5.9a 线速度v = (lv + rv) * 0.5
    double v = (curr_data.lv_ + curr_data.rv_) * 0.5;
    // 式5.9b 只能给出x轴方向的速度
    Eigen::Vector3d velo_vec(v, 0, 0);   // x, y, z 三个轴的线速度

    //--------------------------------------------------------------------------------
    // cur_state_ptr是当前时刻状态的智能指针，该状态包含位姿(四元数)、协方差等信息
    // last_state_ptr是上一时刻状态的智能指针
    std::shared_ptr<State> last_state_ptr;
    state_manager_ptr_->GetNearestState(last_state_ptr);
    std::shared_ptr<State> cur_state_ptr = std::make_shared<State>();
    cur_state_ptr->time_ = curr_data.time_;
    // 上一时刻陀螺仪偏置作为当前时刻陀螺仪偏置
    cur_state_ptr->bg_ = last_state_ptr->bg_;

    //--------------------------------------------------------------------------------
    // 该模式下不存在加速度计，因此不考虑速度和加速度计偏置的更新，比较简单
    // 计算当前时刻状态的旋转（使用IMU的角速度）
    // 当前时刻数据经过偏置补偿后的角速度
    Eigen::Vector3d cur_unbias_angular_vel = curr_data.w_ + last_state_ptr->bg_;
    // 上一时刻数据经过偏置补偿后的角速度
    Eigen::Vector3d last_unbias_angular_vel = last_data_.w_ + last_state_ptr->bg_;
    // 角速度的平均值乘以时间间隔，得到旋转向量
    // 这部分虽然与书中推导的式子不同，但数学上是等价的，且更加准确
    Eigen::Vector3d angular_delta =
        0.5 * (cur_unbias_angular_vel + last_unbias_angular_vel) * delta_t;

    // 当前状态的旋转 = 上一时刻状态的旋转 * (角速度*时间间隔)
    // Converter::so3ToQuat表示将旋转向量转成四元数
    // 式5.26a 计算当前时刻状态的旋转，世界坐标系下
    cur_state_ptr->Rwb_ =
        last_state_ptr->Rwb_ * Converter::so3ToQuat(angular_delta);

    // 式5.26b 计算当前时刻状态的位移，世界坐标系下
    cur_state_ptr->twb_ =
        last_state_ptr->twb_ + last_state_ptr->Rwb_ * velo_vec * delta_t;

    //--------------------------------------------------------------------------------
    // 计算协方差矩阵
    // 定义： 理想数值（优质数值） = 估计数值 + 误差
    // param_ptr_->STATE_DIM 表示状态维度，当前为9

    // 计算F矩阵
    Eigen::MatrixXd F =
        Eigen::MatrixXd::Identity(param_ptr_->STATE_DIM, param_ptr_->STATE_DIM);
    // 式4.9 旋转误差与陀螺仪偏置误差的关系
    // Converter::RightJacobianSO3表示取Jr
    F.block<3, 3>(param_ptr_->ORI_INDEX_STATE_, param_ptr_->GYRO_BIAS_INDEX_STATE_) =
        cur_state_ptr->Rwb_ * Converter::RightJacobianSO3(angular_delta) * delta_t;
    // 式5.14 求当前时刻位移误差与上时刻旋转误差的关系
    F.block<3, 3>(param_ptr_->POSI_INDEX, param_ptr_->ORI_INDEX_STATE_) =
        -Converter::Skew(last_state_ptr->Rwb_ * velo_vec * delta_t);

    Eigen::MatrixXd G = Eigen::MatrixXd::Zero(param_ptr_->STATE_DIM, 9);
    // 式4.21 旋转误差与陀螺仪噪声的关系
    G.block<3, 3>(param_ptr_->ORI_INDEX_STATE_, 0) =
        cur_state_ptr->Rwb_ * Converter::RightJacobianSO3(angular_delta) * delta_t;
    // 式4.22b 陀螺仪偏置的误差状态与随机游走的关系
    G.block<3, 3>(param_ptr_->GYRO_BIAS_INDEX_STATE_, 3) =
        Eigen::Matrix3d::Identity();
    // 式5.21 求当前时刻位移误差与线速度误差的关系
    G.block<3, 3>(param_ptr_->POSI_INDEX, 6) =
        last_state_ptr->Rwb_.toRotationMatrix() * delta_t;

    cur_state_ptr->C_ = last_state_ptr->C_;
    // 计算当前时刻误差状态协方差
    // predict_dispersed_noise_cov_ 表示轮速的噪声协方差矩阵
    // 分为陀螺仪误差、陀螺仪随机游走和线速度三部分
    auto last_99_C = last_state_ptr->C_.block(
        0, 0, param_ptr_->STATE_DIM, param_ptr_->STATE_DIM);
    cur_state_ptr->C_.block(0, 0, param_ptr_->STATE_DIM, param_ptr_->STATE_DIM) =
        F * last_99_C * F.transpose() +
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
    last_data_ = curr_data;

    if (viewer_ptr_)
        viewer_ptr_->DrawWheelPose(cur_state_ptr->Rwb_.toRotationMatrix(), cur_state_ptr->twb_);
}
