#include "WheelPredictor.h"

// todo 数个一起来还是一个一个，数个一起来比较好
void WheelPredictor::Run() {
    while(1) {
        RunOnce();
        usleep(100);
    }
}

void WheelPredictor::RunOnce() {
    WheelData cur_data;
    if (!data_manager_ptr_->GetLastWheelData(cur_data, last_data_.time_)) {
        return;
    }
    
    // 第一个数据，推荐只用6维
    if (last_data_.time_ <= 0.0) {
        if (state_manager_ptr_->Empty()) {
            std::shared_ptr<State> cur_state_ptr = std::make_shared<State>();
            cur_state_ptr->time_ = cur_data.time_;
            cur_state_ptr->C_ = Eigen::MatrixXd::Identity(param_ptr_->STATE_DIM, param_ptr_->STATE_DIM);
            cur_state_ptr->C_.block<3, 3>(param_ptr_->POSI_INDEX, param_ptr_->POSI_INDEX) = Eigen::Matrix3d::Identity() * 0.25;
            cur_state_ptr->C_.block<3, 3>(param_ptr_->ORI_INDEX_STATE_, param_ptr_->ORI_INDEX_STATE_) = Eigen::Matrix3d::Identity() * 0.25;
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

    //--------------------------------------------------------------------------------
    // 根据轮速数据计算线速度和角速度
    // cur_data.lv_是左轮速度，cur_data.rv_是右轮速度，单位m/s
    // param_ptr_->wheel_b_是轴长，单位m
    // 式5.9a 线速度v = (lv + rv) * 0.5
    double v = (cur_data.lv_ + cur_data.rv_) * 0.5;
    // 式5.9c 角速度w = (rv - lv) / wheel_b
    double w = (cur_data.rv_ - cur_data.lv_) / param_ptr_->wheel_b_;
    // 式5.9b 只能给出x轴方向的速度
    Eigen::Vector3d velo_vec(v, 0, 0);   // x, y, z 三个轴的线速度
    // 式5.9d 只能给出绕z轴方向的角速度
    Eigen::Vector3d omega_vec(0, 0, w); // x, y, z 三个轴的角速度

    //--------------------------------------------------------------------------------
    // cur_state_ptr是当前时刻状态的智能指针，该状态包含位姿(四元数)、协方差等信息
    // last_state_ptr是上一时刻状态的智能指针
    std::shared_ptr<State> last_state_ptr;
    state_manager_ptr_->GetNearestState(last_state_ptr);

    std::shared_ptr<State> cur_state_ptr = std::make_shared<State>();
    cur_state_ptr->time_ = cur_data.time_;

    //--------------------------------------------------------------------------------
    // 该模式下不存在IMU，因此不考虑速度和偏置的更新，非常简单
    // 式5.10a 计算当前时刻状态的旋转
    cur_state_ptr->Rwb_ =
        last_state_ptr->Rwb_ * Converter::so3ToQuat(omega_vec * delta_t);

    // 式5.10b 计算当前时刻状态的位移，世界坐标系下
    cur_state_ptr->twb_ =
        last_state_ptr->twb_ + last_state_ptr->Rwb_ * velo_vec * delta_t;

    //--------------------------------------------------------------------------------
    // 计算协方差矩阵
    // note: F是状态转移矩阵，G是噪声转移矩阵（对应书中B矩阵）
    // 定义： 理想数值（优质数值） = 估计数值 + 误差

    // 这里只使用位移和旋转作为状态，param_ptr_->STATE_DIM 等于6
    // 初始单位矩阵，已经填好当前时刻位移误差与上时刻位移误差的关系
    // 以及当前时刻旋转误差与上时刻旋转误差的关系
    Eigen::MatrixXd F =
        Eigen::MatrixXd::Identity(param_ptr_->STATE_DIM, param_ptr_->STATE_DIM);
    // 式5.14 求当前时刻位移误差与上时刻旋转误差的关系
    F.block<3, 3>(param_ptr_->POSI_INDEX, param_ptr_->ORI_INDEX_STATE_) =
        -Converter::Skew(last_state_ptr->Rwb_ * velo_vec * delta_t);

    // 噪声转移矩阵
    Eigen::MatrixXd G = Eigen::MatrixXd::Zero(param_ptr_->STATE_DIM, 6);
    // 式5.21 求当前时刻位移误差与线速度误差的关系
    G.block<3, 3>(param_ptr_->POSI_INDEX, 0) =
        last_state_ptr->Rwb_.toRotationMatrix() * delta_t;
    // 式5.20 求当前时刻旋转误差与角速度误差的关系
    // RightJacobianSO3 表示计算向量的右雅可比矩阵
    G.block<3, 3>(param_ptr_->ORI_INDEX_STATE_, 3) =
        cur_state_ptr->Rwb_ *
        Converter::RightJacobianSO3(omega_vec * delta_t) * delta_t;

    cur_state_ptr->C_ = last_state_ptr->C_;
    // 式5.24 计算当前时刻误差状态协方差
    // predict_dispersed_noise_cov_ 表示轮速的噪声协方差矩阵，分为线速度和角速度两部分
    auto last_66_C = last_state_ptr->C_.block(
        0, 0, param_ptr_->STATE_DIM, param_ptr_->STATE_DIM);
    cur_state_ptr->C_.block(0, 0, param_ptr_->STATE_DIM, param_ptr_->STATE_DIM) =
        F * last_66_C * F.transpose() +
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

    // if (viewer_ptr_)
    //     viewer_ptr_->DrawWheelOdom(cur_state_ptr->Rwb_, cur_state_ptr->twb_);
    if (viewer_ptr_)
        viewer_ptr_->DrawWheelPose(cur_state_ptr->Rwb_.toRotationMatrix(), cur_state_ptr->twb_);
}
