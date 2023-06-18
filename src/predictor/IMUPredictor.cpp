#include "IMUPredictor.h"

// todo 数个一起来还是一个一个，数个一起来比较好
void IMUPredictor::Predict() {
    IMUData cur_data = data_manager_ptr_->GetLastIMUData();
    if (cur_data.time_ <= 0.0)
        return;
    std::cout << cur_data.time_ << std::endl;
    std::cout << cur_data.a_.transpose() << std::endl;
    std::cout << cur_data.w_.transpose() << std::endl;
    // 第一个数据
    if (last_data_.time_ <= 0.0) {
        std::shared_ptr<State<IMUData>> cur_state_ptr = std::make_shared<State<IMUData>>();
        cur_state_ptr->time_ = cur_data.time_;
        cur_state_ptr->aligned_data_ = cur_data;
        state_manager_ptr_->PushState(cur_state_ptr);
        last_data_ = cur_data;
        return;
    }
    double delta_t = cur_data.time_ - last_data_.time_;
    if (delta_t <= 0.0)
        return;
    std::shared_ptr<State<IMUData>> cur_state_ptr = std::make_shared<State<IMUData>>();
    cur_state_ptr->time_ = cur_data.time_;
    cur_state_ptr->aligned_data_ = cur_data;

    std::shared_ptr<State<IMUData>> last_state_ptr;
    state_manager_ptr_->GetNearestState(last_state_ptr);

    //-------------------------------------------------------------------------------------------------------------
    // 计算当前状态的角度
    Eigen::Vector3d cur_unbias_angular_vel = cur_data.w_ - last_state_ptr->bg_;
    Eigen::Vector3d last_unbias_angular_vel = last_data_.w_ - last_state_ptr->bg_;
    Eigen::Vector3d angular_delta = 0.5 * (cur_unbias_angular_vel + last_unbias_angular_vel) * delta_t;

    cur_state_ptr->Rwb_ = last_state_ptr->Rwb_ * Eigen::AngleAxisd(angular_delta.norm(), angular_delta.normalized()).toRotationMatrix();

    //-------------------------------------------------------------------------------------------------------------
    // 计算当前状态的速度
    Eigen::Vector3d last_v, avg_a;
    Eigen::Vector3d cur_unbias_a = cur_state_ptr->Rwb_ * (cur_data.a_ - last_state_ptr->bg_) + gw_;
    Eigen::Vector3d last_unbias_a = last_state_ptr->Rwb_ * (last_data_.a_ - last_state_ptr->bg_) + gw_;

    last_v = last_state_ptr->Vw_;
    avg_a = 0.5 * (cur_unbias_a + last_unbias_a);
    cur_state_ptr->Vw_ += delta_t * avg_a;

    //---------------------------------------------------------------------------------------------------
    // 使用计算出的速度算位置变化
    cur_state_ptr->twb_ += delta_t * last_v + 0.5 * avg_a * delta_t * delta_t;

    //---------------------------------------------------------------------------------------------------
    // 计算协方差矩阵
    // 定义： 估计数值 = 理想数值 + 误差

    Eigen::Matrix3d F_13 = Converter::Skew(cur_unbias_a);

    
    // std::cout<<"after"<<std::endl<<F_23<<std::endl;
    mF.block<3, 3>(INDEX_STATE_POSI, INDEX_STATE_ORI) = -0.5 * F_13 * delta_t;
    mF.block<3, 3>(INDEX_STATE_POSI, INDEX_STATE_ACC_BIAS) = 0.5 * cur_state_ptr->Rwb_ * delta_t;

    mF.block<3, 3>(INDEX_STATE_VEL, INDEX_STATE_ORI) = -F_13;
    mF.block<3, 3>(INDEX_STATE_VEL, INDEX_STATE_ACC_BIAS) = cur_state_ptr->Rwb_;
    mF.block<3, 3>(INDEX_STATE_ORI, INDEX_STATE_GYRO_BIAS) = cur_state_ptr->Rwb_; // cur_state_ptr->Rwb_;  Eigen::Matrix3d::Identity()
    mB.block<3, 3>(INDEX_STATE_VEL, 3) = cur_state_ptr->Rwb_;
    mB.block<3, 3>(INDEX_STATE_ORI, 0) = cur_state_ptr->Rwb_; // cur_state_ptr->Rwb_;  Eigen::Matrix3d::Identity()

    Eigen::MatrixXd Fk = Eigen::MatrixXd::Identity(DIM_STATE, DIM_STATE) + mF * delta_t;

    Eigen::MatrixXd Bk = mB * delta_t;
    mFt = mF;

    mX = Fk * mX;
    mState.covariance = Fk * mState.covariance * Fk.transpose() + Bk * mQ * Bk.transpose();

    state_manager_ptr_->PushState(cur_state_ptr);
    last_data_ = cur_data;
}