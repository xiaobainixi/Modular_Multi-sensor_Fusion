#include "WheelObserver.h"

bool WheelObserver::ComputeHZR(
    const WheelData & wheel_data, const std::shared_ptr<State> & state_ptr, Eigen::MatrixXd & H, Eigen::MatrixXd & Z, Eigen::MatrixXd &R)
{
    H_ = Eigen::MatrixXd::Zero(3, param_ptr_->STATE_DIM + state_manager_ptr_->cam_states_.size() * 6);
    H_.block<3, 3>(0, param_ptr_->VEL_INDEX_STATE_) = state_ptr->Rwb_.toRotationMatrix().transpose();
    H_.block<3, 3>(0, param_ptr_->ORI_INDEX_STATE_) = state_ptr->Rwb_.toRotationMatrix().transpose() * Converter::Skew(state_ptr->Vw_);
    Eigen::Vector3d Vb_m((wheel_data.lv_ + wheel_data.rv_) * 0.5, 0.0, 0.0);
    // todo 没有考虑外参
    Eigen::Vector3d v_error = Vb_m - state_ptr->Rwb_.toRotationMatrix().transpose() * state_ptr->Vw_;
    Z = Eigen::VectorXd::Zero(3);
    Z.block<3, 1>(0, 0) = v_error;
    H = H_;
    R = R_;
    return true;
}