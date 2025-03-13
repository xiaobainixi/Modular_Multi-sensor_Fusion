#include "GPSWheelObserver.h"

bool GPSWheelObserver::ComputeHZR(
    const WheelData & wheel_data, 
    const GPSData & gps_data, const std::shared_ptr<State> & state_ptr,
    Eigen::MatrixXd & H, Eigen::MatrixXd & Z, Eigen::MatrixXd &R)
{
    // 1. gps_data
    double x = 0.0, y = 0.0, z = 0.0;
    coo_trans_ptr_->getENH(gps_data.lat_, gps_data.lon_, gps_data.h_, x, y, z);
    if (viewer_ptr_)
        viewer_ptr_->DrawGps(Eigen::Vector3d(x, y, z));

    // 2. wheel
    H_ = Eigen::MatrixXd::Zero(6, param_ptr_->STATE_DIM + state_manager_ptr_->cam_states_.size() * 6);
    H_.block<3, 3>(0, param_ptr_->POSI_INDEX) = Eigen::Matrix3d::Identity();
    H_.block<3, 3>(3, param_ptr_->VEL_INDEX_STATE_) = state_ptr->Rwb_.transpose();
    H_.block<3, 3>(3, param_ptr_->ORI_INDEX_STATE_) = state_ptr->Rwb_.transpose() * Converter::Skew(state_ptr->Vw_);

    // 3. compute
    Eigen::Vector3d gps_enu(x, y, z);
    Eigen::Vector3d t_error = gps_enu - state_ptr->twb_;
    Eigen::Vector3d Vb_m((wheel_data.lv_ + wheel_data.rv_) * 0.5, 0.0, 0.0);
    // todo 没有考虑外参
    Eigen::Vector3d v_error = Vb_m - state_ptr->Rwb_.transpose() * state_ptr->Vw_;
    Z = Eigen::VectorXd::Zero(6);
    Z.block<3, 1>(0, 0) = t_error;
    Z.block<3, 1>(3, 0) = v_error;
    H = H_;
    R = R_;
    return true;
}