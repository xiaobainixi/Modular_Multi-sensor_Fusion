#pragma once

#include "Observer.h"
#include "common/CooTrans.h"
class GPSWheelObserver : public Observer
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    GPSWheelObserver(
        const std::shared_ptr<Parameter> &param_ptr, const std::shared_ptr<DataManager> &data_manager_ptr,
        const std::shared_ptr<StateManager> &state_manager_ptr, std::shared_ptr<Viewer> viewer_ptr = nullptr)
                
    {
        viewer_ptr_ = viewer_ptr;
        param_ptr_ = param_ptr;
        data_manager_ptr_ = data_manager_ptr;
        state_manager_ptr_ = state_manager_ptr;

        R_ = Eigen::MatrixXd::Zero(6, 6);
        R_(0, 0) = param_ptr->gps_x_noise_ * param_ptr->gps_x_noise_;
        R_(1, 1) = param_ptr->gps_y_noise_ * param_ptr->gps_y_noise_;
        R_(2, 2) = param_ptr->gps_z_noise_ * param_ptr->gps_z_noise_;
        R_(3, 3) = param_ptr->wheel_x_noise_ * param_ptr->wheel_x_noise_;
        R_(4, 4) = param_ptr->wheel_y_noise_ * param_ptr->wheel_y_noise_;
        R_(5, 5) = param_ptr->wheel_z_noise_ * param_ptr->wheel_z_noise_;
    }

    bool ComputeHZR(
        const WheelData & wheel_data, const GPSData & gps_data,
        const std::shared_ptr<State> & state_ptr, Eigen::MatrixXd & H,
        Eigen::MatrixXd & Z, Eigen::MatrixXd &R);

private:
    std::shared_ptr<CooTrans> coo_trans_ptr_;
};