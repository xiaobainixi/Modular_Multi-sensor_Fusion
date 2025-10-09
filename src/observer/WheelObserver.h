#pragma once

#include "Observer.h"
class WheelObserver : public Observer
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    WheelObserver(const std::shared_ptr<Parameter> &param_ptr, const std::shared_ptr<DataManager> &data_manager_ptr,
                const std::shared_ptr<StateManager> &state_manager_ptr, std::shared_ptr<Viewer> viewer_ptr = nullptr)
                
    {
        param_ptr_ = param_ptr;
        data_manager_ptr_ = data_manager_ptr;
        state_manager_ptr_ = state_manager_ptr;

        R_ = Eigen::MatrixXd::Zero(3, 3);
        R_(0, 0) = param_ptr->wheel_vel_noise_ * param_ptr->wheel_vel_noise_ * 100.0 * 100.0;
        R_(1, 1) = param_ptr->wheel_vel_noise_ * param_ptr->wheel_vel_noise_ * 100.0 * 100.0;  // 1e-8 * 1e-8;
        R_(2, 2) = param_ptr->wheel_vel_noise_ * param_ptr->wheel_vel_noise_ * 100.0 * 100.0;  // 1e-8 * 1e-8;
    }

    bool ComputeHZR(const WheelData & wheel_data, const std::shared_ptr<State> & state_ptr, Eigen::MatrixXd & H, Eigen::MatrixXd & Z, Eigen::MatrixXd &R);

private:
};