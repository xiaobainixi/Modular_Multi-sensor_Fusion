#pragma once

#include "Observer.h"
class WheelObserver : public Observer
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    WheelObserver(const std::shared_ptr<Parameter> &param_ptr, const std::shared_ptr<DataManager> &data_manager_ptr,
                const std::shared_ptr<StateManager> &state_manager_ptr)
                
    {
        param_ptr_ = param_ptr;
        data_manager_ptr_ = data_manager_ptr;
        state_manager_ptr_ = state_manager_ptr;

        H_ = Eigen::MatrixXd::Zero(3, param_ptr->STATE_DIM);

        R_ = Eigen::MatrixXd::Zero(3, 3);
        R_(0, 0) = param_ptr->wheel_x_noise_;
        R_(1, 1) = param_ptr->wheel_y_noise_;
        R_(2, 2) = param_ptr->wheel_z_noise_;
    }

    bool ComputeHZR(const WheelData & wheel_data, const std::shared_ptr<State> & state_ptr, Eigen::MatrixXd & H, Eigen::MatrixXd & Z, Eigen::MatrixXd &R);

private:
};