#pragma once

#include "Observer.h"
#include "common/CooTrans.h"
class GNSSObserver : public Observer
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    GNSSObserver(const std::shared_ptr<Parameter> &param_ptr, const std::shared_ptr<DataManager> &data_manager_ptr,
                 const std::shared_ptr<CooTrans> &coo_trans_ptr,
                 const std::shared_ptr<StateManager> &state_manager_ptr, std::shared_ptr<Viewer> viewer_ptr = nullptr)
    {
        viewer_ptr_ = viewer_ptr;
        param_ptr_ = param_ptr;
        data_manager_ptr_ = data_manager_ptr;
        state_manager_ptr_ = state_manager_ptr;
        coo_trans_ptr_ = coo_trans_ptr;

        R_ = Eigen::MatrixXd::Zero(3, 3);
        R_(0, 0) = param_ptr->gnss_x_noise_ * param_ptr->gnss_x_noise_;
        R_(1, 1) = param_ptr->gnss_y_noise_ * param_ptr->gnss_y_noise_;
        R_(2, 2) = param_ptr->gnss_z_noise_ * param_ptr->gnss_z_noise_;
    }

    bool ComputeHZR(const GNSSData &gnss_data, const std::shared_ptr<State> &state_ptr, Eigen::MatrixXd &H, Eigen::MatrixXd &Z, Eigen::MatrixXd &R);

private:
    std::shared_ptr<CooTrans> coo_trans_ptr_;
};

// GNSS残差
// struct GNSSResidual
// {
//     GNSSResidual(const Eigen::Vector3d &gnss_pos, const std::shared_ptr<Parameter> &param_ptr)
//         : gnss_pos_(gnss_pos), param_ptr_(param_ptr) {}
//     template <typename T>
//     bool operator()(const T *const pose, T *residual) const
//     {
//         residual[0] = (pose[0] - T(gnss_pos_.x())) / T(param_ptr_->gnss_x_noise_);
//         residual[1] = (pose[1] - T(gnss_pos_.y())) / T(param_ptr_->gnss_y_noise_);
//         residual[2] = (pose[2] - T(gnss_pos_.z())) / T(param_ptr_->gnss_z_noise_);
//         return true;
//     }
//     Eigen::Vector3d gnss_pos_;
//     std::shared_ptr<Parameter> param_ptr_;
// };

class GNSSResidual : public ceres::SizedCostFunction<3, 3>
{
public:
    GNSSResidual(const Eigen::Vector3d &gnss_pos, const std::shared_ptr<Parameter> &param_ptr)
        : gnss_pos_(gnss_pos), param_ptr_(param_ptr) {}
    bool Evaluate(const double *const *parameters, double *residuals, double **jacobians) const override {
        Eigen::Vector3d p{parameters[0][0], parameters[0][1], parameters[0][2]};

        Eigen::Map<Eigen::Matrix<double, 3, 1>> error(residuals);

        error = p - gnss_pos_;

        Eigen::Matrix3d sqrt_info_ = Eigen::Matrix3d::Zero();
        sqrt_info_(0, 0)    = 1.0 / param_ptr_->gnss_x_noise_;
        sqrt_info_(1, 1)    = 1.0 / param_ptr_->gnss_y_noise_;
        sqrt_info_(2, 2)    = 1.0 / param_ptr_->gnss_z_noise_;

        error = sqrt_info_ * error;

        if (jacobians) {
            if (jacobians[0]) {
                Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> jacobian_pose(jacobians[0]);
                jacobian_pose.setZero();
                jacobian_pose.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
                jacobian_pose = sqrt_info_ * jacobian_pose;
            }
        }

        return true;
    }
    Eigen::Vector3d gnss_pos_;
    std::shared_ptr<Parameter> param_ptr_;
};