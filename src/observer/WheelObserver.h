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

class BaseVelocityResidual : public ceres::SizedCostFunction<3, 3, 4>
{
public:
    BaseVelocityResidual(const Eigen::Vector3d &base_velocity, const std::shared_ptr<Parameter> &param_ptr)
        : base_velocity_(base_velocity), param_ptr_(param_ptr) {}
    bool Evaluate(
        const double *const *parameters, double *residuals,
        double **jacobians) const override
    {
        // 待优化的变量
        // 0: Vw 世界坐标系下的速度
        // 1: Qwb 旋转
        Eigen::Vector3d Vw{parameters[0][0], parameters[0][1], parameters[0][2]};
        Eigen::Quaterniond Qwb(
            parameters[1][3], parameters[1][0], parameters[1][1], parameters[1][2]);

        Eigen::Map<Eigen::Matrix<double, 3, 1>> error(residuals);

        // 式5.53 计算残差，将世界坐标系下的速度转换到车体坐标系下在减去轮速
        error = Qwb.toRotationMatrix().transpose() * Vw - base_velocity_;

        // 开根号的信息矩阵
        Eigen::Matrix3d sqrt_info_ = Eigen::Matrix3d::Zero();
        sqrt_info_(0, 0) = 1.0 / param_ptr_->wheel_vel_noise_;
        sqrt_info_(1, 1) = 1.0 / param_ptr_->wheel_vel_noise_;
        sqrt_info_(2, 2) = 1.0 / param_ptr_->wheel_vel_noise_;

        error = sqrt_info_ * error;

        if (jacobians) {
            if (jacobians[0]) {
                Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> J(jacobians[0]);
                J.setZero();
                // 式5.54a 本体速度残差对世界坐标系下速度的雅可比矩阵
                J.block<3, 3>(0, 0) = Qwb.toRotationMatrix().transpose();
                J = sqrt_info_ * J;
            }

            if (jacobians[1]) {
                Eigen::Map<Eigen::Matrix<double, 3, 4, Eigen::RowMajor>> J(jacobians[1]);
                J.setZero();
                // 式5.54b 本体速度残差对旋转的雅可比矩阵
                J.block<3, 3>(0, 0) =
                    Converter::Skew(Qwb.toRotationMatrix().transpose() * Vw);
                J = sqrt_info_ * J;
            }
        }

        return true;
    }
    Eigen::Vector3d base_velocity_;
    std::shared_ptr<Parameter> param_ptr_;
};