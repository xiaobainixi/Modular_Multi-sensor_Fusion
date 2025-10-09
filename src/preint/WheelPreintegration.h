#pragma once

#include "Preintegration.h"

class WheelPreintegration : public Preintegration {
public:
    WheelPreintegration() = delete;
    WheelPreintegration(const WheelData &wheel_data, const State &state, const std::shared_ptr<Parameter> &param_ptr)
        : wheel_data_0_{wheel_data}, first_wheel_data_{wheel_data}
    {
        param_ptr_ = param_ptr;
        jacobian_ = Eigen::Matrix<double, 6, 6>::Identity();
        covariance_ = Eigen::Matrix<double, 6, 6>::Zero();
        delta_p_ = Eigen::Vector3d::Zero();
        delta_q_ = Eigen::Quaterniond::Identity();

        noise_ = Eigen::Matrix<double, 6, 6>::Zero();
        // 轮速计噪声参数（可根据实际传感器调整） 考虑wheel_b_ 误差
        double w_noise = param_ptr_->wheel_vel_noise_ / param_ptr_->wheel_b_;
        noise_(0, 0) = param_ptr_->wheel_vel_noise_ * param_ptr_->wheel_vel_noise_;  
        noise_(1, 1) = param_ptr_->wheel_vel_noise_ * param_ptr_->wheel_vel_noise_;
        noise_(2, 2) = param_ptr_->wheel_vel_noise_ * param_ptr_->wheel_vel_noise_;
        noise_(3, 3) = w_noise * w_noise;
        noise_(4, 4) = w_noise * w_noise;
        noise_(5, 5) = w_noise * w_noise;
    }

    std::shared_ptr<State> predict(std::shared_ptr<State> state)
    {
        auto new_state = std::make_shared<State>();
        // 位置
        new_state->twb_ = state->twb_ + state->Rwb_ * delta_p_;
        // 姿态
        new_state->Rwb_ = state->Rwb_ * delta_q_.normalized();
        new_state->time_ = state->time_ + sum_dt_;
        new_state->preint_ = shared_from_this();
        new_state->last_state_ = state;
        return new_state;
    }

    void Input(const WheelData &wheel_data)
    {
        data_buf_.push_back(wheel_data);
        Propagate(wheel_data);
    }

    void Repropagate()
    {
        sum_dt_ = 0.0;
        wheel_data_0_ = first_wheel_data_;
        delta_p_.setZero();
        delta_q_.setIdentity();
        jacobian_.setIdentity();
        covariance_.setZero();
        for (int i = 0; i < static_cast<int>(data_buf_.size()); i++)
            Propagate(data_buf_[i]);
    }

    void Propagate(const WheelData &wheel_data)
    {
        dt_ = wheel_data.time_ - wheel_data_0_.time_;
        wheel_data_1_ = wheel_data;

        // 差速模型
        double v_avg = 0.5 * (wheel_data_0_.lv_ + wheel_data_0_.rv_ + wheel_data_1_.lv_ + wheel_data_1_.rv_) / 2.0;
        double v_diff = 0.5 * ((wheel_data_1_.rv_ - wheel_data_1_.lv_) + (wheel_data_0_.rv_ - wheel_data_0_.lv_));
        double wheel_base = param_ptr_->wheel_b_;  // 轮距

        // 计算旋转角度（绕z轴）
        double delta_theta = v_diff * dt_ / wheel_base;
        Eigen::AngleAxisd dR(delta_theta, Eigen::Vector3d::UnitZ());

        // 计算速度
        Eigen::Vector3d v_local(v_avg, 0, 0); // 车体坐标系前进
        

        // 协方差递推
        Eigen::Matrix3d V_x;

        V_x << 0, -v_local(2), v_local(1),
            v_local(2), 0, -v_local(0),
            -v_local(1), v_local(0), 0;

        Eigen::Matrix3d dR33 = dR.toRotationMatrix();
        Eigen::MatrixXd F = Eigen::MatrixXd::Zero(6, 6);
        F.block<3, 3>(param_ptr_->POSI_INDEX, param_ptr_->POSI_INDEX) =
            Eigen::Matrix3d::Identity();
        // 注意这里是右乘扰动
        F.block<3, 3>(param_ptr_->POSI_INDEX, param_ptr_->ORI_INDEX_STATE_) =
            -dR33.transpose() * V_x * dt_;

        F.block<3, 3>(param_ptr_->ORI_INDEX_STATE_, param_ptr_->ORI_INDEX_STATE_) =
            dR33.transpose();

        Eigen::MatrixXd V = Eigen::MatrixXd::Zero(6, 6);
        V.block<3, 3>(param_ptr_->POSI_INDEX, 0) =
            Converter::RightJacobianSO3(0.0, 0.0, delta_theta) * dt_;
        V.block<3, 3>(param_ptr_->ORI_INDEX_STATE_, 3) = delta_q_.toRotationMatrix() * dt_;
        // step_jacobian = F;
        // step_V = V;
        jacobian_ = F * jacobian_;
        // LOG(INFO) << "残差: " << std::endl << F << std::endl << V << std::endl << covariance_;
        covariance_ = F * covariance_ * F.transpose() + V * noise_ * V.transpose();

        sum_dt_ += dt_;
        wheel_data_0_ = wheel_data_1_;

        delta_q_ = delta_q_ * Eigen::Quaterniond(dR);
        delta_p_ = delta_q_ * v_local * dt_ + delta_p_;
    }

    // 残差计算（可根据实际需求扩展）
    template <typename T>
    bool Evaluate(
        const T *const Pi, const T *const Qi,
        const T *const Pj, const T *const Qj, T *residuals) const
    {
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> Pi_eig(Pi);
        // 输入Qi顺序为xyzw，Eigen四元数构造顺序为qxyw
        Eigen::Quaternion<T> Qi_eig(Qi[3], Qi[0], Qi[1], Qi[2]);
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> Pj_eig(Pj);
        Eigen::Quaternion<T> Qj_eig(Qj[3], Qj[0], Qj[1], Qj[2]);

        Eigen::Map<Eigen::Matrix<T, 6, 1>> residuals_eig(residuals);

        // 位置残差
        residuals_eig.template segment<3>(0) = Qi_eig.inverse() * (Pj_eig - Pi_eig) - delta_p_.template cast<T>();
        // 姿态残差
        residuals_eig.template segment<3>(3) = T(2.0) * (delta_q_.cast<T>().inverse() * (Qi_eig.inverse() * Qj_eig)).vec();

        return true;
    }

    WheelData wheel_data_0_, wheel_data_1_;
    const WheelData first_wheel_data_;

    std::vector<WheelData> data_buf_;
};

struct WheelPreintegrationResidual
{
    WheelPreintegrationResidual(std::shared_ptr<WheelPreintegration> preint)
        : preint_(preint) {}

    template <typename T>
    bool operator()(const T *const Pi, const T *const Qi,
                    const T *const Pj, const T *const Qj,
                    T *residuals) const
    {
        preint_->Evaluate(
            Pi, Qi, Pj, Qj, residuals);
        Eigen::Matrix<double, 6, 6> sqrt_info =
            Eigen::LLT<Eigen::Matrix<double, 6, 6>>(
                preint_->covariance_.inverse()).matrixL().transpose();
        Eigen::Matrix<T, 6, 6> sqrt_info_T = sqrt_info.template cast<T>();
        Eigen::Map<Eigen::Matrix<T, 6, 1>> residuals_eig(residuals);
        residuals_eig = sqrt_info_T * residuals_eig;
        return true;
    }

    std::shared_ptr<WheelPreintegration> preint_;
};