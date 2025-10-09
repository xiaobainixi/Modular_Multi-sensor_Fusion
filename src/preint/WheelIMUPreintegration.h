#pragma once

#include "Preintegration.h"

class WheelIMUPreintegration : public Preintegration {
public:
    WheelIMUPreintegration() = delete;
    WheelIMUPreintegration(const WheelIMUData &wheel_imu_data, const State &state, const std::shared_ptr<Parameter> &param_ptr)
        : wheel_imu_data_0_{wheel_imu_data}, first_wheel_imu_data_{wheel_imu_data}
    {
        param_ptr_ = param_ptr;
        bg_ = state.bg_;
        jacobian_ = Eigen::Matrix<double, 9, 9>::Identity();
        covariance_ = Eigen::Matrix<double, 9, 9>::Zero();
        delta_p_ = Eigen::Vector3d::Zero();
        delta_q_ = Eigen::Quaterniond::Identity();

        noise_ = Eigen::Matrix<double, 9, 9>::Zero();
        noise_.block<3, 3>(0, 0) = (param_ptr_->gyro_noise_ * param_ptr_->gyro_noise_) * Eigen::Matrix3d::Identity();
        noise_.block<3, 3>(3, 3) = (param_ptr_->gyro_bias_noise_ * param_ptr_->gyro_bias_noise_) * Eigen::Matrix3d::Identity();
        noise_.block<3, 3>(6, 6) = (param_ptr_->wheel_vel_noise_ * param_ptr_->wheel_vel_noise_) * Eigen::Matrix3d::Identity();
    }

    std::shared_ptr<State> predict(std::shared_ptr<State> state)
    {
        auto new_state = std::make_shared<State>();
        // 位置
        new_state->twb_ = state->twb_ + state->Rwb_ * delta_p_;
        // 姿态
        new_state->Rwb_ = state->Rwb_ * delta_q_.normalized();
        // 零偏
        new_state->bg_ = bg_;

        new_state->time_ = state->time_ + sum_dt_;
        new_state->preint_ = shared_from_this();
        new_state->last_state_ = state;
        return new_state;
    }

    void Input(const WheelIMUData &wheel_imu_data)
    {
        data_buf_.push_back(wheel_imu_data);
        Propagate(wheel_imu_data);
    }

    void Repropagate(const Eigen::Vector3d &new_bg)
    {
        sum_dt_ = 0.0;
        wheel_imu_data_0_ = first_wheel_imu_data_;
        delta_p_.setZero();
        delta_q_.setIdentity();
        // 赋上设置的零偏值
        bg_ = new_bg;
        jacobian_.setIdentity();
        covariance_.setZero();
        for (int i = 0; i < static_cast<int>(data_buf_.size()); i++)
            Propagate(data_buf_[i]);
    }

    void Propagate(const WheelIMUData &wheel_imu_data)
    {
        dt_ = wheel_imu_data.time_ - wheel_imu_data_0_.time_;
        wheel_imu_data_1_ = wheel_imu_data;

        // 差速模型
        double v_avg = 0.5 * (wheel_imu_data_0_.lv_ + wheel_imu_data_0_.rv_ + wheel_imu_data_1_.lv_ + wheel_imu_data_1_.rv_) / 2.0;
        // 计算速度
        Eigen::Vector3d v_local(v_avg, 0, 0); // 车体坐标系前进
        // 计算角度改变量
        Eigen::Vector3d un_gyr = 0.5 * (wheel_imu_data.w_ + wheel_imu_data_0_.w_) - bg_;
        Eigen::Quaterniond dR = Eigen::Quaterniond(1, un_gyr(0) * dt_ / 2, un_gyr(1) * dt_ / 2, un_gyr(2) * dt_ / 2);
        // 协方差递推
        Eigen::Matrix3d V_x;

        V_x << 0, -v_local(2), v_local(1),
            v_local(2), 0, -v_local(0),
            -v_local(1), v_local(0), 0;

        Eigen::Matrix3d dR33 = dR.toRotationMatrix();
        Eigen::MatrixXd F = Eigen::MatrixXd::Zero(9, 9);
        F.block<3, 3>(param_ptr_->POSI_INDEX, param_ptr_->POSI_INDEX) =
            Eigen::Matrix3d::Identity();
        // 注意这里是右乘扰动
        F.block<3, 3>(param_ptr_->POSI_INDEX, param_ptr_->ORI_INDEX_STATE_) =
            -dR33.transpose() * V_x * dt_;

        F.block<3, 3>(param_ptr_->ORI_INDEX_STATE_, param_ptr_->ORI_INDEX_STATE_) =
            dR33.transpose();
        F.block<3, 3>(param_ptr_->ORI_INDEX_STATE_, param_ptr_->GYRO_BIAS_INDEX_STATE_) =
            -1.0 * Eigen::MatrixXd::Identity(3, 3) * dt_;
        F.block<3, 3>(param_ptr_->GYRO_BIAS_INDEX_STATE_, param_ptr_->GYRO_BIAS_INDEX_STATE_) =
            Eigen::Matrix3d::Identity();


        Eigen::MatrixXd V = Eigen::MatrixXd::Zero(9, 9);
        V.block<3, 3>(param_ptr_->POSI_INDEX, 6) =
            delta_q_.toRotationMatrix() * dt_;
        V.block<3, 3>(param_ptr_->ORI_INDEX_STATE_, 0) = Eigen::MatrixXd::Identity(3, 3) * dt_;
        V.block<3, 3>(param_ptr_->GYRO_BIAS_INDEX_STATE_, 3) = Eigen::MatrixXd::Identity(3, 3) * dt_;
        // step_jacobian = F;
        // step_V = V;
        jacobian_ = F * jacobian_;
        // LOG(INFO) << "残差: " << std::endl << F << std::endl << V << std::endl << covariance_;
        // LOG(INFO) << "noise_: " << std::endl << noise_;
        covariance_ = F * covariance_ * F.transpose() + V * noise_ * V.transpose();

        sum_dt_ += dt_;
        wheel_imu_data_0_ = wheel_imu_data_1_;

        delta_q_ = delta_q_ * dR;
        delta_p_ = delta_q_ * v_local * dt_ + delta_p_;
    }

    // 残差计算（可根据实际需求扩展）
    template <typename T>
    bool Evaluate(
        const T *const Pi, const T *const Qi, const T *const Bgi,
        const T *const Pj, const T *const Qj, const T *const Bgj, T *residuals) const
    {
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> Pi_eig(Pi);
        // 输入Qi顺序为xyzw，Eigen四元数构造顺序为wxyz
        Eigen::Quaternion<T> Qi_eig(Qi[3], Qi[0], Qi[1], Qi[2]);
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> Bgi_eig(Bgi);
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> Pj_eig(Pj);
        Eigen::Quaternion<T> Qj_eig(Qj[3], Qj[0], Qj[1], Qj[2]);
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> Bgj_eig(Bgj);

        Eigen::Map<Eigen::Matrix<T, 9, 1>> residuals_eig(residuals);

        Eigen::Matrix<T, 3, 3> dp_dbg = jacobian_.block<3, 3>(param_ptr_->POSI_INDEX, param_ptr_->GYRO_BIAS_INDEX_STATE_).template cast<T>();
        Eigen::Matrix<T, 3, 3> dq_dbg = jacobian_.block<3, 3>(param_ptr_->ORI_INDEX_STATE_, param_ptr_->GYRO_BIAS_INDEX_STATE_).template cast<T>();
        Eigen::Matrix<T, 3, 1> dbg = Bgi_eig - bg_.template cast<T>();

        Eigen::Quaternion<T> delta_q_T = delta_q_.cast<T>();
        Eigen::Quaternion<T> corrected_delta_q = delta_q_T * Converter::deltaQ(dq_dbg * dbg);
        Eigen::Matrix<T, 3, 1> corrected_delta_p = delta_p_.template cast<T>() + dp_dbg * dbg;

        // 位置残差
        residuals_eig.template segment<3>(param_ptr_->POSI_INDEX) = Qi_eig.inverse() * (Pj_eig - Pi_eig) - corrected_delta_p;
        // 姿态残差
        residuals_eig.template segment<3>(param_ptr_->ORI_INDEX_STATE_) = T(2.0) * (corrected_delta_q.inverse() * (Qi_eig.inverse() * Qj_eig)).vec();
        residuals_eig.template segment<3>(param_ptr_->GYRO_BIAS_INDEX_STATE_) = Bgj_eig - Bgi_eig;
        return true;
    }

    WheelIMUData wheel_imu_data_0_, wheel_imu_data_1_;
    const WheelIMUData first_wheel_imu_data_;

    std::vector<WheelIMUData> data_buf_;
};

struct WheelIMUPreintegrationResidual
{
    WheelIMUPreintegrationResidual(std::shared_ptr<WheelIMUPreintegration> preint)
        : preint_(preint) {}

    template <typename T>
    bool operator()(const T *const Pi, const T *const Qi, const T *const Bgi,
                    const T *const Pj, const T *const Qj, const T *const Bgj,
                    T *residuals) const
    {
        preint_->Evaluate(
            Pi, Qi, Bgi, Pj, Qj, Bgj, residuals);
        Eigen::Matrix<double, 9, 9> sqrt_info =
            Eigen::LLT<Eigen::Matrix<double, 9, 9>>(
                preint_->covariance_.inverse()).matrixL().transpose();
        Eigen::Matrix<T, 9, 9> sqrt_info_T = sqrt_info.template cast<T>();
        Eigen::Map<Eigen::Matrix<T, 9, 1>> residuals_eig(residuals);
        residuals_eig = sqrt_info_T * residuals_eig;
        return true;
    }

    std::shared_ptr<WheelIMUPreintegration> preint_;
};