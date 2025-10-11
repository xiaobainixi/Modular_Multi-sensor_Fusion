#pragma once

#include "Preintegration.h"

class IMUPreintegration : public Preintegration
{
public:
    IMUPreintegration() = delete;
    IMUPreintegration(const IMUData &imu_data, const State &state, const std::shared_ptr<Parameter> &param_ptr)
        : imu_data_0_{imu_data}, first_imu_data_{imu_data}
    {
        param_ptr_ = param_ptr;
        ba_ = state.ba_;
        bg_ = state.bg_;
        jacobian_ = Eigen::Matrix<double, 15, 15>::Identity();
        covariance_ = Eigen::Matrix<double, 15, 15>::Zero();
        delta_p_ = Eigen::Vector3d::Zero();
        delta_q_ = Eigen::Quaterniond::Identity();
        delta_v_ = Eigen::Vector3d::Zero();

        noise_ = Eigen::Matrix<double, 18, 18>::Zero();
        noise_.block<3, 3>(0, 0) = (param_ptr_->acc_noise_ * param_ptr_->acc_noise_) * Eigen::Matrix3d::Identity();
        noise_.block<3, 3>(3, 3) = (param_ptr_->gyro_noise_ * param_ptr_->gyro_noise_) * Eigen::Matrix3d::Identity();
        noise_.block<3, 3>(6, 6) = (param_ptr_->acc_noise_ * param_ptr_->acc_noise_) * Eigen::Matrix3d::Identity();
        noise_.block<3, 3>(9, 9) = (param_ptr_->gyro_noise_ * param_ptr_->gyro_noise_) * Eigen::Matrix3d::Identity();
        noise_.block<3, 3>(12, 12) = (param_ptr_->acc_bias_noise_ * param_ptr_->acc_bias_noise_) * Eigen::Matrix3d::Identity();
        noise_.block<3, 3>(15, 15) = (param_ptr_->gyro_bias_noise_ * param_ptr_->gyro_bias_noise_) * Eigen::Matrix3d::Identity();
    }

    std::shared_ptr<State> predict(std::shared_ptr<State> state)
    {
        // 1. 创建新状态对象
        auto new_state = std::make_shared<State>();

        // new_state->twb_ = state->twb_;
        // new_state->Rwb_ = state->Rwb_;
        // new_state->Vw_ = state->Vw_;
        // LOG(INFO) << "位置增量 " << (state->twb_).transpose() << ", 速度增量 " << (state->Vw_).transpose()
        //           << ", 姿态增量 " << (state->Rwb_).coeffs().transpose();
        // LOG(INFO) << "预积分结果: 位置增量 " << (state->twb_ + state->Vw_ * sum_dt_ + 0.5 * param_ptr_->gw_ * sum_dt_ * sum_dt_ + state->Rwb_ * delta_p_).transpose() << ", 速度增量 " << (state->Vw_ + param_ptr_->gw_ * sum_dt_ + state->Rwb_ * delta_v_).transpose()
        //           << ", 姿态增量 " << (state->Rwb_ * delta_q_.normalized()).coeffs().transpose();

        // 2. 用预积分结果更新状态
        // 位置
        new_state->twb_ = state->twb_ + state->Vw_ * sum_dt_ + 0.5 * param_ptr_->gw_ * sum_dt_ * sum_dt_ + state->Rwb_ * delta_p_;
        // 姿态
        new_state->Rwb_ = state->Rwb_ * delta_q_.normalized();
        // 速度
        new_state->Vw_ = state->Vw_ + param_ptr_->gw_ * sum_dt_ + state->Rwb_ * delta_v_;
        // 零偏
        new_state->ba_ = ba_;
        new_state->bg_ = bg_;

        new_state->time_ = state->time_ + sum_dt_;
        new_state->preint_ = shared_from_this();
        new_state->last_state_ = state;
        return new_state;
    }
    void Input(const IMUData &imu_data)
    {
        // 相关时间差和传感器数据保留，方便后续repropagate
        data_buf_.push_back(imu_data);
        Propagate(imu_data);
    }

    /**
     * @brief 根据新设置的imu零偏重新对该帧进行预积分
     *
     * @param[in] new_ba
     * @param[in] new_bg
     */
    void Repropagate(const Eigen::Vector3d &new_ba, const Eigen::Vector3d &new_bg)
    {
        // 状态量全部清零
        sum_dt_ = 0.0;
        imu_data_0_ = first_imu_data_;
        delta_p_.setZero();
        delta_q_.setIdentity();
        delta_v_.setZero();
        // 赋上设置的零偏值
        ba_ = new_ba;
        bg_ = new_bg;
        jacobian_.setIdentity();
        covariance_.setZero();
        // 用之前存下来的imu值重新预积分
        for (int i = 0; i < static_cast<int>(data_buf_.size()); i++)
            Propagate(data_buf_[i]);
    }

    void MidPointIntegration(Eigen::Vector3d &result_delta_p, Eigen::Quaterniond &result_delta_q,
                             Eigen::Vector3d &result_delta_v, Eigen::Vector3d &result_ba, Eigen::Vector3d &result_bg)
    {
        // ROS_INFO("midpoint integration");
        //  首先中值积分更新状态量
        Eigen::Vector3d un_acc_0 = delta_q_ * (imu_data_0_.a_ - ba_);
        Eigen::Vector3d un_gyr = 0.5 * (imu_data_0_.w_ + imu_data_1_.w_) - bg_;
        result_delta_q = delta_q_ * Converter::RotVecToQuaternion(un_gyr * dt_);
        // result_delta_q = delta_q_ * Eigen::Quaterniond(1, un_gyr(0) * dt_ / 2, un_gyr(1) * dt_ / 2, un_gyr(2) * dt_ / 2);
        Eigen::Vector3d un_acc_1 = result_delta_q * (imu_data_1_.a_ - ba_);
        Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
        result_delta_p = delta_p_ + delta_v_ * dt_ + 0.5 * un_acc * dt_ * dt_;
        result_delta_v = delta_v_ + un_acc * dt_;
        result_ba = ba_;
        result_bg = bg_;
        // 随后更新方差矩阵及雅克比

        Eigen::Vector3d w_x = 0.5 * (imu_data_0_.w_ + imu_data_1_.w_) - bg_;
        Eigen::Vector3d a_0_x = imu_data_0_.a_ - ba_;
        Eigen::Vector3d a_1_x = imu_data_1_.a_ - ba_;
        Eigen::Matrix3d R_w_x, R_a_0_x, R_a_1_x;

        R_w_x << 0, -w_x(2), w_x(1),
            w_x(2), 0, -w_x(0),
            -w_x(1), w_x(0), 0;
        R_a_0_x << 0, -a_0_x(2), a_0_x(1),
            a_0_x(2), 0, -a_0_x(0),
            -a_0_x(1), a_0_x(0), 0;
        R_a_1_x << 0, -a_1_x(2), a_1_x(1),
            a_1_x(2), 0, -a_1_x(0),
            -a_1_x(1), a_1_x(0), 0;

        Eigen::MatrixXd F = Eigen::MatrixXd::Zero(15, 15);
        F.block<3, 3>(param_ptr_->POSI_INDEX, param_ptr_->POSI_INDEX) =
            Eigen::Matrix3d::Identity();
        F.block<3, 3>(param_ptr_->POSI_INDEX, param_ptr_->ORI_INDEX_STATE_) =
            -0.25 * delta_q_.toRotationMatrix() * R_a_0_x * dt_ * dt_ +
            -0.25 * result_delta_q.toRotationMatrix() * R_a_1_x *
                (Eigen::Matrix3d::Identity() - R_w_x * dt_) * dt_ * dt_;
        F.block<3, 3>(param_ptr_->POSI_INDEX, param_ptr_->VEL_INDEX_STATE_) =
            Eigen::MatrixXd::Identity(3, 3) * dt_;
        F.block<3, 3>(param_ptr_->POSI_INDEX, param_ptr_->ACC_BIAS_INDEX_STATE_) =
            -0.25 * (delta_q_.toRotationMatrix() + result_delta_q.toRotationMatrix()) * dt_ * dt_;
        F.block<3, 3>(param_ptr_->POSI_INDEX, param_ptr_->GYRO_BIAS_INDEX_STATE_) =
            -0.25 * result_delta_q.toRotationMatrix() * R_a_1_x * dt_ * dt_ * -dt_;


        F.block<3, 3>(param_ptr_->ORI_INDEX_STATE_, param_ptr_->ORI_INDEX_STATE_) =
            Eigen::Matrix3d::Identity() - R_w_x * dt_;
        F.block<3, 3>(param_ptr_->ORI_INDEX_STATE_, param_ptr_->GYRO_BIAS_INDEX_STATE_) =
            -1.0 * Eigen::MatrixXd::Identity(3, 3) * dt_;


        F.block<3, 3>(param_ptr_->VEL_INDEX_STATE_, param_ptr_->ORI_INDEX_STATE_) =
            -0.5 * delta_q_.toRotationMatrix() * R_a_0_x * dt_ +
            -0.5 * result_delta_q.toRotationMatrix() * R_a_1_x *
                (Eigen::Matrix3d::Identity() - R_w_x * dt_) * dt_;
        F.block<3, 3>(param_ptr_->VEL_INDEX_STATE_, param_ptr_->VEL_INDEX_STATE_) =
            Eigen::Matrix3d::Identity();
        F.block<3, 3>(param_ptr_->VEL_INDEX_STATE_, param_ptr_->ACC_BIAS_INDEX_STATE_) =
            -0.5 * (delta_q_.toRotationMatrix() + result_delta_q.toRotationMatrix()) * dt_;
        F.block<3, 3>(param_ptr_->VEL_INDEX_STATE_, param_ptr_->GYRO_BIAS_INDEX_STATE_) =
            -0.5 * result_delta_q.toRotationMatrix() * R_a_1_x * dt_ * -dt_;


        F.block<3, 3>(param_ptr_->ACC_BIAS_INDEX_STATE_, param_ptr_->ACC_BIAS_INDEX_STATE_) =
            Eigen::Matrix3d::Identity();
        F.block<3, 3>(param_ptr_->GYRO_BIAS_INDEX_STATE_, param_ptr_->GYRO_BIAS_INDEX_STATE_) =
            Eigen::Matrix3d::Identity();
        // cout<<"A"<<endl<<A<<endl;

        Eigen::MatrixXd V = Eigen::MatrixXd::Zero(15, 18);
        V.block<3, 3>(param_ptr_->POSI_INDEX, 0) =
            0.25 * delta_q_.toRotationMatrix() * dt_ * dt_;
        V.block<3, 3>(param_ptr_->POSI_INDEX, 3) =
            0.25 * -result_delta_q.toRotationMatrix() * R_a_1_x * dt_ * dt_ * 0.5 * dt_;
        V.block<3, 3>(param_ptr_->POSI_INDEX, 6) =
            0.25 * result_delta_q.toRotationMatrix() * dt_ * dt_;
        V.block<3, 3>(param_ptr_->POSI_INDEX, 9) = V.block<3, 3>(param_ptr_->POSI_INDEX, 3);

        V.block<3, 3>(param_ptr_->ORI_INDEX_STATE_, 3) = 0.5 * Eigen::MatrixXd::Identity(3, 3) * dt_;
        V.block<3, 3>(param_ptr_->ORI_INDEX_STATE_, 9) = 0.5 * Eigen::MatrixXd::Identity(3, 3) * dt_;

        V.block<3, 3>(param_ptr_->VEL_INDEX_STATE_, 0) = 0.5 * delta_q_.toRotationMatrix() * dt_;
        V.block<3, 3>(param_ptr_->VEL_INDEX_STATE_, 3) =
            0.5 * -result_delta_q.toRotationMatrix() * R_a_1_x * dt_ * 0.5 * dt_;
        V.block<3, 3>(param_ptr_->VEL_INDEX_STATE_, 6) = 0.5 * result_delta_q.toRotationMatrix() * dt_;
        V.block<3, 3>(param_ptr_->VEL_INDEX_STATE_, 9) = V.block<3, 3>(param_ptr_->VEL_INDEX_STATE_, 3);

        V.block<3, 3>(param_ptr_->ACC_BIAS_INDEX_STATE_, 12) = Eigen::MatrixXd::Identity(3, 3) * dt_;
        V.block<3, 3>(param_ptr_->GYRO_BIAS_INDEX_STATE_, 15) = Eigen::MatrixXd::Identity(3, 3) * dt_;

        // step_jacobian = F;
        // step_V = V;
        jacobian_ = F * jacobian_;
        covariance_ = F * covariance_ * F.transpose() + V * noise_ * V.transpose();
    }

    void EulerIntegration(
        Eigen::Vector3d &result_delta_p, Eigen::Quaterniond &result_delta_q, Eigen::Vector3d &result_delta_v,
        Eigen::Vector3d &result_linearized_ba, Eigen::Vector3d &result_linearized_bg)
    {
        result_delta_p = delta_p_ + delta_v_ * dt_ + 0.5 * (delta_q_ * (imu_data_1_.a_ - ba_)) * dt_ * dt_;
        result_delta_v = delta_v_ + delta_q_ * (imu_data_1_.a_ - ba_) * dt_;
        Eigen::Vector3d omg = imu_data_1_.w_ - bg_;
        Eigen::Quaterniond dR = Converter::RotVecToQuaternion(omg * dt_);
        result_delta_q = (delta_q_ * dR);
        result_linearized_ba = ba_;
        result_linearized_bg = bg_;

        Eigen::Vector3d w_x = imu_data_1_.w_ - bg_;
        Eigen::Vector3d a_x = imu_data_1_.a_ - ba_;
        Eigen::Matrix3d R_w_x, R_a_x;

        R_w_x << 0, -w_x(2), w_x(1),
            w_x(2), 0, -w_x(0),
            -w_x(1), w_x(0), 0;
        R_a_x << 0, -a_x(2), a_x(1),
            a_x(2), 0, -a_x(0),
            -a_x(1), a_x(0), 0;

        Eigen::MatrixXd A = Eigen::MatrixXd::Zero(15, 15);
        // one step euler 0.5
        A.block<3, 3>(param_ptr_->POSI_INDEX, param_ptr_->ORI_INDEX_STATE_) =
            0.5 * (-1 * delta_q_.toRotationMatrix()) * R_a_x * dt_;
        A.block<3, 3>(param_ptr_->POSI_INDEX, param_ptr_->VEL_INDEX_STATE_) =
            Eigen::MatrixXd::Identity(3, 3);
        A.block<3, 3>(param_ptr_->POSI_INDEX, param_ptr_->ACC_BIAS_INDEX_STATE_) =
            0.5 * (-1 * delta_q_.toRotationMatrix()) * dt_;
        A.block<3, 3>(param_ptr_->ORI_INDEX_STATE_, param_ptr_->ORI_INDEX_STATE_) = -R_w_x;
        A.block<3, 3>(param_ptr_->ORI_INDEX_STATE_, param_ptr_->GYRO_BIAS_INDEX_STATE_) =
            -1 * Eigen::MatrixXd::Identity(3, 3);
        A.block<3, 3>(param_ptr_->VEL_INDEX_STATE_, param_ptr_->ORI_INDEX_STATE_) =
            (-1 * delta_q_.toRotationMatrix()) * R_a_x;
        A.block<3, 3>(param_ptr_->VEL_INDEX_STATE_, param_ptr_->ACC_BIAS_INDEX_STATE_) =
            (-1 * delta_q_.toRotationMatrix());
        // cout<<"A"<<endl<<A<<endl;

        Eigen::MatrixXd U = Eigen::MatrixXd::Zero(15, 12);
        U.block<3, 3>(param_ptr_->POSI_INDEX, 0) = 0.5 * delta_q_.toRotationMatrix() * dt_;
        U.block<3, 3>(param_ptr_->ORI_INDEX_STATE_, 3) = Eigen::MatrixXd::Identity(3, 3);
        U.block<3, 3>(param_ptr_->VEL_INDEX_STATE_, 0) = delta_q_.toRotationMatrix();
        U.block<3, 3>(param_ptr_->ACC_BIAS_INDEX_STATE_, 6) = Eigen::MatrixXd::Identity(3, 3);
        U.block<3, 3>(param_ptr_->GYRO_BIAS_INDEX_STATE_, 9) = Eigen::MatrixXd::Identity(3, 3);

        // write F directly
        Eigen::MatrixXd F, V;
        F = (Eigen::MatrixXd::Identity(15, 15) + dt_ * A);
        V = dt_ * U;
        // step_jacobian = F;
        // step_V = V;
        jacobian_ = F * jacobian_;
        covariance_ = F * covariance_ * F.transpose() + V * noise_ * V.transpose();
    }

    void Propagate(const IMUData &imu_data)
    {
        dt_ = imu_data.time_ - imu_data_0_.time_;
        imu_data_1_ = imu_data;
        Eigen::Vector3d result_delta_p;
        Eigen::Quaterniond result_delta_q;
        Eigen::Vector3d result_delta_v;
        Eigen::Vector3d result_ba;
        Eigen::Vector3d result_bg;

        MidPointIntegration(result_delta_p, result_delta_q, result_delta_v, result_ba, result_bg);

        delta_p_ = result_delta_p;
        delta_q_ = result_delta_q;
        delta_v_ = result_delta_v;
        ba_ = result_ba;
        bg_ = result_bg;
        delta_q_.normalize();
        sum_dt_ += dt_;
        imu_data_0_ = imu_data_1_;
    }

    // 计算和给定相邻帧状态量的残差
    Eigen::Matrix<double, 15, 1> Evaluate(
        const Eigen::Vector3d &Pi, const Eigen::Quaterniond &Qi, const Eigen::Vector3d &Vi,
        const Eigen::Vector3d &Bai, const Eigen::Vector3d &Bgi,
        const Eigen::Vector3d &Pj, const Eigen::Quaterniond &Qj, const Eigen::Vector3d &Vj,
        const Eigen::Vector3d &Baj, const Eigen::Vector3d &Bgj)
    {
        Eigen::Matrix<double, 15, 1> residuals;

        Eigen::Matrix3d dp_dba = jacobian_.block<3, 3>(param_ptr_->POSI_INDEX, param_ptr_->ACC_BIAS_INDEX_STATE_);
        Eigen::Matrix3d dp_dbg = jacobian_.block<3, 3>(param_ptr_->POSI_INDEX, param_ptr_->GYRO_BIAS_INDEX_STATE_);

        Eigen::Matrix3d dq_dbg = jacobian_.block<3, 3>(param_ptr_->ORI_INDEX_STATE_, param_ptr_->GYRO_BIAS_INDEX_STATE_);

        Eigen::Matrix3d dv_dba = jacobian_.block<3, 3>(param_ptr_->VEL_INDEX_STATE_, param_ptr_->ACC_BIAS_INDEX_STATE_);
        Eigen::Matrix3d dv_dbg = jacobian_.block<3, 3>(param_ptr_->VEL_INDEX_STATE_, param_ptr_->GYRO_BIAS_INDEX_STATE_);

        Eigen::Vector3d dba = Bai - ba_;
        Eigen::Vector3d dbg = Bgi - bg_;

        Eigen::Quaterniond corrected_delta_q = delta_q_ * Converter::RotVecToQuaternion(dq_dbg * dbg);
        Eigen::Vector3d corrected_delta_v = delta_v_ + dv_dba * dba + dv_dbg * dbg;
        Eigen::Vector3d corrected_delta_p = delta_p_ + dp_dba * dba + dp_dbg * dbg;

        residuals.block<3, 1>(param_ptr_->POSI_INDEX, 0) =
            Qi.inverse() * (-0.5 * param_ptr_->gw_ * sum_dt_ * sum_dt_ + Pj - Pi - Vi * sum_dt_) - corrected_delta_p;
        residuals.block<3, 1>(param_ptr_->ORI_INDEX_STATE_, 0) =
            2.0 * (corrected_delta_q.inverse() * (Qi.inverse() * Qj)).vec();
        residuals.block<3, 1>(param_ptr_->VEL_INDEX_STATE_, 0) =
            Qi.inverse() * (-param_ptr_->gw_ * sum_dt_ + Vj - Vi) - corrected_delta_v;
        residuals.block<3, 1>(param_ptr_->ACC_BIAS_INDEX_STATE_, 0) = Baj - Bai;
        residuals.block<3, 1>(param_ptr_->GYRO_BIAS_INDEX_STATE_, 0) = Bgj - Bgi;
        return residuals;
    }

    // template <typename T>
    // bool Evaluate(
    //     const T *const Pi, const T *const Qi, const T *const Vi,
    //     const T *const Bai, const T *const Bgi,
    //     const T *const Pj, const T *const Qj, const T *const Vj,
    //     const T *const Baj, const T *const Bgj, T *residuals) const
    // {
    //     Eigen::Map<const Eigen::Matrix<T, 3, 1>> Pi_eig(Pi);
    //     // 输入Qi顺序为xyzw，Eigen四元数构造顺序为wxyz
    //     Eigen::Quaternion<T> Qi_eig(Qi[3], Qi[0], Qi[1], Qi[2]);
    //     Eigen::Map<const Eigen::Matrix<T, 3, 1>> Vi_eig(Vi);
    //     Eigen::Map<const Eigen::Matrix<T, 3, 1>> Bai_eig(Bai);
    //     Eigen::Map<const Eigen::Matrix<T, 3, 1>> Bgi_eig(Bgi);

    //     Eigen::Map<const Eigen::Matrix<T, 3, 1>> Pj_eig(Pj);
    //     Eigen::Quaternion<T> Qj_eig(Qj[3], Qj[0], Qj[1], Qj[2]);
    //     Eigen::Map<const Eigen::Matrix<T, 3, 1>> Vj_eig(Vj);
    //     Eigen::Map<const Eigen::Matrix<T, 3, 1>> Baj_eig(Baj);
    //     Eigen::Map<const Eigen::Matrix<T, 3, 1>> Bgj_eig(Bgj);

    //     Eigen::Map<Eigen::Matrix<T, 15, 1>> residuals_eig(residuals);

    //     Eigen::Matrix<T, 3, 3> dp_dba = jacobian_.block<3, 3>(param_ptr_->POSI_INDEX, param_ptr_->ACC_BIAS_INDEX_STATE_).template cast<T>();
    //     Eigen::Matrix<T, 3, 3> dp_dbg = jacobian_.block<3, 3>(param_ptr_->POSI_INDEX, param_ptr_->GYRO_BIAS_INDEX_STATE_).template cast<T>();

    //     Eigen::Matrix<T, 3, 3> dq_dbg = jacobian_.block<3, 3>(param_ptr_->ORI_INDEX_STATE_, param_ptr_->GYRO_BIAS_INDEX_STATE_).template cast<T>();

    //     Eigen::Matrix<T, 3, 3> dv_dba = jacobian_.block<3, 3>(param_ptr_->VEL_INDEX_STATE_, param_ptr_->ACC_BIAS_INDEX_STATE_).template cast<T>();
    //     Eigen::Matrix<T, 3, 3> dv_dbg = jacobian_.block<3, 3>(param_ptr_->VEL_INDEX_STATE_, param_ptr_->GYRO_BIAS_INDEX_STATE_).template cast<T>();

    //     Eigen::Matrix<T, 3, 1> dba = Bai_eig - ba_.template cast<T>();
    //     Eigen::Matrix<T, 3, 1> dbg = Bgi_eig - bg_.template cast<T>();

    //     Eigen::Quaternion<T> delta_q_T = delta_q_.cast<T>();
    //     Eigen::Quaternion<T> corrected_delta_q = delta_q_T * Converter::RotVecToQuaternion(dq_dbg * dbg);
    //     Eigen::Matrix<T, 3, 1> corrected_delta_v = delta_v_.template cast<T>() + dv_dba * dba + dv_dbg * dbg;
    //     Eigen::Matrix<T, 3, 1> corrected_delta_p = delta_p_.template cast<T>() + dp_dba * dba + dp_dbg * dbg;

    //     T sum_dt_T = T(sum_dt_);
    //     Eigen::Matrix<T, 3, 1> gw_T = param_ptr_->gw_.template cast<T>();

    //     // residuals_eig.block<3, 1>(param_ptr_->POSI_INDEX, 0) = Qi_eig.inverse() * (T(0.5) * gw_T * sum_dt_T * sum_dt_T + Pj_eig - Pi_eig - Vi_eig * sum_dt_T) - corrected_delta_p;
    //     // residuals_eig.block<3, 1>(param_ptr_->ORI_INDEX_STATE_, 0) = T(2.0) * (corrected_delta_q.inverse() * (Qi_eig.inverse() * Qj_eig)).vec();
    //     // residuals_eig.block<3, 1>(param_ptr_->VEL_INDEX_STATE_, 0) = Qi_eig.inverse() * (gw_T * sum_dt_T + Vj_eig - Vi_eig) - corrected_delta_v;
    //     // residuals_eig.block<3, 1>(param_ptr_->ACC_BIAS_INDEX_STATE_, 0) = Baj_eig - Bai_eig;
    //     // residuals_eig.block<3, 1>(param_ptr_->GYRO_BIAS_INDEX_STATE_, 0) = Bgj_eig - Bgi_eig;
    //     residuals_eig.template segment<3>(param_ptr_->POSI_INDEX) =
    //         Qi_eig.inverse() * (-T(0.5) * gw_T * sum_dt_T * sum_dt_T + Pj_eig - Pi_eig - Vi_eig * sum_dt_T) - corrected_delta_p;
    //     residuals_eig.template segment<3>(param_ptr_->ORI_INDEX_STATE_) = T(2.0) * (corrected_delta_q.inverse() * (Qi_eig.inverse() * Qj_eig)).vec();
    //     residuals_eig.template segment<3>(param_ptr_->VEL_INDEX_STATE_) = Qi_eig.inverse() * (-gw_T * sum_dt_T + Vj_eig - Vi_eig) - corrected_delta_v;
    //     residuals_eig.template segment<3>(param_ptr_->ACC_BIAS_INDEX_STATE_) = Baj_eig - Bai_eig;
    //     residuals_eig.template segment<3>(param_ptr_->GYRO_BIAS_INDEX_STATE_) = Bgj_eig - Bgi_eig;

    //     return true;
    // }

    IMUData imu_data_0_, imu_data_1_;
    const IMUData first_imu_data_;

    std::vector<IMUData> data_buf_;
};

// struct IMUPreintegrationResidual
// {
//     IMUPreintegrationResidual(std::shared_ptr<IMUPreintegration> preint)
//         : preint_(preint) {}

//     template <typename T>
//     bool operator()(const T *const Pi, const T *const Qi, const T *const Vi,
//                     const T *const Bai, const T *const Bgi,
//                     const T *const Pj, const T *const Qj, const T *const Vj,
//                     const T *const Baj, const T *const Bgj,
//                     T *residuals) const
//     {
//         preint_->Evaluate(
//             Pi, Qi, Vi, Bai, Bgi,
//             Pj, Qj, Vj, Baj, Bgj, residuals);
//         Eigen::Matrix<double, 15, 15> sqrt_info =
//             Eigen::LLT<Eigen::Matrix<double, 15, 15>>(
//                 preint_->covariance_.inverse()).matrixL().transpose();
//         Eigen::Matrix<T, 15, 15> sqrt_info_T = sqrt_info.template cast<T>();
//         Eigen::Map<Eigen::Matrix<T, 15, 1>> residuals_eig(residuals);
//         residuals_eig = sqrt_info_T * residuals_eig;
//         return true;
//     }

//     std::shared_ptr<IMUPreintegration> preint_;
// };

class IMUPreintegrationResidual : public ceres::SizedCostFunction<15, 3, 4, 3, 3, 3, 3, 4, 3, 3, 3>
{
public:
    IMUPreintegrationResidual() = delete;
    IMUPreintegrationResidual(std::shared_ptr<IMUPreintegration> preint, const std::shared_ptr<Parameter> &param_ptr)
        : preint_(preint), param_ptr_(param_ptr)
    {
    }
    /**
     * @brief  使用ceres解析求导，必须重载这个函数
     *
     * @param[in] parameters 这是一个二维数组，每个参数块都是一个double数组，而一个观测会对多个参数块形成约束
     * @param[in] residuals 残差的计算结果，是一个一维数组，残差就是该观测量和约束的状态量通过某种关系形成残差
     * @param[in] jacobians 残差对参数块的雅克比矩阵，这也是一个二维数组，对任意一个参数块的雅克比矩阵都是一个一维数组
     * @return true
     * @return false
     */
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
    {
        // 便于后续计算，把参数块都转换成eigen
        // imu预积分的约束的参数是相邻两帧的位姿 速度和零偏
        Eigen::Vector3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
        Eigen::Quaterniond Qi(parameters[1][3], parameters[1][0], parameters[1][1], parameters[1][2]);

        Eigen::Vector3d Vi(parameters[2][0], parameters[2][1], parameters[2][2]);
        Eigen::Vector3d Bai(parameters[3][0], parameters[3][1], parameters[3][2]);
        Eigen::Vector3d Bgi(parameters[4][0], parameters[4][1], parameters[4][2]);

        Eigen::Vector3d Pj(parameters[5][0], parameters[5][1], parameters[5][2]);
        Eigen::Quaterniond Qj(parameters[6][3], parameters[6][0], parameters[6][1], parameters[6][2]);

        Eigen::Vector3d Vj(parameters[7][0], parameters[7][1], parameters[7][2]);
        Eigen::Vector3d Baj(parameters[8][0], parameters[8][1], parameters[8][2]);
        Eigen::Vector3d Bgj(parameters[9][0], parameters[9][1], parameters[9][2]);


        Eigen::Map<Eigen::Matrix<double, 15, 1>> residual(residuals);
        // 得到残差
        residual = preint_->Evaluate(Pi, Qi, Vi, Bai, Bgi, Pj, Qj, Vj, Baj, Bgj);
        // 因为ceres没有g2o设置信息矩阵的接口，因此置信度直接乘在残差上，这里通过LLT分解，相当于将信息矩阵开根号
        Eigen::Matrix<double, 15, 15> sqrt_info =
            Eigen::LLT<Eigen::Matrix<double, 15, 15>>(
                preint_->covariance_.inverse()).matrixL().transpose();
        //  这就是带有信息矩阵的残差
        residual = sqrt_info * residual;
        // 关于雅克比的计算手动推导一下
        if (jacobians)
        {
            double sum_dt = preint_->sum_dt_;
            Eigen::Matrix3d dp_dba = preint_->jacobian_.template block<3, 3>(param_ptr_->POSI_INDEX, param_ptr_->ACC_BIAS_INDEX_STATE_);
            Eigen::Matrix3d dp_dbg = preint_->jacobian_.template block<3, 3>(param_ptr_->POSI_INDEX, param_ptr_->GYRO_BIAS_INDEX_STATE_);

            Eigen::Matrix3d dq_dbg = preint_->jacobian_.template block<3, 3>(param_ptr_->ORI_INDEX_STATE_, param_ptr_->GYRO_BIAS_INDEX_STATE_);

            Eigen::Matrix3d dv_dba = preint_->jacobian_.template block<3, 3>(param_ptr_->VEL_INDEX_STATE_, param_ptr_->ACC_BIAS_INDEX_STATE_);
            Eigen::Matrix3d dv_dbg = preint_->jacobian_.template block<3, 3>(param_ptr_->VEL_INDEX_STATE_, param_ptr_->GYRO_BIAS_INDEX_STATE_);

            if (preint_->jacobian_.maxCoeff() > 1e8 || preint_->jacobian_.minCoeff() < -1e8)
            {
                LOG(ERROR) << "numerical unstable in preintegration";
            }

            // 下面开始求预积分相对于各个参数的雅克比
            // 9个参数块，分别是 Pi Qi Vi Bai Bgi Pj Qj Vj Baj Bgj
            // 每个参数块的维度分别是 3 4 3 3 3 3 4 3 3 3
            // 0: Pi
            if (jacobians[0])
            {
                Eigen::Map<Eigen::Matrix<double, 15, 3, Eigen::RowMajor>> J(jacobians[0]);
                J.setZero();
                J.block<3,3>(param_ptr_->POSI_INDEX, 0) = -Qi.inverse().toRotationMatrix();
                J = sqrt_info * J;
            }
            // 1: Qi
            if (jacobians[1])
            {
                Eigen::Map<Eigen::Matrix<double, 15, 4, Eigen::RowMajor>> J(jacobians[1]);
                J.setZero();
                J.block<3,3>(param_ptr_->POSI_INDEX, 0) = Converter::Skew(Qi.inverse() * (-0.5 * param_ptr_->gw_ * sum_dt * sum_dt + Pj - Pi - Vi * sum_dt));
                Eigen::Quaterniond corrected_delta_q = preint_->delta_q_ * Converter::RotVecToQuaternion(dq_dbg * (Bgi - preint_->bg_));
                J.block<3,3>(param_ptr_->ORI_INDEX_STATE_, 0) = -(Converter::Qleft(Qj.inverse() * Qi) * Converter::Qright(corrected_delta_q)).bottomRightCorner<3,3>();
                J.block<3,3>(param_ptr_->VEL_INDEX_STATE_, 0) = Converter::Skew(Qi.inverse() * (-param_ptr_->gw_ * sum_dt + Vj - Vi));
                J = sqrt_info * J;
            }
            // 2: Vi
            if (jacobians[2])
            {
                Eigen::Map<Eigen::Matrix<double, 15, 3, Eigen::RowMajor>> J(jacobians[2]);
                J.setZero();
                J.block<3,3>(param_ptr_->POSI_INDEX, 0) = -Qi.inverse().toRotationMatrix() * sum_dt;
                J.block<3,3>(param_ptr_->VEL_INDEX_STATE_, 0) = -Qi.inverse().toRotationMatrix();
                J = sqrt_info * J;
            }
            // 3: Bai
            if (jacobians[3])
            {
                Eigen::Map<Eigen::Matrix<double, 15, 3, Eigen::RowMajor>> J(jacobians[3]);
                J.setZero();
                J.block<3,3>(param_ptr_->POSI_INDEX, 0) = -dp_dba;
                J.block<3,3>(param_ptr_->VEL_INDEX_STATE_, 0) = -dv_dba;
                J.block<3,3>(param_ptr_->ACC_BIAS_INDEX_STATE_, 0) = -Eigen::Matrix3d::Identity();
                J = sqrt_info * J;
            }
            // 4: Bgi
            if (jacobians[4])
            {
                Eigen::Map<Eigen::Matrix<double, 15, 3, Eigen::RowMajor>> J(jacobians[4]);
                J.setZero();
                J.block<3,3>(param_ptr_->POSI_INDEX, 0) = -dp_dbg;
                J.block<3,3>(param_ptr_->ORI_INDEX_STATE_, 0) =
                    -Converter::Qleft(Qj.inverse() * Qi * preint_->delta_q_).bottomRightCorner<3,3>() * dq_dbg;
                J.block<3,3>(param_ptr_->VEL_INDEX_STATE_, 0) = -dv_dbg;
                J.block<3,3>(param_ptr_->GYRO_BIAS_INDEX_STATE_, 0) = -Eigen::Matrix3d::Identity();
                J = sqrt_info * J;
            }
            // 5: Pj
            if (jacobians[5])
            {
                Eigen::Map<Eigen::Matrix<double, 15, 3, Eigen::RowMajor>> J(jacobians[5]);
                J.setZero();
                J.block<3,3>(param_ptr_->POSI_INDEX, 0) = Qi.inverse().toRotationMatrix();
                J = sqrt_info * J;
            }
            // 6: Qj
            if (jacobians[6])
            {
                Eigen::Map<Eigen::Matrix<double, 15, 4, Eigen::RowMajor>> J(jacobians[6]);
                J.setZero();
                Eigen::Quaterniond corrected_delta_q = preint_->delta_q_ * Converter::RotVecToQuaternion(dq_dbg * (Bgi - preint_->bg_));
                J.block<3,3>(param_ptr_->ORI_INDEX_STATE_, 0) = Converter::Qleft(corrected_delta_q.inverse() * Qi.inverse() * Qj).bottomRightCorner<3,3>();
                J = sqrt_info * J;
            }
            // 7: Vj
            if (jacobians[7])
            {
                Eigen::Map<Eigen::Matrix<double, 15, 3, Eigen::RowMajor>> J(jacobians[7]);
                J.setZero();
                J.block<3,3>(param_ptr_->VEL_INDEX_STATE_, 0) = Qi.inverse().toRotationMatrix();
                J = sqrt_info * J;
            }
            // 8: Baj
            if (jacobians[8])
            {
                Eigen::Map<Eigen::Matrix<double, 15, 3, Eigen::RowMajor>> J(jacobians[8]);
                J.setZero();
                J.block<3,3>(param_ptr_->ACC_BIAS_INDEX_STATE_, 0) = Eigen::Matrix3d::Identity();
                J = sqrt_info * J;
            }
            // 9: Bgj
            if (jacobians[9])
            {
                Eigen::Map<Eigen::Matrix<double, 15, 3, Eigen::RowMajor>> J(jacobians[9]);
                J.setZero();
                J.block<3,3>(param_ptr_->GYRO_BIAS_INDEX_STATE_, 0) = Eigen::Matrix3d::Identity();
                J = sqrt_info * J;
            }
        }

        return true;
    }

    // bool Evaluate_Direct(double const *const *parameters, Eigen::Matrix<double, 15, 1> &residuals, Eigen::Matrix<double, 15, 30> &jacobians);

    // void checkCorrection();
    // void checkTransition();
    // void checkJacobian(double **parameters);
    std::shared_ptr<IMUPreintegration> preint_;
    std::shared_ptr<Parameter> param_ptr_;
};
