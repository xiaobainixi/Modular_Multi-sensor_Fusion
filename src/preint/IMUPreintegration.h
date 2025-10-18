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

    /**
     * @brief 中值法（mid-point）单步预积分
     * @details 利用 imu_data_0_ 与 imu_data_1_ 及当前线性化零偏 ba_, bg_ 与历史增量
     *          (delta_p_, delta_q_, delta_v_) 计算从时刻 i 到 j 的增量：
     *          result_delta_q = delta_q_ * Exp( ( (w0+w1)/2 - bg_ ) * dt )
     *          先用旧姿态与新姿态分别去旋转去偏后的加速度，取平均获得中值加速度，再积分
     *          result_delta_v = delta_v_ + a_mid * dt
     *          result_delta_p = delta_p_ + delta_v_ * dt + 0.5 * a_mid * dt^2
     *          其中 a_mid = 0.5*( delta_q_*(a0-ba_) + result_delta_q*(a1-ba_) )
     * @param[out] result_delta_p  位置预积分增量 p_{ij} （表达在 i 时刻坐标系）
     * @param[out] result_delta_q  姿态预积分增量 q_{ij} （从 i 到 j 的旋转，右乘）
     * @param[out] result_delta_v  速度预积分增量 v_{ij} （表达在 i 时刻坐标系）
     * @param[out] result_ba       线加计零偏（此步不更新，直接传递）
     * @param[out] result_bg       陀螺零偏（此步不更新，直接传递）
     * @note 本函数只计算“候选”增量；成员 delta_* 的真正更新在外层 Propagate 中完成。
     *       同时在函数中构建 F 与 V 并更新 jacobian_ 与 covariance_（式6.132, 6.148-6.149）。
     */
    void MidPointIntegration(
        Eigen::Vector3d &result_delta_p, Eigen::Quaterniond &result_delta_q,
        Eigen::Vector3d &result_delta_v, Eigen::Vector3d &result_ba,
        Eigen::Vector3d &result_bg)
    {
        //--------------------------------------------------------------------------------
        // 1. 中值积分更新状态量，注意更新顺序！
        // imu_data_0_ imu_data_1_ 表示相邻两个IMU数据
        // 此处delta_q_ delta_v_ delta_p_ 表示上一次预积分得到的结果
        // result_delta_q result_delta_v result_delta_p 表示本次预积分得到的结果
        Eigen::Vector3d un_acc_0 = delta_q_ * (imu_data_0_.a_ - ba_);
        // 计算经过偏置矫正后的角速度中值
        Eigen::Vector3d un_gyr = 0.5 * (imu_data_0_.w_ + imu_data_1_.w_) - bg_;
        // 式6.132c
        result_delta_q = delta_q_ * Converter::RotVecToQuaternion(un_gyr * dt_);
        Eigen::Vector3d un_acc_1 = result_delta_q * (imu_data_1_.a_ - ba_);
        Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
        // 式6.132a
        result_delta_p = delta_p_ + delta_v_ * dt_ + 0.5 * un_acc * dt_ * dt_;
        // 式6.132b
        result_delta_v = delta_v_ + un_acc * dt_;
        result_ba = ba_;
        result_bg = bg_;
        //--------------------------------------------------------------------------------
        // 2. 计算预积分相对于偏置的雅可比矩阵和预积分的协方差
        // 准备计算FV矩阵需要的元素，这些元素会重复使用
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

        // 计算F矩阵
        Eigen::MatrixXd F = Eigen::MatrixXd::Zero(15, 15);
        // 式6.148
        F.block<3, 3>(param_ptr_->POSI_INDEX, param_ptr_->POSI_INDEX) =
            Eigen::Matrix3d::Identity();
        // 式6.149a F01
        F.block<3, 3>(param_ptr_->POSI_INDEX, param_ptr_->ORI_INDEX_STATE_) =
            -0.25 * delta_q_.toRotationMatrix() * R_a_0_x * dt_ * dt_ +
            -0.25 * result_delta_q.toRotationMatrix() * R_a_1_x *
                (Eigen::Matrix3d::Identity() - R_w_x * dt_) * dt_ * dt_;
        // 式6.148
        F.block<3, 3>(param_ptr_->POSI_INDEX, param_ptr_->VEL_INDEX_STATE_) =
            Eigen::MatrixXd::Identity(3, 3) * dt_;
        // 式6.149b F03
        F.block<3, 3>(param_ptr_->POSI_INDEX, param_ptr_->ACC_BIAS_INDEX_STATE_) =
            -0.25 * (delta_q_.toRotationMatrix() + result_delta_q.toRotationMatrix()) *
            dt_ * dt_;
        // 式6.149c  F04
        F.block<3, 3>(param_ptr_->POSI_INDEX, param_ptr_->GYRO_BIAS_INDEX_STATE_) =
            -0.25 * result_delta_q.toRotationMatrix() * R_a_1_x * dt_ * dt_ * -dt_;

        // 式6.149d  F11
        F.block<3, 3>(param_ptr_->ORI_INDEX_STATE_, param_ptr_->ORI_INDEX_STATE_) =
            Eigen::Matrix3d::Identity() - R_w_x * dt_;
        // 式6.148
        F.block<3, 3>(param_ptr_->ORI_INDEX_STATE_, param_ptr_->GYRO_BIAS_INDEX_STATE_) =
            -1.0 * Eigen::MatrixXd::Identity(3, 3) * dt_;

        // 式6.149e  F21
        F.block<3, 3>(param_ptr_->VEL_INDEX_STATE_, param_ptr_->ORI_INDEX_STATE_) =
            -0.5 * delta_q_.toRotationMatrix() * R_a_0_x * dt_ +
            -0.5 * result_delta_q.toRotationMatrix() * R_a_1_x *
                (Eigen::Matrix3d::Identity() - R_w_x * dt_) * dt_;
        // 式6.148
        F.block<3, 3>(param_ptr_->VEL_INDEX_STATE_, param_ptr_->VEL_INDEX_STATE_) =
            Eigen::Matrix3d::Identity();
        // 式6.149f F23
        F.block<3, 3>(param_ptr_->VEL_INDEX_STATE_, param_ptr_->ACC_BIAS_INDEX_STATE_) =
            -0.5 * (delta_q_.toRotationMatrix() + result_delta_q.toRotationMatrix()) * dt_;
        // 式6.149g F24
        F.block<3, 3>(param_ptr_->VEL_INDEX_STATE_, param_ptr_->GYRO_BIAS_INDEX_STATE_) =
            -0.5 * result_delta_q.toRotationMatrix() * R_a_1_x * dt_ * -dt_;

        // 式6.148
        F.block<3, 3>(param_ptr_->ACC_BIAS_INDEX_STATE_, param_ptr_->ACC_BIAS_INDEX_STATE_) =
            Eigen::Matrix3d::Identity();
        F.block<3, 3>(param_ptr_->GYRO_BIAS_INDEX_STATE_, param_ptr_->GYRO_BIAS_INDEX_STATE_) =
            Eigen::Matrix3d::Identity();

        // 计算V矩阵
        Eigen::MatrixXd V = Eigen::MatrixXd::Zero(15, 18);
        // 式6.149h V00
        V.block<3, 3>(param_ptr_->POSI_INDEX, 0) =
            0.25 * delta_q_.toRotationMatrix() * dt_ * dt_;
        // 式6.149i V01
        V.block<3, 3>(param_ptr_->POSI_INDEX, 3) =
            0.25 * -result_delta_q.toRotationMatrix() * R_a_1_x * dt_ * dt_ * 0.5 * dt_;
        // 式6.149j V02
        V.block<3, 3>(param_ptr_->POSI_INDEX, 6) =
            0.25 * result_delta_q.toRotationMatrix() * dt_ * dt_;
        // 式6.149i V03
        V.block<3, 3>(param_ptr_->POSI_INDEX, 9) =
            V.block<3, 3>(param_ptr_->POSI_INDEX, 3);

        // 式6.148
        V.block<3, 3>(param_ptr_->ORI_INDEX_STATE_, 3) =
            0.5 * Eigen::MatrixXd::Identity(3, 3) * dt_;
        V.block<3, 3>(param_ptr_->ORI_INDEX_STATE_, 9) =
            0.5 * Eigen::MatrixXd::Identity(3, 3) * dt_;

        // 式6.149k V20
        V.block<3, 3>(param_ptr_->VEL_INDEX_STATE_, 0) =
            0.5 * delta_q_.toRotationMatrix() * dt_;
        // 式6.149l V21
        V.block<3, 3>(param_ptr_->VEL_INDEX_STATE_, 3) =
            0.5 * -result_delta_q.toRotationMatrix() * R_a_1_x * dt_ * 0.5 * dt_;
        // 式6.149m V22
        V.block<3, 3>(param_ptr_->VEL_INDEX_STATE_, 6) =
            0.5 * result_delta_q.toRotationMatrix() * dt_;
        // 式6.149n V23
        V.block<3, 3>(param_ptr_->VEL_INDEX_STATE_, 9) =
            V.block<3, 3>(param_ptr_->VEL_INDEX_STATE_, 3);

        // 式6.148
        V.block<3, 3>(param_ptr_->ACC_BIAS_INDEX_STATE_, 12) =
            Eigen::MatrixXd::Identity(3, 3) * dt_;
        V.block<3, 3>(param_ptr_->GYRO_BIAS_INDEX_STATE_, 15) =
            Eigen::MatrixXd::Identity(3, 3) * dt_;

        // 式6.139 更新雅可比矩阵
        jacobian_ = F * jacobian_;
        // 式6.134 更新协方差矩阵
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

    /**
     * @brief 计算预积分残差
     * 
     * @param Pi i时刻位移
     * @param Qi i时刻旋转
     * @param Vi i时刻速度
     * @param Bai i时刻加速度计零偏
     * @param Bgi i时刻陀螺仪零偏
     * @param Pj j时刻位移
     * @param Qj j时刻旋转
     * @param Vj j时刻速度
     * @param Baj j时刻加速度计零偏
     * @param Bgj j时刻陀螺仪零偏
     * @return Eigen::Matrix<double, 15, 1> 残差
     */
    Eigen::Matrix<double, 15, 1> Evaluate(
        const Eigen::Vector3d &Pi, const Eigen::Quaterniond &Qi,
        const Eigen::Vector3d &Vi, const Eigen::Vector3d &Bai,
        const Eigen::Vector3d &Bgi,
        const Eigen::Vector3d &Pj, const Eigen::Quaterniond &Qj,
        const Eigen::Vector3d &Vj, const Eigen::Vector3d &Baj,
        const Eigen::Vector3d &Bgj)
    {
        Eigen::Matrix<double, 15, 1> residuals;

        // 1. 提取预积分相对于偏置的雅可比矩阵，用于计算预积分的更新值
        // 位移预积分相对于加速度计偏置的雅可比
        Eigen::Matrix3d dp_dba = jacobian_.block<3, 3>(
            param_ptr_->POSI_INDEX, param_ptr_->ACC_BIAS_INDEX_STATE_);
        // 位移预积分相对于陀螺仪偏置的雅可比
        Eigen::Matrix3d dp_dbg = jacobian_.block<3, 3>(
            param_ptr_->POSI_INDEX, param_ptr_->GYRO_BIAS_INDEX_STATE_);

        // 旋转预积分相对于陀螺仪偏置的雅可比
        Eigen::Matrix3d dq_dbg = jacobian_.block<3, 3>(
            param_ptr_->ORI_INDEX_STATE_, param_ptr_->GYRO_BIAS_INDEX_STATE_);

        // 速度预积分相对于加速度计偏置的雅可比
        Eigen::Matrix3d dv_dba = jacobian_.block<3, 3>(
            param_ptr_->VEL_INDEX_STATE_, param_ptr_->ACC_BIAS_INDEX_STATE_);
        // 速度预积分相对于陀螺仪偏置的雅可比
        Eigen::Matrix3d dv_dbg = jacobian_.block<3, 3>(
            param_ptr_->VEL_INDEX_STATE_, param_ptr_->GYRO_BIAS_INDEX_STATE_);

        // 2. 计算偏置更新量
        Eigen::Vector3d dba = Bai - ba_;
        Eigen::Vector3d dbg = Bgi - bg_;

        // 3. 计算偏置更新后的预积分
        // 式6.141c 旋转预积分更新
        Eigen::Quaterniond corrected_delta_q =
            delta_q_ * Converter::RotVecToQuaternion(dq_dbg * dbg);
        // 式6.141b 速度预积分更新 
        Eigen::Vector3d corrected_delta_v =
            delta_v_ + dv_dba * dba + dv_dbg * dbg;
        // 式6.141a 位置预积分更新
        Eigen::Vector3d corrected_delta_p =
            delta_p_ + dp_dba * dba + dp_dbg * dbg;

        // 4. 式6.173 计算残差
        // 位置残差
        residuals.block<3, 1>(param_ptr_->POSI_INDEX, 0) =
            Qi.inverse() * (-0.5 * param_ptr_->gw_ * sum_dt_ * sum_dt_ +
                Pj - Pi - Vi * sum_dt_) - corrected_delta_p;
        // 姿态残差
        residuals.block<3, 1>(param_ptr_->ORI_INDEX_STATE_, 0) =
            2.0 * (corrected_delta_q.inverse() * (Qi.inverse() * Qj)).vec();
        // 速度残差
        residuals.block<3, 1>(param_ptr_->VEL_INDEX_STATE_, 0) =
            Qi.inverse() * (-param_ptr_->gw_ * sum_dt_ + Vj - Vi) -
            corrected_delta_v;
        // 偏置残差
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
    virtual bool Evaluate(
        double const *const *parameters, double *residuals,
        double **jacobians) const
    {
        // 便于后续计算，把参数块都转换成eigen
        // imu预积分的约束的参数是相邻两帧的位姿 速度和偏置
        // 读取Ceres参数块
        // i时刻位移
        Eigen::Vector3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
        // i时刻旋转 注意输入顺序为xyzw，Eigen四元数构造顺序为qxyw
        Eigen::Quaterniond Qi(
            parameters[1][3], parameters[1][0], parameters[1][1], parameters[1][2]);

        // i时刻速度
        Eigen::Vector3d Vi(parameters[2][0], parameters[2][1], parameters[2][2]);
        // i时刻加速度计偏置
        Eigen::Vector3d Bai(parameters[3][0], parameters[3][1], parameters[3][2]);
        // i时刻陀螺仪偏置
        Eigen::Vector3d Bgi(parameters[4][0], parameters[4][1], parameters[4][2]);

        // j时刻位移
        Eigen::Vector3d Pj(parameters[5][0], parameters[5][1], parameters[5][2]);
        // j时刻旋转 注意输入顺序为xyzw，Eigen四元数构造顺序为qxyw
        Eigen::Quaterniond Qj(
            parameters[6][3], parameters[6][0], parameters[6][1], parameters[6][2]);

        // j时刻速度
        Eigen::Vector3d Vj(parameters[7][0], parameters[7][1], parameters[7][2]);
        // j时刻加速度计偏置
        Eigen::Vector3d Baj(parameters[8][0], parameters[8][1], parameters[8][2]);
        // j时刻陀螺仪偏置
        Eigen::Vector3d Bgj(parameters[9][0], parameters[9][1], parameters[9][2]);


        Eigen::Map<Eigen::Matrix<double, 15, 1>> residual(residuals);
        // 计算原始残差（不含信息矩阵）
        residual = preint_->Evaluate(Pi, Qi, Vi, Bai, Bgi, Pj, Qj, Vj, Baj, Bgj);
        // 因为ceres没有g2o设置信息矩阵的接口
        // 因此置信度直接乘在残差上，这里通过LLT分解，相当于将信息矩阵开根号
        Eigen::Matrix<double, 15, 15> sqrt_info =
            Eigen::LLT<Eigen::Matrix<double, 15, 15>>(
                preint_->covariance_.inverse()).matrixL().transpose();
        // 这就是带有信息矩阵的残差
        residual = sqrt_info * residual;

        if (!jacobians) return true;

        // 这段预积分的总时间
        double sum_dt = preint_->sum_dt_;
        // 提取预积分相对于偏置的雅可比矩阵，用于计算预积分的更新值
        // 位移预积分相对于加速度计偏置的雅可比
        Eigen::Matrix3d dp_dba =
            preint_->jacobian_.template block<3, 3>(
                param_ptr_->POSI_INDEX, param_ptr_->ACC_BIAS_INDEX_STATE_);
        // 位移预积分相对于陀螺仪偏置的雅可比
        Eigen::Matrix3d dp_dbg =
            preint_->jacobian_.template block<3, 3>(
                param_ptr_->POSI_INDEX, param_ptr_->GYRO_BIAS_INDEX_STATE_);

        // 旋转预积分相对于陀螺仪偏置的雅可比
        Eigen::Matrix3d dq_dbg =
            preint_->jacobian_.template block<3, 3>(
                param_ptr_->ORI_INDEX_STATE_, param_ptr_->GYRO_BIAS_INDEX_STATE_);

        // 速度预积分相对于加速度计偏置的雅可比
        Eigen::Matrix3d dv_dba =
            preint_->jacobian_.template block<3, 3>(
                param_ptr_->VEL_INDEX_STATE_, param_ptr_->ACC_BIAS_INDEX_STATE_);
        // 速度预积分相对于陀螺仪偏置的雅可比
        Eigen::Matrix3d dv_dbg =
            preint_->jacobian_.template block<3, 3>(
                param_ptr_->VEL_INDEX_STATE_, param_ptr_->GYRO_BIAS_INDEX_STATE_);

        // 数值稳定性检查
        if (
            preint_->jacobian_.maxCoeff() > 1e8 ||
            preint_->jacobian_.minCoeff() < -1e8)
        {
            LOG(ERROR) << "numerical unstable in preintegration";
        }
        // 式6.135c 更新后的旋转预积分 
        Eigen::Quaterniond corrected_delta_q =
            preint_->delta_q_ * Converter::RotVecToQuaternion(
                dq_dbg * (Bgi - preint_->bg_));
        // 下面开始求预积分相对于各个参数的雅克比
        // 9个参数块，分别是 Pi Qi Vi Bai Bgi Pj Qj Vj Baj Bgj
        // 每个参数块的维度分别是 3 4 3 3 3 3 4 3 3 3
        // 0: 残差关于位移Pi的雅可比矩阵
        if (jacobians[0])
        {
            Eigen::Map<Eigen::Matrix<double, 15, 3, Eigen::RowMajor>> J(jacobians[0]);
            J.setZero();
            // 式6.174 位置残差关于Pi的雅可比
            J.block<3,3>(param_ptr_->POSI_INDEX, 0) = -Qi.inverse().toRotationMatrix();
            J = sqrt_info * J;
        }
        // 1: 残差关于旋转Qi的雅可比矩阵
        if (jacobians[1])
        {
            Eigen::Map<Eigen::Matrix<double, 15, 4, Eigen::RowMajor>> J(jacobians[1]);
            J.setZero();
            // 式6.174 位置残差关于Qi的雅可比
            J.block<3,3>(param_ptr_->POSI_INDEX, 0) =
                Converter::Skew(Qi.inverse() *
                    (-0.5 * param_ptr_->gw_ * sum_dt * sum_dt + Pj - Pi - Vi * sum_dt));
            // 式6.174 姿态残差关于Qi的雅可比
            J.block<3,3>(param_ptr_->ORI_INDEX_STATE_, 0) =
                -(Converter::Qleft(Qj.inverse() * Qi) *
                    Converter::Qright(corrected_delta_q)).bottomRightCorner<3,3>();
            // 式6.174 速度残差关于Qi的雅可比
            J.block<3,3>(param_ptr_->VEL_INDEX_STATE_, 0) =
                Converter::Skew(Qi.inverse() * (-param_ptr_->gw_ * sum_dt + Vj - Vi));
            J = sqrt_info * J;
        }
        // 2: 残差关于速度Vi的雅可比矩阵
        if (jacobians[2])
        {
            Eigen::Map<Eigen::Matrix<double, 15, 3, Eigen::RowMajor>> J(jacobians[2]);
            J.setZero();
            // 式6.180 位置残差关于Vi的雅可比 
            J.block<3,3>(param_ptr_->POSI_INDEX, 0) =
                -Qi.inverse().toRotationMatrix() * sum_dt;
            // 式6.180 速度残差关于Vi的雅可比
            J.block<3,3>(param_ptr_->VEL_INDEX_STATE_, 0) =
                -Qi.inverse().toRotationMatrix();
            J = sqrt_info * J;
        }
        // 3: 残差关于加速度计偏置Bai的雅可比矩阵
        if (jacobians[3])
        {
            Eigen::Map<Eigen::Matrix<double, 15, 3, Eigen::RowMajor>> J(jacobians[3]);
            J.setZero();
            // 式6.180 位置残差关于Bai的雅可比
            J.block<3,3>(param_ptr_->POSI_INDEX, 0) = -dp_dba;
            // 式6.180 速度残差关于Bai的雅可比
            J.block<3,3>(param_ptr_->VEL_INDEX_STATE_, 0) = -dv_dba;
            // 式6.180 偏置残差关于Bai的雅可比
            J.block<3,3>(param_ptr_->ACC_BIAS_INDEX_STATE_, 0) =
                -Eigen::Matrix3d::Identity();
            J = sqrt_info * J;
        }
        // 4: 残差关于陀螺仪偏置Bgi的雅可比矩阵
        if (jacobians[4])
        {
            Eigen::Map<Eigen::Matrix<double, 15, 3, Eigen::RowMajor>> J(jacobians[4]);
            J.setZero();
            // 式6.180 位置残差关于Bgi的雅可比
            J.block<3,3>(param_ptr_->POSI_INDEX, 0) = -dp_dbg;
            // 式6.180 姿态残差关于Bgi的雅可比
            J.block<3,3>(param_ptr_->ORI_INDEX_STATE_, 0) =
                -Converter::Qleft(Qj.inverse() * Qi *
                    corrected_delta_q).bottomRightCorner<3,3>() * dq_dbg;
            // 式6.180 速度残差关于Bgi的雅可比
            J.block<3,3>(param_ptr_->VEL_INDEX_STATE_, 0) = -dv_dbg;
            // 式6.180 偏置残差关于Bgi的雅可比
            J.block<3,3>(param_ptr_->GYRO_BIAS_INDEX_STATE_, 0) =
                -Eigen::Matrix3d::Identity();
            J = sqrt_info * J;
        }
        // 5: 残差关于位移Pj的雅可比矩阵
        if (jacobians[5])
        {
            Eigen::Map<Eigen::Matrix<double, 15, 3, Eigen::RowMajor>> J(jacobians[5]);
            J.setZero();
            // 式6.181 位置残差关于Pj的雅可比
            J.block<3,3>(param_ptr_->POSI_INDEX, 0) =
                Qi.inverse().toRotationMatrix();
            J = sqrt_info * J;
        }
        // 6: 残差关于Qj的雅可比矩阵
        if (jacobians[6])
        {
            Eigen::Map<Eigen::Matrix<double, 15, 4, Eigen::RowMajor>> J(jacobians[6]);
            J.setZero();
            // 式6.181 姿态残差关于Qj的雅可比
            J.block<3,3>(param_ptr_->ORI_INDEX_STATE_, 0) =
                Converter::Qleft(corrected_delta_q.inverse() *
                    Qi.inverse() * Qj).bottomRightCorner<3,3>();
            J = sqrt_info * J;
        }
        // 7: 残差关于速度Vj的雅可比矩阵
        if (jacobians[7])
        {
            Eigen::Map<Eigen::Matrix<double, 15, 3, Eigen::RowMajor>> J(jacobians[7]);
            J.setZero();
            // 式6.182 速度残差关于Vj的雅可比
            J.block<3,3>(param_ptr_->VEL_INDEX_STATE_, 0) =
                Qi.inverse().toRotationMatrix();
            J = sqrt_info * J;
        }
        // 8: 残差关于加速度计偏置Baj的雅可比矩阵
        if (jacobians[8])
        {
            Eigen::Map<Eigen::Matrix<double, 15, 3, Eigen::RowMajor>> J(jacobians[8]);
            J.setZero();
            // 式6.182 偏置残差关于Baj的雅可比
            J.block<3,3>(param_ptr_->ACC_BIAS_INDEX_STATE_, 0) =
                Eigen::Matrix3d::Identity();
            J = sqrt_info * J;
        }
        // 9: 残差关于陀螺仪偏置Bgj的雅可比矩阵
        if (jacobians[9])
        {
            Eigen::Map<Eigen::Matrix<double, 15, 3, Eigen::RowMajor>> J(jacobians[9]);
            J.setZero();
            // 式6.182 偏置残差关于Bgj的雅可比
            J.block<3,3>(param_ptr_->GYRO_BIAS_INDEX_STATE_, 0) =
                Eigen::Matrix3d::Identity();
            J = sqrt_info * J;
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
