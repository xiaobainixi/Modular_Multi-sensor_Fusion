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
        // jacobian_ = Eigen::Matrix<double, 9, 9>::Identity();
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
        // jacobian_.setIdentity();
        covariance_.setZero();
        for (int i = 0; i < static_cast<int>(data_buf_.size()); i++)
            Propagate(data_buf_[i]);
    }

    // IMU轮速联合预积分递推
    // 模型假设：平面差速驱动，只积分 x 方向线速度；误差状态采用右乘扰动，状态顺序 [p(3), so3(3) bg(3)]
    // 协方差离散化：X_k+1 = F X_k + G w，使用一阶线性化，过程噪声映射为 V (此处命名保持与其它预积分一致)
    void Propagate(const WheelIMUData &wheel_imu_data)
    {
        //--------------------------------------------------------------------------------
        // 1. 准备数据
        // 计算上一个数据与当前数据的时间间隔
        // 缓存当前原始轮速数据
        // wheel_data wheel_data_1_ 表示当前数据
        // wheel_data_0_ 表示上一个数据
        dt_ = wheel_imu_data.time_ - wheel_imu_data_0_.time_;
        wheel_imu_data_1_ = wheel_imu_data;

        // 差速模型：左右轮线速度平均得到前向速度（梯形积分：两帧求平均）
        double v_avg =
            0.25 * (
                wheel_imu_data_0_.lv_ + wheel_imu_data_0_.rv_ +
                wheel_imu_data_1_.lv_ + wheel_imu_data_1_.rv_);
        // 计算速度
        Eigen::Vector3d v_local(v_avg, 0, 0); // 车体坐标系前进
        // 计算角度改变量
        Eigen::Vector3d un_gyr = 0.5 * (wheel_imu_data.w_ + wheel_imu_data_0_.w_) - bg_;
        Eigen::Quaterniond dR =
            Eigen::Quaterniond(
                1, un_gyr(0) * dt_ / 2,
                un_gyr(1) * dt_ / 2, un_gyr(2) * dt_ / 2);
        Eigen::Matrix3d dR33 = dR.toRotationMatrix();

        //--------------------------------------------------------------------------------
        // 2. 计算协方差矩阵
        // 定义： 理想数值（优质数值） = 估计数值 + 误差
        // param_ptr_->STATE_DIM 表示状态维度，当前为9
        // 计算F矩阵
        Eigen::MatrixXd F =
            Eigen::MatrixXd::Zero(param_ptr_->STATE_DIM, param_ptr_->STATE_DIM);
        // 式5.37 当前时刻位移预积分误差与上时刻位移预积分误差的关系
        F.block<3, 3>(param_ptr_->POSI_INDEX, param_ptr_->POSI_INDEX) =
            Eigen::Matrix3d::Identity();
        // 式5.37 当前时刻位移预积分误差与上时刻旋转预积分误差的关系
        // 注意这里是右乘扰动 此时delta_q_表示上时刻的旋转预积分
        // Converter::Skew(v_local) 是v_local的反对称矩阵
        F.block<3, 3>(param_ptr_->POSI_INDEX, param_ptr_->ORI_INDEX_STATE_) =
            -delta_q_.toRotationMatrix() * Converter::Skew(v_local) * dt_;

        // 式5.47 当前时刻旋转预积分误差与上时刻旋转预积分误差的关系 △R_(j j-1)
        F.block<3, 3>(param_ptr_->ORI_INDEX_STATE_, param_ptr_->ORI_INDEX_STATE_) =
            dR33.transpose();
        // 式5.47 当前时刻陀螺仪偏置误差与上时刻陀螺仪偏置误差的关系
        F.block<3, 3>(
            param_ptr_->GYRO_BIAS_INDEX_STATE_, param_ptr_->GYRO_BIAS_INDEX_STATE_) =
            Eigen::Matrix3d::Identity();

        // 这里的9表示数据误差维度
        Eigen::MatrixXd V = Eigen::MatrixXd::Zero(param_ptr_->STATE_DIM, 9);
        // 式5.37 当前时刻位移预积分误差与线速度误差的关系
        V.block<3, 3>(param_ptr_->POSI_INDEX, 6) =
            delta_q_.toRotationMatrix() * dt_;
        // 式5.47 当前时刻旋转预积分误差与角速度误差的关系
        V.block<3, 3>(param_ptr_->ORI_INDEX_STATE_, 0) =
            Converter::RightJacobianSO3(un_gyr * dt_) * dt_;
        // 式5.47 当前时刻陀螺仪偏置误差与随机游走的关系
        V.block<3, 3>(param_ptr_->GYRO_BIAS_INDEX_STATE_, 3) =
            Eigen::MatrixXd::Identity(3, 3);
        
        // 更新雅克比矩阵，注意更新顺序不要错
        // 式5.48b 位移预积分相对于陀螺仪偏置的雅克比更新
        dp_dbg_ = dp_dbg_ -
            delta_q_.toRotationMatrix() * Converter::Skew(v_local) * dq_dbg_ * dt_;
        // 式5.48a 旋转预积分相对于陀螺仪偏置的雅克比更新
        dq_dbg_ =
            dR33.transpose() * dq_dbg_ - Converter::RightJacobianSO3(un_gyr * dt_) * dt_;
        // 更新协方差矩阵
        covariance_ = F * covariance_ * F.transpose() + V * noise_ * V.transpose();

        //--------------------------------------------------------------------------------
        // 预积分总体时间间隔
        sum_dt_ += dt_;
        // 更新上一个数据，用于下一次迭代
        wheel_imu_data_0_ = wheel_imu_data_1_;

        // 3. 更新预积分 注意先后顺序
        // 式5.46b 更新位移预积分
        delta_p_ = delta_q_ * v_local * dt_ + delta_p_;
        // 式5.46a 更新旋转预积分
        delta_q_ = delta_q_ * dR;
    }

    // template <typename T>
    // bool Evaluate(
    //     const T *const Pi, const T *const Qi, const T *const Bgi,
    //     const T *const Pj, const T *const Qj, const T *const Bgj, T *residuals) const
    // {
    //     Eigen::Map<const Eigen::Matrix<T, 3, 1>> Pi_eig(Pi);
    //     // 输入Qi顺序为xyzw，Eigen四元数构造顺序为wxyz
    //     Eigen::Quaternion<T> Qi_eig(Qi[3], Qi[0], Qi[1], Qi[2]);
    //     Eigen::Map<const Eigen::Matrix<T, 3, 1>> Bgi_eig(Bgi);
    //     Eigen::Map<const Eigen::Matrix<T, 3, 1>> Pj_eig(Pj);
    //     Eigen::Quaternion<T> Qj_eig(Qj[3], Qj[0], Qj[1], Qj[2]);
    //     Eigen::Map<const Eigen::Matrix<T, 3, 1>> Bgj_eig(Bgj);

    //     Eigen::Map<Eigen::Matrix<T, 9, 1>> residuals_eig(residuals);

    //     Eigen::Matrix<T, 3, 3> dp_dbg = jacobian_.block<3, 3>(param_ptr_->POSI_INDEX, param_ptr_->GYRO_BIAS_INDEX_STATE_).template cast<T>();
    //     Eigen::Matrix<T, 3, 3> dq_dbg = jacobian_.block<3, 3>(param_ptr_->ORI_INDEX_STATE_, param_ptr_->GYRO_BIAS_INDEX_STATE_).template cast<T>();
    //     Eigen::Matrix<T, 3, 1> dbg = Bgi_eig - bg_.template cast<T>();

    //     Eigen::Quaternion<T> delta_q_T = delta_q_.cast<T>();
    //     Eigen::Quaternion<T> corrected_delta_q = delta_q_T * Converter::RotVecToQuaternion(dq_dbg * dbg);
    //     Eigen::Matrix<T, 3, 1> corrected_delta_p = delta_p_.template cast<T>() + dp_dbg * dbg;

    //     // 位置残差
    //     residuals_eig.template segment<3>(param_ptr_->POSI_INDEX) = Qi_eig.inverse() * (Pj_eig - Pi_eig) - corrected_delta_p;
    //     // 姿态残差
    //     residuals_eig.template segment<3>(param_ptr_->ORI_INDEX_STATE_) = T(2.0) * (corrected_delta_q.inverse() * (Qi_eig.inverse() * Qj_eig)).vec();
    //     residuals_eig.template segment<3>(param_ptr_->GYRO_BIAS_INDEX_STATE_) = Bgj_eig - Bgi_eig;
    //     return true;
    // }

    /**
     * @brief 计算预积分残差
     * 
     * @param Pi i时刻位移
     * @param Qi i时刻旋转
     * @param Bgi i时刻陀螺仪零偏
     * @param Pj j时刻位移
     * @param Qj j时刻旋转
     * @param Bgj j时刻陀螺仪零偏
     * @return Eigen::Matrix<double, 9, 1> 残差
     */
    Eigen::Matrix<double, 9, 1> Evaluate(
        const Eigen::Vector3d &Pi, const Eigen::Quaterniond &Qi, const Eigen::Vector3d &Bgi,
        const Eigen::Vector3d &Pj, const Eigen::Quaterniond &Qj, const Eigen::Vector3d &Bgj)
    {
        Eigen::Matrix<double, 9, 1> residuals;
        Eigen::Vector3d dbg = Bgi - bg_;

        // 计算更新后的预积分量
        Eigen::Quaterniond corrected_delta_q = delta_q_ * Converter::RotVecToQuaternion(dq_dbg_ * dbg);
        Eigen::Vector3d corrected_delta_p = delta_p_ + dp_dbg_ * dbg;

        // 式5.50c 位移残差
        residuals.block<3, 1>(param_ptr_->POSI_INDEX, 0) =
            Qi.inverse() * (Pj - Pi) - corrected_delta_p;
        // 式5.50b 旋转残差 四元数近似转旋转向量
        residuals.block<3, 1>(param_ptr_->ORI_INDEX_STATE_, 0) =
            2.0 * (corrected_delta_q.inverse() * (Qi.inverse() * Qj)).vec();
        // 式5.50d 偏置残差
        residuals.block<3, 1>(param_ptr_->GYRO_BIAS_INDEX_STATE_, 0) = Bgj - Bgi;
        return residuals;
    }

    // 预积分雅克比块缓存
    Eigen::Matrix3d dp_dbg_ = Eigen::Matrix3d::Zero(), dq_dbg_ = Eigen::Matrix3d::Zero();
    WheelIMUData wheel_imu_data_0_, wheel_imu_data_1_;
    const WheelIMUData first_wheel_imu_data_;

    std::vector<WheelIMUData> data_buf_;
};

// struct WheelIMUPreintegrationResidual
// {
//     WheelIMUPreintegrationResidual(std::shared_ptr<WheelIMUPreintegration> preint)
//         : preint_(preint) {}

//     template <typename T>
//     bool operator()(const T *const Pi, const T *const Qi, const T *const Bgi,
//                     const T *const Pj, const T *const Qj, const T *const Bgj,
//                     T *residuals) const
//     {
//         preint_->Evaluate(
//             Pi, Qi, Bgi, Pj, Qj, Bgj, residuals);
//         Eigen::Matrix<double, 9, 9> sqrt_info =
//             Eigen::LLT<Eigen::Matrix<double, 9, 9>>(
//                 preint_->covariance_.inverse()).matrixL().transpose();
//         Eigen::Matrix<T, 9, 9> sqrt_info_T = sqrt_info.template cast<T>();
//         Eigen::Map<Eigen::Matrix<T, 9, 1>> residuals_eig(residuals);
//         residuals_eig = sqrt_info_T * residuals_eig;
//         return true;
//     }

//     std::shared_ptr<WheelIMUPreintegration> preint_;
// };


class WheelIMUPreintegrationResidual : public ceres::SizedCostFunction<9, 3, 4, 3, 3, 4, 3>
{
public:
    WheelIMUPreintegrationResidual() = delete;
    explicit WheelIMUPreintegrationResidual(
        std::shared_ptr<WheelIMUPreintegration> preint, const std::shared_ptr<Parameter> &param_ptr)
        : preint_(preint), param_ptr_(param_ptr) {}

    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
    {
        // 读取Ceres参数块
        // i时刻位移
        Eigen::Vector3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
        // i时刻旋转 注意输入顺序为xyzw，Eigen四元数构造顺序为qxyw
        Eigen::Quaterniond Qi(parameters[1][3], parameters[1][0], parameters[1][1], parameters[1][2]);
        // i时刻陀螺仪偏置
        Eigen::Vector3d Bgi(parameters[2][0], parameters[2][1], parameters[2][2]);

        // j时刻位移
        Eigen::Vector3d Pj(parameters[3][0], parameters[3][1], parameters[3][2]);
        // j时刻旋转 注意输入顺序为xyzw，Eigen四元数构造顺序为qxyw
        Eigen::Quaterniond Qj(parameters[4][3], parameters[4][0], parameters[4][1], parameters[4][2]);
        // j时刻陀螺仪偏置
        Eigen::Vector3d Bgj(parameters[5][0], parameters[5][1], parameters[5][2]);

        // 计算原始残差（不含信息矩阵）
        Eigen::Map<Eigen::Matrix<double, 9, 1>> res_info(residuals);
        auto res = preint_->Evaluate(Pi, Qi, Bgi, Pj, Qj, Bgj);

        // 信息矩阵平方根
        Eigen::Matrix<double, 9, 9> sqrt_info =
            Eigen::LLT<Eigen::Matrix<double, 9, 9>>(
                preint_->covariance_.inverse()).matrixL().transpose();
        // 由于Ceres中没有像G2O那样明确的定义信息矩阵，因此需要将信息矩阵融于残差中
        res_info = sqrt_info * res;

        if (!jacobians) return true;

        // 计算更新后的预积分
        Eigen::Vector3d dbg = Bgi - preint_->bg_;
        Eigen::Quaterniond corrected_delta_q =
            preint_->delta_q_ * Converter::RotVecToQuaternion(preint_->dq_dbg_ * dbg);
        corrected_delta_q.normalize();

        // 0: 残差关于位移Pi的雅可比矩阵
        if (jacobians[0])
        {
            Eigen::Map<Eigen::Matrix<double, 9, 3, Eigen::RowMajor>> J(jacobians[0]);
            J.setZero();
            // 式5.52d
            J.block<3,3>(param_ptr_->POSI_INDEX,0) = -Qi.inverse().toRotationMatrix();
            J = sqrt_info * J;
        }

        // 1: 残差关于旋转Qi的雅可比矩阵
        if (jacobians[1])
        {
            Eigen::Map<Eigen::Matrix<double, 9, 4, Eigen::RowMajor>> J(jacobians[1]);
            J.setZero();
            // 式5.52f 位置对 Qi
            J.block<3,3>(param_ptr_->POSI_INDEX,0) =
                Converter::Skew(Qi.inverse() * (Pj - Pi));

            // 式5.52a 姿态对 Qi
            J.block<3,3>(param_ptr_->ORI_INDEX_STATE_,0) =
                -Converter::InverseRightJacobianSO3(
                    res.block<3, 1>(param_ptr_->ORI_INDEX_STATE_, 0)) *
                (Qj.inverse() * Qi).toRotationMatrix();
            // 式6.179 这部分在第六章介绍，是一致的，等价于上面
            // J.block<3,3>(param_ptr_->ORI_INDEX_STATE_,0) =
            //     -(Converter::Qleft(Qj.inverse() * Qi) *
            //       Converter::Qright(corrected_delta_q)).bottomRightCorner<3,3>();

            // 这里无速度项
            J = sqrt_info * J;
        }

        // 2: 残差关于陀螺仪偏置Bgi的雅可比矩阵
        if (jacobians[2])
        {
            Eigen::Map<Eigen::Matrix<double, 9, 3, Eigen::RowMajor>> J(jacobians[2]);
            J.setZero();
            // 式5.52g 位置对 Bgi
            J.block<3,3>(param_ptr_->POSI_INDEX,0) = -preint_->dp_dbg_;
            // 式5.52c 姿态对 Bgi
            J.block<3,3>(param_ptr_->ORI_INDEX_STATE_,0) =
                -Converter::InverseRightJacobianSO3(res.block<3, 1>(param_ptr_->ORI_INDEX_STATE_, 0)) *
                Converter::ExpSO3(-res.block<3, 1>(param_ptr_->ORI_INDEX_STATE_, 0)) *
                Converter::RightJacobianSO3(preint_->dq_dbg_ * dbg) * preint_->dq_dbg_;
            // 式6.180 这部分在第六章介绍，是一致的，等价于上面
            // J.block<3,3>(param_ptr_->ORI_INDEX_STATE_,0) =
            //     -Converter::Qleft(Qj.inverse() * Qi * corrected_delta_q).bottomRightCorner<3,3>() *
            //     preint_->dq_dbg_;

            // 式5.52h 零偏残差对 Bgi
            J.block<3,3>(param_ptr_->GYRO_BIAS_INDEX_STATE_,0) =
                -Eigen::Matrix3d::Identity();
            J = sqrt_info * J;
        }

        // 3: 残差关于位移Pj的雅可比矩阵
        if (jacobians[3])
        {
            Eigen::Map<Eigen::Matrix<double, 9, 3, Eigen::RowMajor>> J(jacobians[3]);
            J.setZero();
            // 式5.52e 位置对 Pj
            J.block<3,3>(param_ptr_->POSI_INDEX,0) =
                Qi.inverse().toRotationMatrix();
            J = sqrt_info * J;
        }

        // 4: 残差关于Qj的雅可比矩阵
        if (jacobians[4])
        {
            Eigen::Map<Eigen::Matrix<double, 9, 4, Eigen::RowMajor>> J(jacobians[4]);
            J.setZero();
            // 式5.52b 姿态对 Qj
            J.block<3,3>(param_ptr_->ORI_INDEX_STATE_,0) =
                Converter::InverseRightJacobianSO3(res.block<3, 1>(param_ptr_->ORI_INDEX_STATE_, 0));
            // 式6.181 这部分在第六章介绍，是一致的，等价于上面
            // J.block<3,3>(param_ptr_->ORI_INDEX_STATE_,0) =
            //     (Converter::Qleft(corrected_delta_q.inverse() * Qi.inverse() * Qj))
            //     .bottomRightCorner<3,3>();
            J = sqrt_info * J;
        }

        // 5: 残差关于陀螺仪偏置Bgj的雅可比矩阵
        if (jacobians[5])
        {
            Eigen::Map<Eigen::Matrix<double, 9, 3, Eigen::RowMajor>> J(jacobians[5]);
            J.setZero();
            // 式5.52i 零偏残差对 Bgj
            J.block<3,3>(param_ptr_->GYRO_BIAS_INDEX_STATE_,0) =
                Eigen::Matrix3d::Identity();
            J = sqrt_info * J;
        }

        return true;
    }

private:
    std::shared_ptr<WheelIMUPreintegration> preint_;
    std::shared_ptr<Parameter> param_ptr_;
};