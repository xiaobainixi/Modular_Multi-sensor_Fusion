#pragma once

#include "Preintegration.h"

class WheelPreintegration : public Preintegration {
public:
    WheelPreintegration() = delete;
    WheelPreintegration(const WheelData &wheel_data, const State &state, const std::shared_ptr<Parameter> &param_ptr)
        : wheel_data_0_{wheel_data}, first_wheel_data_{wheel_data}
    {
        param_ptr_ = param_ptr;
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
        covariance_.setZero();
        for (int i = 0; i < static_cast<int>(data_buf_.size()); i++)
            Propagate(data_buf_[i]);
    }

    // 轮速预积分推进
    // 模型假设：平面差速驱动，只积分 x 前进与 z 轴旋转；误差状态采用右乘扰动，状态顺序 [p(3), so3(3)]
    // 协方差离散化：X_k+1 = F X_k + G w，使用一阶线性化，过程噪声映射为 V (此处命名保持与其它预积分一致)
    void Propagate(const WheelData &wheel_data)
    {
        //--------------------------------------------------------------------------------
        // 1. 准备数据
        // 计算上一个数据与当前数据的时间间隔
        // 缓存当前原始轮速数据
        // wheel_data wheel_data_1_ 表示当前数据
        // wheel_data_0_ 表示上一个数据
        dt_ = wheel_data.time_ - wheel_data_0_.time_;
        wheel_data_1_ = wheel_data;

        // 差速模型：左右轮线速度平均得到前向速度（梯形积分：两帧求平均）
        double v_avg =
            0.25 * (
                wheel_data_0_.lv_ + wheel_data_0_.rv_ +
                wheel_data_1_.lv_ + wheel_data_1_.rv_);
        // 左右轮速度差平均（两帧）得到角速度（尚未除以轴距）
        double v_diff =
            0.5 * (
                (wheel_data_1_.rv_ - wheel_data_1_.lv_) +
                (wheel_data_0_.rv_ - wheel_data_0_.lv_));
        // 轮距（轴长）
        double wheel_base = param_ptr_->wheel_b_;  // 轴长

        // 本段航向角增量（绕 z 轴）
        double delta_theta = v_diff * dt_ / wheel_base;
        Eigen::AngleAxisd dR(delta_theta, Eigen::Vector3d::UnitZ());

        // 计算速度
        Eigen::Vector3d v_local(v_avg, 0, 0); // 车体坐标系前进

        // 增量旋转（SO3） △R_(j-1 j)
        Eigen::Matrix3d dR33 = dR.toRotationMatrix();

        //--------------------------------------------------------------------------------
        // 2. 计算协方差矩阵
        // 定义： 理想数值（优质数值） = 估计数值 + 误差
        // param_ptr_->STATE_DIM 表示状态维度，当前为15
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

        // 式5.36 当前时刻旋转预积分误差与上时刻旋转预积分误差的关系 △R_(j j-1)
        F.block<3, 3>(param_ptr_->ORI_INDEX_STATE_, param_ptr_->ORI_INDEX_STATE_) =
            dR33.transpose();

        // 这里的6表示数据误差维度
        Eigen::MatrixXd V = Eigen::MatrixXd::Zero(param_ptr_->STATE_DIM, 6);
        // 式5.37 当前时刻位移预积分误差与线速度误差的关系
        V.block<3, 3>(param_ptr_->POSI_INDEX, 0) =
            delta_q_.toRotationMatrix() * dt_;
        // 式5.36 当前时刻旋转预积分误差与角速度误差的关系
        V.block<3, 3>(param_ptr_->ORI_INDEX_STATE_, 3) =
            Converter::RightJacobianSO3(0.0, 0.0, delta_theta) * dt_;

        // 更新协方差矩阵
        covariance_ = F * covariance_ * F.transpose() + V * noise_ * V.transpose();

        //--------------------------------------------------------------------------------
        // 预积分总体时间间隔
        sum_dt_ += dt_;
        // 更新上一个数据，用于下一次迭代
        wheel_data_0_ = wheel_data_1_;
        // 3. 更新预积分 注意先后顺序
        // 式5.34b 更新位移预积分
        delta_p_ = delta_q_ * v_local * dt_ + delta_p_;
        // 式5.34a 更新旋转预积分
        delta_q_ = delta_q_ * Eigen::Quaterniond(dR);
    }

    // template <typename T>
    // bool Evaluate(
    //     const T *const Pi, const T *const Qi,
    //     const T *const Pj, const T *const Qj, T *residuals) const
    // {
    //     Eigen::Map<const Eigen::Matrix<T, 3, 1>> Pi_eig(Pi);
    //     // 输入Qi顺序为xyzw，Eigen四元数构造顺序为qxyw
    //     Eigen::Quaternion<T> Qi_eig(Qi[3], Qi[0], Qi[1], Qi[2]);
    //     Eigen::Map<const Eigen::Matrix<T, 3, 1>> Pj_eig(Pj);
    //     Eigen::Quaternion<T> Qj_eig(Qj[3], Qj[0], Qj[1], Qj[2]);

    //     Eigen::Map<Eigen::Matrix<T, 6, 1>> residuals_eig(residuals);

    //     // 位置残差
    //     residuals_eig.template segment<3>(0) = Qi_eig.inverse() * (Pj_eig - Pi_eig) - delta_p_.template cast<T>();
    //     // 姿态残差
    //     residuals_eig.template segment<3>(3) = T(2.0) * (delta_q_.cast<T>().inverse() * (Qi_eig.inverse() * Qj_eig)).vec();

    //     return true;
    // }

    /**
     * @brief 计算预积分残差
     * 
     * @param Pi i时刻位移
     * @param Qi i时刻旋转
     * @param Pj j时刻位移
     * @param Qj j时刻旋转
     * @return Eigen::Matrix<double, 6, 1> 残差
     */
    Eigen::Matrix<double, 6, 1> Evaluate(
        const Eigen::Vector3d &Pi, const Eigen::Quaterniond &Qi,
        const Eigen::Vector3d &Pj, const Eigen::Quaterniond &Qj)
    {
        Eigen::Matrix<double, 6, 1> residuals;
        // 式5.40a 旋转残差
        residuals.block<3, 1>(param_ptr_->ORI_INDEX_STATE_, 0) =
            2.0 * (delta_q_.inverse() * (Qi.inverse() * Qj)).vec();
        // 式5.40b 位移残差
        residuals.block<3, 1>(param_ptr_->POSI_INDEX, 0) =
            Qi.inverse() * (Pj - Pi) - delta_p_;
        return residuals;
    }

    WheelData wheel_data_0_, wheel_data_1_;
    const WheelData first_wheel_data_;

    std::vector<WheelData> data_buf_;
};

// struct WheelPreintegrationResidual
// {
//     WheelPreintegrationResidual(std::shared_ptr<WheelPreintegration> preint)
//         : preint_(preint) {}

//     template <typename T>
//     bool operator()(const T *const Pi, const T *const Qi,
//                     const T *const Pj, const T *const Qj,
//                     T *residuals) const
//     {
//         preint_->Evaluate(
//             Pi, Qi, Pj, Qj, residuals);
//         Eigen::Matrix<double, 6, 6> sqrt_info =
//             Eigen::LLT<Eigen::Matrix<double, 6, 6>>(
//                 preint_->covariance_.inverse()).matrixL().transpose();
//         Eigen::Matrix<T, 6, 6> sqrt_info_T = sqrt_info.template cast<T>();
//         Eigen::Map<Eigen::Matrix<T, 6, 1>> residuals_eig(residuals);
//         residuals_eig = sqrt_info_T * residuals_eig;
//         return true;
//     }

//     std::shared_ptr<WheelPreintegration> preint_;
// };

class WheelPreintegrationResidual : public ceres::SizedCostFunction<6, 3, 4, 3, 4>
{
public:
    WheelPreintegrationResidual() = delete;
    WheelPreintegrationResidual(
        std::shared_ptr<WheelPreintegration> preint, const std::shared_ptr<Parameter>& param_ptr)
        : preint_(preint), param_ptr_(param_ptr) {}

    virtual bool Evaluate(
        double const *const *parameters, double *residuals, double **jacobians) const
    {
        // 读取Ceres参数块
        // i时刻位移
        Eigen::Vector3d Pi(
            parameters[0][0], parameters[0][1], parameters[0][2]);
        // i时刻旋转 注意输入顺序为xyzw，Eigen四元数构造顺序为qxyw
        Eigen::Quaterniond Qi(
            parameters[1][3], parameters[1][0], parameters[1][1], parameters[1][2]);

        // j时刻位移
        Eigen::Vector3d Pj(
            parameters[2][0], parameters[2][1], parameters[2][2]);
        // j时刻旋转 注意输入顺序为xyzw，Eigen四元数构造顺序为qxyw
        Eigen::Quaterniond Qj(
            parameters[3][3], parameters[3][0], parameters[3][1], parameters[3][2]);

        // 计算原始残差（不含信息矩阵）
        Eigen::Map<Eigen::Matrix<double, 6, 1>> res_info(residuals);
        // 计算预积分残差，返回残差向量
        auto res = preint_->Evaluate(Pi, Qi, Pj, Qj);

        // 信息矩阵平方根
        Eigen::Matrix<double, 6, 6> sqrt_info =
            Eigen::LLT<Eigen::Matrix<double, 6, 6>>(
                preint_->covariance_.inverse()).matrixL().transpose();
        // 由于Ceres中没有像G2O那样明确的定义信息矩阵，因此需要将信息矩阵融于残差中
        res_info = sqrt_info * res;

        if (!jacobians) return true;

        // 会多次使用的变量，旋转矩阵
        Eigen::Matrix3d Riw = Qi.inverse().toRotationMatrix();

        // 0: 残差关于位移Pi的雅可比矩阵
        if (jacobians[0])
        {
            Eigen::Map<Eigen::Matrix<double, 6, 3, Eigen::RowMajor>> J(jacobians[0]);
            J.setZero();

            // 式5.44c 位置残差对位移Pi的雅可比矩阵
            J.block<3,3>(param_ptr_->POSI_INDEX,0) = -Riw;

            // 由于Ceres中没有像G2O那样明确的定义信息矩阵，因此需要将信息矩阵融于雅可比矩阵中
            J = sqrt_info * J;
        }

        // 1: 残差关于旋转Qi的雅可比矩阵
        if (jacobians[1])
        {
            // 正常情况下这里的雅可比矩阵应该是残差相对于旋转四元数的雅可比矩阵
            // 在四元数中也要定义四元数相对于旋转向量的雅可比矩阵
            // 当四元数相对于旋转向量的雅可比矩阵对角线为1，其他位置为0时
            // 此处可直接存放残差相对于旋转向量的雅可比矩阵，因为经过求导连式法则后是一样的
            Eigen::Map<Eigen::Matrix<double, 6, 4, Eigen::RowMajor>> J(jacobians[1]);
            J.setZero();

            // 式5.44e 位置残差对旋转Qi的雅可比矩阵 Skew(R_i^T (Pj - Pi))
            J.block<3,3>(param_ptr_->POSI_INDEX,0) = Converter::Skew(Riw * (Pj - Pi));

            // 式5.44a 姿态残差对旋转Qi的雅可比矩阵
            J.block<3,3>(param_ptr_->ORI_INDEX_STATE_,0) =
                -Converter::InverseRightJacobianSO3(
                    res.block<3, 1>(param_ptr_->ORI_INDEX_STATE_, 0)) *
                (Qj.inverse() * Qi).toRotationMatrix();

            J = sqrt_info * J;
        }

        // 2: 残差关于位移Pj的雅可比矩阵
        if (jacobians[2])
        {
            Eigen::Map<Eigen::Matrix<double, 6, 3, Eigen::RowMajor>> J(jacobians[2]);
            J.setZero();
            // 式5.44d 位置残差对位移Pj的雅可比矩阵
            J.block<3,3>(param_ptr_->POSI_INDEX,0) = Riw;

            J = sqrt_info * J;
        }

        // 3: 残差关于Qj的雅可比矩阵
        if (jacobians[3])
        {
            Eigen::Map<Eigen::Matrix<double, 6, 4, Eigen::RowMajor>> J(jacobians[3]);
            J.setZero();
            // 式5.44b 姿态残差对旋转Qj的雅可比矩阵
            J.block<3,3>(param_ptr_->ORI_INDEX_STATE_,0) =
                Converter::InverseRightJacobianSO3(
                    res.block<3, 1>(param_ptr_->ORI_INDEX_STATE_, 0));

            J = sqrt_info * J;
        }

        return true;
    }

private:
    std::shared_ptr<WheelPreintegration> preint_;
    std::shared_ptr<Parameter> param_ptr_;
};