#include "initializers/InitialAlignment.h"

/**
 * @brief 求解陀螺仪零偏，同时利用求出来的零偏重新进行预积分
 * @param[in] param_ptr 系统相关参数
 * @param[in] all_image_frame 滑动窗口内所有图像帧
 * @param[out] Bgs 求得的陀螺仪偏置
 */
void solveGyroscopeBias(
    const std::shared_ptr<Parameter> &param_ptr,
    std::map<double, ImageFrame> &all_image_frame, Eigen::Vector3d *Bgs)
{
    // AB矩阵
    Eigen::Matrix3d A;
    Eigen::Vector3d b;
    // 待求的陀螺仪偏置改变量
    Eigen::Vector3d delta_bg;
    A.setZero();
    b.setZero();
    std::map<double, ImageFrame>::iterator frame_i;
    std::map<double, ImageFrame>::iterator frame_j;
    // 计算窗口内所有相邻帧之间的约束，式6.160，默认这个时间段的陀螺仪偏置是一个值
    for (
        frame_i = all_image_frame.begin();
        next(frame_i) != all_image_frame.end(); frame_i++)
    {
        frame_j = next(frame_i);
        Eigen::MatrixXd tmp_A(3, 3);
        tmp_A.setZero();
        Eigen::VectorXd tmp_b(3);
        tmp_b.setZero();
        // 由图像计算出的相邻帧的相对旋转
        Eigen::Quaterniond q_ij(
            frame_i->second.R.transpose() * frame_j->second.R);
        // 式6.160
        tmp_A = frame_j->second.pre_integration->jacobian_.template block<3, 3>(
            param_ptr->ORI_INDEX_STATE_, param_ptr->GYRO_BIAS_INDEX_STATE_);
        tmp_b =
            2 * (frame_j->second.pre_integration->delta_q_.inverse() * q_ij).vec();
        A += tmp_A.transpose() * tmp_A;
        b += tmp_A.transpose() * tmp_b;
    }
    // 求陀螺仪偏置改变量
    delta_bg = A.ldlt().solve(b);
    LOG(INFO) << "gyroscope bias initial calibration " << delta_bg.transpose();
    // 滑窗中的零偏设置为求解出来的零偏
    for (int i = 0; i <= param_ptr->WINDOW_SIZE; i++)
        Bgs[i] += delta_bg;
    // 对all_image_frame中预积分量根据当前零偏重新积分
    for (
        frame_i = all_image_frame.begin();
        next(frame_i) != all_image_frame.end(); frame_i++)
    {
        frame_j = next(frame_i);
        frame_j->second.pre_integration->Repropagate(
            Eigen::Vector3d::Zero(), Bgs[0]);
    }
}


/**
 * @brief 构造重力方向的切空间基底
 * @param[in] g0 当前重力向量
 * @return 返回一个3x2的矩阵，列向量为切空间的两个正交基
 */
Eigen::MatrixXd TangentBasis(Eigen::Vector3d &g0)
{
    Eigen::Vector3d b, c;
    // 单位化重力方向
    Eigen::Vector3d a = g0.normalized(); 
    // 默认z轴
    Eigen::Vector3d tmp(0, 0, 1);
    // 如果重力方向刚好是z轴，则用x轴
    if (a == tmp)
        tmp << 1, 0, 0; 
    // b为tmp在a方向上的投影的正交分量
    b = (tmp - a * (a.transpose() * tmp)).normalized();
    // c为a和b的叉乘，保证正交
    c = a.cross(b);
    Eigen::MatrixXd bc(3, 2);
    bc.block<3, 1>(0, 0) = b;
    bc.block<3, 1>(0, 1) = c;
    return bc;
}

/**
 * @brief 得到了一个初始的重力向量，引入重力大小作为先验，再进行几次迭代优化，求解最终的变量
 * @param[in] param_ptr 系统相关参数
 * @param[in] all_image_frame
 * @param[out] g 枢纽帧的重力向量
 * @param[out] x 待求值组成的向量
 */
void RefineGravity(
    const std::shared_ptr<Parameter> &param_ptr,
    std::map<double, ImageFrame> &all_image_frame,
    Eigen::Vector3d &g, Eigen::VectorXd &x)
{
    // 式6.167 重力模长乘重力方向的部分
    Eigen::Vector3d g0 = g.normalized() * param_ptr->gw_.norm();
    // Eigen::Vector3d lx, ly;
    // Eigen::VectorXd x;
    // 式6.170b 待求量的维度计算
    int all_frame_count = all_image_frame.size();
    // 注意此处重力向量的更新量维度为2维
    int n_state = all_frame_count * 3 + 2 + 1;

    // 1. 准备AB矩阵
    Eigen::MatrixXd A{n_state, n_state};
    A.setZero();
    Eigen::VectorXd b{n_state};
    b.setZero();

    std::map<double, ImageFrame>::iterator frame_i;
    std::map<double, ImageFrame>::iterator frame_j;
    // 2. 通过迭代的方式求得更加精确的结果
    for (int k = 0; k < 4; k++)
    {
        // 式6.167 中的b向量
        Eigen::MatrixXd lxly(3, 2);
        // 2.1 构造重力方向的切空间基底
        lxly = TangentBasis(g0);
        int i = 0;
        // 2.2 填充AB矩阵
        for (
            frame_i = all_image_frame.begin();
            next(frame_i) != all_image_frame.end(); frame_i++, i++)
        {
            frame_j = next(frame_i);

            Eigen::MatrixXd tmp_A(6, 9);
            tmp_A.setZero();
            Eigen::VectorXd tmp_b(6);
            tmp_b.setZero();

            double dt = frame_j->second.pre_integration->sum_dt_;

            // 式6.170a 第一行
            // 速度约束部分
            tmp_A.block<3, 3>(0, 0) = -dt * Eigen::Matrix3d::Identity();
            // 重力方向扰动部分
            tmp_A.block<3, 2>(0, 6) =
                frame_i->second.R.transpose() * dt * dt / 2 *
                Eigen::Matrix3d::Identity() * lxly;
            // 尺度部分，同样 /100
            tmp_A.block<3, 1>(0, 8) =
                frame_i->second.R.transpose() *
                (frame_j->second.T - frame_i->second.T) / 100.0;

            // 式6.169 第一行
            // 位置观测残差
            tmp_b.block<3, 1>(0, 0) =
                frame_j->second.pre_integration->delta_p_ +
                frame_i->second.R.transpose() * frame_j->second.R * param_ptr->tbc_ -
                param_ptr->tbc_ -
                frame_i->second.R.transpose() * dt * dt / 2 * g0;

            // 式6.170a 第二行
            // 速度约束部分
            tmp_A.block<3, 3>(3, 0) = -Eigen::Matrix3d::Identity();
            tmp_A.block<3, 3>(3, 3) =
                frame_i->second.R.transpose() * frame_j->second.R;
            // 重力方向扰动部分
            tmp_A.block<3, 2>(3, 6) =
                frame_i->second.R.transpose() * dt * Eigen::Matrix3d::Identity() * lxly;
            // 式6.169 第二行
            // 速度观测残差
            tmp_b.block<3, 1>(3, 0) =
                frame_j->second.pre_integration->delta_v_ -
                frame_i->second.R.transpose() * dt * Eigen::Matrix3d::Identity() * g0;


            Eigen::Matrix<double, 6, 6> cov_inv = Eigen::Matrix<double, 6, 6>::Zero();
            cov_inv.setIdentity();

            // 构造J^T*J和J^T*b
            Eigen::MatrixXd r_A = tmp_A.transpose() * cov_inv * tmp_A;
            Eigen::VectorXd r_b = tmp_A.transpose() * cov_inv * tmp_b;

            // 填充到大矩阵A和b
            // 整个大矩阵包含了所有滑窗的速度
            // 因此这里根据本次for循环到的帧的索引将关于速度的部分放到对应位置
            A.block<6, 6>(i * 3, i * 3) += r_A.topLeftCorner<6, 6>();
            b.segment<6>(i * 3) += r_b.head<6>();

            // 重力修正+尺度
            A.bottomRightCorner<3, 3>() += r_A.bottomRightCorner<3, 3>();
            b.tail<3>() += r_b.tail<3>();

            // 当前ij相邻两帧速度相对于 重力修正+尺度
            A.block<6, 3>(i * 3, n_state - 3) += r_A.topRightCorner<6, 3>();
            A.block<3, 6>(n_state - 3, i * 3) += r_A.bottomLeftCorner<3, 6>();
        }
        // 数值放大，提升稳定性
        A = A * 1000.0;
        b = b * 1000.0;

        // 2.3 求解
        x = A.ldlt().solve(b);
        // 取出重力向量的更新量，更新到g0，供下次迭代时计算切向量
        Eigen::VectorXd dg = x.segment<2>(n_state - 3);
        // 更新重力方向
        g0 = (g0 + lxly * dg).normalized() * param_ptr->gw_.norm();
        // double s = x(n_state - 1);
    }
    g = g0;
}

/**
 * @brief 求解各帧的速度，枢纽帧的重力向量，及尺度
 *
 * @param[in] all_image_frame
 * @param[out] g 枢纽帧的重力向量
 * @param[out] x 待求值组成的向量
 * @return true
 * @return false
 */
bool LinearAlignment(
    const std::shared_ptr<Parameter> &param_ptr,
    std::map<double, ImageFrame> &all_image_frame,
    Eigen::Vector3d &g, Eigen::VectorXd &x)
{
    // 1. 构建AB矩阵
    // 式6.161 待求量的维度计算
    int all_frame_count = all_image_frame.size();
    int n_state = all_frame_count * 3 + 3 + 1;

    Eigen::MatrixXd A{n_state, n_state};
    A.setZero();
    Eigen::VectorXd b{n_state};
    b.setZero();

    // 2. 向AB矩阵填数
    std::map<double, ImageFrame>::iterator frame_i;
    std::map<double, ImageFrame>::iterator frame_j;
    int i = 0;
    // 遍历所有初始化窗口中的图像帧
    for (
        frame_i = all_image_frame.begin();
        next(frame_i) != all_image_frame.end(); frame_i++, i++)
    {
        frame_j = next(frame_i);

        // 式6.165 的维度
        // 10 维分别表示  V_k V_k+1 g s
        Eigen::MatrixXd tmp_A(6, 10);
        tmp_A.setZero();
        Eigen::VectorXd tmp_b(6);
        tmp_b.setZero();

        double dt = frame_j->second.pre_integration->sum_dt_;

        // 式6.165 第一行
        tmp_A.block<3, 3>(0, 0) = -dt * Eigen::Matrix3d::Identity();
        tmp_A.block<3, 3>(0, 6) =
            frame_i->second.R.transpose() * dt * dt / 2 * Eigen::Matrix3d::Identity();
        // 为了数值稳定，这里主动 / 100 
        tmp_A.block<3, 1>(0, 9) =
            frame_i->second.R.transpose() *
            (frame_j->second.T - frame_i->second.T) / 100.0;
        // 式6.164 第一行
        tmp_b.block<3, 1>(0, 0) =
            frame_j->second.pre_integration->delta_p_ +
            frame_i->second.R.transpose() * frame_j->second.R * param_ptr->tbc_ -
            param_ptr->tbc_;
        
        // 式6.165 第二行
        tmp_A.block<3, 3>(3, 0) = -Eigen::Matrix3d::Identity();
        tmp_A.block<3, 3>(3, 3) =
            frame_i->second.R.transpose() * frame_j->second.R;
        tmp_A.block<3, 3>(3, 6) =
            frame_i->second.R.transpose() * dt * Eigen::Matrix3d::Identity();
        // 式6.164 第二行
        tmp_b.block<3, 1>(3, 0) = frame_j->second.pre_integration->delta_v_;

        // 同样组成 J^T*J的形式
        Eigen::Matrix<double, 6, 6> cov_inv = Eigen::Matrix<double, 6, 6>::Zero();
        // cov.block<6, 6>(0, 0) = IMU_cov[i + 1];
        // Eigen::MatrixXd cov_inv = cov.inverse();
        cov_inv.setIdentity();

        Eigen::MatrixXd r_A = tmp_A.transpose() * cov_inv * tmp_A;
        Eigen::VectorXd r_b = tmp_A.transpose() * cov_inv * tmp_b;

        // 整个大矩阵包含了所有滑窗的速度
        // 因此这里根据本次for循环到的帧的索引将关于速度的部分放到对应位置
        A.block<6, 6>(i * 3, i * 3) += r_A.topLeftCorner<6, 6>();
        b.segment<6>(i * 3) += r_b.head<6>();

        // 重力向量跟尺度
        A.bottomRightCorner<4, 4>() += r_A.bottomRightCorner<4, 4>();
        b.tail<4>() += r_b.tail<4>();

        // 当前ij相邻两帧速度相对于 重力向量跟尺度
        A.block<6, 4>(i * 3, n_state - 4) += r_A.topRightCorner<6, 4>();
        A.block<4, 6>(n_state - 4, i * 3) += r_A.bottomLeftCorner<4, 6>();
    }
    // 3. 求解
    // 增强数值稳定性，都乘1000.0 情况下求得的结果不变
    A = A * 1000.0;
    b = b * 1000.0;
    x = A.ldlt().solve(b);
    // 4. 取尺度值，由于A矩阵中对应位置 /100 因此结果会是待求尺度的100倍。所以这里除掉了
    double s = x(n_state - 1) / 100.0;
    LOG(INFO) << "estimated scale: " << s;
    // 5. 取出枢纽帧重力向量
    g = x.segment<3>(n_state - 4);
    LOG(INFO) << " result g norm=" << g.norm() << " g=" << g.transpose();
    // 检查求得的结果是否符合常理，即重力加速度跟9.8是否差别过大，以及尺度不可能出现负值
    if (fabs(g.norm() - param_ptr->gw_.norm()) > 2.0 || s < 0)
    {
        return false;
    }
    // 6. 重力修复，进一步优化重力跟尺度
    RefineGravity(param_ptr, all_image_frame, g, x);
    // 得到最终真实尺度
    s = (x.tail<1>())(0) / 100.0;
    (x.tail<1>())(0) = s;
    LOG(INFO) << " refine g norm=" << g.norm() << " g=" << g.transpose();
    if (s < 0.0)
        return false;
    else
        return true;
}

/**
 * @brief
 *
 * @param[in] all_image_frame 每帧的位姿和对应的预积分量
 * @param[out] Bgs 陀螺仪零偏
 * @param[out] g 重力向量
 * @param[out] x 其他状态量
 * @return true
 * @return false
 */

bool VisualIMUAlignment(
    const std::shared_ptr<Parameter> &param_ptr,
    std::map<double, ImageFrame> &all_image_frame, Eigen::Vector3d *Bgs, Eigen::Vector3d &g, Eigen::VectorXd &x)
{
    // 求解陀螺仪零偏
    solveGyroscopeBias(param_ptr, all_image_frame, Bgs);

    // 求解各帧的速度，枢纽帧的重力向量，及尺度
    if (LinearAlignment(param_ptr, all_image_frame, g, x))
        return true;
    else
        return false;
}
