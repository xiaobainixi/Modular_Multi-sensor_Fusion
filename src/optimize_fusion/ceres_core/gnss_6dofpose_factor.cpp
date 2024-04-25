#include "gnss_6dofpose_factor.h"

bool Gnss6DofPoseFactor::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
{
    Eigen::Map<const Eigen::Matrix<double, 7, 1>> pose(parameters[0]);
    Eigen::Map<Eigen::Matrix<double, 6, 1>> res(residuals);
    res.head<3>() = pose.head<3>() - t_obs_;
    const Eigen::Quaterniond curr_q(pose.tail<4>());
    res.tail<3>() = 2.0 * (curr_q*q_obs_.inverse()).vec();
    res = sqrt_info_matrix_ * res;
    if (jacobians && jacobians[0])
    {
        Eigen::Map<Eigen::Matrix<double, 6, 7, Eigen::RowMajor>> J(jacobians[0]);
        J.setZero();
        J.topLeftCorner<3, 3>().setIdentity();

        Eigen::Quaterniond anchor_q_inv = q_obs_.inverse();
        Eigen::Matrix3d J_q;
        J_q << anchor_q_inv.w(),  anchor_q_inv.z(), -anchor_q_inv.y(),
              -anchor_q_inv.z(),  anchor_q_inv.w(),  anchor_q_inv.x(),
               anchor_q_inv.y(), -anchor_q_inv.x(),  anchor_q_inv.w();
        J.block<3, 3>(3, 3) = J_q;
        J = sqrt_info_matrix_ * J;
    }

    return true;
}

bool GnssPositionFactor::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const {
    Eigen::Map<const Eigen::Matrix<double,3,1>> pos(parameters[0]);
    Eigen::Map<Eigen::Matrix<double,3,1>> res(residuals);
    res = pos - p_bos_;
    res = sqrt_info_matrix_ * res;
    // 计算雅可比矩阵
    if (jacobians && jacobians[0]) {
        Eigen::Map<Eigen::Matrix<double, 3, 7, Eigen::RowMajor>> J(jacobians[0]);
        J.setZero();
        J.topLeftCorner<3, 3>().setIdentity();
        J = sqrt_info_matrix_ * J;
    }
    return true;
}