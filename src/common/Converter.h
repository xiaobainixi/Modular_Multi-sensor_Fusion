#pragma once

#include <Eigen/Core>
class Converter
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    static Eigen::Matrix4d InversePose(const Eigen::Matrix4d &T) {
        Eigen::Matrix4d T_inv = Eigen::Matrix4d::Identity();
        T_inv.block<3, 3>(0, 0) = T.block<3, 3>(0, 0).transpose();
        T_inv.block<3, 1>(0, 3) = -T_inv.block<3, 3>(0, 0) * T.block<3, 1>(0, 3);
        return T_inv;
    }

    static Eigen::Matrix3d Skew(const Eigen::Vector3d & in) {
        Eigen::Matrix3d matrix;
        matrix << 0.0, -in[2], in[1],
                  in[2], 0.0, -in[0],
                 -in[1], in[0], 0.0;
        return matrix;
    }
};