#pragma once

#include <Eigen/Geometry>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/SVD>
#include <Eigen/QR>
#include <Eigen/SparseCore>
#include <Eigen/SPQRSupport>
class Converter
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    static Eigen::Matrix4d InversePose(const Eigen::Matrix4d &T)
    {
        Eigen::Matrix4d T_inv = Eigen::Matrix4d::Identity();
        T_inv.block<3, 3>(0, 0) = T.block<3, 3>(0, 0).transpose();
        T_inv.block<3, 1>(0, 3) = -T_inv.block<3, 3>(0, 0) * T.block<3, 1>(0, 3);
        return T_inv;
    }

    static Eigen::Matrix3d Skew(const Eigen::Vector3d &in)
    {
        Eigen::Matrix3d matrix;
        matrix << 0.0, -in[2], in[1],
            in[2], 0.0, -in[0],
            -in[1], in[0], 0.0;
        return matrix;
    }

    // static Eigen::Matrix3d NormalizeRotation(const Eigen::Matrix3d &R)
    // {
    //     Eigen::JacobiSVD<Eigen::Matrix3d> svd(R, Eigen::ComputeFullU | Eigen::ComputeFullV);
    //     return svd.matrixU() * svd.matrixV().transpose();
    // }

    static Eigen::Matrix3d ExpSO3(const Eigen::Vector3d &vec)
    {
        Eigen::Matrix3d I = Eigen::Matrix3d::Identity(), deltaR;

        const float d2 = vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2];
        const float d = sqrt(d2);

        Eigen::Matrix3d W;
        W << 0, -vec[2], vec[1],
            vec[2], 0, -vec[0],
            -vec[1], vec[0], 0;
        if (d < 1e-4)     // 10的-4次方
            return I + W;
        else
            return I + W * sin(d) / d + W * W * (1.0f - cos(d)) / d2;
    }

    static Eigen::Vector3d LogSO3(const Eigen::Matrix3d &R)
    {
        const double tr = R(0, 0) + R(1, 1) + R(2, 2);
        Eigen::Vector3d w;
        w << (R(2, 1) - R(1, 2)) / 2, (R(0, 2) - R(2, 0)) / 2, (R(1, 0) - R(0, 1)) / 2;
        const double costheta = (tr - 1.0) * 0.5f;
        if (costheta > 1 || costheta < -1)
            return w;
        const double theta = acos(costheta);
        const double s = sin(theta);
        if (fabs(s) < 1e-5)
            return w;
        else
            return theta * w / s;
    }

    /**
     *  @brief 反对称矩阵
     *  @note Performs the operation:
     *  w   ->  [  0 -w3  w2]
     *          [ w3   0 -w1]
     *          [-w2  w1   0]
     */
    static Eigen::Matrix3d SkewSymmetric(const Eigen::Vector3d &w)
    {
        Eigen::Matrix3d w_hat;
        w_hat(0, 0) = 0;
        w_hat(0, 1) = -w(2);
        w_hat(0, 2) = w(1);
        w_hat(1, 0) = w(2);
        w_hat(1, 1) = 0;
        w_hat(1, 2) = -w(0);
        w_hat(2, 0) = -w(1);
        w_hat(2, 1) = w(0);
        w_hat(2, 2) = 0;
        return w_hat;
    }

    static Eigen::Matrix3d Euler2Matrix(const Eigen::Vector3d &euler) {
        return Eigen::Matrix3d(Eigen::AngleAxisd(euler[2], Eigen::Vector3d::UnitZ()) *
                        Eigen::AngleAxisd(euler[1], Eigen::Vector3d::UnitY()) *
                        Eigen::AngleAxisd(euler[0], Eigen::Vector3d::UnitX()));
    }
};