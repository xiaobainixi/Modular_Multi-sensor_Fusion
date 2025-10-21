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

    // static Eigen::Matrix3d NormalizeRotation(const Eigen::Matrix3d &R)
    // {
    //     Eigen::JacobiSVD<Eigen::Matrix3d> svd(R, Eigen::ComputeFullU | Eigen::ComputeFullV);
    //     return svd.matrixU() * svd.matrixV().transpose();
    // }

    // 旋转向量（so(3)）转四元数
    static Eigen::Quaterniond so3ToQuat(const Eigen::Vector3d& vec)
    {
        Eigen::AngleAxisd aa(vec.norm(), vec.normalized());
        if (vec.norm() < 1e-10) {
            return Eigen::Quaterniond::Identity();
        }
        return Eigen::Quaterniond(aa);
    }

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
    static Eigen::Matrix3d Skew(const Eigen::Vector3d &w)
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


    // SO3 FUNCTIONS
    static Eigen::Matrix3d InverseRightJacobianSO3(const Eigen::Vector3d &v)
    {
        return InverseRightJacobianSO3(v[0],v[1],v[2]);
    }

    static Eigen::Matrix3d InverseRightJacobianSO3(const Eigen::Quaterniond& q)
    {
        return InverseRightJacobianSO3(QuaternionToRotVec(q));
    }

    static Eigen::Matrix3d InverseRightJacobianSO3(const double x, const double y, const double z)
    {
        const double d2 = x*x+y*y+z*z;
        const double d = sqrt(d2);

        Eigen::Matrix3d W;
        W << 0.0, -z, y,z, 0.0, -x,-y,  x, 0.0;
        if(d<1e-5)
            return Eigen::Matrix3d::Identity();
        else
            return Eigen::Matrix3d::Identity() + W/2 + W*W*(1.0/d2 - (1.0+cos(d))/(2.0*d*sin(d)));
    }

    static Eigen::Matrix3d RightJacobianSO3(const Eigen::Vector3d &v)
    {
        return RightJacobianSO3(v[0],v[1],v[2]);
    }

    static Eigen::Matrix3d RightJacobianSO3(const double x, const double y, const double z)
    {
        const double d2 = x*x+y*y+z*z;
        const double d = sqrt(d2);

        Eigen::Matrix3d W;
        W << 0.0, -z, y,z, 0.0, -x,-y,  x, 0.0;
        if(d<1e-5)
        {
            return Eigen::Matrix3d::Identity();
        }
        else
        {
            return Eigen::Matrix3d::Identity() - W*(1.0-cos(d))/d2 + W*W*(d-sin(d))/(d2*d);
        }
    }

    // 四元数 -> 旋转向量 (axis-angle 的 angle*axis 形式, so(3)向量)
    static Eigen::Vector3d QuaternionToRotVec(const Eigen::Quaterniond& q_in) {
        Eigen::Quaterniond q = q_in.normalized();
        double w = q.w();
        Eigen::Vector3d v = q.vec();
        double nv = v.norm();
        // 若接近单位，使用一阶近似避免除零
        if (nv < 1e-12) {
            // q ≈ [ε, 1]；旋转向量 ≈ 2 * v
            return 2.0 * v;
        }
        double angle = 2.0 * std::atan2(nv, w);
        // 将角度映射到 (-pi, pi] 以保持唯一
        if (angle > M_PI) {
            angle -= 2.0 * M_PI;
        }
        Eigen::Vector3d axis = v / nv;
        return angle * axis;
    }

    template <typename Derived>
    static Eigen::Quaternion<typename Derived::Scalar> RotVecToQuaternion(
        const Eigen::MatrixBase<Derived> &theta)
    {
        typedef typename Derived::Scalar Scalar_t;
        // Scalar_t angle = theta.norm();
        // Eigen::Vector3<Scalar_t> vec = theta.normalized();
        // return Eigen::Quaternion<Scalar_t>(Eigen::AngleAxis<Scalar_t>(angle, vec));
        Scalar_t theta_norm = theta.norm();
        Eigen::Quaternion<Scalar_t> dq;
        if (theta_norm < Scalar_t(1e-8)) {
            dq.w() = Scalar_t(1.0);
            dq.x() = theta.x() * Scalar_t(0.5);
            dq.y() = theta.y() * Scalar_t(0.5);
            dq.z() = theta.z() * Scalar_t(0.5);
        } else {
            Scalar_t half_theta = theta_norm * Scalar_t(0.5);
            Scalar_t sin_half_theta = sin(half_theta);
            Scalar_t scale = sin_half_theta / theta_norm;
            dq.w() = cos(half_theta);
            dq.x() = theta.x() * scale;
            dq.y() = theta.y() * scale;
            dq.z() = theta.z() * scale;
        }
        dq.normalize();
        return dq;
    }

    template <typename Derived>
    static Eigen::Matrix<typename Derived::Scalar, 4, 4> Qleft(const Eigen::QuaternionBase<Derived> &q)
    {
        Eigen::Quaternion<typename Derived::Scalar> qq = q;
        Eigen::Matrix<typename Derived::Scalar, 4, 4> ans;
        ans(0, 0) = qq.w(), ans.template block<1, 3>(0, 1) = -qq.vec().transpose();
        ans.template block<3, 1>(1, 0) = qq.vec(), ans.template block<3, 3>(1, 1) = qq.w() * Eigen::Matrix<typename Derived::Scalar, 3, 3>::Identity() + Skew(qq.vec());
        return ans;
    }

    template <typename Derived>
    static Eigen::Matrix<typename Derived::Scalar, 4, 4> Qright(const Eigen::QuaternionBase<Derived> &p)
    {
        Eigen::Quaternion<typename Derived::Scalar> pp = p;
        Eigen::Matrix<typename Derived::Scalar, 4, 4> ans;
        ans(0, 0) = pp.w(), ans.template block<1, 3>(0, 1) = -pp.vec().transpose();
        ans.template block<3, 1>(1, 0) = pp.vec(), ans.template block<3, 3>(1, 1) = pp.w() * Eigen::Matrix<typename Derived::Scalar, 3, 3>::Identity() - Skew(pp.vec());
        return ans;
    }

    static Eigen::Vector3d R2ypr(const Eigen::Matrix3d &R)
    {
        Eigen::Vector3d n = R.col(0);
        Eigen::Vector3d o = R.col(1);
        Eigen::Vector3d a = R.col(2);

        Eigen::Vector3d ypr(3);
        double y = atan2(n(1), n(0));
        double p = atan2(-n(2), n(0) * cos(y) + n(1) * sin(y));
        double r = atan2(a(0) * sin(y) - a(1) * cos(y), -o(0) * sin(y) + o(1) * cos(y));
        ypr(0) = y;
        ypr(1) = p;
        ypr(2) = r;

        return ypr / M_PI * 180.0;
    }

    template <typename Derived>
    static Eigen::Matrix<typename Derived::Scalar, 3, 3> ypr2R(const Eigen::MatrixBase<Derived> &ypr)
    {
        typedef typename Derived::Scalar Scalar_t;

        Scalar_t y = ypr(0) / 180.0 * M_PI;
        Scalar_t p = ypr(1) / 180.0 * M_PI;
        Scalar_t r = ypr(2) / 180.0 * M_PI;

        Eigen::Matrix<Scalar_t, 3, 3> Rz;
        Rz << cos(y), -sin(y), 0,
            sin(y), cos(y), 0,
            0, 0, 1;

        Eigen::Matrix<Scalar_t, 3, 3> Ry;
        Ry << cos(p), 0., sin(p),
            0., 1., 0.,
            -sin(p), 0., cos(p);

        Eigen::Matrix<Scalar_t, 3, 3> Rx;
        Rx << 1., 0., 0.,
            0., cos(r), -sin(r),
            0., sin(r), cos(r);

        return Rz * Ry * Rx;
    }

    static Eigen::Matrix3d g2R(const Eigen::Vector3d &g)
    {
        Eigen::Matrix3d R0;
        Eigen::Vector3d ng1 = g.normalized();
        Eigen::Vector3d ng2{0, 0, 1.0};
        R0 = Eigen::Quaterniond::FromTwoVectors(ng1, ng2).toRotationMatrix();
        double yaw = R2ypr(R0).x();
        R0 = ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
        // R0 = Utility::ypr2R(Eigen::Vector3d{-90, 0, 0}) * R0;
        return R0;
    }

};