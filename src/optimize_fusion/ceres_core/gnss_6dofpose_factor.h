#ifndef GPS_6DOFPOSE_FACTOR_H_
#define GPS_6DOFPOSE_FACTOR_H_

#include <Eigen/Dense>
#include <ceres/ceres.h>
#include <Eigen/Core>

/* 
**  parameters[0]: pose which needs to be anchored to a constant value
 */
class Gnss6DofPoseFactor : public ceres::SizedCostFunction<6, 7>
{
    public: 
        Gnss6DofPoseFactor() = delete;
        // 传入Gnss的6dof的pose（旋转与平移），以及对应的信息矩阵（注意信息矩阵的维度应该是残差*残差的维度）
        Gnss6DofPoseFactor(const Eigen::Quaterniond &q_obs, const Eigen::Vector3d &t_obs,
                           const Eigen::Matrix<double, 6, 6> &sqrt_info_matrix)
            : q_obs_(q_obs), t_obs_(t_obs), sqrt_info_matrix_(sqrt_info_matrix) {}
        virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;
    private:
        Eigen::Quaterniond q_obs_;
        Eigen::Vector3d t_obs_;
        Eigen::Matrix<double, 6, 6> sqrt_info_matrix_; // 平移误差在前，旋转误差在后
};

// 创建gps位置相关的factor: 传入的优化量是关于位置和姿态的，不过这里只对位置产生约束
class GnssPositionFactor : public ceres::SizedCostFunction<3, 7> {
    public:
        GnssPositionFactor() = delete;
        GnssPositionFactor(const Eigen::Vector3d& p_obs, const Eigen::Matrix3d& sqrt_info_matrix) :
            p_bos_(p_obs), sqrt_info_matrix_(sqrt_info_matrix) {}
        virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;
    private:
        Eigen::Vector3d p_bos_;
        Eigen::Matrix3d sqrt_info_matrix_;
};

#endif