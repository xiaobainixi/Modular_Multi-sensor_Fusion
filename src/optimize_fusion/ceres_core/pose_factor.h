#ifndef POSE_FACTOR_H_
#define POSE_FACTOR_H_

#include <Eigen/Dense>
#include <ceres/ceres.h>
#include <Eigen/Core>

/* 
**  parameters[0]: pose which needs to be anchored to a constant value
 */
class PoseFactor : public ceres::SizedCostFunction<6, 7>
{
    public: 
        PoseFactor() = delete;
        PoseFactor(const std::vector<double> anchor_value);
        virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;
    private:
        Eigen::Matrix<double, 7, 1> _anchor_point;
        constexpr static double sqrt_info = 120;
};

#endif