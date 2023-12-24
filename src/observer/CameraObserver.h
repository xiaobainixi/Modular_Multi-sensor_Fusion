#pragma once

#include "Observer.h"
class CameraObserver : public Observer
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    struct CamState
    {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        // Time when the state is recorded
        double time = -1.0;

        // Orientation
        // Take a vector from the world frame to the camera frame.
        Eigen::Matrix3d Rwc_ = Eigen::Matrix3d::Identity();

        // Position of the camera frame in the world frame.
        Eigen::Vector3d twc_ = Eigen::Vector3d::Zero();

        // These two variables should have the same physical
        // interpretation with `orientation` and `position`.
        // There two variables are used to modify the measurement
        // Jacobian matrices to make the observability matrix
        // have proper null space.
        // 使可观测性矩阵具有适当的零空间的旋转平移
        Eigen::Matrix3d Rwc_null_ = Eigen::Matrix3d::Identity();
        Eigen::Vector3d twc_null_ = Eigen::Vector3d::Zero();
    };

    CameraObserver(const std::shared_ptr<Parameter> &param_ptr, const std::shared_ptr<DataManager> &data_manager_ptr,
                const std::shared_ptr<StateManager> &state_manager_ptr)
                
    {
        param_ptr_ = param_ptr;
        data_manager_ptr_ = data_manager_ptr;
        state_manager_ptr_ = state_manager_ptr;

    }

    bool ComputeHZR(
        const FeatureData & feature_data, const std::shared_ptr<State> & state_ptr,
        Eigen::MatrixXd & H, Eigen::MatrixXd & Z, Eigen::MatrixXd &R);

private:

    // eskf
    std::map<double, std::shared_ptr<CamState>> cam_states_;
    Eigen::MatrixXd state_cov;
};