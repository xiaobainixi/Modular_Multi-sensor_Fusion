#include "CameraObserver.h"

bool CameraObserver::ComputeHZR(
    const FeatureData & feature_data, const std::shared_ptr<State> & state_ptr,
    Eigen::MatrixXd & H, Eigen::MatrixXd & Z, Eigen::MatrixXd &R)
{
    std::shared_ptr<State> feature_data_state;
    if (!state_manager_ptr_->GetNearestState(feature_data_state, feature_data.time_))
        return false;
    
    
    return true;
}