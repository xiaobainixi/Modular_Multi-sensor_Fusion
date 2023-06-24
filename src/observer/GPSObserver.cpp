#include "GPSObserver.h"

bool GPSObserver::ComputeHZR(
    const GPSData & gps_data, const std::shared_ptr<State> & state_ptr, Eigen::MatrixXd & H, Eigen::MatrixXd & Z, Eigen::MatrixXd &R)
{
    // 1. gps_data
    double x = 0.0, y = 0.0, z = 0.0;
    if (!coo_trans_ptr_)
        coo_trans_ptr_ = std::make_shared<CooTrans>(gps_data.lat, gps_data.lon, gps_data.h);
    else
        coo_trans_ptr_->getENH(gps_data.lat, gps_data.lon, gps_data.h, x, y, z);

    // 2. compute
    Eigen::Vector3d gps_enu(x, y, z);
    Eigen::Vector3d t_error = gps_enu - state_ptr->twb_;
    Z = Eigen::VectorXd::Zero(param_ptr_->STATE_DIM);
    Z.block<3, 1>(param_ptr_->POSI_INDEX, 0) = t_error;
    H = H_;
    R = R_;
    return true;
}