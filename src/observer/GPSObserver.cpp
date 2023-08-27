#include "GPSObserver.h"

bool GPSObserver::ComputeHZR(
    const GPSData & gps_data, const std::shared_ptr<State> & state_ptr, Eigen::MatrixXd & H, Eigen::MatrixXd & Z, Eigen::MatrixXd &R)
{
    // 1. gps_data
    double x = 0.0, y = 0.0, z = 0.0;
    if (!coo_trans_ptr_)
        coo_trans_ptr_ = std::make_shared<CooTrans>(gps_data.lat_, gps_data.lon_, gps_data.h_);
    else
        coo_trans_ptr_->getENH(gps_data.lat_, gps_data.lon_, gps_data.h_, x, y, z);

    // 2. compute
    Eigen::Vector3d gps_enu(x, y, z);
    Eigen::Vector3d t_error = gps_enu - state_ptr->twb_;
    Z = Eigen::VectorXd::Zero(3);
    Z.block<3, 1>(0, 0) = t_error;
    H = H_;
    R = R_;
    return true;
}