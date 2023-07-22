#pragma once
#include "Updater.h"

enum OBSType {
    GPS = 1,
    WHEEL,
    CAMERA,
};

class Filter : public Updater
{
public:
    Filter(const std::shared_ptr<Parameter> &param_ptr, const std::shared_ptr<DataManager> &data_manager_ptr,
           const std::shared_ptr<StateManager> &state_manager_ptr)
    {
        param_ptr_ = param_ptr;
        data_manager_ptr_ = data_manager_ptr;
        state_manager_ptr_ = state_manager_ptr;
        gps_observer_ptr_ = std::make_shared<GPSObserver>(param_ptr, data_manager_ptr, state_manager_ptr);

        run_thread_ptr_ = std::make_shared<std::thread>(&Filter::Run, this);
    }

    
private:
    void Run();
    void DelayedRun();
    void Update(const std::shared_ptr<State> & state_ptr, const GPSData & gps_data);
    void ESKFUpdate(const Eigen::MatrixXd & H, const Eigen::MatrixXd & C, const Eigen::MatrixXd & R, Eigen::MatrixXd & Z, Eigen::MatrixXd & C_new, Eigen::VectorXd & X) {
        Eigen::MatrixXd K = C * H.transpose() * (H * C * H.transpose() + R).inverse();
        C_new = (Eigen::MatrixXd::Identity(param_ptr_->STATE_DIM, param_ptr_->STATE_DIM) - K * H) * C;
        X = K * Z;
    }
    std::shared_ptr<Parameter> param_ptr_;
    std::shared_ptr<DataManager> data_manager_ptr_;
    std::shared_ptr<StateManager> state_manager_ptr_;

    // todo add
    std::shared_ptr<GPSObserver> gps_observer_ptr_;
};