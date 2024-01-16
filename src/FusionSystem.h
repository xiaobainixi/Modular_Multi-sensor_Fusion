#pragma once
#include <atomic>
#include <unistd.h>

#include "updater/Filter.h"

class FusionSystem {
public:
    FusionSystem(std::shared_ptr<Parameter> param_ptr, std::shared_ptr<StateManager> state_manager_ptr, std::shared_ptr<DataManager> data_manager_ptr) {
        param_ptr_ = param_ptr;
        state_manager_ptr_ = state_manager_ptr;
        data_manager_ptr_ = data_manager_ptr;
        viewer_ptr_ = std::make_shared<Viewer>();

        if (param_ptr_->fusion_model_ == 0) {
            updater_ptr_ = std::make_shared<Filter>(param_ptr_, data_manager_ptr_, state_manager_ptr_, viewer_ptr_);
        } else if (param_ptr_->fusion_model_ == 1) {
            // todo opt
        }
    }

    void Input(const IMUData &imu_data) {
        data_manager_ptr_->Input(imu_data);
    }
    void Input(const GPSData &gps_data) {
        data_manager_ptr_->Input(gps_data);
    }

    // State GetNewestState() {

    // }
private:

    std::shared_ptr<StateManager> state_manager_ptr_;
    std::shared_ptr<DataManager> data_manager_ptr_;
    std::shared_ptr<Parameter> param_ptr_;
    std::shared_ptr<Updater> updater_ptr_;
    std::shared_ptr<Viewer> viewer_ptr_;
};