#pragma once
#include <atomic>
#include <unistd.h>

#include "predictor/IMUPredictor.h"
#include "predictor/WheelPredictor.h"
#include "predictor/WheelIMUPredictor.h"

#include "updater/Filter.h"

class FusionSystem {
public:
    FusionSystem() {
        param_ptr_ = std::make_shared<Parameter>("");
        state_manager_ptr_ = std::make_shared<StateManager>(param_ptr_);
        data_manager_ptr_ = std::make_shared<DataManager>();

        {
            predictor_ptr_ = std::make_shared<IMUPredictor>(state_manager_ptr_, param_ptr_, data_manager_ptr_);
        }

        {
            filt_updater_ptr_ = std::make_shared<Filter>(param_ptr_, data_manager_ptr_, state_manager_ptr_);
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

    /**********************************/
    void FilterProcessing();
    /**********************************/

    /**********************************/
    void DelayedFilterProcessing();
    /**********************************/

    /**********************************/
    void OptimizeProcessing();
    /**********************************/


    std::shared_ptr<StateManager> state_manager_ptr_;
    std::shared_ptr<DataManager> data_manager_ptr_;
    std::shared_ptr<Parameter> param_ptr_;
    std::shared_ptr<Predictor> predictor_ptr_;
    std::shared_ptr<Filter> filt_updater_ptr_;
};