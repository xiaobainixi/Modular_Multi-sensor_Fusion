#pragma once

#include "predictor/IMUPredictor.h"
#include "predictor/WheelPredictor.h"
#include "predictor/WheelIMUPredictor.h"

class FusionSystem {
public:
    FusionSystem() {
        data_manager_ptr_ = std::make_shared<DataManager>();
        param_ptr_ = std::make_shared<Parameter>("");
        state_manager_ptr_ = std::make_shared<StateManager>(param_ptr_);

        {
            // std::shared_ptr<IMUPredictor> imu_predictor_ptr = std::make_shared<IMUPredictor>(state_manager_ptr_, param_ptr_);
            predictor_ptr_ = std::make_shared<IMUPredictor>(state_manager_ptr_, param_ptr_, data_manager_ptr_);
        }
    }

    // void Input(std::shared_ptr<IMUPredictor::IMUData> imu_data_ptr);
    // void Input(std::shared_ptr<IMUPredictor::IMUData> imu_data_ptr);
    // void Input(std::shared_ptr<IMUPredictor::IMUData> imu_data_ptr);
    // void Input(std::shared_ptr<IMUPredictor::IMUData> imu_data_ptr);

    State GetNewestState() {

    }
private:
    std::shared_ptr<StateManager> state_manager_ptr_;
    std::shared_ptr<DataManager> data_manager_ptr_;
    
    std::shared_ptr<Parameter> param_ptr_;
    std::shared_ptr<Predictor> predictor_ptr_;
};