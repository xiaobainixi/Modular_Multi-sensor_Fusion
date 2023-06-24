#pragma once

#include "Predictor.h"

class IMUPredictor : public Predictor {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    IMUPredictor(std::shared_ptr<StateManager> state_manager_ptr, std::shared_ptr<Parameter> param_ptr, std::shared_ptr<DataManager> data_manager_ptr) {
        state_manager_ptr_ = state_manager_ptr;
        param_ptr_ = param_ptr;
        data_manager_ptr_ = data_manager_ptr;
    }

    void Predict();

private:
    std::shared_ptr<StateManager> state_manager_ptr_;
    IMUData last_data_;
};