#pragma once
#include "Predictor.h"

class WheelPredictor : public Predictor{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    WheelPredictor(std::shared_ptr<StateManager> state_manager_ptr, std::shared_ptr<Parameter> param_ptr, std::shared_ptr<DataManager> data_manager_ptr) {
        state_manager_ptr_ = state_manager_ptr;
        param_ptr_ = param_ptr;
        data_manager_ptr_ = data_manager_ptr;
    }

    void Predict();

private:
    WheelData last_data_;
};