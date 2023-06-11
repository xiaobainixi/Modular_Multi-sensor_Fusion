#pragma once
#include <iostream>
#include <memory>
#include <mutex>
#include <Eigen/Core>

#include "common/StateManager.h"
#include "common/DataManager.h"

class Predictor {
public:
    Predictor() = default;

    virtual void Predict() = 0;

protected:
    std::shared_ptr<StateManager> state_manager_ptr_;
    std::shared_ptr<Parameter> param_ptr_;
    std::shared_ptr<DataManager> data_manager_ptr_;
};
    