#pragma once
#include <iostream>
#include <memory>
#include <mutex>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>


#include "common/StateManager.h"
#include "common/DataManager.h"

class Predictor {
public:
    Predictor() = default;

    virtual void Predict() = 0;

protected:
    std::shared_ptr<Parameter> param_ptr_;
    std::shared_ptr<DataManager> data_manager_ptr_;
    double g_ = 9.81;
    Eigen::Vector3d gw_ = Eigen::Vector3d(0.0, 0.0, -g_);
};
    