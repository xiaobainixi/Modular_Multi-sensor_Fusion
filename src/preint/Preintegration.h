#pragma once

#include <iostream>
#include <memory>
#include <unistd.h>
#include <mutex>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <ceres/ceres.h>

#include "common/StateManager.h"
#include "common/DataManager.h"
#include "common/Converter.h"
class State;
class Preintegration : public std::enable_shared_from_this<Preintegration>
{
public:
    Preintegration() {}

    virtual void Repropagate(const Eigen::Vector3d &new_ba, const Eigen::Vector3d &new_bg)
    {
        LOG(INFO) << "你不应该被执行的";
    }

    virtual std::shared_ptr<State> predict(std::shared_ptr<State> state) {return nullptr;}

    std::shared_ptr<Parameter> param_ptr_;

    double dt_ = 0.0;
    double sum_dt_ = 0.0;
    Eigen::MatrixXd jacobian_, covariance_;
    Eigen::MatrixXd noise_;

    Eigen::Vector3d delta_p_;
    Eigen::Quaterniond delta_q_;
    Eigen::Vector3d delta_v_;
    Eigen::Vector3d ba_, bg_;
};