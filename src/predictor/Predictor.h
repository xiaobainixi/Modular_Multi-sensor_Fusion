#pragma once
#include <iostream>
#include <memory>
#include <unistd.h>
#include <mutex>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>


#include "common/StateManager.h"
#include "common/DataManager.h"

#include "viewer/Viewer.h"

#include "../optimize_fusion/ceres_fusion.hpp"

class Predictor {
public:
    Predictor() = default;
    virtual void RunOnce() {}
    virtual bool IsInit() {return true;}
    virtual std::shared_ptr<CeresBasedFusionInterface> getInterface() {return nullptr;}
    virtual double GetLastImuTime() {return -1;}
protected:
    virtual void Run() = 0;
    std::shared_ptr<std::thread> run_thread_ptr_;

    std::shared_ptr<Parameter> param_ptr_;
    std::shared_ptr<DataManager> data_manager_ptr_;
    double g_ = 9.81;
    Eigen::Vector3d gw_ = Eigen::Vector3d(0.0, 0.0, -g_);
    std::shared_ptr<Viewer> viewer_ptr_;
};
    