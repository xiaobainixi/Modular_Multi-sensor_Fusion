#pragma once

#include "Predictor.h"
// #include "preint/imu/preintegration.h"
#include "optimize_fusion/ceres_fusion.hpp"
#include <memory>

class CeresPredictor : public Predictor {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    CeresPredictor(
        std::shared_ptr<StateManager> state_manager_ptr, std::shared_ptr<Parameter> param_ptr,
        std::shared_ptr<DataManager> data_manager_ptr, std::shared_ptr<Viewer> viewer_ptr = nullptr)
    {
        viewer_ptr_ = viewer_ptr;
        state_manager_ptr_ = state_manager_ptr;
        param_ptr_ = param_ptr;
        data_manager_ptr_ = data_manager_ptr;

        // run_thread_ptr_ = std::make_shared<std::thread>(&CeresPredictor::Run, this);
        ceres_fusion_ptr_ = std::make_shared<CeresBasedFusionInterface>();
    }
    void RunOnce();

    virtual bool IsInit() {
        return ceres_fusion_ptr_->IsInit();
    }

    virtual double GetLastImuTime() {return last_data_.time_;}

    virtual std::shared_ptr<CeresBasedFusionInterface> getInterface() {return ceres_fusion_ptr_;}
private:
    void Run();
    std::shared_ptr<CeresBasedFusionInterface> ceres_fusion_ptr_;
    std::shared_ptr<StateManager> state_manager_ptr_;
    IMUData last_data_;
};