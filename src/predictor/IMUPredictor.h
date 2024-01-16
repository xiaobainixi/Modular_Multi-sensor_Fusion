#pragma once

#include "Predictor.h"

class IMUPredictor : public Predictor {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    IMUPredictor(
        std::shared_ptr<StateManager> state_manager_ptr, std::shared_ptr<Parameter> param_ptr,
        std::shared_ptr<DataManager> data_manager_ptr, std::shared_ptr<Viewer> viewer_ptr = nullptr)
    {
        viewer_ptr_ = viewer_ptr;
        state_manager_ptr_ = state_manager_ptr;
        param_ptr_ = param_ptr;
        data_manager_ptr_ = data_manager_ptr;

        // run_thread_ptr_ = std::make_shared<std::thread>(&IMUPredictor::Run, this);
    }
    void RunOnce();
private:
    void Run();
    
    std::shared_ptr<StateManager> state_manager_ptr_;
    IMUData last_data_;
};