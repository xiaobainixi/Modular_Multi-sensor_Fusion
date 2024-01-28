#pragma once

#include "Predictor.h"

class WheelIMUPredictor : public Predictor {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    WheelIMUPredictor(
        std::shared_ptr<StateManager> state_manager_ptr, std::shared_ptr<Parameter> param_ptr,
        std::shared_ptr<DataManager> data_manager_ptr, std::shared_ptr<Viewer> viewer_ptr = nullptr)
    {
        viewer_ptr_ = viewer_ptr;
        state_manager_ptr_ = state_manager_ptr;
        param_ptr_ = param_ptr;
        data_manager_ptr_ = data_manager_ptr;

        // run_thread_ptr_ = std::make_shared<std::thread>(&WheelIMUPredictor::Run, this);
    }
    void RunOnce();

private:
    void Run();
    std::shared_ptr<StateManager> state_manager_ptr_;
    // IMUData last_imu_data_;
    // WheelData last_wheel_data_;
    WheelIMUData last_data_;
};
