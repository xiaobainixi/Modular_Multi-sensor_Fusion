#pragma once

#include "Predictor.h"
#include <sophus/so3.hpp>

<<<<<<< HEAD
class WheelPredictor : public Predictor {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    WheelPredictor(
        std::shared_ptr<StateManager> state_manager_ptr, std::shared_ptr<Parameter> param_ptr,
        std::shared_ptr<DataManager> data_manager_ptr, std::shared_ptr<Viewer> viewer_ptr = nullptr)
    {
        viewer_ptr_ = viewer_ptr;
        state_manager_ptr_ = state_manager_ptr;
        param_ptr_ = param_ptr;
        data_manager_ptr_ = data_manager_ptr;

        run_thread_ptr_ = std::make_shared<std::thread>(&WheelPredictor::Run, this);
=======

#pragma once

#include "Predictor.h"

class WheelPredictor : public Predictor {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    WheelPredictor(
        std::shared_ptr<StateManager> state_manager_ptr, std::shared_ptr<Parameter> param_ptr,
        std::shared_ptr<DataManager> data_manager_ptr, std::shared_ptr<Viewer> viewer_ptr = nullptr)
    {
        viewer_ptr_ = viewer_ptr;
        state_manager_ptr_ = state_manager_ptr;
        param_ptr_ = param_ptr;
        data_manager_ptr_ = data_manager_ptr;

        run_thread_ptr_ = std::make_shared<std::thread>(&IMUPredictor::Run, this);
>>>>>>> dc08928f99acf1c61e5b91f2bb97164b37a63618
    }

private:
    void Run();
    std::shared_ptr<StateManager> state_manager_ptr_;
    WheelData last_data_;
};