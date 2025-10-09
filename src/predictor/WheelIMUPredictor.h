#pragma once

#include "Predictor.h"
#include "preint/WheelIMUPreintegration.h"

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

    std::shared_ptr<Preintegration> CreatePreintegration(
        const double start_time,
        const double end_time,
        const Eigen::Vector3d& ba,
        const Eigen::Vector3d& bg)
    {
        std::vector<WheelIMUData> datas;
        data_manager_ptr_->GetDatasBetween(datas, start_time, end_time);
        if (datas.size() < 3) return nullptr;
        return CreatePreintegration(datas, ba, bg);
    }

private:
    void Run();

    std::shared_ptr<WheelIMUPreintegration> CreatePreintegration(
        const std::vector<WheelIMUData>& datas,
        const Eigen::Vector3d& ba,
        const Eigen::Vector3d& bg)
    {
        if (datas.empty()) return nullptr;
        // 构造初始状态
        State init_state;
        init_state.bg_ = bg;

        // 实例化预积分对象
        auto preint_ptr = std::make_shared<WheelIMUPreintegration>(datas.front(), init_state, param_ptr_);
        for (size_t i = 1; i < datas.size(); ++i) {
            preint_ptr->Input(datas[i]);
        }
        return preint_ptr;
    }

    std::shared_ptr<StateManager> state_manager_ptr_;
    // IMUData last_imu_data_;
    // WheelIMUData last_wheel_data_;
    WheelIMUData last_data_;
};
