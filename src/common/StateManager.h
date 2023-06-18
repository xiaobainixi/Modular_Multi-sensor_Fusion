#pragma once

#include <iostream>
#include <vector>
#include <memory>
#include <mutex>
#include <Eigen/Core>

#include "Parameter.h"
#include "DataManager.h"

template<class AlignedData>
class State {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    // todo 是否有必要加锁
    // std::mutex state_mtx_;
    double time_;
    Eigen::MatrixXd C_;
    Eigen::Vector3d Vw_ = Eigen::Vector3d::Zero();
    Eigen::Vector3d twb_ = Eigen::Vector3d::Zero();
    Eigen::Matrix3d Rwb_ = Eigen::Matrix3d::Identity();
    Eigen::Vector3d Ba_ = Eigen::Vector3d::Zero();
    Eigen::Vector3d Bw_ = Eigen::Vector3d::Zero();

    AlignedData aligned_data_;
};

template<class AlignedData>
class StateManager {
public:
    StateManager(std::shared_ptr<Parameter> param_ptr) {
        param_ptr_ = param_ptr;
    }

    inline bool GetNearestState(std::shared_ptr<State<AlignedData>> & state) {
        std::unique_lock<std::mutex> lock(states_mtx_);
        if (states_.empty()) {
            return false;
        } else {
            state = states_[states_.size() - 1];
            return true;
        }
    }

    inline bool PushState(const std::shared_ptr<State<AlignedData>> & state) {
        if (!state)
            return false;

        std::unique_lock<std::mutex> lock(states_mtx_);
        if (!states_.empty() && state->time_ <= states_[states_.size() - 1]->time_)
            return false;
        states_.push_back(state);
        return true;
    }

    inline std::vector<std::shared_ptr<State<AlignedData>>> GetState() {
        std::unique_lock<std::mutex> lock(states_mtx_);
        return states_;
    }

    // todo getbetween 差值等
private:
    // 不同线程访问，一定要加锁
    std::mutex states_mtx_;
    // 只保留最近10s状态？
    std::vector<std::shared_ptr<State<AlignedData>>> states_;
    std::shared_ptr<Parameter> param_ptr_;
};