#pragma once

#include <iostream>
#include <vector>
#include <memory>
#include <mutex>
#include <Eigen/Core>

#include "Parameter.h"
#include "DataManager.h"
#include "Converter.h"

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
    Eigen::Vector3d ba_ = Eigen::Vector3d::Zero();
    Eigen::Vector3d bg_ = Eigen::Vector3d::Zero();

    void Update(std::shared_ptr<Parameter> param_ptr, const Eigen::VectorXd & X, const Eigen::MatrixXd & C_new) {
        if (param_ptr->state_type_ == 0) {
            Rwb_ = Converter::ExpSO3(X.block<3, 1>(param_ptr->ORI_INDEX_STATE_, 0)) * Rwb_;
            Vw_ += X.block<3, 1>(param_ptr->VEL_INDEX_STATE_, 0);
            twb_ += X.block<3, 1>(param_ptr->POSI_INDEX, 0);
            ba_ += X.block<3, 1>(param_ptr->ACC_BIAS_INDEX_STATE_, 0);
            bg_ += X.block<3, 1>(param_ptr->GYRO_BIAS_INDEX_STATE_, 0);
        } else {
            // todo
        }
        C_ = C_new;
    }
};

struct CompareTime {
	bool operator() (const std::shared_ptr<State>& s1,  const std::shared_ptr<State>& s2) {	
		return s1->time_ < s2->time_;
	}
};

class StateManager {
public:
    StateManager(std::shared_ptr<Parameter> param_ptr) {
        param_ptr_ = param_ptr;
    }

    inline bool GetNearestState(std::shared_ptr<State> & state) {
        std::unique_lock<std::mutex> lock(states_mtx_);
        if (states_.empty()) {
            return false;
        } else {
            state = states_[states_.size() - 1];
            return true;
        }
    }

    inline bool PushState(const std::shared_ptr<State> & state) {
        if (!state)
            return false;

        std::unique_lock<std::mutex> lock(states_mtx_);
        if (!states_.empty() && state->time_ <= states_[states_.size() - 1]->time_)
            return false;
        states_.push_back(state);

        // 只保留最近10s状态,二分查找下，从头找也行不差很多
        std::shared_ptr<State> tar = std::make_shared<State>();
        tar->time_ = state->time_ - 10.0;
        auto iter = lower_bound(states_.begin(), states_.end(), tar, CompareTime());
        if (iter != states_.begin())
            states_ = std::vector<std::shared_ptr<State>>(iter, states_.end());
        return true;
    }

    inline std::vector<std::shared_ptr<State>> GetState() {
        std::unique_lock<std::mutex> lock(states_mtx_);
        return states_;
    }

    // todo getbetween 差值等
private:
    // 不同线程访问，一定要加锁
    std::mutex states_mtx_;
    // 只保留最近10s状态？
    std::vector<std::shared_ptr<State>> states_;
    std::shared_ptr<Parameter> param_ptr_;
};