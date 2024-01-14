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
    Eigen::MatrixXd C_;  // 协方差矩阵
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
            Rwb_ = Converter::ExpSO3(X.block<3, 1>(param_ptr->ORI_INDEX_STATE_, 0)) * Rwb_;
            twb_ += X.block<3, 1>(param_ptr->POSI_INDEX, 0);
        }
        C_ = C_new;
    }
};

struct CamState
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    CamState() : id(0), time(0)
    {
    }

    CamState(const int &new_id) : id(new_id), time(0)
    {
    }
    // Time when the state is recorded
    double time = -1.0;
    int id;

    // Orientation
    // Take a vector from the world frame to the camera frame.
    Eigen::Matrix3d Rwc_ = Eigen::Matrix3d::Identity();

    // Position of the camera frame in the world frame.
    Eigen::Vector3d twc_ = Eigen::Vector3d::Zero();

    // These two variables should have the same physical
    // interpretation with `orientation` and `position`.
    // There two variables are used to modify the measurement
    // Jacobian matrices to make the observability matrix
    // have proper null space.
    // 使可观测性矩阵具有适当的零空间的旋转平移
    Eigen::Matrix3d Rwc_null_ = Eigen::Matrix3d::Identity();
    Eigen::Vector3d twc_null_ = Eigen::Vector3d::Zero();
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

    inline bool GetNearestState(std::shared_ptr<State> & state, double time = -1.0) {
        if (states_.empty())
            return false;
        if (time < 0.0) {
            std::unique_lock<std::mutex> lock(states_mtx_);
            state = states_[states_.size() - 1];
            return true;
        } else if (states_.size() > 1) {
            std::shared_ptr<State> tar = std::make_shared<State>();
            tar->time_ = time;
            auto iter = lower_bound(states_.begin(), states_.end(), tar, CompareTime());
            if (iter != states_.begin() && iter != states_.end()) {
                auto iter_sub = iter;
                iter_sub--;
                state = std::abs((*iter)->time_ - time) < std::abs((*iter_sub)->time_ - time) ? (*iter) : (*iter_sub);
                return true;
            }
        }
        return false;
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

    std::map<int, std::shared_ptr<CamState>, std::less<int>, 
        Eigen::aligned_allocator<std::pair<const int, std::shared_ptr<CamState>>>> cam_states_;
private:
    // 不同线程访问，一定要加锁
    std::mutex states_mtx_;
    // 只保留最近10s状态？
    std::vector<std::shared_ptr<State>> states_;
    std::shared_ptr<Parameter> param_ptr_;
};