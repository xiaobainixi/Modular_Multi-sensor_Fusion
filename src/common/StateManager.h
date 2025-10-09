#pragma once

#include <iostream>
#include <vector>
#include <memory>
#include <mutex>
#include <Eigen/Core>

#include "Parameter.h"
#include "DataManager.h"
#include "Converter.h"
#include "preint/Preintegration.h"
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
    Eigen::Quaterniond Rwc_ = Eigen::Quaterniond::Identity();

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

class Preintegration;
class State {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    // todo 是否有必要加锁
    // std::mutex state_mtx_;
    double time_;
    Eigen::MatrixXd C_;  // 协方差矩阵
    Eigen::Vector3d Vw_ = Eigen::Vector3d::Zero();
    Eigen::Vector3d twb_ = Eigen::Vector3d::Zero();
    Eigen::Quaterniond Rwb_ = Eigen::Quaterniond::Identity();
    Eigen::Vector3d ba_ = Eigen::Vector3d::Zero();
    Eigen::Vector3d bg_ = Eigen::Vector3d::Zero();

    GNSSData cur_gnss_data_;
    WheelData cur_wheel_data_;
    FeatureData feature_data_;
    std::shared_ptr<State> last_state_;
    std::shared_ptr<Preintegration> preint_;

    // 左乘
    void Update(std::shared_ptr<Parameter> param_ptr, const Eigen::VectorXd & X, const Eigen::MatrixXd & C_new, std::map<int, std::shared_ptr<CamState>, std::less<int>, 
        Eigen::aligned_allocator<std::pair<const int, std::shared_ptr<CamState>>>> & cam_states) {
        if (param_ptr->state_type_ == 0) {
            Rwb_ = Converter::so3ToQuat(X.block<3, 1>(param_ptr->ORI_INDEX_STATE_, 0)) * Rwb_;
            Vw_ += X.block<3, 1>(param_ptr->VEL_INDEX_STATE_, 0);
            twb_ += X.block<3, 1>(param_ptr->POSI_INDEX, 0);
            ba_ += X.block<3, 1>(param_ptr->ACC_BIAS_INDEX_STATE_, 0);
            bg_ += X.block<3, 1>(param_ptr->GYRO_BIAS_INDEX_STATE_, 0);
            LOG(INFO) << "Update State (type 0):";
            LOG(INFO) << "Rwb_: " << Rwb_.coeffs().transpose();
            LOG(INFO) << "Vw_: " << Vw_.transpose();
            LOG(INFO) << "twb_: " << twb_.transpose();
            LOG(INFO) << "ba_: " << ba_.transpose();
            LOG(INFO) << "bg_: " << bg_.transpose();
        } else if(param_ptr->state_type_ == 1) {
            Rwb_ = Converter::so3ToQuat(X.block<3, 1>(param_ptr->ORI_INDEX_STATE_, 0)) * Rwb_;
            twb_ += X.block<3, 1>(param_ptr->POSI_INDEX, 0);
            LOG(INFO) << "Update State (type 1):";
            LOG(INFO) << "Rwb_: " << Rwb_.coeffs().transpose();
            LOG(INFO) << "twb_: " << twb_.transpose();
        } else if (param_ptr->state_type_ == 2) {
            Rwb_ = Converter::so3ToQuat(X.block<3, 1>(param_ptr->ORI_INDEX_STATE_, 0)) * Rwb_;
            twb_ += X.block<3, 1>(param_ptr->POSI_INDEX, 0);
            bg_ += X.block<3, 1>(param_ptr->GYRO_BIAS_INDEX_STATE_, 0);
            LOG(INFO) << "Update State (type 2):";
            LOG(INFO) << "Rwb_: " << Rwb_.coeffs().transpose();
            LOG(INFO) << "twb_: " << twb_.transpose();
            LOG(INFO) << "bg_: " << bg_.transpose();
        } else {
            LOG(ERROR) << "未知状态类型";
            exit(0);
        }

        // 3
        // Update the camera states.
        // 更新相机姿态
        auto cam_state_iter = cam_states.begin();
        for (int i = 0; i < cam_states.size(); ++i, ++cam_state_iter)
        {
            const Eigen::VectorXd &delta_x_cam = X.segment<6>(param_ptr->STATE_DIM + i * 6);
            cam_state_iter->second->Rwc_ =
                Converter::so3ToQuat(delta_x_cam.head<3>()) * cam_state_iter->second->Rwc_;
            cam_state_iter->second->twc_ += delta_x_cam.tail<3>();
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

        // 优化模式下状态删除单独维护
        if (param_ptr_->fusion_model_ == 1)
            return true;
        // 只保留最近5s状态,二分查找下，从头找也行不差很多
        std::shared_ptr<State> tar = std::make_shared<State>();
        tar->time_ = state->time_ - 5.0;
        auto iter = lower_bound(states_.begin(), states_.end(), tar, CompareTime());
        if (iter != states_.begin())
            states_ = std::vector<std::shared_ptr<State>>(iter, states_.end());
        return true;
    }

    inline bool PopFrontState() {
        std::unique_lock<std::mutex> lock(states_mtx_);
        if (states_.empty())
            return false;
        states_.erase(states_.begin());
        if (!states_.empty()) {
            states_[0]->last_state_ = nullptr;
            states_[0]->preint_ = nullptr;
        }
        return true;
    }

    inline std::vector<std::shared_ptr<State>> GetAllStates() {
        std::unique_lock<std::mutex> lock(states_mtx_);
        return states_;
    }

    inline bool Empty() {
        std::unique_lock<std::mutex> lock(states_mtx_);
        return states_.empty();
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