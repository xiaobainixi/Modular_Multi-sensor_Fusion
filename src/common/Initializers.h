#pragma once

#include <iostream>
#include <vector>
#include <memory>
#include <mutex>
#include <Eigen/Core>
#include <cmath>

#include "Parameter.h"
#include "DataManager.h"
#include "StateManager.h"
#include "Converter.h"
#include "CooTrans.h"

// 零速阈值, rad/s, m/s^2
static constexpr double ZERO_VELOCITY_GYR_THRESHOLD = 0.002;
static constexpr double ZERO_VELOCITY_ACC_THRESHOLD = 0.1;


class Initializers {
public:
    Initializers(std::shared_ptr<Parameter> param_ptr, std::shared_ptr<DataManager> data_manager_ptr,
        const std::shared_ptr<CooTrans> &coo_trans_ptr,
        std::shared_ptr<StateManager> state_manager_ptr) {
        param_ptr_ = param_ptr;
        data_manager_ptr_ = data_manager_ptr;
        state_manager_ptr_ = state_manager_ptr;
        coo_trans_ptr_ = coo_trans_ptr;
    }

    bool Initialization() {
        if (param_ptr_->use_gnss_)
            return GNSSInitialization();
        else if (param_ptr_->use_camera_)
            return VisualInitialization();
        else {
            LOG(INFO) << "没有可用于初始化的传感器，不做初始化";
            return true;
        }
    }
private:

    template <typename T>
    bool DetectZeroVelocity(const std::vector<T> &data_buffer, std::vector<double> &average) {

        auto size = static_cast<double>(data_buffer.size());
        double size_invert = 1.0 / size;

        double data_rate = size / (data_buffer[data_buffer.size() - 1].time_ - data_buffer[0].time_);

        double sum[6];
        double std[6];

        average.resize(6);
        average[0] = average[1] = average[2] = average[3] = average[4] = average[5] = 0;
        for (const auto &data : data_buffer) {
            average[0] += data.w_.x();
            average[1] += data.w_.y();
            average[2] += data.w_.z();
            average[3] += data.a_.x();
            average[4] += data.a_.y();
            average[5] += data.a_.z();
        }

        average[0] *= size_invert;
        average[1] *= size_invert;
        average[2] *= size_invert;
        average[3] *= size_invert;
        average[4] *= size_invert;
        average[5] *= size_invert;

        sum[0] = sum[1] = sum[2] = sum[3] = sum[4] = sum[5] = 0;
        for (const auto &data : data_buffer) {
            sum[0] += (data.w_.x() - average[0]) * (data.w_.x() - average[0]);
            sum[1] += (data.w_.y() - average[1]) * (data.w_.y() - average[1]);
            sum[2] += (data.w_.z() - average[2]) * (data.w_.z() - average[2]);
            sum[3] += (data.a_.x() - average[3]) * (data.a_.x() - average[3]);
            sum[4] += (data.a_.y() - average[4]) * (data.a_.y() - average[4]);
            sum[5] += (data.a_.z() - average[5]) * (data.a_.z() - average[5]);
        }

        // 速率形式
        std[0] = sqrt(sum[0] * size_invert) * data_rate;
        std[1] = sqrt(sum[1] * size_invert) * data_rate;
        std[2] = sqrt(sum[2] * size_invert) * data_rate;
        std[3] = sqrt(sum[3] * size_invert) * data_rate;
        std[4] = sqrt(sum[4] * size_invert) * data_rate;
        std[5] = sqrt(sum[5] * size_invert) * data_rate;

        if ((std[0] < ZERO_VELOCITY_GYR_THRESHOLD) && (std[1] < ZERO_VELOCITY_GYR_THRESHOLD) &&
            (std[2] < ZERO_VELOCITY_GYR_THRESHOLD) && (std[3] < ZERO_VELOCITY_ACC_THRESHOLD) &&
            (std[4] < ZERO_VELOCITY_ACC_THRESHOLD) && (std[5] < ZERO_VELOCITY_ACC_THRESHOLD)) {
            return true;
        }
        return false;
    }
    
    bool DetectZeroVelocity(const std::vector<WheelData> &data_buffer) {
        double data_number = static_cast<double>(data_buffer.size()) * 2.0;
        double vel_sum = 0.0;
        for (auto data : data_buffer) {
            vel_sum += (abs(data.lv_) + abs(data.rv_));
        }
    
        // 尽量速度快点，gnss远点
        if (vel_sum / data_number < 0.5)
            return true;
        return false;
    }


    bool GNSSInitialization() {
        GNSSData cur_gnss_data;
        if (!data_manager_ptr_->GetLastGNSSData(cur_gnss_data, last_gnss_data_.time_))
            return false;
        if (last_gnss_data_.time_ < 0.0) {
            coo_trans_ptr_->SetECEFOw(cur_gnss_data.lat_, cur_gnss_data.lon_, cur_gnss_data.h_);
            last_gnss_data_ = cur_gnss_data;
            return false;
        }

        // 零速检测估计陀螺零偏和横滚俯仰角
        // Obtain the gyroscope biases and roll and pitch angles
        std::vector<double> average;
        Eigen::Vector3d bg{0, 0, 0};
        Eigen::Vector3d initatt{0, 0, 0};
        Eigen::Vector3d velocity = Eigen::Vector3d::Zero();
        // bool is_has_zero_velocity = false;
        bool is_zero_velocity = false;

        if (param_ptr_->state_type_ != 1) {
            // 模式2可以考虑增加轮速判断
            std::vector<IMUData> datas;
            data_manager_ptr_->GetDatasBetween(datas, last_gnss_data_.time_, cur_gnss_data.time_);

            if (datas.size() < 50) {
                return false;
            }

            // 打印所有 IMUData
            // LOG(INFO) << "IMU datas: " << std::to_string(last_gnss_data_.time_) << " " << std::to_string(cur_gnss_data.time_);
            // for (const auto& d : datas) {
            //     LOG(INFO) << "t=" << std::to_string(d.time_) 
            //         << " a=[" << d.a_.transpose() << "] w=[" << d.w_.transpose() << "]";
            // }
            // 从零速开始
            is_zero_velocity = DetectZeroVelocity(datas, average);
            // 静止初始化
            // if (is_zero_velocity) {
            //     // 陀螺零偏
            //     bg = Eigen::Vector3d(average[0], average[1], average[2]);

            //     // 重力调平获取横滚俯仰角
            //     Eigen::Vector3d fb(average[3], average[4], average[5]);

            //     initatt[0] = -asin(fb[1] / param_ptr_->g_);
            //     initatt[1] = asin(fb[0] / param_ptr_->g_);

            //     LOG(INFO) << "Zero velocity get gyroscope bias " << bg.transpose() << ", roll " << initatt[0]
            //         << ", pitch " << initatt[1];
            //     is_has_zero_velocity = true;
            // }
        } else {
            std::vector<WheelData> datas;
            data_manager_ptr_->GetDatasBetween(datas, last_gnss_data_.time_, cur_gnss_data.time_);
            if (datas.size() < 50) {
                return false;
            }

            is_zero_velocity = DetectZeroVelocity(datas);
        }

        // 非零速状态
        // Initialization conditions
        if (!is_zero_velocity) {
            Eigen::Vector3d vel = coo_trans_ptr_->getENH(cur_gnss_data.lat_, cur_gnss_data.lon_, cur_gnss_data.h_)
                - coo_trans_ptr_->getENH(last_gnss_data_.lat_, last_gnss_data_.lon_, last_gnss_data_.h_);
            if (vel.norm() < 0.5) {
                LOG(INFO) << "速度太低 重新计算：" << vel.norm();
                // 重置
                last_gnss_data_.time_ = -1.0;
                return false;
            }
            velocity = vel / (cur_gnss_data.time_ - last_gnss_data_.time_);

            initatt[0] = 0;
            initatt[1] = atan(-vel.z() / sqrt(vel.x() * vel.x() + vel.y() * vel.y()));
            LOG(INFO) << "Initialized pitch from GNSS as " << initatt[1] * R2D << " deg";

            initatt[2] = atan2(vel.y(), vel.x());
            LOG(INFO) << "Initialized heading from GNSS as " << initatt[2] * R2D << " deg";

        } else {
            LOG(INFO) << "GNSS 初始化必须运动，现在处于静止，请运动";
            // 重置
            last_gnss_data_.time_ = -1.0;
            return false;
        }

        // 初始状态, 没有加杆臂！！！！ Converter::Euler2Matrix(initatt) * antlever_
        // The initialization cur_state_ptr
        std::shared_ptr<State> cur_state_ptr = std::make_shared<State>();
        cur_state_ptr->time_ = cur_gnss_data.time_;
        cur_state_ptr->Rwb_ = Converter::Euler2Matrix(initatt);
        cur_state_ptr->twb_ = coo_trans_ptr_->getENH(cur_gnss_data.lat_, cur_gnss_data.lon_, cur_gnss_data.h_);

        cur_state_ptr->C_ = Eigen::MatrixXd::Identity(param_ptr_->STATE_DIM, param_ptr_->STATE_DIM);
        cur_state_ptr->C_.block<3, 3>(param_ptr_->POSI_INDEX, param_ptr_->POSI_INDEX) = Eigen::Matrix3d::Identity() * 0.0025;
        cur_state_ptr->C_.block<3, 3>(param_ptr_->ORI_INDEX_STATE_, param_ptr_->ORI_INDEX_STATE_) = Eigen::Matrix3d::Identity() * 0.0025;
        if (param_ptr_->state_type_ == 0) {
            cur_state_ptr->bg_ = bg;
            cur_state_ptr->Vw_ = velocity;
            
            cur_state_ptr->C_.block<3, 3>(param_ptr_->VEL_INDEX_STATE_, param_ptr_->VEL_INDEX_STATE_) = Eigen::Matrix3d::Identity() * 0.0025;
            cur_state_ptr->C_.block<3, 3>(param_ptr_->GYRO_BIAS_INDEX_STATE_, param_ptr_->GYRO_BIAS_INDEX_STATE_) = Eigen::Matrix3d::Identity() * 0.01;
            cur_state_ptr->C_.block<3, 3>(param_ptr_->ACC_BIAS_INDEX_STATE_, param_ptr_->ACC_BIAS_INDEX_STATE_) = Eigen::Matrix3d::Identity() * 0.01;
        } else if (param_ptr_->state_type_ == 2) {
            cur_state_ptr->bg_ = bg;

            cur_state_ptr->C_.block<3, 3>(param_ptr_->GYRO_BIAS_INDEX_STATE_, param_ptr_->GYRO_BIAS_INDEX_STATE_) = Eigen::Matrix3d::Identity() * 0.01;
        }
        
        state_manager_ptr_->PushState(cur_state_ptr);
        LOG(INFO) << "GNSS 初始化完毕";
        LOG(INFO) << "初始化状态量：";
        LOG(INFO) << "时间: " << cur_state_ptr->time_;
        LOG(INFO) << "位姿: " << cur_state_ptr->twb_.transpose();
        LOG(INFO) << "速度: " << cur_state_ptr->Vw_.transpose();
        LOG(INFO) << "加速度零偏: " << cur_state_ptr->ba_.transpose();
        LOG(INFO) << "陀螺零偏: " << cur_state_ptr->bg_.transpose();
        LOG(INFO) << "旋转矩阵:\n" << cur_state_ptr->Rwb_;
        return true;
    }

    bool VisualInitialization() {
        LOG(INFO) << "视觉惯性初始化完毕";
        return true;
    }

    std::shared_ptr<DataManager> data_manager_ptr_;
    std::shared_ptr<StateManager> state_manager_ptr_;
    std::mutex states_mtx_;
    std::shared_ptr<Parameter> param_ptr_;

    // tmp data
    GNSSData last_gnss_data_;
    // WheelData last_wheel_data_;
    // FeatureData last_feature_data_;


    std::shared_ptr<CooTrans> coo_trans_ptr_;


    const double D2R = (M_PI / 180.0);
    const double R2D = (180.0 / M_PI);
};