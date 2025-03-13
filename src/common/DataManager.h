#pragma once

#include <iostream>
#include <fstream>
#include <memory>
#include <mutex>
#include <Eigen/Core>
#include <opencv2/opencv.hpp>

#include "Parameter.h"
struct IMUData {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    double time_ = -1.0;
    // m/s^2  rad/s
    Eigen::Vector3d a_, w_;
};

struct WheelData {
    double time_ = -1.0;
    double lv_, rv_;
};

struct WheelIMUData {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    double time_ = -1.0;
    // m/s^2  rad/s
    Eigen::Vector3d a_, w_;
    double lv_, rv_;
};

struct GPSData {
    double time_ = -1.0;
    double lat_;
    double lon_;
    double h_;
};

struct CameraData {
    double time_ = -1.0;
    cv::Mat image_;
};

struct FeaturePoint {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Eigen::Vector2d point_;
    int id_ = -1;
};
struct FeatureData {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    double time_ = -1.0;
    std::vector<FeaturePoint> features_;
    Eigen::Vector3d twc_ = Eigen::Vector3d::Zero();
    Eigen::Matrix3d Rwc_ = Eigen::Matrix3d::Identity();
    Eigen::Vector3d twb_ = Eigen::Vector3d::Zero();
    Eigen::Matrix3d Rwb_ = Eigen::Matrix3d::Identity();
    Eigen::MatrixXd C_;
};

template<typename T>
typename std::deque<T>::iterator upperBoundDeque(std::deque<T>& deque, const double& time) {
    auto begin = deque.begin();
    auto end = deque.end();
    while (begin < end) {
        auto mid = begin + std::distance(begin, end) / 2;
        if ((*mid).time_ <= time) {
            begin = mid + 1;
        } else {
            end = mid;
        }
    }
    return begin;
}

class DataManager {
public:
    DataManager(const std::shared_ptr<Parameter>& param_ptr) { param_ptr_ = param_ptr; }
    void Input(const IMUData & imu_data) {
        std::unique_lock<std::mutex> lock(imu_datas_mtx_);
        if (!imu_datas_.empty() && imu_datas_[imu_datas_.size() - 1].time_ >= imu_data.time_)
            return;
        imu_datas_.push_back(imu_data);

        if (imu_datas_.size() > 1000)
            imu_datas_.pop_front();
    }

    void Input(const WheelData & wheel_data) {
        {
            std::unique_lock<std::mutex> lock(wheel_datas_mtx_);
            if (!wheel_datas_.empty() && wheel_datas_[wheel_datas_.size() - 1].time_ >= wheel_data.time_)
                return;
            wheel_datas_.push_back(wheel_data);

            if (wheel_datas_.size() > 1000)
                wheel_datas_.pop_front();
        }

        // 应该用高频去差值对齐低频，kaist数据集差不多就用imu差值吧
        if (param_ptr_->use_imu_ && param_ptr_->wheel_use_type_ == 1 && imu_datas_.size() > 1) {
            std::deque<IMUData>::iterator imu_data_iter;
            {
                std::unique_lock<std::mutex> lock(imu_datas_mtx_);
                imu_data_iter = std::lower_bound(imu_datas_.begin(), imu_datas_.end(), wheel_data.time_, [](IMUData a, double b) { return a.time_ < b; });
                if (imu_data_iter == imu_datas_.begin())
                    return;
            }
            auto imu_data1 = imu_data_iter == imu_datas_.end() ? *(imu_data_iter - 2) : *(imu_data_iter - 1);
            auto imu_data2 = imu_data_iter == imu_datas_.end() ? *(imu_data_iter - 1) : *imu_data_iter;

            double delta_imu_time = imu_data2.time_ - imu_data1.time_;
            double delta_wheel_imu_time = imu_data2.time_ - wheel_data.time_;
            if (abs(delta_imu_time) < abs(delta_wheel_imu_time))
                return;
            
            WheelIMUData wheel_imu_data;
            wheel_imu_data.time_ = wheel_data.time_;
            wheel_imu_data.a_ = imu_data1.a_ + (imu_data2.a_ - imu_data1.a_) * (wheel_data.time_ - imu_data1.time_) / delta_imu_time;
            wheel_imu_data.w_ = imu_data1.w_ + (imu_data2.w_ - imu_data1.w_) * (wheel_data.time_ - imu_data1.time_) / delta_imu_time;
            wheel_imu_data.lv_ = wheel_data.lv_;
            wheel_imu_data.rv_ = wheel_data.rv_;
            // LOG(INFO) << std::to_string(wheel_data.time_) << " " << std::to_string(imu_data2.time_) << " "
            //         << std::to_string(imu_data1.time_) << " " << std::to_string((imu_datas_.end() - 1)->time_);
            {
                std::unique_lock<std::mutex> lock(wheel_imu_datas_mtx_);
                imu_wheel_datas_.push_back(wheel_imu_data);
                

                if (imu_wheel_datas_.size() > 1000)
                    imu_wheel_datas_.pop_front();
            }
                
        }        
    }

    void Input(const GPSData & gps_data) {
        std::unique_lock<std::mutex> lock(gps_datas_mtx_);
        if (!gps_datas_.empty() && gps_datas_[gps_datas_.size() - 1].time_ >= gps_data.time_)
            return;
        gps_datas_.push_back(gps_data);

        if (gps_datas_.size() > 1000)
            gps_datas_.pop_front();
    }

    void Input(const CameraData & camera_data) {
        if (camera_data_.time_ > 0.0 && camera_data_.time_ >= camera_data.time_)
            return;

        std::unique_lock<std::mutex> lock(camera_data_mtx_);
        camera_data_ = camera_data;
    }

    void Input(const FeatureData & feature_data) {
        if (feature_data_.time_ > 0.0 && feature_data_.time_ >= feature_data.time_)
            return;

        std::unique_lock<std::mutex> lock(feature_data_mtx_);
        feature_data_ = feature_data;
    }

    bool GetLastIMUData(IMUData & imu_data, double last_data_time = -1.0) {
        std::unique_lock<std::mutex> lock(imu_datas_mtx_);
        if (imu_datas_.empty() || imu_datas_.back().time_ <= last_data_time)
            return false;
        imu_data = imu_datas_.back();
        return true;
    }

    bool GetLastGPSData(GPSData & gps_data, double last_data_time = -1.0) {
        std::unique_lock<std::mutex> lock(gps_datas_mtx_);
        if (gps_datas_.empty() || gps_datas_.back().time_ <= last_data_time)
            return false;
        gps_data = gps_datas_.back();
        return true;
    }

    bool GetLastWheelData(WheelData & wheel_data, double last_data_time = -1.0) {
        std::unique_lock<std::mutex> lock(wheel_datas_mtx_);
        if (wheel_datas_.empty() || wheel_datas_.back().time_ <= last_data_time)
            return false;
        wheel_data = wheel_datas_.back();
        return true;
    }

    bool GetLastWheelIMUData(WheelIMUData & wheel_imu_data, double last_data_time = -1.0) {
        std::unique_lock<std::mutex> lock(wheel_imu_datas_mtx_);
        if (imu_wheel_datas_.empty() || imu_wheel_datas_.back().time_ <= last_data_time)
            return false;
        wheel_imu_data = imu_wheel_datas_.back();
        return true;
    }

    bool GetNewCameraData(CameraData & camera_data, double last_data_time = -1.0) {
        if (camera_data_.time_ <= last_data_time)
            return false;
        std::unique_lock<std::mutex> lock(camera_data_mtx_);
        camera_data = camera_data_;
        return true;
    }

    bool GetNewFeatureData(FeatureData & feature_data, double last_data_time = -1.0) {
        if (feature_data_.time_ <= last_data_time)
            return false;
        std::unique_lock<std::mutex> lock(feature_data_mtx_);
        feature_data = feature_data_;
        return true;
    }

    bool GetDatasBetween(std::vector<IMUData>& datas, const double& start, const double& end) {
        std::unique_lock<std::mutex> lock(imu_datas_mtx_);
        if (imu_datas_.back().time_ < end || imu_datas_.front().time_ > start)
            return false;

        auto begin_iter = upperBoundDeque(imu_datas_, start);
        begin_iter--;
        auto end_iter = upperBoundDeque(imu_datas_, end);
        end_iter++;
        datas = std::vector<IMUData>(begin_iter, end_iter);
    }
private:
    std::mutex imu_datas_mtx_, wheel_datas_mtx_, wheel_imu_datas_mtx_, gps_datas_mtx_, camera_data_mtx_, feature_data_mtx_;
    std::deque<IMUData> imu_datas_;
    std::deque<WheelData> wheel_datas_;
    std::deque<WheelIMUData> imu_wheel_datas_;
    std::deque<GPSData> gps_datas_;
    CameraData camera_data_;
    FeatureData feature_data_;

    std::shared_ptr<Parameter> param_ptr_;
};