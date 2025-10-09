#pragma once

#include <iostream>
#include <fstream>
#include <memory>
#include <mutex>
#include <Eigen/Core>
#include <Eigen/Dense>
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

struct GNSSData {
    double time_ = -1.0;
    double lat_;
    double lon_;
    double h_;
    double x_, y_, z_;
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
    Eigen::Quaterniond Rwc_ = Eigen::Quaterniond::Identity();
    Eigen::Vector3d twb_ = Eigen::Vector3d::Zero();
    Eigen::Quaterniond Rwb_ = Eigen::Quaterniond::Identity();
    Eigen::MatrixXd C_;
};

template<typename T>
bool FindNearestPair(std::deque<T>& deque, const double& time,
    typename std::deque<T>::iterator& left,
    typename std::deque<T>::iterator& right,
    double max_interval = 0.05)
{
    if (deque.empty()) return false;
    right = std::lower_bound(deque.begin(), deque.end(), time, [](const T& a, double b){ return a.time_ < b; });
    // 边界情况：time等于第一个元素
    if (right == deque.begin()) {
        left = right;
        // 只返回第一个元素本身
        if (right->time_ != time) return false;
        return true;
    }
    // 边界情况：time等于最后一个元素
    if (right == deque.end()) {
        left = right - 1;
        if ((right - 1)->time_ != time) return false;
        right = right - 1;
        return true;
    }
    left = right - 1;
    if ((right->time_ - left->time_) > max_interval) return false;
    return true;
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

    void Input(const GNSSData & gnss_data) {
        std::unique_lock<std::mutex> lock(gnss_datas_mtx_);
        if (!gnss_datas_.empty() && gnss_datas_[gnss_datas_.size() - 1].time_ >= gnss_data.time_)
            return;
        gnss_datas_.push_back(gnss_data);

        if (gnss_datas_.size() > 1000)
            gnss_datas_.pop_front();
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

    bool GetLastGNSSData(GNSSData & gnss_data, double last_data_time = -1.0) {
        std::unique_lock<std::mutex> lock(gnss_datas_mtx_);
        if (gnss_datas_.empty() || gnss_datas_.back().time_ <= last_data_time)
            return false;
        gnss_data = gnss_datas_.back();
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

    // IMUData插值函数
    IMUData InterpolateIMU(const IMUData& left, const IMUData& right, double time) {
        double ratio = (time - left.time_) / (right.time_ - left.time_);
        IMUData result;
        result.time_ = time;
        result.a_ = left.a_ + (right.a_ - left.a_) * ratio;
        result.w_ = left.w_ + (right.w_ - left.w_) * ratio;
        return result;
    }

    // WheelData插值函数
    WheelData InterpolateWheel(const WheelData& left, const WheelData& right, double time) {
        double ratio = (time - left.time_) / (right.time_ - left.time_);
        WheelData result;
        result.time_ = time;
        result.lv_ = left.lv_ + (right.lv_ - left.lv_) * ratio;
        result.rv_ = left.rv_ + (right.rv_ - left.rv_) * ratio;
        return result;
    }

    WheelIMUData InterpolateWheelIMU(const WheelIMUData& left, const WheelIMUData& right, double time) {
        double ratio = (time - left.time_) / (right.time_ - left.time_);
        WheelIMUData result;
        result.time_ = time;
        result.a_ = left.a_ + (right.a_ - left.a_) * ratio;
        result.w_ = left.w_ + (right.w_ - left.w_) * ratio;
        result.lv_ = left.lv_ + (right.lv_ - left.lv_) * ratio;
        result.rv_ = left.rv_ + (right.rv_ - left.rv_) * ratio;
        return result;
    }

    // 修改后的 GetDatasBetween
    bool GetDatasBetween(std::vector<IMUData>& datas, const double& start, const double& end, double max_interval = 0.05) {
        std::unique_lock<std::mutex> lock(imu_datas_mtx_);
        if (imu_datas_.empty() || imu_datas_.front().time_ > start || imu_datas_.back().time_ < end)
            return false;

        std::deque<IMUData>::iterator start_left, start_right, end_left, end_right;
        if (!FindNearestPair(imu_datas_, start, start_left, start_right, max_interval)) return false;
        if (!FindNearestPair(imu_datas_, end, end_left, end_right, max_interval)) return false;

        datas.clear();
        datas.push_back(InterpolateIMU(*start_left, *start_right, start));
        for (auto it = start_right; it != end_left + 1; ++it)
            datas.push_back(*it);
        datas.push_back(InterpolateIMU(*end_left, *end_right, end));
        return true;
    }

    bool GetDatasBetween(std::vector<WheelData>& datas, const double& start, const double& end, double max_interval = 0.05) {
        std::unique_lock<std::mutex> lock(wheel_datas_mtx_);
        if (wheel_datas_.empty() || wheel_datas_.front().time_ > start || wheel_datas_.back().time_ < end)
            return false;

        std::deque<WheelData>::iterator start_left, start_right, end_left, end_right;
        if (!FindNearestPair(wheel_datas_, start, start_left, start_right, max_interval)) return false;
        if (!FindNearestPair(wheel_datas_, end, end_left, end_right, max_interval)) return false;

        datas.clear();
        datas.push_back(InterpolateWheel(*start_left, *start_right, start));
        for (auto it = start_right; it != end_left + 1; ++it)
            datas.push_back(*it);
        datas.push_back(InterpolateWheel(*end_left, *end_right, end));
        return true;
    }

    bool GetDatasBetween(std::vector<WheelIMUData>& datas, const double& start, const double& end, double max_interval = 0.05) {
        std::unique_lock<std::mutex> lock(wheel_imu_datas_mtx_);
        if (imu_wheel_datas_.empty() || imu_wheel_datas_.front().time_ > start || imu_wheel_datas_.back().time_ < end)
            return false;

        std::deque<WheelIMUData>::iterator start_left, start_right, end_left, end_right;
        if (!FindNearestPair(imu_wheel_datas_, start, start_left, start_right, max_interval)) return false;
        if (!FindNearestPair(imu_wheel_datas_, end, end_left, end_right, max_interval)) return false;

        datas.clear();
        datas.push_back(InterpolateWheelIMU(*start_left, *start_right, start));
        for (auto it = start_right; it != end_left + 1; ++it)
            datas.push_back(*it);
        datas.push_back(InterpolateWheelIMU(*end_left, *end_right, end));
        return true;
    }
private:
    std::mutex imu_datas_mtx_, wheel_datas_mtx_, wheel_imu_datas_mtx_, gnss_datas_mtx_, camera_data_mtx_, feature_data_mtx_;
    std::deque<IMUData> imu_datas_;
    std::deque<WheelData> wheel_datas_;
    std::deque<WheelIMUData> imu_wheel_datas_;
    std::deque<GNSSData> gnss_datas_;
    CameraData camera_data_;
    FeatureData feature_data_;

    std::shared_ptr<Parameter> param_ptr_;
};