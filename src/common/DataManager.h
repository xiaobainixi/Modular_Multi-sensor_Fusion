#pragma once

#include <iostream>
#include <fstream>
#include <memory>
#include <mutex>
#include <Eigen/Core>
#include <opencv2/opencv.hpp>

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

class DataManager {
public:
    void Input(const IMUData & imu_data) {
        std::unique_lock<std::mutex> lock(imu_datas_mtx_);
        if (!imu_datas_.empty() && imu_datas_[imu_datas_.size() - 1].time_ >= imu_data.time_)
            return;
        imu_datas_.push_back(imu_data);
    }

    void Input(const WheelData & wheel_data) {
        std::unique_lock<std::mutex> lock(wheel_datas_mtx_);
        if (!wheel_datas_.empty() && wheel_datas_[wheel_datas_.size() - 1].time_ >= wheel_data.time_)
            return;
        wheel_datas_.push_back(wheel_data);
    }

    void Input(const GPSData & gps_data) {
        std::unique_lock<std::mutex> lock(gps_datas_mtx_);
        if (!gps_datas_.empty() && gps_datas_[gps_datas_.size() - 1].time_ >= gps_data.time_)
            return;
        gps_datas_.push_back(gps_data);
    }

    void Input(const CameraData & camera_data) {
        std::unique_lock<std::mutex> lock(camera_datas_mtx_);
        if (!camera_datas_.empty() && camera_datas_[camera_datas_.size() - 1].time_ >= camera_data.time_)
            return;
        camera_datas_.push_back(camera_data);
    }

    bool GetLastIMUData(IMUData & imu_data, double last_data_time = -1.0) {
        std::unique_lock<std::mutex> lock(imu_datas_mtx_);
        if (imu_datas_.empty() || imu_datas_[imu_datas_.size() - 1].time_ <= last_data_time)
            return false;
        imu_data = imu_datas_[imu_datas_.size() - 1];
        return true;
    }

    bool GetLastGPSData(GPSData & gps_data, double last_data_time = -1.0) {
        std::unique_lock<std::mutex> lock(gps_datas_mtx_);
        if (gps_datas_.empty() || gps_datas_[gps_datas_.size() - 1].time_ <= last_data_time)
            return false;
        gps_data = gps_datas_[gps_datas_.size() - 1];
        return true;
    }

    bool GetLastWheelData(WheelData & wheel_data, double last_data_time = -1.0) {
        std::unique_lock<std::mutex> lock(wheel_datas_mtx_);
        if (wheel_datas_.empty() || wheel_datas_[wheel_datas_.size() - 1].time_ <= last_data_time)
            return false;
        wheel_data = wheel_datas_[wheel_datas_.size() - 1];
        return true;
    }

    // WheelData GetLastWheelData() {
    //     std::unique_lock<std::mutex> lock(wheel_datas_mtx_);
    //     return wheel_datas_.empty() ? WheelData() : wheel_datas_[wheel_datas_.size() - 1];
    // }

    // GPSData GetLastGPSData() {
    //     std::unique_lock<std::mutex> lock(gps_datas_mtx_);
    //     return gps_datas_.empty() ? GPSData() : gps_datas_[gps_datas_.size() - 1];
    // }

    CameraData GetLastCameraData() {
        std::unique_lock<std::mutex> lock(camera_datas_mtx_);
        return camera_datas_.empty() ? CameraData() : camera_datas_[camera_datas_.size() - 1];
    }
private:
    std::mutex imu_datas_mtx_, wheel_datas_mtx_, gps_datas_mtx_, camera_datas_mtx_;
    std::vector<IMUData> imu_datas_;
    std::vector<WheelData> wheel_datas_;
    std::vector<GPSData> gps_datas_;
    std::vector<CameraData> camera_datas_;
};