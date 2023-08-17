#pragma once
#include <iostream>
#include <math.h>
#include <fstream>

#include <eigen3/Eigen/Core>
#include <opencv2/opencv.hpp>

struct InputData {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    double time_ = -1.0;
    // 0 imu 1 wheel 2 gps 3 camera
    int data_type_ = -1;
    // imu m/s^2  rad/s
    Eigen::Vector3d a_, w_;

    // wheel
    double lv_, rv_;

    // gps
    double lat_, lat_error_;
    double lon_, lon_error_;
    double h_, h_error_;

    // camera
    cv::Mat image_;
};

class DataLoader {
public:
    DataLoader(const std::string & data_path);
    bool ReadIMU(const std::string & path);
    bool ReadWheel(const std::string & path);
    bool ReadGPS(const std::string & path);
    bool ReadImage(const std::string & path);

    InputData GetNextData();
private:
    std::queue<InputData> datas_;
    double d2r = 0.017453292519943295;
};