#pragma once
#include <iostream>
#include <math.h>
#include <fstream>
#include <sys/time.h>
#include <unistd.h>

#include <eigen3/Eigen/Core>
#include <opencv2/opencv.hpp>

#include "common/Parameter.h"

struct InputData {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    double time_ = -1.0;
    // 0 imu 1 wheel 2 gnss 3 camera
    int data_type_ = -1;
    // imu m/s^2  rad/s
    Eigen::Vector3d a_, w_;

    // wheel
    double lv_, rv_;

    // gnss
    double lat_, lat_error_;
    double lon_, lon_error_;
    double h_, h_error_;

    // camera
    std::string img_path_;
    // cv::Mat image_;
};

class DataLoader {
public:
    DataLoader(const std::shared_ptr<Parameter> & param_ptr);
    bool ReadIMU(const std::string & path);
    bool ReadWheel(const std::string & path);
    bool ReadGNSS(const std::string & path);
    bool ReadImage(const std::string & path);

    InputData GetNextData();
private:
    std::shared_ptr<Parameter> param_ptr_;
    std::queue<InputData> datas_, gnss_datas_, imu_datas_, image_datas_, wheel_datas_;
    double d2r_ = 0.017453292519943295;
    double last_data_time_ = -1.0;
    struct timeval t1_, t2_;
};