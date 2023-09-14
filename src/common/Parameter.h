#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <Eigen/Core>

class Parameter {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Parameter(const std::string & file) {
        imu_continuous_noise_cov_ =
            Eigen::Matrix<double, 12, 12>::Zero();
        imu_continuous_noise_cov_.block<3, 3>(0, 0) =
            Eigen::Matrix3d::Identity() * gyro_noise_ * gyro_noise_;
        imu_continuous_noise_cov_.block<3, 3>(3, 3) =
            Eigen::Matrix3d::Identity() * gyro_bias_noise_ * gyro_bias_noise_;
        imu_continuous_noise_cov_.block<3, 3>(6, 6) =
            Eigen::Matrix3d::Identity() * acc_noise_ * acc_noise_;
        imu_continuous_noise_cov_.block<3, 3>(9, 9) =
            Eigen::Matrix3d::Identity() * acc_bias_noise_ * acc_bias_noise_;
            std::cout << imu_continuous_noise_cov_ << std::endl;
    }
    

    void ConfigureStatusDim(int type) {
        // 除非特殊定义，否则默认只使用IMU
        if (type == 1) {

        }
    }

    // 不同模式状态不同，默认是纯IMU做预测
    int state_type_ = 0;
    int STATE_DIM = 15;
    int POSI_INDEX = 0;
    int VEL_INDEX_STATE_ = 3;
    int ORI_INDEX_STATE_ = 6;
    int GYRO_BIAS_INDEX_STATE_ = 9;
    int ACC_BIAS_INDEX_STATE_ = 12;

    // imu
    double gyro_noise_ = 2.6e-03;
    double gyro_bias_noise_ = 2.6e-04;
    double acc_noise_ = 2.6e-02;
    double acc_bias_noise_ = 2.6e-03;

    // gps
    double gps_x_noise_ = 0.001;
    double gps_y_noise_ = 0.001;
    double gps_z_noise_ = 0.001;
    Eigen::Matrix<double, 12, 12> imu_continuous_noise_cov_;

    // wheel
    double encoder_resolution_ = 0.00047820240382508;
    double wheel_x_noise_ = 0.001;
    double wheel_y_noise_ = 0.001;
    double wheel_z_noise_ = 0.001;
    bool fix_yz_in_eskf_ = false;


    // dataloader
    double play_speed_ = 6.0;
};