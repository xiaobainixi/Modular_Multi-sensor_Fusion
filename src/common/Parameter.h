#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <memory>

#include <glog/logging.h>
#include <Eigen/Core>
#include <opencv2/opencv.hpp>

#include "GlobalDefination.h"
class Parameter
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Parameter(const std::string &file)
    {
        cv::FileStorage f_settings;
        std::string global_file_path = WORK_SPACE_PATH + "/config/" + file;
        LOG(INFO) << "config path: " << global_file_path;
        if (file.empty() || !f_settings.open(global_file_path, cv::FileStorage::READ))
        {
            LOG(ERROR) << "Can not open config file: " << global_file_path;
            exit(0);
        }

        cv::FileNode node = f_settings["fusion_model"];
        if (!node.empty() && node.isInt())
            fusion_model_ = int(node.real());
        LOG(INFO) << "fusion_model: " << fusion_model_;

        // 推出预测方式与状态
        node = f_settings["use_imu"];
        if (!node.empty() && node.isInt())
            use_imu_ = int(node.real()) == 1;
        LOG(INFO) << "use_imu: " << use_imu_;

        // wheel_use_type_ 为1时表示作为预测，为2时表示作为观测，为其他时表示不使用
        node = f_settings["wheel_use_type"];
        if (!node.empty() && node.isInt())
            wheel_use_type_ = int(node.real());
        LOG(INFO) << "wheel_use_type: " << wheel_use_type_;

        if (!use_imu_ && wheel_use_type_ != 1)
        {
            LOG(ERROR) << "至少选择一个传感器作为预测";
            exit(0);
        }
        else if (!use_imu_ && wheel_use_type_ == 1)
        {
            state_type_ = 1;
            STATE_DIM = 6;
            POSI_INDEX = 0;
            ORI_INDEX_STATE_ = 3;
        }

        // 观测
        node = f_settings["use_gps"];
        if (!node.empty() && node.isInt())
            use_gps_ = int(node.real()) == 1;
        LOG(INFO) << "use_gps: " << use_gps_;

        node = f_settings["use_camera"];
        if (!node.empty() && node.isInt())
            use_camera_ = int(node.real()) == 1;
        LOG(INFO) << "use_camera: " << use_camera_;

        node = f_settings["gyro_noise"];
        if (!node.empty() && node.isReal())
            gyro_noise_ = node.real();
        LOG(INFO) << "gyro_noise: " << gyro_noise_;

        node = f_settings["gyro_bias_noise"];
        if (!node.empty() && node.isReal())
            gyro_bias_noise_ = node.real();
        LOG(INFO) << "gyro_bias_noise: " << gyro_bias_noise_;

        node = f_settings["acc_noise"];
        if (!node.empty() && node.isReal())
            acc_noise_ = node.real();
        LOG(INFO) << "acc_noise: " << acc_noise_;

        node = f_settings["acc_bias_noise"];
        if (!node.empty() && node.isReal())
            acc_bias_noise_ = node.real();
        LOG(INFO) << "acc_bias_noise: " << acc_bias_noise_;

        node = f_settings["gps_x_noise"];
        if (!node.empty() && node.isReal())
            gps_x_noise_ = node.real();
        LOG(INFO) << "gps_x_noise: " << gps_x_noise_;

        node = f_settings["gps_y_noise"];
        if (!node.empty() && node.isReal())
            gps_y_noise_ = node.real();
        LOG(INFO) << "gps_y_noise: " << gps_y_noise_;

        node = f_settings["gps_z_noise"];
        if (!node.empty() && node.isReal())
            gps_z_noise_ = node.real();
        LOG(INFO) << "gps_z_noise: " << gps_z_noise_;

        node = f_settings["wheel_kl"];
        if (!node.empty() && node.isReal())
            wheel_kl_ = node.real();
        LOG(INFO) << "wheel_kl: " << wheel_kl_;

        node = f_settings["wheel_kr"];
        if (!node.empty() && node.isReal())
            wheel_kr_ = node.real();
        LOG(INFO) << "wheel_kr: " << wheel_kr_;

        node = f_settings["wheel_b"];
        if (!node.empty() && node.isReal())
            wheel_b_ = node.real();
        LOG(INFO) << "wheel_b: " << wheel_b_;

        node = f_settings["wheel_noise_factor"];
        if (!node.empty() && node.isReal())
            wheel_noise_factor_ = node.real();
        LOG(INFO) << "wheel_noise_factor: " << wheel_noise_factor_;

        node = f_settings["encoder_resolution"];
        if (!node.empty() && node.isReal())
            encoder_resolution_ = node.real();
        LOG(INFO) << "encoder_resolution: " << encoder_resolution_;

        node = f_settings["wheel_x_noise"];
        if (!node.empty() && node.isReal())
            wheel_x_noise_ = node.real();
        LOG(INFO) << "wheel_x_noise: " << wheel_x_noise_;

        node = f_settings["wheel_y_noise"];
        if (!node.empty() && node.isReal())
            wheel_y_noise_ = node.real();
        LOG(INFO) << "wheel_y_noise: " << wheel_y_noise_;

        node = f_settings["wheel_z_noise"];
        if (!node.empty() && node.isReal())
            wheel_z_noise_ = node.real();
        LOG(INFO) << "wheel_z_noise: " << wheel_z_noise_;

        node = f_settings["fix_yz_in_eskf"];
        if (!node.empty() && node.isInt())
            fix_yz_in_eskf_ = int(node.real()) == 1;
        LOG(INFO) << "fix_yz_in_eskf: " << fix_yz_in_eskf_;

        node = f_settings["play_speed"];
        if (!node.empty() && node.isReal())
            play_speed_ = node.real();
        LOG(INFO) << "play_speed: " << play_speed_;

        node = f_settings["data_path"];
        if (!node.empty() && node.isString())
            data_path_ = node.string();
        LOG(INFO) << "data folder: " << data_path_;

        imu_dispersed_noise_cov_ =
            Eigen::Matrix<double, 12, 12>::Zero();
        imu_dispersed_noise_cov_.block<3, 3>(0, 0) =
            Eigen::Matrix3d::Identity() * gyro_noise_ * gyro_noise_;
        imu_dispersed_noise_cov_.block<3, 3>(3, 3) =
            Eigen::Matrix3d::Identity() * gyro_bias_noise_ * gyro_bias_noise_;
        imu_dispersed_noise_cov_.block<3, 3>(6, 6) =
            Eigen::Matrix3d::Identity() * acc_noise_ * acc_noise_;
        imu_dispersed_noise_cov_.block<3, 3>(9, 9) =
            Eigen::Matrix3d::Identity() * acc_bias_noise_ * acc_bias_noise_;
    }

    void ConfigureStatusDim(int type)
    {
        // 除非特殊定义，否则默认只使用IMU
        if (type == 1)
        {
        }
    }

    // 0 for kf 1 for ba
    int fusion_model_ = 0;
    bool use_imu_ = true;
    // 0 unused 1 predict 2 obs
    int wheel_use_type_ = 0;
    bool use_gps_ = false;
    bool use_camera_ = false;

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
    Eigen::Matrix<double, 12, 12> imu_dispersed_noise_cov_;

    // wheel
    // note: new param for wheel
    double wheel_kl_ = 0.00047820240382508;
    double wheel_kr_ = 0.00047768621928995;
    double wheel_b_ = 1.52439;
    double wheel_noise_factor_ = 0.2;
    // note: old param for wheel
    double encoder_resolution_ = 0.00047820240382508;
    double wheel_x_noise_ = 0.001;
    double wheel_y_noise_ = 0.001;
    double wheel_z_noise_ = 0.001;
    bool fix_yz_in_eskf_ = false;

    // Camera
    std::string cam_distortion_model;
    cv::Vec4d cam_intrinsics;
    cv::Vec4d cam_distortion_coeffs;

    cv::Matx33d R_cam_imu;
    cv::Vec3d t_cam_imu;

    // dataloader
    double play_speed_ = 6.0;
    std::string data_path_ = "../data/";
};