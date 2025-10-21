#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <memory>

#include <glog/logging.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>

#include "GlobalDefination.h"

/**
 * @brief OptimizationConfig Configuration parameters
 *    for 3d feature position optimization.
 * 优化参数
 */
struct MSCKFOptimizationConfig
{
    // 位移是否足够，用于判断点是否能做三角化
    double translation_threshold;
    // huber参数
    double huber_epsilon;
    // 修改量阈值，优化的每次迭代都会有更新量，这个量如果太小则表示与目标值接近
    double estimation_precision;
    // LM算法lambda的初始值
    double initial_damping;

    // 内外轮最大迭代次数
    int outer_loop_max_iteration;
    int inner_loop_max_iteration;

    MSCKFOptimizationConfig()
        : translation_threshold(0.2),
            huber_epsilon(0.01),
            estimation_precision(5e-7),
            initial_damping(1e-3),
            outer_loop_max_iteration(10),
            inner_loop_max_iteration(10)
    {
        return;
    }
};
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

        // 观测
        node = f_settings["use_gnss"];
        if (!node.empty() && node.isInt())
            use_gnss_ = int(node.real()) == 1;
        LOG(INFO) << "use_gnss: " << use_gnss_;

        node = f_settings["gnss_wheel_align"];
        if (!node.empty() && node.isInt())
            gnss_wheel_align_ = int(node.real()) == 1;
        if (wheel_use_type_ != 2 || !use_gnss_) {
            gnss_wheel_align_ = false;
            LOG(INFO) << "不使用轮速或gnss作为观测, gnss与轮速不会对齐使用, 如果想对齐观测, 请同时打开gnss与wheel观测";
        }
        LOG(INFO) << "gnss_wheel_align: " << gnss_wheel_align_;

        

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

        node = f_settings["gnss_x_noise"];
        if (!node.empty() && node.isReal())
            gnss_x_noise_ = node.real();
        LOG(INFO) << "gnss_x_noise: " << gnss_x_noise_;

        node = f_settings["gnss_y_noise"];
        if (!node.empty() && node.isReal())
            gnss_y_noise_ = node.real();
        LOG(INFO) << "gnss_y_noise: " << gnss_y_noise_;

        node = f_settings["gnss_z_noise"];
        if (!node.empty() && node.isReal())
            gnss_z_noise_ = node.real();
        LOG(INFO) << "gnss_z_noise: " << gnss_z_noise_;

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

        node = f_settings["wheel_vel_noise"];
        if (!node.empty() && node.isReal())
            wheel_vel_noise_ = node.real();
        LOG(INFO) << "wheel_vel_noise: " << wheel_vel_noise_;

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

        node = f_settings["data_type"];
        if (!node.empty() && node.isString())
            data_type_ = node.string();
        LOG(INFO) << "data type: " << data_type_;

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

            double w_noise = wheel_vel_noise_ / wheel_b_;
            predict_dispersed_noise_cov_ =
                Eigen::Matrix<double, 6, 6>::Zero();
            predict_dispersed_noise_cov_(0, 0) = wheel_vel_noise_ * wheel_vel_noise_;
            // 考虑wheel_b_ 误差?
            predict_dispersed_noise_cov_(5, 5) = w_noise * w_noise;
        }
        else if (use_imu_ && wheel_use_type_ == 1)
        {
            state_type_ = 2;
            STATE_DIM = 9;
            POSI_INDEX = 0;
            ORI_INDEX_STATE_ = 3;
            GYRO_BIAS_INDEX_STATE_ = 6;

            predict_dispersed_noise_cov_ =
                Eigen::Matrix<double, 9, 9>::Zero();
            predict_dispersed_noise_cov_.block<3, 3>(0, 0) =
                Eigen::Matrix3d::Identity() * gyro_noise_ * gyro_noise_;
            predict_dispersed_noise_cov_.block<3, 3>(3, 3) =
                Eigen::Matrix3d::Identity() * gyro_bias_noise_ * gyro_bias_noise_;
            predict_dispersed_noise_cov_(6, 6) = wheel_vel_noise_ * wheel_vel_noise_;
        } else {
            predict_dispersed_noise_cov_ =
                Eigen::Matrix<double, 12, 12>::Zero();
            predict_dispersed_noise_cov_.block<3, 3>(0, 0) =
                Eigen::Matrix3d::Identity() * gyro_noise_ * gyro_noise_;
            predict_dispersed_noise_cov_.block<3, 3>(3, 3) =
                Eigen::Matrix3d::Identity() * gyro_bias_noise_ * gyro_bias_noise_;
            predict_dispersed_noise_cov_.block<3, 3>(6, 6) =
                Eigen::Matrix3d::Identity() * acc_noise_ * acc_noise_;
            predict_dispersed_noise_cov_.block<3, 3>(9, 9) =
                Eigen::Matrix3d::Identity() * acc_bias_noise_ * acc_bias_noise_;
        }

        node = f_settings["visual_observation_noise"];
        if (!node.empty() && node.isReal())
            visual_observation_noise_ = node.real();
        LOG(INFO) << "visual_observation_noise: " << visual_observation_noise_;
        node = f_settings["camera_fx"];
        if (!node.empty() && node.isReal())
            cam_intrinsics_[0] = node.real();
        node = f_settings["camera_fy"];
        if (!node.empty() && node.isReal())
            cam_intrinsics_[1] = node.real();
        node = f_settings["camera_cx"];
        if (!node.empty() && node.isReal())
            cam_intrinsics_[2] = node.real();
        node = f_settings["camera_cy"];
        if (!node.empty() && node.isReal())
            cam_intrinsics_[3] = node.real();

        node = f_settings["camera_k1"];
        if (!node.empty() && node.isReal())
            cam_distortion_coeffs_[0] = node.real();
        node = f_settings["camera_k2"];
        if (!node.empty() && node.isReal())
            cam_distortion_coeffs_[1] = node.real();
        node = f_settings["camera_p1"];
        if (!node.empty() && node.isReal())
            cam_distortion_coeffs_[2] = node.real();
        node = f_settings["camera_p2"];
        if (!node.empty() && node.isReal())
            cam_distortion_coeffs_[3] = node.real();

        cv::Mat Rbc_mat, tbc_mat;
        f_settings["Rbc"] >> Rbc_mat;
        f_settings["tbc"] >> tbc_mat;
        cv2eigen(Rbc_mat, Rbc_);
        cv2eigen(tbc_mat, tbc_);
        // todo
        // node = f_settings["camera_k3"];
        // if (!node.empty() && node.isReal())
        //     cam_distortion_coeffs_ = node.real();

        MIN_PARALLAX = f_settings["keyframe_parallax"]; // 根据视差确定关键帧
        MIN_PARALLAX = MIN_PARALLAX / FOCAL_LENGTH;
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
    bool use_imu_ = false;
    // 0 unused 1 predict 2 obs
    int wheel_use_type_ = 0;
    bool gnss_wheel_align_ = false;
    bool use_gnss_ = false;
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

    // gnss
    double gnss_x_noise_ = 0.001;
    double gnss_y_noise_ = 0.001;
    double gnss_z_noise_ = 0.001;
    Eigen::MatrixXd predict_dispersed_noise_cov_;

    // wheel
    // note: new param for wheel
    double wheel_kl_ = 0.00047820240382508;
    double wheel_kr_ = 0.00047768621928995;
    double wheel_b_ = 1.52439;
    double wheel_vel_noise_ = 0.2;  // wheel的速度噪声
    bool fix_yz_in_eskf_ = false;

    // Camera
    int WINDOW_SIZE = 20;
    MSCKFOptimizationConfig msckf_optimization_config_;
    double visual_observation_noise_ = 0.01;
    std::string cam_distortion_model_;
    cv::Vec4d cam_intrinsics_;
    cv::Vec4d cam_distortion_coeffs_;

    Eigen::Matrix3d Rbc_;
    Eigen::Vector3d tbc_;

    // VINS
    const double FOCAL_LENGTH = 460.0;
    double MIN_PARALLAX;

    // dataloader
    double play_speed_ = 6.0;
    std::string data_path_ = "../data/";
    std::string data_type_ = " ";


    double g_ = 9.81;
    Eigen::Vector3d gw_ = Eigen::Vector3d(0.0, 0.0, -g_);
};