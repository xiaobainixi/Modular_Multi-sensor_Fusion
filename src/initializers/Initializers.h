#pragma once

#include <iostream>
#include <vector>
#include <memory>
#include <mutex>
#include <Eigen/Core>
#include <cmath>

#include "common/Parameter.h"
#include "common/DataManager.h"
#include "common/StateManager.h"
#include "common/Converter.h"
#include "common/CooTrans.h"
#include "visual/VinsFeatureManager.h"
#include "observer/CameraObserver.h"
#include "predictor/IMUPredictor.h"

#include "initializers/Solve5pts.h"
#include "initializers/InitialSFM.h"
#include "initializers/InitialAlignment.h"

// 零速阈值, rad/s, m/s^2
static constexpr double ZERO_VELOCITY_GYR_THRESHOLD = 0.002;
static constexpr double ZERO_VELOCITY_ACC_THRESHOLD = 0.1;


class Initializers {
public:
    Initializers(std::shared_ptr<Parameter> param_ptr, std::shared_ptr<DataManager> data_manager_ptr,
        const std::shared_ptr<CooTrans> &coo_trans_ptr,
        std::shared_ptr<StateManager> state_manager_ptr,
        std::shared_ptr<Viewer> viewer_ptr = nullptr) {
        param_ptr_ = param_ptr;
        data_manager_ptr_ = data_manager_ptr;
        state_manager_ptr_ = state_manager_ptr;
        coo_trans_ptr_ = coo_trans_ptr;
        if (param_ptr->use_camera_) {
            // 只有纯IMU模式下才会用到视觉初始化，因为只有他使用了加速度计
            predictor_ptr_ = std::make_shared<IMUPredictor>(
                state_manager_ptr_, param_ptr_, data_manager_ptr_, viewer_ptr);
            vins_feature_manager_ptr_ = std::make_shared<VinsFeatureManager>(param_ptr_, viewer_ptr);
        }
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
    // MSCKF特征管理
    MapServer map_server_;
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
        LOG(INFO) << "位置: " << cur_state_ptr->twb_.transpose();
        LOG(INFO) << "速度: " << cur_state_ptr->Vw_.transpose();
        LOG(INFO) << "加速度零偏: " << cur_state_ptr->ba_.transpose();
        LOG(INFO) << "陀螺零偏: " << cur_state_ptr->bg_.transpose();
        LOG(INFO) << "旋转矩阵:\n" << cur_state_ptr->Rwb_;
        return true;
    }

    // VINS 视觉惯性初始化
    bool VisualInitialization() {
        if (param_ptr_->state_type_ != 0) {
            LOG(INFO) << "无需视觉惯性初始化";

            std::shared_ptr<State> cur_state_ptr = std::make_shared<State>();

            cur_state_ptr->C_ = Eigen::MatrixXd::Identity(param_ptr_->STATE_DIM, param_ptr_->STATE_DIM);
            cur_state_ptr->C_.block<3, 3>(param_ptr_->POSI_INDEX, param_ptr_->POSI_INDEX) = Eigen::Matrix3d::Identity() * 0.0025;
            cur_state_ptr->C_.block<3, 3>(param_ptr_->ORI_INDEX_STATE_, param_ptr_->ORI_INDEX_STATE_) = Eigen::Matrix3d::Identity() * 0.0025;
            if (param_ptr_->state_type_ == 2) {
                cur_state_ptr->C_.block<3, 3>(param_ptr_->GYRO_BIAS_INDEX_STATE_, param_ptr_->GYRO_BIAS_INDEX_STATE_) = Eigen::Matrix3d::Identity() * 0.01;
            }
            return true;
        }

        std::shared_ptr<State> last_state;
        FeatureData feature_data;
        if (data_manager_ptr_->GetNewFeatureData(feature_data, last_feature_data_.time_))
        {
            if(!state_manager_ptr_->GetNearestState(last_state)) {
                // 第一帧
                std::shared_ptr<State> cur_state_ptr = std::make_shared<State>();
                cur_state_ptr->time_ = feature_data.time_;
                state_manager_ptr_->PushState(cur_state_ptr);
                vins_feature_manager_ptr_->addFeatureCheckParallax(
                    0, feature_data.features_, cur_state_ptr);
                return false;
            }

            double dt = std::abs(last_state->time_ - feature_data.time_);
            if (dt > 0.08)
            {
                std::vector<IMUData> datas;
                data_manager_ptr_->GetDatasBetween(datas, last_state->time_, feature_data.time_);
                std::vector<double> average;
                // LOG(INFO) << "Visual Initialization between " << std::to_string(last_state->time_) 
                //     << " and " << std::to_string(feature_data.time_) << " with dt " << std::to_string(dt)
                //     << " and IMU datas " << std::to_string(datas.size());
                if (datas.empty() || DetectZeroVelocity(datas, average)) {
                    ResetVisualInitialization();
                    return false;
                }
                auto preint = predictor_ptr_->CreatePreintegration(
                    last_state->time_, feature_data.time_,
                    last_state->ba_, last_state->bg_);
                std::shared_ptr<State> new_state = preint->predict(last_state);
                last_state = new_state;

                int start_frame_idx = state_manager_ptr_->GetAllStates().size();
                LOG(INFO) << "Visual Initialization add new state at time " << std::to_string(feature_data.time_)
                    << ", frame index " << std::to_string(start_frame_idx)
                    << ", point num: " << feature_data.features_.size();
                vins_feature_manager_ptr_->addFeatureCheckParallax(
                    start_frame_idx, feature_data.features_, last_state);
                state_manager_ptr_->PushState(new_state);
                last_state->feature_data_ = feature_data;
            }
            
            last_feature_data_ = feature_data;
        }

        if (state_manager_ptr_->GetAllStates().size() > param_ptr_->WINDOW_SIZE) {
            LOG(INFO) << "凑够了初始化帧数";
            if (InitialStructure()) {
                LOG(INFO) << "视觉惯性初始化完毕";
                return true;
            } else {
                ResetVisualInitialization();
                return false;
            }
        } else {
           return false; 
        }
    }

    /**
     * @brief VIO初始化，将滑窗中的P V Q恢复到第0帧并且和重力对齐
     * 
     * @return true 
     * @return false 
     */
    bool InitialStructure()
    {
        // 这里与VINS不同在于使用了所有关键帧做，普通帧不考虑，因为前面做了0速检测
        std::map<double, ImageFrame> all_image_frame;
        auto all_states = state_manager_ptr_->GetAllStates();
        int frame_count = all_states.size();
        for (int i = 0; i < frame_count; i++)
        {

            ImageFrame imageframe(
                all_states[i]->feature_data_.features_, all_states[i]->time_);
            imageframe.pre_integration =
                dynamic_cast<IMUPreintegration*>(all_states[i]->preint_.get());
            // 这里就是简单的把图像和预积分绑定在一起，这里预积分就是两帧之间的，滑窗中实际上是两个KF之间的
            // 实际上是准备用来初始化的相关数据
            all_image_frame.insert(std::make_pair(all_states[i]->time_, imageframe));
        }

        // Step 1 check imu observibility
        // Step 2 global sfm
        // 做一个纯视觉slam
        Eigen::Quaterniond Q[frame_count];
        Eigen::Vector3d T[frame_count];
        std::map<int, Eigen::Vector3d> sfm_tracked_points;
        std::vector<SFMFeature> sfm_f;   // 保存每个特征点的信息
        // 遍历所有的特征点
        for (auto &it_per_id : vins_feature_manager_ptr_->feature)
        {
            int imu_j = it_per_id.start_frame - 1;  // 这个跟imu无关，就是存储观测特征点的帧的索引
            SFMFeature tmp_feature; // 用来后续做sfm
            tmp_feature.state = false;
            tmp_feature.id = it_per_id.feature_id;
            for (auto &it_per_frame : it_per_id.feature_per_frame)
            {
                imu_j++;
                Eigen::Vector3d pts_j = it_per_frame.point;
                // 索引以及各自坐标系下的坐标
                tmp_feature.observation.push_back(
                    std::make_pair(imu_j, Eigen::Vector2d{pts_j.x(), pts_j.y()}));
            }
            sfm_f.push_back(tmp_feature);
        } 
        Eigen::Matrix3d relative_R;
        Eigen::Vector3d relative_T;
        int l;
        if (!RelativePose(relative_R, relative_T, l))
        {
            LOG(INFO) << "Not enough features or parallax; Move device around";
            return false;
        }
        GlobalSFM sfm;
        // 进行sfm的求解
        if(!sfm.construct(frame_count, Q, T, l,
                relative_R, relative_T,
                sfm_f, sfm_tracked_points))
        {
            LOG(INFO) << "global SFM failed!";
            return false;
        } else {
            LOG(INFO) << "global SFM success!";
        }

        // Step 3 solve pnp for all frame
        // step2只是针对KF进行sfm，初始化需要all_image_frame中的所有元素，因此下面通过KF来求解其他的非KF的位姿
        for (int i = 0; i < frame_count; i++)
        {
            double time = all_states[i]->time_;
            all_image_frame[time].R = Q[i].toRotationMatrix() * param_ptr_->Rbc_.transpose();
            all_image_frame[time].T = T[i];
        }

        // 到此就求解出用来做视觉惯性对齐的所有视觉帧的位姿
        // Step 4 视觉惯性对齐
        Eigen::VectorXd x;
        Eigen::Vector3d g;
        Eigen::Vector3d Ps[frame_count];
        Eigen::Vector3d Vs[frame_count];
        Eigen::Matrix3d Rs[frame_count];
        // Eigen::Vector3d Bas[frame_count];
        Eigen::Vector3d Bgs[frame_count];
        for (int i = 0; i < frame_count; i++)
        {
            Rs[i].setIdentity();
            Ps[i].setZero();
            Vs[i].setZero();
            // Bas[i].setZero();
            Bgs[i].setZero();
        }

        //solve scale
        bool result = VisualIMUAlignment(
            param_ptr_, all_image_frame, Bgs, g, x);
        if(!result)
        {
            LOG(INFO) << "solve g failed!";
            return false;
        }

        // change state
        // 首先把对齐后KF的位姿附给滑窗中的值，Rwi twc
        LOG(INFO) << "all_image_frame: " << all_image_frame.size();
        for (auto [ti, imagedd] : all_image_frame)
            LOG(INFO) << "all_image_frame1: " << std::to_string(ti);
        for (int i = 0; i < frame_count; i++)
        {
            LOG(INFO) << "frame_count: " << std::to_string(all_states[i]->time_);
            Eigen::Matrix3d Ri = all_image_frame[all_states[i]->time_].R;
            Eigen::Vector3d Pi = all_image_frame[all_states[i]->time_].T;
            Ps[i] = Pi;
            Rs[i] = Ri;
            all_image_frame[all_states[i]->time_].is_key_frame = true;
        }

        Eigen::VectorXd dep = vins_feature_manager_ptr_->getDepthVector();  // 根据有效特征点数初始化这个动态向量
        for (int i = 0; i < dep.size(); i++)
            dep[i] = -1;    // 深度预设都是-1
        vins_feature_manager_ptr_->clearDepth(dep);  // 特征管理器把所有的特征点逆深度也设置为-1

        //triangulat on cam pose , no tic
        Eigen::Vector3d TIC_TMP[1];
        Eigen::Matrix3d RIC[1];
        for(int i = 0; i < 1; i++) {
            TIC_TMP[i].setZero();
            RIC[i] = param_ptr_->Rbc_;
        }
        // 多约束三角化所有的特征点，注意，仍带是尺度模糊的
        vins_feature_manager_ptr_->triangulate(Ps, Rs, &(TIC_TMP[0]), &(RIC[0]));

        double s = (x.tail<1>())(0);
        // 将滑窗中的预积分重新计算
        for (int i = 1; i <= param_ptr_->WINDOW_SIZE; i++)
        {
            all_states[i]->preint_->Repropagate(Eigen::Vector3d::Zero(), Bgs[i]);
        }
        // 下面开始把所有的状态对齐到第0帧的imu坐标系
        for (int i = frame_count; i >= 0; i--)
            // twi - tw0 = toi,就是把所有的平移对齐到滑窗中的第0帧
            Ps[i] = s * Ps[i] - Rs[i] * param_ptr_->tbc_ - (s * Ps[0] - Rs[0] * param_ptr_->tbc_);
        int kv = -1;
        std::map<double, ImageFrame>::iterator frame_i;
        // 把求解出来KF的速度赋给滑窗中
        for (frame_i = all_image_frame.begin(); frame_i != all_image_frame.end(); frame_i++)
        {
            if(frame_i->second.is_key_frame)
            {
                kv++;
                // 当时求得速度是imu系，现在转到world系
                Vs[kv] = frame_i->second.R * x.segment<3>(kv * 3);
            }
        }
        // 把尺度模糊的3d点恢复到真实尺度下
        for (auto &it_per_id : vins_feature_manager_ptr_->feature)
        {
            it_per_id.used_num = it_per_id.feature_per_frame.size();
            if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < param_ptr_->WINDOW_SIZE - 2))
                continue;
            it_per_id.estimated_depth *= s;
        }
        // 所有的P V Q全部对齐到第0帧的，同时和对齐到重力方向
        Eigen::Matrix3d R0 = Converter::g2R(g);  // g是枢纽帧下的重力方向，得到R_w_j
        double yaw = Converter::R2ypr(R0 * Rs[0]).x();    // Rs[0]实际上是R_j_0
        R0 = Converter::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;  // 第一帧yaw赋0
        g = R0 * g;
        //Eigen::Matrix3d rot_diff = R0 * Rs[0].transpose();
        Eigen::Matrix3d rot_diff = R0;
        for (int i = 0; i < frame_count; i++)
        {
            // Ps[i] = rot_diff * Ps[i];
            // Rs[i] = rot_diff * Rs[i];   // 全部对齐到重力下，同时yaw角对齐到第一帧
            // Vs[i] = rot_diff * Vs[i];

            all_states[i]->twb_ = rot_diff * Ps[i];
            all_states[i]->Rwb_ = Eigen::Quaterniond(rot_diff * Rs[i]);
            all_states[i]->Vw_ = rot_diff * Vs[i];
            all_states[i]->bg_ = Bgs[i];
        }

        int newest_idx = frame_count - 1;
        all_states[newest_idx]->C_ = Eigen::MatrixXd::Identity(param_ptr_->STATE_DIM, param_ptr_->STATE_DIM);
        all_states[newest_idx]->C_.block<3, 3>(param_ptr_->POSI_INDEX, param_ptr_->POSI_INDEX) = Eigen::Matrix3d::Identity() * 0.0025;
        all_states[newest_idx]->C_.block<3, 3>(param_ptr_->ORI_INDEX_STATE_, param_ptr_->ORI_INDEX_STATE_) = Eigen::Matrix3d::Identity() * 0.0025;
        all_states[newest_idx]->C_.block<3, 3>(param_ptr_->VEL_INDEX_STATE_, param_ptr_->VEL_INDEX_STATE_) = Eigen::Matrix3d::Identity() * 0.0025;
        all_states[newest_idx]->C_.block<3, 3>(param_ptr_->GYRO_BIAS_INDEX_STATE_, param_ptr_->GYRO_BIAS_INDEX_STATE_) = Eigen::Matrix3d::Identity() * 0.01;
        all_states[newest_idx]->C_.block<3, 3>(param_ptr_->ACC_BIAS_INDEX_STATE_, param_ptr_->ACC_BIAS_INDEX_STATE_) = Eigen::Matrix3d::Identity() * 0.01;
        LOG(INFO) << "g0     " << g.transpose();
        LOG(INFO) << "my R0  " << Converter::R2ypr(rot_diff * Rs[0]).transpose();

        auto cur_state_ptr = all_states[newest_idx];
        LOG(INFO) << "初始化状态量：";
        LOG(INFO) << "时间: " << std::to_string(cur_state_ptr->time_);
        LOG(INFO) << "位置: " << cur_state_ptr->twb_.transpose();
        LOG(INFO) << "速度: " << cur_state_ptr->Vw_.transpose();
        LOG(INFO) << "加速度零偏: " << cur_state_ptr->ba_.transpose();
        LOG(INFO) << "陀螺零偏: " << cur_state_ptr->bg_.transpose();
        LOG(INFO) << "旋转矩阵:\n" << cur_state_ptr->Rwb_;
        if (param_ptr_->fusion_model_ == 0) {
            // 初始化填充cam_states_和map_server
            // 只保留第二个帧以及之后的
            state_manager_ptr_->cam_states_.clear();

            // 填充cam_states_
            for (int i = 0; i < frame_count; ++i) {
                auto cam_state = std::make_shared<CamState>(i);
                cam_state->time = all_states[i]->time_;
                cam_state->Rwc_ = all_states[i]->Rwb_ * param_ptr_->Rbc_;
                cam_state->twc_ = all_states[i]->Rwb_ * param_ptr_->tbc_ + all_states[i]->twb_;
                state_manager_ptr_->cam_states_[i] = cam_state;
            }

            // 填充协方差矩阵 C_
            int N = state_manager_ptr_->cam_states_.size();
            int state_dim = 15 + 6 * N;
            cur_state_ptr->C_ = Eigen::MatrixXd::Identity(state_dim, state_dim);

            // 主状态部分（前15维）赋初值
            cur_state_ptr->C_.block<3,3>(param_ptr_->POSI_INDEX, param_ptr_->POSI_INDEX) = Eigen::Matrix3d::Identity() * 0.0025;
            cur_state_ptr->C_.block<3,3>(param_ptr_->ORI_INDEX_STATE_, param_ptr_->ORI_INDEX_STATE_) = Eigen::Matrix3d::Identity() * 0.0025;
            cur_state_ptr->C_.block<3,3>(param_ptr_->VEL_INDEX_STATE_, param_ptr_->VEL_INDEX_STATE_) = Eigen::Matrix3d::Identity() * 0.0025;
            cur_state_ptr->C_.block<3,3>(param_ptr_->GYRO_BIAS_INDEX_STATE_, param_ptr_->GYRO_BIAS_INDEX_STATE_) = Eigen::Matrix3d::Identity() * 0.01;
            cur_state_ptr->C_.block<3,3>(param_ptr_->ACC_BIAS_INDEX_STATE_, param_ptr_->ACC_BIAS_INDEX_STATE_) = Eigen::Matrix3d::Identity() * 0.01;

            // 每个cam_state的协方差（后6N维），可设置为较大初值
            for (int i = 0; i < N; ++i) {
                int idx = 15 + i * 6;
                cur_state_ptr->C_.block<6,6>(idx, idx) = Eigen::Matrix<double,6,6>::Identity() * 0.01;
            }

            // 填充map_server，只保留三角化成功的点
            int valid_feature_count = 0;
            std::ofstream ofs("sfm_points2.txt");
            for (const auto& feat : vins_feature_manager_ptr_->feature) {
                if (feat.estimated_depth > 0) {
                    // 取第一个观测帧
                    const auto& first_frame = feat.feature_per_frame[0];
                    // 归一化相机坐标
                    Eigen::Vector3d uv = first_frame.point;
                    // 相机坐标系下三维点
                    Eigen::Vector3d pts_cam = uv * feat.estimated_depth;

                    // 取该帧的位姿
                    auto first_obs_frame = state_manager_ptr_->cam_states_[feat.start_frame];
                    // 转到世界坐标系
                    Eigen::Vector3d position = first_obs_frame->Rwc_ * pts_cam + first_obs_frame->twc_;

                    MsckfFeature msckf_feat(feat.feature_id, param_ptr_);
                    msckf_feat.position = position;
                    ofs << feat.feature_id << " " << position.x() << " " << position.y() << " " << position.z() << " " << 
                        feat.estimated_depth << " " << uv.transpose() << " " << first_obs_frame->twc_.y() << " " << first_obs_frame->twc_.z() << "\n";
                    msckf_feat.is_initialized = true;
                    // 观测填充
                    for (int idx = 0; idx < feat.feature_per_frame.size(); idx++) {
                        int frame_idx = idx + feat.start_frame;
                        msckf_feat.observations[frame_idx] = feat.feature_per_frame[idx].point.head<2>();
                    }
                    // 加入map_server
                    map_server_[feat.feature_id] = msckf_feat;
                    valid_feature_count++;
                }
            }
            LOG(INFO) << "Map server initialized with " << valid_feature_count << " features.";
        }
        return true;
    }

    /**
     * @brief 寻找滑窗内一个帧作为枢纽帧，要求和最后一帧既有足够的共视也要有足够的视差
     *        这样其他帧都对齐到这个枢纽帧上
     *        得到T_l_last
     * @param[in] relative_R 
     * @param[in] relative_T 
     * @param[in] l 
     * @return true 
     * @return false 
     */

    bool RelativePose(Eigen::Matrix3d &relative_R, Eigen::Vector3d &relative_T, int &l)
    {
        MotionEstimator m_estimator;
        // find previous frame which contians enough correspondance and parallex with newest frame
        // 优先从最前面开始
        for (int i = 0; i < param_ptr_->WINDOW_SIZE; i++)
        {
            std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> corres;
            corres = vins_feature_manager_ptr_->getCorresponding(i, param_ptr_->WINDOW_SIZE);
            // LOG(INFO) << "frame " << i << " to newest frame has correspondance " << corres.size();
            // 要求共视的特征点足够多
            if (corres.size() > 20)
            {
                double sum_parallax = 0;
                double average_parallax;
                for (int j = 0; j < int(corres.size()); j++)
                {
                    Eigen::Vector2d pts_0(corres[j].first(0), corres[j].first(1));
                    Eigen::Vector2d pts_1(corres[j].second(0), corres[j].second(1));
                    double parallax = (pts_0 - pts_1).norm();   // 计算了视差
                    sum_parallax = sum_parallax + parallax;

                }
                // 计算每个特征点的平均视差
                average_parallax = 1.0 * sum_parallax / int(corres.size());
                // LOG(INFO) << "find frame " << i << " has enough correspondance " << corres.size()
                //           << " average parallax " << average_parallax * 460;
                // 有足够的视差在通过本质矩阵恢复第i帧和最后一帧之间的 R t T_i_last
                if(average_parallax * 460 > 30 && m_estimator.solveRelativeRT(corres, relative_R, relative_T))
                {
                    l = i;
                    LOG(INFO) << "average_parallax " << average_parallax * 460 << " choose l " << l <<
                        " and newest frame to triangulate the whole structure";
                    LOG(INFO) << "relative_R: " << std::endl << relative_R << std::endl <<  relative_T.transpose();
                    return true;
                }
            }
        }
        return false;
    }

    void ResetVisualInitialization() {
        state_manager_ptr_->Reset();
        vins_feature_manager_ptr_->clearState();
        last_feature_data_.time_ = -1.0;
    }

    std::shared_ptr<DataManager> data_manager_ptr_;
    std::shared_ptr<StateManager> state_manager_ptr_;
    std::shared_ptr<Predictor> predictor_ptr_;
    std::shared_ptr<VinsFeatureManager> vins_feature_manager_ptr_;
    std::mutex states_mtx_;
    std::shared_ptr<Parameter> param_ptr_;

    // tmp data
    GNSSData last_gnss_data_;
    // WheelData last_wheel_data_;
    FeatureData last_feature_data_;

    std::shared_ptr<CooTrans> coo_trans_ptr_;

    const double D2R = (M_PI / 180.0);
    const double R2D = (180.0 / M_PI);
};