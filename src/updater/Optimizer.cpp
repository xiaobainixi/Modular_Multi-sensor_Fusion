#pragma once
#include "Optimizer.h"

// 右乘更新 ceres自带 EigenQuaternionParameterization 为左乘
bool QLocalParameterization::Plus(const double *x, const double *delta, double *x_plus_delta) const
{
    Eigen::Map<const Eigen::Quaterniond> _q(x);
    Eigen::Quaterniond dq = Converter::RotVecToQuaternion(Eigen::Map<const Eigen::Vector3d>(delta));
    Eigen::Map<Eigen::Quaterniond> q(x_plus_delta);
    q = (_q * dq).normalized();
    return true;
}
#if CERES_VERSION_MAJOR >= 2
bool QLocalParameterization::PlusJacobian(const double *x, double *jacobian) const
#else
bool QLocalParameterization::ComputeJacobian(const double *x, double *jacobian) const
#endif
{
    // 这里要注意！！！！！！！
    // 自己算雅可比时，关于旋转的雅可比直接对应的是旋转向量，所以这里是单位阵
    Eigen::Map<Eigen::Matrix<double, 4, 3, Eigen::RowMajor>> j(jacobian);
    j.topRows<3>().setIdentity();
    j.bottomRows<1>().setZero();

    // 自动求导时求的是关于四元数的雅可比 所以这里要加四元数相对于旋转向量的雅可比
    // Eigen::Map<const Eigen::Quaterniond> q(x);            // (x,y,z,w)
    // const double xq = q.x();
    // const double yq = q.y();
    // const double zq = q.z();
    // const double wq = q.w();

    // Eigen::Map<Eigen::Matrix<double,4,3,Eigen::RowMajor>> J(jacobian);
    // J <<  0.5*wq,   -0.5*zq,   0.5*yq,
    //       0.5*zq,    0.5*wq,  -0.5*xq,
    //      -0.5*yq,    0.5*xq,   0.5*wq,
    //      -0.5*xq,   -0.5*yq,  -0.5*zq;
    // return true;
    return true;
}

#if CERES_VERSION_MAJOR >= 2
bool QLocalParameterization::Minus(const double *y, const double *x, double *y_minus_x) const
{
    Eigen::Map<const Eigen::Quaterniond> q_x(x);
    Eigen::Map<const Eigen::Quaterniond> q_y(y);
    Eigen::Map<Eigen::Vector3d> delta(y_minus_x);
    delta = Converter::QuaternionToRotVec((q_x.conjugate() * q_y).normalized());
    return true;
}

bool QLocalParameterization::MinusJacobian(const double *x, double *jacobian) const
{
    Eigen::Map<Eigen::Matrix<double, 3, 4, Eigen::RowMajor>> j(jacobian);
    j.setZero();
    j.block<3, 3>(0, 0).setIdentity();
    return true;
}
#endif

void Optimizer::SlideWindow(bool slide_old)
{
    std::vector<std::shared_ptr<State>> window_states = state_manager_ptr_->GetAllStates();
    if (window_states.size() <= param_ptr_->WINDOW_SIZE)
        return;

    std::shared_ptr<MarginalizationInfo> marginalization_info = std::make_shared<MarginalizationInfo>();

    // 指定每个参数块独立的ID, 用于索引参数
    // key 表示参数块的内存地址, value表示参数块的ID
    std::unordered_map<long, long> parameters_ids;
    parameters_ids.clear();
    long parameters_id = 0;

    std::vector<FeaturePerId *> depth_feats_marg;

    {
        // 边缘化参数
        // Marginalization parameters
        for (auto &last_marginalization_parameter_block : last_marginalization_parameter_blocks_)
        {
            parameters_ids[reinterpret_cast<long>(last_marginalization_parameter_block)] = parameters_id++;
        }

        // 状态参数
        // State parameters
        for (size_t i = 0; i < window_states.size(); ++i)
        {
            // IMU模式
            if (param_ptr_->state_type_ == 0)
            {
                parameters_ids[reinterpret_cast<long>(window_states[i]->twb_.data())] = parameters_id++;
                parameters_ids[reinterpret_cast<long>(window_states[i]->Rwb_.coeffs().data())] = parameters_id++;
                parameters_ids[reinterpret_cast<long>(window_states[i]->Vw_.data())] = parameters_id++;
                parameters_ids[reinterpret_cast<long>(window_states[i]->ba_.data())] = parameters_id++;
                parameters_ids[reinterpret_cast<long>(window_states[i]->bg_.data())] = parameters_id++;
            }
            // 轮速计模式
            else if (param_ptr_->state_type_ == 1)
            {
                parameters_ids[reinterpret_cast<long>(window_states[i]->twb_.data())] = parameters_id++;
                parameters_ids[reinterpret_cast<long>(window_states[i]->Rwb_.coeffs().data())] = parameters_id++;
            }
            // IMU+轮速计模式
            else if (param_ptr_->state_type_ == 2)
            {
                parameters_ids[reinterpret_cast<long>(window_states[i]->twb_.data())] = parameters_id++;
                parameters_ids[reinterpret_cast<long>(window_states[i]->Rwb_.coeffs().data())] = parameters_id++;
                parameters_ids[reinterpret_cast<long>(window_states[i]->bg_.data())] = parameters_id++;
            }
        }

        // 逆深度参数（仅边缘化最老帧时使用）
        if (slide_old && vins_feature_manager_ptr_ && param_ptr_->use_camera_)
        {
            depth_feats_marg.reserve(vins_feature_manager_ptr_->feature.size());
            for (auto &it_per_id : vins_feature_manager_ptr_->feature)
            {
                it_per_id.used_num = static_cast<int>(it_per_id.feature_per_frame.size());
                if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < param_ptr_->WINDOW_SIZE - 2))
                    continue;
                if (it_per_id.start_frame != 0)
                    continue;
                if (it_per_id.estimated_depth <= 0)
                    continue;

                auto idx_it = inv_depth_index_.find(&it_per_id);
                if (idx_it == inv_depth_index_.end())
                    continue;
                double *inv_depth_ptr = &inv_depths_[idx_it->second];
                parameters_ids[reinterpret_cast<long>(inv_depth_ptr)] = parameters_id++;
                depth_feats_marg.push_back(&it_per_id);
            }
        }

        // // 逆深度参数
        // // Inverse depth parameters
        // frame = map_->keyframes().at(keyframeids[0]);
        // auto features = frame->features();
        // for (auto const &feature : features)
        // {
        //     auto mappoint = feature.second->getMapPoint();
        //     if (feature.second->isOutlier() || !mappoint || mappoint->isOutlier())
        //     {
        //         continue;
        //     }

        //     if (mappoint->referenceFrame() != frame)
        //     {
        //         continue;
        //     }

        //     double *invdepth = &invdepthlist_[mappoint->id()];
        //     parameters_ids[reinterpret_cast<long>(invdepth)] = parameters_id++;
        // }

        // 更新参数块的特定ID, 必要的
        // Update the IS for parameters
        marginalization_info->updateParamtersIds(parameters_ids);
    }

    // 边缘化因子
    // The prior factor
    if (last_marginalization_info_ && last_marginalization_info_->isValid())
    {
        LOG(INFO) << "[Marg] last prior blocks: " << last_marginalization_parameter_blocks_.size()
                  << ", slide_old: " << (slide_old ? 1 : 0);
        for (size_t k = 0; k < last_marginalization_parameter_blocks_.size(); ++k)
        {
            LOG(INFO) << "[Marg] prior block[" << k << "] addr="
                      << static_cast<const void *>(last_marginalization_parameter_blocks_[k]);
        }
        // 存放本次要被边缘化的参数块的index
        std::vector<int> marginalized_index;
        if (slide_old)
        {
            // 边缘化最老帧
            // 用当前窗口状态的参数指针进行匹配
            for (size_t k = 0; k < last_marginalization_parameter_blocks_.size(); k++)
            {
                if ((param_ptr_->state_type_ == 0 &&
                        (last_marginalization_parameter_blocks_[k] == window_states[0]->twb_.data() ||
                        last_marginalization_parameter_blocks_[k] == window_states[0]->Rwb_.coeffs().data() ||
                        last_marginalization_parameter_blocks_[k] == window_states[0]->Vw_.data() ||
                        last_marginalization_parameter_blocks_[k] == window_states[0]->ba_.data() ||
                        last_marginalization_parameter_blocks_[k] == window_states[0]->bg_.data())) ||
                    (param_ptr_->state_type_ == 1 &&
                        (last_marginalization_parameter_blocks_[k] == window_states[0]->twb_.data() ||
                        last_marginalization_parameter_blocks_[k] == window_states[0]->Rwb_.coeffs().data())) ||
                    (param_ptr_->state_type_ == 2 &&
                        (last_marginalization_parameter_blocks_[k] == window_states[0]->twb_.data() ||
                        last_marginalization_parameter_blocks_[k] == window_states[0]->Rwb_.coeffs().data() ||
                        last_marginalization_parameter_blocks_[k] == window_states[0]->bg_.data())))
                {
                    marginalized_index.push_back((int)k);
                }
            }
        }
        else
        {
            // 边缘化次新帧
            size_t i = window_states.size() - 2;
            for (size_t k = 0; k < last_marginalization_parameter_blocks_.size(); k++)
            {
                // if ((param_ptr_->state_type_ == 0 &&
                //      (last_marginalization_parameter_blocks_[k] == window_states[i]->twb_.data() ||
                //       last_marginalization_parameter_blocks_[k] == window_states[i]->Rwb_.coeffs().data() ||
                //       last_marginalization_parameter_blocks_[k] == window_states[i]->Vw_.data() ||
                //       last_marginalization_parameter_blocks_[k] == window_states[i]->ba_.data() ||
                //       last_marginalization_parameter_blocks_[k] == window_states[i]->bg_.data())) ||
                //     (param_ptr_->state_type_ == 1 &&
                //      (last_marginalization_parameter_blocks_[k] == window_states[i]->twb_.data() ||
                //       last_marginalization_parameter_blocks_[k] == window_states[i]->Rwb_.coeffs().data())) ||
                //     (param_ptr_->state_type_ == 2 &&
                //      (last_marginalization_parameter_blocks_[k] == window_states[i]->twb_.data() ||
                //       last_marginalization_parameter_blocks_[k] == window_states[i]->Rwb_.coeffs().data() ||
                //       last_marginalization_parameter_blocks_[k] == window_states[i]->bg_.data())))
                // {
                //     marginalized_index.push_back((int)k);
                // }
                if (last_marginalization_parameter_blocks_[k] == window_states[i]->twb_.data() ||
                    last_marginalization_parameter_blocks_[k] == window_states[i]->Rwb_.coeffs().data())
                {
                    marginalized_index.push_back((int)k);
                }
            }
        }

        auto factor = std::make_shared<MarginalizationFactor>(last_marginalization_info_);
        auto residual = std::make_shared<ResidualBlockInfo>(
            factor, nullptr, last_marginalization_parameter_blocks_, marginalized_index);
        marginalization_info->addResidualBlockInfo(residual);
        LOG(INFO) << "[Marg] added prior residual, marginalized_index size: " << marginalized_index.size();
    }

    if (slide_old)
    {
        // 视觉重投影因子（与第0帧建立约束）
        if (vins_feature_manager_ptr_ && param_ptr_->use_camera_)
        {
            auto loss_function = std::make_shared<ceres::HuberLoss>(1.0);
            for (size_t feat_idx = 0; feat_idx < depth_feats_marg.size(); ++feat_idx)
            {
                auto &it_per_id = *depth_feats_marg[feat_idx];
                int imu_i = it_per_id.start_frame;
                int imu_j = imu_i - 1;
                if (imu_i != 0)
                    continue;

                auto idx_it = inv_depth_index_.find(&it_per_id);
                if (idx_it == inv_depth_index_.end())
                    continue;
                double *inv_depth_ptr = &inv_depths_[idx_it->second];

                Eigen::Vector3d pts_i = it_per_id.feature_per_frame[0].point;
                for (const auto &it_per_frame : it_per_id.feature_per_frame)
                {
                    imu_j++;
                    if (imu_i == imu_j)
                        continue;

                    Eigen::Vector3d pts_j = it_per_frame.point;
                    auto factor = std::make_shared<ProjectionFactor>(pts_i, pts_j, param_ptr_);
                    auto residual = std::make_shared<ResidualBlockInfo>(
                        factor, loss_function,
                        std::vector<double *>{
                            window_states[imu_i]->twb_.data(),
                            window_states[imu_i]->Rwb_.coeffs().data(),
                            window_states[imu_j]->twb_.data(),
                            window_states[imu_j]->Rwb_.coeffs().data(),
                            inv_depth_ptr},
                        std::vector<int>{0, 1, 4});
                    marginalization_info->addResidualBlockInfo(residual);
                }
            }
        }

        // GNSS 因子
        if (window_states[0]->cur_gnss_data_.time_ > 0)
        {
            auto gnss_data = window_states[0]->cur_gnss_data_;
            auto factor = std::make_shared<GNSSResidual>(
                Eigen::Vector3d(gnss_data.x_, gnss_data.y_, gnss_data.z_), param_ptr_);
            auto residual = std::make_shared<ResidualBlockInfo>(
                factor, nullptr, std::vector<double *>{window_states[0]->twb_.data()}, std::vector<int>{0});
            marginalization_info->addResidualBlockInfo(residual);
        }

        // BaseVelocityResidual 因子
        if (window_states[0]->cur_wheel_data_.time_ > 0)
        {
            auto wheel_data = window_states[0]->cur_wheel_data_;
            auto factor = std::make_shared<BaseVelocityResidual>(
                Eigen::Vector3d((wheel_data.lv_ + wheel_data.rv_) * 0.5, 0.0, 0.0), param_ptr_);
            auto residual = std::make_shared<ResidualBlockInfo>(
                factor, nullptr, std::vector<double *>{window_states[0]->Vw_.data(), window_states[0]->Rwb_.coeffs().data()},
                std::vector<int>{0, 1});
            marginalization_info->addResidualBlockInfo(residual);
        }

        // 预积分因子
        std::vector<int> marg_index;
        auto preint_ptr = window_states[0 + 1]->preint_;
        if (param_ptr_->state_type_ == 0) // IMU
        {
            // 一次滑多帧使用
            // if (0 == (num_marg - 1))
            // {
            //     marg_index = {0, 1, 2, 3, 4};
            // }
            // else
            // {
            //     marg_index = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
            // }
            marg_index = {0, 1, 2, 3, 4};
            auto factor = std::make_shared<IMUPreintegrationResidual>(
                std::dynamic_pointer_cast<IMUPreintegration>(preint_ptr), param_ptr_);
            auto residual = std::make_shared<ResidualBlockInfo>(
                factor, nullptr,
                std::vector<double *>{window_states[0]->twb_.data(), window_states[0]->Rwb_.coeffs().data(),
                                    window_states[0]->Vw_.data(), window_states[0]->ba_.data(), window_states[0]->bg_.data(),
                                    window_states[0 + 1]->twb_.data(), window_states[0 + 1]->Rwb_.coeffs().data(),
                                    window_states[0 + 1]->Vw_.data(), window_states[0 + 1]->ba_.data(), window_states[0 + 1]->bg_.data()},
                marg_index);
            marginalization_info->addResidualBlockInfo(residual);
        }
        else if (param_ptr_->state_type_ == 1) // 轮速计
        {
            // if (0 == (num_marg - 1))
            // {
            //     marg_index = {0, 1};
            // }
            // else
            // {
            //     marg_index = {0, 1, 2, 3};
            // }
            marg_index = {0, 1};
            auto factor = std::make_shared<WheelPreintegrationResidual>(
                std::dynamic_pointer_cast<WheelPreintegration>(preint_ptr), param_ptr_);
            auto residual = std::make_shared<ResidualBlockInfo>(
                factor, nullptr,
                std::vector<double *>{window_states[0]->twb_.data(), window_states[0]->Rwb_.coeffs().data(),
                                    window_states[0 + 1]->twb_.data(), window_states[0 + 1]->Rwb_.coeffs().data()},
                marg_index);
            marginalization_info->addResidualBlockInfo(residual);
        }
        else if (param_ptr_->state_type_ == 2) // IMU+轮速计
        {
            // if (0 == (num_marg - 1))
            // {
            //     marg_index = {0, 1, 2};
            // }
            // else
            // {
            //     marg_index = {0, 1, 2, 3, 4, 5};
            // }
            marg_index = {0, 1, 2};
            auto factor = std::make_shared<WheelIMUPreintegrationResidual>(
                std::dynamic_pointer_cast<WheelIMUPreintegration>(preint_ptr), param_ptr_);
            auto residual = std::make_shared<ResidualBlockInfo>(
                factor, nullptr,
                std::vector<double *>{window_states[0]->twb_.data(), window_states[0]->Rwb_.coeffs().data(), window_states[0]->bg_.data(),
                                    window_states[0 + 1]->twb_.data(), window_states[0 + 1]->Rwb_.coeffs().data(), window_states[0 + 1]->bg_.data()},
                marg_index);
            marginalization_info->addResidualBlockInfo(residual);
        }
    }
    else
    {
        // 边缘化次新帧
        size_t i = window_states.size() - 2;
        // i-1 ~ i
        auto preint_prev = window_states[i]->preint_;
        // i ~ i+1
        auto preint_curr = window_states[i + 1]->preint_;

        // 1) 合并两段预积分（i-1 -> i） + （i -> i+1） => （i-1 -> i+1）
        if (param_ptr_->state_type_ == 0)
        {
            auto prev = std::dynamic_pointer_cast<IMUPreintegration>(preint_prev);
            auto curr = std::dynamic_pointer_cast<IMUPreintegration>(preint_curr);
            if (prev && curr)
            {
                prev->Merge(*curr);
            }
        }
        else if (param_ptr_->state_type_ == 1)
        {
            auto prev = std::dynamic_pointer_cast<WheelPreintegration>(preint_prev);
            auto curr = std::dynamic_pointer_cast<WheelPreintegration>(preint_curr);
            if (prev && curr)
            {
                prev->Merge(*curr);
            }
        }
        else if (param_ptr_->state_type_ == 2)
        {
            auto prev = std::dynamic_pointer_cast<WheelIMUPreintegration>(preint_prev);
            auto curr = std::dynamic_pointer_cast<WheelIMUPreintegration>(preint_curr);
            if (prev && curr)
            {
                prev->Merge(*curr);
            }
        }
        window_states[i + 1]->preint_ = preint_prev;

        // 2) 只边缘化 GNSS / 轮速（对应次新帧）
        if (window_states[i]->cur_gnss_data_.time_ > 0)
        {
            auto gnss_data = window_states[i]->cur_gnss_data_;
            auto factor = std::make_shared<GNSSResidual>(
                Eigen::Vector3d(gnss_data.x_, gnss_data.y_, gnss_data.z_), param_ptr_);
            auto residual = std::make_shared<ResidualBlockInfo>(
                factor, nullptr, std::vector<double *>{window_states[i]->twb_.data()}, std::vector<int>{0});
            marginalization_info->addResidualBlockInfo(residual);
        }

        if (window_states[i]->cur_wheel_data_.time_ > 0)
        {
            auto wheel_data = window_states[i]->cur_wheel_data_;
            auto factor = std::make_shared<BaseVelocityResidual>(
                Eigen::Vector3d((wheel_data.lv_ + wheel_data.rv_) * 0.5, 0.0, 0.0), param_ptr_);
            auto residual = std::make_shared<ResidualBlockInfo>(
                factor, nullptr,
                std::vector<double *>{window_states[i]->Vw_.data(), window_states[i]->Rwb_.coeffs().data()},
                std::vector<int>{0, 1});
            marginalization_info->addResidualBlockInfo(residual);
        }
    }

    // 边缘化处理
    // Do marginalization
    marginalization_info->marginalization();

    // 保留的数据, address 存放parameters_id : 参数地址
    // Update the address
    std::unordered_map<long, double *> address;
    auto add_state_blocks = [&](const std::shared_ptr<State> &state) {
        if (param_ptr_->state_type_ == 0)
        {
            address[parameters_ids[reinterpret_cast<long>(state->twb_.data())]] = state->twb_.data();
            address[parameters_ids[reinterpret_cast<long>(state->Rwb_.coeffs().data())]] = state->Rwb_.coeffs().data();
            address[parameters_ids[reinterpret_cast<long>(state->Vw_.data())]] = state->Vw_.data();
            address[parameters_ids[reinterpret_cast<long>(state->ba_.data())]] = state->ba_.data();
            address[parameters_ids[reinterpret_cast<long>(state->bg_.data())]] = state->bg_.data();
        }
        else if (param_ptr_->state_type_ == 1)
        {
            address[parameters_ids[reinterpret_cast<long>(state->twb_.data())]] = state->twb_.data();
            address[parameters_ids[reinterpret_cast<long>(state->Rwb_.coeffs().data())]] = state->Rwb_.coeffs().data();
        }
        else if (param_ptr_->state_type_ == 2)
        {
            address[parameters_ids[reinterpret_cast<long>(state->twb_.data())]] = state->twb_.data();
            address[parameters_ids[reinterpret_cast<long>(state->Rwb_.coeffs().data())]] = state->Rwb_.coeffs().data();
            address[parameters_ids[reinterpret_cast<long>(state->bg_.data())]] = state->bg_.data();
        }
    };

    for (size_t k = 0; k < window_states.size(); ++k)
    {
        if (slide_old)
        {
            if (k == 0)
                continue; // 边缘化最老帧
        }
        else
        {
            if (k == window_states.size() - 2)
                continue; // 边缘化次新帧
        }
        add_state_blocks(window_states[k]);
    }
    // 本次边缘化后将会受到约束的参数块
    last_marginalization_parameter_blocks_ = marginalization_info->getParameterBlocks(address);
    last_marginalization_info_ = std::move(marginalization_info);
    LOG(INFO) << "[Marg] retained blocks after marginalization: "
              << last_marginalization_parameter_blocks_.size();

    // 移除边缘化的数据
    // Remove the marginalized data

    // // 保存移除的路标点, 用于可视化
    // // The marginalized mappoints, for visualization
    // frame = map_->keyframes().at(keyframeids[0]);
    // features = frame->features();
    // for (const auto &feature : features)
    // {
    //     auto mappoint = feature.second->getMapPoint();
    //     if (feature.second->isOutlier() || !mappoint || mappoint->isOutlier())
    //     {
    //         continue;
    //     }
    //     auto &pw = mappoint->pos();

    //     if (is_use_visualization_)
    //     {
    //         drawer_->addNewFixedMappoint(pw);
    //     }

    //     // 保存路标点
    //     // Save these mappoints to file
    //     ptsfilesaver_->dump(vector<double>{pw.x(), pw.y(), pw.z()});
    // }

    // 关键帧
    // The marginalized keyframe
    LOG(INFO) << "window_states: " << window_states.size() << ", num_marg: " << 1;
    if (slide_old)
    {
        state_manager_ptr_->PopFrontState();
        // 同步更新视觉特征管理器（滑窗后修正 start_frame / feature_per_frame）
        if (vins_feature_manager_ptr_ && param_ptr_->use_camera_)
        {
            // 采用“丢最老帧”的滑窗策略，使用 removeBackShiftDepth
            // marg_R, marg_P 为被移除帧的位姿；new_R, new_P 为新的最老帧位姿
            const auto &marg_state = window_states.front();
            const auto &new_state = window_states[1];
            vins_feature_manager_ptr_->removeBackShiftDepth(
                marg_state->Rwb_.toRotationMatrix(), marg_state->twb_,
                new_state->Rwb_.toRotationMatrix(), new_state->twb_);
        }
    }
    else
    {
        // 边缘化次新帧（倒数第二帧）
        state_manager_ptr_->PopSecondLastState();
        // 同步更新视觉特征管理器（滑窗后修正 start_frame / feature_per_frame）
        if (vins_feature_manager_ptr_ && param_ptr_->use_camera_)
        {
            vins_feature_manager_ptr_->removeFront(param_ptr_->WINDOW_SIZE);
        }
    }
}
void Optimizer::Optimization()
{
    const int window_size = 2;
    std::vector<std::shared_ptr<State>> window_states = state_manager_ptr_->GetAllStates();
    if (window_states.size() < window_size)
        return;

    ceres::Problem problem;

    // LOG(INFO) << "窗口大小: " << window_states.size();
    // 保存首帧 yaw/位置，用于优化后对齐
    const Eigen::Matrix3d origin_R0 = window_states[0]->Rwb_.toRotationMatrix();
    const Eigen::Vector3d origin_ypr = Converter::R2ypr(origin_R0);
    const Eigen::Vector3d origin_P0 = window_states[0]->twb_;
    auto normalize_angle_deg = [](double deg) {
        while (deg > 180.0)
            deg -= 360.0;
        while (deg < -180.0)
            deg += 360.0;
        return deg;
    };
    for (size_t i = 0; i < window_states.size(); ++i)
    {
        auto state = window_states[i];
        // 应该区分不同的传感器组合，但是那样看起来会比较乱，这么写工整一些
        // 不同组合个别变量虽然参与计算但是没有梯度，也不会用
        problem.AddParameterBlock(state->twb_.data(), 3);
        problem.AddParameterBlock(state->Rwb_.coeffs().data(), 4, new QLocalParameterization());
        problem.AddParameterBlock(state->Vw_.data(), 3);
        problem.AddParameterBlock(state->ba_.data(), 3);
        problem.AddParameterBlock(state->bg_.data(), 3);
        // 不锁首帧，优化后再对齐到首帧 yaw/位置
        // LOG(INFO) << "state[" << i << "] twb_: " << static_cast<void*>(state->twb_.data())
        //     << " (" << state->twb_.transpose() << ")"
        //     << ", Rwb_: " << static_cast<void*>(state->Rwb_.coeffs().data())
        //     << " (" << state->Rwb_.coeffs().transpose() << ")"
        //     << ", Vw_: " << static_cast<void*>(state->Vw_.data())
        //     << " (" << state->Vw_.transpose() << ")"
        //     << ", ba_: " << static_cast<void*>(state->ba_.data())
        //     << " (" << state->ba_.transpose() << ")"
        //     << ", bg_: " << static_cast<void*>(state->bg_.data())
        //     << " (" << state->bg_.transpose() << ")";
    }
    // for (auto last_block_address : last_marginalization_parameter_blocks_)
    //     LOG(INFO) << "last_block_address: " << last_block_address << " " << last_block_address[0];
    // 边缘化残差
    // The prior factor
    if (last_marginalization_info_ && last_marginalization_info_->isValid())
    {
        auto factor = new MarginalizationFactor(last_marginalization_info_);
        problem.AddResidualBlock(factor, nullptr, last_marginalization_parameter_blocks_);
    }

    // GNSS约束
    for (size_t i = 0; i < window_states.size(); ++i)
    {
        if (window_states[i]->cur_gnss_data_.time_ > 0)
        {
            auto gnss_data = window_states[i]->cur_gnss_data_;
            ceres::CostFunction *gnss_cost =
                new GNSSResidual(Eigen::Vector3d(gnss_data.x_, gnss_data.y_, gnss_data.z_), param_ptr_);
            problem.AddResidualBlock(gnss_cost, nullptr, window_states[i]->twb_.data());
        }
    }

    // Wheel约束
    for (size_t i = 0; i < window_states.size(); ++i)
    {
        if (window_states[i]->cur_wheel_data_.time_ > 0)
        {
            auto wheel_data = window_states[i]->cur_wheel_data_;
            Eigen::Vector3d Vb((wheel_data.lv_ + wheel_data.rv_) * 0.5, 0.0, 0.0);
            ceres::CostFunction *wheel_cost =
                new BaseVelocityResidual(Vb, param_ptr_);
            problem.AddResidualBlock(
                wheel_cost, nullptr,
                window_states[i]->Vw_.data(),
                window_states[i]->Rwb_.coeffs().data());
        }
    }

    // IMU预积分约束（窗口内相邻状态）
    for (size_t i = 1; i < window_states.size(); ++i)
    {
        auto state_i = window_states[i - 1];
        auto state_j = window_states[i];
        if (param_ptr_->state_type_ == 0)
        {
            std::shared_ptr<IMUPreintegration> preint_ptr =
                std::dynamic_pointer_cast<IMUPreintegration>(window_states[i]->preint_);
            ceres::CostFunction *imu_cost = new IMUPreintegrationResidual(preint_ptr, param_ptr_);
            problem.AddResidualBlock(imu_cost, nullptr,
                                     state_i->twb_.data(), state_i->Rwb_.coeffs().data(), state_i->Vw_.data(), state_i->ba_.data(), state_i->bg_.data(),
                                     state_j->twb_.data(), state_j->Rwb_.coeffs().data(), state_j->Vw_.data(), state_j->ba_.data(), state_j->bg_.data());
            // 打印最后一个 preint_ptr 信息
            // if (i == window_states.size() - 1 && preint_ptr)
            // {
            //     LOG(INFO) << "preint_ptr sum_dt: " << preint_ptr->sum_dt_;
            //     LOG(INFO) << "preint_ptr delta_p: " << preint_ptr->delta_p_.transpose();
            //     LOG(INFO) << "preint_ptr delta_v: " << preint_ptr->delta_v_.transpose();
            //     LOG(INFO) << "preint_ptr delta_q (wxyz): " << preint_ptr->delta_q_.w() << ", "
            //               << preint_ptr->delta_q_.x() << ", " << preint_ptr->delta_q_.y() << ", " << preint_ptr->delta_q_.z();
            //     LOG(INFO) << "preint_ptr covariance max: " << preint_ptr->covariance_.maxCoeff()
            //               << " min: " << preint_ptr->covariance_.minCoeff();
            // }
        }
        else if (param_ptr_->state_type_ == 1)
        {
            // 仅轮速计
            std::shared_ptr<WheelPreintegration> preint_ptr =
                std::dynamic_pointer_cast<WheelPreintegration>(window_states[i]->preint_);
            ceres::CostFunction *wheel_cost = new WheelPreintegrationResidual(preint_ptr, param_ptr_);
            problem.AddResidualBlock(wheel_cost, nullptr,
                                     state_i->twb_.data(), state_i->Rwb_.coeffs().data(),
                                     state_j->twb_.data(), state_j->Rwb_.coeffs().data());
        }
        else if (param_ptr_->state_type_ == 2)
        {
            // imu+轮速计
            std::shared_ptr<WheelIMUPreintegration> preint_ptr =
                std::dynamic_pointer_cast<WheelIMUPreintegration>(window_states[i]->preint_);
            ceres::CostFunction *wheelimu_cost = new WheelIMUPreintegrationResidual(preint_ptr, param_ptr_);
            problem.AddResidualBlock(wheelimu_cost, nullptr,
                                     state_i->twb_.data(), state_i->Rwb_.coeffs().data(), state_i->bg_.data(),
                                     state_j->twb_.data(), state_j->Rwb_.coeffs().data(), state_j->bg_.data());
        }
    }

    // 视觉重投影约束
    inv_depths_.clear();
    depth_feats_.clear();
    inv_depth_index_.clear();
    if (vins_feature_manager_ptr_ && param_ptr_->use_camera_)
    {
        ceres::LossFunction *loss_function = new ceres::HuberLoss(1.0);
        inv_depths_.reserve(vins_feature_manager_ptr_->feature.size());
        depth_feats_.reserve(vins_feature_manager_ptr_->feature.size());

        int feature_index = -1;
        int residual_count = 0;

        for (auto &it_per_id : vins_feature_manager_ptr_->feature)
        {
            int used_num = static_cast<int>(it_per_id.feature_per_frame.size());
            if (!(used_num >= 2 && it_per_id.start_frame < static_cast<int>(window_states.size()) - 2))
                continue;

            if (it_per_id.estimated_depth <= 0)
                continue;

            ++feature_index;

            // 使用逆深度参数
            double inv_depth = 1.0 / it_per_id.estimated_depth;
            inv_depths_.push_back(inv_depth);
            depth_feats_.push_back(&it_per_id);
            inv_depth_index_[&it_per_id] = inv_depths_.size() - 1;

            double *inv_depth_ptr = &inv_depths_.back();
            problem.AddParameterBlock(inv_depth_ptr, 1);

            int imu_i = it_per_id.start_frame;
            int imu_j = imu_i - 1;
            Eigen::Vector3d pts_i = it_per_id.feature_per_frame[0].point;

            for (const auto &it_per_frame : it_per_id.feature_per_frame)
            {
                imu_j++;
                if (imu_i == imu_j)
                    continue;

                Eigen::Vector3d pts_j = it_per_frame.point;

                if (imu_j >= static_cast<int>(window_states.size()) || imu_i < 0)
                    LOG(INFO) << "imu_j out of range: " << imu_j << ",imu_i" << imu_i << ", window size: " << window_states.size();
                ProjectionFactor *f = new ProjectionFactor(pts_i, pts_j, param_ptr_);
                problem.AddResidualBlock(
                    f, loss_function,
                    window_states[imu_i]->twb_.data(),
                    window_states[imu_i]->Rwb_.coeffs().data(),
                    window_states[imu_j]->twb_.data(),
                    window_states[imu_j]->Rwb_.coeffs().data(),
                    inv_depth_ptr);

                ++residual_count;
            }
        }

        LOG(INFO) << "逆深度参数个数: " << inv_depths_.size()
                  << ", 视觉重投影残差个数: " << residual_count;
    }

    // auto cur_state = window_states.back();
    // LOG(INFO) << "优化前位姿: " << cur_state->twb_.transpose();
    // LOG(INFO) << "优化前速度: " << cur_state->Vw_.transpose();
    // LOG(INFO) << "优化前加速度零偏: " << cur_state->ba_.transpose();
    // LOG(INFO) << "优化前陀螺零偏: " << cur_state->bg_.transpose();
    // LOG(INFO) << "优化前旋转矩阵:\n"
    //             << cur_state->Rwb_ << cur_state->Rwb_.coeffs().transpose();

    // 优化
    ceres::Solver::Options options;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.num_threads = 4;
    options.max_num_iterations = 24;
    ceres::Solver::Summary summary;

    /**
    // 打印误差
    std::vector<ceres::ResidualBlockId> residual_block_ids;
    problem.GetResidualBlocks(&residual_block_ids);
    int residual_block_idx = 0;
    for (const auto& residual_block : residual_block_ids) {
        std::vector<double*> parameter_blocks;
        problem.GetParameterBlocksForResidualBlock(residual_block, &parameter_blocks);
        const ceres::CostFunction* cost_function = problem.GetCostFunctionForResidualBlock(residual_block);
        int num_residuals = cost_function->num_residuals();
        std::vector<const double*> parameter_block_ptrs(parameter_blocks.begin(), parameter_blocks.end());
        std::vector<double> residuals(num_residuals);
        cost_function->Evaluate(parameter_block_ptrs.data(), residuals.data(), nullptr);
        double norm = Eigen::Map<Eigen::VectorXd>(residuals.data(), num_residuals).norm();

        std::ostringstream oss;
        oss << "ResidualBlock[" << residual_block_idx << "] residuals: [";
        for (size_t i = 0; i < residuals.size(); ++i) {
            oss << residuals[i];
            if (i != residuals.size() - 1) oss << ", ";
        }
        oss << "] | param ids: ";
        for (auto* ptr : parameter_blocks) {
            oss << static_cast<const void*>(ptr) << " ";
        }
        LOG(INFO) << oss.str();
        residual_block_idx++;
    }
    */

    // LOG(INFO) << "优化前状态：";
    ceres::Solve(options, &problem, &summary);

    // 参考 VINS：优化后对齐到首帧 yaw/位置（保持第1帧yaw与位置不变）
    const Eigen::Matrix3d R0_after = window_states[0]->Rwb_.toRotationMatrix();
    const Eigen::Vector3d ypr_after = Converter::R2ypr(R0_after);
    const Eigen::Vector3d P0_after = window_states[0]->twb_;
    const double y_diff = normalize_angle_deg(origin_ypr.x() - ypr_after.x());
    Eigen::Matrix3d rot_diff = Converter::ypr2R(Eigen::Vector3d(y_diff, 0.0, 0.0));
    LOG(INFO) << "优化后首帧 yaw 对齐差值: " << y_diff << " deg";
    LOG(INFO) << "origin_ypr: " << origin_ypr.transpose() << " deg";
    LOG(INFO) << "ypr_after: " << ypr_after.transpose() << " deg";
    // 接近万象节死锁时，直接用首帧姿态对齐
    if (std::abs(std::abs(origin_ypr.y()) - 90.0) < 1.0 || std::abs(std::abs(ypr_after.y()) - 90.0) < 1.0)
    {
        LOG(INFO) << "euler singular point!";
        rot_diff = origin_R0 * R0_after.transpose();
    }
    // rot_diff = origin_R0 * R0_after.transpose();
    for (auto &state : window_states)
    {
        state->Rwb_ = Eigen::Quaterniond(rot_diff * state->Rwb_.toRotationMatrix());
        state->twb_ = rot_diff * (state->twb_ - P0_after) + origin_P0;
        state->Vw_ = rot_diff * state->Vw_;
    }
    LOG(INFO) << "window_states[0]->Rwb_.toRotationMatrix()" << Converter::R2ypr(window_states[0]->Rwb_.toRotationMatrix()).transpose();
    LOG(INFO) << "origin_P0: " << origin_P0.transpose();
    LOG(INFO) << "P0_after: " << P0_after.transpose();
    LOG(INFO) << "window_states[0]->twb_: " << window_states[0]->twb_.transpose();
    auto rpy_delta = Converter::R2ypr(window_states[0]->Rwb_.toRotationMatrix()) - origin_ypr;
    if (abs(rpy_delta.y()) > 2.0 || abs(rpy_delta.z()) > 2.0)
    {
        LOG(INFO) << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << rpy_delta.transpose();
    }
    if (vins_feature_manager_ptr_ && param_ptr_->use_camera_)
    {
        for (size_t i = 0; i < depth_feats_.size(); ++i)
        {
            if (inv_depths_[i] > 0)
                depth_feats_[i]->estimated_depth = 1.0 / inv_depths_[i];
        }
    }
    // cur_state = window_states.back();
    // LOG(INFO) << "优化后位姿: " << cur_state->twb_.transpose();
    // LOG(INFO) << "优化后速度: " << cur_state->Vw_.transpose();
    // LOG(INFO) << "优化后加速度零偏: " << cur_state->ba_.transpose();
    // LOG(INFO) << "优化后陀螺零偏: " << cur_state->bg_.transpose();
    // LOG(INFO) << "优化后旋转矩阵:\n"
    //             << cur_state->Rwb_ << cur_state->Rwb_.coeffs().transpose();
    // auto gnss_data = cur_state->cur_gnss_data_;
    // LOG(INFO) << "GNSS: " << std::to_string(gnss_data.time_) << ", " << gnss_data.x_ << ", " << gnss_data.y_ << ", " << gnss_data.z_;
}
// todo add other sensor
void Optimizer::Run()
{
    std::ofstream result_file;
    result_file.open("./result_file.txt");
    // 循环读数据
    while (1)
    {
        if (!initialized_)
        {
            if (initializers_ptr_->Initialization())
            {
                vins_feature_manager_ptr_ = initializers_ptr_->vins_feature_manager_ptr_;
                initialized_ = true;
                LOG(INFO) << state_manager_ptr_->GetAllStates().size() << " 个初始状态已加入滑动窗口";
            }
            else
            {
                usleep(1000);
                continue;
            }
        }

        bool slide_old = true;
        std::shared_ptr<State> last_state;
        state_manager_ptr_->GetNearestState(last_state);

        // ==========================================
        // 模式 1: 有相机 (Camera Mode)
        // 以图像时间戳为主驱动
        // ==========================================
        if (param_ptr_->use_camera_)
        {
            FeatureData feature_data;
            if (!data_manager_ptr_->GetNewFeatureData(feature_data, last_feature_data_.time_))
            {
                usleep(1000);
                continue;
            }

            double dt = std::abs(last_state->time_ - feature_data.time_);

            // 时间间隔过小，仅更新最后状态的特征，不生成新关键帧
            if (dt <= 0.02)
            {
                continue;
            }

            // 创建预积分
            auto preint = predictor_ptr_->CreatePreintegration(
                last_state->time_, feature_data.time_,
                last_state->ba_, last_state->bg_);

            if (!preint)
            {
                usleep(1000);
                continue;
            }

            // 预测新状态
            std::shared_ptr<State> new_state = preint->predict(last_state);

            // [辅助] 查找附近 GNSS 数据
            if (param_ptr_->use_gnss_)
            {
                GNSSData gnss_data;
                // 查找图像时间戳附近的GNSS
                if (data_manager_ptr_->GetDatas(gnss_data, feature_data.time_))
                {
                    // 仅当时间偏差很小时才认为有效 (例如 50ms)
                    if (std::abs(gnss_data.time_ - feature_data.time_) < 0.05)
                    {
                        coo_trans_ptr_->getENH(
                            gnss_data.lat_, gnss_data.lon_, gnss_data.h_,
                            gnss_data.x_, gnss_data.y_, gnss_data.z_);

                        new_state->cur_gnss_data_ = gnss_data;
                        if (viewer_ptr_)
                            viewer_ptr_->DrawGps(Eigen::Vector3d(gnss_data.x_, gnss_data.y_, gnss_data.z_));
                    }
                }
            }

            // [辅助] 查找附近 轮速 数据
            if (param_ptr_->wheel_use_type_ == 2)
            {
                WheelData cur_wheel_data;
                if (data_manager_ptr_->GetDatas(cur_wheel_data, feature_data.time_))
                {
                    new_state->cur_wheel_data_ = cur_wheel_data;
                }
            }

            // 更新状态
            state_manager_ptr_->PushState(new_state);
            last_state = new_state;

            // 视觉视差处理
            std::vector<std::shared_ptr<State>> window_states = state_manager_ptr_->GetAllStates();
            // 此处提前加入了状态，因此索引为 size-1
            slide_old = vins_feature_manager_ptr_->addFeatureCheckParallax(
                window_states.size() - 1, feature_data.features_, last_state);
            vins_feature_manager_ptr_->triangulate();
            // 更新追踪器
            last_state->feature_data_ = feature_data;
            LOG(INFO) << "新关键帧: " << std::to_string(feature_data.time_)
                      << ", 特征点数: " << feature_data.features_.size() << " " << window_states.size();
            last_feature_data_ = feature_data;
        }
        // ==========================================
        // 模式 2: 无相机，仅有 GNSS (GNSS Mode)
        // 以GNSS时间戳为主驱动
        // ==========================================
        else if (param_ptr_->use_gnss_)
        {
            GNSSData cur_gnss_data;
            if (!data_manager_ptr_->GetLastGNSSData(cur_gnss_data, last_gnss_data_.time_))
            {
                usleep(1000);
                continue;
            }

            // 转换坐标用于显示或计算
            coo_trans_ptr_->getENH(
                cur_gnss_data.lat_, cur_gnss_data.lon_, cur_gnss_data.h_,
                cur_gnss_data.x_, cur_gnss_data.y_, cur_gnss_data.z_);

            if (viewer_ptr_)
                viewer_ptr_->DrawGps(Eigen::Vector3d(cur_gnss_data.x_, cur_gnss_data.y_, cur_gnss_data.z_));

            double dt = std::abs(last_state->time_ - cur_gnss_data.time_);

            // 时间间隔过小，仅更新状态数据
            if (dt <= 0.04)
            {
                last_state->cur_gnss_data_ = cur_gnss_data;
                last_gnss_data_ = cur_gnss_data;
                continue;
            }

            // 创建预积分
            auto preint = predictor_ptr_->CreatePreintegration(
                last_state->time_, cur_gnss_data.time_,
                last_state->ba_, last_state->bg_);

            if (!preint)
            {
                usleep(1000);
                continue;
            }

            // 预测新状态
            std::shared_ptr<State> new_state = preint->predict(last_state);
            new_state->cur_gnss_data_ = cur_gnss_data;

            // [辅助] 查找附近 轮速 数据
            if (param_ptr_->wheel_use_type_ == 2)
            {
                WheelData cur_wheel_data;
                if (data_manager_ptr_->GetDatas(cur_wheel_data, cur_gnss_data.time_))
                {
                    new_state->cur_wheel_data_ = cur_wheel_data;
                }
            }

            // 更新状态
            state_manager_ptr_->PushState(new_state);
            last_state = new_state;
            last_gnss_data_ = cur_gnss_data;
        }
        // ==========================================
        // 无有效主传感器
        // ==========================================
        else
        {
            usleep(1000);
            continue;
        }

        // 统一执行优化与滑窗
        Optimization();
        SlideWindow(slide_old);

        // 显示相关
        if (viewer_ptr_)
        {
            viewer_ptr_->DrawWheelPose(last_state->Rwb_.toRotationMatrix(), last_state->twb_);
            if (param_ptr_->use_camera_)
            {
                // 显示相机位姿跟三维点
                std::vector<std::shared_ptr<State>> window_states = state_manager_ptr_->GetAllStates();
                std::vector<std::pair<Eigen::Matrix3d, Eigen::Vector3d>> cameras;
                for (auto state : window_states)
                {
                    // 将 IMU 位姿转换为相机位姿: T_wc = T_wb * T_bc
                    Eigen::Matrix3d Rwc = state->Rwb_.toRotationMatrix() * param_ptr_->Rbc_;
                    Eigen::Vector3d twc = state->Rwb_.toRotationMatrix() * param_ptr_->tbc_ + state->twb_;
                    cameras.push_back(std::make_pair(Rwc, twc));
                }
                viewer_ptr_->DrawCameras(cameras);

                // 绘制所有可显示的三维点（已初始化且数值有效）
                std::vector<Eigen::Vector3d> map_points_all;
                const auto &features = vins_feature_manager_ptr_->feature;
                map_points_all.reserve(features.size());

                for (const auto &it_per_id : features)
                {
                    // 仅显示深度已收敛/有效的点
                    if (it_per_id.estimated_depth > 0)
                    {
                        int imu_i = it_per_id.start_frame;
                        if (imu_i >= window_states.size())
                            continue;
                        // 1. 恢复相机坐标系下的点 Pc = normalized_point * depth
                        Eigen::Vector3d pts_c = it_per_id.feature_per_frame[0].point * it_per_id.estimated_depth;
                        // 2. 转到 IMU 坐标系 Pb = R_bc * Pc + t_bc
                        Eigen::Vector3d pts_b = param_ptr_->Rbc_ * pts_c + param_ptr_->tbc_;
                        // 3. 转到 世界 坐标系 Pw = R_wb * Pb + t_wb
                        Eigen::Vector3d pts_w = window_states[imu_i]->Rwb_ * pts_b + window_states[imu_i]->twb_;
                        map_points_all.push_back(pts_w);
                    }
                }

                LOG(INFO) << "Displaying map points #: " << map_points_all.size();
                viewer_ptr_->DrawFeatures(map_points_all);
            }
        }

        // result_file << state_ptr->twb_.x() << "," << state_ptr->twb_.y() << "," << state_ptr->twb_.z() << std::endl;
        usleep(100);
    }
}
