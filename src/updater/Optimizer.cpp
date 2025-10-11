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
bool QLocalParameterization::ComputeJacobian(const double *x, double *jacobian) const
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


void Optimizer::SlideWindow()
{
    std::vector<std::shared_ptr<State>> window_states = state_manager_ptr_->GetAllStates();
    if (window_states.size() <= 20)
        return;
    size_t num_marg = window_states.size() - 20;
    std::shared_ptr<MarginalizationInfo> marginalization_info = std::make_shared<MarginalizationInfo>();

    // 指定每个参数块独立的ID, 用于索引参数
    // key 表示参数块的内存地址, value表示参数块的ID
    std::unordered_map<long, long> parameters_ids;
    parameters_ids.clear();
    long parameters_id = 0;

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
            if (preintegration_type_ == 0)
            {
                parameters_ids[reinterpret_cast<long>(window_states[i]->twb_.data())] = parameters_id++;
                parameters_ids[reinterpret_cast<long>(window_states[i]->Rwb_.coeffs().data())] = parameters_id++;
                parameters_ids[reinterpret_cast<long>(window_states[i]->Vw_.data())] = parameters_id++;
                parameters_ids[reinterpret_cast<long>(window_states[i]->ba_.data())] = parameters_id++;
                parameters_ids[reinterpret_cast<long>(window_states[i]->bg_.data())] = parameters_id++;
            }
            // 轮速计模式
            else if (preintegration_type_ == 1)
            {
                parameters_ids[reinterpret_cast<long>(window_states[i]->twb_.data())] = parameters_id++;
                parameters_ids[reinterpret_cast<long>(window_states[i]->Rwb_.coeffs().data())] = parameters_id++;
            }
            // IMU+轮速计模式
            else if (preintegration_type_ == 2)
            {
                parameters_ids[reinterpret_cast<long>(window_states[i]->twb_.data())] = parameters_id++;
                parameters_ids[reinterpret_cast<long>(window_states[i]->Rwb_.coeffs().data())] = parameters_id++;
                parameters_ids[reinterpret_cast<long>(window_states[i]->bg_.data())] = parameters_id++;
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
        // 存放本次要被边缘化的参数块的index
        std::vector<int> marginalized_index;
        for (size_t i = 0; i < num_marg; i++)
        {
            
            for (size_t k = 0; k < last_marginalization_parameter_blocks_.size(); k++)
            {
                // 用当前窗口状态的参数指针进行匹配
                if ((preintegration_type_ == 0 &&
                    (last_marginalization_parameter_blocks_[k] == window_states[i]->twb_.data() ||
                    last_marginalization_parameter_blocks_[k] == window_states[i]->Rwb_.coeffs().data() ||
                    last_marginalization_parameter_blocks_[k] == window_states[i]->Vw_.data() ||
                    last_marginalization_parameter_blocks_[k] == window_states[i]->ba_.data() ||
                    last_marginalization_parameter_blocks_[k] == window_states[i]->bg_.data()))
                || (preintegration_type_ == 1 &&
                    (last_marginalization_parameter_blocks_[k] == window_states[i]->twb_.data() ||
                    last_marginalization_parameter_blocks_[k] == window_states[i]->Rwb_.coeffs().data()))
                || (preintegration_type_ == 2 &&
                    (last_marginalization_parameter_blocks_[k] == window_states[i]->twb_.data() ||
                    last_marginalization_parameter_blocks_[k] == window_states[i]->Rwb_.coeffs().data() ||
                    last_marginalization_parameter_blocks_[k] == window_states[i]->bg_.data())))
                {
                    marginalized_index.push_back((int)k);
                }
            }
        }

        auto factor = std::make_shared<MarginalizationFactor>(last_marginalization_info_);
        auto residual = std::make_shared<ResidualBlockInfo>(
            factor, nullptr, last_marginalization_parameter_blocks_, marginalized_index);
        marginalization_info->addResidualBlockInfo(residual);
    }

    for (size_t i = 0; i < num_marg; ++i)
    {
        // GNSS因子
        if (window_states[i]->cur_gnss_data_.time_ > 0)
        {
            auto gnss_data = window_states[i]->cur_gnss_data_;
            auto factor = std::make_shared<GNSSResidual>(
                Eigen::Vector3d(gnss_data.x_, gnss_data.y_, gnss_data.z_), param_ptr_);
            auto residual = std::make_shared<ResidualBlockInfo>(
                factor, nullptr, std::vector<double *>{window_states[i]->twb_.data()}, std::vector<int>{0});
            marginalization_info->addResidualBlockInfo(residual);
        }

        // 预积分因子
        std::vector<int> marg_index;
        auto preint_ptr = window_states[i + 1]->preint_;
        if (preintegration_type_ == 0) // IMU
        {
            if (i == (num_marg - 1))
            {
                marg_index = {0, 1, 2, 3, 4};
            }
            else
            {
                marg_index = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
            }
            auto factor = std::make_shared<IMUPreintegrationResidual>(
                std::dynamic_pointer_cast<IMUPreintegration>(preint_ptr), param_ptr_);
            auto residual = std::make_shared<ResidualBlockInfo>(
                factor, nullptr,
                std::vector<double *>{window_states[i]->twb_.data(), window_states[i]->Rwb_.coeffs().data(),
                    window_states[i]->Vw_.data(), window_states[i]->ba_.data(), window_states[i]->bg_.data(),
                    window_states[i + 1]->twb_.data(), window_states[i + 1]->Rwb_.coeffs().data(),
                    window_states[i + 1]->Vw_.data(), window_states[i + 1]->ba_.data(), window_states[i + 1]->bg_.data()},
                marg_index);
            marginalization_info->addResidualBlockInfo(residual);
        }
        else if (preintegration_type_ == 1) // 轮速计
        {
            if (i == (num_marg - 1))
            {
                marg_index = {0, 1};
            }
            else
            {
                marg_index = {0, 1, 2, 3};
            }
            auto factor = std::make_shared<WheelPreintegrationResidual>(
                std::dynamic_pointer_cast<WheelPreintegration>(preint_ptr), param_ptr_);
            auto residual = std::make_shared<ResidualBlockInfo>(
                factor, nullptr,
                std::vector<double *>{window_states[i]->twb_.data(), window_states[i]->Rwb_.coeffs().data(),
                                    window_states[i + 1]->twb_.data(), window_states[i + 1]->Rwb_.coeffs().data()},
                marg_index);
            marginalization_info->addResidualBlockInfo(residual);
        }
        else if (preintegration_type_ == 2) // IMU+轮速计
        {
            if (i == (num_marg - 1))
            {
                marg_index = {0, 1, 2};
            }
            else
            {
                marg_index = {0, 1, 2, 3, 4, 5};
            }
            auto factor = std::make_shared<WheelIMUPreintegrationResidual>(
                std::dynamic_pointer_cast<WheelIMUPreintegration>(preint_ptr), param_ptr_);
            auto residual = std::make_shared<ResidualBlockInfo>(
                factor, nullptr,
                std::vector<double *>{window_states[i]->twb_.data(), window_states[i]->Rwb_.coeffs().data(), window_states[i]->bg_.data(),
                                    window_states[i + 1]->twb_.data(), window_states[i + 1]->Rwb_.coeffs().data(), window_states[i + 1]->bg_.data()},
                marg_index);
            marginalization_info->addResidualBlockInfo(residual);
        }
    }

    // // 重投影因子, 最老的关键帧
    // // The visual reprojection factors
    // frame = map_->keyframes().at(keyframeids[0]);
    // auto features = frame->features();

    // auto loss_function = std::make_shared<ceres::HuberLoss>(1.0);
    // for (auto const &feature : features)
    // {
    //     auto mappoint = feature.second->getMapPoint();
    //     if (feature.second->isOutlier() || !mappoint || mappoint->isOutlier())
    //     {
    //         continue;
    //     }

    //     auto ref_frame = mappoint->referenceFrame();
    //     if (ref_frame != frame)
    //     {
    //         continue;
    //     }

    //     auto ref_frame_pc = camera_->pixel2cam(mappoint->referenceKeypoint());
    //     size_t ref_frame_index = getStateDataIndex(ref_frame->stamp());
    //     if (ref_frame_index < 0)
    //     {
    //         continue;
    //     }

    //     double *invdepth = &invdepthlist_[mappoint->id()];

    //     auto ref_feature = ref_frame->features().find(mappoint->id())->second;

    //     auto observations = mappoint->observations();
    //     for (auto &observation : observations)
    //     {
    //         auto obs_feature = observation.lock();
    //         if (!obs_feature || obs_feature->isOutlier())
    //         {
    //             continue;
    //         }
    //         auto obs_frame = obs_feature->getFrame();
    //         if (!obs_frame || !obs_frame->isKeyFrame() || !map_->isKeyFrameInMap(obs_frame) ||
    //             (obs_frame == ref_frame))
    //         {
    //             continue;
    //         }

    //         auto obs_frame_pc = camera_->pixel2cam(obs_feature->keyPoint());
    //         size_t obs_frame_index = getStateDataIndex(obs_frame->stamp());

    //         if ((obs_frame_index < 0) || (ref_frame_index == obs_frame_index))
    //         {
    //             LOGE << "Wrong matched mapoint keyframes " << Logging::doubleData(ref_frame->stamp()) << " with "
    //                  << Logging::doubleData(obs_frame->stamp());
    //             continue;
    //         }

    //         auto factor = std::make_shared<ReprojectionFactor>(
    //             ref_frame_pc, obs_frame_pc, ref_feature->velocityInPixel(), obs_feature->velocityInPixel(),
    //             ref_frame->timeDelay(), obs_frame->timeDelay(), optimize_reprojection_error_std_);
    //         auto residual = std::make_shared<ResidualBlockInfo>(factor, nullptr,
    //                                                             vector<double *>{window_states[ref_frame_index].pose,
    //                                                                              window_states[obs_frame_index].pose,
    //                                                                              extrinsic_, invdepth, &extrinsic_[7]},
    //                                                             vector<int>{0, 3});
    //         marginalization_info->addResidualBlockInfo(residual);
    //     }
    // }

    // 边缘化处理
    // Do marginalization
    marginalization_info->marginalization();

    // 保留的数据, address 存放parameters_id : 参数地址
    // Update the address
    std::unordered_map<long, double *> address;
    for (size_t k = num_marg; k < window_states.size(); k++)
    {
        // IMU模式
        if (preintegration_type_ == 0)
        {
            address[parameters_ids[reinterpret_cast<long>(window_states[k]->twb_.data())]] = window_states[k]->twb_.data();
            address[parameters_ids[reinterpret_cast<long>(window_states[k]->Rwb_.coeffs().data())]] = window_states[k]->Rwb_.coeffs().data();
            address[parameters_ids[reinterpret_cast<long>(window_states[k]->Vw_.data())]] = window_states[k]->Vw_.data();
            address[parameters_ids[reinterpret_cast<long>(window_states[k]->ba_.data())]] = window_states[k]->ba_.data();
            address[parameters_ids[reinterpret_cast<long>(window_states[k]->bg_.data())]] = window_states[k]->bg_.data();
        }
        // 轮速计模式
        else if (preintegration_type_ == 1)
        {
            address[parameters_ids[reinterpret_cast<long>(window_states[k]->twb_.data())]] = window_states[k]->twb_.data();
            address[parameters_ids[reinterpret_cast<long>(window_states[k]->Rwb_.coeffs().data())]] = window_states[k]->Rwb_.coeffs().data();
        }
        // IMU+轮速计模式
        else if (preintegration_type_ == 2)
        {
            address[parameters_ids[reinterpret_cast<long>(window_states[k]->twb_.data())]] = window_states[k]->twb_.data();
            address[parameters_ids[reinterpret_cast<long>(window_states[k]->Rwb_.coeffs().data())]] = window_states[k]->Rwb_.coeffs().data();
            address[parameters_ids[reinterpret_cast<long>(window_states[k]->bg_.data())]] = window_states[k]->bg_.data();
        }
    }
    // 本次边缘化后将会受到约束的参数块
    last_marginalization_parameter_blocks_ = marginalization_info->getParameterBlocks(address);
    last_marginalization_info_ = std::move(marginalization_info);

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
    state_manager_ptr_->PopFrontState();
}
void Optimizer::Optimization()
{
    const int window_size = 2;
    std::vector<std::shared_ptr<State>> window_states = state_manager_ptr_->GetAllStates();
    if (window_states.size() < window_size)
        return;

    ceres::Problem problem;

    // LOG(INFO) << "窗口大小: " << window_states.size();
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
    if (last_marginalization_info_ && last_marginalization_info_->isValid()) {
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

    // IMU预积分约束（窗口内相邻状态）
    for (size_t i = 1; i < window_states.size(); ++i)
    {
        auto state_i = window_states[i - 1];
        auto state_j = window_states[i];
        if (preintegration_type_ == 0)
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
        else if (preintegration_type_ == 1)
        {
            // 仅轮速计
            std::shared_ptr<WheelPreintegration> preint_ptr =
                std::dynamic_pointer_cast<WheelPreintegration>(window_states[i]->preint_);
            ceres::CostFunction *wheel_cost = new WheelPreintegrationResidual(preint_ptr, param_ptr_);
            problem.AddResidualBlock(wheel_cost, nullptr,
                state_i->twb_.data(), state_i->Rwb_.coeffs().data(),
                state_j->twb_.data(), state_j->Rwb_.coeffs().data());
        }
        else if (preintegration_type_ == 2)
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
                initialized_ = true;
            else
                continue;
        }

        std::shared_ptr<State> last_state;
        state_manager_ptr_->GetNearestState(last_state);
        bool need_opt = false;
        GNSSData cur_gnss_data;
        if (param_ptr_->use_gnss_ &&
            data_manager_ptr_->GetLastGNSSData(cur_gnss_data, last_gnss_data_.time_))
        {
            coo_trans_ptr_->getENH(
                cur_gnss_data.lat_, cur_gnss_data.lon_, cur_gnss_data.h_,
                cur_gnss_data.x_, cur_gnss_data.y_, cur_gnss_data.z_);
            if (viewer_ptr_)
                viewer_ptr_->DrawGps(Eigen::Vector3d(cur_gnss_data.x_, cur_gnss_data.y_, cur_gnss_data.z_));
            double dt = std::abs(last_state->time_ - cur_gnss_data.time_);
            if (dt > 0.04)
            {
                auto preint = predictor_ptr_->CreatePreintegration(
                    last_state->time_, cur_gnss_data.time_,
                    last_state->ba_, last_state->bg_);
                if (preint)
                {
                    std::shared_ptr<State> new_state = preint->predict(last_state);
                    state_manager_ptr_->PushState(new_state);
                    last_state = new_state;
                    last_state->cur_gnss_data_ = cur_gnss_data;
                    last_gnss_data_ = cur_gnss_data;
                    need_opt = true;
                }
            }
            else
            {
                last_state->cur_gnss_data_ = cur_gnss_data;
                last_gnss_data_ = cur_gnss_data;
                need_opt = true;
            }
        }
        WheelData cur_wheel_data;
        if (param_ptr_->wheel_use_type_ == 2 &&
            data_manager_ptr_->GetLastWheelData(cur_wheel_data, last_wheel_data_.time_))
        {
            double dt = std::abs(last_state->time_ - cur_wheel_data.time_);
            if (dt > 0.04)
            {
                auto preint = predictor_ptr_->CreatePreintegration(
                    last_state->time_, cur_wheel_data.time_,
                    last_state->ba_, last_state->bg_);
                std::shared_ptr<State> new_state = preint->predict(last_state);
                state_manager_ptr_->PushState(new_state);
                last_state = new_state;
            }
            last_state->cur_wheel_data_ = cur_wheel_data;
            last_wheel_data_ = cur_wheel_data;
        }

        FeatureData feature_data;
        if (param_ptr_->use_camera_ &&
            data_manager_ptr_->GetNewFeatureData(feature_data, last_feature_data_.time_))
        {
            double dt = std::abs(last_state->time_ - feature_data.time_);
            if (dt > 0.04)
            {
                auto preint = predictor_ptr_->CreatePreintegration(
                    last_state->time_, feature_data.time_,
                    last_state->ba_, last_state->bg_);
                std::shared_ptr<State> new_state = preint->predict(last_state);
                state_manager_ptr_->PushState(new_state);
                last_state = new_state;
            }
            last_state->feature_data_ = feature_data;
            last_feature_data_ = feature_data;
        }
        if (!need_opt)
        {
            usleep(1000);
            continue;
        }
        Optimization();
        SlideWindow();
        if (viewer_ptr_) {
            viewer_ptr_->DrawWheelPose(last_state->Rwb_.toRotationMatrix(), last_state->twb_);
        }
            
        // result_file << state_ptr->twb_.x() << "," << state_ptr->twb_.y() << "," << state_ptr->twb_.z() << std::endl;
        usleep(100);
    }
}