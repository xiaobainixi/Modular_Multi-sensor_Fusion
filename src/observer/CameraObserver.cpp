#include "CameraObserver.h"

// todo feature_data较旧 state_ptr最新
// todo 预测观测多线程 cam_states_ 协方差等容易出现同时操作
bool CameraObserver::ComputeHZR(
    const FeatureData & feature_data, const std::shared_ptr<State> & state_ptr,
    Eigen::MatrixXd & H, Eigen::MatrixXd & Z, Eigen::MatrixXd &R)
{
    cam_states_next_id_++;
    // Add a new camera state to the state server.
    // 2. 注册新的相机状态到状态库中
    // 嗯。。。说人话就是找个记录的，不然咋更新
    state_manager_ptr_->cam_states_[cam_states_next_id_] = std::make_shared<CamState>(cam_states_next_id_);
    std::shared_ptr<CamState> cam_state_ptr = state_manager_ptr_->cam_states_[cam_states_next_id_];

    // 严格上讲这个时间不对，但是几乎没影响
    cam_state_ptr->time = feature_data.time_;
    cam_state_ptr->Rwc_ = feature_data.Rwc_;
    cam_state_ptr->twc_ = feature_data.twc_;

    // 记录第一次被估计的数据，不能被改变，因为改变了就破坏了之前的0空间
    cam_state_ptr->Rwc_null_ = cam_state_ptr->Rwc_;
    cam_state_ptr->twc_null_ = cam_state_ptr->twc_;

    // Update the covariance matrix of the state.
    // To simplify computation, the matrix J below is the nontrivial block
    // in Equation (16) in "A Multi-State Constraint Kalman Filter for Vision
    // -aided Inertial Navigation".


    // 3. 这个雅可比可以认为是cam0位姿相对于imu的状态量的求偏导,注意imu状态是最新的，cam状态是有延迟的
    // 此时我们首先要知道相机位姿是 Rcw  twc
    // Rwc = Rwb * Rbc   twc = twb + Rwb * tbc

    // Twb_old = Twb_new * Tnew_old
    // Rwb_old = Rwb_new * Rnew_old
    // twb_old = twb_new + Rwb_new * tnew_old
    // Rwc = Rwb_new * Rnew_old * Rbc   twc = twb_new + Rwb_new * tnew_old + Rwb_new * Rnew_old * tbc
    // twc = twb_new + Rwb_new * (tnew_old + Rnew_old * tbc)
    // 其中 Rnew_old = Rwb_new.t() * Rwb_old    tnew_old = Rwb_new.t() * (twb_old - twb_new) 这俩暂时认为是固定值
    Eigen::Matrix3d Rnew_old = state_ptr->Rwb_.transpose() * feature_data.Rwb_;
    Eigen::Vector3d tnew_old = state_ptr->Rwb_.transpose() * (feature_data.twb_ - state_ptr->twb_);
    Eigen::MatrixXd J = Eigen::MatrixXd::Zero(6, param_ptr_->STATE_DIM);
    // Rwc对Rwb_new的左扰动导数
    J.block<3, 3>(0, param_ptr_->ORI_INDEX_STATE_) = Eigen::Matrix3d::Identity();
    // Rcw对Rci的左扰动导数
    // J.block<3, 3>(0, 15) = Eigen::Matrix3d::Identity();

    // twc对Rwb_new的左扰动导数
    // twc = twb_new + Exp(φ) * Rwb_new * (tnew_old + Rnew_old * tbc)
    //     = twb_new + (I + φ^) * Rwb_new * (tnew_old + Rnew_old * tbc)
    //     = twb_new + Rwb_new * (tnew_old + Rnew_old * tbc) + φ^ * Rwb_new * (tnew_old + Rnew_old * tbc)
    //     = twi + Rwi * tic - (Rwb_new * (tnew_old + Rnew_old * tbc))^ * φ
    J.block<3, 3>(3, param_ptr_->ORI_INDEX_STATE_) = -Converter::SkewSymmetric(state_ptr->Rwb_ * (tnew_old + Rnew_old * param_ptr_->tbc_));

    // twc对twb_new的左扰动导数
    J.block<3, 3>(3, param_ptr_->POSI_INDEX) = Eigen::Matrix3d::Identity();
    // // twc对tic的左扰动导数
    // J.block<3, 3>(3, 18) = R_w_i.transpose();

    // 4. 增广协方差矩阵
    // 简单地说就是原来的协方差是 param_ptr_->STATE_DIM + 6n 维的，现在新来了一个伙计，维度要扩了
    // 并且对应位置的值要根据雅可比跟这个时刻（也就是最新时刻）的imu协方差计算
    // 4.1 扩展矩阵大小 conservativeResize函数不改变原矩阵对应位置的数值
    // Resize the state covariance matrix.
    size_t old_rows = state_ptr->C_.rows();
    size_t old_cols = state_ptr->C_.cols();
    state_ptr->C_.conservativeResize(old_rows + 6, old_cols + 6);

    // Rename some matrix blocks for convenience.
    // imu的协方差矩阵
    const Eigen::MatrixXd &P11 = 
        state_ptr->C_.block(0, 0, param_ptr_->STATE_DIM, param_ptr_->STATE_DIM);

    // imu相对于各个相机状态量的协方差矩阵（不包括最新的）
    const Eigen::MatrixXd &P12 =
        state_ptr->C_.block(0, param_ptr_->STATE_DIM, param_ptr_->STATE_DIM, old_cols - param_ptr_->STATE_DIM);

    // Fill in the augmented state covariance.
    // 4.2 计算协方差矩阵
    // 左下角
    state_ptr->C_.block(old_rows, 0, 6, old_cols) << J * P11, J * P12;

    // 右上角
    state_ptr->C_.block(0, old_cols, old_rows, 6) =
        state_ptr->C_.block(old_rows, 0, 6, old_cols).transpose();

    // 右下角，关于相机部分的J都是0所以省略了
    state_ptr->C_.block<6, 6>(old_rows, old_cols) =
        J * P11 * J.transpose();

    // Fix the covariance to be symmetric
    // 强制对称
    Eigen::MatrixXd state_cov_fixed = (state_ptr->C_ +
                                state_ptr->C_.transpose()) /
                                2.0;
    state_ptr->C_ = state_cov_fixed;


    // 这是个long long int 嗯。。。。直接当作int理解吧
    // 这个id会在 batchImuProcessing 更新
    // 1. 获取当前窗口内特征点数量
    int curr_feature_num = map_server.size();
    int tracked_feature_num = 0;

    // Add new observations for existing features or new
    // features in the map server.
    // 2. 添加新来的点，做的花里胡哨，其实就是在现有的特征管理里面找，
    // id已存在说明是跟踪的点，在已有的上面更新
    // id不存在说明新来的点，那么就新添加一个
    for (const auto &feature : feature_data.features_)
    {
        if (map_server.find(feature.id_) == map_server.end())
        {
            // This is a new feature.
            map_server[feature.id_] = Feature(feature.id_);
            map_server[feature.id_].observations[cam_states_next_id_] = feature.point_;
        }
        else
        {
            // This is an old feature.
            map_server[feature.id_].observations[cam_states_next_id_] = feature.point_;
            ++tracked_feature_num;
        }
    }

    // 这个东西计算了当前进来的跟踪的点中在总数里面的占比（进来的点有可能是新提的）
    double tracking_rate =
        static_cast<double>(tracked_feature_num) /
        static_cast<double>(curr_feature_num);

    // Remove the features that lost track.
    // BTW, find the size the final Jacobian matrix and residual vector.
    int jacobian_row_size = 0;
    // FeatureIDType 这是个long long int 嗯。。。。直接当作int理解吧
    std::vector<FeatureIDType> invalid_feature_ids(0);  // 无效点，最后要删的
    std::vector<FeatureIDType> processed_feature_ids(0);  // 待参与更新的点，用完也被无情的删掉

    int aa = 0, bb = 0, cc = 0, dd = 0;
    
    // 遍历所有特征管理里面的点，包括新进来的
    for (auto iter = map_server.begin();
            iter != map_server.end(); ++iter)
    {
        // Rename the feature to be checked.
        // 引用，改变feature相当于改变iter->second，类似于指针的效果
        auto &feature = iter->second;

        // Pass the features that are still being tracked.
        // 1. 这个点被当前状态观测到，说明这个点后面还有可能被跟踪
        // 跳过这些点
        if (feature.observations.find(cam_states_next_id_) !=
            feature.observations.end()) {
                aa++;
                continue;
            }
            

        // 2. 跟踪小于3帧的点，认为是质量不高的点
        // 也好理解，三角化起码要两个观测，但是只有两个没有其他观测来验证
        if (feature.observations.size() < 4)
        {
            bb++;
            invalid_feature_ids.push_back(feature.id_);
            continue;
        }

        // Check if the feature can be initialized if it
        // has not been.
        // 3. 如果这个特征没有被初始化，尝试去初始化
        // 初始化就是三角化
        if (!feature.is_initialized)
        {
            // 3.1 看看运动是否足够，没有足够视差或者平移小旋转多这种不符合三角化
            // 所以就不要这些点了
            if (!feature.checkMotion(state_manager_ptr_->cam_states_))
            {
                cc++;
                invalid_feature_ids.push_back(feature.id_);
                continue;
            }
            else
            {
                // 3.3 尝试三角化，失败也不要了
                if (!feature.initializePosition(state_manager_ptr_->cam_states_))
                {
                    dd++;
                    invalid_feature_ids.push_back(feature.id_);
                    continue;
                }
            }
        }

        // 4. 到这里表示这个点能用于更新，所以准备下一步计算
        // 一个观测代表一帧，一帧有左右两个观测
        // 也就是算重投影误差时维度将会是4 * feature.observations.size()
        // 这里为什么减3下面会提到
        jacobian_row_size += 2 * feature.observations.size() - 3;
        // 接下来要参与优化的点加入到这个变量中
        processed_feature_ids.push_back(feature.id_);
    }
    LOG(INFO) << aa << " " << bb << " " << cc << " " << dd;
    // cout << "invalid/processed feature #: " <<
    //   invalid_feature_ids.size() << "/" <<
    //   processed_feature_ids.size() << endl;
    // cout << "jacobian row #: " << jacobian_row_size << endl;

    // Remove the features that do not have enough measurements.
    // 5. 删掉非法点
    for (const auto &feature_id : invalid_feature_ids)
        map_server.erase(feature_id);

    // Return if there is no lost feature to be processed.
    if (processed_feature_ids.size() == 0) {
    } else {
        // 准备好误差相对于状态量的雅可比
        Eigen::MatrixXd H_x = Eigen::MatrixXd::Zero(jacobian_row_size,
                                        param_ptr_->STATE_DIM + 6 * state_manager_ptr_->cam_states_.size());
        Eigen::VectorXd r = Eigen::VectorXd::Zero(jacobian_row_size);
        int stack_cntr = 0;

        // Process the features which lose track.
        // 6. 处理特征点
        for (const auto &feature_id : processed_feature_ids)
        {
            auto &feature = map_server[feature_id];

            std::vector<int> cam_state_ids(0);
            for (const auto &measurement : feature.observations)
                cam_state_ids.push_back(measurement.first);

            Eigen::MatrixXd H_xj;
            Eigen::VectorXd r_j;
            // 6.1 计算雅可比，计算重投影误差
            featureJacobian(feature.id_, cam_state_ids, H_xj, r_j);

            // 6.2 卡方检验，剔除错误点，并不是所有点都用
            if (gatingTest(H_xj, r_j, cam_state_ids.size() - 1, state_ptr->C_))
            {
                H_x.block(stack_cntr, 0, H_xj.rows(), H_xj.cols()) = H_xj;
                r.segment(stack_cntr, r_j.rows()) = r_j;
                stack_cntr += H_xj.rows();
            }

            // Put an upper bound on the row size of measurement Jacobian,
            // which helps guarantee the executation time.
            // 限制最大更新量
            if (stack_cntr > 1500)
                break;
        }

        // resize成实际大小
        H_x.conservativeResize(stack_cntr, H_x.cols());
        r.conservativeResize(stack_cntr);

        // Perform the measurement update step.
        // 7. 使用误差及雅可比更新状态
        measurementUpdate(H_x, r, state_ptr);

        // Remove all processed features from the map.
        // 8. 删除用完的点
        for (const auto &feature_id : processed_feature_ids)
            map_server.erase(feature_id);
    }

    // 数量还不到该删的程度，配置文件里面是20个
    if (state_manager_ptr_->cam_states_.size() >= 20) {
        // Find two camera states to be removed.
        // 1. 找出该删的相机状态的id，两个
        std::vector<int> rm_cam_state_ids(0);
        // Move the iterator to the key position.
        // 1. 找到倒数第四个相机状态，作为关键状态
        auto key_cam_state_iter = state_manager_ptr_->cam_states_.end();
        for (int i = 0; i < 4; ++i)
            --key_cam_state_iter;

        // 倒数第三个相机状态
        auto cam_state_iter = key_cam_state_iter;
        ++cam_state_iter;

        // 序列中，第一个相机状态
        auto first_cam_state_iter = state_manager_ptr_->cam_states_.begin();

        // Pose of the key camera state.
        // 2. 关键状态的位姿
        const Eigen::Vector3d key_position =
            key_cam_state_iter->second->twc_;
        const Eigen::Matrix3d key_rotation = key_cam_state_iter->second->Rwc_;

        // Mark the camera states to be removed based on the
        // motion between states.
        // 3. 遍历两次，必然删掉两个状态，有可能是相对新的，有可能是最旧的
        // 但是永远删不到最新的
        for (int i = 0; i < 2; ++i)
        {
            // 从倒数第三个开始
            const Eigen::Vector3d position =
                cam_state_iter->second->twc_;
            const Eigen::Matrix3d rotation = cam_state_iter->second->Rwc_;

            // 计算相对于关键相机状态的平移与旋转
            double distance = (position - key_position).norm();
            double angle = Eigen::AngleAxisd(rotation * key_rotation.transpose()).angle();

            // 判断大小以及跟踪率，就是cam_state_iter这个状态与关键相机状态的相似度，
            // 且当前的点跟踪率很高
            // 删去这个帧，否则删掉最老的
            // todo 加入配置文件
            // if (angle < rotation_threshold &&
            //     distance < translation_threshold &&
            //     tracking_rate > tracking_rate_threshold)
            if (angle < 0.2618 &&
                distance < 4 &&
                tracking_rate > 0.3)
            {
                rm_cam_state_ids.push_back(cam_state_iter->first);
                ++cam_state_iter;
            }
            else
            {
                rm_cam_state_ids.push_back(first_cam_state_iter->first);
                ++first_cam_state_iter;
            }
        }
        // Sort the elements in the output vector.
        // 4. 排序
        sort(rm_cam_state_ids.begin(), rm_cam_state_ids.end());

        // Find the size of the Jacobian matrix.
        // 2. 找到删减帧涉及的观测雅可比大小
        jacobian_row_size = 0;
        int aa = 0, bb = 0, cc = 0, dd = 0, ee = 0;
        std::vector<Eigen::Vector3d> map_points;
        for (auto &item : map_server)
        {
            auto &feature = item.second;
            if (feature.is_initialized && viewer_ptr_)
                map_points.push_back(feature.position);
            // Check how many camera states to be removed are associated
            // with this feature.
            // 2.1 在待删去的帧中统计能观测到这个特征的帧
            std::vector<int> involved_cam_state_ids(0);
            for (const auto &cam_id : rm_cam_state_ids)
            {
                if (feature.observations.find(cam_id) !=
                    feature.observations.end())
                    involved_cam_state_ids.push_back(cam_id);
            }

            if (involved_cam_state_ids.size() == 0) {
                aa++;
                continue;
            }
                
            // 2.2 这个点只在一个里面有观测那就直接删
            // 只用一个观测更新不了状态
            if (involved_cam_state_ids.size() == 1)
            {
                feature.observations.erase(involved_cam_state_ids[0]);
                bb++;
                continue;
            }
            // 程序到这里的时候说明找到了一个特征，先不说他一共被几帧观测到
            // 到这里说明被两帧或两帧以上待删减的帧观测到
            // 2.3 如果没有做过三角化，做一下三角化，如果失败直接删
            if (!feature.is_initialized)
            {
                // Check if the feature can be initialize.
                if (!feature.checkMotion(state_manager_ptr_->cam_states_))
                {
                    // If the feature cannot be initialized, just remove
                    // the observations associated with the camera states
                    // to be removed.
                    for (const auto &cam_id : involved_cam_state_ids)
                        feature.observations.erase(cam_id);
                    cc++;
                    continue;
                }
                else
                {
                    if (!feature.initializePosition(state_manager_ptr_->cam_states_))
                    {
                        for (const auto &cam_id : involved_cam_state_ids)
                            feature.observations.erase(cam_id);
                        dd++;
                        continue;
                    } else if (viewer_ptr_) {
                        map_points.push_back(feature.position);
                    }
                }
            }
            // 2.4 最后的最后得出了行数
            // 意味着有involved_cam_state_ids.size() 数量的观测要被删去
            // 但是因为待删去的帧间有共同观测的关系，直接删会损失这部分信息
            // 所以临删前做最后一次更新
            ee += involved_cam_state_ids.size();
            jacobian_row_size += 2 * involved_cam_state_ids.size() - 3;
        }
        LOG(INFO) << aa << " " << bb << " " << cc << " " << dd << " " << ee << " " << map_points.size();
        if (viewer_ptr_)
            viewer_ptr_->DrawFeatures(map_points);
        // Compute the Jacobian and residual.
        // 3. 计算待删掉的这部分观测的雅可比与误差
        // 预设大小
        Eigen::MatrixXd H_x = Eigen::MatrixXd::Zero(jacobian_row_size,
                                        param_ptr_->STATE_DIM + 6 * state_manager_ptr_->cam_states_.size());
        Eigen::VectorXd r = Eigen::VectorXd::Zero(jacobian_row_size);
        int stack_cntr = 0;

        // 又做了一遍类似上面的遍历，只不过该三角化的已经三角化，该删的已经删了
        for (auto &item : map_server)
        {
            auto &feature = item.second;
            // Check how many camera states to be removed are associated
            // with this feature.
            // 这段就是判断一下这个点是否都在待删除帧中有观测
            std::vector<int> involved_cam_state_ids(0);
            for (const auto &cam_id : rm_cam_state_ids)
            {
                if (feature.observations.find(cam_id) !=
                    feature.observations.end())
                    involved_cam_state_ids.push_back(cam_id);
            }

            // 一个的情况已经被删掉了
            if (involved_cam_state_ids.size() == 0)
                continue;

            // 计算出待删去的这部分的雅可比
            // 这个点假如有多个观测，但本次只用待删除帧上的观测
            Eigen::MatrixXd H_xj;
            Eigen::VectorXd r_j;
            featureJacobian(feature.id_, involved_cam_state_ids, H_xj, r_j);

            if (gatingTest(H_xj, r_j, involved_cam_state_ids.size(), state_ptr->C_))
            {
                H_x.block(stack_cntr, 0, H_xj.rows(), H_xj.cols()) = H_xj;
                r.segment(stack_cntr, r_j.rows()) = r_j;
                stack_cntr += H_xj.rows();
            }

            // 删去观测
            for (const auto &cam_id : involved_cam_state_ids)
                feature.observations.erase(cam_id);
        }

        H_x.conservativeResize(stack_cntr, H_x.cols());
        r.conservativeResize(stack_cntr);

        // Perform measurement update.
        // 4. 用待删去的这些观测更新一下
        measurementUpdate(H_x, r, state_ptr);

        // 5. 直接删掉对应的行列，直接干掉
        // 为啥没有做类似于边缘化的操作？
        // 个人认为是上面做最后的更新了，所以信息已经更新到了各个地方
        for (const auto &cam_id : rm_cam_state_ids)
        {
            int cam_sequence = std::distance(
                state_manager_ptr_->cam_states_.begin(), state_manager_ptr_->cam_states_.find(cam_id));
            int cam_state_start = param_ptr_->STATE_DIM + 6 * cam_sequence;
            int cam_state_end = cam_state_start + 6;

            // Remove the corresponding rows and columns in the state
            // covariance matrix.
            if (cam_state_end < state_ptr->C_.rows())
            {
                state_ptr->C_.block(cam_state_start, 0,
                                                state_ptr->C_.rows() - cam_state_end,
                                                state_ptr->C_.cols()) =
                    state_ptr->C_.block(cam_state_end, 0,
                                                    state_ptr->C_.rows() - cam_state_end,
                                                    state_ptr->C_.cols());

                state_ptr->C_.block(0, cam_state_start,
                                                state_ptr->C_.rows(),
                                                state_ptr->C_.cols() - cam_state_end) =
                    state_ptr->C_.block(0, cam_state_end,
                                                    state_ptr->C_.rows(),
                                                    state_ptr->C_.cols() - cam_state_end);

                state_ptr->C_.conservativeResize(
                    state_ptr->C_.rows() - 6, state_ptr->C_.cols() - 6);
            }
            else
            {
                state_ptr->C_.conservativeResize(
                    state_ptr->C_.rows() - 6, state_ptr->C_.cols() - 6);
            }

            // Remove this camera state in the state vector.
            state_manager_ptr_->cam_states_.erase(cam_id);
        }
    }

    if (viewer_ptr_) {
        std::vector<std::pair<Eigen::Matrix3d, Eigen::Vector3d>> cameras;
        for (auto [id, cam_state] : state_manager_ptr_->cam_states_) {
            cameras.push_back(std::make_pair(cam_state->Rwc_, cam_state->twc_));
        }
        viewer_ptr_->DrawCameras(cameras);
    }
    return true;
}