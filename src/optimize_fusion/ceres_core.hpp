#pragma once
#include "ceres_core/gnss_6dofpose_factor.h"
#include "ceres_core/imu_factor.h"
#include "ceres_core/integration_base.h"
#include "ceres_core/marginalization_factor.h"
#include "ceres_core/pose_local_parameterization.h"
#include <iostream>

struct CeresFusion {
  static void quat_to_euler(const Eigen::Quaterniond &q, double &yaw,
                            double &pitch, double &roll) {
    const double &q0 = q.w();
    const double &q1 = q.x();
    const double &q2 = q.y();
    const double &q3 = q.z();
    roll = atan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 * q1 + q2 * q2));
    pitch = asin(2 * (q0 * q2 - q3 * q1));
    yaw = atan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3));
  }

public:
  // debug : 调试用
  double latest_imu_time_ = -1;
  struct NavState {
    NavState() = default;
    double timestamp;
    Eigen::Vector3d p = Eigen::Vector3d::Zero();
    Eigen::Vector3d v = Eigen::Vector3d::Zero();
    Eigen::Quaterniond q = Eigen::Quaterniond::Identity();
    Eigen::Vector3d ba = Eigen::Vector3d::Zero();
    Eigen::Vector3d bg = Eigen::Vector3d::Zero();
  };
  CeresFusion() {
    // 初始化优化过程中的变量的尺寸以及对应的参数
    pre_integrations_.resize(windonw_size_, nullptr);
    para_Pose.resize(windonw_size_, std::vector<double>(SIZE_POSE));
    para_SpeedBias.resize(windonw_size_, std::vector<double>(SIZE_SPEEDBIAS));
    // imu_datas_.resize(windonw_size_);
    gnss_data_.resize(windonw_size_);
    states_.resize(windonw_size_);
    last_bias_.setZero();

    imu_preint_config_.ACC_N = 0.08;
    imu_preint_config_.ACC_W = 0.000001;
    imu_preint_config_.GYR_N = 0.004;
    imu_preint_config_.GYR_W = 0.000001;
    imu_preint_config_.G = Eigen::Vector3d{0, 0, 9.81};
  }

  bool GetLastestState(NavState& state) {
    if (!IsInit()) {
      return false;
    }
    state = lastest_state_;
    return true;
  }

  // 添加imu数据
  void AddImuData(double timestamp,
                  const Eigen::Matrix<double, 6, 1> &imu_data) {
    if (!get_first_gps_) {
      last_imu_time_ = timestamp;
      last_imu_ = imu_data;
      return;
    }

    assert(cur_frame_index_ >= 0);

    latest_imu_time_ = timestamp;

    if (!pre_integrations_[cur_frame_index_]) {
      pre_integrations_[cur_frame_index_] = new IntegrationBase{
          last_imu_.head(3), last_imu_.tail(3), last_bias_.head(3),
          last_bias_.tail(3), imu_preint_config_};
    }

    // 将imu的数据保存进来
    double dt_imu = timestamp - last_imu_time_;
    if (last_imu_time_ < 0) {
      dt_imu = 0.005;
    }

    pre_integrations_[cur_frame_index_]->push_back(dt_imu, imu_data.head(3),
                                                   imu_data.tail(3));

    last_imu_time_ = timestamp;
    last_imu_ = imu_data;

    // 用当前的imu预积分结果
    if (init_) {
      auto preint = pre_integrations_[cur_frame_index_];
      auto last_state = states_[cur_frame_index_];
      auto imu_preint_config_temp = imu_preint_config_;
      imu_preint_config_temp.G *= -1;
      lastest_state_ = last_state;
      lastest_state_.q = last_state.q * preint->delta_q;
      lastest_state_.q.normalize();

      lastest_state_.v = last_state.v + last_state.q * preint->delta_v +
                         imu_preint_config_temp.G * preint->sum_dt;

      lastest_state_.p =
          last_state.p + last_state.q * preint->delta_p +
          last_state.v * preint->sum_dt +
          0.5 * imu_preint_config_temp.G * preint->sum_dt * preint->sum_dt;
      lastest_state_.ba = last_state.ba;
      lastest_state_.bg = last_state.bg;
      lastest_state_.timestamp = timestamp;
    }
  }

  bool IsInit() {
    return init_;
  }

  // 基于滑窗的gps（目前主要是存在gps和imu同时存在的情况）
  void AddGpsData(double debug_time, const Eigen::Vector3d &gps_enu) {
    static size_t cur_gps_cnt = 0;
    if (cur_gps_cnt++ % process_gps_every_n != 0) {
      return;
    }

    if (!get_first_gps_) {
      get_first_gps_ = true;
      cur_frame_index_ = 0;

      GnssData cur_gns_data;
      cur_gns_data.timestamp = debug_time;
      cur_gns_data.enu_xyh = gps_enu;
      gnss_data_[cur_frame_index_] = cur_gns_data;

      return;
    }

    cur_frame_index_++;

    // save gnss data to buf
    GnssData cur_gns_data;
    cur_gns_data.timestamp = debug_time;
    cur_gns_data.enu_xyh = gps_enu;
    gnss_data_[cur_frame_index_] = cur_gns_data;

    TryToInit();

    if (!init_) {
      std::cout << "the filter had not been initializated" << std::endl;
      return;
    }

    // 用上一时刻的imu数据来进行状态的预测当前state的优化起始结果
    if (pre_integrations_[cur_frame_index_ - 1]) {
      auto preint = pre_integrations_[cur_frame_index_ - 1];
      auto imu_preint_config_temp = imu_preint_config_;
      imu_preint_config_temp.G *= -1;
      states_[cur_frame_index_].q =
          states_[cur_frame_index_ - 1].q * preint->delta_q;
      states_[cur_frame_index_].q.normalize();

      states_[cur_frame_index_].v =
          states_[cur_frame_index_ - 1].v +
          states_[cur_frame_index_ - 1].q * preint->delta_v +
          imu_preint_config_temp.G * preint->sum_dt;

      states_[cur_frame_index_].p =
          states_[cur_frame_index_ - 1].p +
          states_[cur_frame_index_ - 1].q * preint->delta_p +
          states_[cur_frame_index_ - 1].v * preint->sum_dt +
          0.5 * imu_preint_config_temp.G * preint->sum_dt * preint->sum_dt;
      states_[cur_frame_index_].ba = states_[cur_frame_index_ - 1].ba;
      states_[cur_frame_index_].bg = states_[cur_frame_index_ - 1].bg;
    } else {
      std::cout << "warnning, can not get last imu preint" << std::endl;
    }

    Optimize();

    // margin
    if (cur_frame_index_ == windonw_size_ - 2) {
      MarginalizationInfo *marginalization_info = new MarginalizationInfo();
      vector2double();

      if (last_marginalization_info) {
        std::vector<int> drop_set;
        for (int i = 0;
             i < static_cast<int>(last_marginalization_parameter_blocks.size());
             i++) {
          if (last_marginalization_parameter_blocks[i] == para_Pose[0].data() ||
              last_marginalization_parameter_blocks[i] ==
                  para_SpeedBias[0].data()) {
            drop_set.push_back(i);
          }
        }
        MarginalizationFactor *marginalization_factor =
            new MarginalizationFactor(last_marginalization_info);
        ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(
            marginalization_factor, nullptr,
            last_marginalization_parameter_blocks, drop_set);
        marginalization_info->addResidualBlockInfo(residual_block_info);
      }

      // 边缘化掉对应的imu预积分的factor(0~1)对应的两帧的imu数据
      {
        IMUFactor *imu_factor = new IMUFactor(pre_integrations_[0]);
        ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(
            imu_factor, NULL,
            std::vector<double *>{para_Pose[0].data(), para_SpeedBias[0].data(),
                                  para_Pose[1].data(),
                                  para_SpeedBias[1].data()},
            std::vector<int>{0, 1});
        marginalization_info->addResidualBlockInfo(residual_block_info);
      }

      // 边缘化掉gps的factor
      {
        Eigen::Matrix3d gps_info = Eigen::Matrix3d::Identity() * 1 / 0.01;
        GnssPositionFactor *gnss_factor =
            new GnssPositionFactor(gnss_data_[0].enu_xyh, gps_info);
        ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(
            gnss_factor, NULL, std::vector<double *>{para_Pose[0].data()},
            std::vector<int>{0});
        marginalization_info->addResidualBlockInfo(residual_block_info);
      }

      marginalization_info->preMarginalize();
      marginalization_info->marginalize();

      // 进行滑动窗口内部参数的移动
      std::unordered_map<long, double *> addr_shift;
      for (int i = 1; i <= cur_frame_index_; i++) {
        addr_shift[reinterpret_cast<long>(para_Pose[i].data())] =
            para_Pose[i - 1].data();
        addr_shift[reinterpret_cast<long>(para_SpeedBias[i].data())] =
            para_SpeedBias[i - 1].data();
      }
      std::vector<double *> parameter_blocks =
          marginalization_info->getParameterBlocks(addr_shift);

      if (last_marginalization_info) {
        delete last_marginalization_info;
      }
      last_marginalization_info = marginalization_info;
      last_marginalization_parameter_blocks = parameter_blocks;

      // 交换滑动窗口内的数据
      for (int i = 0; i < cur_frame_index_; i++) {
        states_[i] = states_[i + 1];
        gnss_data_[i] = gnss_data_[i + 1];
        std::swap(pre_integrations_[i], pre_integrations_[i + 1]);
      }

      // 删除最后一个imu预积分的结果
      delete pre_integrations_[cur_frame_index_ - 1];
      pre_integrations_[cur_frame_index_ - 1] = nullptr;
      cur_frame_index_--;
    }

    // TODO : 只测试30s的数据
    // static size_t cur_cnt = 0;
    // if (cur_cnt++ > windonw_size_ + 300) {
    //   exit(0);
    // }
  }

private:
  // 内部使用的数据结构
  struct GnssData {
    GnssData() = default;
    double timestamp;
    Eigen::Vector3d enu_xyh = Eigen::Vector3d::Zero();
  };

  NavState lastest_state_;

  // try to init
  void TryToInit() {
    if (!init_) {
      // 条件1 ：当前的滑窗帧数需要足够多
      if (cur_frame_index_ == windonw_size_ - 2) {
        // 条件2：gps水平方向需要有足够多的运动
        double total_dis = 0;
        for (int i = 1; i <= cur_frame_index_; i++) {
          total_dis += (gnss_data_[i].enu_xyh - gnss_data_[i - 1].enu_xyh)
                           .head(2)
                           .norm();
        }

        std::cout << "total_dis : " << total_dis << std::endl;
        double init_pos_th = 0.5;
        if (total_dis > init_pos_th) {
          if (Optimize()) {
            init_ = true;
          }
        }

        // 如果当前的滑窗达到了最大值，却initialization失败，则移除第一帧的结果
        if (!init_) {
          // 交换滑动窗口内的数据（imu预积分数据以及state的状态量）
          for (int i = 0; i < cur_frame_index_; i++) {
            states_[i] = states_[i + 1];
            gnss_data_[i] = gnss_data_[i + 1];
            std::swap(pre_integrations_[i], pre_integrations_[i + 1]);
          }

          // 删除最后一个imu预积分的结果
          delete pre_integrations_[cur_frame_index_ - 1];
          pre_integrations_[cur_frame_index_ - 1] = nullptr;
          cur_frame_index_--;
        }
      }
    }
  }

  void vector2double() {
    for (int i = 0; i <= cur_frame_index_; i++) {
      auto Ps = states_[i].p;
      auto q = states_[i].q;
      auto Vs = states_[i].v;
      auto Bas = states_[i].ba;
      auto Bgs = states_[i].bg;

      para_Pose[i][0] = Ps.x();
      para_Pose[i][1] = Ps.y();
      para_Pose[i][2] = Ps.z();
      para_Pose[i][3] = q.x();
      para_Pose[i][4] = q.y();
      para_Pose[i][5] = q.z();
      para_Pose[i][6] = q.w();

      para_SpeedBias[i][0] = Vs.x();
      para_SpeedBias[i][1] = Vs.y();
      para_SpeedBias[i][2] = Vs.z();

      para_SpeedBias[i][3] = Bas.x();
      para_SpeedBias[i][4] = Bas.y();
      para_SpeedBias[i][5] = Bas.z();

      para_SpeedBias[i][6] = Bgs.x();
      para_SpeedBias[i][7] = Bgs.y();
      para_SpeedBias[i][8] = Bgs.z();
    }
  }

  void double2vector() {
    for (int i = 0; i <= cur_frame_index_; i++) {
      states_[i].q = Eigen::Quaterniond(para_Pose[i][6], para_Pose[i][3],
                                        para_Pose[i][4], para_Pose[i][5]);
      states_[i].p =
          Vector3d(para_Pose[i][0], para_Pose[i][1], para_Pose[i][2]);
      states_[i].v = Vector3d(para_SpeedBias[i][0], para_SpeedBias[i][1],
                              para_SpeedBias[i][2]);
      states_[i].ba = Vector3d(para_SpeedBias[i][3], para_SpeedBias[i][4],
                               para_SpeedBias[i][5]);
      states_[i].bg = Vector3d(para_SpeedBias[i][6], para_SpeedBias[i][7],
                               para_SpeedBias[i][8]);
    }
  }

  bool Optimize() {
    vector2double();

    ceres::Problem problem;
    ceres::LossFunction *loss_function;
    loss_function = new ceres::CauchyLoss(1.0);

    // add param
    for (int i = 0; i <= cur_frame_index_; i++) {
      ceres::LocalParameterization *local_parameterization =
          new PoseLocalParameterization();
      problem.AddParameterBlock(para_Pose[i].data(), SIZE_POSE,
                                local_parameterization);
      problem.AddParameterBlock(para_SpeedBias[i].data(), SIZE_SPEEDBIAS);
    }

    // add imu factor
    for (int i = 0; i < cur_frame_index_; i++) {
      assert(pre_integrations_[i]);
      IMUFactor *imu_factor = new IMUFactor(pre_integrations_[i]);
      problem.AddResidualBlock(
          imu_factor, nullptr, para_Pose[i].data(), para_SpeedBias[i].data(),
          para_Pose[i + 1].data(), para_SpeedBias[i + 1].data());
    }

    // add gps factor
    for (int i = 0; i <= cur_frame_index_; i++) {
      Eigen::Matrix3d gps_info = Eigen::Matrix3d::Identity() * 1 / 0.1;

      // fix first gps position data
      if (i == 0) {
        // gps_info *= 1e10;
      }

      GnssPositionFactor *gnss_factor =
          new GnssPositionFactor(gnss_data_[i].enu_xyh, gps_info);
      problem.AddResidualBlock(gnss_factor, nullptr, para_Pose[i].data());
    }

    if (last_marginalization_info) {
      MarginalizationFactor *marginalization_factor =
          new MarginalizationFactor(last_marginalization_info);
      problem.AddResidualBlock(marginalization_factor, NULL,
                               last_marginalization_parameter_blocks);
    }

    // solve
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    // options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.trust_region_strategy_type = ceres::DOGLEG;
    options.max_num_iterations = 5;
    options.use_nonmonotonic_steps = true;
    // options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    double2vector();

    last_bias_.head(3) = states_[cur_frame_index_].ba;
    last_bias_.tail(3) = states_[cur_frame_index_].bg;

    return summary.IsSolutionUsable();
  }

  // ---------------------------------------------------------------------
  bool init_ = false;

  size_t cur_frame_index_ = -1;
  const size_t windonw_size_ = 10;
  size_t process_gps_every_n = 1;
  std::vector<IntegrationBase *> pre_integrations_;

  bool get_first_imu_ = false;
  bool get_first_gps_ = false;
  double last_imu_time_ = -1;
  Eigen::Matrix<double, 6, 1> last_imu_;  // acc, gyro
  Eigen::Matrix<double, 6, 1> last_bias_; // acc, gyro
  IntegrationBase::Config imu_preint_config_;

  std::vector<NavState> states_;
  std::vector<GnssData> gnss_data_;
  std::vector<std::vector<double>> para_Pose;
  std::vector<std::vector<double>> para_SpeedBias;

  MarginalizationInfo *last_marginalization_info = nullptr;
  std::vector<double *> last_marginalization_parameter_blocks;
  Eigen::Vector3d first_gps_pose_;
};