#include "WheelPredictor.h"

// todo 数个一起来还是一个一个，数个一起来比较好
void WheelPredictor::Run() {
    while(1) {
        WheelData cur_data;
        if (!data_manager_ptr_->GetLastWheelData(cur_data, last_data_.time_)) {
            usleep(100);
            continue;
        }
        
<<<<<<< HEAD
        // 第一个数据，推荐只用6维
=======
        // 第一个数据
>>>>>>> dc08928f99acf1c61e5b91f2bb97164b37a63618
        if (last_data_.time_ <= 0.0) {
            std::shared_ptr<State> cur_state_ptr = std::make_shared<State>();
            cur_state_ptr->time_ = cur_data.time_;
            cur_state_ptr->C_ = Eigen::MatrixXd::Identity(param_ptr_->STATE_DIM, param_ptr_->STATE_DIM);
            cur_state_ptr->C_.block<3, 3>(param_ptr_->POSI_INDEX, param_ptr_->POSI_INDEX) = Eigen::Matrix3d::Identity() * 0.25;
<<<<<<< HEAD
            cur_state_ptr->C_.block<3, 3>(param_ptr_->ORI_INDEX_STATE_, param_ptr_->ORI_INDEX_STATE_) = Eigen::Matrix3d::Identity() * 0.25;
=======
            cur_state_ptr->C_.block<3, 3>(param_ptr_->VEL_INDEX_STATE_, param_ptr_->VEL_INDEX_STATE_) = Eigen::Matrix3d::Identity() * 0.25;
            cur_state_ptr->C_.block<3, 3>(param_ptr_->ORI_INDEX_STATE_, param_ptr_->ORI_INDEX_STATE_) = Eigen::Matrix3d::Identity() * 0.25;
            cur_state_ptr->C_.block<3, 3>(param_ptr_->GYRO_BIAS_INDEX_STATE_, param_ptr_->GYRO_BIAS_INDEX_STATE_) = Eigen::Matrix3d::Identity() * 0.01;
            cur_state_ptr->C_.block<3, 3>(param_ptr_->ACC_BIAS_INDEX_STATE_, param_ptr_->ACC_BIAS_INDEX_STATE_) = Eigen::Matrix3d::Identity() * 0.01;
>>>>>>> dc08928f99acf1c61e5b91f2bb97164b37a63618
            state_manager_ptr_->PushState(cur_state_ptr);
            last_data_ = cur_data;
            usleep(100);
            continue;
        }
        double delta_t = cur_data.time_ - last_data_.time_;
        if (delta_t <= 0.0) {
            usleep(100);
            continue;
        }

        // note: 轮速计的数据是左轮速和右轮速，需要转换成线速度和角速度1英寸d                                                                                                                                                                                                                                                                                                                                                                                                                                                               
        // note: 根据原文，后轮就是汽车中心，车轮使用18英寸的轮胎，直径为45.72cm，半径r = 22.86cm
        // note: encoder是增量式的，所以需要用现在的值减去上一个值，得到增量
<<<<<<< HEAD
        // todo 针对某个数据集的处理放到读数据的代码中
        // 计算增量
        double delta_time = cur_data.time_ - last_data_.time_;
        double l_encoder_delta = cur_data.lv_ - last_data_.lv_;
        double r_encoder_delta = cur_data.rv_ - last_data_.rv_;
=======
        // 计算增量
        double delta_time = cur_data.time_ - last_data_.time_;
        double l_encoder_delta = cur_data.lv_ - last_data_.lv_;
        double r_encoder_delta = cur_data.lv_ - last_data_.lv_;
>>>>>>> dc08928f99acf1c61e5b91f2bb97164b37a63618

        const double wheel_base = 1.52439; 
        const double l_d = 0.623479;
        const double r_d = 0.622806;
<<<<<<< HEAD

        // todo 这里已经算好了速度，可以直接乘时间表示距离
=======
>>>>>>> dc08928f99acf1c61e5b91f2bb97164b37a63618
        // 计算左右轮走过的距离delta dist：计算转过的角度，pi * D * delta / 4096
        double l_dd = M_PI * l_d * l_encoder_delta / 4096;
        double r_dd = M_PI * r_d * r_encoder_delta / 4096;
        double l_v = l_dd / delta_time;
        double r_v = r_dd / delta_time;
        // 根据速度公式计算速度
        double delta_dist = (l_dd + r_dd) / 2;
        double delta_angle = (r_dd - l_dd) / wheel_base;
        // 计算线速度和角速度
        double v = (l_v + r_v) / 2;
<<<<<<< HEAD
        double w = (r_v - l_v) / wheel_base;

        // note: v是线速度，在车中，只观测x轴方向的速度
        // 将速度包装成向量的形式
        Eigen::Vector3d velo_vec(v, 0, 0);   // x, y, z 三个轴的线速度
        Eigen::Vector3d omega_vec(0, 0, w); // x, y, z 三个轴的角速度
=======
        double w = (rv - lv) / wheel_base;

        // note: v是线速度，在车中，只观测x轴方向的速度
        // 将速度包装成向量的形式
        Eigen::Vector3d velo_vec = [v, 0, 0];   // x, y, z 三个轴的线速度
        Eigen::Vector3d omega_vec = [0, 0, w]; // x, y, z 三个轴的角速度
>>>>>>> dc08928f99acf1c61e5b91f2bb97164b37a63618

        // 需要根据外参，将速度转换为body坐标系的速度，目前假设body坐标系在后轴中心
        // Dead Reckoning
        std::shared_ptr<State> last_state_ptr;
        state_manager_ptr_->GetNearestState(last_state_ptr);

        
        // 计算当前状态
        std::shared_ptr<State> cur_state_ptr = std::make_shared<State>();

        
        cur_state_ptr->time_ = cur_data.time_;

        // 世界坐标系下的速度
        cur_state_ptr->Vw_ = last_state_ptr->Rwb_ * velo_vec;
        // 世界坐标系的位置
        cur_state_ptr->twb_ = last_state_ptr->twb_ + cur_state_ptr->Vw_ * delta_time;
        // 世界坐标系下的角度
<<<<<<< HEAD
        cur_state_ptr->Rwb_ = last_state_ptr->Rwb_ * Converter::ExpSO3(omega_vec * delta_t);

        //---------------------------------------------------------------------------------------------------
        // 计算协方差矩阵
        // todo 这里推荐只用位移跟旋转作为状态，一共就6维
=======
        cur_state_ptr->Rwb_ = last_state_ptr->Rwb_ * Sophus::SO3d::exp(omega_vec * dt);

        //---------------------------------------------------------------------------------------------------
        // 计算协方差矩阵
>>>>>>> dc08928f99acf1c61e5b91f2bb97164b37a63618
        // note: 这里既要考虑速度，又要考虑角度（假设角度的观测是准的）
        // note: 同时还要考虑速度和角度的误差，构建协方差矩阵
        // note: F是状态转移矩阵，G是噪声转移矩阵
        // 定义： 理想数值（优质数值） = 估计数值 + 误差
        // note: 里程计的观测是世界坐标系中的速度，以及世界坐标系中的角度，所以状态转移也与这两个量有关
        Eigen::MatrixXd F = Eigen::MatrixXd::Zero(param_ptr_->STATE_DIM, param_ptr_->STATE_DIM);
        // note: 速度转换到直接坐标系下，可以看做是对世界坐标系下的速度的直接观测，导数为单位矩阵
        F.block<3, 3>(param_ptr_->VEL_INDEX_STATE_, param_ptr_->VEL_INDEX_STATE_) = Eigen::Matrix3d::Identity();
        // note: 角度转换到直接坐标系下，可以看做是对世界坐标系下的角度的直接观测，导数为单位矩阵
        F.block<3, 3>(param_ptr_->ORI_INDEX_STATE_, param_ptr_->ORI_INDEX_STATE_) = Eigen::Matrix3d::Identity();
        Eigen::MatrixXd Phi = Eigen::MatrixXd::Identity(param_ptr_->STATE_DIM, param_ptr_->STATE_DIM) + F * delta_t;

        // note: 噪声直接累加
        Eigen::MatrixXd G = Eigen::MatrixXd::Zero(param_ptr_->STATE_DIM, 12);
        G.block<3, 3>(param_ptr_->ORI_INDEX_STATE_, 0) = Eigen::Matrix3d::Identity() * delta_t;
        G.block<3, 3>(param_ptr_->VEL_INDEX_STATE_, 6) = Eigen::Matrix3d::Identity() * delta_t;
        cur_state_ptr->C_ = Phi * last_state_ptr->C_ * Phi.transpose() + G * param_ptr_->imu_dispersed_noise_cov_ * G.transpose();

        Eigen::MatrixXd state_cov_fixed = 
            (cur_state_ptr->C_ + cur_state_ptr->C_.transpose()) / 2.0;
        cur_state_ptr->C_ = state_cov_fixed;

        state_manager_ptr_->PushState(cur_state_ptr);
        last_data_ = cur_data;

        if (viewer_ptr_)
            viewer_ptr_->DrawWheelPose(cur_state_ptr->Rwb_, cur_state_ptr->twb_);
        usleep(100);
    }
    
}