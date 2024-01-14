#include "WheelPredictor.h"

// todo 数个一起来还是一个一个，数个一起来比较好
void WheelPredictor::Run() {
    while(1) {
        WheelData cur_data;
        if (!data_manager_ptr_->GetLastWheelData(cur_data, last_data_.time_)) {
            usleep(100);
            continue;
        }
        
        // 第一个数据，推荐只用6维
        if (last_data_.time_ <= 0.0) {
            std::shared_ptr<State> cur_state_ptr = std::make_shared<State>();
            cur_state_ptr->time_ = cur_data.time_;
            cur_state_ptr->C_ = Eigen::MatrixXd::Identity(param_ptr_->STATE_DIM, param_ptr_->STATE_DIM);
            cur_state_ptr->C_.block<3, 3>(param_ptr_->POSI_INDEX, param_ptr_->POSI_INDEX) = Eigen::Matrix3d::Identity() * 0.25;
            cur_state_ptr->C_.block<3, 3>(param_ptr_->ORI_INDEX_STATE_, param_ptr_->ORI_INDEX_STATE_) = Eigen::Matrix3d::Identity() * 0.25;
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
        // todo 针对某个数据集的处理放到读数据的代码中

        // 计算线速度和角速度
        double v = (cur_data.lv_ + cur_data.rv_) * 0.5;
        double w = (cur_data.rv_ - cur_data.lv_) / param_ptr_->wheel_b_;
        // note: v是线速度，在车中，只观测x轴方向的速度
        // 将速度包装成向量的形式
        Eigen::Vector3d velo_vec(v, 0, 0);   // x, y, z 三个轴的线速度
        Eigen::Vector3d omega_vec(0, 0, w); // x, y, z 三个轴的角速度

        // std::cout << "delta_t :" << delta_t << "\n" << "v: " << v << " w: " << w 
        //           << " curr_data.lv : " << cur_data.lv_ << " curr_data.rv : " << cur_data.rv_ << std::endl;
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
        cur_state_ptr->twb_ = last_state_ptr->twb_ + cur_state_ptr->Vw_ * delta_t;

        // 世界坐标系的位置
        // cur_state_ptr->twb_ = last_state_ptr->twb_ + last_state_ptr->Rwb_ * velo_vec * delta_t;
        // 世界坐标系下的角度
        cur_state_ptr->Rwb_ = last_state_ptr->Rwb_ * Converter::ExpSO3(omega_vec * delta_t);


        //---------------------------------------------------------------------------------------------------
        // 计算协方差矩阵
        // todo 这里推荐只用位移跟旋转作为状态，一共就6维
        // note: 这里既要考虑速度，又要考虑角度（假设角度的观测是准的）
        // note: 同时还要考虑速度和角度的误差，构建协方差矩阵
        // note: F是状态转移矩阵，G是噪声转移矩阵
        // 定义： 理想数值（优质数值） = 估计数值 + 误差
        // note: 里程计的观测是世界坐标系中的速度，以及世界坐标系中的角度，所以状态转移也与这两个量有关
        // Eigen::MatrixXd F = Eigen::MatrixXd::Zero(param_ptr_->STATE_DIM, param_ptr_->STATE_DIM);
        // // note: 速度转换到直接坐标系下，可以看做是对世界坐标系下的速度的直接观测，导数为单位矩阵
        // F.block<3, 3>(param_ptr_->VEL_INDEX_STATE_, param_ptr_->VEL_INDEX_STATE_) = Eigen::Matrix3d::Identity();
        // // note: 角度转换到直接坐标系下，可以看做是对世界坐标系下的角度的直接观测，导数为单位矩阵
        // F.block<3, 3>(param_ptr_->ORI_INDEX_STATE_, param_ptr_->ORI_INDEX_STATE_) = Eigen::Matrix3d::Identity();
        // Eigen::MatrixXd Phi = Eigen::MatrixXd::Identity(param_ptr_->STATE_DIM, param_ptr_->STATE_DIM) + F * delta_t;

        // 这里先试用位移和旋转作为状态，不考虑使用速度（v, t, R, ba, bg，一共15维的状态）
        Eigen::MatrixXd F = Eigen::MatrixXd::Zero(param_ptr_->STATE_DIM, param_ptr_->STATE_DIM);
        // 雅可比矩阵：
        // dX / dt = - R_g_o * skew(v * dt) | I
        // dX / dR = exp(w * dt)            | 0
        F.block<3, 3>(param_ptr_->POSI_INDEX, param_ptr_->ORI_INDEX_STATE_) = -cur_state_ptr->Rwb_ * Converter::Skew(velo_vec * delta_t);
        F.block<3, 3>(param_ptr_->POSI_INDEX, param_ptr_->POSI_INDEX) = Eigen::Matrix3d::Identity();
        F.block<3, 3>(param_ptr_->ORI_INDEX_STATE_, param_ptr_->ORI_INDEX_STATE_) = Converter::ExpSO3(omega_vec * delta_t);

        // note: 噪声直接累加
        // 增加位移和旋转的噪声，其中，旋转的噪声是Identity，位移的噪声是旋转*Identity
        Eigen::MatrixXd G = Eigen::MatrixXd::Zero(param_ptr_->STATE_DIM, 6);
        G.block<3, 3>(param_ptr_->POSI_INDEX, 0) = Eigen::Matrix3d::Identity() * delta_t;
        G.block<3, 3>(param_ptr_->ORI_INDEX_STATE_, 3) = last_state_ptr->Rwb_ * delta_t;

        Eigen::Matrix<double, 6, 6> odom_dispersed_noise_cov = Eigen::Matrix<double, 6, 6>::Zero();
        odom_dispersed_noise_cov(0, 0) = 0.0;  // 这几维不作观测，所以不需要噪声
        odom_dispersed_noise_cov(1, 1) = 0.0;
        odom_dispersed_noise_cov(2, 2) = w * w * param_ptr_->wheel_noise_factor_ * param_ptr_->wheel_noise_factor_;
        odom_dispersed_noise_cov(3, 3) = v * v * param_ptr_->wheel_noise_factor_ * param_ptr_->wheel_noise_factor_;
        odom_dispersed_noise_cov(4, 4) = v * v * param_ptr_->wheel_noise_factor_ * param_ptr_->wheel_noise_factor_;
        odom_dispersed_noise_cov(5, 5) = 0.0;

        Eigen::MatrixXd Phi = Eigen::MatrixXd::Identity(param_ptr_->STATE_DIM, param_ptr_->STATE_DIM) + F * delta_t;
        cur_state_ptr->C_ = Phi * last_state_ptr->C_ * Phi.transpose() + G * odom_dispersed_noise_cov * G.transpose();

        Eigen::MatrixXd state_cov_fixed = 
            (cur_state_ptr->C_ + cur_state_ptr->C_.transpose()) / 2.0;
        cur_state_ptr->C_ = state_cov_fixed;

        state_manager_ptr_->PushState(cur_state_ptr);
        last_data_ = cur_data;

        if (viewer_ptr_)
            viewer_ptr_->DrawWheelOdom(cur_state_ptr->Rwb_, cur_state_ptr->twb_);
        usleep(100);
    }
    
}