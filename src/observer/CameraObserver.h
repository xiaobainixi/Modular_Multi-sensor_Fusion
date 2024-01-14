#pragma once
#include <boost/math/distributions/chi_squared.hpp>

#include "Observer.h"
#include "visual/Feature.hpp"

class CameraObserver : public Observer
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    CameraObserver(const std::shared_ptr<Parameter> &param_ptr, const std::shared_ptr<DataManager> &data_manager_ptr,
                const std::shared_ptr<StateManager> &state_manager_ptr, std::shared_ptr<Viewer> viewer_ptr = nullptr)
    {
        param_ptr_ = param_ptr;
        data_manager_ptr_ = data_manager_ptr;
        state_manager_ptr_ = state_manager_ptr;
        viewer_ptr_ = viewer_ptr;
        // 卡方检验表
        // Initialize the chi squared test table with confidence
        // level 0.95.
        for (int i = 1; i < 100; ++i)
        {
            boost::math::chi_squared chi_squared_dist(i);
            chi_squared_test_table_[i] =
                boost::math::quantile(chi_squared_dist, 0.05);
        }
    }

    bool ComputeHZR(
        const FeatureData & feature_data, const std::shared_ptr<State> & state_ptr,
        Eigen::MatrixXd & H, Eigen::MatrixXd & Z, Eigen::MatrixXd &R);

private:

    /**
     * @brief 更新
     * @param  H 雅可比
     * @param  r 误差
     */
    void measurementUpdate(
        const Eigen::MatrixXd &H, const Eigen::VectorXd &r, const std::shared_ptr<State> & state_ptr)
    {

        if (H.rows() == 0 || r.rows() == 0)
            return;

        // Decompose the final Jacobian matrix to reduce computational
        // complexity as in Equation (28), (29).
        Eigen::MatrixXd H_thin;
        Eigen::VectorXd r_thin;

        if (H.rows() > H.cols())
        {
            // Convert H to a sparse matrix.
            Eigen::SparseMatrix<double> H_sparse = H.sparseView();

            // Perform QR decompostion on H_sparse.
            // 利用H矩阵稀疏性，QR分解
            // 这段结合零空间投影一起理解，主要作用就是降低计算量
            Eigen::SPQR<Eigen::SparseMatrix<double>> spqr_helper;
            spqr_helper.setSPQROrdering(SPQR_ORDERING_NATURAL);
            spqr_helper.compute(H_sparse);

            Eigen::MatrixXd H_temp;
            Eigen::VectorXd r_temp;
            (spqr_helper.matrixQ().transpose() * H).evalTo(H_temp);
            (spqr_helper.matrixQ().transpose() * r).evalTo(r_temp);

            H_thin = H_temp.topRows(param_ptr_->STATE_DIM + state_manager_ptr_->cam_states_.size() * 6);
            r_thin = r_temp.head(param_ptr_->STATE_DIM + state_manager_ptr_->cam_states_.size() * 6);

            // HouseholderQR<Eigen::MatrixXd> qr_helper(H);
            // Eigen::MatrixXd Q = qr_helper.householderQ();
            // Eigen::MatrixXd Q1 = Q.leftCols(param_ptr_->STATE_DIM+cam_states_.size()*6);

            // H_thin = Q1.transpose() * H;
            // r_thin = Q1.transpose() * r;
        }
        else
        {
            H_thin = H;
            r_thin = r;
        }

        // 2. 标准的卡尔曼计算过程
        // Compute the Kalman gain.
        const Eigen::MatrixXd &P = state_ptr->C_;
        Eigen::MatrixXd S = H_thin * P * H_thin.transpose() +
                        param_ptr_->visual_observation_noise_ * Eigen::MatrixXd::Identity(
                                                        H_thin.rows(), H_thin.rows());
        // Eigen::MatrixXd K_transpose = S.fullPivHouseholderQr().solve(H_thin*P);
        Eigen::MatrixXd K_transpose = S.ldlt().solve(H_thin * P);
        Eigen::MatrixXd K = K_transpose.transpose();

        // Compute the error of the state.
        Eigen::VectorXd delta_x = K * r_thin;

        // Update the IMU state.
        // const Eigen::VectorXd &delta_x_imu = delta_x.segment(0, param_ptr_->STATE_DIM);

        // if ( // delta_x_imu.segment<3>(0).norm() > 0.15 ||
        //         // delta_x_imu.segment<3>(3).norm() > 0.15 ||
        //     delta_x_imu.segment<3>(6).norm() > 0.5 ||
        //     // delta_x_imu.segment<3>(9).norm() > 0.5 ||
        //     delta_x_imu.segment<3>(12).norm() > 1.0)
        // {
        //     printf("delta velocity: %f\n", delta_x_imu.segment<3>(6).norm());
        //     printf("delta position: %f\n", delta_x_imu.segment<3>(12).norm());
        //     // return;
        // }

        // Update state covariance.
        // 4. 更新协方差
        Eigen::MatrixXd I_KH = Eigen::MatrixXd::Identity(K.rows(), H_thin.cols()) - K * H_thin;
        // state_ptr->C_ = I_KH*state_ptr->C_*I_KH.transpose() +
        //   K*K.transpose()*Feature::observation_noise;
        Eigen::MatrixXd C_tmp = I_KH * state_ptr->C_;

        // Fix the covariance to be symmetric
        C_tmp = (C_tmp + C_tmp.transpose()) / 2.0;
        state_ptr->Update(param_ptr_, delta_x, C_tmp, state_manager_ptr_->cam_states_);
    }

    /**
     * @brief 计算一个路标点的雅可比
     * @param  feature_id 路标点id
     * @param  cam_state_ids 这个点对应的所有的相机状态id
     * @param  H_x 雅可比
     * @param  r 误差
     */
    void featureJacobian(
        const FeatureIDType &feature_id,
        const std::vector<int> &cam_state_ids,
        Eigen::MatrixXd &H_x, Eigen::VectorXd &r)
    {
        // 取出特征
        const auto &feature = map_server[feature_id];

        // Check how many camera states in the provided camera
        // id camera has actually seen this feature.
        // 1. 统计有效观测的相机状态，因为对应的个别状态有可能被滑走了
        std::vector<int> valid_cam_state_ids(0);
        for (const auto &cam_id : cam_state_ids)
        {
            if (feature.observations.find(cam_id) ==
                feature.observations.end())
                continue;

            valid_cam_state_ids.push_back(cam_id);
        }

        int jacobian_row_size = 0;
        // 行数等于4*观测数量，一个观测在双目上都有，所以是2*2
        // 此时还没有0空间投影
        jacobian_row_size = 2 * valid_cam_state_ids.size();

        // 误差相对于状态量的雅可比，没有约束列数，因为列数一直是最新的
        Eigen::MatrixXd H_xj = Eigen::MatrixXd::Zero(jacobian_row_size,
                                        param_ptr_->STATE_DIM + state_manager_ptr_->cam_states_.size() * 6);
        // 误差相对于三维点的雅可比
        Eigen::MatrixXd H_fj = Eigen::MatrixXd::Zero(jacobian_row_size, 3);
        // 误差
        Eigen::VectorXd r_j = Eigen::VectorXd::Zero(jacobian_row_size);
        int stack_cntr = 0;

        // 2. 计算每一个观测（同一帧左右目这里被叫成一个观测）的雅可比与误差
        for (const auto &cam_id : valid_cam_state_ids)
        {

            Eigen::Matrix<double, 2, 6> H_xi = Eigen::Matrix<double, 2, 6>::Zero();
            Eigen::Matrix<double, 2, 3> H_fi = Eigen::Matrix<double, 2, 3>::Zero();
            Eigen::Vector2d r_i = Eigen::Vector2d::Zero();
            // 2.1 计算一个左右目观测的雅可比
            measurementJacobian(cam_id, feature.id_, H_xi, H_fi, r_i);

            // 计算这个cam_id在整个矩阵的列数，因为要在大矩阵里面放
            auto cam_state_iter = state_manager_ptr_->cam_states_.find(cam_id);
            int cam_state_cntr = std::distance(
                state_manager_ptr_->cam_states_.begin(), cam_state_iter);

            // Stack the Jacobians.
            H_xj.block<2, 6>(stack_cntr, param_ptr_->STATE_DIM + 6 * cam_state_cntr) = H_xi;
            H_fj.block<2, 3>(stack_cntr, 0) = H_fi;
            r_j.segment<2>(stack_cntr) = r_i;
            stack_cntr += 2;
        }

        // Project the residual and Jacobians onto the nullspace
        // of H_fj.
        // 零空间投影
        Eigen::JacobiSVD<Eigen::MatrixXd> svd_helper(H_fj, Eigen::ComputeFullU | Eigen::ComputeThinV);
        Eigen::MatrixXd A = svd_helper.matrixU().rightCols(
            jacobian_row_size - 3);

        // 上面的效果跟QR分解一样，下面的代码可以测试打印对比
        // Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(H_fj);
        // Eigen::MatrixXd Q = qr.matrixQ();
        // std::cout << "spqr_helper.matrixQ(): " << std::endl << Q << std::endl << std::endl;
        // std::cout << "A: " << std::endl << A << std::endl;

        // 0空间投影
        H_x = A.transpose() * H_xj;
        r = A.transpose() * r_j;

        return;
    }

    /**
     * @brief 计算一个路标点的雅可比
     * @param  cam_state_id 有效的相机状态id
     * @param  feature_id 路标点id
     * @param  H_x 误差相对于位姿的雅可比
     * @param  H_f 误差相对于三维点的雅可比
     * @param  r 误差
     */
    void measurementJacobian(
        const int &cam_state_id,
        const FeatureIDType &feature_id,
        Eigen::Matrix<double, 2, 6> &H_x, Eigen::Matrix<double, 2, 3> &H_f, Eigen::Vector2d &r)
    {

        // Prepare all the required data.
        // 1. 取出相机状态与特征
        const std::shared_ptr<CamState> &cam_state = state_manager_ptr_->cam_states_[cam_state_id];
        const Feature &feature = map_server[feature_id];

        // 2. 取出左目位姿，根据外参计算右目位姿
        // Cam0 pose.

        // 3. 取出三维点坐标与归一化的坐标点，因为前端发来的是归一化坐标的
        // 3d feature position in the world frame.
        // And its observation with the stereo cameras.
        const Eigen::Vector3d &p_w = feature.position;
        const Eigen::Vector2d &z = feature.observations.find(cam_state_id)->second;

        // 4. 转到左右目相机坐标系下
        // Convert the feature position from the world frame to
        // the cam0 and cam1 frame.
        Eigen::Vector3d p_c0 = cam_state->Rwc_ * (p_w - cam_state->twc_);
        // p_c1 = R_c0_c1 * cam_state->Rwc_ * (p_w - cam_state->twc_ + R_w_c1.transpose() * t_cam0_cam1)
        //      = R_c0_c1 * (p_c0 + cam_state->Rwc_ * R_w_c1.transpose() * t_cam0_cam1)
        //      = R_c0_c1 * (p_c0 + R_c0_c1 * t_cam0_cam1)

        // Compute the Jacobians.
        // 5. 计算雅可比
        // 左相机归一化坐标点相对于左相机坐标系下的点的雅可比
        // (x, y) = (X / Z, Y / Z)
        Eigen::Matrix<double, 2, 3> dz_dpc0 = Eigen::Matrix<double, 2, 3>::Zero();
        dz_dpc0(0, 0) = 1 / p_c0(2);
        dz_dpc0(1, 1) = 1 / p_c0(2);
        dz_dpc0(0, 2) = -p_c0(0) / (p_c0(2) * p_c0(2));
        dz_dpc0(1, 2) = -p_c0(1) / (p_c0(2) * p_c0(2));

        // 左相机坐标系下的三维点相对于左相机位姿的雅可比 先r后t
        Eigen::Matrix<double, 3, 6> dpc0_dxc = Eigen::Matrix<double, 3, 6>::Zero();
        dpc0_dxc.leftCols(3) = Converter::SkewSymmetric(p_c0);
        dpc0_dxc.rightCols(3) = -cam_state->Rwc_;

        // Vector3d p_c0 = cam_state->Rwc_ * (p_w - cam_state->twc_);
        // p_c0 对 p_w
        Eigen::Matrix3d dpc0_dpg = cam_state->Rwc_;

        // 两个雅可比
        H_x = dz_dpc0 * dpc0_dxc;
        H_f = dz_dpc0 * dpc0_dpg;

        // Modifty the measurement Jacobian to ensure
        // observability constrain.
        // 6. OC
        Eigen::Matrix<double, 2, 6> A = H_x;
        Eigen::Matrix<double, 6, 1> u = Eigen::Matrix<double, 6, 1>::Zero();
        u.block<3, 1>(0, 0) = 
            cam_state->Rwc_null_ * param_ptr_->gw_;
        u.block<3, 1>(3, 0) =
            Converter::SkewSymmetric(p_w - cam_state->twc_null_) * param_ptr_->gw_;
        H_x = A - A * u * (u.transpose() * u).inverse() * u.transpose();
        H_f = -H_x.block<2, 3>(0, 3);

        // Compute the residual.
        // 7. 计算归一化平面坐标误差
        r = z - Eigen::Vector2d(p_c0(0) / p_c0(2), p_c0(1) / p_c0(2));

        return;
    }

    bool gatingTest(
        const Eigen::MatrixXd &H, const Eigen::VectorXd &r, const int &dof, const Eigen::MatrixXd & C)
    {
        // 输入的dof的值是所有相机观测，且没有去掉滑窗的
        // 而且按照维度这个卡方的维度也不对
        // 
        Eigen::MatrixXd P1 = H * C * H.transpose();
        Eigen::MatrixXd P2 = param_ptr_->visual_observation_noise_ *
                        Eigen::MatrixXd::Identity(H.rows(), H.rows());
        double gamma = r.transpose() * (P1 + P2).ldlt().solve(r);

        // cout << dof << " " << gamma << " " <<
        //   chi_squared_test_table[dof] << " ";

        if (gamma < chi_squared_test_table_[dof])
        {
            // cout << "passed" << endl;
            return true;
        }
        else
        {
            // cout << "failed" << endl;
            return false;
        }
    }
    // eskf
    
    int cam_states_next_id_ = 0;
    std::map<int, double> chi_squared_test_table_;
    // 理解为一种特征管理，本质是一个map，key为特征点id，value为特征类
    MapServer map_server;
};