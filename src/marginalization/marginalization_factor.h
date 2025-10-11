#pragma once

#include "marginalization/marginalization_info.h"

#include <ceres/ceres.h>
#include <memory>

class MarginalizationFactor : public ceres::CostFunction {

public:
    MarginalizationFactor() = delete;
    explicit MarginalizationFactor(std::shared_ptr<MarginalizationInfo> marg_info)
        : marg_info_(std::move(marg_info)) {

        // 给定每个参数块数据大小
        for (auto size : marg_info_->remainedBlockSize()) {
            mutable_parameter_block_sizes()->push_back(size);
        }

        // 残差大小
        set_num_residuals(marg_info_->remainedSize());
    }

    bool Evaluate(const double *const *parameters, double *residuals, double **jacobians) const override {
        int marginalizaed_size = marg_info_->marginalizedSize();
        int remained_size      = marg_info_->remainedSize();

        const std::vector<int> &remained_block_index     = marg_info_->remainedBlockIndex();
        const std::vector<int> &remained_block_size      = marg_info_->remainedBlockSize();
        const std::vector<double *> &remained_block_data = marg_info_->remainedBlockData();

        Eigen::VectorXd dx(remained_size);
        // LOG(INFO) << "remained_block_size: " << marginalizaed_size << " " << remained_block_size.size() << " " << remained_size;
        for (size_t i = 0; i < remained_block_size.size(); i++) {
            int size  = remained_block_size[i];
            int index = remained_block_index[i] - marginalizaed_size;

            // LOG(INFO) << "size: " << size << " " << index << " " << parameters[i] << " " << parameters[i][0] << " " << remained_block_data[i][0];
            Eigen::VectorXd x  = Eigen::Map<const Eigen::VectorXd>(parameters[i], size);
            Eigen::VectorXd x0 = Eigen::Map<const Eigen::VectorXd>(remained_block_data[i], size);

            // dx = x - x0
            if (size == POSE_GLOBAL_SIZE) {
                Eigen::Quaterniond dq(Eigen::Quaterniond(x0(3), x0(0), x0(1), x0(2)).inverse() *
                                      Eigen::Quaterniond(x(3), x(0), x(1), x(2)));
                dx.segment(index, 3) = 2.0 * dq.vec();
                if (dq.w() < 0) {
                    dx.segment<3>(index) = -2.0 * dq.vec();
                }
            } else {
                dx.segment(index, size) = x - x0;
            }
        }

        // e = e0 + J0 * dx
        Eigen::Map<Eigen::VectorXd>(residuals, remained_size) =
            marg_info_->linearizedResiduals() + marg_info_->linearizedJacobians() * dx;

        if (jacobians) {

            for (size_t i = 0; i < remained_block_size.size(); i++) {
                if (jacobians[i]) {
                    int size       = remained_block_size[i];
                    int index      = remained_block_index[i] - marginalizaed_size;
                    int local_size = marg_info_->localSize(size);

                    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> jacobian(
                        jacobians[i], remained_size, size);

                    // J = J0
                    jacobian.setZero();
                    jacobian.leftCols(local_size) = marg_info_->linearizedJacobians().middleCols(index, local_size);
                }
            }
        }

        return true;
    }

private:
    std::shared_ptr<MarginalizationInfo> marg_info_;
};
