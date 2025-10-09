#pragma once
#include "Updater.h"

#include "marginalization/marginalization_factor.h"
#include "marginalization/marginalization_info.h"
#include "marginalization/residual_block_info.h"

class QLocalParameterization : public ceres::LocalParameterization
{
    virtual bool Plus(const double *x, const double *delta, double *x_plus_delta) const;
    virtual bool ComputeJacobian(const double *x, double *jacobian) const;
    virtual int GlobalSize() const { return 4; };
    virtual int LocalSize() const { return 3; };
};

class Optimizer : public Updater
{
public:
    Optimizer(const std::shared_ptr<Parameter> &param_ptr, const std::shared_ptr<DataManager> &data_manager_ptr,
           const std::shared_ptr<StateManager> &state_manager_ptr, std::shared_ptr<Viewer> viewer_ptr = nullptr)
    {
        viewer_ptr_ = viewer_ptr;
        param_ptr_ = param_ptr;
        data_manager_ptr_ = data_manager_ptr;
        state_manager_ptr_ = state_manager_ptr;

        if (param_ptr_->use_imu_ && param_ptr_->wheel_use_type_ != 1) {
            preintegration_type_ = 0;
            predictor_ptr_ = std::make_shared<IMUPredictor>(state_manager_ptr_, param_ptr_, data_manager_ptr_, viewer_ptr_);
        } else if (param_ptr_->use_imu_ && param_ptr_->wheel_use_type_ == 1) {
            // todo imu+wheel
            preintegration_type_ = 2;
            predictor_ptr_ = std::make_shared<WheelIMUPredictor>(state_manager_ptr_, param_ptr_, data_manager_ptr_, viewer_ptr_);
        } else if (!param_ptr_->use_imu_ && param_ptr_->wheel_use_type_ == 1) {
            // todo wheel
            preintegration_type_ = 1;
            predictor_ptr_ = std::make_shared<WheelPredictor>(state_manager_ptr_, param_ptr_, data_manager_ptr_, viewer_ptr_);
        }

        if (!predictor_ptr_) {
            LOG(ERROR) << "没有初始化预测模块，程序将不会运行，请检查配置文件";
            return;
        }

        coo_trans_ptr_ = std::make_shared<CooTrans>();
        if (param_ptr->gnss_wheel_align_)
            gnss_wheel_observer_ptr_ = std::make_shared<GNSSWheelObserver>(param_ptr, data_manager_ptr, coo_trans_ptr_, state_manager_ptr, viewer_ptr_);
        else {
            if (param_ptr->use_gnss_)
                gnss_observer_ptr_ = std::make_shared<GNSSObserver>(param_ptr, data_manager_ptr, coo_trans_ptr_, state_manager_ptr, viewer_ptr_);
            
            if (param_ptr->wheel_use_type_ == 2)
                wheel_observer_ptr_ = std::make_shared<WheelObserver>(param_ptr, data_manager_ptr, state_manager_ptr, viewer_ptr_);
        }

        if (param_ptr->use_camera_) {
            image_processor_ptr_ = std::make_shared<ImageProcessor>(param_ptr, data_manager_ptr, state_manager_ptr);
            camera_observer_ptr_ = std::make_shared<CameraObserver>(param_ptr, data_manager_ptr, state_manager_ptr, viewer_ptr_);
        }


        initializers_ptr_ = std::make_shared<Initializers>(param_ptr, data_manager_ptr, coo_trans_ptr_, state_manager_ptr);
        run_thread_ptr_ = std::make_shared<std::thread>(&Optimizer::Run, this);
    }

    
private:
    int preintegration_type_; // 0 imu, 1 wheel, 2 imu+wheel
    void Run();
    void SlideWindow();
    void Optimization();
    std::shared_ptr<Parameter> param_ptr_;
    std::shared_ptr<DataManager> data_manager_ptr_;
    std::shared_ptr<StateManager> state_manager_ptr_;
    std::shared_ptr<Preintegration> preintegration_ptr_;

    std::shared_ptr<CooTrans> coo_trans_ptr_;
    // todo add
    std::shared_ptr<GNSSObserver> gnss_observer_ptr_;
    std::shared_ptr<WheelObserver> wheel_observer_ptr_;
    std::shared_ptr<GNSSWheelObserver> gnss_wheel_observer_ptr_;
    std::shared_ptr<ImageProcessor> image_processor_ptr_;
    std::shared_ptr<CameraObserver> camera_observer_ptr_;

    // 边缘化
    // Marginalization variables
    std::shared_ptr<MarginalizationInfo> last_marginalization_info_{nullptr};
    std::vector<double *> last_marginalization_parameter_blocks_;

    // tmp data
    GNSSData last_gnss_data_;
    WheelData last_wheel_data_;
    FeatureData last_feature_data_;
};