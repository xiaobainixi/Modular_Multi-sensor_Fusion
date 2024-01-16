#pragma once
#include "Updater.h"

enum OBSType {
    GPS = 1,
    WHEEL,
    CAMERA,
};

class Filter : public Updater
{
public:
    Filter(const std::shared_ptr<Parameter> &param_ptr, const std::shared_ptr<DataManager> &data_manager_ptr,
           const std::shared_ptr<StateManager> &state_manager_ptr, std::shared_ptr<Viewer> viewer_ptr = nullptr)
    {
        viewer_ptr_ = viewer_ptr;
        param_ptr_ = param_ptr;
        data_manager_ptr_ = data_manager_ptr;
        state_manager_ptr_ = state_manager_ptr;

        if (param_ptr_->use_imu_ && param_ptr_->wheel_use_type_ != 1) {
            predictor_ptr_ = std::make_shared<IMUPredictor>(state_manager_ptr_, param_ptr_, data_manager_ptr_, viewer_ptr_);
        } else if (param_ptr_->use_imu_ && param_ptr_->wheel_use_type_ == 1) {
            // todo imu+wheel
        } else if (!param_ptr_->use_imu_ && param_ptr_->wheel_use_type_ == 1) {
            // todo wheel
            predictor_ptr_ = std::make_shared<WheelPredictor>(state_manager_ptr_, param_ptr_, data_manager_ptr_, viewer_ptr_);
        }

        if (param_ptr->gps_wheel_align_)
            gps_wheel_observer_ptr_ = std::make_shared<GPSWheelObserver>(param_ptr, data_manager_ptr, state_manager_ptr, viewer_ptr_);
        else {
            if (param_ptr->use_gps_)
                gps_observer_ptr_ = std::make_shared<GPSObserver>(param_ptr, data_manager_ptr, state_manager_ptr, viewer_ptr_);
            
            if (param_ptr->wheel_use_type_ == 2)
                wheel_observer_ptr_ = std::make_shared<WheelObserver>(param_ptr, data_manager_ptr, state_manager_ptr, viewer_ptr_);
        }

        if (param_ptr->use_camera_) {
            image_processor_ptr_ = std::make_shared<ImageProcessor>(param_ptr, data_manager_ptr, state_manager_ptr);
            camera_observer_ptr_ = std::make_shared<CameraObserver>(param_ptr, data_manager_ptr, state_manager_ptr, viewer_ptr_);
        }

        run_thread_ptr_ = std::make_shared<std::thread>(&Filter::Run, this);
    }

    
private:
    void Run();
    void DelayedRun();
    void UpdateFromGPS(const std::shared_ptr<State> & state_ptr);
    void UpdateFromWheel(const std::shared_ptr<State> & state_ptr);
    void UpdateFromGPSWheel(const std::shared_ptr<State> & state_ptr);
    void UpdateFromCamera(const std::shared_ptr<State> & state_ptr);
    void ESKFUpdate(
        const Eigen::MatrixXd & H, const Eigen::MatrixXd & C, const Eigen::MatrixXd & R,
        Eigen::MatrixXd & Z, Eigen::MatrixXd & C_new, Eigen::VectorXd & X);
    std::shared_ptr<Parameter> param_ptr_;
    std::shared_ptr<DataManager> data_manager_ptr_;
    std::shared_ptr<StateManager> state_manager_ptr_;

    // todo add
    std::shared_ptr<GPSObserver> gps_observer_ptr_;
    std::shared_ptr<WheelObserver> wheel_observer_ptr_;
    std::shared_ptr<GPSWheelObserver> gps_wheel_observer_ptr_;
    std::shared_ptr<ImageProcessor> image_processor_ptr_;
    std::shared_ptr<CameraObserver> camera_observer_ptr_;

    // tmp data
    GPSData last_gps_data_;
    WheelData last_wheel_data_;
    FeatureData last_feature_data_;
};