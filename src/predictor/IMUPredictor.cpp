#include "IMUPredictor.h"

// todo 数个一起来还是一个一个，数个一起来比较好
void IMUPredictor::Predict() {
    IMUData imu_data = data_manager_ptr_->GetLastIMUData();
    if (imu_data.time_ <= 0.0)
        return;
    std::cout << imu_data.time_ << std::endl;
    std::cout << imu_data.a_.transpose() << std::endl;
    std::cout << imu_data.w_.transpose() << std::endl;
    // 第一个数据
    if (last_data_.time_ <= 0.0) {
        last_data_ = imu_data;
        return;
    }
    double delta_t = imu_data.time_ - last_data_.time_;
    if (delta_t <= 0.0)
        return;
    std::shared_ptr<State> state_ptr = std::make_shared<State>();
    state_ptr->time_ = imu_data.time_;

    last_data_ = imu_data;
}