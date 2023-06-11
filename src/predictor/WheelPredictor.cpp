#include "WheelPredictor.h"

void WheelPredictor::Predict() {
    WheelData wheel_data = data_manager_ptr_->GetLastWheelData();
    if (wheel_data.time_ <= 0.0)
        return;
    std::cout << wheel_data.time_ << std::endl;
    std::cout << wheel_data.lv_ << " " << wheel_data.rv_ << std::endl;
}