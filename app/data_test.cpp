#include "FusionSystem.h"

int main() {
    // data test
    std::shared_ptr<DataManager> data_manager_ptr_ = std::make_shared<DataManager>();
    std::shared_ptr<Parameter> param_ptr_ = std::make_shared<Parameter>("");
    std::shared_ptr<StateManager> state_manager_ptr_ = std::make_shared<StateManager>(param_ptr_);
    
    std::shared_ptr<IMUPredictor> imu_predictor_ptr = std::make_shared<IMUPredictor>(state_manager_ptr_, param_ptr_, data_manager_ptr_);
    std::shared_ptr<Predictor> predictor_ptr_ = std::dynamic_pointer_cast<Predictor>(imu_predictor_ptr);

    IMUData imu_data;
    imu_data.time_ = 250;
    data_manager_ptr_->Input(imu_data);
    predictor_ptr_->Predict();
    imu_data.time_ = 251;
    data_manager_ptr_->Input(imu_data);
    predictor_ptr_->Predict();
    return 0;
}