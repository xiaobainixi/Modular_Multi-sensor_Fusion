#include "FusionSystem.h"

int main() {
    // data test
    std::shared_ptr<DataManager> data_manager_ptr = std::make_shared<DataManager>();
    std::shared_ptr<Parameter> param_ptr = std::make_shared<Parameter>("");
    std::shared_ptr<StateManager> state_manager_ptr = std::make_shared<StateManager>(param_ptr);
    
    FusionSystem fusion_system(param_ptr, state_manager_ptr, data_manager_ptr);

    IMUData imu_data;
    imu_data.time_ = 250;
    data_manager_ptr->Input(imu_data);
    imu_data.time_ = 251;
    data_manager_ptr->Input(imu_data);

    
    while(1) {
        usleep(100);
    }
    return 0;
}