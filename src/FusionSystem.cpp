#include "FusionSystem.h"

void FusionSystem::FilterProcessing()
{
    GPSData last_gps_data;
    // 循环读数据
    while (1)
    {
        predictor_ptr_->Predict();

        std::shared_ptr<State> state_ptr;
        if(!state_manager_ptr_->GetNearestState(state_ptr)) {
            usleep(100);
            continue;
        }

        if (filt_updater_ptr_) {
            GPSData cur_gps_data = data_manager_ptr_->GetLastGPSData();
            if (cur_gps_data.time_ < 0.0 || cur_gps_data.time_ <= last_gps_data.time_) {
                usleep(100);
                continue;
            }

            filt_updater_ptr_->Update(state_ptr, cur_gps_data);
        }
        usleep(100);
    }
}