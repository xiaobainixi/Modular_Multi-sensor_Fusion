#include "tool/DataLoader.h"

#include "FusionSystem.h"

int main() {
    // data test
    std::shared_ptr<DataManager> data_manager_ptr = std::make_shared<DataManager>();
    std::shared_ptr<Parameter> param_ptr = std::make_shared<Parameter>("config.yaml");
    std::shared_ptr<StateManager> state_manager_ptr = std::make_shared<StateManager>(param_ptr);
    
    FusionSystem fusion_system(param_ptr, state_manager_ptr, data_manager_ptr);

    DataLoader data_loader(param_ptr);
    while(1) {
        InputData input_data = data_loader.GetNextData();
        if (input_data.data_type_ == 0) {
            IMUData imu_data;
            imu_data.time_ = input_data.time_;
            imu_data.a_ = input_data.a_;
            imu_data.w_ = input_data.w_;
            data_manager_ptr->Input(imu_data);
        } else if (input_data.data_type_ == 1) {
            WheelData wheel_data;
            wheel_data.time_ = input_data.time_;
            wheel_data.lv_ = input_data.lv_;
            wheel_data.rv_ = input_data.rv_;
            data_manager_ptr->Input(wheel_data);
        } else if (input_data.data_type_ == 2) {
            GPSData gps_data;
            gps_data.time_ = input_data.time_;
            gps_data.lat_ = input_data.lat_;
            gps_data.lon_ = input_data.lon_;
            gps_data.h_ = input_data.h_;
            data_manager_ptr->Input(gps_data);
            // todo error add
        } else if (input_data.data_type_ == 3) {
            CameraData camera_data;
            camera_data.time_ = input_data.time_;
            camera_data.image_ = cv::imread(input_data.img_path_, -1);
            cv::cvtColor(camera_data.image_, camera_data.image_,
                                 cv::COLOR_BayerRG2RGB);
            cv::cvtColor(camera_data.image_, camera_data.image_,
                            cv::COLOR_RGB2GRAY);
            // cv::imshow("aaaa", camera_data.image_);
            // cv::waitKey(1);
            if (!camera_data.image_.empty()) {
                // cv::cvtColor(camera_data.image_, camera_data.image_, CV_BayerRG2RGB);
                camera_data.image_ = camera_data.image_.rowRange(0, camera_data.image_.rows / 2);
                data_manager_ptr->Input(camera_data);
            } else {
                LOG(ERROR) << "读取图片失败，图片路径为： " << input_data.img_path_;
            }
                
        }
    }
    return 0;
}