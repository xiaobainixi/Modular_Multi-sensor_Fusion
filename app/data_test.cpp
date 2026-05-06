#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <string>
#include <unistd.h>

#include "tool/DataLoader.h"
#include "FusionSystem.h"

namespace {

void DumpLatestState(const std::shared_ptr<StateManager> &state_manager_ptr,
                     std::ofstream &trajectory_file,
                     std::ofstream &state_csv_file,
                     double &last_dumped_time,
                     size_t &dumped_state_count) {
    std::shared_ptr<State> state_ptr;
    if (!state_manager_ptr->GetNearestState(state_ptr) || !state_ptr)
        return;
    if (state_ptr->time_ <= last_dumped_time)
        return;

    const Eigen::Quaterniond q = state_ptr->Rwb_.normalized();
    trajectory_file << std::fixed << std::setprecision(9)
                    << state_ptr->time_ << " "
                    << state_ptr->twb_.x() << " "
                    << state_ptr->twb_.y() << " "
                    << state_ptr->twb_.z() << " "
                    << q.x() << " "
                    << q.y() << " "
                    << q.z() << " "
                    << q.w() << "\n";

    state_csv_file << std::fixed << std::setprecision(9)
                   << state_ptr->time_ << ","
                   << state_ptr->twb_.x() << ","
                   << state_ptr->twb_.y() << ","
                   << state_ptr->twb_.z() << ","
                   << q.x() << ","
                   << q.y() << ","
                   << q.z() << ","
                   << q.w() << ","
                   << state_ptr->Vw_.x() << ","
                   << state_ptr->Vw_.y() << ","
                   << state_ptr->Vw_.z() << "\n";

    trajectory_file.flush();
    state_csv_file.flush();
    last_dumped_time = state_ptr->time_;
    ++dumped_state_count;
}

std::string GetArg(int argc, char **argv, const std::string &flag, const std::string &default_value) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::string(argv[i]) == flag)
            return argv[i + 1];
    }
    return default_value;
}

}  // namespace

int main(int argc, char **argv) {
    const std::string config_path = GetArg(argc, argv, "--config", "euroc.yaml");
    const std::string output_dir = GetArg(argc, argv, "--output-dir", "./outputs/default_run");
    std::filesystem::create_directories(output_dir);

    std::ofstream trajectory_file(output_dir + "/estimated_trajectory.tum", std::ios::out | std::ios::trunc);
    std::ofstream state_csv_file(output_dir + "/estimated_states.csv", std::ios::out | std::ios::trunc);
    std::ofstream summary_file(output_dir + "/run_summary.txt", std::ios::out | std::ios::trunc);
    state_csv_file << "timestamp,px,py,pz,qx,qy,qz,qw,vx,vy,vz\n";

    std::shared_ptr<Parameter> param_ptr = std::make_shared<Parameter>(config_path);
    std::shared_ptr<DataManager> data_manager_ptr = std::make_shared<DataManager>(param_ptr);
    std::shared_ptr<StateManager> state_manager_ptr = std::make_shared<StateManager>(param_ptr);

    FusionSystem fusion_system(param_ptr, state_manager_ptr, data_manager_ptr);
    DataLoader data_loader(param_ptr);

    double last_dumped_time = -1.0;
    size_t dumped_state_count = 0;
    InputData input_data;
    while (data_loader.GetNextData(input_data)) {
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
            GNSSData gnss_data;
            gnss_data.time_ = input_data.time_;
            gnss_data.lat_ = input_data.lat_;
            gnss_data.lon_ = input_data.lon_;
            gnss_data.h_ = input_data.h_;
            data_manager_ptr->Input(gnss_data);
        } else if (input_data.data_type_ == 3) {
            CameraData camera_data;
            camera_data.time_ = input_data.time_;
            camera_data.image_ = cv::imread(input_data.img_path_, cv::IMREAD_UNCHANGED);

            if (!camera_data.image_.empty()) {
                if (camera_data.image_.channels() == 1 && param_ptr->camera_input_is_bayer_) {
                    cv::Mat rgb_image;
                    cv::cvtColor(camera_data.image_, rgb_image, cv::COLOR_BayerRG2RGB);
                    cv::cvtColor(rgb_image, camera_data.image_, cv::COLOR_RGB2GRAY);
                } else if (camera_data.image_.channels() == 3) {
                    cv::cvtColor(camera_data.image_, camera_data.image_, cv::COLOR_BGR2GRAY);
                } else if (camera_data.image_.channels() == 4) {
                    cv::cvtColor(camera_data.image_, camera_data.image_, cv::COLOR_BGRA2GRAY);
                }
                data_manager_ptr->Input(camera_data);
            } else {
                LOG(ERROR) << "读取图片失败，图片路径为： " << input_data.img_path_;
            }
        }

        DumpLatestState(state_manager_ptr, trajectory_file, state_csv_file, last_dumped_time, dumped_state_count);
    }

    for (int idle_round = 0; idle_round < 500; ++idle_round) {
        const double previous_dump_time = last_dumped_time;
        usleep(10000);
        DumpLatestState(state_manager_ptr, trajectory_file, state_csv_file, last_dumped_time, dumped_state_count);
        if (last_dumped_time > previous_dump_time)
            idle_round = 0;
    }

    summary_file << "config=" << config_path << "\n";
    summary_file << "output_dir=" << output_dir << "\n";
    summary_file << "state_count=" << dumped_state_count << "\n";
    summary_file << "last_state_time=" << std::fixed << std::setprecision(9) << last_dumped_time << "\n";
    summary_file.flush();
    trajectory_file.flush();
    state_csv_file.flush();

    std::exit(0);
}
