#include "DataLoader.h"

DataLoader::DataLoader(const std::string & data_path) {
    ReadIMU(data_path + "/imu.txt");
    ReadGPS(data_path + "/gps.txt");
    ReadWheel(data_path + "/encoder.txt");
    ReadImage(data_path + "/image.txt");

    while (!gps_datas_.empty() || !imu_datas_.empty() || !wheel_datas_.empty() || !image_datas_.empty()) {
        double gps_time = !gps_datas_.empty() ? gps_datas_.front().time_ : DBL_MAX;
        double imu_time = !imu_datas_.empty() ? imu_datas_.front().time_ : DBL_MAX;
        double wheel_time = !wheel_datas_.empty() ? wheel_datas_.front().time_ : DBL_MAX;
        double image_time = !image_datas_.empty() ? image_datas_.front().time_ : DBL_MAX;
        double current_data_time = std::min(gps_time, std::min(imu_time, std::min(wheel_time, image_time)));
        if (current_data_time == gps_time) {
            datas_.push(gps_datas_.front());
            gps_datas_.pop();
        }
        else if (current_data_time == imu_time) {
            datas_.push(imu_datas_.front());
            imu_datas_.pop();
        }
        else if (current_data_time == wheel_time) {
            datas_.push(wheel_datas_.front());
            wheel_datas_.pop();
        }
        else {
            datas_.push(image_datas_.front());
            image_datas_.pop();
        }
    }
}

InputData DataLoader::GetNextData() {
    InputData output_data = datas_.front();
    // todo add read image
    datas_.pop();

    if (last_data_time_ > 0.0) {
        gettimeofday(&t2_, NULL);
        double used_time = (t2_.tv_sec - t1_.tv_sec) + (double)(t2_.tv_usec - t1_.tv_usec)/1000000.0;
        double delta_time = output_data.time_ - last_data_time_ - used_time;
        if (delta_time > 0)
            usleep(delta_time * 1e6);
    }

    gettimeofday(&t1_, NULL);
    last_data_time_ = output_data.time_;
    return output_data;
}

bool DataLoader::ReadIMU(const std::string & path)
{
    std::ifstream imu_file(path, std::ios::in);

    if (!imu_file.is_open())
    {
        std::cerr << "failure to open gps file" << std::endl;
        return false;
    }

    std::string imu_data_line;
    std::string temp;

    while (std::getline(imu_file, imu_data_line))
    {
        std::stringstream imu_data_ss;
        imu_data_ss << imu_data_line;
        InputData data;

        std::getline(imu_data_ss, temp, ',');
        data.time_ = std::stod(temp.substr(5, 5)) + std::stod(temp.substr(10, 4)) * 0.0001;;

        std::getline(imu_data_ss, temp, ',');
        std::getline(imu_data_ss, temp, ',');
        std::getline(imu_data_ss, temp, ',');
        std::getline(imu_data_ss, temp, ',');
        std::getline(imu_data_ss, temp, ',');
        std::getline(imu_data_ss, temp, ',');
        std::getline(imu_data_ss, temp, ',');

        std::getline(imu_data_ss, temp, ',');
        data.w_.x() = std::stod(temp); // * D2R;  // 这个数据集是弧度
        std::getline(imu_data_ss, temp, ',');
        data.w_.y() = std::stod(temp); // * D2R;
        std::getline(imu_data_ss, temp, ',');
        data.w_.z() = std::stod(temp); // * D2R;

        std::getline(imu_data_ss, temp, ',');
        data.a_.x() = std::stod(temp);
        std::getline(imu_data_ss, temp, ',');
        data.a_.y() = std::stod(temp);
        std::getline(imu_data_ss, temp, ',');
        data.a_.z() = std::stod(temp);

        data.data_type_ = 0;
        imu_datas_.push(data);
    }
    imu_file.close();
    return true;
}

bool DataLoader::ReadGPS(const std::string & path) {
    std::ifstream gps_file(path, std::ios::in);

    if (!gps_file.is_open())
    {
        std::cerr << "failure to open gps file" << std::endl;
        return false;
    }

    std::string gps_data_line;
    std::string temp;

    while (std::getline(gps_file, gps_data_line))
    {
        std::stringstream gps_data_ss;
        gps_data_ss << gps_data_line;
        InputData data;

        std::getline(gps_data_ss, temp, ',');
        data.time_ = std::stod(temp.substr(5, 5)) + std::stod(temp.substr(10, 4)) * 0.0001;;

        std::getline(gps_data_ss, temp, ',');
        data.lat_ = std::stod(temp) * d2r_;
        std::getline(gps_data_ss, temp, ',');
        data.lon_ = std::stod(temp) * d2r_;
        std::getline(gps_data_ss, temp, ',');
        data.h_ = std::stod(temp);

        std::getline(gps_data_ss, temp, ',');
        data.lat_error_ = std::stod(temp);
        std::getline(gps_data_ss, temp, ',');
        data.lon_error_  = std::stod(temp);
        std::getline(gps_data_ss, temp, ',');
        data.h_error_  = std::stod(temp);

        data.data_type_ = 2;
        gps_datas_.push(data);
    }
    gps_file.close();
}

bool DataLoader::ReadWheel(const std::string & path) {

}

bool DataLoader::ReadImage(const std::string & path) {

}