#include "DataLoader.h"

DataLoader::DataLoader(const std::shared_ptr<Parameter> & param_ptr) {
    param_ptr_ = param_ptr;
    skip_seconds_ = param_ptr_->skip_seconds_;
    std::cout << "data path: " << param_ptr_->data_path_ << std::endl;
    if (param_ptr->data_type_ == "euroc") {
        ReadEurocIMU(param_ptr_->data_path_ + "mav0/imu0/data.csv");
        ReadEurocImage(param_ptr_->data_path_ + "mav0/cam0/data.csv");
    } else {
        ReadIMU(param_ptr_->data_path_ + "/imu.txt");
        ReadGNSS(param_ptr_->data_path_ + "/gnss.txt");
        ReadWheel(param_ptr_->data_path_ + "/encoder.txt");
        ReadImage(param_ptr_->data_path_ + "/image.txt");
    }

    while (!gnss_datas_.empty() || !imu_datas_.empty() || !wheel_datas_.empty() || !image_datas_.empty()) {
        double gnss_time = !gnss_datas_.empty() ? gnss_datas_.front().time_ : DBL_MAX;
        double imu_time = !imu_datas_.empty() ? imu_datas_.front().time_ : DBL_MAX;
        double wheel_time = !wheel_datas_.empty() ? wheel_datas_.front().time_ : DBL_MAX;
        double image_time = !image_datas_.empty() ? image_datas_.front().time_ : DBL_MAX;
        double current_data_time = std::min(gnss_time, std::min(imu_time, std::min(wheel_time, image_time)));
        if (current_data_time == gnss_time) {
            datas_.push(gnss_datas_.front());
            gnss_datas_.pop();
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

    if (!datas_.empty()) {
        double skip_time = datas_.front().time_ + skip_seconds_;
        while (!datas_.empty() && datas_.front().time_ < skip_time) {
            datas_.pop();
    }
}
}

InputData DataLoader::GetNextData() {
    if (datas_.empty()) {
        std::cerr << "data empty" << std::endl;
        exit(0);
    }
    InputData output_data = datas_.front();
    // todo add read image
    datas_.pop();

    if (last_data_time_ > 0.0) {
        gettimeofday(&t2_, NULL);
        double used_time = (t2_.tv_sec - t1_.tv_sec) + (double)(t2_.tv_usec - t1_.tv_usec)/1000000.0;
        double delta_time = output_data.time_ - last_data_time_ - used_time;
        if (delta_time > 0)
            usleep(delta_time * 1e6 / param_ptr_->play_speed_);
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
        std::cerr << "failure to open imu file: " << path << std::endl;
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
        if (temp.size() > 10)
            data.time_ = std::stod(temp.substr(5, 5)) + std::stod(temp.substr(10, 4)) * 0.0001;
        else
            data.time_ = std::stod(temp);

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
    LOG(INFO) << "imu data size: " << imu_datas_.size() << std::endl;
    return true;
}

bool DataLoader::ReadGNSS(const std::string & path) {
    std::ifstream gnss_file(path, std::ios::in);

    if (!gnss_file.is_open())
    {
        std::cerr << "failure to open gnss file: " << path << std::endl;
        return false;
    }

    std::string gnss_data_line;
    std::string temp;

    while (std::getline(gnss_file, gnss_data_line))
    {
        std::stringstream gnss_data_ss;
        gnss_data_ss << gnss_data_line;
        InputData data;

        std::getline(gnss_data_ss, temp, ',');
        if (temp.size() > 10)
            data.time_ = std::stod(temp.substr(5, 5)) + std::stod(temp.substr(10, 4)) * 0.0001;
        else
            data.time_ = std::stod(temp);

        std::getline(gnss_data_ss, temp, ',');
        data.lat_ = std::stod(temp) * d2r_;
        std::getline(gnss_data_ss, temp, ',');
        data.lon_ = std::stod(temp) * d2r_;
        std::getline(gnss_data_ss, temp, ',');
        data.h_ = std::stod(temp);

        std::getline(gnss_data_ss, temp, ',');
        data.lat_error_ = std::stod(temp);
        std::getline(gnss_data_ss, temp, ',');
        data.lon_error_  = std::stod(temp);
        std::getline(gnss_data_ss, temp, ',');
        data.h_error_  = std::stod(temp);

        data.data_type_ = 2;
        gnss_datas_.push(data);
    }
    LOG(INFO) << "gnss data size: " << gnss_datas_.size() << std::endl;
    gnss_file.close();
    return true;
}

bool DataLoader::ReadWheel(const std::string & path) {
    std::ifstream wheel_file(path, std::ios::in);

    if (!wheel_file.is_open())
    {
        std::cerr << "failure to open wheel file: " << path << std::endl;
        return false;
    }

    std::string wheel_data_line;
    std::string temp;

    double last_left_encoder_data = 0.0, last_right_encoder_data = 0.0, last_encoder_data_time = -1.0;
    while (std::getline(wheel_file, wheel_data_line))
    {
        std::stringstream wheel_data_ss;
        wheel_data_ss << wheel_data_line;
        

        std::getline(wheel_data_ss, temp, ',');
        double cur_encoder_data_time = std::stod(temp.substr(5, 5)) + std::stod(temp.substr(10, 4)) * 0.0001;

        std::getline(wheel_data_ss, temp, ',');
        double cur_left_encoder_data = std::stod(temp);
        std::getline(wheel_data_ss, temp, ',');
        double cur_right_encoder_data = std::stod(temp);

        if (last_encoder_data_time < 0.0) {
            last_encoder_data_time = cur_encoder_data_time;
            last_left_encoder_data = cur_left_encoder_data;
            last_right_encoder_data = cur_right_encoder_data;
            continue;
        }

        InputData data;
        data.time_ = cur_encoder_data_time;
        data.data_type_ = 1;
        double delta_time = cur_encoder_data_time - last_encoder_data_time;
        data.lv_ = (cur_left_encoder_data - last_left_encoder_data) * param_ptr_->wheel_kl_ / delta_time;
        data.rv_ = (cur_right_encoder_data - last_right_encoder_data) * param_ptr_->wheel_kr_ / delta_time;

        wheel_datas_.push(data);

        last_encoder_data_time = cur_encoder_data_time;
        last_left_encoder_data = cur_left_encoder_data;
        last_right_encoder_data = cur_right_encoder_data;
    }
    wheel_file.close();
    LOG(INFO) << "wheel data size: " << wheel_datas_.size() << std::endl;
    return true;
}

bool DataLoader::ReadImage(const std::string & path) {
    std::ifstream img_file(path, std::ios::in);

    if (!img_file.is_open())
    {
        std::cerr << "failure to open image file: " << path << std::endl;
        return false;
    }

    std::string img_data_line;
    std::string temp;

    while (std::getline(img_file, img_data_line))
    {
        std::stringstream img_data_ss;
        img_data_ss << img_data_line;
        std::string tmp;
        while (std::getline(img_data_ss, tmp, '/'))
        {
        }

        InputData data;
        data.time_ = std::stod(tmp.substr(5, 5)) + std::stod(tmp.substr(10, 3)) * 0.001;
        data.data_type_ = 3;
        data.img_path_ = param_ptr_->data_path_ + "/" + img_data_line;
        image_datas_.push(data);

    }
    img_file.close();
    LOG(INFO) << "image data size: " << image_datas_.size() << std::endl;
    return true;
}

bool DataLoader::ReadEurocImage(const std::string & path) {
    std::ifstream img_file(path, std::ios::in);
    if (!img_file.is_open()) {
        std::cerr << "failure to open image file: " << path << std::endl;
        return false;
    }

    std::string line;
    // 跳过表头: #timestamp [ns],filename
    if (std::getline(img_file, line)) {}

    while (std::getline(img_file, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string ts_str, file_str;
        std::getline(ss, ts_str, ',');
        std::getline(ss, file_str, ',');
        if (ts_str.empty() || file_str.empty()) continue;

        // 去除可能的回车/空格
        file_str.erase(
            std::remove_if(file_str.begin(), file_str.end(),
                           [](unsigned char c){ return c == '\r' || c == '\n' || std::isspace(c); }),
            file_str.end());

        // Euroc时间戳是纳秒
        long long ts_ns = std::stoll(ts_str);
        double t_sec = static_cast<double>(ts_ns) * 1e-9;

        InputData data;
        data.time_ = t_sec;
        data.data_type_ = 3;
        data.img_path_ = param_ptr_->data_path_ + "/mav0/cam0/data/" + file_str; // 文件名如 1403636579763555584.png
        image_datas_.push(data);
    }
    img_file.close();
    LOG(INFO) << "image data size: " << image_datas_.size() << std::endl;
    return true;
}

bool DataLoader::ReadEurocIMU(const std::string & path)
{
    std::ifstream imu_file(path, std::ios::in);
    if (!imu_file.is_open()) {
        std::cerr << "failure to open imu file: " << path << std::endl;
        return false;
    }

    std::string line;
    // 跳过表头
    if (std::getline(imu_file, line)) {
        // 例如: #timestamp [ns],w_RS_S_x...
    }

    while (std::getline(imu_file, line)) {
        if (line.empty()) continue;
        if (line[0] == '#') continue;

        std::stringstream ss(line);
        std::string ts_str;
        std::string wx_str, wy_str, wz_str;
        std::string ax_str, ay_str, az_str;

        // 期望 7 个字段
        if (!std::getline(ss, ts_str, ',')) continue;
        if (!std::getline(ss, wx_str, ',')) continue;
        if (!std::getline(ss, wy_str, ',')) continue;
        if (!std::getline(ss, wz_str, ',')) continue;
        if (!std::getline(ss, ax_str, ',')) continue;
        if (!std::getline(ss, ay_str, ',')) continue;
        if (!std::getline(ss, az_str, ',')) continue;

        // 时间戳 ns -> s
        long long ts_ns = std::stoll(ts_str);
        double t_sec = static_cast<double>(ts_ns) * 1e-9;

        InputData data;
        data.time_ = t_sec;
        data.data_type_ = 0;
        data.w_.x() = std::stod(wx_str);
        data.w_.y() = std::stod(wy_str);
        data.w_.z() = std::stod(wz_str);
        data.a_.x() = std::stod(ax_str);
        data.a_.y() = std::stod(ay_str);
        data.a_.z() = std::stod(az_str);

        imu_datas_.push(data);
    }
    imu_file.close();
    LOG(INFO) << "imu data size: " << imu_datas_.size();
    return true;
}