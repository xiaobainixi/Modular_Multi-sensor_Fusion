#include "DataLoader.h"

bool DataLoader::ReadGPS(const std::string & path)
{
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
        data.lat_ = std::stod(temp) * d2r;
        std::getline(gps_data_ss, temp, ',');
        data.lon_ = std::stod(temp) * d2r;
        std::getline(gps_data_ss, temp, ',');
        data.h_ = std::stod(temp);

        std::getline(gps_data_ss, temp, ',');
        data.lat_error_ = std::stod(temp);
        std::getline(gps_data_ss, temp, ',');
        data.lon_error_  = std::stod(temp);
        std::getline(gps_data_ss, temp, ',');
        data.h_error_  = std::stod(temp);

        datas_.push(data);
    }
    gps_file.close();
    return true;
}