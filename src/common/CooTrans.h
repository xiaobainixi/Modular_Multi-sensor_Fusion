#pragma once

#include <iostream>
#include <math.h>

#include <eigen3/Eigen/Core>

class CooTrans
{
public:
    CooTrans() {}
    // 输入为纬经高初始化东北高坐标原点
    CooTrans(double lat, double lon, double h)
    {
        // lon = lon * iPI;
        // lat = lat * iPI;
        double N = a / sqrt(1 - e2 * sin(lat) * sin(lat));

        double x = (N + h) * cos(lat) * cos(lon);
        double y = (N + h) * cos(lat) * sin(lon);
        double z = (N * (1 - e2) + h) * sin(lat);

        r << -sin(lon), cos(lon), 0,
            -sin(lat) * cos(lon), -sin(lat) * sin(lon), cos(lat),
            cos(lat) * cos(lon), cos(lat) * sin(lon), sin(lat);

        ecef_Ow[0] = x;
        ecef_Ow[1] = y;
        ecef_Ow[2] = z;
    }
    // 重新设置原点
    void SetECEFOw(double lat, double lon, double h)
    {
        // lon = lon * iPI;
        // lat = lat * iPI;
        double N = a / sqrt(1 - e2 * sin(lat) * sin(lat));

        double x = (N + h) * cos(lat) * cos(lon);
        double y = (N + h) * cos(lat) * sin(lon);
        double z = (N * (1 - e2) + h) * sin(lat);

        r << -sin(lon), cos(lon), 0,
            -sin(lat) * cos(lon), -sin(lat) * sin(lon), cos(lat),
            cos(lat) * cos(lon), cos(lat) * sin(lon), sin(lat);

        ecef_Ow[0] = x;
        ecef_Ow[1] = y;
        ecef_Ow[2] = z;
    }
    // 经纬高转东北高， 输入为纬经高，弧度
    void getENH(double lat, double lon, double h, double &x, double &y, double &z)
    {
        double N_ = a / sqrt(1 - e2 * sin(lat) * sin(lat));
        Eigen::Vector3d t((N_ + h) * cos(lat) * cos(lon), (N_ + h) * cos(lat) * sin(lon), (N_ * (1 - e2) + h) * sin(lat));
        t = r * (t - ecef_Ow);
        x = t[0];
        y = t[1];
        z = t[2];
    }

    Eigen::Vector3d getENH(double lat, double lon, double h)
    {
        double N_ = a / sqrt(1 - e2 * sin(lat) * sin(lat));
        Eigen::Vector3d t((N_ + h) * cos(lat) * cos(lon), (N_ + h) * cos(lat) * sin(lon), (N_ * (1 - e2) + h) * sin(lat));
        t = r * (t - ecef_Ow);
        return t;
    }

    // 输入为一个东北高坐标，返回其经纬高
    Eigen::Vector3d ECEF2LLA(Eigen::Vector3d t)
    {
        double X, Y, Z;
        t = r.transpose() * t + ecef_Ow;
        X = t[0];
        Y = t[1];
        Z = t[2];

        double B0, R, N;
        double B_, L_;
        R = sqrt(X * X + Y * Y);
        B0 = atan2(Z, R);
        while (1)
        {
            N = a / sqrt(1.0 - e2 * sin(B0) * sin(B0));
            B_ = atan2(Z + N * e2 * sin(B0), R);
            if (fabs(B_ - B0) < 1.0e-10)
                break;

            B0 = B_;
        }
        L_ = atan2(Y, X);

        Eigen::Vector3d LLA;
        LLA(2) = R / cos(B_) - N;
        // 弧度转换成经纬度
        LLA(0) = B_ * 180 / M_PI;
        LLA(1) = L_ * 180 / M_PI;

        return LLA;
    }
private:
    // double iPI = 0.0174532925199433; // 3.1415926535898/180.0
    // double PI = 3.1415926535898;
    // 54年北京坐标系参数
    // double a = 6378245.0;   // 长轴
    // double f = 1.0 / 298.3; // 扁率   (a-b)/a

    // 80年西安坐标系参数
    // double a = 6378140.0;
    // double f = 1 / 298.25722101;

    // WGS84坐标系参数
    double a = 6378137.0;
    double f = 1 / 298.257223563;

    double e2 = 2 * f - f * f; // e为第一偏心率，可以算可以直接提供，e2 = e * e

    Eigen::Vector3d ecef_Ow;
    Eigen::Matrix3d r;
};
