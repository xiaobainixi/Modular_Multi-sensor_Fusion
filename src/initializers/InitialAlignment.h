#pragma once
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <map>

#include "preint/IMUPreintegration.h"
#include "visual/VinsFeatureManager.h"

class ImageFrame
{
public:
    ImageFrame() {};
    ImageFrame(
        const std::vector<FeaturePoint> &_points, double _t)
        : t{_t}, is_key_frame{false}
    {
        points = _points;
    };
    std::vector<FeaturePoint> points;
    double t;
    Eigen::Matrix3d R;
    Eigen::Vector3d T;
    IMUPreintegration *pre_integration;
    bool is_key_frame;
};

bool VisualIMUAlignment(
    const std::shared_ptr<Parameter> &param_ptr,
    std::map<double, ImageFrame> &all_image_frame,
    Eigen::Vector3d *Bgs, Eigen::Vector3d &g, Eigen::VectorXd &x);