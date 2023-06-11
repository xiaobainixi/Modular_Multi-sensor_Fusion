#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <Eigen/Core>

class Parameter {
public:
    Parameter(const std::string & file) {

    }
    

    void ConfigureStatusDim(int type) {
        // 除非特殊定义，否则默认只使用IMU
        if (type == 1) {

        }
    }

    // 不同模式状态不同，默认是纯IMU做预测
    int STATE_DIM = 15;
    int POSI_INDEX = 0;
    int VEL_INDEX_STATE_ = 3;
    int ORI_INDEX_STATE_ = 6;
    int GYRO_BIAS_INDEX_STATE_ = 9;
    int ACC_BIAS_INDEX_STATE_ = 12;
};