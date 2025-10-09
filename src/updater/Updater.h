#pragma once
#include <unistd.h>

#include "common/StateManager.h"
#include "common/DataManager.h"
#include "common/Initializers.h"

#include "predictor/IMUPredictor.h"
#include "predictor/WheelPredictor.h"
#include "predictor/WheelIMUPredictor.h"

#include "observer/GNSSObserver.h"
#include "observer/WheelObserver.h"
#include "observer/CameraObserver.h"
#include "observer/GNSSWheelObserver.h"

#include "visual/ImageProcessor.h"

#include "viewer/Viewer.h"

class Updater {
public:
    Updater() {}
protected:
    virtual void Run() = 0;
    std::shared_ptr<std::thread> run_thread_ptr_;
    std::shared_ptr<Viewer> viewer_ptr_;
    std::shared_ptr<Predictor> predictor_ptr_;
    std::shared_ptr<Initializers> initializers_ptr_;
    
    bool initialized_ = false;

    // tmp data
    GNSSData last_gnss_data_;
    WheelData last_wheel_data_;
    FeatureData last_feature_data_;
};