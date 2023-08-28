#pragma once
#include <unistd.h>

#include "common/StateManager.h"
#include "common/DataManager.h"

#include "observer/GPSObserver.h"
#include "observer/WheelObserver.h"
#include "observer/CameraObserver.h"

#include "viewer/Viewer.h"

class Updater {
public:
    Updater() {}
protected:
    virtual void Run() = 0;
    std::shared_ptr<std::thread> run_thread_ptr_;
    std::shared_ptr<Viewer> viewer_ptr_;
};