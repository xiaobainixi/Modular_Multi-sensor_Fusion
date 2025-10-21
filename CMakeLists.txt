cmake_minimum_required(VERSION 3.0)
project(Modular_Multi-sensor_Fusion)

add_definitions(-w)

IF(NOT CMAKE_BUILD_TYPE)
    SET(CMAKE_BUILD_TYPE Debug)
ENDIF()

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3 -march=native")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
set(CMAKE_CXX_STANDARD 17)

include(cmake/FindGLEW.cmake)
include(cmake/FindGlog.cmake)
include(cmake/global_defination.cmake)
include(cmake/FindSuiteSparse.cmake)

# find
include_directories(${GLOG_INCLUDE_DIRS})

find_package(OpenCV 4)
if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV > 4 not found.")
    find_package(OpenCV 3)
endif()
MESSAGE("OPENCV VERSION:")
MESSAGE(${OpenCV_VERSION})

find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)
find_package(Pangolin REQUIRED)
include_directories(${Pangolin_INCLUDE_DIRS})

include_directories(
    /opt/local/include   #mac os
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}
    ${EIGEN3_INCLUDE_DIR}
    ${CERES_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIR}
    ${SUITESPARSE_INCLUDE_DIRS}
)

add_library(${PROJECT_NAME} SHARED
    src/observer/CameraObserver.cpp
    src/observer/GNSSObserver.cpp
    src/observer/LidarObserver.cpp
    src/observer/WheelObserver.cpp
    src/observer/GNSSWheelObserver.cpp

    src/predictor/IMUPredictor.cpp
    src/predictor/WheelIMUPredictor.cpp
    src/predictor/WheelPredictor.cpp

    src/visual/ImageProcessor.cpp
    src/visual/VinsFeatureManager.cpp
    src/common/InitialSFM.cpp
    src/common/InitialAlignment.cpp
    src/common/Solve5pts.cpp

    src/updater/Optimizer.cpp
    src/updater/Filter.cpp

    src/viewer/Viewer.cpp

    src/FusionSystem.cpp

    tool/DataLoader.cpp
)

target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}
    ${EIGEN3_LIBS}
    ${CERES_LIBRARIES}
    ${Pangolin_LIBRARIES}
    ${GLOG_LIBRARIES}
    ${SUITESPARSE_LIBRARIES}
)

add_executable(data_test app/data_test.cpp)
target_link_libraries(data_test ${PROJECT_NAME})
