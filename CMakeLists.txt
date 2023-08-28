cmake_minimum_required(VERSION 3.0)
project(Modular_Multi-sensor_Fusion)

add_definitions(-w)

IF(NOT CMAKE_BUILD_TYPE)
    SET(CMAKE_BUILD_TYPE Debug)
ENDIF()

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -O3 -march=native")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3 -march=native")

# Check C++17 or C++0x support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++17" COMPILER_SUPPORTS_CXX17)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)

if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
    add_definitions(-DCOMPILEDWITHC11)
    message(STATUS "Using flag -std=c++17.")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
    add_definitions(-DCOMPILEDWITHC0X)
    message(STATUS "Using flag -std=c++0x.")
else()
    message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++17 support. Please use a different C++ compiler.")
endif()

include(cmake/FindGLEW.cmake)

find_package(OpenCV 4)

if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV > 4 not found.")
    find_package(OpenCV 3)
endif()

MESSAGE("OPENCV VERSION:")
MESSAGE(${OpenCV_VERSION})

find_package(Eigen3 REQUIRED)

find_package(Pangolin REQUIRED)
include_directories(${Pangolin_INCLUDE_DIRS})
include_directories(
    /opt/local/include   #mac os
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}
    ${EIGEN3_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIR}
)

add_library(${PROJECT_NAME} SHARED
    src/observer/CameraObserver.cpp
    src/observer/GPSObserver.cpp
    src/observer/LidarObserver.cpp
    src/observer/WheelObserver.cpp

    src/predictor/IMUPredictor.cpp
    src/predictor/WheelIMUPredictor.cpp
    src/predictor/WheelPredictor.cpp

    src/updater/Filter.cpp

    src/viewer/Viewer.cpp

    src/FusionSystem.cpp

    tool/DataLoader.cpp
)

target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}
    ${EIGEN3_LIBS}
    ${Pangolin_LIBRARIES}
)

add_executable(data_test app/data_test.cpp)
target_link_libraries(data_test ${PROJECT_NAME})
