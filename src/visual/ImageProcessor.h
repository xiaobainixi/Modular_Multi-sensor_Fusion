#pragma once

#include <vector>
#include <map>
#include <thread>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h>

#include "common/StateManager.h"
#include "common/DataManager.h"

/**
 * @brief ImageProcessor Detects and tracks features
 *    in image sequences.
 */
class ImageProcessor
{
public:
    // Constructor
    ImageProcessor(const std::shared_ptr<Parameter> &param_ptr, const std::shared_ptr<DataManager> &data_manager_ptr,
           const std::shared_ptr<StateManager> &state_manager_ptr);
    // Disable copy and assign constructors.
    ImageProcessor(const ImageProcessor &) = delete;
    ImageProcessor operator=(const ImageProcessor &) = delete;

    // Destructor
    ~ImageProcessor();

    // Initialize the object.
    bool initialize();

    typedef std::shared_ptr<ImageProcessor> Ptr;
    typedef std::shared_ptr<const ImageProcessor> ConstPtr;

private:
    void Run();

    bool loadParameters();

    void undistortPoints(
        const std::vector<cv::Point2f> &pts_in,
        const cv::Vec4d &intrinsics,
        const std::string &distortion_model,
        const cv::Vec4d &distortion_coeffs,
        std::vector<cv::Point2f> &pts_out,
        const cv::Matx33d &rectification_matrix = cv::Matx33d::eye(),
        const cv::Vec4d &new_intrinsics = cv::Vec4d(1, 1, 0, 0));
    void undistortPoint(
        const cv::Point2f &pt_in, cv::Point2f &pt_out, const cv::Vec4d &intrinsics,
        const std::string &distortion_model,
        const cv::Vec4d &distortion_coeffs, const cv::Matx33d &rectification_matrix = cv::Matx33d::eye(),
        const cv::Vec4d &new_intrinsics = cv::Vec4d(1, 1, 0, 0));

    void readImage(const cv::Mat &_img,double _cur_time);
    void setMask();
    void addPoints();
    bool updateID(unsigned int i);
    void showUndistortion(const std::string &name);
    void rejectWithF();
    void undistortedPoints();
    bool inBorder(const cv::Point2f &pt);
    void reduceVector(std::vector<cv::Point2f> &v, std::vector<uchar> status);
    void reduceVector(std::vector<int> &v, std::vector<uchar> status);

    // Camera calibration parameters
    std::string cam0_distortion_model;
    cv::Vec4d cam0_intrinsics;
    cv::Vec4d cam0_distortion_coeffs;

    // 外部类
    std::shared_ptr<Parameter> param_ptr_;
    std::shared_ptr<DataManager> data_manager_ptr_;
    std::shared_ptr<StateManager> state_manager_ptr_;

    // 
    std::shared_ptr<std::thread> run_thread_ptr_;


    cv::Mat mask;
    cv::Mat fisheye_mask;
    cv::Mat prev_img, cur_img, forw_img;
    std::vector<cv::Point2f> n_pts;
    std::vector<cv::Point2f> prev_pts, cur_pts, forw_pts;
    std::vector<cv::Point2f> prev_un_pts, cur_un_pts;
    std::vector<cv::Point2f> pts_velocity;
    std::vector<int> ids;
    std::vector<int> track_cnt;
    std::map<int, cv::Point2f> cur_un_pts_map;
    std::map<int, cv::Point2f> prev_un_pts_map;
    // camodocal::CameraPtr m_camera;
    double cur_time;
    double prev_time;

    static int n_id;

    int MAX_CNT = 500;
    int MIN_DIST = 10;
    int WINDOW_SIZE = 10;
    // int FREQ = 30;
    double F_THRESHOLD = 1.0;
    // int SHOW_TRACK = 1;
    // int STEREO_TRACK = 1;
    int EQUALIZE = 1;
    int ROW = 480;
    int COL = 752;
    // int FOCAL_LENGTH;
    int FISHEYE = 0;

};

typedef ImageProcessor::Ptr ImageProcessorPtr;
typedef ImageProcessor::ConstPtr ImageProcessorConstPtr;

