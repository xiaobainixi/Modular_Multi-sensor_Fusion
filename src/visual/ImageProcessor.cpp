#include <iostream>
#include <algorithm>
#include <set>
#include <Eigen/Dense>
#include <cstdlib>
#include <ctime>

// #include <random_numbers/random_numbers.h>
#include "visual/ImageProcessor.h"

ImageProcessor::ImageProcessor(
    const std::shared_ptr<Parameter> &param_ptr, const std::shared_ptr<DataManager> &data_manager_ptr,
    const std::shared_ptr<StateManager> &state_manager_ptr)
{
    param_ptr_ = param_ptr;
    data_manager_ptr_ = data_manager_ptr;
    state_manager_ptr_ = state_manager_ptr;

    if (!param_ptr_ || !data_manager_ptr_ || !state_manager_ptr_)
        return;

    loadParameters();

    run_thread_ptr_ = std::make_shared<std::thread>(&ImageProcessor::Run, this);
}

ImageProcessor::~ImageProcessor()
{
    cv::destroyAllWindows();
}

void ImageProcessor::Run() {
    double last_image_time = -1.0;
    CameraData camera_data;
    while (1)
    {
        bool get = data_manager_ptr_->GetNewCameraData(camera_data, last_image_time);
        if (!get) {
            usleep(1000);
            continue;
        }
        last_image_time = camera_data.time_;
        readImage(camera_data.image_, camera_data.time_);
        usleep(1000);
    }
}

/**
 * @brief 导入节点launch时提供的各种参数
 * @return 成功或失败 一直为true
 */
bool ImageProcessor::loadParameters()
{
    LOG(INFO) << "Load vo param";
    // // Camera calibration parameters
    // // 1. 畸变模型，默认都是用的radtan模型
    cam0_distortion_model = param_ptr_->cam_distortion_model_;
    cam0_intrinsics[0] = param_ptr_->cam_intrinsics_[0];
    cam0_intrinsics[1] = param_ptr_->cam_intrinsics_[1];
    cam0_intrinsics[2] = param_ptr_->cam_intrinsics_[2];
    cam0_intrinsics[3] = param_ptr_->cam_intrinsics_[3];

    cam0_distortion_coeffs = param_ptr_->cam_distortion_coeffs_;
    return true;
}

/**
 * @brief 根据图像点的像素坐标、相机内参矩阵、畸变模型和系数，得到图像点在相机系下去畸变的像素坐标，如果使用默认的最后两个参数则是归一化坐标
 * @param  pts_in 输入像素点
 * @param  intrinsics fx, fy, cx, cy
 * @param  distortion_model 相机畸变模型
 * @param  distortion_coeffs 畸变系数
 * @param  pts_out 输出矫正后的像素点（归一化的点）
 * @param  rectification_matrix R矩阵
 * @param  new_intrinsics 矫正后的内参矩阵
 */
void ImageProcessor::undistortPoints(
    const std::vector<cv::Point2f> &pts_in, const cv::Vec4d &intrinsics, const std::string &distortion_model,
    const cv::Vec4d &distortion_coeffs, std::vector<cv::Point2f> &pts_out, const cv::Matx33d &rectification_matrix,
    const cv::Vec4d &new_intrinsics)
{

    if (pts_in.size() == 0)
        return;

    const cv::Matx33d K(
        intrinsics[0], 0.0, intrinsics[2],
        0.0, intrinsics[1], intrinsics[3],
        0.0, 0.0, 1.0);

    const cv::Matx33d K_new(
        new_intrinsics[0], 0.0, new_intrinsics[2],
        0.0, new_intrinsics[1], new_intrinsics[3],
        0.0, 0.0, 1.0);

    // undistortPoints 计算流程
    /**
     * 输入 uv fx fy cx cy 新的内参矩阵（无畸变内参） fx' fy' cx' cy'  输出为无畸变的像素点u' v'
     * x" = (u - cx)/fx     y" = (v - cy)/fy
     * (x', y') = undistort(x", y", dist_coeffs)
     * [X, Y, W]^T = R*[x', y', 1]^T
     * x = X/W   y = Y/W
     * u' = x * fx' + cx'
     * v' = y * fy' + cy'
     * 有可能新内参矩阵是3*4的P投影矩阵
     * 其中P矩阵[K, t] t为3*1向量，其值等于K*[基线长, 0, 0]^T
     */
    if (distortion_model == "radtan")
    {
        // 步骤为 将pts_in根据K反投到归一化平面上，在使用distortion_coeffs去畸变
        // 然后使用rectification_matrix转一下，在这里可以理解成将相机1归一化坐标系啊下的点转到了相机2归一化下
        // 最后使用K_new转到图像平面上，实现去畸变+转坐标，如果K_new为单位矩阵，那么最后的结果就是归一化坐标
        cv::undistortPoints(pts_in, pts_out, K, distortion_coeffs, rectification_matrix, K_new);
    }
    else if (distortion_model == "equidistant")
    {
        cv::fisheye::undistortPoints(pts_in, pts_out, K, distortion_coeffs, rectification_matrix, K_new);
    }
    else
    {
        // LOG(INFO) << "The model " << distortion_model << "is unrecognized, use radtan instead...";
        cv::undistortPoints(pts_in, pts_out, K, distortion_coeffs, rectification_matrix, K_new);
    }

    return;
}

/**
 * @brief 根据单个图像点的像素坐标、相机内参矩阵、畸变模型和系数，得到去畸变的坐标
 * @param  pt_in 输入像素点
 * @param  intrinsics fx, fy, cx, cy
 * @param  distortion_model 相机畸变模型
 * @param  distortion_coeffs 畸变系数
 * @param  pt_out 输出矫正后的像素点（归一化的点）
 * @param  rectification_matrix R矩阵
 * @param  new_intrinsics 矫正后的内参矩阵
 */
void ImageProcessor::undistortPoint(
    const cv::Point2f &pt_in, cv::Point2f &pt_out, const cv::Vec4d &intrinsics, const std::string &distortion_model,
    const cv::Vec4d &distortion_coeffs, const cv::Matx33d &rectification_matrix,
    const cv::Vec4d &new_intrinsics)
{
    std::vector<cv::Point2f> pts_in_vec(1, pt_in);
    std::vector<cv::Point2f> pts_out_vec(1);

    undistortPoints(pts_in_vec, intrinsics, distortion_model, distortion_coeffs, pts_out_vec, rectification_matrix, new_intrinsics);

    pt_out = pts_out_vec[0];
}


int ImageProcessor::n_id = 0;

bool ImageProcessor::inBorder(const cv::Point2f &pt)
{
    const int BORDER_SIZE = 1;
    int img_x = cvRound(pt.x);
    int img_y = cvRound(pt.y);
    return BORDER_SIZE <= img_x && img_x < COL - BORDER_SIZE && BORDER_SIZE <= img_y && img_y < ROW - BORDER_SIZE;
}

// 根据状态位，进行“瘦身”
void ImageProcessor::reduceVector(std::vector<cv::Point2f> &v, std::vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

void ImageProcessor::reduceVector(std::vector<int> &v, std::vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}


// 给现有的特征点设置mask，目的为了特征点的均匀化
void ImageProcessor::setMask()
{
    if(FISHEYE)
        mask = fisheye_mask.clone();
    else
        mask = cv::Mat(ROW, COL, CV_8UC1, cv::Scalar(255));
    

    // prefer to keep features that are tracked for long time
    std::vector<std::pair<int, std::pair<cv::Point2f, int>>> cnt_pts_id;

    for (unsigned int i = 0; i < forw_pts.size(); i++)
        cnt_pts_id.push_back(std::make_pair(track_cnt[i], std::make_pair(forw_pts[i], ids[i])));
    // 利用光流特点，追踪多的稳定性好，排前面
    sort(cnt_pts_id.begin(), cnt_pts_id.end(), [](
        const std::pair<int, std::pair<cv::Point2f, int>> &a, const std::pair<int, std::pair<cv::Point2f, int>> &b)
        {
        return a.first > b.first;
        });

    forw_pts.clear();
    ids.clear();
    track_cnt.clear();

    for (auto &it : cnt_pts_id)
    {
        if (mask.at<uchar>(it.second.first) == 255)
        {
            // 把挑选剩下的特征点重新放进容器
            forw_pts.push_back(it.second.first);
            ids.push_back(it.second.second);
            track_cnt.push_back(it.first);
            // opencv函数，把周围一个圆内全部置0,这个区域不允许别的特征点存在，避免特征点过于集中
            cv::circle(mask, it.second.first, MIN_DIST, 0, -1);
        }
    }
}

// 把新的点加入容器，id给-1作为区分
void ImageProcessor::addPoints()
{
    for (auto &p : n_pts)
    {
        forw_pts.push_back(p);
        ids.push_back(-1);
        track_cnt.push_back(1);
    }
}

/**
 * @brief 
 * 
 * @param[in] _img 输入图像
 * @param[in] _cur_time 图像的时间戳
 * 1、图像均衡化预处理
 * 2、光流追踪
 * 3、提取新的特征点（如果发布）
 * 4、所有特征点去畸变，计算速度
 */
void ImageProcessor::readImage(const cv::Mat &_img, double _cur_time)
{
    cv::Mat img;
    cur_time = _cur_time;

    if (EQUALIZE)
    {
        // 图像太暗或者太亮，提特征点比较难，所以均衡化一下
        // ! opencv 函数看一下
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        clahe->apply(_img, img);
    }
    else
        img = _img;

    // 这里forw表示当前，cur表示上一帧
    if (forw_img.empty())   // 第一次输入图像，prev_img这个没用
    {
        prev_img = cur_img = forw_img = img;
    }
    else
    {
        forw_img = img;
    }

    forw_pts.clear();

    if (cur_pts.size() > 0) // 上一帧有特征点，就可以进行光流追踪了
    {
        std::vector<uchar> status;
        std::vector<float> err;
        // 调用opencv函数进行光流追踪
        // Step 1 通过opencv光流追踪给的状态位剔除outlier
        cv::calcOpticalFlowPyrLK(cur_img, forw_img, cur_pts, forw_pts, status, err, cv::Size(21, 21), 3);

        for (int i = 0; i < int(forw_pts.size()); i++)
            // Step 2 通过图像边界剔除outlier
            if (status[i] && !inBorder(forw_pts[i]))    // 追踪状态好检查在不在图像范围
                status[i] = 0;
        reduceVector(prev_pts, status); // 没用到
        reduceVector(cur_pts, status);
        reduceVector(forw_pts, status);
        reduceVector(ids, status);  // 特征点的id
        reduceVector(cur_un_pts, status);   // 去畸变后的坐标
        reduceVector(track_cnt, status);    // 追踪次数
    }
    // 被追踪到的是上一帧就存在的，因此追踪数+1
    for (auto &n : track_cnt)
        n++;

    // Step 3 通过对级约束来剔除outlier
    rejectWithF();
    setMask();
    int n_max_cnt = MAX_CNT - static_cast<int>(forw_pts.size());
    if (n_max_cnt > 0)
    {
        if(mask.empty())
            std::cout << "mask is empty " << std::endl;
        if (mask.type() != CV_8UC1)
            std::cout << "mask type wrong " << std::endl;
        if (mask.size() != forw_img.size())
            std::cout << "wrong size " << std::endl;
        // 只有发布才可以提取更多特征点，同时避免提的点进mask
        // 会不会这些点集中？会，不过没关系，他们下一次作为老将就得接受均匀化的洗礼
        cv::goodFeaturesToTrack(forw_img, n_pts, MAX_CNT - forw_pts.size(), 0.01, MIN_DIST, mask);
    }
    else
        n_pts.clear();

    addPoints();

    prev_img = cur_img;
    prev_pts = cur_pts;
    prev_un_pts = cur_un_pts;   // 以上三个量无用
    cur_img = forw_img; // 实际上是上一帧的图像
    cur_pts = forw_pts; // 上一帧的特征点
    undistortedPoints();
    prev_time = cur_time;

    // 更新id
    for (unsigned int i = 0;; i++)
    {
        bool completed = false;
        completed = updateID(i);    // 单目的情况下可以直接用=号
        if (!completed)
            break;
    }

    FeatureData feature_data;
    feature_data.time_ = cur_time;
    feature_data.features_.reserve(cur_un_pts.size());
    for (int i = 0; i < cur_un_pts.size(); ++i)
    {
        if (track_cnt[i] < 2)
            continue;
        FeaturePoint feature_point;
        feature_point.id_ = ids[i];
        feature_point.point_.x() = cur_un_pts[i].x;
        feature_point.point_.y() = cur_un_pts[i].y;
        feature_data.features_.push_back(feature_point);
    }
    data_manager_ptr_->Input(feature_data);

    cv::Mat stereo_img = _img.clone();
    cv::cvtColor(stereo_img, stereo_img, CV_GRAY2RGB);
    for (unsigned int j = 0; j < cur_pts.size(); j++)
    {
        double len = std::min(1.0, 1.0 * track_cnt[j] / 10);
        cv::circle(stereo_img, cur_pts[j], 2, cv::Scalar(255 * (1 - len), 0, 255 * len), 2);
        //draw speed line
        /*
        Vector2d tmp_cur_un_pts (cur_un_pts[j].x, cur_un_pts[j].y);
        Vector2d tmp_pts_velocity (pts_velocity[j].x, pts_velocity[j].y);
        Vector3d tmp_prev_un_pts;
        tmp_prev_un_pts.head(2) = tmp_cur_un_pts - 0.10 * tmp_pts_velocity;
        tmp_prev_un_pts.z() = 1;
        Vector2d tmp_prev_uv;
        m_camera->spaceToPlane(tmp_prev_un_pts, tmp_prev_uv);
        cv::line(stereo_img, cur_pts[j], cv::Point2f(tmp_prev_uv.x(), tmp_prev_uv.y()), cv::Scalar(255 , 0, 0), 1 , 8, 0);
        */
        //char name[10];
        //sprintf(name, "%d", ids[j]);
        //cv::putText(stereo_img, name, cur_pts[j], cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    }
    cv::imshow("a", stereo_img);
    cv::waitKey(1);
}

/**
 * @brief 
 * 
 */
void ImageProcessor::rejectWithF()
{
    // 当前被追踪到的光流至少8个点
    if (forw_pts.size() >= 8)
    {
        std::vector<cv::Point2f> un_cur_pts(cur_pts.size()), un_forw_pts(forw_pts.size());
        undistortPoints(
                cur_pts, cam0_intrinsics, cam0_distortion_model,
                cam0_distortion_coeffs, un_cur_pts, cv::Matx33d::eye(), cv::Vec4d(460, 460, 0, 0));
        undistortPoints(
                forw_pts, cam0_intrinsics, cam0_distortion_model,
                cam0_distortion_coeffs, un_forw_pts, cv::Matx33d::eye(), cv::Vec4d(460, 460, 0, 0));

        std::vector<uchar> status;
        // opencv接口计算本质矩阵，某种意义也是一种对级约束的outlier剔除
        cv::findFundamentalMat(un_cur_pts, un_forw_pts, cv::FM_RANSAC, F_THRESHOLD, 0.99, status);
        int size_a = cur_pts.size();
        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        reduceVector(forw_pts, status);
        reduceVector(cur_un_pts, status);
        reduceVector(ids, status);
        reduceVector(track_cnt, status);
    }
}

/**
 * @brief 
 * 
 * @param[in] i 
 * @return true 
 * @return false 
 *  给新的特征点赋上id,越界就返回false
 */
bool ImageProcessor::updateID(unsigned int i)
{
    if (i < ids.size())
    {
        if (ids[i] == -1)
            ids[i] = n_id++;
        return true;
    }
    else
        return false;
}


// void ImageProcessor::showUndistortion(const string &name)
// {
//     cv::Mat undistortedImg(ROW + 600, COL + 600, CV_8UC1, cv::Scalar(0));
//     vector<Eigen::Vector2d> distortedp, undistortedp;
//     for (int i = 0; i < COL; i++)
//         for (int j = 0; j < ROW; j++)
//         {
//             Eigen::Vector2d a(i, j);
//             Eigen::Vector3d b;
//             m_camera->liftProjective(a, b);
//             distortedp.push_back(a);
//             undistortedp.push_back(Eigen::Vector2d(b.x() / b.z(), b.y() / b.z()));
//             //printf("%f,%f->%f,%f,%f\n)\n", a.x(), a.y(), b.x(), b.y(), b.z());
//         }
//     for (int i = 0; i < int(undistortedp.size()); i++)
//     {
//         cv::Mat pp(3, 1, CV_32FC1);
//         pp.at<float>(0, 0) = undistortedp[i].x() * FOCAL_LENGTH + COL / 2;
//         pp.at<float>(1, 0) = undistortedp[i].y() * FOCAL_LENGTH + ROW / 2;
//         pp.at<float>(2, 0) = 1.0;
//         //cout << trackerData[0].K << endl;
//         //printf("%lf %lf\n", p.at<float>(1, 0), p.at<float>(0, 0));
//         //printf("%lf %lf\n", pp.at<float>(1, 0), pp.at<float>(0, 0));
//         if (pp.at<float>(1, 0) + 300 >= 0 && pp.at<float>(1, 0) + 300 < ROW + 600 && pp.at<float>(0, 0) + 300 >= 0 && pp.at<float>(0, 0) + 300 < COL + 600)
//         {
//             undistortedImg.at<uchar>(pp.at<float>(1, 0) + 300, pp.at<float>(0, 0) + 300) = cur_img.at<uchar>(distortedp[i].y(), distortedp[i].x());
//         }
//         else
//         {
//             //ROS_ERROR("(%f %f) -> (%f %f)", distortedp[i].y, distortedp[i].x, pp.at<float>(1, 0), pp.at<float>(0, 0));
//         }
//     }
//     cv::imshow(name, undistortedImg);
//     cv::waitKey(0);
// }

// 当前帧所有点统一去畸变，同时计算特征点速度，用来后续时间戳标定
void ImageProcessor::undistortedPoints()
{
    cur_un_pts.clear();
    cur_un_pts_map.clear();
    //cv::undistortPoints(cur_pts, un_pts, K, cv::Mat());
    for (unsigned int i = 0; i < cur_pts.size(); i++)
    {
        cv::Point2f un_cur_pt;
        undistortPoint(
            cur_pts[i], un_cur_pt, cam0_intrinsics, cam0_distortion_model,
            cam0_distortion_coeffs);
        cur_un_pts.push_back(un_cur_pt);
        // id->坐标的map
        cur_un_pts_map.insert(std::make_pair(ids[i], un_cur_pt));
        //printf("cur pts id %d %f %f", ids[i], cur_un_pts[i].x, cur_un_pts[i].y);
    }
    // caculate points velocity
    if (!prev_un_pts_map.empty())
    {
        double dt = cur_time - prev_time;
        pts_velocity.clear();
        for (unsigned int i = 0; i < cur_un_pts.size(); i++)
        {
            if (ids[i] != -1)
            {
                std::map<int, cv::Point2f>::iterator it;
                it = prev_un_pts_map.find(ids[i]);
                // 找到同一个特征点
                if (it != prev_un_pts_map.end())
                {
                    double v_x = (cur_un_pts[i].x - it->second.x) / dt;
                    double v_y = (cur_un_pts[i].y - it->second.y) / dt;
                    // 得到在归一化平面的速度
                    pts_velocity.push_back(cv::Point2f(v_x, v_y));
                }
                else
                    pts_velocity.push_back(cv::Point2f(0, 0));
            }
            else
            {
                pts_velocity.push_back(cv::Point2f(0, 0));
            }
        }
    }
    else
    {
        // 第一帧的情况
        for (unsigned int i = 0; i < cur_pts.size(); i++)
        {
            pts_velocity.push_back(cv::Point2f(0, 0));
        }
    }
    prev_un_pts_map = cur_un_pts_map;
}
