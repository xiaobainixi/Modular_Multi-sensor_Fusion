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
    : is_first_img(true), // img_transport(n),
      prev_features_ptr(new GridFeatures()),
      curr_features_ptr(new GridFeatures())
{
    param_ptr_ = param_ptr;
    data_manager_ptr_ = data_manager_ptr;
    state_manager_ptr_ = state_manager_ptr;

    if (!param_ptr_ || !data_manager_ptr_ || !state_manager_ptr_)
        return;

    loadParameters();
    detector_ptr = cv::FastFeatureDetector::create(processor_config.fast_threshold);

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
        ProcessImage(camera_data.image_, camera_data.time_);
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
    // nh.param<string>("cam0/distortion_model", cam0_distortion_model, string("radtan"));
    // nh.param<string>("cam1/distortion_model", cam1_distortion_model, string("radtan"));

    // // 3. 左右目内参
    // vector<double> cam0_intrinsics_temp(4);
    // nh.getParam("cam0/intrinsics", cam0_intrinsics_temp);
    cam0_intrinsics[0] = param_ptr_->cam_intrinsics_[0];
    cam0_intrinsics[1] = param_ptr_->cam_intrinsics_[1];
    cam0_intrinsics[2] = param_ptr_->cam_intrinsics_[2];
    cam0_intrinsics[3] = param_ptr_->cam_intrinsics_[3];

    // vector<double> cam1_intrinsics_temp(4);
    // nh.getParam("cam1/intrinsics", cam1_intrinsics_temp);
    // cam1_intrinsics[0] = cam1_intrinsics_temp[0];
    // cam1_intrinsics[1] = cam1_intrinsics_temp[1];
    // cam1_intrinsics[2] = cam1_intrinsics_temp[2];
    // cam1_intrinsics[3] = cam1_intrinsics_temp[3];

    // // 4. 左右目畸变
    // vector<double> cam0_distortion_coeffs_temp(4);
    // nh.getParam("cam0/distortion_coeffs",
    //             cam0_distortion_coeffs_temp);
    cam0_distortion_coeffs = param_ptr_->cam_distortion_coeffs_;
    // cam0_distortion_coeffs[0] = cam0_distortion_coeffs_temp[0];
    // cam0_distortion_coeffs[1] = cam0_distortion_coeffs_temp[1];
    // cam0_distortion_coeffs[2] = cam0_distortion_coeffs_temp[2];
    // cam0_distortion_coeffs[3] = cam0_distortion_coeffs_temp[3];

    // vector<double> cam1_distortion_coeffs_temp(4);
    // nh.getParam("cam1/distortion_coeffs",
    //             cam1_distortion_coeffs_temp);
    // cam1_distortion_coeffs[0] = cam1_distortion_coeffs_temp[0];
    // cam1_distortion_coeffs[1] = cam1_distortion_coeffs_temp[1];
    // cam1_distortion_coeffs[2] = cam1_distortion_coeffs_temp[2];
    // cam1_distortion_coeffs[3] = cam1_distortion_coeffs_temp[3];

    // // 5. 左右目外参
    // // 这里面的T_imu_cam0 表示 imu 到 cam0 的外参
    // cv::Mat T_imu_cam0 = utils::getTransformCV(nh, "cam0/T_cam_imu");
    // cv::Matx33d R_imu_cam0(T_imu_cam0(cv::Rect(0, 0, 3, 3)));
    // cv::Vec3d t_imu_cam0 = T_imu_cam0(cv::Rect(3, 0, 1, 3));
    // R_cam0_imu = R_imu_cam0.t();
    // t_cam0_imu = -R_imu_cam0.t() * t_imu_cam0;

    // cv::Mat T_cam0_cam1 = utils::getTransformCV(nh, "cam1/T_cn_cnm1");
    // cv::Mat T_imu_cam1 = T_cam0_cam1 * T_imu_cam0;

    // 剩下的都是显示了
    // ROS_INFO("===========================================");
    // ROS_INFO("cam0_intrinscs: %f, %f, %f, %f",
    //          cam0_intrinsics[0], cam0_intrinsics[1],
    //          cam0_intrinsics[2], cam0_intrinsics[3]);
    // ROS_INFO("cam0_distortion_model: %s",
    //          cam0_distortion_model.c_str());
    // ROS_INFO("cam0_distortion_coefficients: %f, %f, %f, %f",
    //          cam0_distortion_coeffs[0], cam0_distortion_coeffs[1],
    //          cam0_distortion_coeffs[2], cam0_distortion_coeffs[3]);

    // ROS_INFO("cam1_resolution: %d, %d",
    //          cam1_resolution[0], cam1_resolution[1]);
    // ROS_INFO("cam1_intrinscs: %f, %f, %f, %f",
    //          cam1_intrinsics[0], cam1_intrinsics[1],
    //          cam1_intrinsics[2], cam1_intrinsics[3]);
    // ROS_INFO("cam1_distortion_model: %s",
    //          cam1_distortion_model.c_str());
    // ROS_INFO("cam1_distortion_coefficients: %f, %f, %f, %f",
    //          cam1_distortion_coeffs[0], cam1_distortion_coeffs[1],
    //          cam1_distortion_coeffs[2], cam1_distortion_coeffs[3]);

    // std::cout << R_imu_cam0 << std::endl;
    // std::cout << t_imu_cam0.t() << std::endl;

    // ROS_INFO("grid_row: %d",
    //          processor_config.grid_row);
    // ROS_INFO("grid_col: %d",
    //          processor_config.grid_col);
    // ROS_INFO("grid_min_feature_num: %d",
    //          processor_config.grid_min_feature_num);
    // ROS_INFO("grid_max_feature_num: %d",
    //          processor_config.grid_max_feature_num);
    // ROS_INFO("pyramid_levels: %d",
    //          processor_config.pyramid_levels);
    // ROS_INFO("patch_size: %d",
    //          processor_config.patch_size);
    // ROS_INFO("fast_threshold: %d",
    //          processor_config.fast_threshold);
    // ROS_INFO("max_iteration: %d",
    //          processor_config.max_iteration);
    // ROS_INFO("track_precision: %f",
    //          processor_config.track_precision);
    // ROS_INFO("ransac_threshold: %f",
    //          processor_config.ransac_threshold);
    // ROS_INFO("===========================================");
    return true;
}

/**
 * @brief 处理双目图像
 * @param  cam0_img 左图消息
 * @param  cam1_img 右图消息
 */
void ImageProcessor::ProcessImage(const cv::Mat &image, const double &stamp)
{
    // Get the current image.
    if (image.empty() || stamp < 0.0)
        return;
    cam0_curr_img_ptr = std::make_shared<Frame>();
    cam0_curr_img_ptr->image_ = image;
    cam0_curr_img_ptr->stamp_ = stamp;

    // Build the image pyramids once since they're used at multiple places
    // 2. 创建尺度金字塔，其实就是把输入图片根据输入的层数给他弄成多张大小不同的图片
    createImagePyramids();

    // Detect features in the first frame.
    if (is_first_img)
    {
        // ros::Time start_time = ros::Time::now();
        // 3.1 初始化第一批特征：
        // 利用了LKT光流法来寻找两帧间的匹配点
        // 利用了对极约束筛选外点
        // 将特征点划分到不同的图像grid中
        // 同时有数量限制
        initializeFirstFrame();
        // ROS_INFO("Detection time: %f",
        //     (ros::Time::now()-start_time).toSec());
        is_first_img = false;

        // Draw results.
        // start_time = ros::Time::now();
        // 3.2 当有其他节点订阅了debug_stereo_image消息时
        // 将双目图像拼接起来并画出特征点位置，作为消息发出去
        drawFeaturesMono();
        // ROS_INFO("Draw features: %f",
        //     (ros::Time::now()-start_time).toSec());
    }
    else
    {
        // Track the feature in the previous image.
        // ros::Time start_time = ros::Time::now();
        // 4.1 第二帧开始就跟踪了，此时只是跟踪上一帧而已，并没有衍生出新的点
        trackFeatures();
        // ROS_INFO("Tracking time: %f",
        //     (ros::Time::now()-start_time).toSec());

        // Add new features into the current image.
        // start_time = ros::Time::now();
        // 4.2 在左目提取新的特征，通过左右目光流跟踪去外点，向变量添加新的特征
        addNewFeatures();
        // ROS_INFO("Addition time: %f",
        //     (ros::Time::now()-start_time).toSec());

        // Add new features into the current image.
        // start_time = ros::Time::now();
        // 4.3 剔除每个格多余的点
        pruneGridFeatures();
        // ROS_INFO("Prune grid features: %f",
        //     (ros::Time::now()-start_time).toSec());

        // Draw results.
        // start_time = ros::Time::now();
        // 4.4 当有其他节点订阅了debug_stereo_image消息时，将双目图像拼接起来并画出特征点位置，作为消息发送出去
        drawFeaturesMono();
        // ROS_INFO("Draw features: %f",
        //     (ros::Time::now()-start_time).toSec());
    }

    // ros::Time start_time = ros::Time::now();
    // updateFeatureLifetime();
    // ROS_INFO("Statistics: %f",
    //     (ros::Time::now()-start_time).toSec());

    // Publish features in the current image.
    // ros::Time start_time = ros::Time::now();
    // 5. 发布图片特征点跟踪的结果
    publish();
    // ROS_INFO("Publishing: %f",
    //     (ros::Time::now()-start_time).toSec());

    // Update the previous image and previous features.
    // 保存的都是上一帧左目的，只有curr_features_ptr包含右目的点
    cam0_prev_img_ptr = cam0_curr_img_ptr;
    prev_features_ptr = curr_features_ptr;
    std::swap(prev_cam0_pyramid_, curr_cam0_pyramid_);

    // Initialize the current features to empty vectors.
    // curr_features_ptr指向new GridFeatures()，分配网格
    // 每次都是新的
    curr_features_ptr.reset(new GridFeatures());
    for (int code = 0; code < processor_config.grid_row * processor_config.grid_col; ++code)
    {
        (*curr_features_ptr)[code] = std::vector<FeatureMetaData>(0);
    }

    return;
}

/**
 * @brief 为输入图像分割金字塔层
 * @see ImageProcessor::ProcessImage
 */
void ImageProcessor::createImagePyramids()
{
    const cv::Mat &curr_cam0_img = cam0_curr_img_ptr->image_;
    // 将输入的图像根据参数“缩放图片”，存放至curr_cam0_pyramid_，注意里面存放并非连续
    // 以euroc为例输入图像为752*480，取三层金字塔，curr_cam0_pyramid_的第0个是原图
    // 第2个是长宽缩小1倍的图像 376*240
    // 第4个在第二个的基础上又缩小一倍
    // 第6个也是

    // patch_size表示卷积处理窗口，必须不少于calcOpticalFlowPyrLK的winSize参数。
    // @param withDerivatives 设置为每个金字塔等级预计算梯度。 如果金字塔是在没有梯度的情况下构建的，那么calcOpticalFlowPyrLK将在内部对其进行计算
    // @param pyrBorder 金字塔图层的边框模式
    // @param derivBorder 梯度边框模式
    // @param tryReuseInputImage put ROI of input image into the pyramid if possible. You can pass false
    // to force data copying.
    // @return number of levels in constructed pyramid. Can be less than maxLevel.

    // 1)BORDER_REPLICATE:重复：                    aaaaaa|abcdefgh|hhhhhhh
    // 2)BORDER_REFLECT:反射:                       fedcba|abcdefgh|hgfedcb
    // 3)BORDER_REFLECT_101:反射101:                gfedcb|abcdefgh|gfedcba
    // 4)BORDER_WRAP:外包装：                       cdefgh|abcdefgh|abcdefg
    // 5)BORDER_CONSTANT:常量复制：                 iiiiii|abcdefgh|iiiiiii(i的值由后一个参数Scalar()确定，如Scalar::all(0) )
    // borderValue:若上一参数为BORDER_CONSTANT，则由此参数确定补充上去的像素值。可选用默认值。
    buildOpticalFlowPyramid(
        curr_cam0_img, curr_cam0_pyramid_,
        cv::Size(processor_config.patch_size, processor_config.patch_size),
        processor_config.pyramid_levels, true, cv::BORDER_REFLECT_101,
        cv::BORDER_CONSTANT, false);
}

/**
 * @brief
 * 1. 从cam0的图像中提取FAST特征
 * 2. 利用LKT光流法在cam1的图像中寻找匹配的像素点
 * 3. 利用双目外参构成的对极几何约束进行野点筛选。
 * 4. 然后根据cam0中所有匹配特征点的位置将它们分配到不同的grid中
 * 5. 按提取FAST特征时的response对每个grid中的特征进行排序
 * 6. 最后将它们存储到相应的类成员变量中（每个grid特征数有限制）。
 * @see ImageProcessor::ProcessImage
 */
void ImageProcessor::initializeFirstFrame()
{
    // Size of each grid.
    // 1. 查看分的格的大小
    const cv::Mat &img = cam0_curr_img_ptr->image_;
    static int grid_height = img.rows / processor_config.grid_row;
    static int grid_width = img.cols / processor_config.grid_col;

    // Detect new features on the frist image.
    // 2. 计算原图fast特征点
    std::vector<cv::KeyPoint> new_features(0);
    detector_ptr->detect(img, new_features);

    // Find the stereo matched points for the newly
    // detected features.
    // 格式转为Point2f
    std::vector<cv::Point2f> cam0_points(new_features.size());
    std::vector<float> response_inliers(new_features.size());
    for (int i = 0; i < new_features.size(); ++i)
    {
        cam0_points[i] = new_features[i].pt;
        // 响应程度，代表该点强壮大小，即该点是特征点的程度，用白话说就是值越大这个点看起来更像是角点
        response_inliers[i] = new_features[i].response;
    }

    // Group the features into grids
    // 4. 分格存入

    // 预备好格
    GridFeatures grid_new_features; // typedef std::map<int, std::vector<FeatureMetaData>>
    for (int code = 0; code < processor_config.grid_row * processor_config.grid_col; ++code)
        grid_new_features[code] = std::vector<FeatureMetaData>(0);

    // 便利所有匹配的内点，根据像素坐标存放对应的网格
    for (int i = 0; i < cam0_points.size(); ++i)
    {
        const cv::Point2f &cam0_point = cam0_points[i];
        const float &response = response_inliers[i];

        int row = static_cast<int>(cam0_point.y / grid_height);
        int col = static_cast<int>(cam0_point.x / grid_width);

        // 在第几个格
        int code = row * processor_config.grid_col + col;

        // 存放
        FeatureMetaData new_feature;
        new_feature.response = response;
        new_feature.cam0_point = cam0_point;
        grid_new_features[code].push_back(new_feature);
    }

    // Sort the new features in each grid based on its response.
    // 5. 根据左目提取的点的响应值，每一格按照响应值的从大到小排序
    for (auto &item : grid_new_features)
        std::sort(item.second.begin(), item.second.end(), &ImageProcessor::featureCompareByResponse);

    // Collect new features within each grid with high response.
    // 6. 向已有的curr_features_ptr里面放入新的特征点，同时也不是全部加入，有数量限制
    // 在此之前 curr_features_ptr 是空的，就是 GridFeatures 类型，每次都是新的
    for (int code = 0; code < processor_config.grid_row * processor_config.grid_col; ++code)
    {
        // 获取引用，即使curr_features_ptr里面没有这个值也能获取，相当于直接创建
        std::vector<FeatureMetaData> &features_this_grid = (*curr_features_ptr)[code];
        std::vector<FeatureMetaData> &new_features_this_grid = grid_new_features[code];

        // 向 features_this_grid 放数据， 可以理解成 grid_new_features 是一个临时的，没有数量限制的
        // 这步就是根据相应排好之后按照数量存放
        for (int k = 0; k < processor_config.grid_min_feature_num && k < new_features_this_grid.size(); ++k)
        {
            features_this_grid.push_back(new_features_this_grid[k]);
            features_this_grid.back().id = next_feature_id++;
            features_this_grid.back().lifetime = 1;
        }
    }

    return;
}

/**
 * @brief 利用输入的前一帧特征点图像坐标、前一帧到当前帧的旋转矩阵以及相机内参，预测当前帧中的特征点图像坐标。
 * 作用是给LKT光流一个initial guess。
 * @param  input_pts 上一帧的像素点
 * @param  Rcp 旋转，左相机的上一帧到当前帧
 * @param  intrinsics 内参
 * @param  compensated_pts 输出预测的点
 * @see ImageProcessor::trackFeatures()
 */
void ImageProcessor::predictFeatureTracking(
    const std::vector<cv::Point2f> &input_pts, const cv::Matx33f &Rcp,
    const cv::Vec4d &intrinsics, std::vector<cv::Point2f> &compensated_pts)
{

    // Return directly if there are no input features.
    // 1. 如果输入为0，清空返回
    if (input_pts.size() == 0)
    {
        compensated_pts.clear();
        return;
    }
    compensated_pts.resize(input_pts.size());

    // Intrinsic matrix.
    cv::Matx33f K(
        intrinsics[0], 0.0, intrinsics[2],
        0.0, intrinsics[1], intrinsics[3],
        0.0, 0.0, 1.0);
    // 2. 投到归一化坐标，旋转，再转回像素坐标
    // 没有去畸变，反正只是帮助预测，又不是定死了的位置，节省计算资源
    cv::Matx33f H = K * Rcp * K.inv();

    for (int i = 0; i < input_pts.size(); ++i)
    {
        cv::Vec3f p1(input_pts[i].x, input_pts[i].y, 1.0f);
        cv::Vec3f p2 = H * p1;
        compensated_pts[i].x = p2[0] / p2[2];
        compensated_pts[i].y = p2[1] / p2[2];
    }

    return;
}

/**
 * @brief
 * 跟踪上一帧，去除外点
 */
void ImageProcessor::trackFeatures()
{
    // Size of each grid.
    // 网格大小，在很下面才会用到
    static int grid_height = cam0_curr_img_ptr->image_.rows / processor_config.grid_row;
    static int grid_width = cam0_curr_img_ptr->image_.cols / processor_config.grid_col;

    // Organize the features in the previous image.
    // 2. 相当于把上一帧所有点的相关信息全取出来
    std::vector<FeatureIDType> prev_ids(0);
    std::vector<int> prev_lifetime(0);
    std::vector<cv::Point2f> prev_cam0_points(0);

    for (const auto &item : *prev_features_ptr)
    {
        for (const auto &prev_feature : item.second)
        {
            prev_ids.push_back(prev_feature.id);
            prev_lifetime.push_back(prev_feature.lifetime);
            prev_cam0_points.push_back(prev_feature.cam0_point);
        }
    }

    // Number of the features before tracking.
    // 跟踪前上一帧的点数
    before_tracking = prev_cam0_points.size();

    // Abort tracking if there is no features in
    // the previous frame.
    // 上一帧没有点，return
    if (prev_ids.size() == 0)
        return;

    // Track features using LK optical flow method.
    // 3. 利用输入的前一帧特征点图像坐标、前一帧到当前帧的旋转矩阵以及相机内参，预测当前帧中的特征点图像坐标
    std::vector<cv::Point2f> curr_cam0_points(0);
    std::vector<unsigned char> track_inliers(0);
    // todo
    // predictFeatureTracking(prev_cam0_points, cam0_R_p_c, cam0_intrinsics, curr_cam0_points);
    curr_cam0_points = prev_cam0_points;

    // 4. 光流跟踪，都是在未去畸变的点上做的
    cv::calcOpticalFlowPyrLK(
        prev_cam0_pyramid_, curr_cam0_pyramid_, // 左相机上一帧与当前帧的图片（多层金字塔）
        prev_cam0_points, curr_cam0_points,     // 前后特征点
        track_inliers, cv::noArray(),
        cv::Size(processor_config.patch_size, processor_config.patch_size),
        processor_config.pyramid_levels,
        cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS,
                         processor_config.max_iteration,
                         processor_config.track_precision),
        cv::OPTFLOW_USE_INITIAL_FLOW); // OPTFLOW_USE_INITIAL_FLOW使用存储在curr_cam0_points中的初始估计值

    // Mark those tracked points out of the image region
    // as untracked.
    // 去除边缘以外的点
    for (int i = 0; i < curr_cam0_points.size(); ++i)
    {
        if (track_inliers[i] == 0)
            continue;
        if (curr_cam0_points[i].y < 0 ||
            curr_cam0_points[i].y > cam0_curr_img_ptr->image_.rows - 1 ||
            curr_cam0_points[i].x < 0 ||
            curr_cam0_points[i].x > cam0_curr_img_ptr->image_.cols - 1)
            track_inliers[i] = 0;
    }

    // Collect the tracked points.
    std::vector<FeatureIDType> prev_tracked_ids(0);
    std::vector<int> prev_tracked_lifetime(0);
    std::vector<cv::Point2f> prev_tracked_cam0_points(0);
    std::vector<cv::Point2f> prev_tracked_cam1_points(0);
    std::vector<cv::Point2f> curr_tracked_cam0_points(0);

    // 5. 删除外点，且剩下的依然是一一对应的点
    removeUnmarkedElements(prev_ids, track_inliers, prev_tracked_ids);
    removeUnmarkedElements(prev_lifetime, track_inliers, prev_tracked_lifetime);
    removeUnmarkedElements(prev_cam0_points, track_inliers, prev_tracked_cam0_points);
    removeUnmarkedElements(curr_cam0_points, track_inliers, curr_tracked_cam0_points);

    // Number of features left after tracking.
    // 跟踪后当前帧点的数量
    after_tracking = curr_tracked_cam0_points.size();

    // Step 2 and 3: RANSAC on temporal image pairs of cam0 and cam1.
    std::vector<int> cam0_ransac_inliers(0);

    // todo
    // twoPointRansac(prev_tracked_cam0_points, curr_tracked_cam0_points,
    //                cam0_R_p_c, cam0_intrinsics, cam0_distortion_model,
    //                cam0_distortion_coeffs, processor_config.ransac_threshold,
    //                0.99, cam0_ransac_inliers);
    cam0_ransac_inliers = std::vector<int>(prev_tracked_cam0_points.size(), 1);

    // Number of features after ransac.
    after_ransac = 0;
    // 7. 如果有一个是外点，那就判定为外点
    for (int i = 0; i < cam0_ransac_inliers.size(); ++i)
    {
        if (cam0_ransac_inliers[i] == 0)
            continue;
        // 确定当前帧的内点根据坐标确定在哪个格后分格存入，供下一帧使用
        int row = static_cast<int>(curr_tracked_cam0_points[i].y / grid_height);
        int col = static_cast<int>(curr_tracked_cam0_points[i].x / grid_width);
        int code = row * processor_config.grid_col + col;
        (*curr_features_ptr)[code].push_back(FeatureMetaData());

        FeatureMetaData &grid_new_feature = (*curr_features_ptr)[code].back();
        grid_new_feature.id = prev_tracked_ids[i];
        grid_new_feature.lifetime = ++prev_tracked_lifetime[i];
        grid_new_feature.cam0_point = curr_tracked_cam0_points[i];

        ++after_ransac;
    }

    // 8. 后面是统计了
    // Compute the tracking rate.
    // int prev_feature_num = 0;
    // for (const auto &item : *prev_features_ptr)
    //     prev_feature_num += item.second.size();

    // int curr_feature_num = 0;
    // for (const auto &item : *curr_features_ptr)
    //     curr_feature_num += item.second.size();

    // ROS_INFO_THROTTLE(
    //     0.5, "\033[0;32m candidates: %d; track: %d; match: %d; ransac: %d/%d=%f\033[0m",
    //     before_tracking, after_tracking, after_matching,
    //     curr_feature_num, prev_feature_num,
    //     static_cast<double>(curr_feature_num) /
    //         (static_cast<double>(prev_feature_num) + 1e-5));
    // printf(
    //     "\033[0;32m candidates: %d; raw track: %d; stereo match: %d; ransac: %d/%d=%f\033[0m\n",
    //     before_tracking, after_tracking, after_matching,
    //     curr_feature_num, prev_feature_num,
    //     static_cast<double>(curr_feature_num)/
    //     (static_cast<double>(prev_feature_num)+1e-5));

    return;
}

/**
 * @brief 在左目提取新的特征，通过左右目光流跟踪去外点，向变量添加新的特征
 */
void ImageProcessor::addNewFeatures()
{
    // 取出当前左目图片
    const cv::Mat &curr_img = cam0_curr_img_ptr->image_;

    // Size of each grid.
    static int grid_height = cam0_curr_img_ptr->image_.rows / processor_config.grid_row;
    static int grid_width = cam0_curr_img_ptr->image_.cols / processor_config.grid_col;

    // Create a mask to avoid redetecting existing features.
    cv::Mat mask(curr_img.rows, curr_img.cols, CV_8U, cv::Scalar(1));
    // 1. 已经有特征的区域在mask上填充0，一定范围内，避免重复提取
    // 便利所有跟踪成功的点
    for (const auto &features : *curr_features_ptr)
    {
        for (const auto &feature : features.second)
        {
            const int y = static_cast<int>(feature.cam0_point.y);
            const int x = static_cast<int>(feature.cam0_point.x);

            // 划片，就是这个点附近都不能再有点了
            int up_lim = y - 2, bottom_lim = y + 3,
                left_lim = x - 2, right_lim = x + 3;
            if (up_lim < 0)
                up_lim = 0;
            if (bottom_lim > curr_img.rows)
                bottom_lim = curr_img.rows;
            if (left_lim < 0)
                left_lim = 0;
            if (right_lim > curr_img.cols)
                right_lim = curr_img.cols;

            cv::Range row_range(up_lim, bottom_lim);
            cv::Range col_range(left_lim, right_lim);
            mask(row_range, col_range) = 0;
        }
    }

    // Detect new features.
    std::vector<cv::KeyPoint> new_features(0);
    // 2. 提取新的fast特征点
    detector_ptr->detect(curr_img, new_features, mask);

    // Collect the new detected features based on the grid.
    // Select the ones with top response within each grid afterwards.
    // 3. 新提取的特征点按照网格存入计算数量
    std::vector<std::vector<cv::KeyPoint>> new_feature_sieve(processor_config.grid_row * processor_config.grid_col);
    for (const auto &feature : new_features)
    {
        int row = static_cast<int>(feature.pt.y / grid_height);
        int col = static_cast<int>(feature.pt.x / grid_width);
        new_feature_sieve[row * processor_config.grid_col + col].push_back(feature);
    }

    // 3.1 分格主要用于限制每格内点数，限制完之后重新放入new_features
    new_features.clear();
    for (auto &item : new_feature_sieve)
    {
        // 删除超过数量的点
        if (item.size() > processor_config.grid_max_feature_num)
        {
            std::sort(item.begin(), item.end(), &ImageProcessor::keyPointCompareByResponse);
            item.erase(item.begin() + processor_config.grid_max_feature_num, item.end());
        }
        new_features.insert(new_features.end(), item.begin(), item.end());
    }

    int detected_new_features = new_features.size();
    // 4. 左右目匹配，步骤跟初始化基本一致
    // Find the stereo matched points for the newly
    // detected features.
    // 转成cv::Point2f

    std::vector<float> response_inliers(0);
    std::vector<cv::Point2f> cam0_points(new_features.size());
    for (int i = 0; i < new_features.size(); ++i) {
        cam0_points[i] = new_features[i].pt;
        response_inliers.push_back(new_features[i].response);
    }

    // Group the features into grids
    // 5. 分格存入
    GridFeatures grid_new_features;
    for (int code = 0; code < processor_config.grid_row * processor_config.grid_col; ++code)
        grid_new_features[code] = std::vector<FeatureMetaData>(0);

    for (int i = 0; i < cam0_points.size(); ++i)
    {
        const cv::Point2f &cam0_point = cam0_points[i];
        const float &response = response_inliers[i];

        int row = static_cast<int>(cam0_point.y / grid_height);
        int col = static_cast<int>(cam0_point.x / grid_width);
        int code = row * processor_config.grid_col + col;

        FeatureMetaData new_feature;
        new_feature.response = response;
        new_feature.cam0_point = cam0_point;
        grid_new_features[code].push_back(new_feature);
    }

    // 6. 与本身跟踪的点共同放入curr_features_ptr，同时确保每个格里面的点数不超过设定值
    // Sort the new features in each grid based on its response.
    for (auto &item : grid_new_features)
        std::sort(item.second.begin(), item.second.end(), &ImageProcessor::featureCompareByResponse);

    int new_added_feature_num = 0;
    // Collect new features within each grid with high response.
    // 遍历所有格
    for (int code = 0; code < processor_config.grid_row * processor_config.grid_col; ++code)
    {
        std::vector<FeatureMetaData> &features_this_grid = (*curr_features_ptr)[code];
        std::vector<FeatureMetaData> &new_features_this_grid = grid_new_features[code];

        // 如果这个格内跟踪的点已经够数了，就不要新的点了
        if (features_this_grid.size() >= processor_config.grid_min_feature_num)
            continue;

        // 否则用心点按照响应值大小往里面填
        int vacancy_num = processor_config.grid_min_feature_num - features_this_grid.size();
        for (int k = 0; k < vacancy_num && k < new_features_this_grid.size(); ++k)
        {
            features_this_grid.push_back(new_features_this_grid[k]);
            features_this_grid.back().id = next_feature_id++;
            features_this_grid.back().lifetime = 1;

            ++new_added_feature_num;
        }
    }

    // printf("\033[0;33m detected: %d; matched: %d; new added feature: %d\033[0m\n",
    //     detected_new_features, matched_new_features, new_added_feature_num);

    return;
}

/**
 * @brief 剔除每个格多余的点
 */
void ImageProcessor::pruneGridFeatures()
{
    for (auto &item : *curr_features_ptr)
    {
        auto &grid_features = item.second;
        // Continue if the number of features in this grid does
        // not exceed the upper bound.
        if (grid_features.size() <= processor_config.grid_max_feature_num)
            continue;
        std::sort(grid_features.begin(), grid_features.end(), &ImageProcessor::featureCompareByLifetime);
        grid_features.erase(grid_features.begin() + processor_config.grid_max_feature_num, grid_features.end());
    }
    return;
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
 * @brief 根据归一化平面上的坐标、相机内参矩阵、畸变模型和系数，得到图像上带有畸变的像素坐标
 * @param  pts_in 归一化坐标的前两维
 * @param  intrinsics 内参
 * @param  distortion_model 畸变模型
 * @param  distortion_coeffs 畸变系数
 * @return 像素坐标的点
 */
std::vector<cv::Point2f> ImageProcessor::distortPoints(
    const std::vector<cv::Point2f> &pts_in, const cv::Vec4d &intrinsics, const std::string &distortion_model,
    const cv::Vec4d &distortion_coeffs)
{

    // 1. 内参矩阵
    const cv::Matx33d K(
        intrinsics[0], 0.0, intrinsics[2],
        0.0, intrinsics[1], intrinsics[3],
        0.0, 0.0, 1.0);

    std::vector<cv::Point2f> pts_out;
    // 2. 不同模型的加畸变函数
    if (distortion_model == "radtan")
    {
        std::vector<cv::Point3f> homogenous_pts;
        // 转成齐次
        cv::convertPointsToHomogeneous(pts_in, homogenous_pts);
        cv::projectPoints(homogenous_pts, cv::Vec3d::zeros(), cv::Vec3d::zeros(), K, distortion_coeffs, pts_out);
    }
    else if (distortion_model == "equidistant")
    {
        cv::fisheye::distortPoints(pts_in, pts_out, K, distortion_coeffs);
    }
    else
    {
        LOG(INFO) << "The model " << distortion_model << "is unrecognized, use radtan instead...";
        std::vector<cv::Point3f> homogenous_pts;
        cv::convertPointsToHomogeneous(pts_in, homogenous_pts);
        cv::projectPoints(homogenous_pts, cv::Vec3d::zeros(), cv::Vec3d::zeros(), K, distortion_coeffs, pts_out);
    }

    return pts_out;
}


// TODO
/**
 * @brief 估计系数，用于算像素单位与归一化平面单位的一个乘数
 * @param  pts1 非归一化坐标
 * @param  pts2 归一化坐标
 * @param  scaling_factor 尺度
 */
void ImageProcessor::rescalePoints(std::vector<cv::Point2f> &pts1, std::vector<cv::Point2f> &pts2, float &scaling_factor)
{

    scaling_factor = 0.0f;

    // 求数个sqrt(x^2 + y^2)的和
    for (int i = 0; i < pts1.size(); ++i)
    {
        scaling_factor += sqrt(pts1[i].dot(pts1[i]));
        scaling_factor += sqrt(pts2[i].dot(pts2[i]));
    }
    // 个数/和 × 1.414， 这里面为什么还有个根号2呢，前面得出的尺度系数是在斜边的基础上求的
    // 但实际应用的时候是长宽级别的，所以要除根号2
    scaling_factor = (pts1.size() + pts2.size()) / scaling_factor * sqrt(2.0f);
    // 每个点乘这个数
    for (int i = 0; i < pts1.size(); ++i)
    {
        pts1[i] *= scaling_factor;
        pts2[i] *= scaling_factor;
    }

    return;
}

/**
 * @brief 通过ransac进一步剔除外点
 * @param  pts1 上一帧内的点
 * @param  pts2 当前帧内的点
 * @param  Rcp 上一帧到当前帧的旋转,IMU预测的
 * @param  intrinsics 内参
 * @param  distortion_model 畸变模型
 * @param  distortion_coeffs 畸变参数
 * @param  inlier_error 内点误差，像素，也就是ransac的阈值
 * @param  success_probability ransac参数，结果里这个点是内点的概率
 * @param  inlier_markers 内外点
 * @see ImageProcessor::trackFeatures()
 */
void ImageProcessor::twoPointRansac(
    const std::vector<cv::Point2f> &pts1, const std::vector<cv::Point2f> &pts2, const cv::Matx33f &Rcp,
    const cv::Vec4d &intrinsics,
    const std::string &distortion_model, const cv::Vec4d &distortion_coeffs, const double &inlier_error,
    const double &success_probability, std::vector<int> &inlier_markers)
{

    // Check the size of input point size.
    if (pts1.size() != pts2.size())
        return;

    double norm_pixel_unit = 2.0 / (intrinsics[0] + intrinsics[1]);
    // ransac迭代次数
    // success_probability 表示希望RANSAC得到正确模型的概率
    // 认为内点比率是0.7，每次取两对点
    int iter_num = static_cast<int>(ceil(log(1 - success_probability) / log(1 - 0.7 * 0.7)));

    // Initially, mark all points as inliers.
    inlier_markers.clear();
    inlier_markers.resize(pts1.size(), 1);

    // Undistort all the points.
    std::vector<cv::Point2f> pts1_undistorted(pts1.size());
    std::vector<cv::Point2f> pts2_undistorted(pts2.size());
    // 1. 两批点全部投到矫正后的归一化平面坐标下
    undistortPoints(pts1, intrinsics, distortion_model, distortion_coeffs, pts1_undistorted);
    undistortPoints(pts2, intrinsics, distortion_model, distortion_coeffs, pts2_undistorted);

    // Compenstate the points in the previous image with
    // the relative rotation.
    // 2. 通过旋转将pts1_undistorted点转换到pts2_undistorted的相机坐标下，注意这里没做平移，且没有将第三维弄成1
    for (auto &pt : pts1_undistorted)
    {
        cv::Vec3f pt_h(pt.x, pt.y, 1.0f);
        // Vec3f pt_hc = dR * pt_h;
        cv::Vec3f pt_hc = Rcp * pt_h;
        pt.x = pt_hc[0];
        pt.y = pt_hc[1];
    }

    // Normalize the points to gain numerical stability.
    float scaling_factor = 0.0f;
    // 求出两波点的平均尺度，每个点再做尺度归一化
    rescalePoints(pts1_undistorted, pts2_undistorted, scaling_factor);
    // 焦距倒数乘尺度，阈值与点的坐标尺度保持一致
    norm_pixel_unit *= scaling_factor;

    // Compute the difference between previous and current points,
    // which will be used frequently later.
    // 3. 计算对应的两批点在尺度归一化的坐标差
    std::vector<cv::Point2d> pts_diff(pts1_undistorted.size());
    for (int i = 0; i < pts1_undistorted.size(); ++i)
        pts_diff[i] = pts1_undistorted[i] - pts2_undistorted[i];

    // Mark the point pairs with large difference directly.
    // BTW, the mean distance of the rest of the point pairs
    // are computed.
    // 4. 通过上面的差值剔除一些较大的，因为要做RANSAC，所以尽量的先把外点剔除一部分
    double mean_pt_distance = 0.0;
    int raw_inlier_cntr = 0;
    for (int i = 0; i < pts_diff.size(); ++i)
    {
        double distance = sqrt(pts_diff[i].dot(pts_diff[i]));
        // 25 pixel distance is a pretty large tolerance for normal motion.
        // However, to be used with aggressive motion, this tolerance should
        // be increased significantly to match the usage.
        // 50 表示 50个像素
        if (distance > 50.0 * norm_pixel_unit)
        {
            inlier_markers[i] = 0;
        }
        else
        {
            mean_pt_distance += distance;
            ++raw_inlier_cntr;
        }
    }
    // 计算保留下来的差值的均值
    mean_pt_distance /= raw_inlier_cntr;

    // If the current number of inliers is less than 3, just mark
    // all input as outliers. This case can happen with fast
    // rotation where very few features are tracked.
    // 5. 快速旋转将导致内点很少，返回。
    // 前端没有所谓的是否丢失，最多是没有点跟踪上，导致后端没办法更新而已
    if (raw_inlier_cntr < 3)
    {
        for (auto &marker : inlier_markers)
            marker = 0;
        return;
    }

    // Before doing 2-point RANSAC, we have to check if the motion
    // is degenerated, meaning that there is no translation between
    // the frames, in which case, the model of the RANSAC does not
    // work. If so, the distance between the matched points will
    // be almost 0.
    // if (mean_pt_distance < inlier_error*norm_pixel_unit) {
    // 6. 如果平均差较小， 一个像素 ，就不做ransac了，因为此时平移几乎是0，运动退化，没必要在估计t了，所以不用下面的RANSAC
    // 使用经验值进一步剔除外点
    if (mean_pt_distance < norm_pixel_unit)
    {
        // ROS_WARN_THROTTLE(1.0, "Degenerated motion...");
        for (int i = 0; i < pts_diff.size(); ++i)
        {
            if (inlier_markers[i] == 0)
                continue;
            if (sqrt(pts_diff[i].dot(pts_diff[i])) > inlier_error * norm_pixel_unit)
                inlier_markers[i] = 0;
        }
        return;
    }

    // In the case of general motion, the RANSAC model can be applied.
    // The three column corresponds to tx, ty, and tz respectively.
    // 7. ransac
    Eigen::MatrixXd coeff_t(pts_diff.size(), 3);
    // 7.1 构建方程 y1 - y2     -(x1 - x2)    x1y2 - x2y1
    for (int i = 0; i < pts_diff.size(); ++i)
    {
        coeff_t(i, 0) = pts_diff[i].y;
        coeff_t(i, 1) = -pts_diff[i].x;
        coeff_t(i, 2) = pts1_undistorted[i].x * pts2_undistorted[i].y - pts1_undistorted[i].y * pts2_undistorted[i].x;
    }

    // 7.2 根据剩下的内点来更新模型
    std::vector<int> raw_inlier_idx;
    for (int i = 0; i < inlier_markers.size(); ++i)
    {
        if (inlier_markers[i] != 0)
            raw_inlier_idx.push_back(i);
    }

    std::vector<int> best_inlier_set;
    double best_error = 1e10;

    srand((unsigned)time(NULL));
    for (int iter_idx = 0; iter_idx < iter_num; ++iter_idx)
    {
        // Randomly select two point pairs.
        // Although this is a weird way of selecting two pairs, but it
        // is able to efficiently avoid selecting repetitive pairs.
        // 7.3 选择非重复的两个点
        int select_idx1 = rand() % raw_inlier_idx.size();
        int select_idx_diff = (rand() % (raw_inlier_idx.size() - 1)) + 1;
        // 到这里上面两个数有可能出现重复的
        // 经过下面处理实现获取两个不相等的id
        int select_idx2 =
            select_idx1 + select_idx_diff < raw_inlier_idx.size() ? select_idx1 + select_idx_diff : select_idx1 + select_idx_diff - raw_inlier_idx.size();

        int pair_idx1 = raw_inlier_idx[select_idx1];
        int pair_idx2 = raw_inlier_idx[select_idx2];

        // | y1 - y2, -(x1 - x2), x1y2 - x2y1 |         tx       0
        // |                                  |    *    ty   ~=
        // | y3 - y4, -(x3 - x4), x3y4 - x4y3 |         tz       0

        // Construct the model;
        // coeff_tx = [y1 - y2, y3 - y4]
        // coeff_ty = [-(x1 - x2), -(x3 - x4)]
        // coeff_tz = [x1y2 - x2y1, x3y4 - x4y3]
        Eigen::Vector2d coeff_tx(coeff_t(pair_idx1, 0), coeff_t(pair_idx2, 0));
        Eigen::Vector2d coeff_ty(coeff_t(pair_idx1, 1), coeff_t(pair_idx2, 1));
        Eigen::Vector2d coeff_tz(coeff_t(pair_idx1, 2), coeff_t(pair_idx2, 2));
        std::vector<double> coeff_l1_norm(3);
        // 1范数，所有元素绝对值的合
        coeff_l1_norm[0] = coeff_tx.lpNorm<1>();
        coeff_l1_norm[1] = coeff_ty.lpNorm<1>();
        coeff_l1_norm[2] = coeff_tz.lpNorm<1>();
        // 这个越小说明离0越近
        int base_indicator = min_element(coeff_l1_norm.begin(), coeff_l1_norm.end()) - coeff_l1_norm.begin();

        // 这里想要取其中两个做成方阵，到底选哪两个？
        // 首先不管选哪两个，得出的方阵由于有误差的存在，它是绝对可逆的
        // 所以挑选了数值相对较大的两个，因为在相同误差下，数值大的误差比例小
        // 求出的逆更加“可靠”
        Eigen::Vector3d model(0.0, 0.0, 0.0);
        if (base_indicator == 0)
        {
            Eigen::Matrix2d A;
            A << coeff_ty, coeff_tz;
            Eigen::Vector2d solution = A.inverse() * (-coeff_tx);
            model(0) = 1.0;
            model(1) = solution(0);
            model(2) = solution(1);
        }
        else if (base_indicator == 1)
        {
            Eigen::Matrix2d A;
            A << coeff_tx, coeff_tz;
            Eigen::Vector2d solution = A.inverse() * (-coeff_ty);
            model(0) = solution(0);
            model(1) = 1.0;
            model(2) = solution(1);
        }
        else
        {
            Eigen::Matrix2d A;
            A << coeff_tx, coeff_ty;
            Eigen::Vector2d solution = A.inverse() * (-coeff_tz);
            model(0) = solution(0);
            model(1) = solution(1);
            model(2) = 1.0;
        }

        // 7.4 选出内点
        // Find all the inliers among point pairs.
        Eigen::VectorXd error = coeff_t * model;

        std::vector<int> inlier_set;
        for (int i = 0; i < error.rows(); ++i)
        {
            if (inlier_markers[i] == 0)
                continue;

            // 点越准，这个error越接近0
            if (std::abs(error(i)) < inlier_error * norm_pixel_unit)
                inlier_set.push_back(i);
        }

        // If the number of inliers is small, the current
        // model is probably wrong.
        // 内点数量太少，跳过这组结果，换其他2个点再计算
        if (inlier_set.size() < 0.2 * pts1_undistorted.size())
            continue;

        // 7.5 下面这段类似于上面计算t的过程，只不过使用的内点算得
        // Refit the model using all of the possible inliers.
        Eigen::VectorXd coeff_tx_better(inlier_set.size());
        Eigen::VectorXd coeff_ty_better(inlier_set.size());
        Eigen::VectorXd coeff_tz_better(inlier_set.size());
        for (int i = 0; i < inlier_set.size(); ++i)
        {
            coeff_tx_better(i) = coeff_t(inlier_set[i], 0);
            coeff_ty_better(i) = coeff_t(inlier_set[i], 1);
            coeff_tz_better(i) = coeff_t(inlier_set[i], 2);
        }

        Eigen::Vector3d model_better(0.0, 0.0, 0.0);
        if (base_indicator == 0)
        {
            Eigen::MatrixXd A(inlier_set.size(), 2);
            A << coeff_ty_better, coeff_tz_better;
            // 计算伪逆
            Eigen::Vector2d solution =
                (A.transpose() * A).inverse() * A.transpose() * (-coeff_tx_better);
            model_better(0) = 1.0;
            model_better(1) = solution(0);
            model_better(2) = solution(1);
        }
        else if (base_indicator == 1)
        {
            Eigen::MatrixXd A(inlier_set.size(), 2);
            A << coeff_tx_better, coeff_tz_better;
            Eigen::Vector2d solution =
                (A.transpose() * A).inverse() * A.transpose() * (-coeff_ty_better);
            model_better(0) = solution(0);
            model_better(1) = 1.0;
            model_better(2) = solution(1);
        }
        else
        {
            Eigen::MatrixXd A(inlier_set.size(), 2);
            A << coeff_tx_better, coeff_ty_better;
            Eigen::Vector2d solution =
                (A.transpose() * A).inverse() * A.transpose() * (-coeff_tz_better);
            model_better(0) = solution(0);
            model_better(1) = solution(1);
            model_better(2) = 1.0;
        }

        // Compute the error and upate the best model if possible.
        Eigen::VectorXd new_error = coeff_t * model_better;

        // 统计内点数量与得分
        double this_error = 0.0;
        for (const auto &inlier_idx : inlier_set)
            this_error += std::abs(new_error(inlier_idx));
        this_error /= inlier_set.size();

        if (inlier_set.size() > best_inlier_set.size())
        {
            best_error = this_error;
            best_inlier_set = inlier_set;
        }
    }

    // 8. 迭代很多次之后最后的结果输出
    // Fill in the markers.
    inlier_markers.clear();
    inlier_markers.resize(pts1.size(), 0);
    for (const auto &inlier_idx : best_inlier_set)
        inlier_markers[inlier_idx] = 1;

    // printf("inlier ratio: %lu/%lu\n",
    //     best_inlier_set.size(), inlier_markers.size());

    return;
}

/**
 * @brief 发送消息
 */
void ImageProcessor::publish()
{

    // 1. 取出当前帧所有跟踪上的+新提取的特征点
    std::vector<FeatureIDType> curr_ids(0);
    std::vector<cv::Point2f> curr_cam0_points(0);

    for (const auto &grid_features : (*curr_features_ptr))
    {
        for (const auto &feature : grid_features.second)
        {
            curr_ids.push_back(feature.id);
            curr_cam0_points.push_back(feature.cam0_point);
        }
    }

    std::vector<cv::Point2f> curr_cam0_points_undistorted(0);
    // 2. 去畸变，注意！！！！！这里的输入最后的相机内参矩阵没有，所以输出为归一化的坐标
    undistortPoints(
        curr_cam0_points, cam0_intrinsics, cam0_distortion_model,
        cam0_distortion_coeffs, curr_cam0_points_undistorted);
    // 3. 发送消息，存放的是所有点以及id
    FeatureData feature_data;
    feature_data.time_ = cam0_curr_img_ptr->stamp_;
    feature_data.features_.reserve(curr_ids.size());
    for (int i = 0; i < curr_ids.size(); ++i)
    {
        FeaturePoint feature_point;

        feature_point.id_ = curr_ids[i];
        feature_point.point_.x() = curr_cam0_points_undistorted[i].x;
        feature_point.point_.y() = curr_cam0_points_undistorted[i].y;
        feature_data.features_.push_back(feature_point);
    }
    data_manager_ptr_->Input(feature_data);
}

/**
 * @brief 当有其他节点订阅了debug_stereo_image消息时，将双目图像拼接起来并画出特征点位置，作为消息发送出去
 */
void ImageProcessor::drawFeaturesMono()
{
    // Colors for different features.
    // 跟踪的点是绿色的
    cv::Scalar tracked(0, 255, 0);
    // 新点是金色的
    cv::Scalar new_feature(0, 255, 255);

    // 网格大小
    static int grid_height =
        cam0_curr_img_ptr->image_.rows / processor_config.grid_row;
    static int grid_width =
        cam0_curr_img_ptr->image_.cols / processor_config.grid_col;

    // Create an output image.
    // 1. 把左右目图片拼一块
    int img_height = cam0_curr_img_ptr->image_.rows;
    int img_width = cam0_curr_img_ptr->image_.cols;
    cv::Mat out_img = cam0_curr_img_ptr->image_.clone();
    cv::cvtColor(out_img, out_img, CV_GRAY2RGB);

    // Draw grids on the image.
    // 2. 画格
    for (int i = 1; i < processor_config.grid_row; ++i)
    {
        cv::Point pt1(0, i * grid_height);
        cv::Point pt2(img_width * 2, i * grid_height);
        cv::line(out_img, pt1, pt2, cv::Scalar(255, 0, 0));
    }
    for (int i = 1; i < processor_config.grid_col; ++i)
    {
        cv::Point pt1(i * grid_width, 0);
        cv::Point pt2(i * grid_width, img_height);
        cv::line(out_img, pt1, pt2, cv::Scalar(255, 0, 0));
    }

    // Collect features ids in the previous frame.
    // 3. 取出上一帧特征点id
    std::vector<FeatureIDType> prev_ids(0);
    for (const auto &grid_features : *prev_features_ptr)
        for (const auto &feature : grid_features.second)
            prev_ids.push_back(feature.id);

    // Collect feature points in the previous frame.
    // 4. 取出上一帧特征点id与对应的特征点
    std::map<FeatureIDType, cv::Point2f> prev_cam0_points;
    for (const auto &grid_features : *prev_features_ptr)
        for (const auto &feature : grid_features.second)
        {
            prev_cam0_points[feature.id] = feature.cam0_point;
        }

    // Collect feature points in the current frame.
    // 5. 取出当前帧特征点id与对应的特征点
    std::map<FeatureIDType, cv::Point2f> curr_cam0_points;
    for (const auto &grid_features : *curr_features_ptr)
        for (const auto &feature : grid_features.second)
        {
            curr_cam0_points[feature.id] = feature.cam0_point;
        }

    // Draw tracked features.
    // 6. 画匹配的点，画线，拖尾
    for (const auto &id : prev_ids)
    {
        if (prev_cam0_points.find(id) != prev_cam0_points.end() &&
            curr_cam0_points.find(id) != curr_cam0_points.end())
        {
            cv::Point2f prev_pt0 = prev_cam0_points[id];
            cv::Point2f curr_pt0 = curr_cam0_points[id];

            circle(out_img, curr_pt0, 3, tracked, -1);
            line(out_img, prev_pt0, curr_pt0, tracked, 1);

            prev_cam0_points.erase(id);
            curr_cam0_points.erase(id);
        }
    }

    // Draw new features.
    // 7. 画新添加的点
    for (const auto &new_cam0_point : curr_cam0_points)
    {
        cv::Point2f pt0 = new_cam0_point.second;
        circle(out_img, pt0, 3, new_feature, -1);
    }

    cv::imshow("Feature", out_img);
    cv::waitKey(5);

    return;
}

/**
 * @brief 没用到，计算特征点的生命周期
 */
void ImageProcessor::updateFeatureLifetime()
{
    for (int code = 0; code < processor_config.grid_row * processor_config.grid_col; ++code)
    {
        std::vector<FeatureMetaData> &features = (*curr_features_ptr)[code];
        for (const auto &feature : features)
        {
            if (feature_lifetime.find(feature.id) == feature_lifetime.end())
                feature_lifetime[feature.id] = 1;
            else
                ++feature_lifetime[feature.id];
        }
    }
    return;
}

/**
 * @brief 没用到，输出每个点的生命周期
 */
void ImageProcessor::featureLifetimeStatistics()
{

    std::map<int, int> lifetime_statistics;
    for (const auto &data : feature_lifetime) // std::map<FeatureIDType, int>
    {
        if (lifetime_statistics.find(data.second) == lifetime_statistics.end())
            lifetime_statistics[data.second] = 1;
        else
            ++lifetime_statistics[data.second];
    }

    for (const auto &data : lifetime_statistics)
        std::cout << data.first << " : " << data.second << std::endl;

    return;
}