#ifndef FEATURE_MANAGER_H
#define FEATURE_MANAGER_H

#include <list>
#include <algorithm>
#include <vector>
#include <numeric>
#include <eigen3/Eigen/Dense>

#include "common/StateManager.h"
#include "viewer/Viewer.h"

class FeaturePerFrame
{
  public:
    FeaturePerFrame(
        const Eigen::Vector3d &point, const std::shared_ptr<State> & state_ptr)
    {
        this->point = point;
        this->state_ptr_ = state_ptr;
    }
    Eigen::Vector3d point;
    double z;
    bool is_used;
    double parallax;
    Eigen::MatrixXd A;
    Eigen::VectorXd b;
    double dep_gradient;

    std::shared_ptr<State> state_ptr_;
};

class FeaturePerId
{
  public:
    const int feature_id;
    int start_frame;
    std::vector<FeaturePerFrame> feature_per_frame;  // 该id对应的特征点在每个帧中的属性

    int used_num;
    bool is_outlier;
    bool is_margin;
    double estimated_depth;
    int solve_flag; // 0 haven't solve yet; 1 solve succ; 2 solve fail;

    Eigen::Vector3d gt_p;

    FeaturePerId(int _feature_id, int _start_frame)
        : feature_id(_feature_id), start_frame(_start_frame),
          used_num(0), estimated_depth(-1.0), solve_flag(0)
    {
    }

    int endFrame();
};

class VinsFeatureManager
{
  public:
    VinsFeatureManager(
        const std::shared_ptr<Parameter> &param_ptr,
        const std::shared_ptr<Viewer> &viewer_ptr = nullptr);


    void clearState();

    int getFeatureCount();

    bool addFeatureCheckParallax(
        int frame_count, const std::vector<FeaturePoint> & features,
        const std::shared_ptr<State> & state_ptr);
    void debugShow();
    std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> getCorresponding(int frame_count_l, int frame_count_r);

    //void updateDepth(const VectorXd &x);
    void setDepth(const Eigen::VectorXd &x);
    void removeFailures();
    void clearDepth(const Eigen::VectorXd &x);
    Eigen::VectorXd getDepthVector();
    // 使用每个帧的 state_ptr_ (IMU位姿) 与外参 Rbc_, tbc_ 进行三角化
    void triangulate();
    void triangulate(
        Eigen::Vector3d Ps[], Eigen::Matrix3d Rs[], Eigen::Vector3d tic[], Eigen::Matrix3d ric[]);
    void removeBackShiftDepth(Eigen::Matrix3d marg_R, Eigen::Vector3d marg_P, Eigen::Matrix3d new_R, Eigen::Vector3d new_P);
    void removeBack();
    void removeFront(int frame_count);
    void removeOutlier();
    std::list<FeaturePerId> feature;
    int last_track_num;

  private:
    double compensatedParallax2(const FeaturePerId &it_per_id, int frame_count);
    // const Eigen::Matrix3d *Rs;
    // Eigen::Matrix3d ric[NUM_OF_CAM];
    std::shared_ptr<Parameter> param_ptr_;
    std::shared_ptr<Viewer> viewer_ptr_;
};

#endif