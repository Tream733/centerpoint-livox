#pragma once
#include <algorithm>
#include <queue>
#include <numeric>
#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "geometry_msgs/msg/pose_array.hpp"
#include "ws_msgs/msg/cluster_array.hpp"
#include "ws_msgs/msg/bbox.hpp"
#include "ws_msgs/msg/bbox_array.hpp"
#include <pcl/filters/passthrough.h>
#include "centerpoint.h"
#include "process.h"
#include "postprocess.h"
#include "yaml-cpp/yaml.h"
#include <deque>
#include <chrono>


using Bbox = ws_msgs::msg::Bbox;
using bboxArray = ws_msgs::msg::BboxArray;
static size_t BoxFeature = 7;
using Clock = std::chrono::high_resolution_clock;
class Detection : public rclcpp::Node
{

public:
    Detection( const std::string& name_space,
        const rclcpp::NodeOptions& options=rclcpp::NodeOptions());

    Detection(const rclcpp::NodeOptions& options=rclcpp::NodeOptions());


private:
    void cloudCallbak(const sensor_msgs::msg::PointCloud2::ConstPtr &input);
    void makeOutput(std::vector<Box> &out_detections,rclcpp::Time& stamp);

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;

    rclcpp::Publisher<ws_msgs::msg::BboxArray>::SharedPtr pub_bbox_array_;

    rclcpp::Clock::SharedPtr clock_;

    std::unique_ptr<CenterPoint> centerpoint_ = nullptr;

    void loadParam(std::string & param_path);

    bboxArray bbox_array_;

    Clock::time_point m_sync_start_time_;

    bool use_onnx_ = false;
    std::string rpn_file_;
    std::string centerpoint_config_;
    std::string file_name_;
    std::string param_path_ = "/home/txy/wspace/src/detection_lidar/centerpoint/cfgs/centerpoint.yaml";

    int pub_count_ = 0;

    int sub_count_ = 0;

};



