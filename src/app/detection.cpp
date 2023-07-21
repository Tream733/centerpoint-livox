#include "app/detection.h"
#include "ws_msgs/msg/bbox_array.hpp"
#include <pcl/features/moment_of_inertia_estimation.h>

using namespace std::chrono_literals;

Detection::Detection(
    const rclcpp::NodeOptions& options
    ):Detection("",options) 
{}

Detection::Detection(
    const std::string& name_space,
    const rclcpp::NodeOptions& options
    ):Node("Detection",name_space,options)
{
    RCLCPP_INFO(this->get_logger(),"Clustering init complete");

    // Store clock
    clock_ = this->get_clock();

    loadParam(param_path_);

    // center_point_ 初始化
    centerpoint_.reset(new CenterPoint(use_onnx_,rpn_file_,centerpoint_config_));

    // Create a ROS subscriber for the input cloud
    rclcpp::QoS subscription_qos(1);
    std::function<void(const sensor_msgs::msg::PointCloud2::SharedPtr)> subscription_callback = std::bind(&Detection::cloudCallbak,this, std::placeholders::_1);
    subscription_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        "livox/point_cloud_front",5,subscription_callback
    );
    rclcpp::QoS qos(1);
    qos.best_effort();
    pub_bbox_array_ = this->create_publisher<ws_msgs::msg::BboxArray>("bbox_array",10);
}

void Detection::loadParam(std::string & param_path)
{
    YAML::Node config = YAML::LoadFile(param_path);
    rpn_file_ = config["RpnFile"].as<std::string>();
    centerpoint_config_ = config["ModelConfig"].as<std::string>();
    use_onnx_ = config["UseOnnx"].as<bool>();
}

void Detection::cloudCallbak(const sensor_msgs::msg::PointCloud2::ConstPtr &input){
    std::cout<<"  ======================sub========================   ok     index "<< ++sub_count_ << std::endl;
        
    m_sync_start_time_ = Clock::now();
    RCLCPP_INFO(this->get_logger(),"points_size(%d,%d)",input->height,input->width);
    PointICloudPtr cloud(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::fromROSMsg(*input, *cloud);
    float *points_array;
    int in_num_points = PointCloud2Array(points_array,*cloud);
    std::vector<Box> out_detections;
    out_detections.clear();
    cudaDeviceSynchronize();
    centerpoint_->DoInference(points_array, in_num_points, out_detections);
    cudaDeviceSynchronize();
    rclcpp::Time stamp = input->header.stamp;
    makeOutput(out_detections,stamp);
    double sync_duration_ms = std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - m_sync_start_time_).count() / 1e6;
    std::cout<<" --------------cloudCallbak-------------------------- :   "<<sync_duration_ms<<std::endl;
}

void Detection::makeOutput(std::vector<Box> &out_detections,rclcpp::Time& stamp)
{
    size_t num_objects = out_detections.size();
    int obj_index;
    bbox_array_.header.frame_id = "livox";
    bbox_array_.header.stamp = stamp;
    bbox_array_.boxes.clear();
    for(obj_index = 0; obj_index < num_objects; obj_index++) {
        ws_msgs::msg::Bbox bbox;
        float x = out_detections[obj_index].x;
        float y = out_detections[obj_index].y;
        float z = out_detections[obj_index].z;
        float dx = out_detections[obj_index].l;
        float dy = out_detections[obj_index].w;
        float dz = out_detections[obj_index].h;
        float yaw = out_detections[obj_index].r;
        yaw += M_PI / 2;
        yaw = std::atan2(sinf(yaw),cosf(yaw));
        yaw = - yaw;

        float top_z = z + dz ;
        float bot_Z = z;
        // std::cout<<"-------------------------------------------"<<std::endl;
        // std::cout<<"  x     y    z     dx    dy    dz   :   "<<x<<"  "<<y<<"    "<<z<<"     "<<dx<<"     "<<dy<<"     "<<dz<<std::endl;
        
        float c_s = cos(yaw);
        float s_s = sin(yaw);
        bbox.center.x = x;
        bbox.center.y = y;
        bbox.center.z = z;
        bbox.size.x = dy;
        bbox.size.y = dx;
        bbox.size.z = dz;
        bbox.heading = yaw;
        bbox.type = out_detections[obj_index].label;
        // p1 上右前
        bbox.corner_3d[0].x = dx/2*c_s - dy /2 *s_s + x;
        bbox.corner_3d[0].y = dy /2 * c_s + dx / 2 * s_s + y;
        bbox.corner_3d[0].z = top_z;
        // p2  上左前
        bbox.corner_3d[1].x = (-dx)/2*c_s - dy /2 *s_s + x;
        bbox.corner_3d[1].y = dy /2 * c_s + (-dx) / 2 * s_s + y;
        bbox.corner_3d[1].z = top_z;
        // p3  上左后
        bbox.corner_3d[2].x = (-dx)/2*c_s - (-dy) /2 *s_s + x;
        bbox.corner_3d[2].y = (-dy) /2 * c_s + (-dx) / 2 * s_s + y;
        bbox.corner_3d[2].z = top_z;
        // p4   上右后
        bbox.corner_3d[3].x = dx/2*c_s - (-dy) /2 *s_s + x;
        bbox.corner_3d[3].y = (-dy) /2 * c_s + dx / 2 * s_s + y;
        bbox.corner_3d[3].z = top_z;
        // p5   下右前
        bbox.corner_3d[4].x = dx/2*c_s - dy /2 *s_s + x;
        bbox.corner_3d[4].y = dy /2 * c_s + dx / 2 * s_s + y;
        bbox.corner_3d[4].z = bot_Z;
        // p6   下左前
        bbox.corner_3d[5].x = (-dx)/2*c_s - dy /2 *s_s + x;
        bbox.corner_3d[5].y = dy /2 * c_s + (-dx) / 2 * s_s + y;
        bbox.corner_3d[5].z = bot_Z;
        // p7   下左后
        bbox.corner_3d[6].x = (-dx)/2*c_s - (-dy) /2 *s_s + x;
        bbox.corner_3d[6].y = (-dy) /2 * c_s + (-dx) / 2 * s_s + y;
        bbox.corner_3d[6].z = bot_Z;
        // p8   下右后
        bbox.corner_3d[7].x = dx/2*c_s - (-dy) /2 *s_s + x;
        bbox.corner_3d[7].y = (-dy) /2 * c_s + dx / 2 * s_s + y;
        bbox.corner_3d[7].z = bot_Z;

        bbox_array_.boxes.push_back(bbox);
    }
    pub_bbox_array_->publish(bbox_array_);

    std::cout<<"  ------------publish   ok     index--------- "<< ++pub_count_<<"   detected objects:  "<<num_objects<<std::endl;
}