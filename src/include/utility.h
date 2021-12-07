// This code partly draws on  lio_sam
// Author of  EPSC-LOAM : QZ Wang
// Email wangqingzhi27@outlook.com

#ifndef _UTILITY_LIDAR_ODOMETRY_H_
#define _UTILITY_LIDAR_ODOMETRY_H_

#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/NavSatFix.h>
#include <sensor_msgs/PointCloud2.h>
#include <std_msgs/Float64MultiArray.h>
#include <std_msgs/Header.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

//#include <opencv/cv.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/crop_box.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/range_image/range_image.h>
#include <pcl/registration/icp.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/transforms.h>
#include <tf/LinearMath/Quaternion.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_listener.h>

#include <algorithm>
#include <array>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <ctime>
#include <deque>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <pcl/search/impl/search.hpp>
#include <queue>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

using namespace std;

/*************************************
0: 0      # "unlabeled", and others ignored
1: 10     # "car"
2: 11     # "bicycle"
3: 15     # "motorcycle"
4: 18     # "truck"
5: 20     # "other-vehicle"
6: 30     # "person"
7: 31     # "bicyclist"
8: 32     # "motorcyclist"
9: 40     # "road"
10: 44    # "parking"
11: 48    # "sidewalk"
12: 49    # "other-ground"
13: 50    # "building"
14: 51    # "fence"
15: 70    # "vegetation"
16: 71    # "trunk"
17: 72    # "terrain"
18: 80    # "pole"
19: 81    # "traffic-sign"
1              # "dynamic-object"
*************************************/
extern map<int, string> LABEL;
// LABEL[0] = "unlabeled";
// LABEL[10] = "car";
// LABEL[11] = "bicycle";
// LABEL[15] = "motorcycle";
// LABEL[18] = "truck";
// LABEL[20] = "other-vehicle";
// LABEL[30] = "person";
// LABEL[31] = "bicyclist";
// LABEL[32] = "motorcyclist";
// LABEL[40] = "road";
// LABEL[44] = "parking";
// LABEL[48] = "sidewalk";
// LABEL[49] = "other-ground";
// LABEL[50] = "building";
// LABEL[51] = "fence";
// LABEL[70] = "vegetation";
// LABEL[71] = "trunk";
// LABEL[72] = "terrain";
// LABEL[80] = "pole";
// LABEL[81] = "traffic-sign";
// LABEL[1] = "dynamic-object";

// Velodyne
struct PointXYZIRT {
  PCL_ADD_POINT4D
  PCL_ADD_INTENSITY;
  uint16_t ring;
  float time;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT(
    PointXYZIRT,
    (float, x, x)(float, y, y)(float, z, z)(float, intensity,
                                            intensity)(uint16_t, ring,
                                                       ring)(float, time, time))

struct PointXYZIL {
  PCL_ADD_POINT4D
  PCL_ADD_INTENSITY;
  uint16_t label;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT(
    PointXYZIL,
    (float, x, x)(float, y, y)(float, z, z)(float, intensity,
                                            intensity)(uint16_t, label, label))

struct PointXYZIRTL {
  PCL_ADD_POINT4D
  PCL_ADD_INTENSITY;
  uint16_t ring;
  float time;
  uint16_t label;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT(
    PointXYZIRTL,
    (float, x, x)(float, y, y)(float, z, z)(float, intensity,
                                            intensity)(uint16_t, ring,
                                                       ring)(float, time,
                                                             time)(uint16_t,
                                                                   label,
                                                                   label))

typedef pcl::PointXYZI PointType;

class ParamServer {
 public:
  ros::NodeHandle nh;

  std::string robot_id;

  bool useImu;

  std::string RESULT_PATH;
  std::string MODEL_PATH;

  // Topics
  string pointCloudTopic;
  string imuTopic;
  string odomTopic;
  string gpsTopic;

  // Frames
  string lidarFrame;
  string baselinkFrame;
  string odometryFrame;
  string mapFrame;

  // GPS Settings
  bool useImuHeadingInitialization;
  bool useGpsElevation;
  float gpsCovThreshold;
  float poseCovThreshold;

  // Save pcd
  bool savePCD;
  string savePCDDirectory;

  // Velodyne Sensor Configuration: Velodyne
  int N_SCAN;
  int Horizon_SCAN;

  int downsampleRate;
  float lidarMinRange;
  float lidarMaxRange;

  // IMU
  float imuAccNoise;
  float imuGyrNoise;
  float imuAccBiasN;
  float imuGyrBiasN;
  float imuGravity;
  float imuRPYWeight;
  vector<double> extRotV;
  vector<double> extRPYV;
  vector<double> extTransV;
  Eigen::Matrix3d extRot;
  Eigen::Matrix3d extRPY;
  Eigen::Vector3d extTrans;
  Eigen::Quaterniond extQRPY;

  // LOAM
  float edgeThreshold;
  float surfThreshold;
  int edgeFeatureMinValidNum;
  int surfFeatureMinValidNum;

  // voxel filter paprams
  float odometrySurfLeafSize;
  float mappingCornerLeafSize;
  float mappingSurfLeafSize;

  //新增 submap voxel filter
  float subMapCornerLeafSize;
  float subMapSurfLeafSize;
  float subMapLeafSize;

  float z_tollerance;
  float rotation_tollerance;

  // CPU Params
  int numberOfCores;
  double mappingProcessInterval;

  // Surrounding map
  float surroundingkeyframeAddingDistThreshold;
  float surroundingkeyframeAddingAngleThreshold;
  float surroundingKeyframeDensity;
  float surroundingKeyframeSearchRadius;

  // Loop closure
  bool loopClosureEnableFlag;
  float loopClosureFrequency;
  int surroundingKeyframeSize;
  float historyKeyframeSearchRadius;
  float historyKeyframeSearchTimeDiff;
  int historyKeyframeSearchNum;
  float historyKeyframeFitnessScore;

  //新增
  float distanceKeyframeThresh;
  int accumDistanceIndexThresh;
  int historyAccumDistanceIndexThresh;
  int distanceFromLastIndexThresh;

  float loopClosureCornerLeafSize;
  float loopClosureSurfLeafSize;

  //新增 make Submap
  int subMapFramesSize;
  float keyFrameMiniDistance;
  float subMapYawMax;
  float subMapMaxTime;
  float keyFrameMiniYaw;

  float subMapOptmizationDistanceThresh;
  float subMapOptmizationYawThresh;

  int subMapOptmizationFirstSize;

  float subMapOptmizationWeights;
  float odometerAndOptimizedDistanceDifference;
  float odometerAndOptimizedAngleDifference;

  bool useOdometryPitchPrediction;

  // global map visualization radius
  float globalMapVisualizationSearchRadius;
  float globalMapVisualizationPoseDensity;
  float globalMapVisualizationLeafSize;

  ParamServer() {
    nh.param<std::string>("/robot_id", robot_id, "roboat");

    nh.param<bool>("lis_slam/useImu", useImu, true);

    nh.param<std::string>("lis_slam/RESULT_PATH", RESULT_PATH,
                          "../assets/trajectory/test_pred.txt");
    nh.param<std::string>("lis_slam/MODEL_PATH", MODEL_PATH, "./");

    nh.param<std::string>("lis_slam/pointCloudTopic", pointCloudTopic,
                          "points_raw");
    nh.param<std::string>("lis_slam/imuTopic", imuTopic, "imu_correct");
    nh.param<std::string>("lis_slam/odomTopic", odomTopic, "odometry/imu");
    nh.param<std::string>("lis_slam/gpsTopic", gpsTopic, "odometry/gps");

    nh.param<std::string>("lis_slam/lidarFrame", lidarFrame, "base_link");
    nh.param<std::string>("lis_slam/baselinkFrame", baselinkFrame, "base_link");
    nh.param<std::string>("lis_slam/odometryFrame", odometryFrame, "odom");
    nh.param<std::string>("lis_slam/mapFrame", mapFrame, "map");

    nh.param<bool>("lis_slam/useImuHeadingInitialization",
                   useImuHeadingInitialization, false);
    nh.param<bool>("lis_slam/useGpsElevation", useGpsElevation, false);
    nh.param<float>("lis_slam/gpsCovThreshold", gpsCovThreshold, 2.0);
    nh.param<float>("lis_slam/poseCovThreshold", poseCovThreshold, 25.0);

    nh.param<bool>("lis_slam/savePCD", savePCD, false);
    nh.param<std::string>("lis_slam/savePCDDirectory", savePCDDirectory,
                          "/Downloads/LOAM/");

    nh.param<int>("lis_slam/N_SCAN", N_SCAN, 16);
    nh.param<int>("lis_slam/Horizon_SCAN", Horizon_SCAN, 1800);

    nh.param<int>("lis_slam/downsampleRate", downsampleRate, 1);
    nh.param<float>("lis_slam/lidarMinRange", lidarMinRange, 1.0);
    nh.param<float>("lis_slam/lidarMaxRange", lidarMaxRange, 1000.0);

    nh.param<float>("lis_slam/imuAccNoise", imuAccNoise, 0.01);
    nh.param<float>("lis_slam/imuGyrNoise", imuGyrNoise, 0.001);
    nh.param<float>("lis_slam/imuAccBiasN", imuAccBiasN, 0.0002);
    nh.param<float>("lis_slam/imuGyrBiasN", imuGyrBiasN, 0.00003);
    nh.param<float>("lis_slam/imuGravity", imuGravity, 9.80511);
    nh.param<float>("lis_slam/imuRPYWeight", imuRPYWeight, 0.01);
    nh.param<vector<double>>("lis_slam/extrinsicRot", extRotV,
                             vector<double>());
    nh.param<vector<double>>("lis_slam/extrinsicRPY", extRPYV,
                             vector<double>());
    nh.param<vector<double>>("lis_slam/extrinsicTrans", extTransV,
                             vector<double>());
    extRot = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(
        extRotV.data(), 3, 3);
    extRPY = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(
        extRPYV.data(), 3, 3);
    extTrans = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(
        extTransV.data(), 3, 1);
    extQRPY = Eigen::Quaterniond(extRPY);

    nh.param<float>("lis_slam/edgeThreshold", edgeThreshold, 0.1);
    nh.param<float>("lis_slam/surfThreshold", surfThreshold, 0.1);
    nh.param<int>("lis_slam/edgeFeatureMinValidNum", edgeFeatureMinValidNum,
                  10);
    nh.param<int>("lis_slam/surfFeatureMinValidNum", surfFeatureMinValidNum,
                  100);

    nh.param<float>("lis_slam/odometrySurfLeafSize", odometrySurfLeafSize, 0.2);
    nh.param<float>("lis_slam/mappingCornerLeafSize", mappingCornerLeafSize,
                    0.2);
    nh.param<float>("lis_slam/mappingSurfLeafSize", mappingSurfLeafSize, 0.2);

    //新增 submap voxel filter
    nh.param<float>("lis_slam/subMapCornerLeafSize", subMapCornerLeafSize, 0.2);
    nh.param<float>("lis_slam/subMapSurfLeafSize", subMapSurfLeafSize, 0.4);
    nh.param<float>("lis_slam/subMapLeafSize", subMapLeafSize, 0.4);

    nh.param<float>("lis_slam/z_tollerance", z_tollerance, FLT_MAX);
    nh.param<float>("lis_slam/rotation_tollerance", rotation_tollerance,
                    FLT_MAX);

    nh.param<int>("lis_slam/numberOfCores", numberOfCores, 2);
    nh.param<double>("lis_slam/mappingProcessInterval", mappingProcessInterval,
                     0.15);

    nh.param<float>("lis_slam/surroundingkeyframeAddingDistThreshold",
                    surroundingkeyframeAddingDistThreshold, 1.0);
    nh.param<float>("lis_slam/surroundingkeyframeAddingAngleThreshold",
                    surroundingkeyframeAddingAngleThreshold, 0.2);
    nh.param<float>("lis_slam/surroundingKeyframeDensity",
                    surroundingKeyframeDensity, 1.0);
    nh.param<float>("lis_slam/surroundingKeyframeSearchRadius",
                    surroundingKeyframeSearchRadius, 50.0);

    nh.param<bool>("lis_slam/loopClosureEnableFlag", loopClosureEnableFlag,
                   false);
    nh.param<float>("lis_slam/loopClosureFrequency", loopClosureFrequency, 1.0);
    nh.param<int>("lis_slam/surroundingKeyframeSize", surroundingKeyframeSize,
                  50);
    nh.param<float>("lis_slam/historyKeyframeSearchRadius",
                    historyKeyframeSearchRadius, 10.0);
    nh.param<float>("lis_slam/historyKeyframeSearchTimeDiff",
                    historyKeyframeSearchTimeDiff, 30.0);
    nh.param<int>("lis_slam/historyKeyframeSearchNum", historyKeyframeSearchNum,
                  25);
    nh.param<float>("lis_slam/historyKeyframeFitnessScore",
                    historyKeyframeFitnessScore, 0.3);

    //新增
    nh.param<float>("lis_slam/distanceKeyframeThresh", distanceKeyframeThresh,
                    15.0);
    nh.param<int>("lis_slam/accumDistanceIndexThresh", accumDistanceIndexThresh,
                  10);
    nh.param<int>("lis_slam/historyAccumDistanceIndexThresh",
                  historyAccumDistanceIndexThresh, 5);
    nh.param<int>("lis_slam/distanceFromLastIndexThresh",
                  distanceFromLastIndexThresh, 3);

    nh.param<float>("lis_slam/loopClosureCornerLeafSize",
                    loopClosureCornerLeafSize, 0.2);
    nh.param<float>("lis_slam/loopClosureSurfLeafSize", loopClosureSurfLeafSize,
                    0.5);

    //新增subMapMaxTime
    nh.param<float>("lis_slam/subMapMaxTime", subMapMaxTime, 2.0);
    nh.param<int>("lis_slam/subMapFramesSize", subMapFramesSize, 10);
    nh.param<float>("lis_slam/subMapYawMax", subMapYawMax, 0.5);

    nh.param<float>("lis_slam/keyFrameMiniDistance", keyFrameMiniDistance, 0.1);
    nh.param<float>("lis_slam/keyFrameMiniYaw", keyFrameMiniYaw, 0.2);

    nh.param<float>("lis_slam/subMapOptmizationDistanceThresh",
                    subMapOptmizationDistanceThresh, 0.2);
    nh.param<float>("lis_slam/subMapOptmizationYawThresh",
                    subMapOptmizationYawThresh, 0.5);

    nh.param<int>("lis_slam/subMapOptmizationFirstSize",
                  subMapOptmizationFirstSize, 5);

    nh.param<float>("lis_slam/subMapOptmizationWeights",
                    subMapOptmizationWeights, 0.5);
    nh.param<float>("lis_slam/odometerAndOptimizedDistanceDifference",
                    odometerAndOptimizedDistanceDifference, 0.2);
    nh.param<float>("lis_slam/odometerAndOptimizedAngleDifference",
                    odometerAndOptimizedAngleDifference, 0.25);

    nh.param<bool>("lis_slam/useOdometryPitchPrediction",
                   useOdometryPitchPrediction, true);

    nh.param<float>("lis_slam/globalMapVisualizationSearchRadius",
                    globalMapVisualizationSearchRadius, 1e3);
    nh.param<float>("lis_slam/globalMapVisualizationPoseDensity",
                    globalMapVisualizationPoseDensity, 10.0);
    nh.param<float>("lis_slam/globalMapVisualizationLeafSize",
                    globalMapVisualizationLeafSize, 1.0);

    usleep(100);
  }

  sensor_msgs::Imu imuConverter(const sensor_msgs::Imu &imu_in) {
    sensor_msgs::Imu imu_out = imu_in;
    // rotate acceleration
    Eigen::Vector3d acc(imu_in.linear_acceleration.x,
                        imu_in.linear_acceleration.y,
                        imu_in.linear_acceleration.z);
    acc = extRot * acc;
    imu_out.linear_acceleration.x = acc.x();
    imu_out.linear_acceleration.y = acc.y();
    imu_out.linear_acceleration.z = acc.z();
    // rotate gyroscope
    Eigen::Vector3d gyr(imu_in.angular_velocity.x, imu_in.angular_velocity.y,
                        imu_in.angular_velocity.z);
    gyr = extRot * gyr;
    imu_out.angular_velocity.x = gyr.x();
    imu_out.angular_velocity.y = gyr.y();
    imu_out.angular_velocity.z = gyr.z();
    // rotate roll pitch yaw
    Eigen::Quaterniond q_from(imu_in.orientation.w, imu_in.orientation.x,
                              imu_in.orientation.y, imu_in.orientation.z);
    Eigen::Quaterniond q_final = q_from * extQRPY;
    imu_out.orientation.x = q_final.x();
    imu_out.orientation.y = q_final.y();
    imu_out.orientation.z = q_final.z();
    imu_out.orientation.w = q_final.w();

    if (sqrt(q_final.x() * q_final.x() + q_final.y() * q_final.y() +
             q_final.z() * q_final.z() + q_final.w() * q_final.w()) < 0.1) {
      ROS_ERROR("Invalid quaternion, please use a 9-axis IMU!");
      ros::shutdown();
    }

    return imu_out;
  }
};

//新增
static sensor_msgs::PointCloud2 publishRawCloud(
    ros::Publisher *thisPub, pcl::PointCloud<PointXYZIRTL>::Ptr thisCloud,
    ros::Time thisStamp, std::string thisFrame) {
  sensor_msgs::PointCloud2 tempCloud;
  pcl::toROSMsg(*thisCloud, tempCloud);
  tempCloud.header.stamp = thisStamp;
  tempCloud.header.frame_id = thisFrame;
  if (thisPub->getNumSubscribers() != 0) thisPub->publish(tempCloud);
  return tempCloud;
}

static sensor_msgs::PointCloud2 publishCloud(ros::Publisher *thisPub,
                                      pcl::PointCloud<PointType>::Ptr thisCloud,
                                      ros::Time thisStamp,
                                      std::string thisFrame) {
  sensor_msgs::PointCloud2 tempCloud;
  pcl::toROSMsg(*thisCloud, tempCloud);
  tempCloud.header.stamp = thisStamp;
  tempCloud.header.frame_id = thisFrame;
  if (thisPub->getNumSubscribers() != 0) thisPub->publish(tempCloud);
  return tempCloud;
}

template <typename T>
double ROS_TIME(T msg) {
  return msg->header.stamp.toSec();
}

template <typename T>
void imuAngular2rosAngular(sensor_msgs::Imu *thisImuMsg, T *angular_x,
                           T *angular_y, T *angular_z) {
  *angular_x = thisImuMsg->angular_velocity.x;
  *angular_y = thisImuMsg->angular_velocity.y;
  *angular_z = thisImuMsg->angular_velocity.z;
}

template <typename T>
void imuAccel2rosAccel(sensor_msgs::Imu *thisImuMsg, T *acc_x, T *acc_y,
                       T *acc_z) {
  *acc_x = thisImuMsg->linear_acceleration.x;
  *acc_y = thisImuMsg->linear_acceleration.y;
  *acc_z = thisImuMsg->linear_acceleration.z;
}

template <typename T>
void imuRPY2rosRPY(sensor_msgs::Imu *thisImuMsg, T *rosRoll, T *rosPitch,
                   T *rosYaw) {
  double imuRoll, imuPitch, imuYaw;
  tf::Quaternion orientation;
  tf::quaternionMsgToTF(thisImuMsg->orientation, orientation);
  tf::Matrix3x3(orientation).getRPY(imuRoll, imuPitch, imuYaw);

  *rosRoll = imuRoll;
  *rosPitch = imuPitch;
  *rosYaw = imuYaw;
}

template <typename PointT>
float pointDistance(PointT p)
// float pointDistance(PointType p)
{
  return sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
}

template <typename PointT>
float pointDistance(PointT p1, PointT p2)
// float pointDistance(PointType p1, PointType p2)
{
  return sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y) +
              (p1.z - p2.z) * (p1.z - p2.z));
}

#endif
