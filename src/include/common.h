// Author of  EPSC-LOAM : QZ Wang
// Email wangqingzhi27@outlook.com

#ifndef _COMMON_H_
#define _COMMON_H_

#include "utility.h"

using namespace std;

typedef pcl::PointXYZI PointType;

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
    (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)
    (uint16_t, ring, ring)(float, time, time))

struct PointXYZIL {
    PCL_ADD_POINT4D
    PCL_ADD_INTENSITY;
    uint16_t label;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT(
    PointXYZIL,
    (float, x, x)(float, y, y)(float, z, z)
    (float, intensity, intensity)(uint16_t, label, label))

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
    (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)
    (uint16_t, ring, ring)(float, time, time)(uint16_t, label, label))

/*
 * A point cloud type that has 6D pose info ([x,y,z,roll,pitch,yaw] intensity is
 * time stamp)
 */
struct PointXYZIRPYT {
    PCL_ADD_POINT4D
    PCL_ADD_INTENSITY;  // preferred way of adding a XYZ+padding
    float roll;
    float pitch;
    float yaw;
    double time;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW  // make sure our new allocators are aligned
} EIGEN_ALIGN16;  // enforce SSE padding for correct memory alignment

POINT_CLOUD_REGISTER_POINT_STRUCT(
    PointXYZIRPYT,
    (float, x, x)(float, y, y)(float, z, z)(float, intensity,intensity)
    (float, roll, roll)(float, pitch, pitch)(float, yaw, yaw)(double, time, time))

typedef PointXYZIRPYT PointTypePose;


template <typename T>
double ROS_TIME(T msg) 
{
    return msg->header.stamp.toSec();
}

template <typename T>
void imuAngular2rosAngular(sensor_msgs::Imu *thisImuMsg, T *angular_x,
                           T *angular_y, T *angular_z) 
{
    *angular_x = thisImuMsg->angular_velocity.x;
    *angular_y = thisImuMsg->angular_velocity.y;
    *angular_z = thisImuMsg->angular_velocity.z;
}

template <typename T>
void imuAccel2rosAccel(sensor_msgs::Imu *thisImuMsg, T *acc_x, T *acc_y, T *acc_z) 
{
    *acc_x = thisImuMsg->linear_acceleration.x;
    *acc_y = thisImuMsg->linear_acceleration.y;
    *acc_z = thisImuMsg->linear_acceleration.z;
}

template <typename T>
void imuRPY2rosRPY(sensor_msgs::Imu *thisImuMsg, T *rosRoll, T *rosPitch, T *rosYaw) 
{
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
{
    return sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
}

template <typename PointT>
float pointDistance(PointT p1, PointT p2)
{
    return sqrt((p1.x - p2.x) * (p1.x - p2.x) + 
                (p1.y - p2.y) * (p1.y - p2.y) +
                (p1.z - p2.z) * (p1.z - p2.z));
}




//新增
sensor_msgs::PointCloud2 publishRawCloud(
    ros::Publisher *thisPub, pcl::PointCloud<PointXYZIRTL>::Ptr thisCloud,
    ros::Time thisStamp, std::string thisFrame);

sensor_msgs::PointCloud2 publishCloud(
    ros::Publisher *thisPub, pcl::PointCloud<PointType>::Ptr thisCloud,
    ros::Time thisStamp, std::string thisFrame);

sensor_msgs::PointCloud2 publishLabelCloud(
    ros::Publisher *thisPub, pcl::PointCloud<PointXYZIL>::Ptr thisCloud,
    ros::Time thisStamp, std::string thisFrame);

Eigen::Affine3f pclPointToAffine3f(PointTypePose thisPoint);

Eigen::Affine3f trans2Affine3f(float transformIn[]);

PointTypePose trans2PointTypePose(float transformIn[]);
PointTypePose trans2PointTypePose(float transformIn[], int id, double time);

PointType trans2PointType(float transformIn[]);
PointType trans2PointType(float transformIn[], int id);

pcl::PointCloud<PointType>::Ptr transformPointCloud(pcl::PointCloud<PointType>::Ptr cloudIn, Eigen::Affine3f &transformIn);

pcl::PointCloud<PointType>::Ptr transformPointCloud(pcl::PointCloud<PointType>::Ptr cloudIn, const PointTypePose* transformIn);

pcl::PointCloud<PointXYZIL>::Ptr transformPointCloud(pcl::PointCloud<PointXYZIL>::Ptr cloudIn, Eigen::Affine3f &transCur);

pcl::PointCloud<PointXYZIL>::Ptr transformPointCloud(pcl::PointCloud<PointXYZIL>::Ptr cloudIn, const PointTypePose *transformIn);

pcl::PointCloud<PointXYZIL>::Ptr trans2LabelPointCloud(pcl::PointCloud<PointType>::Ptr cloudIn);

float constraintTransformation(float value, float limit);

float constraintTransformation(float value, float limit, float now, float pre);

#endif  // _COMMON_H_