// This code partly draws on  lio_sam
// Author of  EPSC-LOAM : QZ Wang
// Email wangqingzhi27@outlook.com

#ifndef _LASER_PROCESSING_H_
#define _LASER_PROCESSING_H_

#include "lis_slam/cloud_info.h"
#include "utility.h"
#include "common.h"

extern std::mutex imuLock;
extern std::mutex odoLock;
extern std::mutex cloLock;

extern std::deque<sensor_msgs::Imu> imuQueue;
extern std::deque<nav_msgs::Odometry> odomQueue;
extern std::deque<sensor_msgs::PointCloud2> cloudQueue;

extern const int queueLength;

struct smoothness_t 
{
  float value;
  size_t ind;
};

struct by_value 
{
  bool operator()(smoothness_t const &left, smoothness_t const &right) 
  {
    return left.value < right.value;
  }
};

class LaserProcessing : public ParamServer 
{
public:
  LaserProcessing() : deskewFlag(0) 
  {
    allocateMemory();
    resetParameters();

    pcl::console::setVerbosityLevel(pcl::console::L_ERROR);
  }

  void allocateMemory();
  void resetParameters();

  bool distortionRemoval();
  void featureExtraction();

  bool cachePointCloud();
  bool deskewInfo();
  void imuDeskewInfo();
  void odomDeskewInfo();
  void findRotation(double pointTime, float *rotXCur, float *rotYCur, float *rotZCur);
  void findPosition(double relTime, float *posXCur, float *posYCur, float *posZCur);
  PointXYZIRT deskewPoint(PointXYZIRT *point, double relTime);
  void projectPointCloud();
  void cloudExtraction();

  void calculateSmoothness();
  void markOccludedPoints();
  void extractFeatures();

  void assignCouldInfo();

  lis_slam::cloud_info getCloudInfo() { return cloudInfo; }
  
  void freeCloudInfoMemory() 
  {
    cloudInfo.imuAvailable = false;
    cloudInfo.odomAvailable = false;

    cloudInfo.imuRollInit = 0.0;
    cloudInfo.imuPitchInit = 0.0;
    cloudInfo.imuPitchInit = 0.0;

    cloudInfo.imuPitchInit = 0.0;
    cloudInfo.imuPitchInit = 0.0;
    cloudInfo.imuPitchInit = 0.0;
    cloudInfo.imuPitchInit = 0.0;
    cloudInfo.imuPitchInit = 0.0;
    cloudInfo.imuPitchInit = 0.0;

    // cloudInfo.cloud_deskewed.clear();
    // cloudInfo.cloud_corner.clear();
    // cloudInfo.cloud_surface.clear();
  }

private:
  sensor_msgs::PointCloud2 currentCloudMsg;
  std_msgs::Header cloudHeader;

  lis_slam::cloud_info cloudInfo;

  pcl::PointCloud<PointXYZIRT>::Ptr laserCloudIn;

  pcl::PointCloud<PointXYZIRT>::Ptr fullCloud;
  pcl::PointCloud<PointXYZIRT>::Ptr extractedCloud;

  pcl::PointCloud<PointXYZIRT>::Ptr cornerCloud;
  pcl::PointCloud<PointXYZIRT>::Ptr surfaceCloud;

  double *imuTime = new double[queueLength];
  double *imuRotX = new double[queueLength];
  double *imuRotY = new double[queueLength];
  double *imuRotZ = new double[queueLength];

  int imuPointerCur;
  bool firstPointFlag;

  Eigen::Affine3f transStartInverse;

  int deskewFlag;
  cv::Mat rangeMat;

  bool odomDeskewFlag;
  float odomIncreX;
  float odomIncreY;
  float odomIncreZ;

  double timeScanCur;
  double timeScanEnd;

  int32_t *startRingIndex;
  int32_t *endRingIndex;

  int32_t *pointColInd;
  float *pointRange;

  std::vector<smoothness_t> cloudSmoothness;
  float *cloudCurvature;
  int *cloudNeighborPicked;
  int *cloudLabel;

  // pcl::VoxelGrid<PointXYZIRT> downSizeFilter;
};

#endif  // _LASER_PROCESSING_H_
