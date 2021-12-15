// Author of  EPSC-LOAM : QZ Wang
// Email wangqingzhi27@outlook.com

#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>

#include "epscGeneration.h"
#include "keyFrame.h"
#include "subMap.h"

#include "lis_slam/semantic_info.h"
#include "utility.h"

using namespace gtsam;

using symbol_shorthand::B;  // Bias  (ax,ay,az,gx,gy,gz)
using symbol_shorthand::G;  // GPS pose
using symbol_shorthand::V;  // Vel   (xdot,ydot,zdot)
using symbol_shorthand::X;  // Pose3 (x,y,z,r,p,y)

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
    (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)
    (float, roll,roll)(float, pitch,pitch)(float, yaw,yaw)(double,time,time))

typedef PointXYZIRPYT PointTypePose;


class SubMapOptmizationNode : public ParamServer, SemanticLabelParam {
 public:
  ros::Subscriber subCloud;
  ros::Subscriber subGPS;

  std::mutex seMtx;
  std::mutex gpsMtx;
  std::deque<nav_msgs::Odometry> gpsQueue;
  std::deque<lis_slam::semantic_info> seInfoQueue;

  map<int, keyframe_Ptr> keyFrameInfo;
  int keyFrameID = 0;

  ros::Publisher pubCloudRegisteredRaw;
  ros::Publisher pubCloudCurSubMap;
  ros::Publisher pubCloudMap;

  ros::Publisher pubSubMapOdometryGlobal;
  ros::Publisher pubKeyFrameOdometryGlobal;

  ros::Publisher pubKeyFramePoseGlobal;
  ros::Publisher pubKeyFramePath;
  ros::Publisher pubSubMapConstraintEdge;

  ros::Publisher pubLoopConstraintEdge;

  void allocateMemory() {}

  SubMapOptmizationNode() {
    subGPS = nh.subscribe<nav_msgs::Odometry>(
        gpsTopic, 2000, &SubMapOptmizationNode::gpsHandler, this,
        ros::TransportHints().tcpNoDelay());
    subCloud = nh.subscribe<lis_slam::semantic_info>(
        "lis_slam/semantic_fusion/semantic_info", 10,
        &SubMapOptmizationNode::semanticInfoHandler, this,
        ros::TransportHints().tcpNoDelay());

    pubSubMapOdometryGlobal =
        nh.advertise<nav_msgs::Odometry>("lis_slam/mapping/submap_odometry", 1);
    pubKeyFrameOdometryGlobal = nh.advertise<nav_msgs::Odometry>(
        "lis_slam/mapping/keyframe_odometry", 1);

    pubKeyFramePoseGlobal = nh.advertise<sensor_msgs::PointCloud2>(
        "lis_slam/mapping/trajectory", 1);
    pubKeyFramePath =
        nh.advertise<nav_msgs::Path>("lis_slam/mapping/keyframe_path", 1);
    pubSubMapConstraintEdge = nh.advertise<visualization_msgs::MarkerArray>(
        "lis_slam/mapping/submap_constraints", 1);

    pubCloudRegisteredRaw = nh.advertise<sensor_msgs::PointCloud2>(
        "lis_slam/mapping/registered_raw", 1);
    pubCloudCurSubMap =
        nh.advertise<sensor_msgs::PointCloud2>("lis_slam/mapping/submap", 1);
    pubCloudMap = nh.advertise<sensor_msgs::PointCloud2>(
        "lis_slam/mapping/map_global", 1);

    pubLoopConstraintEdge = nh.advertise<visualization_msgs::MarkerArray>(
        "lis_slam/mapping/loop_closure_constraints", 1);

    allocateMemory();
  }

  void semanticInfoHandler(const lis_slam::semantic_infoConstPtr &msgIn) {
    std::lock_guard<std::mutex> lock(seMtx);
    seInfoQueue.push_back(*msgIn);

    if (seInfoQueue.size() > 0) {
      lis_slam::semantic_info currentInfoMsg = seInfoQueue.front();
      cloudQueue.pop_front();

      keyframe_Ptr currentKeyFrame;
      currentKeyFrame->keyframe_id = keyFrameID;
      currentKeyFrame->timeInfoStamp = currentInfoMsg.header.stamp;
      currentKeyFrame->submap_id = -1;
      currentKeyFrame->id_in_submap = -1;

      if (currentInfoMsg.odomAvailable == true) {
        currentKeyFrame->init_pose = pcl::getTransformation(
            currentInfoMsg.initialGuessX, currentInfoMsg.initialGuessY,
            currentInfoMsg.initialGuessZ, currentInfoMsg.initialGuessRoll,
            currentInfoMsg.initialGuessPitch, currentInfoMsg.initialGuessYaw);
      } else if (currentInfoMsg.imuAvailable == true) {
        currentKeyFrame->init_pose = pcl::getTransformation(
            0, 0, 0, currentInfoMsg.imuRollInit, currentInfoMsg.imuPitchInit,
            currentInfoMsg.imuYawInit);
      }
      pcl::PointCloud<PointXYZIL>::Ptr semantic_pointcloud_in(
          new pcl::PointCloud<PointXYZIL>());
      pcl::PointCloud<PointXYZIL>::Ptr dynamic_pointcloud_in(
          new pcl::PointCloud<PointXYZIL>());
      pcl::PointCloud<PointXYZIL>::Ptr static_pointcloud_in(
          new pcl::PointCloud<PointXYZIL>());
      pcl::PointCloud<PointXYZIL>::Ptr outlier_pointcloud_in(
          new pcl::PointCloud<PointXYZIL>());

      pcl::PointCloud<pcl::PointXYZI>::Ptr corner_pointcloud_in(
          new pcl::PointCloud<pcl::PointXYZI>());
      pcl::PointCloud<pcl::PointXYZI>::Ptr surf_pointcloud_in(
          new pcl::PointCloud<pcl::PointXYZI>());

      pcl::fromROSMsg(currentInfoMsg.cloud_semantic, *semantic_pointcloud_in);
      pcl::fromROSMsg(currentInfoMsg.cloud_dynamic, *dynamic_pointcloud_in);
      pcl::fromROSMsg(currentInfoMsg.cloud_static, *static_pointcloud_in);
      pcl::fromROSMsg(currentInfoMsg.cloud_outlier, *outlier_pointcloud_in);

      pcl::fromROSMsg(currentInfoMsg.cloud_corner, *corner_pointcloud_in);
      pcl::fromROSMsg(currentInfoMsg.cloud_surface, *surf_pointcloud_in);

      currentKeyFrame->cloud_semantic = semantic_pointcloud_in;
      currentKeyFrame->cloud_dynamic = dynamic_pointcloud_in;
      currentKeyFrame->cloud_static = static_pointcloud_in;
      currentKeyFrame->cloud_outlier = outlier_pointcloud_in;

      currentKeyFrame->cloud_corner = corner_pointcloud_in;
      currentKeyFrame->cloud_surface = surf_pointcloud_in;

      //   pcl::PointCloud<PointXYZIL>::Ptr semantic_pointcloud_DS(
      //       new pcl::PointCloud<PointXYZIL>());
      //   pcl::PointCloud<PointXYZIL>::Ptr dynamic_pointcloud_DS(
      //       new pcl::PointCloud<PointXYZIL>());
      //   pcl::PointCloud<PointXYZIL>::Ptr static_pointcloud_DS(
      //       new pcl::PointCloud<PointXYZIL>());
      //   pcl::PointCloud<PointXYZIL>::Ptr outlier_pointcloud_DS(
      //       new pcl::PointCloud<PointXYZIL>());

      //   pcl::PointCloud<pcl::PointXYZI>::Ptr corner_pointcloud_DS(
      //       new pcl::PointCloud<pcl::PointXYZI>());
      //   pcl::PointCloud<pcl::PointXYZI>::Ptr surf_pointcloud_DS(
      //       new pcl::PointCloud<pcl::PointXYZI>());

      //   pcl::PointCloud<PointXYZIL>::Ptr cloud_semantic_down;
      //   pcl::PointCloud<PointXYZIL>::Ptr cloud_dynamic_down;
      //   pcl::PointCloud<PointXYZIL>::Ptr cloud_static_down;
      //   pcl::PointCloud<PointXYZIL>::Ptr cloud_outlier_down;

      //   pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_corner_down;
      //   pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_surface_down;

      keyFrameInfo[keyFrameID] = currentKeyFrame;
      keyFrameID++;
    }

    while (seInfoQueue.size() > 6) seInfoQueue.pop_front();
  }

  void gpsHandler(const nav_msgs::Odometry::ConstPtr &gpsMsg) {
    std::lock_guard<std::mutex> lock(gpsMtx);
    gpsQueue.push_back(*gpsMsg);
  }

  void makeSubMapThread(){

  }

  void loopClosureThread(){
    int processID = 0;
    EPSCGeneration epscGen;

    while(ros::ok()){
      if(processID<=keyFrameID){
        keyframe_Ptr curKeyFramePtr;
        if(keyFrameInfo.find(processID) != keyFrameInfo.end())
          curKeyFramePtr = keyFrameInfo[processID];
        
        epscGen.loopDetection(curKeyFramePtr->cloud_corner, curKeyFramePtr->cloud_surface, 
        curKeyFramePtr->cloud_semantic,curKeyFramePtr->cloud_static,
        curKeyFramePtr->init_pose);

        
      }
    }
  }

  void subMapOptmizationThread(){

  }

  void pointAssociateToMap(PointType const *const pi, PointType *const po) {
    po->x = transPointAssociateToMap(0, 0) * pi->x +
            transPointAssociateToMap(0, 1) * pi->y +
            transPointAssociateToMap(0, 2) * pi->z +
            transPointAssociateToMap(0, 3);
    po->y = transPointAssociateToMap(1, 0) * pi->x +
            transPointAssociateToMap(1, 1) * pi->y +
            transPointAssociateToMap(1, 2) * pi->z +
            transPointAssociateToMap(1, 3);
    po->z = transPointAssociateToMap(2, 0) * pi->x +
            transPointAssociateToMap(2, 1) * pi->y +
            transPointAssociateToMap(2, 2) * pi->z +
            transPointAssociateToMap(2, 3);
    po->intensity = pi->intensity;
  }

  pcl::PointCloud<PointType>::Ptr transformPointCloud(
      pcl::PointCloud<PointType>::Ptr cloudIn, PointTypePose *transformIn) {
    pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

    PointType *pointFrom;

    int cloudSize = cloudIn->size();
    cloudOut->resize(cloudSize);

    Eigen::Affine3f transCur = pcl::getTransformation(
        transformIn->x, transformIn->y, transformIn->z, transformIn->roll,
        transformIn->pitch, transformIn->yaw);

// #pragma omp parallel for num_threads(numberOfCores)
#pragma omp for
    for (int i = 0; i < cloudSize; ++i) {
      pointFrom = &cloudIn->points[i];
      cloudOut->points[i].x = transCur(0, 0) * pointFrom->x +
                              transCur(0, 1) * pointFrom->y +
                              transCur(0, 2) * pointFrom->z + transCur(0, 3);
      cloudOut->points[i].y = transCur(1, 0) * pointFrom->x +
                              transCur(1, 1) * pointFrom->y +
                              transCur(1, 2) * pointFrom->z + transCur(1, 3);
      cloudOut->points[i].z = transCur(2, 0) * pointFrom->x +
                              transCur(2, 1) * pointFrom->y +
                              transCur(2, 2) * pointFrom->z + transCur(2, 3);
      cloudOut->points[i].intensity = pointFrom->intensity;
    }
    return cloudOut;
  }

  gtsam::Pose3 pclPointTogtsamPose3(PointTypePose thisPoint) {
    return gtsam::Pose3(
        gtsam::Rot3::RzRyRx(double(thisPoint.roll), double(thisPoint.pitch),
                            double(thisPoint.yaw)),
        gtsam::Point3(double(thisPoint.x), double(thisPoint.y),
                      double(thisPoint.z)));
  }

  gtsam::Pose3 trans2gtsamPose(float transformIn[]) {
    return gtsam::Pose3(
        gtsam::Rot3::RzRyRx(transformIn[0], transformIn[1], transformIn[2]),
        gtsam::Point3(transformIn[3], transformIn[4], transformIn[5]));
  }

  Eigen::Affine3f pclPointToAffine3f(PointTypePose thisPoint) {
    return pcl::getTransformation(thisPoint.x, thisPoint.y, thisPoint.z,
                                  thisPoint.roll, thisPoint.pitch,
                                  thisPoint.yaw);
  }

  Eigen::Affine3f trans2Affine3f(float transformIn[]) {
    return pcl::getTransformation(transformIn[3], transformIn[4],
                                  transformIn[5], transformIn[0],
                                  transformIn[1], transformIn[2]);
  }

  PointTypePose trans2PointTypePose(float transformIn[]) {
    PointTypePose thisPose6D;
    thisPose6D.x = transformIn[3];
    thisPose6D.y = transformIn[4];
    thisPose6D.z = transformIn[5];
    thisPose6D.roll = transformIn[0];
    thisPose6D.pitch = transformIn[1];
    thisPose6D.yaw = transformIn[2];
    return thisPose6D;
  }
};

int main(int argc, char **argv) {
  ros::init(argc, argv, "lis_slam");

  SubMapOptmizationNode SON;

  std::thread make_submap_process(&SubMapOptmizationNode::makeSubMapThread,
                                  &SON);
  std::thread loop_closure_process(&SubMapOptmizationNode::loopClosureThread,
                                   &SON);
  std::thread submap_optmization_process(
      &SubMapOptmizationNode::subMapOptmizationThread, &SON);

  ROS_INFO("\033[1;32m----> SubMap Optmization Node Started.\033[0m");

  //   ros::MultiThreadedSpinner spinner(3);
  //   spinner.spin();
  ros::spin();

  make_submap_process.join();
  loop_closure_process.join();
  submap_optmization_process.join();

  return 0;
}