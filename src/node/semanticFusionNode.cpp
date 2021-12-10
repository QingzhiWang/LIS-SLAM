// Author of  EPSC-LOAM : QZ Wang
// Email wangqingzhi27@outlook.com

#include "lis_slam/cloud_info.h"
#include "lis_slam/semantic_info.h"
#include "rangenetAPI.h"

class SemanticFusionNode : public ParamServer, SemanticLabelParam {
 private:
  double total_time = 0;
  int total_frame = 0;

  ros::Subscriber subCloudRaw;
  ros::Subscriber subCloudInfo;

  ros::Publisher pubSemanticInfo;
  ros::Publisher pubSemanticCloud;
  ros::Publisher pubSemanticRGBCloud;
  //   ros::Publisher pubDynObjCloud;

  std::deque<lis_slam::cloud_info> cloudInfoQueue;
  std::mutex mtx;

  lis_slam::cloud_info cloudInfo;
  std_msgs::Header cloudHeader;

  sensor_msgs::PointCloud2 currentCloudMsg;
  pcl::PointCloud<PointType>::Ptr currentCloudIn;
  pcl::PointCloud<PointType>::Ptr currentCloudInDS;

  // sensor_msgs::PointCloud2 currentCloudCornerMsg;
  // pcl::PointCloud<PointType>::Ptr currentCloudCornerIn;
  // pcl::PointCloud<PointType>::Ptr currentCloudCornerInDS;

  // sensor_msgs::PointCloud2 currentCloudSurfMsg;
  // pcl::PointCloud<PointType>::Ptr currentCloudSurfIn;
  // pcl::PointCloud<PointType>::Ptr currentCloudSurfInDS;

  pcl::VoxelGrid<PointType> downSizeFilter;

  pcl::PointCloud<pcl::PointXYZRGBL>::Ptr semanticCloudOut;
  pcl::PointCloud<pcl::PointXYZRGBL>::Ptr dynamicCloudOut;
  pcl::PointCloud<pcl::PointXYZRGBL>::Ptr staticCloudOut;
  pcl::PointCloud<pcl::PointXYZRGBL>::Ptr outlierCloudOut;

  lis_slam::semantic_info semanticInfo;

  // RangenetAPI range_net;
  RangenetAPI range_net = RangenetAPI(MODEL_PATH);

 public:
  SemanticFusionNode() {
    // subCloudRaw = nh.subscribe<sensor_msgs::PointCloud2>(
    //     pointCloudTopic, 10,
    //     &SemanticFusionNode::laserCloudRawHandler, this);

    subCloudInfo = nh.subscribe<lis_slam::cloud_info>(
        "lis_slam/odom_estimation/cloud_info", 10,
        &SemanticFusionNode::laserCloudInfoHandler, this);

    pubSemanticInfo = nh.advertise<lis_slam::semantic_info>(
        "lis_slam/semantic_fusion/semantic_info", 10);

    pubSemanticRGBCloud = nh.advertise<sensor_msgs::PointCloud2>(
        "lis_slam/semantic_fusion/semantic_cloud_rgb", 10);

    pubSemanticCloud = nh.advertise<sensor_msgs::PointCloud2>(
        "lis_slam/semantic_fusion/semantic_cloud", 10);

    downSizeFilter.setLeafSize(mappingCornerLeafSize, mappingCornerLeafSize,
                               mappingCornerLeafSize);

    allocateMemory();
  }

  void allocateMemory() {
    currentCloudIn.reset(new pcl::PointCloud<PointType>());
    currentCloudInDS.reset(new pcl::PointCloud<PointType>());
    // currentCloudCornerIn.reset(new pcl::PointCloud<PointType>());
    // currentCloudCornerInDS.reset(new pcl::PointCloud<PointType>());
    // currentCloudSurfIn.reset(new pcl::PointCloud<PointType>());
    // currentCloudSurfInDS.reset(new pcl::PointCloud<PointType>());

    semanticCloudOut.reset(new pcl::PointCloud<pcl::PointXYZRGBL>());
    dynamicCloudOut.reset(new pcl::PointCloud<pcl::PointXYZRGBL>());
    staticCloudOut.reset(new pcl::PointCloud<pcl::PointXYZRGBL>());
    outlierCloudOut.reset(new pcl::PointCloud<pcl::PointXYZRGBL>());
  }

  void resetParameters() {
    currentCloudIn->clear();
    currentCloudInDS->clear();
    // currentCloudCornerIn->clear();
    // currentCloudCornerInDS->clear();
    // currentCloudSurfIn->clear();
    // currentCloudSurfInDS->clear();

    semanticCloudOut->clear();
    dynamicCloudOut->clear();
    staticCloudOut->clear();
    outlierCloudOut->clear();
  }

  void laserCloudRawHandler(const sensor_msgs::PointCloud2ConstPtr& msgIn) {
    std::chrono::time_point<std::chrono::system_clock> start, end;
    start = std::chrono::system_clock::now();

    currentCloudMsg = *msgIn;
    cloudHeader = msgIn->header;

    pcl::fromROSMsg(currentCloudMsg, *currentCloudIn);

    range_net.infer(*currentCloudIn);
    semanticCloudOut = range_net.getSemanticCloud();

    publishCloudInfo();

    resetParameters();

    end = std::chrono::system_clock::now();
    std::chrono::duration<float> elapsed_seconds = end - start;
    total_frame++;
    float time_temp = elapsed_seconds.count() * 1000;
    total_time += time_temp;
    ROS_INFO("Average semantic fusion time %f ms \n", total_time / total_frame);
  }

  void laserCloudInfoHandler(const lis_slam::cloud_infoConstPtr& msgIn) {
    std::chrono::time_point<std::chrono::system_clock> start, end;
    start = std::chrono::system_clock::now();

    // extract info and feature cloud
    cloudInfo = *msgIn;
    cloudHeader = cloudInfo.header;

    currentCloudMsg = cloudInfo.cloud_deskewed;
    // currentCloudCornerMsg = cloudInfo.cloud_corner;
    // currentCloudSurfMsg = cloudInfo.cloud_surface;

    pcl::fromROSMsg(currentCloudMsg, *currentCloudIn);
    // pcl::fromROSMsg(currentCloudCornerMsg, *currentCloudCornerIn);
    // pcl::fromROSMsg(currentCloudSurfMsg, *currentCloudSurfIn);

    range_net.infer(*currentCloudIn);
    semanticCloudOut = range_net.getSemanticCloud();

    categoryMapping();

    publishCloudInfo();

    resetParameters();

    end = std::chrono::system_clock::now();
    std::chrono::duration<float> elapsed_seconds = end - start;
    total_frame++;
    float time_temp = elapsed_seconds.count() * 1000;
    total_time += time_temp;
    ROS_INFO("Average semantic fusion time %f ms \n \n",
             total_time / total_frame);
  }


  void categoryMapping() {
    uint32_t num_points = semanticCloudOut->size();

    for (int i = 0; i < num_points; ++i) {
      if (UsingLableMap[semanticCloudOut->points[i].label] == 10)
        dynamicCloudOut->points.push_back(semanticCloudOut->points[i]);
      else if (UsingLableMap[semanticCloudOut->points[i].label] == 40 ||
               UsingLableMap[semanticCloudOut->points[i].label] == 81)
        staticCloudOut->points.push_back(semanticCloudOut->points[i]);
      else
        outlierCloudOut->points.push_back(semanticCloudOut->points[i]);
    }
  }

  void publishCloudInfo() {
    semanticInfo.header = cloudHeader;
    semanticInfo.imuAvailable = cloudInfo.imuAvailable;
    semanticInfo.odomAvailable = cloudInfo.odomAvailable;
    
    semanticInfo.imuRollInit = cloudInfo.imuRollInit;
    semanticInfo.imuPitchInit = cloudInfo.imuPitchInit;
    semanticInfo.imuYawInit = cloudInfo.imuYawInit;

    semanticInfo.initialGuessX = cloudInfo.initialGuessX;
    semanticInfo.initialGuessY = cloudInfo.initialGuessY;
    semanticInfo.initialGuessZ = cloudInfo.initialGuessZ;
    semanticInfo.initialGuessRoll = cloudInfo.initialGuessRoll;
    semanticInfo.initialGuessPitch = cloudInfo.initialGuessPitch;
    semanticInfo.initialGuessYaw = cloudInfo.initialGuessYaw;

    semanticInfo.cloud_corner = cloudInfo.cloud_corner;
    semanticInfo.cloud_surface = cloudInfo.cloud_surface;


    sensor_msgs::PointCloud2 tempCloud;

    pcl::toROSMsg(*semanticCloudOut, tempCloud);
    tempCloud.header.stamp = cloudHeader.stamp;
    tempCloud.header.frame_id = lidarFrame;
    pubSemanticCloud.publish(tempCloud);
    semanticInfo.cloud_semantic = tempCloud;

    pcl::toROSMsg(*dynamicCloudOut, tempCloud);
    tempCloud.header.stamp = cloudHeader.stamp;
    tempCloud.header.frame_id = lidarFrame;
    semanticInfo.cloud_dynamic = tempCloud;

    pcl::toROSMsg(*staticCloudOut, tempCloud);
    tempCloud.header.stamp = cloudHeader.stamp;
    tempCloud.header.frame_id = lidarFrame;
    semanticInfo.cloud_static = tempCloud;

    pcl::toROSMsg(*outlierCloudOut, tempCloud);
    tempCloud.header.stamp = cloudHeader.stamp;
    tempCloud.header.frame_id = lidarFrame;
    semanticInfo.cloud_outlier = tempCloud;

  }
};

int main(int argc, char** argv) {
  ros::init(argc, argv, "lis_slam");

  SemanticFusionNode SFN;

  ROS_INFO("\033[1;32m----> Semantic Fusion Node Started.\033[0m");

  ros::spin();

  return 0;
}