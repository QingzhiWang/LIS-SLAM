// Author of  EPSC-LOAM : QZ Wang
// Email wangqingzhi27@outlook.com

#include "RangenetAPI.h"
#include "lis_slam/cloud_info.h"
#include "utility.h"

class SemanticFusionNode : public Paramserver {
 private:
  double total_time = 0;
  int total_frame = 0;

  ros::Subscriber subCloudInfo;

  ros::Publisher pubSemanticInfo;
  ros::Publisher pubSemanticCloud;
  ros::Publisher pubSemanticRGBCloud
      //   ros::Publisher pubDynObjCloud;

      std::deque<lis_slam::cloud_info>
          cloudInfoQueue;
  std::mutex mtx;

  lis_slam::cloud_info cloudInfo;
  std_msgs::Header cloudHeader;

  sensor_msgs::PointCloud2 currentCloudMsg;
  pcl::PointCloud<PointType>::Ptr currentCloudIn;
  pcl::PointCloud<PointType>::Ptr currentCloudInDS;

  sensor_msgs::PointCloud2 currentCloudCornerMsg;
  pcl::PointCloud<PointType>::Ptr currentCloudCornerIn;
  pcl::PointCloud<PointType>::Ptr currentCloudCornerInDS;

  sensor_msgs::PointCloud2 currentCloudSurfMsg;
  pcl::PointCloud<PointType>::Ptr currentCloudSurfIn;
  pcl::PointCloud<PointType>::Ptr currentCloudSurfInDS;

  pcl::VoxelGrid<PointType> downSizeFilter;

  pcl::PointCloud<pcl::PointXYZRGB> RGBCloudOut;
  pcl::PointCloud<PointXYZIL>::Ptr semanticCloudOut;
  lis_slam::semantic_info semanticInfo;

 public:
  SemanticFusionNode() {
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
    currentCloudCornerIn.reset(new pcl::PointCloud<PointType>());
    currentCloudCornerInDS.reset(new pcl::PointCloud<PointType>());
    currentCloudSurfIn.reset(new pcl::PointCloud<PointType>());
    currentCloudSurfInDS.reset(new pcl::PointCloud<PointType>());

    RGBCloudOut.reset(new pcl::PointCloud<pcl::PointXYZRGB>());

    semanticCloudOut.reset(new pcl::PointCloud<PointXYZIL>());
  }

  void laserCloudInfoHandler(const lis_slam::cloud_infoConstPtr& msgIn) {
    std::lock_guard<std::mutex> lock(mtx);
    cloudInfoQueue.push_back(*msgIn);
  }

  void SemanticFusionNodeThread() {
    while (ros::ok()) {
      std::chrono::time_point<std::chrono::system_clock> start, end;
      start = std::chrono::system_clock::now();

      if (cloudInfoQueue.empty()) {
        continue;
      }

      // extract info and feature cloud
      mtx.lock();
      cloudInfo = cloudInfoQueue.front();
      cloudInfoQueue.pop_front();
      mtx.unlock();

      // extract time stamp
      cloudHeader = cloudInfo.header;

      currentCloudMsg = cloudInfo.cloud_deskewed;
      currentCloudCornerMsg = cloudInfo.cloud_corner;
      currentCloudSurfMsg = cloudInfo.cloud_surface;

      pcl::fromROSMsg(currentCloudMsg, *currentCloudIn);
      pcl::fromROSMsg(currentCloudCornerMsg, *currentCloudCornerIn);
      pcl::fromROSMsg(currentCloudSurfMsg, *currentCloudSurfIn);

      semanticSegmentation();

      publishCloudInfo();

      end = std::chrono::system_clock::now();
      std::chrono::duration<float> elapsed_seconds = end - start;
      total_frame++;
      float time_temp = elapsed_seconds.count() * 1000;
      total_time += time_temp;
      ROS_INFO("Average semantic fusion time %f ms \n \n",
               total_time / total_frame);
    }
  }

  void semanticSegmentation() {
    class RangenetAPI range_net(MODEL_PATH);

    uint32_t num_points = currentCloudIn->points.size();

    std::cout << "点云数量：" << num_points << std::endl;

    std::vector<float> values;

    for (size_t i = 0; i < num_points; i++) {
      values.push_back(currentCloudIn->points[i].x);
      values.push_back(currentCloudIn->points[i].y);
      values.push_back(currentCloudIn->points[i].z);
      values.push_back(currentCloudIn->points[i].intensity);
    }

    range_net.infer(values, num_points);
    std::vector<std::vector<float>> semantic_scan = range_net.getSemanticScan();

    std::vector<int> label_map = range_net.getLabelMap();
    std::map<uint32_t, semantic_color> color_map = range_net.getColorMap();

    std::vector<cv::Vec3f> points = range_net.getPointCloud();
    std::vector<cv::Vec3b> color_mask = range_net.getColorMask();

    std::cout << "num_points size: " << num_points.size() << std::endl;
    std::cout << "semantic_scan size: " << semantic_scan.size() << std::endl;
    std::cout << "points size: " << points.size() << std::endl;

    std::vector<uint32_t> labels;
    std::vector<float> labels_prob;
    labels.resize(num_points);
    labels_prob.resize(num_points);

    for (uint32_t i = 0; i < num_points; ++i) {
      labels_prob[i] = 0;
      for (int32_t j = 0; j < _n_classes; ++j) {
        if (labels_prob[i] <= semantic_scan[i][j]) {
          labels[i] = label_map[j];
          labels_prob[i] = semantic_scan[i][j];
        }
      }
    }

    for (size_t i = 0; i < points.size(); i++) {
      pcl::PointXYZRGB p;

      // 剔除动态物体(假剔除，因为将静态的目标也剔除掉了) 目前仅剔除 car
      // if(color_mask[i][0] == 245 && color_mask[i][1] == 150 &&
      // color_mask[i][2] == 100)
      //     continue;

      p.x = points[i][0];
      p.y = points[i][1];
      p.z = points[i][2];
      p.b = color_mask[i][0];
      p.g = color_mask[i][1];
      p.r = color_mask[i][2];
      RGBCloudOut->points.push_back(p);

      PointXYZIL point;

      point.x = currentCloudIn->points[i].x;
      point.y = currentCloudIn->points[i].y;
      point.z = currentCloudIn->points[i].z;
      point.intensity = currentCloudIn->points[i].intensity;
      point.label = currentCloudIn->points[i].label;
      semanticCloudOut->points.push_back(point);
    }
  }

  void publishCloudInfo() {
    sensor_msgs::PointCloud2 tempCloud;

    pcl::toROSMsg(*RGBCloudOut, tempCloud);
    tempCloud.header.stamp = cloudHeader.stamp;
    tempCloud.header.frame_id = lidarFrame;
    pubSemanticRGBCloud.publish(tempCloud);

    pcl::toROSMsg(*semanticCloudOut, tempCloud);
    tempCloud.header.stamp = cloudHeader.stamp;
    tempCloud.header.frame_id = lidarFrame;
    pubSemanticCloud.publish(tempCloud);

    
  }
};

int main(int argc, char** argv) {
  ros::init(argc, argv, "lis_slam");

  SemanticFusionNode SFN;

  ROS_INFO("\033[1;32m----> Semantic Fusion Node Started.\033[0m");

  std::thread semantic_fusion_thread(
      &SemanticFusionNode::SemanticFusionNodeThread, &SFN);

  ros::spin();

  semantic_fusion_thread.join();

  return 0;
}