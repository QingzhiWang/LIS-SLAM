#include "RangenetAPI.h"
#include "lis_slam/cloud_info.h"
#include "utility.h"

class SemanticFusion : public Paramserver {
 private:
  ros::Subscriber subLaserCloud;
  ros::Publisher pubLaserCloud;

  ros::Publisher pubExtractedCloud;
  ros::Publisher pubLaserCloudInfo;

  std::deque<sensor_msgs::PointCloud2> cloudQueue;
  sensor_msgs::PointCloud2 currentCloudMsg;

  pcl::PointCloud<PointXYZIRT>::Ptr laserCloudIn;

  pcl::PointCloud<PointXYZIRT>::Ptr fullCloud;
  pcl::PointCloud<PointXYZIRT>::Ptr extractedCloud;

  lis_slam::cloud_info cloudInfo;
  std_msgs::Header cloudHeader;

 public:
}