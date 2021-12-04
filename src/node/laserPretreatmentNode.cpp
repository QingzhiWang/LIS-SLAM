// Author of  EPSC-LOAM : QZ Wang
// Email wangqingzhi27@outlook.com

// local lib
#include "laserPretreatment.h"

LaserPretreatment lpre;

std::mutex cloudLock;
std::deque<sensor_msgs::PointCloud2ConstPtr> cloudQueue;

ros::Subscriber subLaserCloud ros::Publisher pubLaserCloud;

void cloudHandler(const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg) {
  std::lock_guard<std::mutex> lock(cloudLock);
  cloudQueue.push_back(laserCloudMsg);
}

double total_time = 0;
int total_frame = 0;

void laserPretreatment() {
  while (1) {
    if (!cloudQueue.empty()) {
      //激光预处理 增加ring time Label通道供后续步骤使用

      // read data
      cloudLock.lock();
      sensor_msgs::PointCloud2ConstPtr laserCloudMsg =
          *cloudQueue.front() pcl::PointCloud<PointType>::Ptr laserCloudIn(
              new pcl::PointCloud<PointType>());
      pcl::fromROSMsg(*laserCloudMsg, *laserCloudIn);
      cloudQueue.pop();
      cloudLock.unlock();

      // ros::Time  startTime=ros::Time::now();

      // pcl::PointCloud<PointXYZIRTL> laserCloudOut;
      // laserCloudOut=lpre.process(*laserCloudIn);

      // ros::Time  endTime=ros::Time::now();
      // std::cout <<  "Laser Pretreatment  Time: " <<  (endTime -
      // startTime).toSec() << "[sec]" << std::endl;

      // if((endTime - startTime).toSec()> 1)
      //     ROS_WARN("Laser Pretreatment process over 100ms");

      std::chrono::time_point<std::chrono::system_clock> start, end;
      start = std::chrono::system_clock::now();

      pcl::PointCloud<PointXYZIRTL> laserCloudOut;
      laserCloudOut = lpre.process(*laserCloudIn);

      end = std::chrono::system_clock::now();
      std::chrono::duration<float> elapsed_seconds = end - start;
      total_frame++;
      float time_temp = elapsed_seconds.count() * 1000;
      total_time += time_temp;
      ROS_INFO("Average laser pretreatment time %f ms \n \n",
               total_time / total_frame);

      sensor_msgs::PointCloud2 laserCloudOutMsg;
      pcl::toROSMsg(laserCloudOut, laserCloudOutMsg);
      laserCloudOutMsg.header.stamp = laserCloudMsg->header.stamp;
      laserCloudOutMsg.header.frame_id = lidarFrame;
      pubLaserCloud.publish(laserCloudOutMsg);
    }
  }
  // sleep 2 ms every time
  std::chrono::milliseconds dura(2);
  std::this_thread::sleep_for(dura);
}

int main(int argc, char** argv) {
  ros::init(argc, argv, "lis_slam");

  if (N_SCAN != 16 && N_SCAN != 32 && N_SCAN != 64) {
    printf("only support velodyne with 16, 32 or 64 scan line!");
    return 0;
  }

  subLaserCloud =
      nh.subscribe<sensor_msgs::PointCloud2>(pointCloudTopic, 5, &cloudHandler);
  pubLaserCloud = nh.advertise<sensor_msgs::PointCloud2>(
      "lis_slam/points_pretreatmented", 1);

  std::thread pretreatment_process{laserPretreatment};

  ROS_INFO("\033[1;32m----> Laser Pretreatment Node Started.\033[0m");

  ros::spin();

  return 0;
}