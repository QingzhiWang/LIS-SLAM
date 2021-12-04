// Author of  EPSC-LOAM : QZ Wang  
// Email wangqingzhi27@outlook.com

#include "utility.h"
#include "lis_slam/cloud_info.h"

//local lib
#include "laserPretreatment.h"
#include "laserProcessing.h"

#include <cmath>

using std::atan2;
using std::cos;
using std::sin;

const double scanPeriod = 0.1;
const int queueLength = 2000;


LaserPretreatment lpre;
LaserProcessing lpro;

std::mutex imuLock;
std::mutex odoLock;
std::mutex cloLock;

std::deque<sensor_msgs::Imu> imuQueue;
std::deque<nav_msgs::Odometry> odomQueue;
std::deque<sensor_msgs::PointCloud2ConstPtr> cloudQueue;

ros::Subscriber subImu;
ros::Subscriber subOdom;
ros::Subscriber subLaserCloud

ros::Publisher pubCloudInfo;
ros::Publisher pubCornerPoints;
ros::Publisher pubSurfacePoints;


void imuHandler(const sensor_msgs::Imu::ConstPtr& imuMsg){
    sensor_msgs::Imu thisImu = imuConverter(*imuMsg);

    std::lock_guard<std::mutex> lock1(imuLock);
    imuQueue.push_back(thisImu);

    // debug IMU data
    // cout << std::setprecision(6);
    // cout << "IMU acc: " << endl;
    // cout << "x: " << thisImu.linear_acceleration.x << 
    //       ", y: " << thisImu.linear_acceleration.y << 
    //       ", z: " << thisImu.linear_acceleration.z << endl;
    // cout << "IMU gyro: " << endl;
    // cout << "x: " << thisImu.angular_velocity.x << 
    //       ", y: " << thisImu.angular_velocity.y << 
    //       ", z: " << thisImu.angular_velocity.z << endl;
    // double imuRoll, imuPitch, imuYaw;
    // tf::Quaternion orientation;
    // tf::quaternionMsgToTF(thisImu.orientation, orientation);
    // tf::Matrix3x3(orientation).getRPY(imuRoll, imuPitch, imuYaw);
    // cout << "IMU roll pitch yaw: " << endl;
    // cout << "roll: " << imuRoll << ", pitch: " << imuPitch << ", yaw: " << imuYaw << endl << endl;
}

void odometryHandler(const nav_msgs::Odometry::ConstPtr& odometryMsg){
    std::lock_guard<std::mutex> lock2(odoLock);
    odomQueue.push_back(*odometryMsg);
}

void cloudHandler(const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg){
    std::lock_guard<std::mutex> lock3(cloLock);
    cloudQueue.push_back(laserCloudMsg);
}


void laserProcessing(){
    while(1){
        if(!cloudQueue.empty()){
            //激光预处理 增加ring time Label通道供后续步骤使用
            ros::Time  startTime=ros::Time::now();

            sensor_msgs::PointCloud2ConstPtr laserCloudMsg = *cloudQueue.front()

            pcl::PointCloud<PointType> laserCloudIn;
            pcl::fromROSMsg(*cloudQueue.front(), laserCloudIn);

            pcl::PointCloud<PointXYZIRT> laserCloudOut;
            laserCloudOut=lpre.process(laserCloudIn);

            ros::Time  endTime=ros::Time::now();
            // std::cout <<  "Laser Pretreatment  Time: " <<  (endTime - startTime).toSec() << "[sec]" << std::endl;
            
            if((endTime - startTime).toSec()> 1)
                ROS_WARN("Laser Pretreatment process over 100ms");

            //激光运动畸变去除
            


            
        }
    }
    //sleep 2 ms every time
    std::chrono::milliseconds dura(2);
    std::this_thread::sleep_for(dura);
}


int main(int argc, char** argv)
{
    ros::init(argc, argv, "lis_slam");

    if(N_SCAN != 16 && N_SCAN != 32 && N_SCAN != 64)
    {
        printf("only support velodyne with 16, 32 or 64 scan line!");
        return 0;
    }

    if(useImu==true){
        subImu        = nh.subscribe<sensor_msgs::Imu>(imuTopic, 2000, &ImageProjection::imuHandler, this, ros::TransportHints().tcpNoDelay());
        subOdom       = nh.subscribe<nav_msgs::Odometry>(odomTopic+"_incremental", 2000, &ImageProjection::odometryHandler, this, ros::TransportHints().tcpNoDelay());
        subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>(pretreatmentedCloudTopic, 5, &ImageProjection::cloudHandler, this, ros::TransportHints().tcpNoDelay());
        ROS_WARN("useImu==true!");
    }else{
        subOdom       = nh.subscribe<nav_msgs::Odometry>("odometry/fusion_incremental", 2000, &ImageProjection::odometryHandler, this, ros::TransportHints().tcpNoDelay());
        subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>(pretreatmentedCloudTopic, 5, &ImageProjection::cloudHandler, this, ros::TransportHints().tcpNoDelay());
        ROS_WARN("useImu==false!");
    }

    pubCloudInfo = nh.advertise<lis_slam::cloud_info> ("lis_slam/laser_process/cloud_info", 1);

    std::thread laser_processing_process{laserProcessing};


    ROS_INFO("\033[1;32m----> Laser Processing Node Started.\033[0m");
    
    ros::MultiThreadedSpinner spinner(3);
    spinner.spin();

    return 0;
}