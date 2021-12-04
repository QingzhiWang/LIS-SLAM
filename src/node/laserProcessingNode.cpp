// Author of  EPSC-LOAM : QZ Wang  
// Email wangqingzhi27@outlook.com


//local lib
#include "laserProcessing.h"

LaserProcessing lpro;

ros::Subscriber subImu;
ros::Subscriber subOdom;
ros::Subscriber subLaserCloud

ros::Publisher pubCloudInfo;

ros::Publisher pubRawPoints;
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
    cloudQueue.push_back(*laserCloudMsg);
}



double total_time =0;
int total_frame=0;

void laserProcessing(){
    while(1){
        if(!cloudQueue.empty()){

            std::chrono::time_point<std::chrono::system_clock> start, end;
            start = std::chrono::system_clock::now();

            //激光运动畸变去除
            if(lpro.distortionRemoval() == false){
                continue;
            }

            //线面特征提取
            lpro.featureExtraction();

            //更新并获取CouldInfo 
            lpro.assignCouldInfo();
            lis_slam::cloud_info cloudInfoOut =  lpro.getCloudInfo();
            
            lpro.resetParameters();

            end = std::chrono::system_clock::now();
            std::chrono::duration<float> elapsed_seconds = end - start;
            total_frame++;
            float time_temp = elapsed_seconds.count() * 1000;
            total_time+=time_temp;
            ROS_INFO("Average laser processing time %f ms \n \n", total_time/total_frame);

            //发布coudInfo
            pubCloudInfo.publish(cloudInfoOut);

            sensor_msgs::PointCloud2 laserCloudOutMsg;

            //发布cloud_deskewed
            pcl::toROSMsg(laserCloudOut.cloud_deskewed, laserCloudOutMsg);
            laserCloudOutMsg.header.stamp = laserCloudMsg->header.stamp;
            laserCloudOutMsg.header.frame_id = lidarFrame;
            pubRawPoints.publish(laserCloudOutMsg);

            //发布cloud_corner
            pcl::toROSMsg(laserCloudOut.cloud_corner, laserCloudOutMsg);
            laserCloudOutMsg.header.stamp = laserCloudMsg->header.stamp;
            laserCloudOutMsg.header.frame_id = lidarFrame;
            pubCornerPoints.publish(laserCloudOutMsg);

            //发布cloud_surface
            pcl::toROSMsg(laserCloudOut.cloud_surface, laserCloudOutMsg);
            laserCloudOutMsg.header.stamp = laserCloudMsg->header.stamp;
            laserCloudOutMsg.header.frame_id = lidarFrame;
            pubSurfacePoints.publish(laserCloudOutMsg);       
        }
    }
    //sleep 2 ms every time
    std::chrono::milliseconds dura(2);
    std::this_thread::sleep_for(dura);
}


int main(int argc, char** argv)
{
    ros::init(argc, argv, "lis_slam");

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
    pubRawPoints = nh.advertise<sensor_msgs::PointCloud2> ("lis_slam/laser_process/cloud_deskewed", 1);
    pubCornerPoints = nh.advertise<sensor_msgs::PointCloud2> ("lis_slam/laser_process/cloud_corner", 1);
    pubSurfacePoints = nh.advertise<sensor_msgs::PointCloud2> ("lis_slam/laser_process/cloud_surface", 1);

    std::thread laser_processing_process{laserProcessing};

    ROS_INFO("\033[1;32m----> Laser Processing Node Started.\033[0m");
    
    ros::MultiThreadedSpinner spinner(3);
    spinner.spin();

    return 0;
}