
#include <cmath>

#include "utility.h"
#include "common.h"

#include "lis_slam/cloud_info.h"

#include "distortionAdjust.h"
#include "laserPretreatment.h"

#include "featureExtraction.h"


//************************************************************************
//对激光雷达数据预处理
//实现在原始sensor_msgs::PointCloud2增加ring 和time 通道供后续步骤使用
//************************************************************************

class DataPretreatNode : public ParamServer 
{
private:
	LaserPretreatment lpre;
	DistortionAdjust dtor;
	FeatureExtraction fext;

	ros::Subscriber subPointCloud;
	ros::Subscriber subImu;
	ros::Subscriber subVel;
	ros::Subscriber subOdom;

	ros::Publisher pubDistortedCloud;
	ros::Publisher pubPretreatmentedCloud;


	ros::Publisher pubCornerPoints;
	ros::Publisher pubSurfacePoints;
	ros::Publisher pubSharpCornerPoints;
	ros::Publisher pubSharpSurfacePoints;
	
	ros::Publisher pubCloudInfo;

	std::mutex cloudMutex;
	std::deque<sensor_msgs::PointCloud2ConstPtr> laserCloudQueue;

	double total_time = 0;
	int total_frame = 0;

public:
	DataPretreatNode(){
		if (N_SCAN != 16 && N_SCAN != 32 && N_SCAN != 64) {
			printf("only support velodyne with 16, 32 or 64 scan line!");
			ROS_BREAK();
		}

		subPointCloud = nh.subscribe<sensor_msgs::PointCloud2>(pointCloudTopic, 10, &DataPretreatNode::laserCloudInfoHandler, this, ros::TransportHints().tcpNoDelay());

		if(useGPSVel == true){
			subVel = nh.subscribe<geometry_msgs::TwistStamped>(gpsVelTopic, 2000, &DataPretreatNode::gpsVelHandler, this, ros::TransportHints().tcpNoDelay());
		}

		if (useImu == true) {
			subImu = nh.subscribe<sensor_msgs::Imu>( imuTopic, 2000, &DataPretreatNode::imuHandler, this, ros::TransportHints().tcpNoDelay());
			// subOdom = nh.subscribe<nav_msgs::Odometry>(odomTopic + "/imu", 2000, &DataPretreatNode::odometryHandler, this, ros::TransportHints().tcpNoDelay());
			ROS_WARN("useImu==true!");
		} else {
			subOdom = nh.subscribe<nav_msgs::Odometry>(odomTopic + "/lidar", 2000, &DataPretreatNode::odometryHandler, this, ros::TransportHints().tcpNoDelay());
			ROS_WARN("useImu==false!");
		}

		pubCloudInfo = nh.advertise<lis_slam::cloud_info>( "lis_slam/data/cloud_info", 100);

		pubPretreatmentedCloud = nh.advertise<sensor_msgs::PointCloud2>("lis_slam/points_pretreatmented", 100);
		pubDistortedCloud = nh.advertise<sensor_msgs::PointCloud2>( "lis_slam/data/distorted_cloud", 100);
		
		pubCornerPoints = nh.advertise<sensor_msgs::PointCloud2>( "lis_slam/feature/cloud_corner", 10);
		pubSurfacePoints = nh.advertise<sensor_msgs::PointCloud2>( "lis_slam/feature/cloud_surface", 10);
		pubSharpCornerPoints = nh.advertise<sensor_msgs::PointCloud2>( "lis_slam/feature/cloud_corner_sharp", 10);
		pubSharpSurfacePoints = nh.advertise<sensor_msgs::PointCloud2>( "lis_slam/feature/cloud_surface_sharp", 10);
		
		allocateMemory();
	}		
	
	~DataPretreatNode(){}


	void allocateMemory() 
	{

	}

	void gpsVelHandler(const geometry_msgs::TwistStampedConstPtr& twist_msg_ptr) 
	{
		geometry_msgs::TwistStamped thisVel = gpsVelConverter(*twist_msg_ptr);
		
		VelocityData velocity_data;
		velocity_data.time = thisVel.header.stamp.toSec();

		velocity_data.linear_velocity.x = thisVel.twist.linear.x;
		velocity_data.linear_velocity.y = thisVel.twist.linear.y;
		velocity_data.linear_velocity.z = thisVel.twist.linear.z;

		velocity_data.angular_velocity.x = thisVel.twist.angular.x;
		velocity_data.angular_velocity.y = thisVel.twist.angular.y;
		velocity_data.angular_velocity.z = thisVel.twist.angular.z;

		std::lock_guard<std::mutex> lock(dtor.vel_buff_mutex_);

		dtor.new_vel_data_.push_back(velocity_data);
	}

	void odometryHandler(const nav_msgs::Odometry::ConstPtr& odometryMsg) 
	{
		std::lock_guard<std::mutex> lock(dtor.odom_buff_mutex_);
		dtor.new_odom_data_.push_back(*odometryMsg);
		
		if(useGPSVel == false)
		{
			VelocityData velocity_data;
			velocity_data.time = odometryMsg->header.stamp.toSec();

			velocity_data.linear_velocity.x = odometryMsg->twist.twist.linear.x;
			velocity_data.linear_velocity.y = odometryMsg->twist.twist.linear.y;
			velocity_data.linear_velocity.z = odometryMsg->twist.twist.linear.z;

			velocity_data.angular_velocity.x = odometryMsg->twist.twist.angular.x;
			velocity_data.angular_velocity.y = odometryMsg->twist.twist.angular.y;
			velocity_data.angular_velocity.z = odometryMsg->twist.twist.angular.z;

			std::lock_guard<std::mutex> lock1(dtor.vel_buff_mutex_);

			dtor.new_vel_data_.push_back(velocity_data);
		}
	}

	void imuHandler(const sensor_msgs::Imu::ConstPtr& imuMsg) 
	{
		sensor_msgs::Imu thisImu = imuConverter(*imuMsg);
		
		IMUData imu_data;
		imu_data.time = thisImu.header.stamp.toSec();

		imu_data.linear_acceleration.x = thisImu.linear_acceleration.x;
		imu_data.linear_acceleration.y = thisImu.linear_acceleration.y;
		imu_data.linear_acceleration.z = thisImu.linear_acceleration.z;

		imu_data.angular_velocity.x = thisImu.angular_velocity.x;
		imu_data.angular_velocity.y = thisImu.angular_velocity.y;
		imu_data.angular_velocity.z = thisImu.angular_velocity.z;

		imu_data.orientation.x = thisImu.orientation.x;
		imu_data.orientation.y = thisImu.orientation.y;
		imu_data.orientation.z = thisImu.orientation.z;
		imu_data.orientation.w = thisImu.orientation.w;
    
		std::lock_guard<std::mutex> lock(dtor.imu_buff_mutex_);
		dtor.new_imu_data_.push_back(imu_data);

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

	void laserCloudInfoHandler(const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg) 
	{

		laserCloudQueue.push_back(laserCloudMsg);

	}

	void dataPretreatThread()
	{

		while (ros::ok()) 
		{
			if (laserCloudQueue.size() > 2) 
			{
				std::chrono::time_point<std::chrono::system_clock> start, end;
				start = std::chrono::system_clock::now();

				cloudMutex.lock();
				sensor_msgs::PointCloud2ConstPtr laserCloudMsg = laserCloudQueue.front();
				laserCloudQueue.pop_front();
				cloudMutex.unlock();
				
				pcl::PointCloud<PointIn>::Ptr laserCloudInTmp(new pcl::PointCloud<PointIn>());
				pcl::PointCloud<PointType>::Ptr laserCloudIn(new pcl::PointCloud<PointType>());
				pcl::PointCloud<PointXYZIRT>::Ptr laserCloudOut(new pcl::PointCloud<PointXYZIRT>());	

				if(lidarIntensity == "i"){
					pcl::fromROSMsg(*laserCloudMsg, *laserCloudInTmp);
					*laserCloudOut = *lpre.Pretreatment(laserCloudInTmp);
				}	
				else
				{
					pcl::fromROSMsg(*laserCloudMsg, *laserCloudIn);
					*laserCloudOut = *lpre.Pretreatment(laserCloudIn);
				}

				sensor_msgs::PointCloud2 laserCloudOutMsg;
				pcl::toROSMsg(*laserCloudOut, laserCloudOutMsg);
				laserCloudOutMsg.header.stamp = laserCloudMsg->header.stamp;
				laserCloudOutMsg.header.frame_id = lidarFrame;
				pubPretreatmentedCloud.publish(laserCloudOutMsg);

				dtor.initCloudPoint(laserCloudOut, laserCloudMsg->header);
				dtor.deskewInfo();
				dtor.assignCloudInfo();
				lis_slam::cloud_info cloudInfo = dtor.current_cloud_info_;
				pubDistortedCloud.publish(cloudInfo.cloud_deskewed);
				
				fext.initCloudInfo(cloudInfo);
				fext.featureExtraction(); //线面特征提取
				fext.assignCouldInfo(); //更新并获取CouldInfo
				lis_slam::cloud_info cloudInfoOut = fext.getCloudInfo();
				fext.resetParameters();
				
				pubCloudInfo.publish(cloudInfoOut);
				
				pubCornerPoints.publish(cloudInfoOut.cloud_corner);
				pubSurfacePoints.publish(cloudInfoOut.cloud_surface);
				pubSharpCornerPoints.publish(cloudInfoOut.cloud_corner_sharp);
				pubSharpSurfacePoints.publish(cloudInfoOut.cloud_surface_sharp);

				end = std::chrono::system_clock::now();
				std::chrono::duration<float> elapsed_seconds = end - start;
				total_frame++;
				float time_temp = elapsed_seconds.count() * 1000;
				total_time += time_temp;
				ROS_INFO("Average Data Pretreat time %f ms", total_time / total_frame);

			}
			
			ros::spinOnce();
			// sleep 2 ms every time
			std::chrono::milliseconds dura(2);
			std::this_thread::sleep_for(dura);
		}
		
	}

};


int main(int argc, char** argv) 
{
	ros::init(argc, argv, "epsc_lio");

	DataPretreatNode DPN;

	std::thread process(&DataPretreatNode::dataPretreatThread, &DPN);

	ROS_INFO("\033[1;32m----> Datd Pretreat Node Started.\033[0m");

	ros::MultiThreadedSpinner spinner(3);
	spinner.spin();

	process.join();

	return 0;
}
