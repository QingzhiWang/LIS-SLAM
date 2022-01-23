// Author of  EPSC-LOAM : QZ Wang
// Email wangqingzhi27@outlook.com

#ifndef _DISTORTION_ADJUST_H_
#define _DISTORTION_ADJUST_H_

#include "utility.h"
#include "common.h"
#include "lis_slam/cloud_info.h"

class VelocityData {
  public:
    struct LinearVelocity {
      double x = 0.0;
      double y = 0.0;
      double z = 0.0;
    };

    struct AngularVelocity {
      double x = 0.0;
      double y = 0.0;
      double z = 0.0;
    };

    double time = 0.0;
    LinearVelocity linear_velocity;
    AngularVelocity angular_velocity;
  
  public:
    static bool SyncData(std::deque<VelocityData>& UnsyncedData, std::deque<VelocityData>& SyncedData, double sync_time);
    void TransformCoordinate(Eigen::Matrix4f transform_matrix);
    void NED2ENU(void);
};


class IMUData {
  public:
    class Orientation {
      public:
        double x = 0.0;
        double y = 0.0;
        double z = 0.0;
        double w = 0.0;
      
      public:
        void Normlize() {
          double norm = sqrt(pow(x, 2.0) + pow(y, 2.0) + pow(z, 2.0) + pow(w, 2.0));
          x /= norm;
          y /= norm;
          z /= norm;
          w /= norm;
        }
    };

    struct LinearAcceleration {
      double x = 0.0;
      double y = 0.0;
      double z = 0.0;
    };

    struct AngularVelocity {
      double x = 0.0;
      double y = 0.0;
      double z = 0.0;
    };

    struct AccelBias {
      double x = 0.0;
      double y = 0.0;
      double z = 0.0;
    };

    struct GyroBias {
      double x = 0.0;
      double y = 0.0;
      double z = 0.0;
    };

    double time = 0.0;

    Orientation orientation;
    
    LinearAcceleration linear_acceleration;
    AngularVelocity angular_velocity;

    AccelBias accel_bias;
    GyroBias gyro_bias;
    
  public:
    // 把四元数转换成旋转矩阵送出去
    Eigen::Matrix3f GetOrientationMatrix();
    static bool SyncData(std::deque<IMUData>& UnsyncedData, std::deque<IMUData>& SyncedData, double sync_time);
};

class DistortionAdjust : public ParamServer
{
public:
	std::mutex imu_buff_mutex_;
	std::deque<IMUData> imu_data_;
	std::deque<IMUData> new_imu_data_;

	std::mutex odom_buff_mutex_;
	std::deque<nav_msgs::Odometry> odom_data_;
	std::deque<nav_msgs::Odometry> new_odom_data_;

	std::mutex vel_buff_mutex_;
	std::deque<VelocityData> vel_data_;
	std::deque<VelocityData> new_vel_data_;
	
	pcl::PointCloud<PointXYZIRT>::Ptr current_cloud_data_;
	lis_slam::cloud_info  current_cloud_info_;
	
	std_msgs::Header cloudHeader;
	double current_cloud_time_;
	double timeScanCur;
	double timeScanEnd;

	IMUData current_imu_data_;
    VelocityData current_velocity_data_;
public:
	DistortionAdjust(){
		current_cloud_data_.reset(new pcl::PointCloud<PointXYZIRT>()); 
	}
	~DistortionAdjust(){}

	void initCloudPoint(pcl::PointCloud<PointXYZIRT>::Ptr& cloud_in, std_msgs::Header header_in);
	
	bool deskewInfo();
	void imuDeskewInfo();
    void odomDeskewInfo();
    bool gpsVelDeskewInfo();
	
	void assignCloudInfo();
public:
    void SetMotionInfo(float scan_period, VelocityData velocity_data);
    bool AdjustCloud();

private:
    inline Eigen::Matrix3f UpdateMatrix(float real_time);

private:
    float scan_period_ = 0.1;
	Eigen::Vector3f velocity_;
    Eigen::Vector3f angular_rate_;
};


#endif  // _DISTORTION_ADJUST_H_