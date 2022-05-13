
#include "distortionAdjust.h"

bool VelocityData::SyncData(std::deque<VelocityData>& UnsyncedData, std::deque<VelocityData>& SyncedData, double sync_time) 
{
    // 传感器数据按时间序列排列，在传感器数据中为同步的时间点找到合适的时间位置
    // 即找到与同步时间相邻的左右两个数据
    // 需要注意的是，如果左右相邻数据有一个离同步时间差值比较大，则说明数据有丢失，时间离得太远不适合做差值
	while (UnsyncedData.size() >= 2) {
		// ROS_WARN("UnsyncedData.front().time: %f , sync_time: %f ", UnsyncedData.front().time, sync_time);

        if (UnsyncedData.front().time > sync_time){
			// ROS_WARN("UnsyncedData.front().time > sync_time");
            return false;
		}
        if (UnsyncedData.at(1).time < sync_time) {
			// ROS_WARN("UnsyncedData.at(1).time: %f , sync_time: %f ", UnsyncedData.at(1).time, sync_time);
			// ROS_WARN("UnsyncedData.at(1).time < sync_time");
            UnsyncedData.pop_front();
            continue;
        }
        if (sync_time - UnsyncedData.front().time > 0.2) {
			// ROS_WARN("sync_time - UnsyncedData.front().time > 0.2");
            UnsyncedData.pop_front();
            return false;
        }
        if (UnsyncedData.at(1).time - sync_time > 0.2) {
			// ROS_WARN("UnsyncedData.at(1).time - sync_time > 0.2");
            UnsyncedData.pop_front();
            return false;
        }
        break;
    }
    if (UnsyncedData.size() < 2){
		// ROS_WARN("UnsyncedData.size() < 2");
        return false;
	}

    VelocityData front_data = UnsyncedData.at(0);
    VelocityData back_data = UnsyncedData.at(1);
    VelocityData synced_data;

    double front_scale = (back_data.time - sync_time) / (back_data.time - front_data.time);
    double back_scale = (sync_time - front_data.time) / (back_data.time - front_data.time);
    synced_data.time = sync_time;
    synced_data.linear_velocity.x = front_data.linear_velocity.x * front_scale + back_data.linear_velocity.x * back_scale;
    synced_data.linear_velocity.y = front_data.linear_velocity.y * front_scale + back_data.linear_velocity.y * back_scale;
    synced_data.linear_velocity.z = front_data.linear_velocity.z * front_scale + back_data.linear_velocity.z * back_scale;
    synced_data.angular_velocity.x = front_data.angular_velocity.x * front_scale + back_data.angular_velocity.x * back_scale;
    synced_data.angular_velocity.y = front_data.angular_velocity.y * front_scale + back_data.angular_velocity.y * back_scale;
    synced_data.angular_velocity.z = front_data.angular_velocity.z * front_scale + back_data.angular_velocity.z * back_scale;

    SyncedData.push_back(synced_data);

    return true;
}

void VelocityData::TransformCoordinate(Eigen::Matrix4f transform_matrix) 
{
    Eigen::Matrix4d matrix = transform_matrix.cast<double>();

    Eigen::Matrix3d R = matrix.block<3,3>(0,0);
    Eigen::Vector3d t = matrix.block<3,1>(0,3);

    // get angular & linear velocities in IMU frame:
    Eigen::Vector3d w(angular_velocity.x, angular_velocity.y, angular_velocity.z);
    Eigen::Vector3d v(linear_velocity.x, linear_velocity.y, linear_velocity.z);

    // a. first, add velocity component generated by rotation:
    Eigen::Vector3d delta_v;
    delta_v(0) = w(1) * t(2) - w(2) * t(1);
    delta_v(1) = w(2) * t(0) - w(0) * t(2);
    delta_v(2) = w(0) * t(1) - w(1) * t(0);
    v += delta_v;

    // b. transform velocities in IMU frame to lidar frame:
    w = R.transpose() * w;
    v = R.transpose() * v;

    // finally:
    angular_velocity.x = w(0);
    angular_velocity.y = w(1);
    angular_velocity.z = w(2);
    linear_velocity.x = v(0);
    linear_velocity.y = v(1);
    linear_velocity.z = v(2);
}

void VelocityData::NED2ENU(void) 
{
    LinearVelocity linear_velocity_enu;

    linear_velocity_enu.x = +linear_velocity.y;
    linear_velocity_enu.y = +linear_velocity.x;
    linear_velocity_enu.z = -linear_velocity.z;

    linear_velocity.x = linear_velocity_enu.x;
    linear_velocity.y = linear_velocity_enu.y;
    linear_velocity.z = linear_velocity_enu.z;

    AngularVelocity angular_velocity_enu;

    angular_velocity_enu.x = +angular_velocity.y;
    angular_velocity_enu.y = +angular_velocity.x;
    angular_velocity_enu.z = -angular_velocity.z;

    angular_velocity.x = angular_velocity_enu.x;
    angular_velocity.y = angular_velocity_enu.y;
    angular_velocity.z = angular_velocity_enu.z;
}





Eigen::Matrix3f IMUData::GetOrientationMatrix() 
{
    Eigen::Quaterniond q(orientation.w, orientation.x, orientation.y, orientation.z);
    Eigen::Matrix3f matrix = q.matrix().cast<float>();

    return matrix;
}

bool IMUData::SyncData(std::deque<IMUData>& UnsyncedData, std::deque<IMUData>& SyncedData, double sync_time) 
{
    // 传感器数据按时间序列排列，在传感器数据中为同步的时间点找到合适的时间位置
    // 即找到与同步时间相邻的左右两个数据
    // 需要注意的是，如果左右相邻数据有一个离同步时间差值比较大，则说明数据有丢失，时间离得太远不适合做差值
    while (UnsyncedData.size() >= 2) {
        // UnsyncedData.front().time should be <= sync_time:
        if (UnsyncedData.front().time > sync_time) 
            return false;
        // sync_time should be <= UnsyncedData.at(1).time:
        if (UnsyncedData.at(1).time < sync_time) {
            UnsyncedData.pop_front();
            continue;
        }

        // sync_time - UnsyncedData.front().time should be <= 0.2:
        if (sync_time - UnsyncedData.front().time > 0.2) {
            UnsyncedData.pop_front();
            return false;
        }
        // UnsyncedData.at(1).time - sync_time should be <= 0.2
        if (UnsyncedData.at(1).time - sync_time > 0.2) {
            return false;
        }
        break;
    }
    if (UnsyncedData.size() < 2)
        return false;

    IMUData front_data = UnsyncedData.at(0);
    IMUData back_data = UnsyncedData.at(1);
    IMUData synced_data;

    double front_scale = (back_data.time - sync_time) / (back_data.time - front_data.time);
    double back_scale = (sync_time - front_data.time) / (back_data.time - front_data.time);
    synced_data.time = sync_time;
    synced_data.linear_acceleration.x = front_data.linear_acceleration.x * front_scale + back_data.linear_acceleration.x * back_scale;
    synced_data.linear_acceleration.y = front_data.linear_acceleration.y * front_scale + back_data.linear_acceleration.y * back_scale;
    synced_data.linear_acceleration.z = front_data.linear_acceleration.z * front_scale + back_data.linear_acceleration.z * back_scale;
    synced_data.angular_velocity.x = front_data.angular_velocity.x * front_scale + back_data.angular_velocity.x * back_scale;
    synced_data.angular_velocity.y = front_data.angular_velocity.y * front_scale + back_data.angular_velocity.y * back_scale;
    synced_data.angular_velocity.z = front_data.angular_velocity.z * front_scale + back_data.angular_velocity.z * back_scale;
    // 四元数插值有线性插值和球面插值，球面插值更准确，但是两个四元数差别不大是，二者精度相当
    // 由于是对相邻两时刻姿态插值，姿态差比较小，所以可以用线性插值
    synced_data.orientation.x = front_data.orientation.x * front_scale + back_data.orientation.x * back_scale;
    synced_data.orientation.y = front_data.orientation.y * front_scale + back_data.orientation.y * back_scale;
    synced_data.orientation.z = front_data.orientation.z * front_scale + back_data.orientation.z * back_scale;
    synced_data.orientation.w = front_data.orientation.w * front_scale + back_data.orientation.w * back_scale;
    // 线性插值之后要归一化
    synced_data.orientation.Normlize();

    SyncedData.push_back(synced_data);

    return true;
}






void DistortionAdjust::initCloudPoint(pcl::PointCloud<PointXYZIRT>::Ptr& cloud_in, std_msgs::Header header_in)
{
	current_cloud_data_->points.clear();
	*current_cloud_data_ = *cloud_in;
	// check dense flag
	if (current_cloud_data_->is_dense == false) {
		ROS_ERROR("Point cloud is not in dense format, please remove NaN points first!");
		ros::shutdown();
	}
	cloudHeader = header_in;
	current_cloud_time_ = cloudHeader.stamp.toSec();

	// get timestamp
	// timeScanCur = current_cloud_time_ - scan_period_ / 2.0;;
	timeScanCur = current_cloud_time_;
	timeScanEnd = timeScanCur + current_cloud_data_->points.back().time;  // Velodyne
	
}

void DistortionAdjust::assignCloudInfo()
{
	current_cloud_info_.header = cloudHeader;

	sensor_msgs::PointCloud2 tempCloud;

	// ROS_WARN("current_cloud_data_.size: %d", current_cloud_data_->size());

	pcl::toROSMsg(*current_cloud_data_, tempCloud);
	tempCloud.header.stamp = cloudHeader.stamp;
	tempCloud.header.frame_id = lidarFrame;
	current_cloud_info_.cloud_deskewed = tempCloud;

}

bool DistortionAdjust::deskewInfo()
{
	std::lock_guard<std::mutex> lock1(imu_buff_mutex_);
	std::lock_guard<std::mutex> lock2(odom_buff_mutex_);
	std::lock_guard<std::mutex> lock3(vel_buff_mutex_);

	if (useImu == true) {
		// make sure IMU data available for the scan
		// if (imu_data_.empty() || imu_data_.front().time > timeScanCur || imu_data_.front().time < timeScanEnd) 
		if (new_imu_data_.empty()) 
		{
			// ROS_WARN("imu_data_.front().time: %f , timeScanCur: %f , timeScanEnd: %f ", imu_data_.front().time, timeScanCur, timeScanEnd);
			ROS_WARN("imu_data size : %d Waiting for IMU data ...", imu_data_.size());
			return false;
		}

		imuDeskewInfo();
	}

	odomDeskewInfo();

	if(!gpsVelDeskewInfo()) {
		ROS_WARN("gpsVelDeskewInfo() == false");
		return false;
	}
	// motion compensation for lidar measurements:
	current_velocity_data_ = vel_data_.front();
	SetMotionInfo(0.1, current_velocity_data_);
	AdjustCloud();

	return true;
}

void DistortionAdjust::imuDeskewInfo()
{
    static std::deque<IMUData> unsynced_imu_;
	
    // pipe all available measurements to output buffer:
    if (new_imu_data_.size() > 0) {
        unsynced_imu_.insert(unsynced_imu_.end(), new_imu_data_.begin(), new_imu_data_.end());
        new_imu_data_.clear();
    }

    bool valid_imu = IMUData::SyncData(unsynced_imu_, imu_data_, current_cloud_time_);

	
	current_cloud_info_.imuAvailable = false;

	if(valid_imu == true)
	{
		double imuRoll, imuPitch, imuYaw;
		tf::Quaternion orientation(imu_data_.front().orientation.x, imu_data_.front().orientation.y, 
								   imu_data_.front().orientation.z, imu_data_.front().orientation.w);
		tf::Matrix3x3(orientation).getRPY(imuRoll, imuPitch, imuYaw);

		current_cloud_info_.imuRollInit = imuRoll;
		current_cloud_info_.imuPitchInit = imuPitch;
		current_cloud_info_.imuYawInit = imuYaw;
		current_cloud_info_.imuAvailable = true;

		ROS_WARN("imuDeskewInfo :  current_cloud_info_.imuAvailable = true!");

	}else{
		ROS_WARN("valid_imu == false");
	}

}

void DistortionAdjust::odomDeskewInfo()
{
	current_cloud_info_.odomAvailable = false;

	if(useImu == false && useGPSVel == false){

		if (odom_data_.size() < 2) {
			// ROS_WARN("odomDeskewInfo :  odomQueue.empty()!");
			return;
		}

		// get start odometry at the beinning of the scan
		static nav_msgs::Odometry firstOdomMsg, nextOdomMsg;
		
		for (int i = 0; i < (int)odom_data_.size() - 1; ++i) 
		{
			firstOdomMsg = odom_data_[i];
			nextOdomMsg = odom_data_[i+1];

			if (ROS_TIME(&firstOdomMsg) < timeScanCur)
				continue;
			else
				break;
		}

		tf::Quaternion orientation;
		double fristRoll, firstPitch, firstYaw;
		double nextRoll, nextPitch, nextYaw;
		tf::quaternionMsgToTF(firstOdomMsg.pose.pose.orientation, orientation);
		tf::Matrix3x3(orientation).getRPY(fristRoll, firstPitch, firstYaw);
		
		tf::quaternionMsgToTF(nextOdomMsg.pose.pose.orientation, orientation);
		tf::Matrix3x3(orientation).getRPY(nextRoll, nextPitch, nextYaw);
		
		double diff_x = nextOdomMsg.pose.pose.position.x - firstOdomMsg.pose.pose.position.x;
		double diff_y = nextOdomMsg.pose.pose.position.y - firstOdomMsg.pose.pose.position.y;
		double diff_z = nextOdomMsg.pose.pose.position.z - firstOdomMsg.pose.pose.position.z;
		double diff_time = nextOdomMsg.header.stamp.toSec() - firstOdomMsg.header.stamp.toSec();
		
		VelocityData velocity_data;
		velocity_data.time = firstOdomMsg.header.stamp.toSec();

		velocity_data.linear_velocity.x = diff_x / diff_time;
		velocity_data.linear_velocity.y = diff_y / diff_time;
		velocity_data.linear_velocity.z = diff_z / diff_time;

		velocity_data.angular_velocity.x = (nextRoll - fristRoll) / diff_time;
		velocity_data.angular_velocity.y = (nextPitch - firstPitch) / diff_time;
		velocity_data.angular_velocity.z = (nextYaw - firstYaw) / diff_time;

		std::lock_guard<std::mutex> lock1(vel_buff_mutex_);

		vel_data_.push_back(velocity_data);
	}

	while (!odom_data_.empty()) 
	{
		if (odom_data_.front().header.stamp.toSec() < timeScanCur - 0.01)
			odom_data_.pop_front();
		else
			break;
	}

	if (odom_data_.empty()) {
		// ROS_WARN("odomDeskewInfo :  odomQueue.empty()!");
		return;
	}

	if (odom_data_.front().header.stamp.toSec() > timeScanCur) {
		//  ROS_WARN("odomDeskewInfo :  odomQueue.front().header.stamp.toSec() > timeScanCur!");
		return;
	}

	// get start odometry at the beinning of the scan
	nav_msgs::Odometry startOdomMsg;
	
	for (int i = 0; i < (int)odom_data_.size(); ++i) 
	{
		startOdomMsg = odom_data_[i];

		if (ROS_TIME(&startOdomMsg) < timeScanCur)
			continue;
		else
			break;
	}

	tf::Quaternion orientation;
	tf::quaternionMsgToTF(startOdomMsg.pose.pose.orientation, orientation);

	double roll, pitch, yaw;
	tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);

	// Initial guess used in mapOptimization
	current_cloud_info_.initialGuessX = startOdomMsg.pose.pose.position.x;
	current_cloud_info_.initialGuessY = startOdomMsg.pose.pose.position.y;
	current_cloud_info_.initialGuessZ = startOdomMsg.pose.pose.position.z;
	current_cloud_info_.initialGuessRoll = roll;
	current_cloud_info_.initialGuessPitch = pitch;
	current_cloud_info_.initialGuessYaw = yaw;

	current_cloud_info_.odomAvailable = true;
	ROS_WARN("odomDeskewInfo :  current_cloud_info_.odomAvailable = true!");

}

bool DistortionAdjust::gpsVelDeskewInfo()
{
    static std::deque<VelocityData> unsynced_velocity_;

    // pipe all available measurements to output buffer:
    if (new_vel_data_.size() > 0) {
        unsynced_velocity_.insert(unsynced_velocity_.end(), new_vel_data_.begin(), new_vel_data_.end());
        new_vel_data_.clear();
    }

	bool valid_velocity = VelocityData::SyncData(unsynced_velocity_, vel_data_, current_cloud_time_);
	
	return valid_velocity;
}






void DistortionAdjust::SetMotionInfo(float scan_period, VelocityData velocity_data) 
{
    scan_period_ = scan_period;
    velocity_ << velocity_data.linear_velocity.x, velocity_data.linear_velocity.y, velocity_data.linear_velocity.z;
    angular_rate_ << velocity_data.angular_velocity.x, velocity_data.angular_velocity.y, velocity_data.angular_velocity.z;
}

bool DistortionAdjust::AdjustCloud() 
{
    pcl::PointCloud<PointXYZIRT>::Ptr origin_cloud_ptr(new pcl::PointCloud<PointXYZIRT>(*current_cloud_data_));
    current_cloud_data_.reset(new pcl::PointCloud<PointXYZIRT>());

    // float orientation_space = 2.0 * M_PI;
    // float delete_space = 5.0 * M_PI / 180.0;
    // float start_orientation = atan2(origin_cloud_ptr->points[0].y, origin_cloud_ptr->points[0].x);

    // Eigen::AngleAxisf t_V(start_orientation, Eigen::Vector3f::UnitZ());
    // Eigen::Matrix3f rotate_matrix = t_V.matrix();
    // Eigen::Matrix4f transform_matrix = Eigen::Matrix4f::Identity();
    // transform_matrix.block<3,3>(0,0) = rotate_matrix.inverse();
    // pcl::transformPointCloud(*origin_cloud_ptr, *origin_cloud_ptr, transform_matrix);

    // velocity_ = rotate_matrix * velocity_;
    // angular_rate_ = rotate_matrix * angular_rate_;

    for (size_t point_index = 1; point_index < origin_cloud_ptr->points.size(); ++point_index) {
        // float orientation = atan2(origin_cloud_ptr->points[point_index].y, origin_cloud_ptr->points[point_index].x);
        // if (orientation < 0.0)
        //     orientation += 2.0 * M_PI;
        
        // if (orientation < delete_space || 2.0 * M_PI - orientation < delete_space)
        //     continue;

        // float real_time = fabs(orientation) / orientation_space * scan_period_ - scan_period_ / 2.0;
        
		float real_time = origin_cloud_ptr->points[point_index].time - scan_period_ / 2.0;

        Eigen::Vector3f origin_point(origin_cloud_ptr->points[point_index].x,
                                     origin_cloud_ptr->points[point_index].y,
                                     origin_cloud_ptr->points[point_index].z);

        Eigen::Matrix3f current_matrix = UpdateMatrix(real_time);
        Eigen::Vector3f rotated_point = current_matrix * origin_point;
        Eigen::Vector3f adjusted_point = rotated_point + velocity_ * real_time;
        PointXYZIRT point;
        point.x = adjusted_point(0);
        point.y = adjusted_point(1);
        point.z = adjusted_point(2);
        point.intensity = origin_cloud_ptr->points[point_index].intensity;
        point.ring = origin_cloud_ptr->points[point_index].ring;
        point.time = origin_cloud_ptr->points[point_index].time;

        current_cloud_data_->points.push_back(point);
    }

    // pcl::transformPointCloud(*current_cloud_data_, *current_cloud_data_, transform_matrix.inverse());
    return true;
}

Eigen::Matrix3f DistortionAdjust::UpdateMatrix(float real_time) 
{
    Eigen::Vector3f angle = angular_rate_ * real_time;
    Eigen::AngleAxisf t_Vz(angle(2), Eigen::Vector3f::UnitZ());
    Eigen::AngleAxisf t_Vy(angle(1), Eigen::Vector3f::UnitY());
    Eigen::AngleAxisf t_Vx(angle(0), Eigen::Vector3f::UnitX());
    Eigen::AngleAxisf t_V;
    t_V = t_Vz * t_Vy * t_Vx;
    return t_V.matrix();
}