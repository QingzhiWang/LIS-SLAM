// Author of  EPSC-LOAM : QZ Wang
// Email wangqingzhi27@outlook.com

#include <cmath>

#include "utility.h"
#include "common.h"

#include "lis_slam/cloud_info.h"


using std::atan2;
using std::cos;
using std::sin;

const double scanPeriod = 0.1;

struct PointIn {
    PCL_ADD_POINT4D;
	float i;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT(
    PointIn,
    (float, x, x)(float, y, y)(float, z, z)(float, i, i))


//************************************************************************
//对激光雷达数据预处理
//实现在原始sensor_msgs::PointCloud2增加ring 和time 通道供后续步骤使用
//************************************************************************
class laserPretreatmentNode : public ParamServer 
{
 public:
	ros::Publisher pubPretreatmentedCloud;
	ros::Subscriber subPointCloud;

	double total_time = 0;
	int total_frame = 0;

	laserPretreatmentNode() {
		if (N_SCAN != 16 && N_SCAN != 32 && N_SCAN != 64) {
			printf("only support velodyne with 16, 32 or 64 scan line!");
			ROS_BREAK();
		}

		subPointCloud = nh.subscribe<sensor_msgs::PointCloud2>(pointCloudTopic, 10, &laserPretreatmentNode::laserCloudInfoHandler, this, ros::TransportHints().tcpNoDelay());
		
		pubPretreatmentedCloud = nh.advertise<sensor_msgs::PointCloud2>("lis_slam/points_pretreatmented", 100);

		allocateMemory();
	}

	void allocateMemory() 
	{
		// laserCloudRaw.reset(new pcl::PointCloud<PointType>());
		// laserCloudRawDS.reset(new pcl::PointCloud<PointType>());
	}


	void laserCloudInfoHandler(const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg) 
	{
		std::chrono::time_point<std::chrono::system_clock> start, end;
		start = std::chrono::system_clock::now();

		ros::Time startTime = ros::Time::now();
		
		pcl::PointCloud<PointIn> laserCloudInTmp;
		pcl::PointCloud<PointType> laserCloudIn;
		pcl::PointCloud<PointXYZIRT> laserCloudOut;

		if(lidarIntensity == "i"){
			pcl::fromROSMsg(*laserCloudMsg, laserCloudInTmp);
			
			std::vector<int> indices;
			pcl::removeNaNFromPointCloud(laserCloudInTmp, laserCloudInTmp, indices);
			removeClosedPointCloud(laserCloudInTmp, laserCloudInTmp, lidarMinRange, lidarMaxRange);
			
			int cloudSize = laserCloudInTmp.points.size();
			float startOri = -atan2(laserCloudInTmp.points[0].y, laserCloudInTmp.points[0].x);
			float endOri = -atan2(laserCloudInTmp.points[cloudSize - 1].y, laserCloudInTmp.points[cloudSize - 1].x) + 2 * M_PI;

			if (endOri - startOri > 3 * M_PI)  endOri -= 2 * M_PI;
			else if (endOri - startOri < M_PI)  endOri += 2 * M_PI;
			
			bool halfPassed = false;
			int count = cloudSize;
			PointXYZIRT point;
			for (int i = 0; i < cloudSize; i++) 
			{
				point.x = laserCloudInTmp.points[i].x;
				point.y = laserCloudInTmp.points[i].y;
				point.z = laserCloudInTmp.points[i].z;
				point.intensity = laserCloudInTmp.points[i].i;

				float angle = atan(point.z / sqrt(point.x * point.x + point.y * point.y)) * 180 / M_PI;
				int scanID = 0;

				if (N_SCAN == 16) {
					scanID = int((angle + 15) / 2 + 0.5);
					if (scanID > (N_SCAN - 1) || scanID < 0) {
						count--;
						continue;
					}
				} 
				else if (N_SCAN == 32) {
					scanID = int((angle + 92.0 / 3.0) * 3.0 / 4.0);
					if (scanID > (N_SCAN - 1) || scanID < 0) {
						count--;
						continue;
					}
				} 
				else if (N_SCAN == 64) {
					if (angle >= -8.83) scanID = int((2 - angle) * 3.0 + 0.5);
					else scanID = N_SCAN / 2 + int((-8.83 - angle) * 2.0 + 0.5);

					// use [0 50]  > 50 remove outlies
					if (angle > 2 || angle < -24.33 || scanID > 50 || scanID < 0) {
						count--;
						continue;
					}
				} 
				else {
					printf("wrong scan number\n");
					ROS_BREAK();
				}

				float ori = -atan2(point.y, point.x);
				if (!halfPassed) {
					if (ori < startOri - M_PI / 2) ori += 2 * M_PI;
					else if (ori > startOri + M_PI * 3 / 2)  ori -= 2 * M_PI;
					
					if (ori - startOri > M_PI) halfPassed = true;
				} else {
					ori += 2 * M_PI;
					if (ori < endOri - M_PI * 3 / 2)  ori += 2 * M_PI;
					else if (ori > endOri + M_PI / 2)  ori -= 2 * M_PI;
					
				}
				float relTime = (ori - startOri) / (endOri - startOri);
				point.ring = scanID;
				point.time = scanPeriod * relTime;
				laserCloudOut.points.push_back(point);
			}
		}
		else
		{
			pcl::fromROSMsg(*laserCloudMsg, laserCloudIn);

			std::vector<int> indices;
			pcl::removeNaNFromPointCloud(laserCloudIn, laserCloudIn, indices);
			removeClosedPointCloud(laserCloudIn, laserCloudIn, lidarMinRange, lidarMaxRange);
			
			int cloudSize = laserCloudIn.points.size();
			float startOri = -atan2(laserCloudIn.points[0].y, laserCloudIn.points[0].x);
			float endOri = -atan2(laserCloudIn.points[cloudSize - 1].y, laserCloudIn.points[cloudSize - 1].x) + 2 * M_PI;

			if (endOri - startOri > 3 * M_PI)  endOri -= 2 * M_PI;
			else if (endOri - startOri < M_PI)  endOri += 2 * M_PI;
			
			bool halfPassed = false;
			int count = cloudSize;
			PointXYZIRT point;
			for (int i = 0; i < cloudSize; i++) 
			{
				point.x = laserCloudIn.points[i].x;
				point.y = laserCloudIn.points[i].y;
				point.z = laserCloudIn.points[i].z;
				point.intensity = laserCloudIn.points[i].intensity;

				float angle = atan(point.z / sqrt(point.x * point.x + point.y * point.y)) * 180 / M_PI;
				int scanID = 0;

				if (N_SCAN == 16) {
					scanID = int((angle + 15) / 2 + 0.5);
					if (scanID > (N_SCAN - 1) || scanID < 0) {
						count--;
						continue;
					}
				} 
				else if (N_SCAN == 32) {
					scanID = int((angle + 92.0 / 3.0) * 3.0 / 4.0);
					if (scanID > (N_SCAN - 1) || scanID < 0) {
						count--;
						continue;
					}
				} 
				else if (N_SCAN == 64) {
					if (angle >= -8.83) scanID = int((2 - angle) * 3.0 + 0.5);
					else scanID = N_SCAN / 2 + int((-8.83 - angle) * 2.0 + 0.5);

					// use [0 50]  > 50 remove outlies
					if (angle > 2 || angle < -24.33 || scanID > 50 || scanID < 0) {
						count--;
						continue;
					}
				} 
				else {
					printf("wrong scan number\n");
					ROS_BREAK();
				}

				float ori = -atan2(point.y, point.x);
				if (!halfPassed) {
					if (ori < startOri - M_PI / 2) ori += 2 * M_PI;
					else if (ori > startOri + M_PI * 3 / 2)  ori -= 2 * M_PI;
					
					if (ori - startOri > M_PI) halfPassed = true;
				} else {
					ori += 2 * M_PI;
					if (ori < endOri - M_PI * 3 / 2)  ori += 2 * M_PI;
					else if (ori > endOri + M_PI / 2)  ori -= 2 * M_PI;
					
				}
				float relTime = (ori - startOri) / (endOri - startOri);
				point.ring = scanID;
				point.time = scanPeriod * relTime;
				laserCloudOut.points.push_back(point);
			}
		}


		ros::Time endTime = ros::Time::now();
		// std::cout <<  "Laser Pretreatment  Time: " <<  (endTime -
		// startTime).toSec() << "[sec]" << std::endl;

		end = std::chrono::system_clock::now();
		std::chrono::duration<float> elapsed_seconds = end - start;
		total_frame++;
		float time_temp = elapsed_seconds.count() * 1000;
		total_time += time_temp;
		ROS_INFO("Average laser Pretreatment time %f ms", total_time / total_frame);

		if ((endTime - startTime).toSec() > 1)
		ROS_WARN("Laser Pretreatment process over 100ms");

		sensor_msgs::PointCloud2 laserCloudOutMsg;
		pcl::toROSMsg(laserCloudOut, laserCloudOutMsg);
		laserCloudOutMsg.header.stamp = laserCloudMsg->header.stamp;
		laserCloudOutMsg.header.frame_id = lidarFrame;
		pubPretreatmentedCloud.publish(laserCloudOutMsg);
	}

	template <typename PointT>
	void removeClosedPointCloud(const pcl::PointCloud<PointT>& cloud_in,
								pcl::PointCloud<PointT>& cloud_out,
								float minthres, float maxthres) 
	{
		if (&cloud_in != &cloud_out) 
		{
			cloud_out.header = cloud_in.header;
			cloud_out.points.resize(cloud_in.points.size());
		}

		size_t j = 0;

		for (size_t i = 0; i < cloud_in.points.size(); ++i) 
		{
			float thisRange = cloud_in.points[i].x * cloud_in.points[i].x + cloud_in.points[i].y * cloud_in.points[i].y + cloud_in.points[i].z * cloud_in.points[i].z;
			if (thisRange < minthres * minthres) continue;
			if (thisRange > maxthres * maxthres) continue;
			cloud_out.points[j] = cloud_in.points[i];
			j++;
		}

		if (j != cloud_in.points.size()) {
			cloud_out.points.resize(j);
		}

		cloud_out.height = 1;
		cloud_out.width = static_cast<uint32_t>(j);
		cloud_out.is_dense = true;
	}
};

int main(int argc, char** argv) 
{
	ros::init(argc, argv, "epsc_lio");

	laserPretreatmentNode LP;

	ROS_INFO("\033[1;32m----> Laser Pretreatment Started.\033[0m");

	ros::spin();

	return 0;
}