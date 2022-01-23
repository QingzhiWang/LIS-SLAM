// Author of  EPSC-LOAM : QZ Wang
// Email wangqingzhi27@outlook.com
#include "utility.h"
#include "common.h"

#include "featureExtraction.h"

#include "lis_slam/cloud_info.h"


class FeatureExtractionNode : public ParamServer 
{
private:
	FeatureExtraction fext;

	ros::Subscriber subCloudInfo;

	ros::Publisher pubCloudInfo;

	ros::Publisher pubRawPoints;
	ros::Publisher pubCornerPoints;
	ros::Publisher pubSurfacePoints;
	ros::Publisher pubSharpCornerPoints;
	ros::Publisher pubSharpSurfacePoints;

	double total_time = 0;
	int total_frame = 0;
public:
	FeatureExtractionNode()
	{
		subCloudInfo = nh.subscribe<lis_slam::cloud_info>( "lis_slam/data/cloud_info", 5, &FeatureExtractionNode::cloudHandler, this);

		pubCloudInfo = nh.advertise<lis_slam::cloud_info>( "lis_slam/feature/cloud_info", 10);
		pubRawPoints = nh.advertise<sensor_msgs::PointCloud2>( "lis_slam/feature/cloud_deskewed", 10);
		pubCornerPoints = nh.advertise<sensor_msgs::PointCloud2>( "lis_slam/feature/cloud_corner", 10);
		pubSurfacePoints = nh.advertise<sensor_msgs::PointCloud2>( "lis_slam/feature/cloud_surface", 10);
		pubSharpCornerPoints = nh.advertise<sensor_msgs::PointCloud2>( "lis_slam/feature/cloud_corner_sharp", 10);
		pubSharpSurfacePoints = nh.advertise<sensor_msgs::PointCloud2>( "lis_slam/feature/cloud_surface_sharp", 10);
	}

	~FeatureExtractionNode() {}

	void cloudHandler(lis_slam::cloud_info laserCloudMsg);

};

void FeatureExtractionNode::cloudHandler(lis_slam::cloud_info laserCloudMsg)
{
	
	std::chrono::time_point<std::chrono::system_clock> start, end;
	start = std::chrono::system_clock::now();

	fext.initCloudInfo(laserCloudMsg);
	//线面特征提取
	fext.featureExtraction();

	//更新并获取CouldInfo
	fext.assignCouldInfo();
	lis_slam::cloud_info cloudInfoOut = fext.getCloudInfo();

	fext.resetParameters();

	end = std::chrono::system_clock::now();
	std::chrono::duration<float> elapsed_seconds = end - start;
	total_frame++;
	float time_temp = elapsed_seconds.count() * 1000;
	total_time += time_temp;
	ROS_INFO("Average Feature Extraction time %f ms", total_time / total_frame);

	//发布coudInfo
	pubCloudInfo.publish(cloudInfoOut);

	pubRawPoints.publish(cloudInfoOut.cloud_deskewed);
	pubCornerPoints.publish(cloudInfoOut.cloud_corner);
	pubSurfacePoints.publish(cloudInfoOut.cloud_surface);
	pubSharpCornerPoints.publish(cloudInfoOut.cloud_corner_sharp);
	pubSharpSurfacePoints.publish(cloudInfoOut.cloud_surface_sharp);

}



int main(int argc, char** argv) 
{
	ros::init(argc, argv, "lis_slam");

	FeatureExtractionNode FEN;

	ROS_INFO("\033[1;32m----> Feature Extraction Node Started.\033[0m");

	ros::spin();

	return 0;
}