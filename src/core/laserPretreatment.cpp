
#include "laserPretreatment.h"

pcl::PointCloud<PointXYZIRT>::Ptr LaserPretreatment::Pretreatment(pcl::PointCloud<PointIn>::Ptr& cloudIn)
{
	
	pcl::PointCloud<PointXYZIRT>::Ptr laserCloudOut(new pcl::PointCloud<PointXYZIRT>());
	
	std::vector<int> indices;
	pcl::removeNaNFromPointCloud(*cloudIn, *cloudIn, indices);
	removeClosedPointCloud(*cloudIn, *cloudIn, lidarMinRange, lidarMaxRange);
		
	int cloudSize = cloudIn->points.size();
	float startOri = -atan2(cloudIn->points[0].y, cloudIn->points[0].x);
	float endOri = -atan2(cloudIn->points[cloudSize - 1].y, cloudIn->points[cloudSize - 1].x) + 2 * M_PI;

	if (endOri - startOri > 3 * M_PI)  endOri -= 2 * M_PI;
	else if (endOri - startOri < M_PI)  endOri += 2 * M_PI;
	
	bool halfPassed = false;
	int count = cloudSize;
	PointXYZIRT point;
	for (int i = 0; i < cloudSize; i++) 
	{
		point.x = cloudIn->points[i].x;
		point.y = cloudIn->points[i].y;
		point.z = cloudIn->points[i].z;
		point.intensity = cloudIn->points[i].i;

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
		laserCloudOut->points.push_back(point);
	}

	return laserCloudOut;
}


pcl::PointCloud<PointXYZIRT>::Ptr LaserPretreatment::Pretreatment(pcl::PointCloud<PointType>::Ptr& cloudIn)
{
	
	pcl::PointCloud<PointXYZIRT>::Ptr laserCloudOut(new pcl::PointCloud<PointXYZIRT>());
	
	std::vector<int> indices;
	pcl::removeNaNFromPointCloud(*cloudIn, *cloudIn, indices);
	removeClosedPointCloud(*cloudIn, *cloudIn, lidarMinRange, lidarMaxRange);
		
	int cloudSize = cloudIn->points.size();
	float startOri = -atan2(cloudIn->points[0].y, cloudIn->points[0].x);
	float endOri = -atan2(cloudIn->points[cloudSize - 1].y, cloudIn->points[cloudSize - 1].x) + 2 * M_PI;

	if (endOri - startOri > 3 * M_PI)  endOri -= 2 * M_PI;
	else if (endOri - startOri < M_PI)  endOri += 2 * M_PI;
	
	bool halfPassed = false;
	int count = cloudSize;
	PointXYZIRT point;
	for (int i = 0; i < cloudSize; i++) 
	{
		point.x = cloudIn->points[i].x;
		point.y = cloudIn->points[i].y;
		point.z = cloudIn->points[i].z;
		point.intensity = cloudIn->points[i].intensity;

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
		laserCloudOut->points.push_back(point);
	}

	return laserCloudOut;
}