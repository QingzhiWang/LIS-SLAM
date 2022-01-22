// This code partly draws on  lio_sam
// Author of  EPSC-LOAM : QZ Wang
// Email wangqingzhi27@outlook.com

#include "featureExtraction.h"


void FeatureExtraction ::allocateMemory() 
{
	laserCloudIn.reset(new pcl::PointCloud<PointXYZIRT>());

	cornerCloud.reset(new pcl::PointCloud<PointXYZIRT>());
	surfaceCloud.reset(new pcl::PointCloud<PointXYZIRT>());
	sharpCornerCloud.reset(new pcl::PointCloud<PointXYZIRT>());
	SharpSurfaceCloud.reset(new pcl::PointCloud<PointXYZIRT>());

	startRingIndex = new int32_t[N_SCAN];
	endRingIndex = new int32_t[N_SCAN];

	pointColInd = new int32_t[N_SCAN * Horizon_SCAN];
	pointRange = new float[N_SCAN * Horizon_SCAN];

	cloudSmoothness.resize(N_SCAN * Horizon_SCAN);
	cloudCurvature = new float[N_SCAN * Horizon_SCAN];
	cloudNeighborPicked = new int[N_SCAN * Horizon_SCAN];
	cloudLabel = new int[N_SCAN * Horizon_SCAN];

}

/****************************************
 *
 *****************************************/
void FeatureExtraction ::resetParameters() 
{
	laserCloudIn->clear();
	
	cornerCloud->clear();
	surfaceCloud->clear();
	sharpCornerCloud->clear();
	SharpSurfaceCloud->clear();

	// reset range matrix for range image projection
	rangeMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_32F, cv::Scalar::all(FLT_MAX));

	cloudSmoothness.clear();

	for (int i=0;i<N_SCAN;++i){
		startRingIndex[i] = 0;
		endRingIndex[i] = 0;
	}

	for (int i = 0; i < N_SCAN * Horizon_SCAN; ++i) 
	{
		pointColInd[i] = 0;
		pointRange[i] = 0;
		cloudCurvature[i] = 0;
		cloudNeighborPicked[i] = 0;
		cloudLabel[i] = 0;
	}
}


void FeatureExtraction ::featureExtraction() 
{
	if(!cachePointCloud()){
		return;
	}

	projectPointCloud();
	cloudExtraction();
	
	calculateSmoothness();
	markOccludedPoints();
	extractFeatures();
}


bool FeatureExtraction ::cachePointCloud() 
{
	pcl::fromROSMsg(cloudInfo.cloud_deskewed, *laserCloudIn);
	cloudHeader = cloudInfo.header;

	// check dense flag
	if (laserCloudIn->is_dense == false) {
		ROS_ERROR("Point cloud is not in dense format, please remove NaN points first!");
		ros::shutdown();
	}

	// check ring channel
	static int ringFlag = 0;
	if (ringFlag == 0) 
	{
		ringFlag = -1;
		for (int i = 0; i < (int)currentCloudMsg.fields.size(); ++i) 
		{
			if (currentCloudMsg.fields[i].name == "ring") {
				ringFlag = 1;
				break;
			}
		}
		if (ringFlag == -1) {
			ROS_ERROR("Point cloud ring channel not available, please configure your point cloud data!");
			ros::shutdown();
		}
	}

	// get timestamp
	cloudHeader = currentCloudMsg.header;
	timeScanCur = cloudHeader.stamp.toSec();
	timeScanEnd = timeScanCur + laserCloudIn->points.back().time;  // Velodyne

	return true;
}



void FeatureExtraction ::projectPointCloud() 
{
	int cloudSize = laserCloudIn->points.size();
	// range image projection
	for (int i = 0; i < cloudSize; ++i) 
	{
		// PointType thisPoint;
		PointXYZIRT thisPoint;
		thisPoint.x = laserCloudIn->points[i].x;
		thisPoint.y = laserCloudIn->points[i].y;
		thisPoint.z = laserCloudIn->points[i].z;
		thisPoint.intensity = laserCloudIn->points[i].intensity;
		//新增
		thisPoint.ring = laserCloudIn->points[i].ring;
		thisPoint.time = laserCloudIn->points[i].time;

		float range = pointDistance(thisPoint);
		if (range < lidarMinRange || range > lidarMaxRange) continue;

		int rowIdn = laserCloudIn->points[i].ring;
		if (rowIdn < 0 || rowIdn >= N_SCAN) continue;

		if (rowIdn % downsampleRate != 0) continue;

		float horizonAngle = atan2(thisPoint.x, thisPoint.y) * 180 / M_PI;

		static float ang_res_x = 360.0 / float(Horizon_SCAN);
		int columnIdn = -round((horizonAngle - 90.0) / ang_res_x) + Horizon_SCAN / 2;
		if (columnIdn >= Horizon_SCAN) columnIdn -= Horizon_SCAN;

		if (columnIdn < 0 || columnIdn >= Horizon_SCAN) continue;

		if (rangeMat.at<float>(rowIdn, columnIdn) != FLT_MAX) continue;

		rangeMat.at<float>(rowIdn, columnIdn) = range;
	}
}

/****************************************
 *
 *****************************************/
void FeatureExtraction ::cloudExtraction() 
{
int count = 0;
// extract segmented cloud for lidar odometry
for (int i = 0; i < N_SCAN; ++i) 
{
	startRingIndex[i] = count - 1 + 5;

	for (int j = 0; j < Horizon_SCAN; ++j) 
	{
	if (rangeMat.at<float>(i, j) != FLT_MAX) 
	{
		// mark the points' column index for marking occlusion later
		pointColInd[count] = j;
		// save range info
		pointRange[count] = rangeMat.at<float>(i, j);
		// save extracted cloud
		++count;
	}
	}
	endRingIndex[i] = count - 1 - 5;
}
}

/****************************************
 *
 *****************************************/
void FeatureExtraction ::calculateSmoothness() 
{
int cloudSize = extractedCloud->points.size();
for (int i = 5; i < cloudSize - 5; i++) 
{
	float diffRange = pointRange[i - 5] + pointRange[i - 4] +
					  pointRange[i - 3] + pointRange[i - 2] +
					  pointRange[i - 1] - pointRange[i] * 10 +
					  pointRange[i + 1] + pointRange[i + 2] +
					  pointRange[i + 3] + pointRange[i + 4] + pointRange[i + 5];

	cloudCurvature[i] = diffRange * diffRange;  // diffX * diffX + diffY * diffY + diffZ * diffZ;

	cloudNeighborPicked[i] = 0;
	cloudLabel[i] = 0;
	// cloudSmoothness for sorting
	cloudSmoothness[i].value = cloudCurvature[i];
	cloudSmoothness[i].ind = i;
}
}

/****************************************
 *
 *****************************************/
void FeatureExtraction ::markOccludedPoints() 
{
	int cloudSize = extractedCloud->points.size();
	// mark occluded points and parallel beam points
	for (int i = 5; i < cloudSize - 6; ++i) 
	{
		// occluded points
		float depth1 = pointRange[i];
		float depth2 = pointRange[i + 1];
		int columnDiff = std::abs(int(pointColInd[i + 1] - pointColInd[i]));

		if (columnDiff < 10) 
		{
			// 10 pixel diff in range image
			if (depth1 - depth2 > 0.3) {
				cloudNeighborPicked[i - 5] = 1;
				cloudNeighborPicked[i - 4] = 1;
				cloudNeighborPicked[i - 3] = 1;
				cloudNeighborPicked[i - 2] = 1;
				cloudNeighborPicked[i - 1] = 1;
				cloudNeighborPicked[i] = 1;
			} else if (depth2 - depth1 > 0.3) {
				cloudNeighborPicked[i + 1] = 1;
				cloudNeighborPicked[i + 2] = 1;
				cloudNeighborPicked[i + 3] = 1;
				cloudNeighborPicked[i + 4] = 1;
				cloudNeighborPicked[i + 5] = 1;
				cloudNeighborPicked[i + 6] = 1;
			}
		}
		// parallel beam
		float diff1 = std::abs(float(pointRange[i - 1] - pointRange[i]));
		float diff2 = std::abs(float(pointRange[i + 1] - pointRange[i]));

		if (diff1 > 0.02 * pointRange[i] && diff2 > 0.02 * pointRange[i])
			cloudNeighborPicked[i] = 1;
	}
}

/****************************************
 *
 *****************************************/
void FeatureExtraction ::extractFeatures() 
{
	cornerCloud->clear();
	surfaceCloud->clear();
	sharpCornerCloud->clear();
	SharpSurfaceCloud->clear();

	pcl::PointCloud<PointXYZIRT>::Ptr surfaceCloudScan(new pcl::PointCloud<PointXYZIRT>());
	pcl::PointCloud<PointXYZIRT>::Ptr surfaceCloudScanDS(new pcl::PointCloud<PointXYZIRT>());

	for (int i = 0; i < N_SCAN; i++) 
	{
		surfaceCloudScan->clear();

		for (int j = 0; j < 6; j++) {
			int sp = (startRingIndex[i] * (6 - j) + endRingIndex[i] * j) / 6;
			int ep = (startRingIndex[i] * (5 - j) + endRingIndex[i] * (j + 1)) / 6 - 1;

			if (sp >= ep) continue;

			std::sort(cloudSmoothness.begin() + sp, cloudSmoothness.begin() + ep, by_value());

			int largestPickedNum = 0;
			for (int k = ep; k >= sp; k--) 
			{
				int ind = cloudSmoothness[k].ind;
				if (cloudNeighborPicked[ind] == 0 && cloudCurvature[ind] > edgeThreshold) 
				{
					largestPickedNum++;
					if (largestPickedNum <= 20) {
						cloudLabel[ind] = 1;
						cornerCloud->push_back(extractedCloud->points[ind]);
						if(largestPickedNum <= 4)
							sharpCornerCloud->push_back(extractedCloud->points[ind]);
					} else {
						break;
					}

					cloudNeighborPicked[ind] = 1;
					for (int l = 1; l <= 5; l++) 
					{
						int columnDiff = std::abs(int(pointColInd[ind + l] - pointColInd[ind + l - 1]));
						if (columnDiff > 10) break;
						cloudNeighborPicked[ind + l] = 1;
					}
					
					for (int l = -1; l >= -5; l--) 
					{
						int columnDiff = std::abs(int(pointColInd[ind + l] - pointColInd[ind + l + 1]));
						if (columnDiff > 10) break;
						cloudNeighborPicked[ind + l] = 1;
					}
				}
			}

			largestPickedNum = 0;
			for (int k = sp; k <= ep; k++) 
			{
				int ind = cloudSmoothness[k].ind;
				if (cloudNeighborPicked[ind] == 0 && cloudCurvature[ind] < surfThreshold) 
				{
					largestPickedNum++;

					cloudLabel[ind] = -1;
					cloudNeighborPicked[ind] = 1;

					if(largestPickedNum <= 10)
						SharpSurfaceCloud->push_back(extractedCloud->points[ind]);

					for (int l = 1; l <= 5; l++) 
					{
						int columnDiff = std::abs(int(pointColInd[ind + l] - pointColInd[ind + l - 1]));
						if (columnDiff > 10) break;

						cloudNeighborPicked[ind + l] = 1;
					}
					for (int l = -1; l >= -5; l--) 
					{
						int columnDiff = std::abs(int(pointColInd[ind + l] - pointColInd[ind + l + 1]));
						if (columnDiff > 10) break;

						cloudNeighborPicked[ind + l] = 1;
					}
				}
			}

			for (int k = sp; k <= ep; k++) 
			{
				if (cloudLabel[k] <= 0) 
				{
					surfaceCloudScan->push_back(extractedCloud->points[k]);
				}
			}
		}

		// surfaceCloudScanDS->clear();
		// downSizeFilter.setInputCloud(surfaceCloudScan);
		// downSizeFilter.filter(*surfaceCloudScanDS);

		// *surfaceCloud += *surfaceCloudScanDS;

		*surfaceCloud += *surfaceCloudScan;
	}
}

/****************************************
 *
 *****************************************/
void FeatureExtraction ::assignCouldInfo() 
{
	cloudInfo.header = cloudHeader;

	sensor_msgs::PointCloud2 tempCloud;

	pcl::toROSMsg(*cornerCloud, tempCloud);
	tempCloud.header.stamp = cloudHeader.stamp;
	tempCloud.header.frame_id = lidarFrame;
	cloudInfo.cloud_corner = tempCloud;

	pcl::toROSMsg(*surfaceCloud, tempCloud);
	tempCloud.header.stamp = cloudHeader.stamp;
	tempCloud.header.frame_id = lidarFrame;
	cloudInfo.cloud_surface = tempCloud;

	pcl::toROSMsg(*sharpCornerCloud, tempCloud);
	tempCloud.header.stamp = cloudHeader.stamp;
	tempCloud.header.frame_id = lidarFrame;
	cloudInfo.cloud_corner_sharp = tempCloud;

	pcl::toROSMsg(*SharpSurfaceCloud, tempCloud);
	tempCloud.header.stamp = cloudHeader.stamp;
	tempCloud.header.frame_id = lidarFrame;
	cloudInfo.cloud_surface_sharp = tempCloud;
}
