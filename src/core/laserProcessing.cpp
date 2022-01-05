// This code partly draws on  lio_sam
// Author of  EPSC-LOAM : QZ Wang
// Email wangqingzhi27@outlook.com

#include "laserProcessing.h"

std::mutex imuLock;
std::mutex odoLock;
std::mutex cloLock;

std::deque<sensor_msgs::Imu> imuQueue;
std::deque<nav_msgs::Odometry> odomQueue;
std::deque<sensor_msgs::PointCloud2> cloudQueue;

const int queueLength = 2000;

/****************************************
 *
 *****************************************/
void LaserProcessing ::allocateMemory() 
{
	laserCloudIn.reset(new pcl::PointCloud<PointXYZIRT>());

	fullCloud.reset(new pcl::PointCloud<PointXYZIRT>());
	extractedCloud.reset(new pcl::PointCloud<PointXYZIRT>());

	cornerCloud.reset(new pcl::PointCloud<PointXYZIRT>());
	surfaceCloud.reset(new pcl::PointCloud<PointXYZIRT>());

	fullCloud->points.resize(N_SCAN * Horizon_SCAN);

	startRingIndex = new int32_t[N_SCAN];
	endRingIndex = new int32_t[N_SCAN];

	pointColInd = new int32_t[N_SCAN * Horizon_SCAN];
	pointRange = new float[N_SCAN * Horizon_SCAN];

	cloudSmoothness.resize(N_SCAN * Horizon_SCAN);
	cloudCurvature = new float[N_SCAN * Horizon_SCAN];
	cloudNeighborPicked = new int[N_SCAN * Horizon_SCAN];
	cloudLabel = new int[N_SCAN * Horizon_SCAN];

	// downSizeFilter.setLeafSize(odometrySurfLeafSize, odometrySurfLeafSize, odometrySurfLeafSize);
}

/****************************************
 *
 *****************************************/
void LaserProcessing ::resetParameters() 
{
	laserCloudIn->clear();
	fullCloud->clear();
	extractedCloud->clear();
	cornerCloud->clear();
	surfaceCloud->clear();

	// reset range matrix for range image projection
	rangeMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_32F, cv::Scalar::all(FLT_MAX));

	imuPointerCur = 0;
	firstPointFlag = true;
	odomDeskewFlag = false;

	for (int i = 0; i < queueLength; ++i) 
	{
		imuTime[i] = 0;
		imuRotX[i] = 0;
		imuRotY[i] = 0;
		imuRotZ[i] = 0;
	}

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

/****************************************
 *
 *****************************************/
bool LaserProcessing ::distortionRemoval() 
{

	if (!cachePointCloud()) return false;

	if (!deskewInfo()) return false;

	projectPointCloud();

	cloudExtraction();

	return true;
}

/****************************************
 *
 *****************************************/
void LaserProcessing ::featureExtraction() 
{
	calculateSmoothness();

	markOccludedPoints();

	extractFeatures();
}

/****************************************
 *
 *****************************************/
bool LaserProcessing ::cachePointCloud() 
{
	// cache point cloud
	if (cloudQueue.size() <= 2) return false;

	// convert cloud
	cloLock.lock();
	currentCloudMsg = cloudQueue.front();
	cloudQueue.pop_front();
	cloLock.unlock();

	pcl::fromROSMsg(currentCloudMsg, *laserCloudIn);

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

	// check point time
	if (deskewFlag == 0) 
	{
		deskewFlag = -1;
		for (int i = 0; i < (int)currentCloudMsg.fields.size(); ++i) {
		if (currentCloudMsg.fields[i].name == "time") {
			deskewFlag = 1;
			break;
		}
		}
		if (deskewFlag == -1)
		ROS_WARN("Point cloud timestamp not available, deskew function disabled, system will drift significantly!");
	}

	// get timestamp
	cloudHeader = currentCloudMsg.header;
	timeScanCur = cloudHeader.stamp.toSec();
	timeScanEnd = timeScanCur + laserCloudIn->points.back().time;  // Velodyne

	return true;
}

/****************************************
 *
 *****************************************/
bool LaserProcessing ::deskewInfo() 
{
	std::lock_guard<std::mutex> lock1(imuLock);
	std::lock_guard<std::mutex> lock2(odoLock);

	if (useImu == true) {
		// make sure IMU data available for the scan
		if (imuQueue.empty() || 
			imuQueue.front().header.stamp.toSec() > timeScanCur ||
			imuQueue.back().header.stamp.toSec() < timeScanEnd) 
		{
			ROS_DEBUG("Waiting for IMU data ...");
			return false;
		}

		imuDeskewInfo();

		odomDeskewInfo();
	} else {
		odomDeskewInfo();
	}

	return true;
}

/****************************************
 *
 *****************************************/
void LaserProcessing ::imuDeskewInfo() 
{
	cloudInfo.imuAvailable = false;

	while (!imuQueue.empty()) 
	{
		if (imuQueue.front().header.stamp.toSec() < timeScanCur - 0.01)
			imuQueue.pop_front();
		else
			break;
	}

	if (imuQueue.empty()) return;

	imuPointerCur = 0;

	for (int i = 0; i < (int)imuQueue.size(); ++i) 
	{
		sensor_msgs::Imu thisImuMsg = imuQueue[i];
		double currentImuTime = thisImuMsg.header.stamp.toSec();

		// get roll, pitch, and yaw estimation for this scan
		if (currentImuTime <= timeScanCur)
		imuRPY2rosRPY(&thisImuMsg, &cloudInfo.imuRollInit, &cloudInfo.imuPitchInit, &cloudInfo.imuYawInit);

		if (currentImuTime > timeScanEnd + 0.01) break;

		if (imuPointerCur == 0) 
		{
			imuRotX[0] = 0;
			imuRotY[0] = 0;
			imuRotZ[0] = 0;
			imuTime[0] = currentImuTime;
			++imuPointerCur;
			continue;
		}

		// get angular velocity
		double angular_x, angular_y, angular_z;
		imuAngular2rosAngular(&thisImuMsg, &angular_x, &angular_y, &angular_z);

		// integrate rotation
		double timeDiff = currentImuTime - imuTime[imuPointerCur - 1];
		imuRotX[imuPointerCur] = imuRotX[imuPointerCur - 1] + angular_x * timeDiff;
		imuRotY[imuPointerCur] = imuRotY[imuPointerCur - 1] + angular_y * timeDiff;
		imuRotZ[imuPointerCur] = imuRotZ[imuPointerCur - 1] + angular_z * timeDiff;
		imuTime[imuPointerCur] = currentImuTime;
		++imuPointerCur;
	}

	--imuPointerCur;

	if (imuPointerCur <= 0) return;

	cloudInfo.imuAvailable = true;
}

/****************************************
 *
 *****************************************/
void LaserProcessing ::odomDeskewInfo() 
{
	// ROS_WARN("odomDeskewInfo start!");
	// std::cout<<"odomQueueSize: "<<odomQueue.size()<<std::endl;
	cloudInfo.odomAvailable = false;

	while (!odomQueue.empty()) 
	{
		if (odomQueue.front().header.stamp.toSec() < timeScanCur - 0.01)
			odomQueue.pop_front();
		else
			break;
	}

	if (odomQueue.empty()) {
		// ROS_WARN("odomDeskewInfo :  odomQueue.empty()!");
		return;
	}

	if (odomQueue.front().header.stamp.toSec() > timeScanCur) {
		//  ROS_WARN("odomDeskewInfo :  odomQueue.front().header.stamp.toSec() >
		//  timeScanCur!");
		return;
	}

	// get start odometry at the beinning of the scan
	nav_msgs::Odometry startOdomMsg;

	for (int i = 0; i < (int)odomQueue.size(); ++i) 
	{
		startOdomMsg = odomQueue[i];

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
	cloudInfo.initialGuessX = startOdomMsg.pose.pose.position.x;
	cloudInfo.initialGuessY = startOdomMsg.pose.pose.position.y;
	cloudInfo.initialGuessZ = startOdomMsg.pose.pose.position.z;
	cloudInfo.initialGuessRoll = roll;
	cloudInfo.initialGuessPitch = pitch;
	cloudInfo.initialGuessYaw = yaw;

	cloudInfo.odomAvailable = true;

	// ROS_WARN("odomDeskewInfo end!");

	// get end odometry at the end of the scan
	odomDeskewFlag = false;

	if (odomQueue.back().header.stamp.toSec() < timeScanEnd) return;

	nav_msgs::Odometry endOdomMsg;

	for (int i = 0; i < (int)odomQueue.size(); ++i) 
	{
		endOdomMsg = odomQueue[i];

		if (ROS_TIME(&endOdomMsg) < timeScanEnd)
			continue;
		else
			break;
	}

	if (int(round(startOdomMsg.pose.covariance[0])) != int(round(endOdomMsg.pose.covariance[0])))
		return;

	Eigen::Affine3f transBegin = pcl::getTransformation(
		startOdomMsg.pose.pose.position.x, startOdomMsg.pose.pose.position.y,
		startOdomMsg.pose.pose.position.z, roll, pitch, yaw);

	tf::quaternionMsgToTF(endOdomMsg.pose.pose.orientation, orientation);
	tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);
	Eigen::Affine3f transEnd = pcl::getTransformation(
		endOdomMsg.pose.pose.position.x, endOdomMsg.pose.pose.position.y,
		endOdomMsg.pose.pose.position.z, roll, pitch, yaw);

	Eigen::Affine3f transBt = transBegin.inverse() * transEnd;

	float rollIncre, pitchIncre, yawIncre;
	pcl::getTranslationAndEulerAngles(transBt, odomIncreX, odomIncreY, odomIncreZ, rollIncre, pitchIncre, yawIncre);

	odomDeskewFlag = true;
}

/****************************************
 *
 *****************************************/
void LaserProcessing ::findRotation(double pointTime, float *rotXCur,
									float *rotYCur, float *rotZCur) 
{
	*rotXCur = 0;
	*rotYCur = 0;
	*rotZCur = 0;

	int imuPointerFront = 0;
	while (imuPointerFront < imuPointerCur) 
	{
		if (pointTime < imuTime[imuPointerFront]) break;
		++imuPointerFront;
	}

	if (pointTime > imuTime[imuPointerFront] || imuPointerFront == 0) 
	{
		*rotXCur = imuRotX[imuPointerFront];
		*rotYCur = imuRotY[imuPointerFront];
		*rotZCur = imuRotZ[imuPointerFront];
	} else {
		int imuPointerBack = imuPointerFront - 1;
		double ratioFront = (pointTime - imuTime[imuPointerBack]) /
							(imuTime[imuPointerFront] - imuTime[imuPointerBack]);
		double ratioBack = (imuTime[imuPointerFront] - pointTime) /
						(imuTime[imuPointerFront] - imuTime[imuPointerBack]);
		*rotXCur = imuRotX[imuPointerFront] * ratioFront +
				imuRotX[imuPointerBack] * ratioBack;
		*rotYCur = imuRotY[imuPointerFront] * ratioFront +
				imuRotY[imuPointerBack] * ratioBack;
		*rotZCur = imuRotZ[imuPointerFront] * ratioFront +
				imuRotZ[imuPointerBack] * ratioBack;
	}
}

/****************************************
 *
 *****************************************/
void LaserProcessing ::findPosition(double relTime, float *posXCur, float *posYCur, float *posZCur) 
{
	*posXCur = 0;
	*posYCur = 0;
	*posZCur = 0;

	// If the sensor moves relatively slow, like walking speed, positional deskew
	// seems to have little benefits. Thus code below is commented.

	// if (cloudInfo.odomAvailable == false || odomDeskewFlag == false)
	//     return;

	// float ratio = relTime / (timeScanEnd - timeScanCur);

	// *posXCur = ratio * odomIncreX;
	// *posYCur = ratio * odomIncreY;
	// *posZCur = ratio * odomIncreZ;
}

/****************************************
 *
 *****************************************/
PointXYZIRT LaserProcessing ::deskewPoint(PointXYZIRT *point, double relTime) 
{
	if (deskewFlag == -1 || cloudInfo.imuAvailable == false) return *point;

	double pointTime = timeScanCur + relTime;

	float rotXCur, rotYCur, rotZCur;
	findRotation(pointTime, &rotXCur, &rotYCur, &rotZCur);

	float posXCur, posYCur, posZCur;
	findPosition(relTime, &posXCur, &posYCur, &posZCur);

	if (firstPointFlag == true) {
		transStartInverse = (pcl::getTransformation(posXCur, posYCur, posZCur,
													rotXCur, rotYCur, rotZCur)).inverse();
		firstPointFlag = false;
	}

	// transform points to start
	Eigen::Affine3f transFinal = pcl::getTransformation(posXCur, posYCur, posZCur, rotXCur, rotYCur, rotZCur);
	Eigen::Affine3f transBt = transStartInverse * transFinal;

	PointXYZIRT newPoint;
	newPoint.x = transBt(0, 0) * point->x + transBt(0, 1) * point->y +
				transBt(0, 2) * point->z + transBt(0, 3);
	newPoint.y = transBt(1, 0) * point->x + transBt(1, 1) * point->y +
				transBt(1, 2) * point->z + transBt(1, 3);
	newPoint.z = transBt(2, 0) * point->x + transBt(2, 1) * point->y +
				transBt(2, 2) * point->z + transBt(2, 3);
	newPoint.intensity = point->intensity;
	//新增
	newPoint.ring = point->ring;
	newPoint.time = point->time;

	return newPoint;
}

/****************************************
 *
 *****************************************/
void LaserProcessing ::projectPointCloud() 
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

		thisPoint = deskewPoint(&thisPoint, laserCloudIn->points[i].time);  // Velodyne
		// thisPoint = deskewPoint(&thisPoint, (float)laserCloudIn->points[i].t /
		// 1000000000.0); // Ouster

		rangeMat.at<float>(rowIdn, columnIdn) = range;

		int index = columnIdn + rowIdn * Horizon_SCAN;
		fullCloud->points[index] = thisPoint;
	}
}

/****************************************
 *
 *****************************************/
void LaserProcessing ::cloudExtraction() 
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
		extractedCloud->push_back(fullCloud->points[j + i * Horizon_SCAN]);
		// size of extracted cloud
		++count;
	}
	}
	endRingIndex[i] = count - 1 - 5;
}
}

/****************************************
 *
 *****************************************/
void LaserProcessing ::calculateSmoothness() 
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
void LaserProcessing ::markOccludedPoints() 
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
void LaserProcessing ::extractFeatures() 
{
	cornerCloud->clear();
	surfaceCloud->clear();

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

			for (int k = sp; k <= ep; k++) 
			{
				int ind = cloudSmoothness[k].ind;
				if (cloudNeighborPicked[ind] == 0 && cloudCurvature[ind] < surfThreshold) 
				{
					cloudLabel[ind] = -1;
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
void LaserProcessing ::assignCouldInfo() 
{
	cloudInfo.header = cloudHeader;

	sensor_msgs::PointCloud2 tempCloud;

	pcl::toROSMsg(*extractedCloud, tempCloud);
	tempCloud.header.stamp = cloudHeader.stamp;
	tempCloud.header.frame_id = lidarFrame;
	cloudInfo.cloud_deskewed = tempCloud;

	pcl::toROSMsg(*cornerCloud, tempCloud);
	tempCloud.header.stamp = cloudHeader.stamp;
	tempCloud.header.frame_id = lidarFrame;
	cloudInfo.cloud_corner = tempCloud;

	pcl::toROSMsg(*surfaceCloud, tempCloud);
	tempCloud.header.stamp = cloudHeader.stamp;
	tempCloud.header.frame_id = lidarFrame;
	cloudInfo.cloud_surface = tempCloud;
}
