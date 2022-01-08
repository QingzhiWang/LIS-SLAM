// Author of  EPSC-LOAM : QZ Wang
// Email wangqingzhi27@outlook.com

#define PCL_NO_PRECOMPILE

#include "common.h"
#include "lis_slam/cloud_info.h"
#include "utility.h"

#define USING_SUBMAP_TARGET false
#define USING_MULTI_FRAME_TARGET true


class OdomEstimationNode : public ParamServer 
{
 public:
	double total_time = 0;
	int total_frame = 0;

	ros::Subscriber subCloudInfo;
	ros::Publisher pubKeyFrameInfo;
	ros::Publisher pubKeyFrameId;

	ros::Publisher pubLaserOdometryIncremental;
	ros::Publisher pubLaserOdometryGlobal;

	lis_slam::cloud_info cloudInfo;
	lis_slam::cloud_info keyFrameInfo;

	pcl::PointCloud<PointType>::Ptr laserCloudCornerLast;  // corner feature set from odoOptimization
	pcl::PointCloud<PointType>::Ptr laserCloudSurfLast;  // surf feature set from odoOptimization
	pcl::PointCloud<PointType>::Ptr laserCloudSharpCornerLast;  
	pcl::PointCloud<PointType>::Ptr laserCloudSharpSurfLast;  

	pcl::VoxelGrid<PointType> downSizeFilterCorner;
	pcl::VoxelGrid<PointType> downSizeFilterSurf;

	int laserCloudSharpCornerLastNum = 0;
	int laserCloudSharpSurfLastNum = 0;

	pcl::PointCloud<PointType>::Ptr laserCloudOri;
	pcl::PointCloud<PointType>::Ptr coeffSel;

	std::vector<PointType> laserCloudOriCornerVec;  // corner point holder for parallel computation
	std::vector<PointType> coeffSelCornerVec;
	std::vector<bool> laserCloudOriCornerFlag;
	std::vector<PointType> laserCloudOriSurfVec;  // surf point holder for parallel computation
	std::vector<PointType> coeffSelSurfVec;
	std::vector<bool> laserCloudOriSurfFlag;

	ros::Time timeLaserInfoStamp;
	double timeLaserInfoCur;

	bool FirstFlag = true;
	uint64 keyFrameId = 0;

	float transformTobeMapped[6];

	float transformCurFrame2PriFrame[6];
	float transformPriFrame[6];

	float transPredictionMapped[6];

	Eigen::Affine3f transPointAssociateToMap;
	Eigen::Affine3f incrementalOdometryAffineFront;
	Eigen::Affine3f incrementalOdometryAffineBack;

	bool isDegenerate = false;
	Eigen::Matrix<float, 6, 6> matP;

	bool isMapOptmization = false;

	pcl::PointCloud<PointType>::Ptr cloudKeyPoses3D;
	pcl::PointCloud<PointTypePose>::Ptr cloudKeyPoses6D;
	vector<pcl::PointCloud<PointType>::Ptr> cornerCloudKeyFrames;
	vector<pcl::PointCloud<PointType>::Ptr> surfCloudKeyFrames;

	map<int, pair<pcl::PointCloud<PointType>, pcl::PointCloud<PointType>>> laserCloudMapContainer;
	pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMap;
	pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMap;
	pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMapDS;
	pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMapDS;

	std::vector<pcl::PointCloud<PointType>::Ptr> laserCloudSurfVec;
	std::vector<pcl::PointCloud<PointType>::Ptr> laserCloudCornerVec;

	pcl::KdTreeFLANN<PointType>::Ptr kdtreeCornerFromMap;
	pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurfFromMap;

	pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurroundingKeyPoses;

	pcl::VoxelGrid<PointType> downSizeFilterSurroundingKeyPoses;  // for surrounding key poses of scan-to-map optimization

	int laserCloudCornerFromMapDSNum = 0;
	int laserCloudSurfFromMapDSNum = 0;

	OdomEstimationNode() {
		subCloudInfo = nh.subscribe<lis_slam::cloud_info>("lis_slam/laser_process/cloud_info", 10, &OdomEstimationNode::laserCloudInfoHandler, this);

		pubKeyFrameInfo = nh.advertise<lis_slam::cloud_info>("lis_slam/odom_estimation/cloud_info", 10);

		pubLaserOdometryGlobal = nh.advertise<nav_msgs::Odometry>(odomTopic + "/front", 200);
		pubLaserOdometryIncremental = nh.advertise<nav_msgs::Odometry>(odomTopic + "/front_incremental", 200);

		pubKeyFrameId = nh.advertise<sensor_msgs::PointCloud2>("lis_slam/odom_estimation/keyframe_id", 10);

		downSizeFilterCorner.setLeafSize(mappingCornerLeafSize, mappingCornerLeafSize, mappingCornerLeafSize);
		downSizeFilterSurf.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize,mappingSurfLeafSize);

		allocateMemory();

		downSizeFilterSurroundingKeyPoses.setLeafSize(surroundingKeyframeDensity, surroundingKeyframeDensity,surroundingKeyframeDensity);  // for surrounding key poses of scan-to-map optimization
	}

	void allocateMemory() 
	{
		laserCloudCornerLast.reset(new pcl::PointCloud<PointType>());
		laserCloudSurfLast.reset(new pcl::PointCloud<PointType>());
		laserCloudSharpCornerLast.reset(new pcl::PointCloud<PointType>());
		laserCloudSharpSurfLast.reset(new pcl::PointCloud<PointType>());

		laserCloudOri.reset(new pcl::PointCloud<PointType>());
		coeffSel.reset(new pcl::PointCloud<PointType>());

		laserCloudOriCornerVec.resize(N_SCAN * Horizon_SCAN);
		coeffSelCornerVec.resize(N_SCAN * Horizon_SCAN);
		laserCloudOriCornerFlag.resize(N_SCAN * Horizon_SCAN);
		laserCloudOriSurfVec.resize(N_SCAN * Horizon_SCAN);
		coeffSelSurfVec.resize(N_SCAN * Horizon_SCAN);
		laserCloudOriSurfFlag.resize(N_SCAN * Horizon_SCAN);

		std::fill(laserCloudOriCornerFlag.begin(), laserCloudOriCornerFlag.end(), false);
		std::fill(laserCloudOriSurfFlag.begin(), laserCloudOriSurfFlag.end(), false);

		for (int i = 0; i < 6; ++i) 
		{
			transformTobeMapped[i] = 0;
			transformCurFrame2PriFrame[i] = 0;
			transformPriFrame[i] = 0;

			transPredictionMapped[i] = 0.0;
		}

		matP.setZero();

		cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
		cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());
		laserCloudCornerFromMap.reset(new pcl::PointCloud<PointType>());
		laserCloudSurfFromMap.reset(new pcl::PointCloud<PointType>());
		laserCloudCornerFromMapDS.reset(new pcl::PointCloud<PointType>());
		laserCloudSurfFromMapDS.reset(new pcl::PointCloud<PointType>());

		kdtreeCornerFromMap.reset(new pcl::KdTreeFLANN<PointType>());
		kdtreeSurfFromMap.reset(new pcl::KdTreeFLANN<PointType>());

		kdtreeSurroundingKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());
	}


	void laserCloudInfoHandler(const lis_slam::cloud_infoConstPtr &msgIn) 
	{
		std::chrono::time_point<std::chrono::system_clock> start, end;
		start = std::chrono::system_clock::now();

		cloudInfo = *msgIn;

		timeLaserInfoStamp = cloudInfo.header.stamp;
		timeLaserInfoCur = timeLaserInfoStamp.toSec();
		
		updateInitialGuess();

		if (FirstFlag) 
		{
			currentCloudInit();
			saveKeyFrames();
			publishOdometry();
			publishCloudInfo();
			FirstFlag = false;
			return;
		}

		#if USING_MULTI_FRAME_TARGET
			laserCloudCornerFromMapDS->clear();
			laserCloudSurfFromMapDS->clear();

			pcl::PointCloud<PointType>::Ptr tmpSurf( new pcl::PointCloud<PointType>());
			pcl::PointCloud<PointType>::Ptr tmpCorner( new pcl::PointCloud<PointType>());
			pcl::copyPointCloud(*laserCloudSurfLast,    *tmpSurf);
			pcl::copyPointCloud(*laserCloudCornerLast,    *tmpCorner);

			*tmpSurf = *transformPointCloud(tmpSurf, &cloudKeyPoses6D->back());
			*tmpCorner = *transformPointCloud(tmpCorner, &cloudKeyPoses6D->back());

			laserCloudSurfVec.push_back(tmpSurf);
			laserCloudCornerVec.push_back(tmpCorner);

			while(laserCloudSurfVec.size() >= 5)
			{
				laserCloudSurfVec.erase(laserCloudSurfVec.begin());
				laserCloudCornerVec.erase(laserCloudCornerVec.begin());
			}
			
			for(int i = 0; i < laserCloudSurfVec.size(); i++){
				*laserCloudCornerFromMapDS += *laserCloudCornerVec[i];
				*laserCloudSurfFromMapDS += *laserCloudSurfVec[i];
			}
		#endif

		#if USING_SUBMAP_TARGET
			extractSurroundingKeyFrames();
		#endif

		currentCloudInit();
		scan2SubMapOptimization();
		publishOdometry();

		if(saveTrajectory)
		{
			saveKeyFrames();
			publishCloudInfo();
			publishCloud(&pubKeyFrameId, cloudKeyPoses3D, timeLaserInfoStamp, mapFrame);
		}
		else
		{
			calculateTranslation();
			if (abs(transformCurFrame2PriFrame[2]) >= keyFrameMiniYaw ||
				abs(transformCurFrame2PriFrame[3]) >= keyFrameMiniDistance ||
				abs(transformCurFrame2PriFrame[4]) >= keyFrameMiniDistance) 
			{
				saveKeyFrames();
				publishCloudInfo();
				publishCloud(&pubKeyFrameId, cloudKeyPoses3D, timeLaserInfoStamp, mapFrame);
			}
		}

		end = std::chrono::system_clock::now();
		std::chrono::duration<float> elapsed_seconds = end - start;
		total_frame++;
		float time_temp = elapsed_seconds.count() * 1000;
		total_time += time_temp;

		ROS_INFO("keyFrameId: %d,  TotalFrameId: %d.", keyFrameId, total_frame);
		ROS_INFO("Average odom estimation time %f ms", total_time / total_frame);
	}



	void pointAssociateToMap(PointType const *const pi, PointType *const po) 
	{
		po->x = transPointAssociateToMap(0, 0) * pi->x +
				transPointAssociateToMap(0, 1) * pi->y +
				transPointAssociateToMap(0, 2) * pi->z +
				transPointAssociateToMap(0, 3);
		po->y = transPointAssociateToMap(1, 0) * pi->x +
				transPointAssociateToMap(1, 1) * pi->y +
				transPointAssociateToMap(1, 2) * pi->z +
				transPointAssociateToMap(1, 3);
		po->z = transPointAssociateToMap(2, 0) * pi->x +
				transPointAssociateToMap(2, 1) * pi->y +
				transPointAssociateToMap(2, 2) * pi->z +
				transPointAssociateToMap(2, 3);
		po->intensity = pi->intensity;
	}

	void currentCloudInit()
	{
		pcl::fromROSMsg(cloudInfo.cloud_corner, *laserCloudCornerLast);
		pcl::fromROSMsg(cloudInfo.cloud_surface, *laserCloudSurfLast);
		pcl::fromROSMsg(cloudInfo.cloud_corner_sharp, *laserCloudSharpCornerLast);
		pcl::fromROSMsg(cloudInfo.cloud_surface_sharp, *laserCloudSharpSurfLast);

		// Downsample cloud from current scan
		// laserCloudSharpCornerLast->clear();
		// downSizeFilterCorner.setInputCloud(laserCloudCornerLast);
		// downSizeFilterCorner.filter(*laserCloudSharpCornerLast);

		// laserCloudSharpSurfLast->clear();
		// downSizeFilterSurf.setInputCloud(laserCloudSurfLast);
		// downSizeFilterSurf.filter(*laserCloudSharpSurfLast);

		laserCloudSharpCornerLastNum = laserCloudSharpCornerLast->size();
		laserCloudSharpSurfLastNum = laserCloudSharpSurfLast->size();
	}



	void calculateTranslation() 
	{
		Eigen::Affine3f transBack = trans2Affine3f(transformPriFrame);
		Eigen::Affine3f transTobe = trans2Affine3f(transformTobeMapped);

		Eigen::Affine3f transIncre = transBack.inverse() * transTobe;

		pcl::getTranslationAndEulerAngles(
			transIncre, transformCurFrame2PriFrame[3], transformCurFrame2PriFrame[4], transformCurFrame2PriFrame[5],
			transformCurFrame2PriFrame[0], transformCurFrame2PriFrame[1], transformCurFrame2PriFrame[2]);
	}

  	void updateInitialGuess() 
	{
		// save current transformation before any processing
		incrementalOdometryAffineFront = trans2Affine3f(transformTobeMapped);

		static Eigen::Affine3f lastImuTransformation;
		// initialization
		static bool firstTransAvailable = false;
		if (firstTransAvailable == false) {
			ROS_WARN("Front: firstTransAvailable!");
			transformTobeMapped[0] = cloudInfo.imuRollInit;
			transformTobeMapped[1] = cloudInfo.imuPitchInit;
			transformTobeMapped[2] = cloudInfo.imuYawInit;

			if (!useImuHeadingInitialization)
			    transformTobeMapped[2] = 0;

			lastImuTransformation = pcl::getTransformation(0, 0, 0, cloudInfo.imuRollInit, cloudInfo.imuPitchInit, cloudInfo.imuYawInit);  // save imu before return;
			firstTransAvailable = true;
			return;
		}

		// use imu pre-integration estimation for pose guess
		static bool lastImuPreTransAvailable = false;
		static Eigen::Affine3f lastImuPreTransformation;
		if (cloudInfo.odomAvailable == true) 
		{
		ROS_WARN("Front: cloudInfo.odomAvailable == true!");
		Eigen::Affine3f transBack = pcl::getTransformation(cloudInfo.initialGuessX, cloudInfo.initialGuessY, cloudInfo.initialGuessZ, 
														   cloudInfo.initialGuessRoll, cloudInfo.initialGuessPitch, cloudInfo.initialGuessYaw);
			if (lastImuPreTransAvailable == false) {
				lastImuPreTransformation = transBack;
				lastImuPreTransAvailable = true;
			} else {
				Eigen::Affine3f transIncre = lastImuPreTransformation.inverse() * transBack;
				Eigen::Affine3f transTobe = trans2Affine3f(transformTobeMapped);
				Eigen::Affine3f transFinal = transTobe * transIncre;
				pcl::getTranslationAndEulerAngles(transFinal, 
						transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5], 
						transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);

				// transPredictionMapped=trans2Affine3f(transformTobeMapped);
				for (int i = 0; i < 6; ++i) 
				{
					transPredictionMapped[i] = transformTobeMapped[i];
				}

				lastImuPreTransformation = transBack;

				lastImuTransformation = pcl::getTransformation(0, 0, 0, cloudInfo.imuRollInit, cloudInfo.imuPitchInit, cloudInfo.imuYawInit);  // save imu before return;
				return;
			}
		}

		static float lastTransformTobeMapped[6] = {0.0};
		// if (cloudInfo.odomAvailable == false && cloudInfo.imuAvailable == false) 
		if (cloudInfo.odomAvailable == false) 
		{
			static bool first = false;
			if (first == false) 
			{
				for (int i = 0; i < 6; ++i) 
				{
					lastTransformTobeMapped[i] = transformTobeMapped[i];
				}

				first = true;
				return;
			}

			ROS_WARN("Front: cloudInfo.odomAvailable == false!");
			Eigen::Affine3f transBack = pcl::getTransformation(
					transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5], 
					transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);

			Eigen::Affine3f transLast = pcl::getTransformation(
					lastTransformTobeMapped[3], lastTransformTobeMapped[4], lastTransformTobeMapped[5], 
					lastTransformTobeMapped[0], lastTransformTobeMapped[1], lastTransformTobeMapped[2]);

			for (int i = 0; i < 6; ++i) 
			{
				lastTransformTobeMapped[i] = transformTobeMapped[i];
			}

			Eigen::Affine3f transIncre = transLast.inverse() * transBack;

			Eigen::Affine3f transTobe = trans2Affine3f(transformTobeMapped);
			Eigen::Affine3f transFinal = transTobe * transIncre;

			pcl::getTranslationAndEulerAngles(transFinal, 
					transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5], 
					transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);

			return;
		}

		// use imu incremental estimation for pose guess (only rotation)
		if (cloudInfo.imuAvailable == true) 
		{
			ROS_WARN("Front: cloudInfo.imuAvailable == true!");
			Eigen::Affine3f transBack = pcl::getTransformation(0, 0, 0, cloudInfo.imuRollInit, cloudInfo.imuPitchInit, cloudInfo.imuYawInit);

			Eigen::Affine3f transIncre = lastImuTransformation.inverse() * transBack;

			Eigen::Affine3f transTobe = trans2Affine3f(transformTobeMapped);
			Eigen::Affine3f transFinal = transTobe * transIncre;

			pcl::getTranslationAndEulerAngles(transFinal, 
					transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5], 
					transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);

			// transPredictionMapped=trans2Affine3f(transformTobeMapped);
			for (int i = 0; i < 6; ++i) 
			{
				transPredictionMapped[i] = transformTobeMapped[i];
			}

			lastImuTransformation = pcl::getTransformation(0, 0, 0, cloudInfo.imuRollInit, cloudInfo.imuPitchInit, cloudInfo.imuYawInit);  // save imu before return;

			return;
		}

  	}

	void saveKeyFrames() 
	{
		PointType thisPose3D;
		thisPose3D.x = transformTobeMapped[3];
		thisPose3D.y = transformTobeMapped[4];
		thisPose3D.z = transformTobeMapped[5];
		thisPose3D.intensity = keyFrameId;
		cloudKeyPoses3D->points.push_back(thisPose3D);

		PointTypePose thisPose6D;
		thisPose6D.x = thisPose3D.x;
		thisPose6D.y = thisPose3D.y;
		thisPose6D.z = thisPose3D.z;
		thisPose6D.intensity = thisPose3D.intensity;  // this can be used as index
		thisPose6D.roll = transformTobeMapped[0];
		thisPose6D.pitch = transformTobeMapped[1];
		thisPose6D.yaw = transformTobeMapped[2];
		thisPose6D.time = timeLaserInfoCur;
		cloudKeyPoses6D->points.push_back(thisPose6D);

		pcl::PointCloud<PointType>::Ptr thisCornerKeyFrame(new pcl::PointCloud<PointType>());
		pcl::PointCloud<PointType>::Ptr thisSurfKeyFrame(new pcl::PointCloud<PointType>());
		pcl::copyPointCloud(*laserCloudCornerLast, *thisCornerKeyFrame);
		pcl::copyPointCloud(*laserCloudSurfLast, *thisSurfKeyFrame);

		// save key frame cloud
		cornerCloudKeyFrames.push_back(thisCornerKeyFrame);
		surfCloudKeyFrames.push_back(thisSurfKeyFrame);

		transformPriFrame[0] = transformTobeMapped[0];
		transformPriFrame[1] = transformTobeMapped[1];
		transformPriFrame[2] = transformTobeMapped[2];
		transformPriFrame[3] = transformTobeMapped[3];
		transformPriFrame[4] = transformTobeMapped[4];
		transformPriFrame[5] = transformTobeMapped[5];

		keyFrameId++;
	}

	void publishCloudInfo() 
	{
		keyFrameInfo = cloudInfo;

		keyFrameInfo.header.stamp = timeLaserInfoStamp;

		keyFrameInfo.imuRollInit = cloudInfo.imuRollInit;
		keyFrameInfo.imuPitchInit = cloudInfo.imuPitchInit;
		keyFrameInfo.imuYawInit = cloudInfo.imuYawInit;

		keyFrameInfo.imuAvailable = cloudInfo.imuAvailable;

		sensor_msgs::PointCloud2 tempCloud;

		pcl::toROSMsg(*laserCloudCornerLast, tempCloud);
		tempCloud.header.stamp = timeLaserInfoStamp;
		tempCloud.header.frame_id = lidarFrame;
		keyFrameInfo.cloud_corner = tempCloud;

		pcl::toROSMsg(*laserCloudSurfLast, tempCloud);
		tempCloud.header.stamp = timeLaserInfoStamp;
		tempCloud.header.frame_id = lidarFrame;
		keyFrameInfo.cloud_surface = tempCloud;

		pcl::toROSMsg(*laserCloudSharpCornerLast, tempCloud);
		tempCloud.header.stamp = timeLaserInfoStamp;
		tempCloud.header.frame_id = lidarFrame;
		keyFrameInfo.cloud_corner_sharp = tempCloud;

		pcl::toROSMsg(*laserCloudSharpSurfLast, tempCloud);
		tempCloud.header.stamp = timeLaserInfoStamp;
		tempCloud.header.frame_id = lidarFrame;
		keyFrameInfo.cloud_surface_sharp = tempCloud;

		keyFrameInfo.initialGuessX = transformTobeMapped[3];
		keyFrameInfo.initialGuessY = transformTobeMapped[4];
		keyFrameInfo.initialGuessZ = transformTobeMapped[5];
		keyFrameInfo.initialGuessRoll = transformTobeMapped[0];
		keyFrameInfo.initialGuessPitch = transformTobeMapped[1];
		keyFrameInfo.initialGuessYaw = transformTobeMapped[2];

		keyFrameInfo.odomAvailable = true;

		pubKeyFrameInfo.publish(keyFrameInfo);
	}

	void extractSurroundingKeyFrames() {
		pcl::PointCloud<PointType>::Ptr surroundingKeyPoses(new pcl::PointCloud<PointType>());
		pcl::PointCloud<PointType>::Ptr surroundingKeyPosesDS(new pcl::PointCloud<PointType>());
		// std::vector<int> pointSearchInd;
		// std::vector<float> pointSearchSqDis;

		// // extract all the nearby key poses and downsample them
		// kdtreeSurroundingKeyPoses->setInputCloud(cloudKeyPoses3D); // create
		// kd-tree kdtreeSurroundingKeyPoses->radiusSearch(cloudKeyPoses3D->back(), (double)surroundingKeyframeSearchRadius, pointSearchInd, pointSearchSqDis); 
		// for (int i = 0; i < (int)pointSearchInd.size(); ++i)
		// {
		//     int id = pointSearchInd[i];
		//     surroundingKeyPoses->push_back(cloudKeyPoses3D->points[id]);
		// }

		// downSizeFilterSurroundingKeyPoses.setInputCloud(surroundingKeyPoses);
		// downSizeFilterSurroundingKeyPoses.filter(*surroundingKeyPosesDS);

		// also extract some latest key frames in case the robot rotates in one position
		int numPoses = cloudKeyPoses3D->size();
		for (int i = numPoses - 1; i >= 0; --i) 
		{
			if (timeLaserInfoCur - cloudKeyPoses6D->points[i].time < 6.0)  // 10.0
				surroundingKeyPosesDS->points.push_back(cloudKeyPoses3D->points[i]);
			else
				break;
		}

		extractCloud(surroundingKeyPosesDS);
	}

	void extractCloud(pcl::PointCloud<PointType>::Ptr cloudToExtract) 
	{
		// fuse the map
		laserCloudCornerFromMap->clear();
		laserCloudSurfFromMap->clear();
		for (int i = 0; i < (int)cloudToExtract->size(); ++i) 
		{
			if (pointDistance(cloudToExtract->points[i], cloudKeyPoses3D->back()) > surroundingKeyframeSearchRadius)
				continue;

			int thisKeyInd = (int)cloudToExtract->points[i].intensity;
			if (laserCloudMapContainer.find(thisKeyInd) != laserCloudMapContainer.end()) {
				// transformed cloud available
				*laserCloudCornerFromMap += laserCloudMapContainer[thisKeyInd].first;
				*laserCloudSurfFromMap += laserCloudMapContainer[thisKeyInd].second;
			} else {
				// transformed cloud not available
				pcl::PointCloud<PointType> laserCloudCornerTemp = *transformPointCloud(cornerCloudKeyFrames[thisKeyInd], &cloudKeyPoses6D->points[thisKeyInd]);
				pcl::PointCloud<PointType> laserCloudSurfTemp = *transformPointCloud(surfCloudKeyFrames[thisKeyInd], &cloudKeyPoses6D->points[thisKeyInd]);
				*laserCloudCornerFromMap += laserCloudCornerTemp;
				*laserCloudSurfFromMap += laserCloudSurfTemp;
				laserCloudMapContainer[thisKeyInd] = make_pair(laserCloudCornerTemp, laserCloudSurfTemp);
			}
		}

		// Downsample the surrounding corner key frames (or map)
		downSizeFilterCorner.setInputCloud(laserCloudCornerFromMap);
		downSizeFilterCorner.filter(*laserCloudCornerFromMapDS);
		laserCloudCornerFromMapDSNum = laserCloudCornerFromMapDS->size();
		// Downsample the surrounding surf key frames (or map)
		downSizeFilterSurf.setInputCloud(laserCloudSurfFromMap);
		downSizeFilterSurf.filter(*laserCloudSurfFromMapDS);
		laserCloudSurfFromMapDSNum = laserCloudSurfFromMapDS->size();

		// clear map cache if too large
		if (laserCloudMapContainer.size() > 1000) 
			laserCloudMapContainer.clear();
	}

	void scan2SubMapOptimization() 
	{
		if (laserCloudSharpCornerLastNum > edgeFeatureMinValidNum && laserCloudSharpSurfLastNum > surfFeatureMinValidNum) 
		{
			// ROS_INFO("laserCloudSharpCornerLastNum: %d laserCloudSharpSurfLastNum: %d .", laserCloudSharpCornerLastNum, laserCloudSharpSurfLastNum);

			kdtreeCornerFromMap->setInputCloud(laserCloudCornerFromMapDS);
			kdtreeSurfFromMap->setInputCloud(laserCloudSurfFromMapDS);

			for (int iterCount = 0; iterCount < 30; iterCount++)  // 30
			{
				laserCloudOri->clear();
				coeffSel->clear();

				cornerOptimization();

				surfOptimization();

				combineOptimizationCoeffs();

				if (LMOptimization(iterCount) == true) break;
			}

			transformUpdate();
		} else {
			ROS_WARN( "Not enough features! Only %d edge and %d planar features available.", laserCloudSharpCornerLastNum, laserCloudSharpSurfLastNum);
		}
	}

	void updatePointAssociateToSubMap() 
	{
		transPointAssociateToMap = trans2Affine3f(transformTobeMapped);
	}

	void cornerOptimization() 
	{
		updatePointAssociateToSubMap();
		
		int numSearch = 0;

		// #pragma omp for
		#pragma omp parallel for num_threads(numberOfCores)
		for (int i = 0; i < laserCloudSharpCornerLastNum; i++) 
		{
			PointType pointOri, pointSel, coeff;
			std::vector<int> pointSearchInd;
			std::vector<float> pointSearchSqDis;

			pointOri = laserCloudSharpCornerLast->points[i];
			pointAssociateToMap(&pointOri, &pointSel);

			kdtreeCornerFromMap->nearestKSearch(pointSel, 5, pointSearchInd,
												pointSearchSqDis);

			cv::Mat matA1(3, 3, CV_32F, cv::Scalar::all(0));
			cv::Mat matD1(1, 3, CV_32F, cv::Scalar::all(0));
			cv::Mat matV1(3, 3, CV_32F, cv::Scalar::all(0));

			if (pointSearchSqDis[4] < 1.0) {
				float cx = 0, cy = 0, cz = 0;
				for (int j = 0; j < 5; j++) {

					cx += laserCloudCornerFromMapDS->points[pointSearchInd[j]].x;
					cy += laserCloudCornerFromMapDS->points[pointSearchInd[j]].y;
					cz += laserCloudCornerFromMapDS->points[pointSearchInd[j]].z;
				}
				cx /= 5; cy /= 5; cz /= 5;

				float a11 = 0, a12 = 0, a13 = 0, a22 = 0, a23 = 0, a33 = 0;
				for (int j = 0; j < 5; j++) 
				{
					float ax = 0, ay = 0, az = 0;

					ax = laserCloudCornerFromMapDS->points[pointSearchInd[j]].x - cx;
					ay = laserCloudCornerFromMapDS->points[pointSearchInd[j]].y - cy;
					az = laserCloudCornerFromMapDS->points[pointSearchInd[j]].z - cz;

					a11 += ax * ax; a12 += ax * ay; a13 += ax * az; a22 += ay * ay; a23 += ay * az; a33 += az * az;
				}
				a11 /= 5; a12 /= 5; a13 /= 5; a22 /= 5; a23 /= 5; a33 /= 5;

				matA1.at<float>(0, 0) = a11;
				matA1.at<float>(0, 1) = a12;
				matA1.at<float>(0, 2) = a13;
				matA1.at<float>(1, 0) = a12;
				matA1.at<float>(1, 1) = a22;
				matA1.at<float>(1, 2) = a23;
				matA1.at<float>(2, 0) = a13;
				matA1.at<float>(2, 1) = a23;
				matA1.at<float>(2, 2) = a33;

				cv::eigen(matA1, matD1, matV1);

				if (matD1.at<float>(0, 0) > 3 * matD1.at<float>(0, 1)) 
				{
					float x0 = pointSel.x;
					float y0 = pointSel.y;
					float z0 = pointSel.z;
					float x1 = cx + 0.1 * matV1.at<float>(0, 0);
					float y1 = cy + 0.1 * matV1.at<float>(0, 1);
					float z1 = cz + 0.1 * matV1.at<float>(0, 2);
					float x2 = cx - 0.1 * matV1.at<float>(0, 0);
					float y2 = cy - 0.1 * matV1.at<float>(0, 1);
					float z2 = cz - 0.1 * matV1.at<float>(0, 2);

					float a012 = sqrt(((x0 - x1) * (y0 - y2) - (x0 - x2) * (y0 - y1)) *
										((x0 - x1) * (y0 - y2) - (x0 - x2) * (y0 - y1)) +
										((x0 - x1) * (z0 - z2) - (x0 - x2) * (z0 - z1)) *
										((x0 - x1) * (z0 - z2) - (x0 - x2) * (z0 - z1)) +
										((y0 - y1) * (z0 - z2) - (y0 - y2) * (z0 - z1)) *
										((y0 - y1) * (z0 - z2) - (y0 - y2) * (z0 - z1)));

					float l12 = sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) + (z1 - z2) * (z1 - z2));

					float la = ((y1 - y2) * ((x0 - x1) * (y0 - y2) - (x0 - x2) * (y0 - y1)) +
								(z1 - z2) * ((x0 - x1) * (z0 - z2) - (x0 - x2) * (z0 - z1))) /
								a012 / l12;

					float lb = -((x1 - x2) * ((x0 - x1) * (y0 - y2) - (x0 - x2) * (y0 - y1)) -
								(z1 - z2) * ((y0 - y1) * (z0 - z2) - (y0 - y2) * (z0 - z1))) /
								a012 / l12;

					float lc = -((x1 - x2) * ((x0 - x1) * (z0 - z2) - (x0 - x2) * (z0 - z1)) +
								(y1 - y2) * ((y0 - y1) * (z0 - z2) - (y0 - y2) * (z0 - z1))) /
								a012 / l12;

					float ld2 = a012 / l12;

					float s = 1 - 0.9 * fabs(ld2);

					coeff.x = s * la;
					coeff.y = s * lb;
					coeff.z = s * lc;
					coeff.intensity = s * ld2;

					if (s > 0.1) 
					{
						laserCloudOriCornerVec[i] = pointOri;
						coeffSelCornerVec[i] = coeff;
						laserCloudOriCornerFlag[i] = true;
						
						numSearch++;
					}
				}
			}
		}

		// ROS_WARN("Corner numSearch: [%d / %d]", numSearch, laserCloudSharpCornerLastNum);
	}

	void surfOptimization() 
	{
		updatePointAssociateToSubMap();
		
		int numSearch = 0;

		// #pragma omp for
		#pragma omp parallel for num_threads(numberOfCores)
		for (int i = 0; i < laserCloudSharpSurfLastNum; i++) 
		{
			PointType pointOri, pointSel, coeff;
			std::vector<int> pointSearchInd;
			std::vector<float> pointSearchSqDis;

			pointOri = laserCloudSharpSurfLast->points[i];
			pointAssociateToMap(&pointOri, &pointSel);

			kdtreeSurfFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);

			Eigen::Matrix<float, 5, 3> matA0;
			Eigen::Matrix<float, 5, 1> matB0;
			Eigen::Vector3f matX0;

			matA0.setZero();
			matB0.fill(-1);
			matX0.setZero();

			if (pointSearchSqDis[4] < 1.0) {
				for (int j = 0; j < 5; j++) {
					matA0(j, 0) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].x;
					matA0(j, 1) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].y;
					matA0(j, 2) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].z;
				}

				matX0 = matA0.colPivHouseholderQr().solve(matB0);

				float pa = matX0(0, 0);
				float pb = matX0(1, 0);
				float pc = matX0(2, 0);
				float pd = 1;

				float ps = sqrt(pa * pa + pb * pb + pc * pc);
				pa /= ps; pb /= ps; pc /= ps; pd /= ps;

				bool planeValid = true;
				for (int j = 0; j < 5; j++) {
					if (fabs(pa * laserCloudSurfFromMapDS->points[pointSearchInd[j]].x +
							pb * laserCloudSurfFromMapDS->points[pointSearchInd[j]].y +
							pc * laserCloudSurfFromMapDS->points[pointSearchInd[j]].z +
							pd) > 0.2) {
						planeValid = false;
						break;
					}
				}

				if (planeValid) {
					float pd2 = pa * pointSel.x + pb * pointSel.y + pc * pointSel.z + pd;

					float s = 1 - 0.9 * fabs(pd2) / sqrt(sqrt(pointSel.x * pointSel.x + pointSel.y * pointSel.y + pointSel.z * pointSel.z));

					coeff.x = s * pa;
					coeff.y = s * pb;
					coeff.z = s * pc;
					coeff.intensity = s * pd2;

					if (s > 0.1) {
						laserCloudOriSurfVec[i] = pointOri;
						coeffSelSurfVec[i] = coeff;
						laserCloudOriSurfFlag[i] = true;
						
						numSearch++;
					}
				}
			}
		}

		// ROS_WARN("Surf numSearch: [%d / %d]", numSearch, laserCloudSharpSurfLastNum);

	}

	void combineOptimizationCoeffs() 
	{
		// combine corner coeffs
		for (int i = 0; i < laserCloudSharpCornerLastNum; ++i) 
		{
			if (laserCloudOriCornerFlag[i] == true) {
				laserCloudOri->push_back(laserCloudOriCornerVec[i]);
				coeffSel->push_back(coeffSelCornerVec[i]);
			}
		}
		// combine surf coeffs
		for (int i = 0; i < laserCloudSharpSurfLastNum; ++i) 
		{
			if (laserCloudOriSurfFlag[i] == true) {
				laserCloudOri->push_back(laserCloudOriSurfVec[i]);
				coeffSel->push_back(coeffSelSurfVec[i]);
			}
		}
		// reset flag for next iteration
		std::fill(laserCloudOriCornerFlag.begin(), laserCloudOriCornerFlag.end(), false);
		std::fill(laserCloudOriSurfFlag.begin(), laserCloudOriSurfFlag.end(), false);
	}

	bool LMOptimization(int iterCount) 
	{
		// This optimization is from the original loam_velodyne by Ji Zhang, need to
		// cope with coordinate transformation lidar <- camera      ---     camera
		// <- lidar x = z                ---     x = y y = x                --- y =
		// z z = y                ---     z = x roll = yaw           ---     roll =
		// pitch pitch = roll         ---     pitch = yaw yaw = pitch          ---
		// yaw = roll

		// lidar -> camera
		float srx = sin(transformTobeMapped[1]);
		float crx = cos(transformTobeMapped[1]);
		float sry = sin(transformTobeMapped[2]);
		float cry = cos(transformTobeMapped[2]);
		float srz = sin(transformTobeMapped[0]);
		float crz = cos(transformTobeMapped[0]);

		int laserCloudSelNum = laserCloudOri->size();
		if (laserCloudSelNum < 50) {
			return false;
		}

		cv::Mat matA(laserCloudSelNum, 6, CV_32F, cv::Scalar::all(0));
		cv::Mat matAt(6, laserCloudSelNum, CV_32F, cv::Scalar::all(0));
		cv::Mat matAtA(6, 6, CV_32F, cv::Scalar::all(0));
		cv::Mat matB(laserCloudSelNum, 1, CV_32F, cv::Scalar::all(0));
		cv::Mat matAtB(6, 1, CV_32F, cv::Scalar::all(0));
		cv::Mat matX(6, 1, CV_32F, cv::Scalar::all(0));
		cv::Mat matP(6, 6, CV_32F, cv::Scalar::all(0));

		PointType pointOri, coeff;

		// #pragma omp for
		#pragma omp parallel for num_threads(numberOfCores)
		for (int i = 0; i < laserCloudSelNum; i++) 
		{
			// lidar -> camera
			pointOri.x = laserCloudOri->points[i].y;
			pointOri.y = laserCloudOri->points[i].z;
			pointOri.z = laserCloudOri->points[i].x;
			// lidar -> camera
			coeff.x = coeffSel->points[i].y;
			coeff.y = coeffSel->points[i].z;
			coeff.z = coeffSel->points[i].x;
			coeff.intensity = coeffSel->points[i].intensity;
			// in camera
			float arx = (crx * sry * srz * pointOri.x + crx * crz * sry * pointOri.y - srx * sry * pointOri.z) * coeff.x +
						(-srx * srz * pointOri.x - crz * srx * pointOri.y - crx * pointOri.z) * coeff.y +
						(crx * cry * srz * pointOri.x + crx * cry * crz * pointOri.y - cry * srx * pointOri.z) * coeff.z;

			float ary = ((cry * srx * srz - crz * sry) * pointOri.x + (sry * srz + cry * crz * srx) * pointOri.y + crx * cry * pointOri.z) * coeff.x +
						((-cry * crz - srx * sry * srz) * pointOri.x + (cry * srz - crz * srx * sry) * pointOri.y - crx * sry * pointOri.z) * coeff.z;

			float arz = ((crz * srx * sry - cry * srz) * pointOri.x + (-cry * crz - srx * sry * srz) * pointOri.y) * coeff.x +
						(crx * crz * pointOri.x - crx * srz * pointOri.y) * coeff.y +
						((sry * srz + cry * crz * srx) * pointOri.x + (crz * sry - cry * srx * srz) * pointOri.y) * coeff.z;
			// lidar -> camera
			matA.at<float>(i, 0) = arz;
			matA.at<float>(i, 1) = arx;
			matA.at<float>(i, 2) = ary;
			matA.at<float>(i, 3) = coeff.z;
			matA.at<float>(i, 4) = coeff.x;
			matA.at<float>(i, 5) = coeff.y;
			matB.at<float>(i, 0) = -coeff.intensity;
		}

		cv::transpose(matA, matAt);
		matAtA = matAt * matA;
		matAtB = matAt * matB;
		cv::solve(matAtA, matAtB, matX, cv::DECOMP_QR);

		if (iterCount == 0) {
			cv::Mat matE(1, 6, CV_32F, cv::Scalar::all(0));
			cv::Mat matV(6, 6, CV_32F, cv::Scalar::all(0));
			cv::Mat matV2(6, 6, CV_32F, cv::Scalar::all(0));

			cv::eigen(matAtA, matE, matV);
			matV.copyTo(matV2);

			isDegenerate = false;
			float eignThre[6] = {100, 100, 100, 100, 100, 100};
			for (int i = 5; i >= 0; i--) 
			{
				if (matE.at<float>(0, i) < eignThre[i]) {
					for (int j = 0; j < 6; j++) 
					{
						matV2.at<float>(i, j) = 0;
					}
					isDegenerate = true;
				} else {
					break;
				}
			}
			matP = matV.inv() * matV2;
		}

		if (isDegenerate) 
		{
			cv::Mat matX2(6, 1, CV_32F, cv::Scalar::all(0));
			matX.copyTo(matX2);
			matX = matP * matX2;
		}

		transformTobeMapped[0] += matX.at<float>(0, 0);
		transformTobeMapped[1] += matX.at<float>(1, 0);
		transformTobeMapped[2] += matX.at<float>(2, 0);
		transformTobeMapped[3] += matX.at<float>(3, 0);
		transformTobeMapped[4] += matX.at<float>(4, 0);
		transformTobeMapped[5] += matX.at<float>(5, 0);

		float deltaR = sqrt(pow(pcl::rad2deg(matX.at<float>(0, 0)), 2) +
							pow(pcl::rad2deg(matX.at<float>(1, 0)), 2) +
							pow(pcl::rad2deg(matX.at<float>(2, 0)), 2));
		float deltaT = sqrt(pow(matX.at<float>(3, 0) * 100, 2) +
							pow(matX.at<float>(4, 0) * 100, 2) +
							pow(matX.at<float>(5, 0) * 100, 2));
		
		if (deltaR < 0.005 && deltaT < 0.05) {
			ROS_WARN("Front ---> iterCount: %d, deltaR: %f, deltaT: %f", iterCount, deltaR, deltaT);
			return true;  // converged
		}
		return false;  // keep optimizing
	}

	void transformUpdate() 
	{
		if (cloudInfo.imuAvailable == true) 
		{
			if (std::abs(cloudInfo.imuPitchInit) < 1.4) 
			{
				double imuWeight = imuRPYWeight;
				tf::Quaternion imuQuaternion;
				tf::Quaternion transformQuaternion;
				double rollMid, pitchMid, yawMid;

				// slerp roll
				transformQuaternion.setRPY(transformTobeMapped[0], 0, 0);
				imuQuaternion.setRPY(cloudInfo.imuRollInit, 0, 0);
				tf::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid);
				transformTobeMapped[0] = rollMid;

				// slerp pitch
				transformQuaternion.setRPY(0, transformTobeMapped[1], 0);
				imuQuaternion.setRPY(0, cloudInfo.imuPitchInit, 0);
				tf::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid);
				transformTobeMapped[1] = pitchMid;
			}
		}

		transformTobeMapped[0] = constraintTransformation(transformTobeMapped[0], rotation_tollerance);
		transformTobeMapped[1] = constraintTransformation(transformTobeMapped[1], rotation_tollerance);
		transformTobeMapped[5] = constraintTransformation(transformTobeMapped[5], z_tollerance);

		incrementalOdometryAffineBack = trans2Affine3f(transformTobeMapped);
	}



	void publishOdometry() 
	{
		// Publish odometry for ROS (global)
		nav_msgs::Odometry laserOdometryROS;
		laserOdometryROS.header.stamp = timeLaserInfoStamp;
		laserOdometryROS.header.frame_id = odometryFrame;
		laserOdometryROS.child_frame_id = "odom_estimation";
		// laserOdometryROS.child_frame_id = lidarFrame;
		laserOdometryROS.pose.pose.position.x = transformTobeMapped[3];
		laserOdometryROS.pose.pose.position.y = transformTobeMapped[4];
		laserOdometryROS.pose.pose.position.z = transformTobeMapped[5];
		laserOdometryROS.pose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);
		pubLaserOdometryGlobal.publish(laserOdometryROS);
		
		// Publish TF
		static tf::TransformBroadcaster br;
		tf::Transform t_odom_to_lidar = tf::Transform(
				tf::createQuaternionFromRPY(transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]),
				tf::Vector3(transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5]));
		tf::StampedTransform trans_odom_to_lidar = tf::StampedTransform(t_odom_to_lidar, timeLaserInfoStamp, odometryFrame, "odom_estimation");
		br.sendTransform(trans_odom_to_lidar);

		// Publish odometry for ROS (incremental)
		static bool lastIncreOdomPubFlag = false;
		static nav_msgs::Odometry laserOdomIncremental;  // incremental odometry msg
		static Eigen::Affine3f increOdomAffine;  // incremental odometry in affine
		if (lastIncreOdomPubFlag == false) {
			lastIncreOdomPubFlag = true;
			laserOdomIncremental = laserOdometryROS;
			increOdomAffine = trans2Affine3f(transformTobeMapped);
		} else {
			Eigen::Affine3f affineIncre = incrementalOdometryAffineFront.inverse() * incrementalOdometryAffineBack;
			increOdomAffine = increOdomAffine * affineIncre;
			float x, y, z, roll, pitch, yaw;
			pcl::getTranslationAndEulerAngles(increOdomAffine, x, y, z, roll, pitch, yaw);

			laserOdomIncremental.header.stamp = timeLaserInfoStamp;
			laserOdomIncremental.header.frame_id = odometryFrame;
			// laserOdomIncremental.child_frame_id = "odom_estimation";
			laserOdometryROS.child_frame_id = lidarFrame;
			laserOdomIncremental.pose.pose.position.x = x;
			laserOdomIncremental.pose.pose.position.y = y;
			laserOdomIncremental.pose.pose.position.z = z;
			laserOdomIncremental.pose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(roll, pitch, yaw);
		}
		pubLaserOdometryIncremental.publish(laserOdomIncremental);
		// ROS_INFO("Finshed  publishOdometry !");
	}
};

int main(int argc, char **argv) 
{
	ros::init(argc, argv, "lis_slam");

	OdomEstimationNode ODN;

	ROS_INFO("\033[1;32m----> Odom Estimation Node Started.\033[0m");

	ros::spin();

	return 0;
}