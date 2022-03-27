// Author of  EPSC-LOAM : QZ Wang
// Email wangqingzhi27@outlook.com
#define PCL_NO_PRECOMPILE
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam_unstable/nonlinear/IncrementalFixedLagSmoother.h>

#include "lis_slam/semantic_info.h"
#include "lis_slam/finishMap.h"

#include "utility.h"
#include "common.h"

#include "epscGeneration.h"
#include "subMap.h"
#include "registration.h"


#define USING_SINGLE_TARGET false
#define USING_SUBMAP_TARGET false
#define USING_SLIDING_TARGET true
#define USING_MULTI_KEYFRAME_TARGET false

#define USING_SEMANTIC_FEATURE false
#define USING_LOAM_FEATURE true



using namespace gtsam;

using symbol_shorthand::B;  // Bias  (ax,ay,az,gx,gy,gz)
using symbol_shorthand::G;  // GPS pose
using symbol_shorthand::V;  // Vel   (xdot,ydot,zdot)
using symbol_shorthand::X;  // Pose3 (x,y,z,r,p,y)

bool FINISHMAP = false;

std::mutex subMapMtx;
std::deque<int> subMapIndexQueue;
map<int, submap_Ptr> subMapInfo;


vector<pair<int, int>> loopIndexQueue;
vector<gtsam::Pose3> loopPoseQueue;
vector<gtsam::noiseModel::Diagonal::shared_ptr> loopNoiseQueue;

map<int, vector<Eigen::Affine3f>> keyFrame2SubMapPose;

multimap<int, int> loopIndexContainer;  // from new to old
multimap<int, int> loopIndexContainerTest;


gtsam::Pose3 pclPointTogtsamPose3(PointTypePose thisPoint) {
    return gtsam::Pose3(gtsam::Rot3::RzRyRx(double(thisPoint.roll), double(thisPoint.pitch), double(thisPoint.yaw)),
                        gtsam::Point3(double(thisPoint.x), double(thisPoint.y), double(thisPoint.z)));
}

gtsam::Pose3 trans2gtsamPose(float transformIn[]) {
    return gtsam::Pose3(gtsam::Rot3::RzRyRx(transformIn[0], transformIn[1], transformIn[2]),
                        gtsam::Point3(transformIn[3], transformIn[4], transformIn[5]));
}

Eigen::Affine3f odom2affine(nav_msgs::Odometry odom) {
	double x, y, z, roll, pitch, yaw;
	x = odom.pose.pose.position.x;
	y = odom.pose.pose.position.y;
	z = odom.pose.pose.position.z;
	tf::Quaternion orientation;
	tf::quaternionMsgToTF(odom.pose.pose.orientation, orientation);
	tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);
	return pcl::getTransformation(x, y, z, roll, pitch, yaw);
}


class SubMapOdometryNode : public SubMapManager<PointXYZIL>
{
 public:
	// bool FINISHMAP = false;

 	ros::ServiceServer service;
    ros::Subscriber subCloud;
    ros::Subscriber subIMU;
	ros::Subscriber subOdom;

	ros::Publisher pubCloudRegisteredRaw;
    
    ros::Publisher pubCloudCurSubMap;
    ros::Publisher pubSubMapId;

    ros::Publisher pubKeyFrameOdometryGlobal;
    ros::Publisher pubKeyFrameOdometryIncremental;
  
    ros::Publisher pubKeyFramePoseGlobal;
    ros::Publisher pubKeyFramePath;

    ros::Publisher pubLoopConstraintEdge;
    ros::Publisher pubSEPSC;
    ros::Publisher pubFEPSC;
    ros::Publisher pubEPSC;
    ros::Publisher pubSC;
    ros::Publisher pubISC;
    ros::Publisher pubSSC;


    std::mutex seMtx;
    std::mutex imuMtx;
    std::mutex odomMtx;
    std::deque<lis_slam::semantic_info> seInfoQueue;
    std::deque<sensor_msgs::Imu> imuQueOpt;
    std::deque<sensor_msgs::Imu> imuQueImu;
    std::deque<nav_msgs::Odometry> odomQueue;

    lis_slam::semantic_info cloudInfo;
    
    int keyFrameID = 0;
    int subMapID = 0;
    // std::deque<int> keyFrameQueue;
    std::deque<keyframe_Ptr> keyFrameQueue;
    map<int, keyframe_Ptr> keyFrameInfo;

    keyframe_Ptr currentKeyFrame = keyframe_Ptr(new keyframe_t);
    submap_Ptr currentSubMap = submap_Ptr(new submap_t);
    localMap_Ptr localMap = localMap_Ptr(new localMap_t);

    pcl::PointCloud<PointTypePose>::Ptr subMapPose6D; 
    pcl::PointCloud<PointType>::Ptr subMapPose3D;
    
    map<int, PointType> subMapPosesIndex3D;
    map<int, PointTypePose> subMapPosesIndex6D;
    
    pcl::PointCloud<PointType>::Ptr keyFramePoses3D;
    pcl::PointCloud<PointTypePose>::Ptr keyFramePoses6D;

	pcl::KdTreeFLANN<PointTypePose>::Ptr kdtreeFromKeyFramePose6D;
    
    map<int, PointType> keyFramePosesIndex3D;
    map<int, PointTypePose> keyFramePosesIndex6D;

	map<int, int> keyframeInSubmapIndex;

    pcl::PointCloud<PointXYZIL>::Ptr laserCloudCornerLast; // corner feature set from odoOptimization
    pcl::PointCloud<PointXYZIL>::Ptr laserCloudSurfLast; // surf feature set from odoOptimization
    pcl::PointCloud<PointXYZIL>::Ptr laserCloudCornerLastDS; // downsampled corner featuer set from odoOptimization
    pcl::PointCloud<PointXYZIL>::Ptr laserCloudSurfLastDS; // downsampled surf featuer set from odoOptimization

    pcl::PointCloud<PointXYZIL>::Ptr laserCloudCornerFromSubMap;
    pcl::PointCloud<PointXYZIL>::Ptr laserCloudSurfFromSubMap;

    pcl::PointCloud<PointXYZIL>::Ptr laserCloudFromPre;


    int laserCloudCornerLastDSNum = 0;
    int laserCloudSurfLastDSNum = 0;

    ros::Time timeLaserInfoStamp;
    ros::Time timeSubMapInfoStamp;
    double timeLaserInfoCur = -1;
    double timeLaserInfoPre = -1;

    float transformTobeSubMapped[6];
    float transPredictionMapped[6];

	Eigen::Affine3f transBef;
	Eigen::Affine3f transBef2Aft = Eigen::Affine3f::Identity();
    
    float transformCurFrame2Submap[6];
    float transformCurSubmap[6];
    float subMapYawSum=0;
    int  curSubMapSize = 0;


    Eigen::Affine3f transPointAssociateToSubMap;
    Eigen::Affine3f incrementalOdometryAffineFront;
    Eigen::Affine3f incrementalOdometryAffineBack;

    pcl::KdTreeFLANN<PointXYZIL>::Ptr kdtreeCornerFromSubMap;
    pcl::KdTreeFLANN<PointXYZIL>::Ptr kdtreeSurfFromSubMap;

    pcl::PointCloud<PointXYZIL>::Ptr laserCloudOri;
    pcl::PointCloud<PointXYZIL>::Ptr coeffSel;

    std::vector<PointXYZIL> laserCloudOriCornerVec; // corner point holder for parallel computation
    std::vector<PointXYZIL> coeffSelCornerVec;
    std::vector<bool> laserCloudOriCornerFlag;
    std::vector<PointXYZIL> laserCloudOriSurfVec; // surf point holder for parallel computation
    std::vector<PointXYZIL> coeffSelSurfVec;
    std::vector<bool> laserCloudOriSurfFlag;
    
    bool isDegenerate = false;
    Eigen::Matrix<float, 6, 6> matP;

	float deltaR = 100;
	float deltaT = 100;

	// ---- IMUPreintegration start ----  
	bool systemInitialized = false;

	gtsam::noiseModel::Diagonal::shared_ptr priorPoseNoise;
	gtsam::noiseModel::Diagonal::shared_ptr priorVelNoise;
	gtsam::noiseModel::Diagonal::shared_ptr priorBiasNoise;
	gtsam::noiseModel::Diagonal::shared_ptr correctionNoise;
	gtsam::Vector noiseModelBetweenBias;

	gtsam::PreintegratedImuMeasurements* imuIntegratorOpt_;
	gtsam::PreintegratedImuMeasurements* imuIntegratorImu_;
	
	gtsam::Pose3 prevPose_;
	gtsam::Vector3 prevVel_;
	gtsam::NavState prevState_;
	gtsam::imuBias::ConstantBias prevBias_;

	gtsam::NavState prevStateOdom;
	gtsam::imuBias::ConstantBias prevBiasOdom;

	bool doneFirstOpt = false;
	double lastImuT_imu = -1;
	double lastImuT_opt = -1;

	gtsam::ISAM2 optimizer;
	gtsam::NonlinearFactorGraph graphFactors;
	gtsam::Values graphValues;

	const double delta_t = 0.05;

	int key = 1;

	// gtsam::Pose3 imu2Lidar =
	// 	gtsam::Pose3(gtsam::Rot3(1, 0, 0, 0),
	// 				 gtsam::Point3(-extTrans.x(), -extTrans.y(), -extTrans.z()));
	// gtsam::Pose3 lidar2Imu =
	// 	gtsam::Pose3(gtsam::Rot3(1, 0, 0, 0),
	// 				 gtsam::Point3(extTrans.x(), extTrans.y(), extTrans.z()));

	gtsam::Pose3 lidar2Imu =
		gtsam::Pose3(gtsam::Rot3(1, 0, 0, 0),
					 gtsam::Point3(-extTrans.x(), -extTrans.y(), -extTrans.z()));
	gtsam::Pose3 imu2Lidar =
		gtsam::Pose3(gtsam::Rot3(1, 0, 0, 0),
					 gtsam::Point3(extTrans.x(), extTrans.y(), extTrans.z()));

	tf::TransformListener tfListener;
	tf::StampedTransform lidar2Baselink;

    std::mutex imuOdomMtx;
	deque<nav_msgs::Odometry> imuOdomQueue;
	
	ros::Publisher pubKeyframeIMUOdometry;
	ros::Publisher pubImuOdometry;

	ros::Publisher pubLidarPath;
	ros::Publisher pubLidarOdometry;
	ros::Publisher pubLidarIMUOdometry;
	
	
	// ---- test publisher ----  
    ros::Publisher pubTest1;
    ros::Publisher pubTest2;

    ros::Publisher pubLoopConstraintEdgeTest;
    
	ros::Publisher pubTestPre;
    ros::Publisher pubTestCur;
    ros::Publisher pubTestCurLoop;
    ros::Publisher pubTestCurICP;

    
    SubMapOdometryNode() 
    {
    	service = nh.advertiseService("finish_map", &SubMapOdometryNode::finish_map_callback, this);

        subCloud = nh.subscribe<lis_slam::semantic_info>( "lis_slam/semantic_fusion/semantic_info", 100, &SubMapOdometryNode::semanticInfoHandler, this, ros::TransportHints().tcpNoDelay());
        subIMU   = nh.subscribe<sensor_msgs::Imu>(imuTopic, 2000, &SubMapOdometryNode::imuHandler, this, ros::TransportHints().tcpNoDelay());
        subOdom   = nh.subscribe<nav_msgs::Odometry>(odomTopic + "/front", 200, &SubMapOdometryNode::odomHandler, this, ros::TransportHints().tcpNoDelay());
        
        pubKeyFrameOdometryGlobal      = nh.advertise<nav_msgs::Odometry> (odomTopic + "/keyframe", 200);
        pubKeyFrameOdometryIncremental = nh.advertise<nav_msgs::Odometry> (odomTopic + "/keyframe_incremental", 200);
      
        pubKeyFramePoseGlobal = nh.advertise<sensor_msgs::PointCloud2>("lis_slam/make_submap/keyframe_id", 1);
        pubKeyFramePath       = nh.advertise<nav_msgs::Path>("lis_slam/make_submap/keyframe_path", 1);

        pubCloudRegisteredRaw = nh.advertise<sensor_msgs::PointCloud2>("lis_slam/make_submap/registered_raw", 1);
        
        pubCloudCurSubMap = nh.advertise<sensor_msgs::PointCloud2>("lis_slam/make_submap/submap", 1); 
        pubSubMapId       = nh.advertise<sensor_msgs::PointCloud2>("lis_slam/make_submap/submap_id", 1);

        pubLoopConstraintEdge = nh.advertise<visualization_msgs::MarkerArray>("lis_slam/make_submap/loop_closure_constraints", 1);
        pubSEPSC = nh.advertise<sensor_msgs::Image>("global_descriptor_sepsc", 100);
        pubFEPSC = nh.advertise<sensor_msgs::Image>("global_descriptor_fepsc", 100);
		pubEPSC = nh.advertise<sensor_msgs::Image>("global_descriptor_epsc", 100);
        pubSC   = nh.advertise<sensor_msgs::Image>("global_descriptor_sc", 100);
        pubISC  = nh.advertise<sensor_msgs::Image>("global_descriptor_isc", 100);
        pubSSC  = nh.advertise<sensor_msgs::Image>("global_descriptor_ssc", 100);
        
  		pubKeyframeIMUOdometry = nh.advertise<nav_msgs::Odometry>(odomTopic + "/keyframe_imu", 2000);
		pubImuOdometry         = nh.advertise<nav_msgs::Odometry>(odomTopic + "/imu", 2000);
    	
		pubLidarPath        = nh.advertise<nav_msgs::Path>("lis_slam/make_submap/lidar_path", 1);
  		pubLidarOdometry    = nh.advertise<nav_msgs::Odometry>(odomTopic + "/lidar", 2000);
  		pubLidarIMUOdometry = nh.advertise<nav_msgs::Odometry>(odomTopic + "/lidar_imu", 2000);
  
  		
        pubTest1 = nh.advertise<sensor_msgs::PointCloud2>("lis_slam/make_submap/test_1", 1);
        pubTest2 = nh.advertise<sensor_msgs::PointCloud2>("lis_slam/make_submap/test_2", 1);
        
        pubLoopConstraintEdgeTest = nh.advertise<visualization_msgs::MarkerArray>("lis_slam/make_submap/loop_closure_constraints_test", 1);

		pubTestPre = nh.advertise<sensor_msgs::PointCloud2>("Pre", 1);
        pubTestCur = nh.advertise<sensor_msgs::PointCloud2>("Cur", 1);
        pubTestCurLoop = nh.advertise<sensor_msgs::PointCloud2>("CurLoop", 1);
        pubTestCurICP = nh.advertise<sensor_msgs::PointCloud2>("CurICP", 1);

        allocateMemory();
    }

    void allocateMemory() 
    {
        subMapPose6D.reset(new pcl::PointCloud<PointTypePose>()); 
        subMapPose3D.reset(new pcl::PointCloud<PointType>());

        keyFramePoses3D.reset(new pcl::PointCloud<PointType>());
        keyFramePoses6D.reset(new pcl::PointCloud<PointTypePose>());

		kdtreeFromKeyFramePose6D.reset(new pcl::KdTreeFLANN<PointTypePose>());

        laserCloudCornerLast.reset(new pcl::PointCloud<PointXYZIL>()); 
        laserCloudSurfLast.reset(new pcl::PointCloud<PointXYZIL>()); 
        laserCloudCornerLastDS.reset(new pcl::PointCloud<PointXYZIL>()); 
        laserCloudSurfLastDS.reset(new pcl::PointCloud<PointXYZIL>()); 

        laserCloudCornerFromSubMap.reset(new pcl::PointCloud<PointXYZIL>());
        laserCloudSurfFromSubMap.reset(new pcl::PointCloud<PointXYZIL>());

		laserCloudFromPre.reset(new pcl::PointCloud<PointXYZIL>());


        for (int i = 0; i < 6; ++i)
        {
            transformTobeSubMapped[i] = 0;
            transPredictionMapped[i]=0.0;
            
            transformCurFrame2Submap[i] = 0;
            transformCurSubmap[i] = 0;
        }

        kdtreeCornerFromSubMap.reset(new pcl::KdTreeFLANN<PointXYZIL>());
        kdtreeSurfFromSubMap.reset(new pcl::KdTreeFLANN<PointXYZIL>());

        laserCloudOri.reset(new pcl::PointCloud<PointXYZIL>());
        coeffSel.reset(new pcl::PointCloud<PointXYZIL>());

        laserCloudOriCornerVec.resize(N_SCAN * Horizon_SCAN);
        coeffSelCornerVec.resize(N_SCAN * Horizon_SCAN);
        laserCloudOriCornerFlag.resize(N_SCAN * Horizon_SCAN);
        laserCloudOriSurfVec.resize(N_SCAN * Horizon_SCAN);
        coeffSelSurfVec.resize(N_SCAN * Horizon_SCAN);
        laserCloudOriSurfFlag.resize(N_SCAN * Horizon_SCAN);
    
        std::fill(laserCloudOriCornerFlag.begin(), laserCloudOriCornerFlag.end(), false);
        std::fill(laserCloudOriSurfFlag.begin(), laserCloudOriSurfFlag.end(), false);
        
        matP.setZero();


		boost::shared_ptr<gtsam::PreintegrationParams> p = gtsam::PreintegrationParams::MakeSharedU(imuGravity);
		p->accelerometerCovariance = gtsam::Matrix33::Identity(3, 3) * pow(imuAccNoise, 2);  // acc white noise in continuous
		p->gyroscopeCovariance = gtsam::Matrix33::Identity(3, 3) * pow(imuGyrNoise, 2);  // gyro white noise in continuous
		p->integrationCovariance = gtsam::Matrix33::Identity(3, 3) * pow(1e-4, 2);  // error committed in integrating position from velocities
		gtsam::imuBias::ConstantBias prior_imu_bias((gtsam::Vector(6) << 0, 0, 0, 0, 0, 0).finished());;  // assume zero initial bias

		priorPoseNoise = gtsam::noiseModel::Diagonal::Sigmas(
			(gtsam::Vector(6) << 1e-2, 1e-2, 1e-3, 1e-4, 1e-4, 1e-4).finished());  // rad,rad,rad,m, m, m
		// priorPoseNoise  = gtsam::noiseModel::Diagonal::Sigmas(
		// 	(gtsam::Vector(6)<< 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1).finished()); // rad,rad,rad,m, m, m
		priorVelNoise = gtsam::noiseModel::Isotropic::Sigma(3, 1e4);  // m/s
		priorBiasNoise = gtsam::noiseModel::Isotropic::Sigma(6, 1e-3);  // 1e-2 ~ 1e-3 seems to be good
		correctionNoise = gtsam::noiseModel::Isotropic::Sigma(6, 1);  // meter
		noiseModelBetweenBias = (gtsam::Vector(6) << imuAccBiasN, imuAccBiasN, imuAccBiasN, imuGyrBiasN, imuGyrBiasN, imuGyrBiasN).finished();

		imuIntegratorImu_ = new gtsam::PreintegratedImuMeasurements(p, prior_imu_bias);  // setting up the IMU integration for IMU message thread
		imuIntegratorOpt_ = new gtsam::PreintegratedImuMeasurements(p, prior_imu_bias);  // setting up the IMU integration for optimization

    }


	void resetOptimization() {
		gtsam::ISAM2Params optParameters;
		optParameters.relinearizeThreshold = 0.1;
		optParameters.relinearizeSkip = 1;
		optimizer = gtsam::ISAM2(optParameters);

		gtsam::NonlinearFactorGraph newGraphFactors;
		graphFactors = newGraphFactors;

		gtsam::Values NewGraphValues;
		graphValues = NewGraphValues;
	}


	void resetParams() {
		lastImuT_imu = -1;
		doneFirstOpt = false;
		systemInitialized = false;
	}


	bool finish_map_callback(lis_slam::finishMap::Request &request, lis_slam::finishMap::Response &response) {
		response.succeed = true;
		FINISHMAP = true;
		return response.succeed;
	}

    void semanticInfoHandler(const lis_slam::semantic_info::ConstPtr &msgIn) 
    {
        std::lock_guard<std::mutex> lock(seMtx);
        seInfoQueue.push_back(*msgIn);
    }


    void imuHandler(const sensor_msgs::Imu::ConstPtr& imu_raw) 
    {
        std::lock_guard<std::mutex> lock(imuMtx);

        sensor_msgs::Imu thisImu = imuConverter(*imu_raw);

        imuQueOpt.push_back(thisImu);
        imuQueImu.push_back(thisImu);

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

		if (doneFirstOpt == false) return;
		double imuTime = ROS_TIME(&thisImu);
		double dt = (lastImuT_imu < 0) ? (1.0 / 100.0) : (imuTime - lastImuT_imu);
		lastImuT_imu = imuTime;

	    // integrate this single imu message
		imuIntegratorImu_->integrateMeasurement(
				gtsam::Vector3(thisImu.linear_acceleration.x, thisImu.linear_acceleration.y, thisImu.linear_acceleration.z),
				gtsam::Vector3(thisImu.angular_velocity.x, thisImu.angular_velocity.y, thisImu.angular_velocity.z),
				dt);

		// predict odometry
    	gtsam::NavState currentState = imuIntegratorImu_->predict(prevStateOdom, prevBiasOdom);

		// publish odometry
		nav_msgs::Odometry odometry;
		odometry.header.stamp = thisImu.header.stamp;
		odometry.header.frame_id = odometryFrame;
		odometry.child_frame_id = "odom_imu";

		// transform imu pose to ldiar
    	gtsam::Pose3 imuPose = gtsam::Pose3(currentState.quaternion(), currentState.position());
    	gtsam::Pose3 lidarPose = imuPose.compose(imu2Lidar);

		odometry.pose.pose.position.x = lidarPose.translation().x();
		odometry.pose.pose.position.y = lidarPose.translation().y();
		odometry.pose.pose.position.z = lidarPose.translation().z();
		odometry.pose.pose.orientation.x = lidarPose.rotation().toQuaternion().x();
		odometry.pose.pose.orientation.y = lidarPose.rotation().toQuaternion().y();
		odometry.pose.pose.orientation.z = lidarPose.rotation().toQuaternion().z();
		odometry.pose.pose.orientation.w = lidarPose.rotation().toQuaternion().w();

		odometry.twist.twist.linear.x = currentState.velocity().x();
		odometry.twist.twist.linear.y = currentState.velocity().y();
		odometry.twist.twist.linear.z = currentState.velocity().z();
		odometry.twist.twist.angular.x = thisImu.angular_velocity.x + prevBiasOdom.gyroscope().x();
		odometry.twist.twist.angular.y = thisImu.angular_velocity.y + prevBiasOdom.gyroscope().y();
		odometry.twist.twist.angular.z = thisImu.angular_velocity.z + prevBiasOdom.gyroscope().z();
		pubImuOdometry.publish(odometry);


        std::lock_guard<std::mutex> lock1(imuOdomMtx);
	    imuOdomQueue.push_back(odometry);

		// cout << std::setprecision(6);
		// cout << "CurrentState IMU velocity: " << endl;
		// cout << "x: " << odometry.twist.twist.linear.x <<
		//       ", y: " << odometry.twist.twist.linear.y <<
		//       ", z: " << odometry.twist.twist.linear.z << endl;
		// cout << "CurrentState IMU gyro: " << endl;
		// cout << "x: " << odometry.twist.twist.angular.x <<
		//       ", y: " << odometry.twist.twist.angular.y <<
		//       ", z: " << odometry.twist.twist.angular.z << endl;
		// cout << "CurrentState IMU roll pitch yaw: " << endl;
		// cout << "roll: " << lidarPose.rotation().roll() << ", pitch: " << lidarPose.rotation().pitch() << ", yaw: " << lidarPose.rotation().yaw() << endl << endl;

    }


	void odomHandler(const nav_msgs::Odometry::ConstPtr &msgIn )
	{
        std::lock_guard<std::mutex> lock(odomMtx);
        odomQueue.push_back(*msgIn);

		if(timeLaserInfoCur == -1 || keyFramePoses3D->points.size() <= 0) 
			return;

		nav_msgs::Odometry thisOdom = odomQueue.front();
		odomQueue.pop_front();

		Eigen::Affine3f odomAffine = odom2affine(thisOdom);
        Eigen::Affine3f lidarOdomAffine = transBef2Aft * odomAffine;

		Eigen::Affine3f submapAffine = trans2Affine3f(transformCurSubmap);
		Eigen::Affine3f lidar2Submap = submapAffine.inverse() * lidarOdomAffine;

		keyFrame2SubMapPose[subMapID].push_back(lidar2Submap);

        float x, y, z, roll, pitch, yaw;
        pcl::getTranslationAndEulerAngles(lidarOdomAffine, x, y, z, roll, pitch, yaw);

        // publish latest odometry
        nav_msgs::Odometry laserOdometry = thisOdom;
        laserOdometry.pose.pose.position.x = x;
        laserOdometry.pose.pose.position.y = y;
        laserOdometry.pose.pose.position.z = z;
        laserOdometry.pose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(roll, pitch, yaw);
        pubLidarOdometry.publish(laserOdometry);

		double lidarOdomTime = thisOdom.header.stamp.toSec();

        std::lock_guard<std::mutex> lock1(imuOdomMtx);
		// get latest odometry (at current IMU stamp)
		while (!imuOdomQueue.empty()) {
		if (imuOdomQueue.front().header.stamp.toSec() <= lidarOdomTime)
			imuOdomQueue.pop_front();
		else
			break;
		}

		// 有问题！
		// Eigen::Affine3f imuOdomAffineFront = odom2affine(imuOdomQueue.front());
		// Eigen::Affine3f imuOdomAffineBack = odom2affine(imuOdomQueue.back());
		// Eigen::Affine3f imuOdomAffineIncre = imuOdomAffineFront.inverse() * imuOdomAffineBack;
		// Eigen::Affine3f imuOdomAffineLast = lidarOdomAffine * imuOdomAffineIncre;
		// pcl::getTranslationAndEulerAngles(imuOdomAffineLast, x, y, z, roll, pitch, yaw);
		
		// // publish latest odometry
		// nav_msgs::Odometry laserIMUOdometry = imuOdomQueue.back();
		// laserIMUOdometry.pose.pose.position.x = x;
		// laserIMUOdometry.pose.pose.position.y = y;
		// laserIMUOdometry.pose.pose.position.z = z;
		// laserIMUOdometry.pose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(roll, pitch, yaw);
		// pubLidarIMUOdometry.publish(laserIMUOdometry);

		// // publish IMU path
		// static nav_msgs::Path imuPath;
		// static double last_path_time = -1;
		// double imuTime = imuOdomQueue.back().header.stamp.toSec();
		// if (imuTime - last_path_time > 0.1) {
		// 	last_path_time = imuTime;
		// 	geometry_msgs::PoseStamped pose_stamped;
		// 	pose_stamped.header.stamp = imuOdomQueue.back().header.stamp;
		// 	pose_stamped.header.frame_id = odometryFrame;
		// 	pose_stamped.pose = laserOdometry.pose.pose;
		// 	imuPath.poses.push_back(pose_stamped);
		// 	while (!imuPath.poses.empty() && imuPath.poses.front().header.stamp.toSec() < lidarOdomTime - 0.1)
		// 		imuPath.poses.erase(imuPath.poses.begin());
		// 	if (pubLidarPath.getNumSubscribers() != 0) {
		// 		imuPath.header.stamp = imuOdomQueue.back().header.stamp;
		// 		imuPath.header.frame_id = odometryFrame;
		// 		pubLidarPath.publish(imuPath);
		// 	}
		// }
		
	}


    /*****************************
     * @brief
     * @param input
     *****************************/
    void makeSubMapThread() 
    {
        double total_time = 0;
        int total_frame = 0;
        bool subMapFirstFlag = true;
        
        float local_map_radius = 80;
        int max_num_pts = 80000; 
        int kept_vertex_num = 800;
        float last_frame_reliable_radius = 60;
        bool map_based_dynamic_removal_on = true;
        float dynamic_removal_center_radius = 30.0;
        float dynamic_dist_thre_min = 0.3;
        float dynamic_dist_thre_max = 3.0;
        float near_dist_thre = 0.03;

        while(ros::ok())
        {
            if (seInfoQueue.size() > 0) 
            {
                std::chrono::time_point<std::chrono::system_clock> start, end;
                start = std::chrono::system_clock::now();
                
                std::lock_guard<std::mutex> lock(seMtx);

                cloudInfo = seInfoQueue.front();
                seInfoQueue.pop_front();
				
                timeLaserInfoStamp = cloudInfo.header.stamp;
				timeLaserInfoPre = timeLaserInfoCur;
                timeLaserInfoCur = cloudInfo.header.stamp.toSec();
                
                if(!keyframeInit())
                    continue;
                
                // ROS_WARN("Now (keyframeInit) time %f ms", ((std::chrono::duration<float>)(std::chrono::system_clock::now() - start)).count()*1000);

                updateInitialGuess();

                if(subMapFirstFlag)
                {
                    currentCloudInit();
					currentSubMapInit();
                    saveKeyFrame();
                    this->fisrt_submap(currentSubMap, currentKeyFrame);
					
					publishOdometry();
                    publishKeyFrameCloud();			
					
				#if USING_SLIDING_TARGET
					this->insert_local_map(localMap, currentKeyFrame, 
							local_map_radius, max_num_pts, kept_vertex_num,
							last_frame_reliable_radius, map_based_dynamic_removal_on,
							dynamic_removal_center_radius, dynamic_dist_thre_min,
							dynamic_dist_thre_max, near_dist_thre);
				#endif

                    subMapFirstFlag=false;
					timeLaserInfoPre = timeLaserInfoCur;
                    continue;
                }
                
				// 首先提取带匹配点云（注意：要在初始化当前帧点云前）
				extractTargetCloud();
                // ROS_WARN("Now (extractCloud) time %f ms", ((std::chrono::duration<float>)(std::chrono::system_clock::now() - start)).count()*1000);

                currentCloudInit();
				
				// scan2SubMapOptimizationICP();
                scan2SubMapOptimization();
                // ROS_WARN("Now (scan2SubMapOptimization) time %f ms", ((std::chrono::duration<float>)(std::chrono::system_clock::now() - start)).count()*1000);

				IMUPreintegration();
                // ROS_WARN("Now (IMUPreintegration) time %f ms", ((std::chrono::duration<float>)(std::chrono::system_clock::now() - start)).count()*1000);

				calculateTranslation();
				float accu_tran = std::max(transformCurFrame2Submap[3], transformCurFrame2Submap[4]); 
				float accu_rot = transformCurFrame2Submap[2]; 
				if(curSubMapSize > 5 && (deltaR > 0.003 || deltaT > 0.03 || 
				judge_new_submap(accu_tran, accu_rot, curSubMapSize, subMapTraMax, subMapYawMax, subMapFramesSize)))
				{
					ROS_WARN("Make %d submap  has %d  Frames !", subMapID, curSubMapSize);
					
					saveSubMap();
					publishSubMapCloud();
					
					currentSubMapInit();
					saveKeyFrame();
					this->fisrt_submap(currentSubMap, currentKeyFrame);
					
					publishOdometry();
					publishKeyFrameCloud();
				}else{

					saveKeyFrame();
					this->insert_submap(currentSubMap, currentKeyFrame, 
							local_map_radius, max_num_pts, kept_vertex_num,
							last_frame_reliable_radius, map_based_dynamic_removal_on,
							dynamic_removal_center_radius, dynamic_dist_thre_min,
							dynamic_dist_thre_max, near_dist_thre);
					subMapInfo[subMapID] = currentSubMap;

					publishOdometry();
					publishKeyFrameCloud();
					
					// ROS_WARN("Current SubMap Static Cloud Size: [%d, %d] .", currentSubMap->submap_pole->points.size(), currentSubMap->submap_ground->points.size());
					pcl::PointCloud<PointXYZIL>::Ptr tmpCloud( new pcl::PointCloud<PointXYZIL>());
					tmpCloud->points.insert(tmpCloud->points.end(), 
											currentSubMap->submap_dynamic->points.begin(), 
											currentSubMap->submap_dynamic->points.end());
					tmpCloud->points.insert(tmpCloud->points.end(), 
											currentSubMap->submap_pole->points.begin(), 
											currentSubMap->submap_pole->points.end());
					tmpCloud->points.insert(tmpCloud->points.end(), 
											currentSubMap->submap_ground->points.begin(), 
											currentSubMap->submap_ground->points.end());
					tmpCloud->points.insert(tmpCloud->points.end(), 
											currentSubMap->submap_building->points.begin(), 
											currentSubMap->submap_building->points.end());
					publishLabelCloud(&pubTest2, tmpCloud, timeLaserInfoStamp, lidarFrame);
	
				}
				
			#if USING_SLIDING_TARGET
				this->insert_local_map(localMap, currentKeyFrame, 
						local_map_radius, max_num_pts, kept_vertex_num,
						last_frame_reliable_radius, map_based_dynamic_removal_on,
						dynamic_removal_center_radius, dynamic_dist_thre_min,
						dynamic_dist_thre_max, near_dist_thre);
			#endif

		

                end = std::chrono::system_clock::now();
                std::chrono::duration<float> elapsed_seconds = end - start;
                total_frame++;
                float time_temp = elapsed_seconds.count() * 1000;
                total_time += time_temp;
                
                ROS_WARN("Average make SubMap time %f ms", total_time / total_frame);
            }

            while (seInfoQueue.size() > 20) seInfoQueue.pop_front();

			// 保存最后SubMap
			if(FINISHMAP == true && seInfoQueue.size() == 0)
			{
				ROS_WARN("Finish Make SubMap ---> Make %d submap  has %d  Frames !", subMapID, curSubMapSize);
                    
				saveSubMap();
				publishSubMapCloud();	

				return;
			}


			ros::spinOnce();
        }
    }

    bool keyframeInit()
    {
        currentKeyFrame->free_all();
        currentKeyFrame->loop_container.clear();
        
        currentKeyFrame->keyframe_id = -1;
        currentKeyFrame->timeInfoStamp = cloudInfo.header.stamp;
        currentKeyFrame->submap_id = -1;
        currentKeyFrame->id_in_submap = curSubMapSize;

        if (cloudInfo.odomAvailable == true) 
        {
            PointTypePose  point6d;
            point6d.x = cloudInfo.initialGuessX;
            point6d.y = cloudInfo.initialGuessY;
            point6d.z = cloudInfo.initialGuessZ;
            point6d.intensity = keyFrameID;
            point6d.roll = cloudInfo.initialGuessRoll;
            point6d.pitch = cloudInfo.initialGuessPitch;
            point6d.yaw = cloudInfo.initialGuessYaw;
            point6d.time = timeLaserInfoCur;

            currentKeyFrame->init_pose = point6d;
        } 
        else if (cloudInfo.imuAvailable == true) 
        {
            PointTypePose  point6d;
            point6d.x = 0;
            point6d.y = 0;
            point6d.z = 0;
            point6d.intensity = keyFrameID;
            point6d.roll = cloudInfo.imuRollInit;
            point6d.pitch = cloudInfo.imuPitchInit;
            point6d.yaw = cloudInfo.imuYawInit;
            point6d.time = timeLaserInfoCur;

            currentKeyFrame->init_pose = point6d;
        }
        
		#if USING_SEMANTIC_FEATURE
			pcl::fromROSMsg(cloudInfo.semantic_raw, *currentKeyFrame->semantic_raw);
			pcl::fromROSMsg(cloudInfo.semantic_dynamic, *currentKeyFrame->semantic_dynamic);
			pcl::fromROSMsg(cloudInfo.semantic_pole, *currentKeyFrame->semantic_pole);
			pcl::fromROSMsg(cloudInfo.semantic_ground, *currentKeyFrame->semantic_ground);
			pcl::fromROSMsg(cloudInfo.semantic_building, *currentKeyFrame->semantic_building);
			pcl::fromROSMsg(cloudInfo.semantic_outlier, *currentKeyFrame->semantic_outlier);

			pcl::fromROSMsg(cloudInfo.cloud_corner, *currentKeyFrame->cloud_corner);
			pcl::fromROSMsg(cloudInfo.cloud_surface, *currentKeyFrame->cloud_surface);     

			SubMapManager::voxel_downsample_pcl(currentKeyFrame->semantic_raw, currentKeyFrame->semantic_raw_down, 0.5);  //0.4
			SubMapManager::voxel_downsample_pcl(currentKeyFrame->semantic_dynamic, currentKeyFrame->semantic_dynamic_down, 0.2); //0.2
			SubMapManager::voxel_downsample_pcl(currentKeyFrame->semantic_pole, currentKeyFrame->semantic_pole_down, 0.05); //0.05
			SubMapManager::voxel_downsample_pcl(currentKeyFrame->semantic_ground, currentKeyFrame->semantic_ground_down, 0.6); //0.6
			SubMapManager::voxel_downsample_pcl(currentKeyFrame->semantic_building, currentKeyFrame->semantic_building_down, 0.4); //0.4
			SubMapManager::voxel_downsample_pcl(currentKeyFrame->semantic_outlier, currentKeyFrame->semantic_outlier_down, 0.6); //0.5

			// SubMapManager::voxel_downsample_pcl(currentKeyFrame->semantic_dynamic, currentKeyFrame->semantic_dynamic_down, 0.5); //0.2
			// SubMapManager::voxel_downsample_pcl(currentKeyFrame->semantic_pole, currentKeyFrame->semantic_pole_down, 0.5); //0.05
			// SubMapManager::voxel_downsample_pcl(currentKeyFrame->semantic_ground, currentKeyFrame->semantic_ground_down, 0.5); //0.6
			// SubMapManager::voxel_downsample_pcl(currentKeyFrame->semantic_building, currentKeyFrame->semantic_building_down, 0.5); //0.4
			// SubMapManager::voxel_downsample_pcl(currentKeyFrame->semantic_outlier, currentKeyFrame->semantic_outlier_down, 0.5); //0.5

			SubMapManager::voxel_downsample_pcl(currentKeyFrame->cloud_corner, currentKeyFrame->cloud_corner_down, 0.2);
        	SubMapManager::voxel_downsample_pcl(currentKeyFrame->cloud_surface, currentKeyFrame->cloud_surface_down, 0.4);
		#endif

		#if USING_LOAM_FEATURE
			pcl::fromROSMsg(cloudInfo.cloud_corner, *currentKeyFrame->cloud_corner);
			pcl::fromROSMsg(cloudInfo.cloud_surface, *currentKeyFrame->cloud_surface);     
			// pcl::fromROSMsg(cloudInfo.cloud_corner_sharp, *currentKeyFrame->cloud_corner_down);
			// pcl::fromROSMsg(cloudInfo.cloud_surface_sharp, *currentKeyFrame->cloud_surface_down);  

			SubMapManager::voxel_downsample_pcl(currentKeyFrame->cloud_corner, currentKeyFrame->cloud_corner_down, 0.1);
        	SubMapManager::voxel_downsample_pcl(currentKeyFrame->cloud_surface, currentKeyFrame->cloud_surface_down, 0.2);

			*currentKeyFrame->semantic_pole  = *trans2LabelPointCloud(currentKeyFrame->cloud_corner, 18);
			*currentKeyFrame->semantic_building = *trans2LabelPointCloud(currentKeyFrame->cloud_surface, 13);
			*currentKeyFrame->semantic_pole_down = *trans2LabelPointCloud(currentKeyFrame->cloud_corner_down, 18);
			*currentKeyFrame->semantic_building_down = *trans2LabelPointCloud(currentKeyFrame->cloud_surface_down, 13);
		#endif	

        //calculate bbx (local)
        // this->get_cloud_bbx(currentKeyFrame->semantic_raw, currentKeyFrame->local_bound);
        this->get_cloud_bbx_cpt(currentKeyFrame->semantic_raw, currentKeyFrame->local_bound, currentKeyFrame->local_cp);

        // ROS_WARN("keyFrameID: %d ,keyFrameInfo Size: %d ",keyFrameID, keyFrameInfo.size());    
		
		std::cout << "Feature point number of last frame (Ori | Ds): " << std::endl 
				  <<  "Dynamic: [" << currentKeyFrame->semantic_dynamic->points.size() << " | " << currentKeyFrame->semantic_dynamic_down->points.size() << "]." << std::endl 
				  <<  "Pole: [" << currentKeyFrame->semantic_pole->points.size() << " | " << currentKeyFrame->semantic_pole_down->points.size() << "]." << std::endl 
				  <<  "Ground: [" << currentKeyFrame->semantic_ground->points.size() << " | " << currentKeyFrame->semantic_ground_down->points.size() << "]." << std::endl 
				  <<  "Building: [" << currentKeyFrame->semantic_building->points.size() << " | " << currentKeyFrame->semantic_building_down->points.size() << "]." << std::endl 
				  <<  "Outlier: [" << currentKeyFrame->semantic_outlier->points.size() << " | " << currentKeyFrame->semantic_outlier_down->points.size() << "]." << std::endl 
				  << std::endl;

        return true;
    }
    
    void currentCloudInit()
    {
        laserCloudCornerLast->clear();
        laserCloudSurfLast->clear();
        laserCloudCornerLastDS->clear();
        laserCloudSurfLastDS->clear();
		
		laserCloudCornerLast->points.insert(laserCloudCornerLast->points.end(), 
											currentKeyFrame->semantic_pole->points.begin(), 
											currentKeyFrame->semantic_pole->points.end());
		laserCloudCornerLastDS->points.insert(laserCloudCornerLastDS->points.end(), 
											currentKeyFrame->semantic_pole_down->points.begin(), 
											currentKeyFrame->semantic_pole_down->points.end());
		
		laserCloudSurfLast->points.insert(laserCloudSurfLast->points.end(), 
											currentKeyFrame->semantic_dynamic->points.begin(), 
											currentKeyFrame->semantic_dynamic->points.end());
		laserCloudSurfLastDS->points.insert(laserCloudSurfLastDS->points.end(), 
											currentKeyFrame->semantic_dynamic_down->points.begin(), 
											currentKeyFrame->semantic_dynamic_down->points.end());
		
		laserCloudSurfLast->points.insert(laserCloudSurfLast->points.end(), 
											currentKeyFrame->semantic_building->points.begin(), 
											currentKeyFrame->semantic_building->points.end());
		laserCloudSurfLastDS->points.insert(laserCloudSurfLastDS->points.end(), 
											currentKeyFrame->semantic_building_down->points.begin(), 
											currentKeyFrame->semantic_building_down->points.end());
					
		laserCloudSurfLast->points.insert(laserCloudSurfLast->points.end(), 
										currentKeyFrame->semantic_ground->points.begin(), 
										currentKeyFrame->semantic_ground->points.end());
		laserCloudSurfLastDS->points.insert(laserCloudSurfLastDS->points.end(), 
											currentKeyFrame->semantic_ground_down->points.begin(), 
											currentKeyFrame->semantic_ground_down->points.end());																						

        laserCloudCornerLastDSNum = laserCloudCornerLastDS->points.size();
        laserCloudSurfLastDSNum = laserCloudSurfLastDS->points.size();
    }


    void updateInitialGuess()
    {
        // save current transformation before any processing
        incrementalOdometryAffineFront = trans2Affine3f(transformTobeSubMapped);

        static Eigen::Affine3f lastImuTransformation;
        // initialization
        static bool firstTransAvailable = false;
        if (firstTransAvailable == false)
        {
            ROS_WARN("MakeSubmap: firstTransAvailable!");
            transformTobeSubMapped[0] = cloudInfo.imuRollInit;
            transformTobeSubMapped[1] = cloudInfo.imuPitchInit;
            transformTobeSubMapped[2] = cloudInfo.imuYawInit;
			
            if (!useImuHeadingInitialization)
                transformTobeSubMapped[2] = 0;

            lastImuTransformation = pcl::getTransformation(0, 0, 0, cloudInfo.imuRollInit, cloudInfo.imuPitchInit, cloudInfo.imuYawInit); // save imu before return;
            firstTransAvailable = true;

            transformCurSubmap[0]=transformTobeSubMapped[0];
            transformCurSubmap[1]=transformTobeSubMapped[1];
            transformCurSubmap[2]=transformTobeSubMapped[2];
            transformCurSubmap[3]=transformTobeSubMapped[3];
            transformCurSubmap[4]=transformTobeSubMapped[4];
            transformCurSubmap[5]=transformTobeSubMapped[5];

            return;
        }

        // use imu pre-integration estimation for pose guess
        static bool lastImuPreTransAvailable = false;
        static Eigen::Affine3f lastImuPreTransformation;
        if (cloudInfo.odomAvailable == true)
        {
            ROS_WARN("MakeSubmap: cloudInfo.odomAvailable == true!");
            Eigen::Affine3f transBack = pcl::getTransformation(cloudInfo.initialGuessX,    cloudInfo.initialGuessY,     cloudInfo.initialGuessZ, 
                                                               cloudInfo.initialGuessRoll, cloudInfo.initialGuessPitch, cloudInfo.initialGuessYaw);
            
			transBef = transBack;

			if (lastImuPreTransAvailable == false) {
                lastImuPreTransformation = transBack;
                lastImuPreTransAvailable = true;
            } else {

                
                Eigen::Affine3f transIncre = lastImuPreTransformation.inverse() * transBack;
                Eigen::Affine3f transTobe = trans2Affine3f(transformTobeSubMapped);
                Eigen::Affine3f transFinal = transTobe * transIncre;
                pcl::getTranslationAndEulerAngles(transFinal, transformTobeSubMapped[3], transformTobeSubMapped[4], transformTobeSubMapped[5], 
                                                              transformTobeSubMapped[0], transformTobeSubMapped[1], transformTobeSubMapped[2]);


                for(int i=0;i<6;++i){
                    transPredictionMapped[i]=transformTobeSubMapped[i];
                }

                lastImuPreTransformation = transBack;
			}

			if (cloudInfo.imuAvailable == true)
				lastImuTransformation = pcl::getTransformation(0, 0, 0, cloudInfo.imuRollInit, cloudInfo.imuPitchInit, cloudInfo.imuYawInit); // save imu before return;
			
			return;
            
        }

        // use imu incremental estimation for pose guess (only rotation)
        if (cloudInfo.imuAvailable == true)
        {
            ROS_WARN("MakeSubmap: cloudInfo.imuAvailable == true!");
            Eigen::Affine3f transBack = pcl::getTransformation(0, 0, 0, cloudInfo.imuRollInit, cloudInfo.imuPitchInit, cloudInfo.imuYawInit);
           
            Eigen::Affine3f transIncre = lastImuTransformation.inverse() * transBack;

            Eigen::Affine3f transTobe = trans2Affine3f(transformTobeSubMapped);
            Eigen::Affine3f transFinal = transTobe * transIncre;
            
            pcl::getTranslationAndEulerAngles(transFinal, transformTobeSubMapped[3], transformTobeSubMapped[4], transformTobeSubMapped[5], 
                                                          transformTobeSubMapped[0], transformTobeSubMapped[1], transformTobeSubMapped[2]);
            
            for(int i=0;i<6;++i)
            {
                transPredictionMapped[i]=transformTobeSubMapped[i];
            }

            lastImuTransformation = pcl::getTransformation(0, 0, 0, cloudInfo.imuRollInit, cloudInfo.imuPitchInit, cloudInfo.imuYawInit); // save imu before return;
     
            return;
        }

        static float lastTransformTobeSubMapped[6]={0.0};
        if (cloudInfo.odomAvailable == false && cloudInfo.imuAvailable == false)
        {
            static bool first = false;
            if (first == false)
            {
                for(int i = 0; i < 6; ++i)
                {
                    lastTransformTobeSubMapped[i]=transformTobeSubMapped[i];
                }

                first = true;
                return;
            }

            // ROS_WARN("MakeSubmap: cloudInfo.imuAvailable == true!");
            Eigen::Affine3f transBack = pcl::getTransformation(transformTobeSubMapped[3], transformTobeSubMapped[4], transformTobeSubMapped[5],
                                                                                                                        transformTobeSubMapped[0], transformTobeSubMapped[1], transformTobeSubMapped[2]);
            
            Eigen::Affine3f transLast = pcl::getTransformation(lastTransformTobeSubMapped[3], lastTransformTobeSubMapped[4], lastTransformTobeSubMapped[5],
                                                                                                                        lastTransformTobeSubMapped[0], lastTransformTobeSubMapped[1], lastTransformTobeSubMapped[2]);
            
            for(int i = 0; i < 6; ++i)
            {
                lastTransformTobeSubMapped[i] = transformTobeSubMapped[i];
            }
            
            Eigen::Affine3f transIncre = transLast.inverse() * transBack;

            Eigen::Affine3f transTobe = trans2Affine3f(transformTobeSubMapped);
            Eigen::Affine3f transFinal = transTobe * transIncre;
            
            pcl::getTranslationAndEulerAngles(transFinal, transformTobeSubMapped[3], transformTobeSubMapped[4], transformTobeSubMapped[5], 
                                                        transformTobeSubMapped[0], transformTobeSubMapped[1], transformTobeSubMapped[2]);
            
            // ROS_WARN("cloudInfo.odomAvailable == true : transformTobeSubMapped[0] : %f",transformTobeSubMapped[0]);
            // ROS_WARN("cloudInfo.odomAvailable == true : transformTobeSubMapped[1] : %f",transformTobeSubMapped[1]);
            // ROS_WARN("cloudInfo.odomAvailable == true : transformTobeSubMapped[2] : %f",transformTobeSubMapped[2]);
            // ROS_WARN("cloudInfo.odomAvailable == true : transformTobeSubMapped[3] : %f",transformTobeSubMapped[3]);
            // ROS_WARN("cloudInfo.odomAvailable == true : transformTobeSubMapped[4] : %f",transformTobeSubMapped[4]);
            // ROS_WARN("cloudInfo.odomAvailable == true : transformTobeSubMapped[5] : %f",transformTobeSubMapped[5]);
        }

    }

    void calculateTranslation()
    {
        Eigen::Affine3f transBack = trans2Affine3f(transformCurSubmap);
        Eigen::Affine3f transTobe = trans2Affine3f(transformTobeSubMapped);

        Eigen::Affine3f transIncre = transBack.inverse() * transTobe;

        pcl::getTranslationAndEulerAngles(transIncre, transformCurFrame2Submap[3], transformCurFrame2Submap[4], transformCurFrame2Submap[5], 
                                                      transformCurFrame2Submap[0], transformCurFrame2Submap[1], transformCurFrame2Submap[2]);
    }


    void saveKeyFrame()
    {
		curSubMapSize++;

		// ROS_WARN("transPredictionMapped: [%f, %f, %f, %f, %f, %f]",
        //         transPredictionMapped[0], transPredictionMapped[1], transPredictionMapped[2],
        //         transPredictionMapped[3], transPredictionMapped[4], transPredictionMapped[5]);

        // ROS_WARN("transformTobeSubMapped: [%f, %f, %f, %f, %f, %f]",
        //         transformTobeSubMapped[0], transformTobeSubMapped[1], transformTobeSubMapped[2],
        //         transformTobeSubMapped[3], transformTobeSubMapped[4], transformTobeSubMapped[5]);
        
        // ROS_WARN("transformCurSubmap: [%f, %f, %f, %f, %f, %f]",
        //         transformCurSubmap[0], transformCurSubmap[1], transformCurSubmap[2],
        //         transformCurSubmap[3], transformCurSubmap[4], transformCurSubmap[5]);

        PointTypePose  point6d = trans2PointTypePose(transformTobeSubMapped, keyFrameID, timeLaserInfoCur);
        PointType  point3d = trans2PointType(transformTobeSubMapped, keyFrameID);

        keyFramePoses3D->push_back(point3d);
        keyFramePoses6D->push_back(point6d);

        keyFramePosesIndex6D[keyFrameID] = point6d;
        keyFramePosesIndex3D[keyFrameID] = point3d;
        
		keyframeInSubmapIndex[keyFrameID] = subMapID;

        currentKeyFrame->keyframe_id = keyFrameID;
        currentKeyFrame->submap_id = subMapID;
        currentKeyFrame->id_in_submap = curSubMapSize;
        currentKeyFrame->optimized_pose = point6d;

        calculateTranslation();
        point6d = trans2PointTypePose(transformCurFrame2Submap, keyFrameID, timeLaserInfoCur);
        currentKeyFrame->relative_pose = point6d;

        //calculate bbx (global)
        Eigen::Affine3f  tran_map = pclPointToAffine3f(currentKeyFrame->optimized_pose);
        this->transform_bbx(currentKeyFrame->local_bound, currentKeyFrame->local_cp, currentKeyFrame->bound, currentKeyFrame->cp, tran_map);

        keyframe_Ptr tmpKeyFrame;
        tmpKeyFrame = keyframe_Ptr(new keyframe_t(*currentKeyFrame, true));
		keyFrameQueue.push_back(tmpKeyFrame);
        // keyFrameInfo.insert(std::make_pair(keyFrameID, tmpKeyFrame));
        

        // ROS_WARN("currentKeyFrame : relative_pose: [%f, %f, %f, %f, %f, %f]",
        //         currentKeyFrame->relative_pose.roll, currentKeyFrame->relative_pose.pitch, currentKeyFrame->relative_pose.yaw,
        //         currentKeyFrame->relative_pose.x, currentKeyFrame->relative_pose.y, currentKeyFrame->relative_pose.z);

        // ROS_WARN("currentKeyFrame : optimized_pose: [%f, %f, %f, %f, %f, %f]",
        //         currentKeyFrame->optimized_pose.roll, currentKeyFrame->optimized_pose.pitch, currentKeyFrame->optimized_pose.yaw,
        //         currentKeyFrame->optimized_pose.x, currentKeyFrame->optimized_pose.y, currentKeyFrame->optimized_pose.z);
        

		keyFrameID++;
    }


	void currentSubMapInit()
	{
        timeSubMapInfoStamp = timeLaserInfoStamp;
        double curSubMapTime = timeSubMapInfoStamp.toSec();

        PointTypePose  point6d = trans2PointTypePose(transformTobeSubMapped, subMapID, curSubMapTime);
        subMapPose6D->points.push_back(point6d);      
            
        PointType  point3d = trans2PointType(transformTobeSubMapped, subMapID);
        subMapPose3D->points.push_back(point3d);  

        subMapPosesIndex3D[subMapID] = point3d;
        subMapPosesIndex6D[subMapID] = point6d;

		currentSubMap->submap_id = subMapID;
		currentSubMap->submap_pose_6D_init = point6d;
		currentSubMap->submap_pose_6D_optimized = point6d;
		currentSubMap->submap_pose_3D_optimized = point3d;
        
        transformCurSubmap[0]=transformTobeSubMapped[0];
        transformCurSubmap[1]=transformTobeSubMapped[1];
        transformCurSubmap[2]=transformTobeSubMapped[2];
        transformCurSubmap[3]=transformTobeSubMapped[3];
        transformCurSubmap[4]=transformTobeSubMapped[4];
        transformCurSubmap[5]=transformTobeSubMapped[5];

	}


    void saveSubMap()
    {
        submap_Ptr tmpSubMap;
        tmpSubMap = submap_Ptr(new submap_t(*currentSubMap, true, true));
        subMapIndexQueue.push_back(subMapID);
		subMapInfo[subMapID] = tmpSubMap;

		curSubMapSize = 0;
		subMapID++;
    }


	void extractTargetCloud()
	{
		// pcl::copyPointCloud(*laserCloudCornerLast,    *laserCloudFromPre);
		// *laserCloudFromPre = *transformPointCloud(laserCloudFromPre, &keyFramePoses6D->back());

		laserCloudFromPre->clear();
		*laserCloudFromPre += *currentSubMap->submap_pole;
		// *laserCloudFromPre += *currentSubMap->submap_dynamic;
		*laserCloudFromPre = *transformPointCloud(laserCloudFromPre, &currentSubMap->submap_pose_6D_optimized);

	#if USING_SINGLE_TARGET
		pcl::copyPointCloud(*laserCloudCornerLast,    *laserCloudCornerFromSubMap);
		pcl::copyPointCloud(*laserCloudSurfLast,    *laserCloudSurfFromSubMap);
		*laserCloudCornerFromSubMap = *transformPointCloud(laserCloudCornerFromSubMap, &keyFramePoses6D->back());
		*laserCloudSurfFromSubMap = *transformPointCloud(laserCloudSurfFromSubMap, &keyFramePoses6D->back());
	#endif

	#if USING_MULTI_KEYFRAME_TARGET
		static std::vector<pcl::PointCloud<PointXYZIL>::Ptr> laserCloudSurfVec;
		static std::vector<pcl::PointCloud<PointXYZIL>::Ptr> laserCloudCornerVec;
		
		laserCloudCornerFromSubMap->clear();
		laserCloudSurfFromSubMap->clear();

		pcl::PointCloud<PointXYZIL>::Ptr tmpSurf( new pcl::PointCloud<PointXYZIL>());
		pcl::PointCloud<PointXYZIL>::Ptr tmpCorner( new pcl::PointCloud<PointXYZIL>());
		pcl::copyPointCloud(*laserCloudSurfLast,    *tmpSurf);
		pcl::copyPointCloud(*laserCloudCornerLast,    *tmpCorner);

		*tmpSurf = *transformPointCloud(tmpSurf, &keyFramePoses6D->back());
		*tmpCorner = *transformPointCloud(tmpCorner, &keyFramePoses6D->back());

		laserCloudSurfVec.push_back(tmpSurf);
		laserCloudCornerVec.push_back(tmpCorner);

		while(laserCloudSurfVec.size() >= 5)
		{
			laserCloudSurfVec.erase(laserCloudSurfVec.begin());
			laserCloudCornerVec.erase(laserCloudCornerVec.begin());
		}
		
		for(int i = 0; i < laserCloudSurfVec.size(); i++){
			*laserCloudCornerFromSubMap += *laserCloudCornerVec[i];
			*laserCloudSurfFromSubMap += *laserCloudSurfVec[i];
		}
		
		pcl::PointCloud<PointXYZIL>::Ptr tmpCloud( new pcl::PointCloud<PointXYZIL>());
		*tmpCloud += *laserCloudCornerFromSubMap;
		*tmpCloud += *laserCloudSurfFromSubMap;
		publishLabelCloud(&pubTest1, laserCloudSurfFromSubMap, timeLaserInfoStamp, odometryFrame);

	#endif

	#if USING_SLIDING_TARGET
		PointTypePose  cur_pose = trans2PointTypePose(transformTobeSubMapped, keyFrameID, timeLaserInfoCur);
		extractSlidingCloud(currentKeyFrame, cur_pose, false);
	#endif

	#if USING_SUBMAP_TARGET    
		int target_submap_id = -1;
		int target_keyframe_id = -1;
		bool using_target_id = false;
		PointTypePose  cur_pose = trans2PointTypePose(transformTobeSubMapped, keyFrameID, timeLaserInfoCur);
		
		extractSurroundingKeyFrames(cur_pose, target_submap_id, target_keyframe_id, using_target_id);
		
		auto it_ = subMapInfo.find(target_submap_id);
		if(it_ != subMapInfo.end()){
			extractSubMapCloud(currentKeyFrame, it_->second, cur_pose, false);
		}else{
			ROS_WARN("Dont extract Target Submap ID from Surrounding KeyFrames! ");  
			ROS_WARN("USING_SINGLE_TARGET !");  
			// @Todo
			pcl::copyPointCloud(*laserCloudCornerLast,    *laserCloudCornerFromSubMap);
			pcl::copyPointCloud(*laserCloudSurfLast,    *laserCloudSurfFromSubMap);
			*laserCloudCornerFromSubMap = *transformPointCloud(laserCloudCornerFromSubMap, &keyFramePoses6D->back());
			*laserCloudSurfFromSubMap = *transformPointCloud(laserCloudSurfFromSubMap, &keyFramePoses6D->back());
		}
	#endif

	}


    void extractSurroundingKeyFrames(PointTypePose &cur_pose, int &target_submap_id, 
                                     int target_keyframe_id = -1, bool using_target_id = false)
    {
        kdtreeFromKeyFramePose6D.reset(new pcl::KdTreeFLANN<PointTypePose>());

		if(subMapPose6D->points.size() <= 0)
			return;   

        if(using_target_id)
        {
            auto it_ = keyFramePosesIndex6D.find(target_keyframe_id);
            if(it_ != keyFramePosesIndex6D.end())
            {
                std::vector<int> pointSearchInd;
                std::vector<float> pointSearchSqDis;

                // extract all the nearby key poses and downsample them
                kdtreeFromKeyFramePose6D->setInputCloud(subMapPose6D); 
                // kdtreeFromKeyFramePose6D->radiusSearch(subMapPose6D->back(), (double)2.0*subMapTraMax, pointSearchInd, pointSearchSqDis);
                kdtreeFromKeyFramePose6D->nearestKSearch(it_->second, 2, pointSearchInd, pointSearchSqDis);
                
                for (int i = 0; i < (int)pointSearchInd.size(); ++i)
                {
                    int id = pointSearchInd[i];
                    PointTypePose tmpKeyPose = subMapPose6D->points[id];
                    if(tmpKeyPose.time < cur_pose.time){
                        target_submap_id = tmpKeyPose.intensity;
                    }
                }
                if(target_submap_id == -1)
                    ROS_WARN("target_submap_id == -1 !");
                else
                    ROS_WARN("target_submap_id = %d!", target_submap_id);

            }
        }
        else
        {
            std::vector<int> pointSearchInd;
            std::vector<float> pointSearchSqDis;

            // extract all the nearby key poses and downsample them
            kdtreeFromKeyFramePose6D->setInputCloud(subMapPose6D); 
            // kdtreeFromKeyFramePose6D->radiusSearch(subMapPose6D->back(), (double)2.0*subMapTraMax, pointSearchInd, pointSearchSqDis);
            kdtreeFromKeyFramePose6D->nearestKSearch(cur_pose, 2, pointSearchInd, pointSearchSqDis);
            
            for (int i = 0; i < (int)pointSearchInd.size(); ++i)
            {
                int id = pointSearchInd[i];
                PointTypePose tmpKeyPose = subMapPose6D->points[id];
                if(tmpKeyPose.time < cur_pose.time){
                    target_submap_id = tmpKeyPose.intensity;
                }
            }
            if(target_submap_id == -1)
                ROS_WARN("target_submap_id == -1 !");
            else
                ROS_WARN("target_submap_id = %d!", target_submap_id);
        }
             
    }

    void extractSubMapCloud(keyframe_Ptr &cur_keyframe, submap_Ptr &cur_submap, 
                            PointTypePose cur_pose, bool using_keyframe_pose = false)
    {
        laserCloudCornerFromSubMap->clear();
        laserCloudSurfFromSubMap->clear();

        if(using_keyframe_pose)
        {
            Eigen::Affine3f tran_map = pclPointToAffine3f(cur_keyframe->optimized_pose);
            this->transform_bbx(cur_keyframe->local_bound, cur_keyframe->local_cp, cur_keyframe->bound, cur_keyframe->cp, tran_map);

        }
        else
        {
            Eigen::Affine3f tran_map = pclPointToAffine3f(cur_pose);
            this->transform_bbx(cur_keyframe->local_bound, cur_keyframe->local_cp, cur_keyframe->bound, cur_keyframe->cp, tran_map);
        }


        Eigen::Affine3f tran_map = pclPointToAffine3f(cur_submap->submap_pose_6D_optimized);
        this->transform_bbx(cur_submap->local_bound, cur_submap->local_cp, cur_submap->bound, cur_submap->cp, tran_map);

        bounds_t bbx_intersection;
        get_intersection_bbx(cur_keyframe->bound, cur_submap->bound, bbx_intersection, 2.0);

        // std::cout << "cur_submap local_bound: [" << cur_submap->local_bound.min_x << ", "
        //                                          << cur_submap->local_bound.min_y << ", " 
        //                                          << cur_submap->local_bound.min_z << ", " 
        //                                          << cur_submap->local_bound.max_x << ", " 
        //                                          << cur_submap->local_bound.max_y << ", " 
        //                                          << cur_submap->local_bound.max_z
        //                                          << "]" << std::endl;

        // std::cout << "cur_submap bound: [" << cur_submap->bound.min_x << ", "
        //                                    << cur_submap->bound.min_y << ", " 
        //                                    << cur_submap->bound.min_z << ", " 
        //                                    << cur_submap->bound.max_x << ", " 
        //                                    << cur_submap->bound.max_y << ", " 
        //                                    << cur_submap->bound.max_z
        //                                    << "]" << std::endl;

        // std::cout << "bbx_intersection: [" << bbx_intersection.min_x << ", "
        //                                    << bbx_intersection.min_y << ", " 
        //                                    << bbx_intersection.min_z << ", " 
        //                                    << bbx_intersection.max_x << ", " 
        //                                    << bbx_intersection.max_y << ", " 
        //                                    << bbx_intersection.max_z  
        //                                    << "]" << std::endl; 

		laserCloudCornerFromSubMap->points.insert(laserCloudCornerFromSubMap->points.end(), 
								  				cur_submap->submap_pole->points.begin(), 
								  				cur_submap->submap_pole->points.end());
		laserCloudSurfFromSubMap->points.insert(laserCloudSurfFromSubMap->points.end(), 
								  				cur_submap->submap_ground->points.begin(), 
								  				cur_submap->submap_ground->points.end());
		laserCloudSurfFromSubMap->points.insert(laserCloudSurfFromSubMap->points.end(), 
								  				cur_submap->submap_building->points.begin(), 
								  				cur_submap->submap_building->points.end());
		
		*laserCloudCornerFromSubMap = *transformPointCloud(laserCloudCornerFromSubMap, &cur_submap->submap_pose_6D_optimized);
        *laserCloudSurfFromSubMap = *transformPointCloud(laserCloudSurfFromSubMap, &cur_submap->submap_pose_6D_optimized);
		
        // Use the intersection bounding box to filter the outlier points
        bbx_filter(laserCloudCornerFromSubMap, bbx_intersection);
        bbx_filter(laserCloudSurfFromSubMap, bbx_intersection);

		pcl::PointCloud<PointXYZIL>::Ptr cloud_temp(new pcl::PointCloud<PointXYZIL>);
		cloud_temp->points.insert(cloud_temp->points.end(), 
								laserCloudCornerFromSubMap->points.begin(), 
								laserCloudCornerFromSubMap->points.end());
		cloud_temp->points.insert(cloud_temp->points.end(), 
								laserCloudSurfFromSubMap->points.begin(), 
								laserCloudSurfFromSubMap->points.end());
        publishLabelCloud(&pubTest1, cloud_temp, timeLaserInfoStamp, lidarFrame);

    }


    void extractSlidingCloud(keyframe_Ptr &cur_keyframe, PointTypePose cur_pose, bool using_keyframe_pose = false)
    {	
        laserCloudCornerFromSubMap->clear();
        laserCloudSurfFromSubMap->clear();

		Eigen::Affine3f tran_map;
		if(using_keyframe_pose) {
			tran_map = pclPointToAffine3f(cur_keyframe->optimized_pose);
		} else {
			tran_map = pclPointToAffine3f(cur_pose);
		}

		// this->transform_bbx(cur_keyframe->local_bound, cur_keyframe->local_cp, cur_keyframe->bound, cur_keyframe->cp, tran_map);
		// bounds_t bbx_intersection;
        // get_intersection_bbx(cur_keyframe->bound, localMap->bound, bbx_intersection, 10.0);
		
		bounds_t cur_bbx;
		cur_bbx.min_x = -70.0; cur_bbx.min_y = -70.0; cur_bbx.min_z = -10.0;
		cur_bbx.max_x = 70.0;  cur_bbx.max_y = 70.0;  cur_bbx.max_z = 20.0;
		centerpoint_t cur_cp;
		get_bound_cpt(cur_bbx, cur_cp);
		this->transform_bbx(cur_bbx, cur_cp, cur_bbx, cur_cp, tran_map);
		bounds_t bbx_intersection;
        get_intersection_bbx(cur_bbx, localMap->bound, bbx_intersection, 2.0);

        SubMapManager::voxel_downsample_pcl(localMap->submap_dynamic, localMap->submap_dynamic, 0.1); // default: 0.05
        SubMapManager::voxel_downsample_pcl(localMap->submap_pole, localMap->submap_pole, 0.05); // default: 0.02
        SubMapManager::voxel_downsample_pcl(localMap->submap_ground, localMap->submap_ground, 0.4); // default: 0.2
        SubMapManager::voxel_downsample_pcl(localMap->submap_building, localMap->submap_building, 0.2); // default: 0.1
        SubMapManager::voxel_downsample_pcl(localMap->submap_outlier, localMap->submap_outlier, 0.6); // default: 0.5
        
		// Use the intersection bounding box to filter the outlier points
        bbx_filter(localMap->submap_dynamic, bbx_intersection);
        bbx_filter(localMap->submap_pole, bbx_intersection);
        bbx_filter(localMap->submap_ground, bbx_intersection);
        bbx_filter(localMap->submap_building, bbx_intersection);
        bbx_filter(localMap->submap_outlier, bbx_intersection);
		

		laserCloudCornerFromSubMap->points.insert(laserCloudCornerFromSubMap->points.end(), 
												localMap->submap_pole->points.begin(), 
												localMap->submap_pole->points.end());
		laserCloudSurfFromSubMap->points.insert(laserCloudSurfFromSubMap->points.end(), 
												localMap->submap_ground->points.begin(), 
												localMap->submap_ground->points.end());
		laserCloudSurfFromSubMap->points.insert(laserCloudSurfFromSubMap->points.end(), 
												localMap->submap_building->points.begin(), 
												localMap->submap_building->points.end());
		laserCloudSurfFromSubMap->points.insert(laserCloudSurfFromSubMap->points.end(), 
												localMap->submap_dynamic->points.begin(), 
												localMap->submap_dynamic->points.end());


		pcl::PointCloud<PointXYZIL>::Ptr cloud_temp(new pcl::PointCloud<PointXYZIL>);
		cloud_temp->points.insert(cloud_temp->points.end(), 
								laserCloudCornerFromSubMap->points.begin(), 
								laserCloudCornerFromSubMap->points.end());
		cloud_temp->points.insert(cloud_temp->points.end(), 
								laserCloudSurfFromSubMap->points.begin(), 
								laserCloudSurfFromSubMap->points.end());
        // std::cout << "cloud_temp size: " << cloud_temp->points.size() << std::endl;
        publishLabelCloud(&pubTest1, cloud_temp, timeLaserInfoStamp, odometryFrame);

	}


	void icpAlignment(pcl::PointCloud<PointXYZIL>::Ptr source_pc, pcl::PointCloud<PointXYZIL>::Ptr target_pc, float transformIn[])
	{

		ROS_WARN("Source PC Size: %d, Target PC Size: %d", source_pc->points.size(), target_pc->points.size());

        std::cout << "matching..." << std::flush;
        auto t1 = ros::Time::now();

        // ICP Settings
        static pcl::IterativeClosestPoint<PointXYZIL, PointXYZIL> icp;
        icp.setMaxCorrespondenceDistance(0.2);
        icp.setMaximumIterations(50);
        icp.setTransformationEpsilon(1e-5);
        icp.setEuclideanFitnessEpsilon(1e-5);
        icp.setRANSACIterations(0);

		Eigen::Affine3f initLidarFrame = trans2Affine3f(transformIn);
		// Align clouds
		pcl::PointCloud<PointXYZIL>::Ptr tmpCloud( new pcl::PointCloud<PointXYZIL>());
		*tmpCloud = *transformPointCloud(source_pc, initLidarFrame);
		icp.setInputSource(tmpCloud);

		icp.setInputTarget(target_pc);
		pcl::PointCloud<PointXYZIL>::Ptr unused_result( new pcl::PointCloud<PointXYZIL>());
		icp.align(*unused_result);
	publishLabelCloud(&pubTestPre, target_pc, timeLaserInfoStamp, odometryFrame);

		double score = icp.getFitnessScore();
		
		Eigen::Affine3f correctionLidarFrame;
		correctionLidarFrame = icp.getFinalTransformation();
        
		auto t2 = ros::Time::now();
        std::cout << " done" << std::endl;
        std::cout << "score: " << score << "    time: " << (t2 - t1).toSec() << "[sec]" << std::endl;
        
		float X, Y, Z, ROLL, PITCH, YAW;
        Eigen::Affine3f tCorrect = correctionLidarFrame * initLidarFrame;  // pre-multiplying -> successive rotation about a fixed frame
        pcl::getTranslationAndEulerAngles(tCorrect, X, Y, Z, ROLL, PITCH, YAW);
        
		std::cout << "TransformIn: [" << ROLL << ", " << PITCH << ", " << YAW << ", " 
                                      << X << ", " << Y << ", " << Z << "]" << std::endl;

	*tmpCloud = *transformPointCloud(source_pc, tCorrect);
	publishLabelCloud(&pubTestCurICP, tmpCloud, timeLaserInfoStamp, odometryFrame);

		if(score <= 3.0) 
		{
			transformTobeSubMapped[2] = YAW;
			transformTobeSubMapped[3] = X;
			transformTobeSubMapped[4] = Y;
		}

	}

    void scan2SubMapOptimizationICP()
	{
		pcl::PointCloud<PointXYZIL>::Ptr sourcePC( new pcl::PointCloud<PointXYZIL>());
		pcl::PointCloud<PointXYZIL>::Ptr targetPC( new pcl::PointCloud<PointXYZIL>());

		// *sourcePC += *currentKeyFrame->semantic_dynamic;
		// *sourcePC += *currentKeyFrame->semantic_pole;
		// *sourcePC += *currentKeyFrame->semantic_ground;
		// *sourcePC += *currentKeyFrame->semantic_building;

		// *targetPC += *laserCloudSurfFromSubMap;
		// *targetPC += *laserCloudCornerFromSubMap;
		
		*targetPC += *laserCloudFromPre;
		*sourcePC += *currentKeyFrame->semantic_pole;
		// *sourcePC += *currentKeyFrame->semantic_dynamic;
		icpAlignment(sourcePC, targetPC, transformTobeSubMapped);
	}

    void scan2SubMapOptimization()
    {
        if (laserCloudCornerLastDSNum > edgeFeatureMinValidNum && laserCloudSurfLastDSNum > surfFeatureMinValidNum)
        {
            // ROS_WARN("laserCloudCornerFromSubMap: %d laserCloudSurfFromSubMap: %d .", laserCloudCornerFromSubMap->points.size(), laserCloudSurfFromSubMap->points.size());
            // ROS_WARN("laserCloudCornerLastDSNum: %d laserCloudSurfLastDSNum: %d .", laserCloudCornerLastDSNum, laserCloudSurfLastDSNum);
            
            kdtreeCornerFromSubMap->setInputCloud(laserCloudCornerFromSubMap);
            kdtreeSurfFromSubMap->setInputCloud(laserCloudSurfFromSubMap);

			int iterCount = 0;
            for (; iterCount < 20; iterCount++)   //30
            {
                laserCloudOri->clear();
                coeffSel->clear();

                cornerOptimization();

                surfOptimization();

                combineOptimizationCoeffs();

                if (LMOptimization(iterCount) == true)
                    break;          
            }

			ROS_WARN("iterCount: %d, deltaR: %f, deltaT: %f", iterCount, deltaR, deltaT);
            transformUpdate();

        } else {
            ROS_WARN("Not enough features! Only %d edge and %d planar features available.", laserCloudCornerLastDSNum, laserCloudSurfLastDSNum);
        }
    }

    void updatePointAssociateToSubMap()
    {
        transPointAssociateToSubMap = trans2Affine3f(transformTobeSubMapped);
    }

    void pointAssociateToSubMap(PointXYZIL const * const pi, PointXYZIL * const po)
    {
        po->x = transPointAssociateToSubMap(0,0) * pi->x + transPointAssociateToSubMap(0,1) * pi->y + transPointAssociateToSubMap(0,2) * pi->z + transPointAssociateToSubMap(0,3);
        po->y = transPointAssociateToSubMap(1,0) * pi->x + transPointAssociateToSubMap(1,1) * pi->y + transPointAssociateToSubMap(1,2) * pi->z + transPointAssociateToSubMap(1,3);
        po->z = transPointAssociateToSubMap(2,0) * pi->x + transPointAssociateToSubMap(2,1) * pi->y + transPointAssociateToSubMap(2,2) * pi->z + transPointAssociateToSubMap(2,3);
        po->intensity = pi->intensity;
        po->label = pi->label;
    }

    void cornerOptimization()
    {
        updatePointAssociateToSubMap();

		int numSearch = 0;

        // #pragma omp for
        #pragma omp parallel for num_threads(numberOfCores)
        for (int i = 0; i < laserCloudCornerLastDSNum; i++)
        {
            PointXYZIL pointOri, pointSel, coeff;
            pointOri = laserCloudCornerLastDS->points[i];
            pointAssociateToSubMap(&pointOri, &pointSel);

			std::vector<int> pointSearchInd;
            std::vector<float> pointSearchSqDis;
            kdtreeCornerFromSubMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);


			// std::vector<int> pointSearchIndPre;
            // std::vector<float> pointSearchSqDisPre;
            // kdtreeCornerFromSubMap->nearestKSearch(pointSel, 10, pointSearchIndPre, pointSearchSqDisPre);

            // std::vector<int> pointSearchInd;
            // std::vector<float> pointSearchSqDis;

            // std::vector<int> pointSearchIndNo;
            // std::vector<float> pointSearchSqDisNo;
				
			// int labelOri = laserCloudCornerLastDS->points[i].label;
			// for(int id = 0; id < pointSearchIndPre.size(); ++id)
			// {
			// 	int labelCur = laserCloudCornerFromSubMap->points[pointSearchIndPre[id]].label;
			// 	if(pointSearchInd.size() <= 5 && labelOri == labelCur && pointSearchSqDisPre[id] < 2.0)
			// 	{
			// 		pointSearchInd.push_back(pointSearchIndPre[id]);
			// 		pointSearchSqDis.push_back(pointSearchSqDisPre[id]);
			// 	}else{
			// 		pointSearchIndNo.push_back(pointSearchIndPre[id]);
			// 		pointSearchSqDisNo.push_back(pointSearchSqDisPre[id]);
			// 	}
			// }
			// int curIndSize = 5 - pointSearchInd.size();
			// for(int id = 0; id < curIndSize; ++id){
			// 	pointSearchInd.push_back(pointSearchIndNo[id]);
			// 	pointSearchSqDis.push_back(pointSearchSqDisNo[id]);
			// }

            cv::Mat matA1(3, 3, CV_32F, cv::Scalar::all(0));
            cv::Mat matD1(1, 3, CV_32F, cv::Scalar::all(0));
            cv::Mat matV1(3, 3, CV_32F, cv::Scalar::all(0));

            // if (pointSearchSqDis.size() == 5 && pointSearchSqDis[4] < 1.0) 
            if (pointSearchSqDis.size() == 5 && pointSearchSqDis[4] < 2.0) 
			{
                float cx = 0, cy = 0, cz = 0;
                for (int j = 0; j < 5; j++) {
                    cx += laserCloudCornerFromSubMap->points[pointSearchInd[j]].x;
                    cy += laserCloudCornerFromSubMap->points[pointSearchInd[j]].y;
                    cz += laserCloudCornerFromSubMap->points[pointSearchInd[j]].z;
                }
                cx /= 5; cy /= 5;  cz /= 5;

                float a11 = 0, a12 = 0, a13 = 0, a22 = 0, a23 = 0, a33 = 0;
                for (int j = 0; j < 5; j++) {
                    float ax=0, ay=0, az=0;

                    ax = laserCloudCornerFromSubMap->points[pointSearchInd[j]].x - cx;
                    ay = laserCloudCornerFromSubMap->points[pointSearchInd[j]].y - cy;
                    az = laserCloudCornerFromSubMap->points[pointSearchInd[j]].z - cz;

                    a11 += ax * ax; a12 += ax * ay; a13 += ax * az;
                    a22 += ay * ay; a23 += ay * az;
                    a33 += az * az;
                }
                a11 /= 5; a12 /= 5; a13 /= 5; a22 /= 5; a23 /= 5; a33 /= 5;

                matA1.at<float>(0, 0) = a11; matA1.at<float>(0, 1) = a12; matA1.at<float>(0, 2) = a13;
                matA1.at<float>(1, 0) = a12; matA1.at<float>(1, 1) = a22; matA1.at<float>(1, 2) = a23;
                matA1.at<float>(2, 0) = a13; matA1.at<float>(2, 1) = a23; matA1.at<float>(2, 2) = a33;

                cv::eigen(matA1, matD1, matV1);

                if (matD1.at<float>(0, 0) > 3 * matD1.at<float>(0, 1)) {

                    float x0 = pointSel.x;
                    float y0 = pointSel.y;
                    float z0 = pointSel.z;
                    float x1 = cx + 0.1 * matV1.at<float>(0, 0);
                    float y1 = cy + 0.1 * matV1.at<float>(0, 1);
                    float z1 = cz + 0.1 * matV1.at<float>(0, 2);
                    float x2 = cx - 0.1 * matV1.at<float>(0, 0);
                    float y2 = cy - 0.1 * matV1.at<float>(0, 1);
                    float z2 = cz - 0.1 * matV1.at<float>(0, 2);

                    float a012 = sqrt(((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) * ((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) 
                                    + ((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1)) * ((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1)) 
                                    + ((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1)) * ((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1)));

                    float l12 = sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2));

                    float la = ((y1 - y2)*((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) 
                              + (z1 - z2)*((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1))) / a012 / l12;

                    float lb = -((x1 - x2)*((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) 
                               - (z1 - z2)*((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1))) / a012 / l12;

                    float lc = -((x1 - x2)*((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1)) 
                               + (y1 - y2)*((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1))) / a012 / l12;

                    float ld2 = a012 / l12;

                    float s = 1 - 0.9 * fabs(ld2);

					float w = 2.0 - LabelSorce[laserCloudCornerLastDS->points[i].label];

                    coeff.x = w * s * la;
                    coeff.y = w * s * lb;
                    coeff.z = w * s * lc;
                    coeff.intensity = w * s * ld2;

                    // coeff.x = s * la;
                    // coeff.y = s * lb;
                    // coeff.z = s * lc;
                    // coeff.intensity = w * s * ld2;

                    if (s > 0.1) {
                        laserCloudOriCornerVec[i] = pointOri;
                        coeffSelCornerVec[i] = coeff;
                        laserCloudOriCornerFlag[i] = true;

						numSearch++;
                    }
                }
            }
        }

		// ROS_WARN("Corner numSearch: [%d / %d]", numSearch, laserCloudCornerLastDSNum);

    }



    void surfOptimization()
    {
        updatePointAssociateToSubMap();

		int numSearch = 0;

        // #pragma omp for
        #pragma omp parallel for num_threads(numberOfCores)
        for (int i = 0; i < laserCloudSurfLastDSNum; i++)
        {
            PointXYZIL pointOri, pointSel, coeff;
            pointOri = laserCloudSurfLastDS->points[i];
            pointAssociateToSubMap(&pointOri, &pointSel);
			
			std::vector<int> pointSearchInd;
            std::vector<float> pointSearchSqDis;
            kdtreeSurfFromSubMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);
			
			
			// std::vector<int> pointSearchIndPre;
            // std::vector<float> pointSearchSqDisPre;
            // kdtreeSurfFromSubMap->nearestKSearch(pointSel, 10, pointSearchIndPre, pointSearchSqDisPre);

            // std::vector<int> pointSearchInd;
            // std::vector<float> pointSearchSqDis;
				
            // std::vector<int> pointSearchIndNo;
            // std::vector<float> pointSearchSqDisNo;
				
			// int labelOri = laserCloudSurfLastDS->points[i].label;
			// for(int id = 0; id < pointSearchIndPre.size(); ++id)
			// {
			// 	int labelCur = laserCloudSurfFromSubMap->points[pointSearchIndPre[id]].label;
			// 	if(pointSearchInd.size() <= 5 && labelOri == labelCur && pointSearchSqDisPre[id] < 2.0)
			// 	{
			// 		pointSearchInd.push_back(pointSearchIndPre[id]);
			// 		pointSearchSqDis.push_back(pointSearchSqDisPre[id]);
			// 	}else{
			// 		pointSearchIndNo.push_back(pointSearchIndPre[id]);
			// 		pointSearchSqDisNo.push_back(pointSearchSqDisPre[id]);

			// 	}
			// }

			// int curIndSize = 5 - pointSearchInd.size();
			// for(int id = 0; id < curIndSize; ++id){
			// 	pointSearchInd.push_back(pointSearchIndNo[id]);
			// 	pointSearchSqDis.push_back(pointSearchSqDisNo[id]);
			// }


            Eigen::Matrix<float, 5, 3> matA0;
            Eigen::Matrix<float, 5, 1> matB0;
            Eigen::Vector3f matX0;

            matA0.setZero();
            matB0.fill(-1);
            matX0.setZero();

            // if (pointSearchSqDis.size() == 5 && pointSearchSqDis[4] < 1.0) 
            if (pointSearchSqDis.size() == 5 && pointSearchSqDis[4] < 2.0) 
			{

                for (int j = 0; j < 5; j++) {
                        matA0(j, 0) = laserCloudSurfFromSubMap->points[pointSearchInd[j]].x;
                        matA0(j, 1) = laserCloudSurfFromSubMap->points[pointSearchInd[j]].y;
                        matA0(j, 2) = laserCloudSurfFromSubMap->points[pointSearchInd[j]].z;
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
                    if (fabs(pa * laserCloudSurfFromSubMap->points[pointSearchInd[j]].x +
                            pb * laserCloudSurfFromSubMap->points[pointSearchInd[j]].y +
                            pc * laserCloudSurfFromSubMap->points[pointSearchInd[j]].z + pd) > 0.2) {
                        planeValid = false;
                        break;
                    }    
                }

                if (planeValid) {
                    float pd2 = pa * pointSel.x + pb * pointSel.y + pc * pointSel.z + pd;

                    float s = 1 - 0.9 * fabs(pd2) / sqrt(sqrt(pointSel.x * pointSel.x
                            + pointSel.y * pointSel.y + pointSel.z * pointSel.z));

					float w = 2.0 - LabelSorce[laserCloudSurfLastDS->points[i].label];

                    coeff.x = w * s * pa;
                    coeff.y = w * s * pb;
                    coeff.z = w * s * pc;
                    coeff.intensity = w * s * pd2;
                    
					// coeff.x = s * pa;
                    // coeff.y = s * pb;
                    // coeff.z = s * pc;
                    // coeff.intensity = w * s * pd2;

                    if (s > 0.1) {
                        laserCloudOriSurfVec[i] = pointOri;
                        coeffSelSurfVec[i] = coeff;
                        laserCloudOriSurfFlag[i] = true;
						
						numSearch++;
                    }
                }
            }
        }

		// ROS_WARN("Surf numSearch: [%d / %d]", numSearch, laserCloudSurfLastDSNum);

    }

    void combineOptimizationCoeffs()
    {
        // combine corner coeffs
        for (int i = 0; i < laserCloudCornerLastDSNum; ++i){
            if (laserCloudOriCornerFlag[i] == true){
                laserCloudOri->push_back(laserCloudOriCornerVec[i]);
                coeffSel->push_back(coeffSelCornerVec[i]);
            }
        }
        // combine surf coeffs
        for (int i = 0; i < laserCloudSurfLastDSNum; ++i){
            if (laserCloudOriSurfFlag[i] == true){
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
        // This optimization is from the original loam_velodyne by Ji Zhang, need to cope with coordinate transformation
        // lidar <- camera      ---     camera <- lidar
        // x = z                ---     x = y
        // y = x                ---     y = z
        // z = y                ---     z = x
        // roll = yaw           ---     roll = pitch
        // pitch = roll         ---     pitch = yaw
        // yaw = pitch          ---     yaw = roll

        // lidar -> camera
        float srx = sin(transformTobeSubMapped[1]);
        float crx = cos(transformTobeSubMapped[1]);
        float sry = sin(transformTobeSubMapped[2]);
        float cry = cos(transformTobeSubMapped[2]);
        float srz = sin(transformTobeSubMapped[0]);
        float crz = cos(transformTobeSubMapped[0]);

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
        for (int i = 0; i < laserCloudSelNum; i++) {
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
            float arx = (crx*sry*srz*pointOri.x + crx*crz*sry*pointOri.y - srx*sry*pointOri.z) * coeff.x
                      + (-srx*srz*pointOri.x - crz*srx*pointOri.y - crx*pointOri.z) * coeff.y
                      + (crx*cry*srz*pointOri.x + crx*cry*crz*pointOri.y - cry*srx*pointOri.z) * coeff.z;

            float ary = ((cry*srx*srz - crz*sry)*pointOri.x 
                      + (sry*srz + cry*crz*srx)*pointOri.y + crx*cry*pointOri.z) * coeff.x
                      + ((-cry*crz - srx*sry*srz)*pointOri.x 
                      + (cry*srz - crz*srx*sry)*pointOri.y - crx*sry*pointOri.z) * coeff.z;

            float arz = ((crz*srx*sry - cry*srz)*pointOri.x + (-cry*crz-srx*sry*srz)*pointOri.y)*coeff.x
                      + (crx*crz*pointOri.x - crx*srz*pointOri.y) * coeff.y
                      + ((sry*srz + cry*crz*srx)*pointOri.x + (crz*sry-cry*srx*srz)*pointOri.y)*coeff.z;
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
            for (int i = 5; i >= 0; i--) {
                if (matE.at<float>(0, i) < eignThre[i]) {
                    for (int j = 0; j < 6; j++) {
                        matV2.at<float>(i, j) = 0;
                    }
                    isDegenerate = true;
                } else {
                    break;
                }
            }
            matP = matV.inv() * matV2;
        }

        if (isDegenerate) {
            cv::Mat matX2(6, 1, CV_32F, cv::Scalar::all(0));
            matX.copyTo(matX2);
            matX = matP * matX2;
        }

        transformTobeSubMapped[0] += matX.at<float>(0, 0);
        transformTobeSubMapped[1] += matX.at<float>(1, 0);
        transformTobeSubMapped[2] += matX.at<float>(2, 0);
        transformTobeSubMapped[3] += matX.at<float>(3, 0);
        transformTobeSubMapped[4] += matX.at<float>(4, 0);
        transformTobeSubMapped[5] += matX.at<float>(5, 0);

        deltaR = sqrt(pow(pcl::rad2deg(matX.at<float>(0, 0)), 2) +
                      pow(pcl::rad2deg(matX.at<float>(1, 0)), 2) +
                      pow(pcl::rad2deg(matX.at<float>(2, 0)), 2));
        deltaT = sqrt(pow(matX.at<float>(3, 0) * 100, 2) +
                      pow(matX.at<float>(4, 0) * 100, 2) +
                      pow(matX.at<float>(5, 0) * 100, 2));

        if (deltaR < 0.003 && deltaT < 0.03) {
            return true; // converged
        }
        return false; // keep optimizing
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
                transformQuaternion.setRPY(transformTobeSubMapped[0], 0, 0);
                imuQuaternion.setRPY(cloudInfo.imuRollInit, 0, 0);
                tf::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid);
                transformTobeSubMapped[0] = rollMid;

                // slerp pitch
                transformQuaternion.setRPY(0, transformTobeSubMapped[1], 0);
                imuQuaternion.setRPY(0, cloudInfo.imuPitchInit, 0);
                tf::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid);
                transformTobeSubMapped[1] = pitchMid;
            }
        }

        transformTobeSubMapped[0] = constraintTransformation(transformTobeSubMapped[0], rotation_tollerance);
        transformTobeSubMapped[1] = constraintTransformation(transformTobeSubMapped[1], rotation_tollerance);
        transformTobeSubMapped[5] = constraintTransformation(transformTobeSubMapped[5], z_tollerance);

        incrementalOdometryAffineBack = trans2Affine3f(transformTobeSubMapped);

		transBef2Aft = incrementalOdometryAffineBack * transBef.inverse();
    }





	void IMUPreintegration()
	{
        std::lock_guard<std::mutex> lock(imuMtx);

    	double currentCorrectionTime = timeLaserInfoCur;
		double preCorrectionTime = timeLaserInfoPre;

		// make sure we have imu data to integrate
		if (imuQueOpt.empty()) return;

		nav_msgs::Odometry laserOdomeROS;
		laserOdomeROS.pose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(transformTobeSubMapped[0], 
																					transformTobeSubMapped[1], 
																					transformTobeSubMapped[2]);

		float p_x = transformTobeSubMapped[3];
		float p_y = transformTobeSubMapped[4];
		float p_z = transformTobeSubMapped[5];
		float r_x = laserOdomeROS.pose.pose.orientation.x;
		float r_y = laserOdomeROS.pose.pose.orientation.y;
		float r_z = laserOdomeROS.pose.pose.orientation.z;
		float r_w = laserOdomeROS.pose.pose.orientation.w;
		gtsam::Pose3 lidarPose = gtsam::Pose3(gtsam::Rot3::Quaternion(r_w, r_x, r_y, r_z), gtsam::Point3(p_x, p_y, p_z));
		
		// 0. initialize system
		if (systemInitialized == false) {
			resetOptimization();

			// pop old IMU message
			while (!imuQueOpt.empty()) {
				if (ROS_TIME(&imuQueOpt.front()) < currentCorrectionTime - delta_t) {
					lastImuT_opt = ROS_TIME(&imuQueOpt.front());
					imuQueOpt.pop_front();
				} else
					break;
			}

			// initial pose
			prevPose_ = lidarPose.compose(lidar2Imu);
			gtsam::PriorFactor<gtsam::Pose3> priorPose(X(0), prevPose_, priorPoseNoise);
			graphFactors.add(priorPose);
			// initial velocity
			prevVel_ = gtsam::Vector3(0, 0, 0);
			gtsam::PriorFactor<gtsam::Vector3> priorVel(V(0), prevVel_, priorVelNoise);
			graphFactors.add(priorVel);
			// initial bias
			prevBias_ = gtsam::imuBias::ConstantBias();
			gtsam::PriorFactor<gtsam::imuBias::ConstantBias> priorBias(B(0), prevBias_, priorBiasNoise);
			graphFactors.add(priorBias);
			// add values
			graphValues.insert(X(0), prevPose_);
			graphValues.insert(V(0), prevVel_);
			graphValues.insert(B(0), prevBias_);	

			// optimize once
			optimizer.update(graphFactors, graphValues);
			graphFactors.resize(0);
			graphValues.clear();

			imuIntegratorImu_->resetIntegrationAndSetBias(prevBias_);
			imuIntegratorOpt_->resetIntegrationAndSetBias(prevBias_);

			key = 1;
			systemInitialized = true;
			return;
		}
		
		// reset graph for speed
		if (key == 100) {
			// get updated noise before reset
			gtsam::noiseModel::Gaussian::shared_ptr updatedPoseNoise = gtsam::noiseModel::Gaussian::Covariance(optimizer.marginalCovariance(X(key - 1)));
			gtsam::noiseModel::Gaussian::shared_ptr updatedVelNoise = gtsam::noiseModel::Gaussian::Covariance(optimizer.marginalCovariance(V(key - 1)));
			gtsam::noiseModel::Gaussian::shared_ptr updatedBiasNoise = gtsam::noiseModel::Gaussian::Covariance(optimizer.marginalCovariance(B(key - 1)));
			// reset graph
			resetOptimization();
			// add pose
			gtsam::PriorFactor<gtsam::Pose3> priorPose(X(0), prevPose_, updatedPoseNoise);
			graphFactors.add(priorPose);
			// add velocity
			gtsam::PriorFactor<gtsam::Vector3> priorVel(V(0), prevVel_, updatedVelNoise);
			graphFactors.add(priorVel);
			// add bias
			gtsam::PriorFactor<gtsam::imuBias::ConstantBias> priorBias(B(0), prevBias_, updatedBiasNoise);
			graphFactors.add(priorBias);
			// add values
			graphValues.insert(X(0), prevPose_);
			graphValues.insert(V(0), prevVel_);
			graphValues.insert(B(0), prevBias_);
			// optimize once
			optimizer.update(graphFactors, graphValues);
			graphFactors.resize(0);
			graphValues.clear();

			key = 1;
		}

		// 1. integrate imu data and optimize
		while (!imuQueOpt.empty()) {
			// pop and integrate imu data that is between two optimizations
			sensor_msgs::Imu* thisImu = &imuQueOpt.front();
			double imuTime = ROS_TIME(thisImu);
			if (imuTime < preCorrectionTime - delta_t){
				lastImuT_opt = imuTime;
				imuQueOpt.pop_front();
			}
			else if (imuTime < currentCorrectionTime - delta_t) {
				double dt = (lastImuT_opt < 0) ? (1.0 / 100.0) : (imuTime - lastImuT_opt);
				imuIntegratorOpt_->integrateMeasurement(
					gtsam::Vector3(thisImu->linear_acceleration.x, thisImu->linear_acceleration.y, thisImu->linear_acceleration.z),
					gtsam::Vector3(thisImu->angular_velocity.x, thisImu->angular_velocity.y, thisImu->angular_velocity.z),
					dt);
				lastImuT_opt = imuTime;
				imuQueOpt.pop_front();
			} else
				break;
		}

		// add imu factor to graph
		const gtsam::PreintegratedImuMeasurements& preint_imu = dynamic_cast<const gtsam::PreintegratedImuMeasurements&>(*imuIntegratorOpt_);
		gtsam::ImuFactor imu_factor(X(key - 1), V(key - 1), X(key), V(key), B(key - 1), preint_imu);
		graphFactors.add(imu_factor);
		// add imu bias between factor
		graphFactors.add(gtsam::BetweenFactor<gtsam::imuBias::ConstantBias>(B(key - 1), B(key), gtsam::imuBias::ConstantBias(), gtsam::noiseModel::Diagonal::Sigmas(sqrt(imuIntegratorOpt_->deltaTij()) * noiseModelBetweenBias)));
		// add pose factor
		gtsam::Pose3 curPose = lidarPose.compose(lidar2Imu);
		gtsam::PriorFactor<gtsam::Pose3> pose_factor(X(key), curPose, correctionNoise);
		graphFactors.add(pose_factor);
		// insert predicted values
		gtsam::NavState propState_ = imuIntegratorOpt_->predict(prevState_, prevBias_);
		graphValues.insert(X(key), propState_.pose());
		graphValues.insert(V(key), propState_.v());
		graphValues.insert(B(key), prevBias_);
		// optimize
		optimizer.update(graphFactors, graphValues);
		optimizer.update();
		graphFactors.resize(0);
		graphValues.clear();
		// Overwrite the beginning of the preintegration for the next step.
		gtsam::Values result = optimizer.calculateEstimate();
		prevPose_ = result.at<gtsam::Pose3>(X(key));
		prevVel_ = result.at<gtsam::Vector3>(V(key));
		prevState_ = gtsam::NavState(prevPose_, prevVel_);
		prevBias_ = result.at<gtsam::imuBias::ConstantBias>(B(key));
		// Reset the optimization preintegration object.
		imuIntegratorOpt_->resetIntegrationAndSetBias(prevBias_);
		// check optimization
		if (failureDetection(prevVel_, prevBias_)) {
			resetParams();
			return;
		}


		// 2. after optiization, re-propagate imu odometry preintegration
		prevStateOdom = prevState_;
		prevBiasOdom = prevBias_;
		// first pop imu message older than current correction data
		double lastImuQT = -1;
		while (!imuQueImu.empty() && ROS_TIME(&imuQueImu.front()) < currentCorrectionTime - delta_t) {
			lastImuQT = ROS_TIME(&imuQueImu.front());
			imuQueImu.pop_front();
		}
		// repropogate
		if (!imuQueImu.empty()) {
			// reset bias use the newly optimized bias
			imuIntegratorImu_->resetIntegrationAndSetBias(prevBiasOdom);
			// integrate imu message from the beginning of this optimization
			for (int i = 0; i < (int)imuQueImu.size(); ++i) {
				sensor_msgs::Imu* thisImu = &imuQueImu[i];
				double imuTime = ROS_TIME(thisImu);
				double dt = (lastImuQT < 0) ? (1.0 / 100.0) : (imuTime - lastImuQT);

				imuIntegratorImu_->integrateMeasurement(
					gtsam::Vector3(thisImu->linear_acceleration.x, thisImu->linear_acceleration.y, thisImu->linear_acceleration.z),
					gtsam::Vector3(thisImu->angular_velocity.x, thisImu->angular_velocity.y, thisImu->angular_velocity.z),
					dt);
				lastImuQT = imuTime;
			}
		}

		++key;
		doneFirstOpt = true;


		// publish odometry
		nav_msgs::Odometry odometry;
		odometry.header.stamp = timeLaserInfoStamp;
		odometry.header.frame_id = odometryFrame;
		odometry.child_frame_id = "odom_imu";

		// transform imu pose to ldiar
		gtsam::Pose3 imuPose = gtsam::Pose3(prevStateOdom.quaternion(), prevStateOdom.position());
		gtsam::Pose3 curlidarPose = imuPose.compose(imu2Lidar);

		odometry.pose.pose.position.x = curlidarPose.translation().x();
		odometry.pose.pose.position.y = curlidarPose.translation().y();
		odometry.pose.pose.position.z = curlidarPose.translation().z();
		odometry.pose.pose.orientation.x = curlidarPose.rotation().toQuaternion().x();
		odometry.pose.pose.orientation.y = curlidarPose.rotation().toQuaternion().y();
		odometry.pose.pose.orientation.z = curlidarPose.rotation().toQuaternion().z();
		odometry.pose.pose.orientation.w = curlidarPose.rotation().toQuaternion().w();

		pubKeyframeIMUOdometry.publish(odometry);

		// ROS_WARN("timeLaserInfoStamp: %f, lastImuT_opt: %f.", timeLaserInfoStamp.toSec(), lastImuT_opt);
		
		// transformTobeSubMapped[3] = curlidarPose.translation().x();
		// transformTobeSubMapped[4] = curlidarPose.translation().y();
		// transformTobeSubMapped[5] = curlidarPose.translation().z();
		// transformTobeSubMapped[0] = curlidarPose.rotation().roll();
		// transformTobeSubMapped[1] = curlidarPose.rotation().pitch();
		// transformTobeSubMapped[2] = curlidarPose.rotation().yaw();

	}


	bool failureDetection(const gtsam::Vector3& velCur,
						  const gtsam::imuBias::ConstantBias& biasCur) {
		Eigen::Vector3f vel(velCur.x(), velCur.y(), velCur.z());
		if (vel.norm() > 30) {
			ROS_WARN("Large velocity, reset IMU-preintegration!");
			return true;
		}

		Eigen::Vector3f ba(biasCur.accelerometer().x(), biasCur.accelerometer().y(), biasCur.accelerometer().z());
		Eigen::Vector3f bg(biasCur.gyroscope().x(), biasCur.gyroscope().y(), biasCur.gyroscope().z());
		if (ba.norm() > 1.0 || bg.norm() > 1.0) {
			ROS_WARN("Large bias, reset IMU-preintegration!");
			return true;
		}

		return false;
	}

    void publishOdometry()
    {
        // Publish odometry for ROS (global)
        nav_msgs::Odometry laserOdometryROS;
        laserOdometryROS.header.stamp = timeLaserInfoStamp;
        laserOdometryROS.header.frame_id = odometryFrame;
        laserOdometryROS.child_frame_id = "odom_mapping";
        laserOdometryROS.pose.pose.position.x = transformTobeSubMapped[3];
        laserOdometryROS.pose.pose.position.y = transformTobeSubMapped[4];
        laserOdometryROS.pose.pose.position.z = transformTobeSubMapped[5];
        laserOdometryROS.pose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(transformTobeSubMapped[0], 
                                                                                         transformTobeSubMapped[1], 
                                                                                         transformTobeSubMapped[2]);
        pubKeyFrameOdometryGlobal.publish(laserOdometryROS);

        // Publish TF
        // static tf::TransformBroadcaster br;
        // tf::Transform t_odom_to_lidar = tf::Transform(tf::createQuaternionFromRPY(transformTobeSubMapped[0], transformTobeSubMapped[1], transformTobeSubMapped[2]),
        //                                               tf::Vector3(transformTobeSubMapped[3], transformTobeSubMapped[4], transformTobeSubMapped[5]));
        // tf::StampedTransform trans_odom_to_lidar = tf::StampedTransform(t_odom_to_lidar, timeLaserInfoStamp, odometryFrame, lidarFrame);
        // br.sendTransform(trans_odom_to_lidar);

        // Publish odometry for ROS (incremental)
        static bool lastIncreOdomPubFlag = false;
        static nav_msgs::Odometry laserOdomIncremental; // incremental odometry msg
        static Eigen::Affine3f increOdomAffine; // incremental odometry in affine
        if (lastIncreOdomPubFlag == false) {
            lastIncreOdomPubFlag = true;
            laserOdomIncremental = laserOdometryROS;
            increOdomAffine = trans2Affine3f(transformTobeSubMapped);
        } else {
            Eigen::Affine3f affineIncre = incrementalOdometryAffineFront.inverse() * incrementalOdometryAffineBack;
            increOdomAffine = increOdomAffine * affineIncre;
            float x, y, z, roll, pitch, yaw;
            pcl::getTranslationAndEulerAngles (increOdomAffine, x, y, z, roll, pitch, yaw);

            laserOdomIncremental.header.stamp = timeLaserInfoStamp;
            laserOdomIncremental.header.frame_id = odometryFrame;
            laserOdomIncremental.child_frame_id = "odom_mapping";
            laserOdomIncremental.pose.pose.position.x = x;
            laserOdomIncremental.pose.pose.position.y = y;
            laserOdomIncremental.pose.pose.position.z = z;
            laserOdomIncremental.pose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(roll, pitch, yaw);
        }
        pubKeyFrameOdometryIncremental.publish(laserOdomIncremental);

        // ROS_WARN("Finshed  publishOdometry !");
    }


    void publishKeyFrameCloud()
    {
        // pubCloudRegisteredRaw;
        pcl::PointCloud<PointXYZIL>::Ptr thisCornerKeyFrame(new pcl::PointCloud<PointXYZIL>());
        pcl::PointCloud<PointXYZIL>::Ptr thisSurfKeyFrame(new pcl::PointCloud<PointXYZIL>());
        pcl::copyPointCloud(*laserCloudCornerLastDS,  *thisCornerKeyFrame);
        pcl::copyPointCloud(*laserCloudSurfLastDS,    *thisSurfKeyFrame);
        
        *thisSurfKeyFrame += *thisCornerKeyFrame;

        Eigen::Affine3f transTobe = trans2Affine3f(transformTobeSubMapped);
        *thisSurfKeyFrame = *transformPointCloud(thisSurfKeyFrame, transTobe);

        publishLabelCloud(&pubCloudRegisteredRaw, thisSurfKeyFrame, timeLaserInfoStamp, odometryFrame);

        // pubKeyFramePath;

        // pubKeyFramePoseGlobal
        publishCloud(&pubKeyFramePoseGlobal, keyFramePoses3D, timeLaserInfoStamp, odometryFrame);
    }


    void publishSubMapCloud()
    {
        pcl::PointCloud<PointXYZIL>::Ptr cloud_raw(new pcl::PointCloud<PointXYZIL>);
        currentSubMap->merge_feature_points(cloud_raw);
        publishLabelCloud(&pubCloudCurSubMap, cloud_raw, timeLaserInfoStamp, lidarFrame);
        
        publishCloud(&pubSubMapId, subMapPose3D, timeLaserInfoStamp, odometryFrame);       
    }




    /*****************************
     * @brief
     * @param input
     *****************************/
    void loopClosureThread() 
    {
        ros::Rate rate(loopClosureFrequency);
        int processID = 0;
        EPSCGeneration epscGen;

        while (ros::ok()) {
			if (loopClosureEnableFlag == false) 
			{
				ros::spinOnce();
				rate.sleep();
				continue;
			}

			if(!keyFrameQueue.empty())
			{
				keyframe_Ptr curKeyFramePtr = keyFrameQueue.front();

				auto thisCurId = subMapInfo.find(curKeyFramePtr->submap_id);
				if (thisCurId == subMapInfo.end()) 
					continue;

				keyFrameQueue.pop_front();

				auto t1 = ros::Time::now();

				Eigen::Affine3f curPose = pclPointToAffine3f(curKeyFramePtr->optimized_pose);
				epscGen.loopDetection(curKeyFramePtr->cloud_corner, curKeyFramePtr->cloud_surface,
									  curKeyFramePtr->semantic_raw, curPose);

				int loopKeyCur = epscGen.current_frame_id;
				std::vector<int> loopKeyPre;
				loopKeyPre.assign(epscGen.matched_frame_id.begin(), epscGen.matched_frame_id.end());
				std::vector<Eigen::Affine3f> matched_init_transform;
				matched_init_transform.assign(epscGen.matched_frame_transform.begin(), epscGen.matched_frame_transform.end());

				cv_bridge::CvImage out_msg;
				out_msg.header.frame_id = lidarFrame;
				out_msg.header.stamp = curKeyFramePtr->timeInfoStamp;
				out_msg.encoding = sensor_msgs::image_encodings::RGB8;
				
				
				if (UsingSEPSCFlag){
					out_msg.image = epscGen.getLastSEPSCRGB();
					pubSEPSC.publish(out_msg.toImageMsg());
				}
				if (UsingEPSCFlag){
					out_msg.image = epscGen.getLastEPSCRGB();
					pubEPSC.publish(out_msg.toImageMsg());
				}
				if (UsingFEPSCFlag){
					out_msg.image = epscGen.getLastFEPSCRGB();
					pubFEPSC.publish(out_msg.toImageMsg());
				}
				if (UsingSCFlag){
					out_msg.image = epscGen.getLastSCRGB();
					pubSC.publish(out_msg.toImageMsg()); 
				}	
				if (UsingISCFlag){
					out_msg.image = epscGen.getLastISCRGB();
					pubISC.publish(out_msg.toImageMsg());   
				}
				if (UsingSSCFlag){
					out_msg.image = epscGen.getLastSSCRGB();
					pubSSC.publish(out_msg.toImageMsg());
				}				
				
				

				// curKeyFramePtr->global_descriptor = epscGen.getLastSEPSCMONO();
					
				ros::Time t2 = ros::Time::now();
				ROS_WARN("Detect Loop Closure Time: %.3f", (t2 - t1).toSec());
				
				if (loopKeyPre.empty()) 
				{
					ROS_WARN("loopKeyPre is empty !");
					continue;
				}

				std::cout << std::endl;
				std::cout << "--- loop detection ---" << std::endl;
				std::cout << "keyframe_id : " << curKeyFramePtr->keyframe_id << std::endl;
				std::cout << "loopKeyCur : " << loopKeyCur << std::endl;
				std::cout << "num_candidates: " << loopKeyPre.size() << std::endl;
				
				for (int i = 0; i < loopKeyPre.size(); i++) 
				{
					loopIndexContainerTest.insert(make_pair(loopKeyCur, loopKeyPre[i]));
					out_msg.image = epscGen.getLastFEPSCRGB(loopKeyPre[i]);
					pubSEPSC.publish(out_msg.toImageMsg());
					std::cout << "loopKeyPre [" << i << "]:" << loopKeyPre[i] << std::endl;
				}

				// visualizeLoopClosureTest();

				// int bestMatched = -1;
				// if (detectLoopClosure(loopKeyCur, loopKeyPre, matched_init_transform, bestMatched) == false)
				// 	continue;
		
				// curKeyFramePtr->loop_container.push_back(bestMatched);	
				
				
				int loopSubMapCur = -1;
				auto it = keyframeInSubmapIndex.find(loopKeyCur);
				if(it != keyframeInSubmapIndex.end()){
					loopSubMapCur = curKeyFramePtr->submap_id;
					std::cout << "Find loopSubMapCur: " << loopSubMapCur << " keyframeInSubmapIndex: " << keyframeInSubmapIndex[loopKeyCur] << std::endl;
				}else{	
					ROS_WARN("loopClosureThread -->> Dont find loopSubMapCur %d in keyframeInSubmapIndex !", loopSubMapCur);
					continue;
				}

				vector<int> loopSubMapPre, loopKeyPreLast;
				vector<Eigen::Affine3f> curKey2PreKeyInitTrans;
				for (int i = 0; i < loopKeyPre.size(); i++) 
				{
					auto it = keyframeInSubmapIndex.find(loopKeyPre[i]);
					if(it != keyframeInSubmapIndex.end())
					{
						auto beg = loopIndexContainer.lower_bound(loopSubMapCur);
						auto end = loopIndexContainer.upper_bound(loopSubMapCur);
						bool isFlag = false;
						for(auto m = beg; m != end; m++)
						{
							if(it->second == m->second)
								isFlag = true;
						}

						if(isFlag) continue;

						loopSubMapPre.push_back(it->second);
						loopKeyPreLast.push_back(loopKeyPre[i]);
						curKey2PreKeyInitTrans.push_back(matched_init_transform[i]);
						std::cout << "loopSubMapPre : " << it->second << std::endl;
					}else{
						ROS_WARN("loopClosureThread -->> Dont find loopKeyPre %d in keyframeInSubmapIndex !", loopKeyPre[i]);
						continue;
					}
				}

				// sort(loopSubMapPre.begin(), loopSubMapPre.end());
				// loopSubMapPre.erase(unique(loopSubMapPre.begin(), loopSubMapPre.end()), loopSubMapPre.end());
				
				if (loopSubMapPre.empty()) 
				{
					ROS_WARN("loopSubMapPre is empty !");
					continue;
				}

				int bestMatched = -1;
				if (detectLoopClosureForSubMap(curKeyFramePtr, loopKeyCur, loopSubMapPre, loopKeyPreLast, curKey2PreKeyInitTrans, bestMatched) == false)
					continue;

				// visualizeLoopClosure();

				curKeyFramePtr->loop_container.push_back(bestMatched);	

				// curKeyFramePtr->free_all();										  
			}

			rate.sleep();
			ros::spinOnce();			
        }
    }



    bool detectLoopClosureForSubMapICP(keyframe_Ptr cur_keyframe, int &loopKeyCur, vector<int> &loopSubMapPre, int &bestMatched) 
    {
        pcl::PointCloud<PointXYZIL>::Ptr cureKeyframeCloud( new pcl::PointCloud<PointXYZIL>());
        pcl::PointCloud<PointXYZIL>::Ptr prevKeyframeCloud( new pcl::PointCloud<PointXYZIL>());

		*cureKeyframeCloud += *cur_keyframe->semantic_dynamic;
		*cureKeyframeCloud += *cur_keyframe->semantic_pole;
		*cureKeyframeCloud += *cur_keyframe->semantic_ground;
		*cureKeyframeCloud += *cur_keyframe->semantic_building;

        std::cout << "matching..." << std::flush;
        auto t1 = ros::Time::now();
		static OptimizedICPGN myICP(30, 10);


        int bestID = -1;
        double bestScore = std::numeric_limits<double>::max();
		Eigen::Affine3f correctionLidarFrame;
		Eigen::Affine3f key2PreSubMapTrans;
        static Eigen::Affine3f correctionKey2PreSubMap = Eigen::Affine3f::Identity();

        for (int i = 0; i < loopSubMapPre.size(); i++) 
        {
            auto thisPreId = subMapInfo.find(loopSubMapPre[i]);
            if (thisPreId != subMapInfo.end()) 
			{
                std::cout << "loopContainerHandler: loopSubMapPre : " << loopSubMapPre[i] << std::endl;
                
				prevKeyframeCloud->clear();
                *prevKeyframeCloud += *subMapInfo[loopSubMapPre[i]]->submap_dynamic;
                *prevKeyframeCloud += *subMapInfo[loopSubMapPre[i]]->submap_pole;
                *prevKeyframeCloud += *subMapInfo[loopSubMapPre[i]]->submap_ground;
                *prevKeyframeCloud += *subMapInfo[loopSubMapPre[i]]->submap_building;
                // *prevKeyframeCloud += *subMapInfo[loopSubMapPre[i]]->submap_outlier;
				myICP.SetTargetCloud(prevKeyframeCloud);

		publishLabelCloud(&pubTestPre, prevKeyframeCloud, timeLaserInfoStamp, odometryFrame);
			
				Eigen::Affine3f subMapTrans = pclPointToAffine3f(subMapInfo[loopSubMapPre[i]]->submap_pose_6D_optimized);
				Eigen::Affine3f keyTrans = pclPointToAffine3f(cur_keyframe->optimized_pose);
				key2PreSubMapTrans = subMapTrans.inverse() * keyTrans;
				key2PreSubMapTrans = correctionKey2PreSubMap * key2PreSubMapTrans;

				// Align clouds
				pcl::PointCloud<PointXYZIL>::Ptr tmpCloud( new pcl::PointCloud<PointXYZIL>());
				*tmpCloud = *transformPointCloud(cureKeyframeCloud, key2PreSubMapTrans);
		publishLabelCloud(&pubTestCurLoop, tmpCloud, timeLaserInfoStamp, odometryFrame);
                
				Eigen::Matrix4f predict_pose = Eigen::Matrix4f::Identity();
				Eigen::Matrix4f result_pose = Eigen::Matrix4f::Identity();
				pcl::PointCloud<PointXYZIL>::Ptr unused_result( new pcl::PointCloud<PointXYZIL>());
				myICP.Match(cureKeyframeCloud, predict_pose, unused_result, result_pose);
		
		publishLabelCloud(&pubTestCurICP, unused_result, timeLaserInfoStamp, odometryFrame);

                double score = myICP.GetFitnessScore();

                if (myICP.HasConverged() == false || score > bestScore) 
                    continue;
                bestScore = score;
                bestMatched = loopSubMapPre[i];
                bestID = i;
                correctionLidarFrame = result_pose;
            
            } else {
                bestMatched = -1;
                ROS_WARN("loopSubMapPre do not find !");
            }
        }

        if (loopKeyCur == -1 || bestMatched == -1) 
            return false;

        auto t2 = ros::Time::now();
        std::cout << " done" << std::endl;
        std::cout << "best_score: " << bestScore << "    time: " << (t2 - t1).toSec() << "[sec]" << std::endl;

        // if (bestScore < 1.0) 
        // {
		// 	correctionKey2PreSubMap = correctionLidarFrame;
        //     std::cout << "correctionKey2PreSubMap update !" << std::endl;
        // }

        if (bestScore > historyKeyframeFitnessScore) 
        {
            std::cout << "loop not found..." << std::endl;
            return false;
        }
        std::cout << "loop found!!" << std::endl;
		
		correctionKey2PreSubMap = correctionLidarFrame;

		Eigen::Affine3f key2CurSubMapTrans;
		int loopSubMapCur = -1;
		auto it = keyframeInSubmapIndex.find(loopKeyCur);
		if(it != keyframeInSubmapIndex.end()){
			key2CurSubMapTrans = pclPointToAffine3f(cur_keyframe->relative_pose);
			loopSubMapCur = cur_keyframe->submap_id;
			std::cout << "Find loopSubMapCur: " << loopSubMapCur << " keyframeInSubmapIndex: " << keyframeInSubmapIndex[loopKeyCur] << std::endl;
		}else{	
			ROS_WARN("loopClosureThread -->> Dont find loopSubMapCur %d in keyframeInSubmapIndex !", loopSubMapCur);
			return false;
		}

        float X, Y, Z, ROLL, PITCH, YAW;
        Eigen::Affine3f tCorrect = correctionLidarFrame * key2PreSubMapTrans * key2CurSubMapTrans.inverse();  // pre-multiplying -> successive rotation about a fixed frame
        pcl::getTranslationAndEulerAngles(tCorrect, X, Y, Z, ROLL, PITCH, YAW);
        gtsam::Pose3 pose = Pose3(Rot3::RzRyRx(ROLL, PITCH, YAW), Point3(X, Y, Z));
        gtsam::Vector Vector6(6);
        float noiseScore = 0.01;
        // float noiseScore = bestScore*0.01;
        Vector6 << noiseScore, noiseScore, noiseScore, noiseScore, noiseScore, noiseScore;
        noiseModel::Diagonal::shared_ptr constraintNoise = noiseModel::Diagonal::Variances(Vector6);

        // Add pose constraint
        // mtx.lock();
        loopIndexQueue.push_back(make_pair(loopSubMapCur, bestMatched));
        loopPoseQueue.push_back(pose);
        loopNoiseQueue.push_back(constraintNoise);
        // mtx.unlock();

        // add loop constriant
        loopIndexContainer.insert(std::make_pair(loopSubMapCur, bestMatched));

        return true;
    }


    bool detectLoopClosureForSubMapTEASER(keyframe_Ptr cur_keyframe, int &loopKeyCur, vector<int> &loopSubMapPre, int &bestMatched) 
    {
        pcl::PointCloud<PointXYZIL>::Ptr cureKeyframeCloud( new pcl::PointCloud<PointXYZIL>());
        pcl::PointCloud<PointXYZIL>::Ptr prevKeyframeCloud( new pcl::PointCloud<PointXYZIL>());

		*cureKeyframeCloud += *cur_keyframe->semantic_dynamic;
		*cureKeyframeCloud += *cur_keyframe->semantic_pole;
		*cureKeyframeCloud += *cur_keyframe->semantic_ground;
		*cureKeyframeCloud += *cur_keyframe->semantic_building;

        std::cout << "matching..." << std::flush;
        auto t1 = ros::Time::now();

        int bestID = -1;
        double bestScore = std::numeric_limits<double>::max();
		Eigen::Affine3f correctionLidarFrame;
		Eigen::Affine3f key2PreSubMapTrans;
        static Eigen::Affine3f correctionKey2PreSubMap = Eigen::Affine3f::Identity();

        for (int i = 0; i < loopSubMapPre.size(); i++) 
        {
            auto thisPreId = subMapInfo.find(loopSubMapPre[i]);
            if (thisPreId != subMapInfo.end()) 
			{
                std::cout << "loopContainerHandler: loopSubMapPre : " << loopSubMapPre[i] << std::endl;
                
				prevKeyframeCloud->clear();
                *prevKeyframeCloud += *subMapInfo[loopSubMapPre[i]]->submap_dynamic;
                *prevKeyframeCloud += *subMapInfo[loopSubMapPre[i]]->submap_pole;
                *prevKeyframeCloud += *subMapInfo[loopSubMapPre[i]]->submap_ground;
                *prevKeyframeCloud += *subMapInfo[loopSubMapPre[i]]->submap_building;
                // *prevKeyframeCloud += *subMapInfo[loopSubMapPre[i]]->submap_outlier;
        		SubMapManager::voxel_downsample_pcl(prevKeyframeCloud, prevKeyframeCloud, 0.1);
		publishLabelCloud(&pubTestPre, prevKeyframeCloud, timeLaserInfoStamp, odometryFrame);
			
				Eigen::Affine3f subMapTrans = pclPointToAffine3f(subMapInfo[loopSubMapPre[i]]->submap_pose_6D_optimized);
				Eigen::Affine3f keyTrans = pclPointToAffine3f(cur_keyframe->optimized_pose);
				key2PreSubMapTrans = subMapTrans.inverse() * keyTrans;
				key2PreSubMapTrans = correctionKey2PreSubMap * key2PreSubMapTrans;

				// Align clouds
				pcl::PointCloud<PointXYZIL>::Ptr tmpCloud( new pcl::PointCloud<PointXYZIL>());
				*tmpCloud = *transformPointCloud(cureKeyframeCloud, key2PreSubMapTrans);
		publishLabelCloud(&pubTestCurLoop, tmpCloud, timeLaserInfoStamp, odometryFrame);
                

				Eigen::Matrix4f result_pose = Eigen::Matrix4f::Identity();
				int result = coarse_reg_teaser(prevKeyframeCloud, tmpCloud, result_pose, 0.2, 8);

                bestScore = result;
                bestMatched = loopSubMapPre[i];
                bestID = i;
                correctionLidarFrame = result_pose;
				
				pcl::PointCloud<PointXYZIL>::Ptr unused_result( new pcl::PointCloud<PointXYZIL>());
				*unused_result = *transformPointCloud(tmpCloud, correctionLidarFrame);
		publishLabelCloud(&pubTestCurICP, unused_result, timeLaserInfoStamp, odometryFrame);
            } else {
                bestMatched = -1;
                ROS_WARN("loopSubMapPre do not find !");
            }
        }

        if (loopKeyCur == -1 || bestMatched == -1) 
            return false;

        auto t2 = ros::Time::now();
        std::cout << " done" << std::endl;
        std::cout << "best_score: " << bestScore << "    time: " << (t2 - t1).toSec() << "[sec]" << std::endl;

        // if (bestScore > historyKeyframeFitnessScore) 
        // {
        //     std::cout << "loop not found..." << std::endl;
        //     return false;
        // }
        std::cout << "loop found!!" << std::endl;
		
		correctionKey2PreSubMap = correctionLidarFrame;

		Eigen::Affine3f key2CurSubMapTrans;
		int loopSubMapCur = -1;
		auto it = keyframeInSubmapIndex.find(loopKeyCur);
		if(it != keyframeInSubmapIndex.end()){
			key2CurSubMapTrans = pclPointToAffine3f(cur_keyframe->relative_pose);
			loopSubMapCur = cur_keyframe->submap_id;
			std::cout << "Find loopSubMapCur: " << loopSubMapCur << " keyframeInSubmapIndex: " << keyframeInSubmapIndex[loopKeyCur] << std::endl;
		}else{	
			ROS_WARN("loopClosureThread -->> Dont find loopSubMapCur %d in keyframeInSubmapIndex !", loopSubMapCur);
			return false;
		}

        float X, Y, Z, ROLL, PITCH, YAW;
        Eigen::Affine3f tCorrect = correctionLidarFrame * key2PreSubMapTrans * key2CurSubMapTrans.inverse();  // pre-multiplying -> successive rotation about a fixed frame
        pcl::getTranslationAndEulerAngles(tCorrect, X, Y, Z, ROLL, PITCH, YAW);
        gtsam::Pose3 pose = Pose3(Rot3::RzRyRx(ROLL, PITCH, YAW), Point3(X, Y, Z));
        gtsam::Vector Vector6(6);
        float noiseScore = 0.01;
        // float noiseScore = bestScore*0.01;
        Vector6 << noiseScore, noiseScore, noiseScore, noiseScore, noiseScore, noiseScore;
        noiseModel::Diagonal::shared_ptr constraintNoise = noiseModel::Diagonal::Variances(Vector6);

        // Add pose constraint
        // mtx.lock();
        loopIndexQueue.push_back(make_pair(loopSubMapCur, bestMatched));
        loopPoseQueue.push_back(pose);
        loopNoiseQueue.push_back(constraintNoise);
        // mtx.unlock();

        // add loop constriant
        loopIndexContainer.insert(std::make_pair(loopSubMapCur, bestMatched));

        return true;
    }


    bool detectLoopClosureForSubMap(
			keyframe_Ptr cur_keyframe,  int &loopKeyCur, 
			vector<int> &loopSubMapPre, vector<int> &loopKeyPreLast,
			vector<Eigen::Affine3f> &curKey2PreKeyInitTrans, 
			int &bestMatched) 
    {
        pcl::PointCloud<PointXYZIL>::Ptr cureKeyframeCloud( new pcl::PointCloud<PointXYZIL>());
        pcl::PointCloud<PointXYZIL>::Ptr prevKeyframeCloud( new pcl::PointCloud<PointXYZIL>());

		*cureKeyframeCloud += *cur_keyframe->semantic_dynamic;
		*cureKeyframeCloud += *cur_keyframe->semantic_pole;
		*cureKeyframeCloud += *cur_keyframe->semantic_ground;
		*cureKeyframeCloud += *cur_keyframe->semantic_building;

        std::cout << "matching..." << std::flush;
        auto t1 = ros::Time::now();

		// static pcl::NormalDistributionsTransform<PointXYZIL, PointXYZIL>::Ptr reg(new pcl::NormalDistributionsTransform<PointXYZIL, PointXYZIL>());
		// reg->setTransformationEpsilon(0.01); //为终止条件设置最小转换差异
		// reg->setStepSize(0.1); //为More-Thuente线搜索设置最大步长
		// reg->setResolution(1.0); //设置NDT网格结构的分辨率（VoxelGridCovariance）
		// reg->setMaximumIterations(35); //设置匹配迭代的最大次数

        // ICP Settings
        static pcl::IterativeClosestPoint<PointXYZIL, PointXYZIL>::Ptr reg(new pcl::IterativeClosestPoint<PointXYZIL, PointXYZIL>());
        // static pcl::GeneralizedIterativeClosestPoint<PointXYZIL, PointXYZIL> reg(new pcl::GeneralizedIterativeClosestPoint<PointXYZIL, PointXYZIL>());
        reg->setMaxCorrespondenceDistance(10);
        reg->setMaximumIterations(30);
        reg->setTransformationEpsilon(1e-4);
        reg->setEuclideanFitnessEpsilon(1e-4);
        reg->setRANSACIterations(0);

		// static pcl::Registration<PointXYZIL, PointXYZIL>::Ptr reg = select_registration_method("FAST_VGICP");

        int bestID = -1;
        double bestScore = std::numeric_limits<double>::max();
		Eigen::Affine3f correctionLidarFrame;
		Eigen::Affine3f key2PreSubMapTrans;
        // static Eigen::Affine3f correctionKey2PreSubMap = Eigen::Affine3f::Identity();

        for (int i = 0; i < loopSubMapPre.size(); i++) 
        {
            auto thisPreId = subMapInfo.find(loopSubMapPre[i]);
            if (thisPreId != subMapInfo.end()) 
			{
                std::cout << "loopContainerHandler: loopSubMapPre : " << loopSubMapPre[i] << std::endl;
                
				prevKeyframeCloud->clear();
                *prevKeyframeCloud += *subMapInfo[loopSubMapPre[i]]->submap_dynamic;
                *prevKeyframeCloud += *subMapInfo[loopSubMapPre[i]]->submap_pole;
                *prevKeyframeCloud += *subMapInfo[loopSubMapPre[i]]->submap_ground;
                *prevKeyframeCloud += *subMapInfo[loopSubMapPre[i]]->submap_building;
                // *prevKeyframeCloud += *subMapInfo[loopSubMapPre[i]]->submap_outlier;
                reg->setInputTarget(prevKeyframeCloud);

		publishLabelCloud(&pubTestPre, prevKeyframeCloud, timeLaserInfoStamp, odometryFrame);
			
	
				auto preKey2SubMapPoses = subMapInfo[loopSubMapPre[i]]->keyframe_poses_6D_map;
				if (preKey2SubMapPoses.find(loopKeyPreLast[i]) != preKey2SubMapPoses.end())
				{
					// 使用 EPSC 检测的位姿 作为初始位姿
					ROS_WARN("Using EPSC Init Pose !");
					Eigen::Affine3f preKey2PreSubMapTrans = pclPointToAffine3f(subMapInfo[loopSubMapPre[i]]->keyframe_poses_6D_map[loopKeyPreLast[i]]);
					key2PreSubMapTrans = preKey2PreSubMapTrans * curKey2PreKeyInitTrans[i];
				} else {
					// 使用 Pose 作为初始位姿
					ROS_WARN("Using SubMap Init Pose !");
					Eigen::Affine3f subMapTrans = pclPointToAffine3f(subMapInfo[loopSubMapPre[i]]->submap_pose_6D_optimized);
					Eigen::Affine3f keyTrans = pclPointToAffine3f(cur_keyframe->optimized_pose);
					key2PreSubMapTrans = subMapTrans.inverse() * keyTrans;
				}
				// key2PreSubMapTrans = correctionKey2PreSubMap * key2PreSubMapTrans;

				// Align clouds
				pcl::PointCloud<PointXYZIL>::Ptr tmpCloud( new pcl::PointCloud<PointXYZIL>());
				*tmpCloud = *transformPointCloud(cureKeyframeCloud, key2PreSubMapTrans);
				reg->setInputSource(tmpCloud);

				// reg->setInputSource(cureKeyframeCloud);


		publishLabelCloud(&pubTestCurLoop, tmpCloud, timeLaserInfoStamp, odometryFrame);

                pcl::PointCloud<PointXYZIL>::Ptr unused_result( new pcl::PointCloud<PointXYZIL>());
                reg->align(*unused_result);
		
		publishLabelCloud(&pubTestCurICP, unused_result, timeLaserInfoStamp, odometryFrame);

                double score = reg->getFitnessScore();
                if (reg->hasConverged() == false || score > bestScore) 
                    continue;
                bestScore = score;
                bestMatched = loopSubMapPre[i];
                bestID = i;
                correctionLidarFrame = reg->getFinalTransformation();

            } else {
                bestMatched = -1;
                ROS_WARN("loopSubMapPre do not find !");
            }
        }

        if (loopKeyCur == -1 || bestMatched == -1) 
            return false;

        auto t2 = ros::Time::now();
        std::cout << " done" << std::endl;
        std::cout << "best_score: " << bestScore << "    time: " << (t2 - t1).toSec() << "[sec]" << std::endl;

        if (bestScore > historyKeyframeFitnessScore) 
        {
            std::cout << "loop not found..." << std::endl;
            return false;
        }
        std::cout << "loop found!!" << std::endl;
		
		// correctionKey2PreSubMap = correctionLidarFrame;

		Eigen::Affine3f key2CurSubMapTrans;
		int loopSubMapCur = -1;
		auto it = keyframeInSubmapIndex.find(loopKeyCur);
		if(it != keyframeInSubmapIndex.end()){
			key2CurSubMapTrans = pclPointToAffine3f(cur_keyframe->relative_pose);
			loopSubMapCur = cur_keyframe->submap_id;
			std::cout << "Find loopSubMapCur: " << loopSubMapCur << " keyframeInSubmapIndex: " << keyframeInSubmapIndex[loopKeyCur] << std::endl;
		}else{	
			ROS_WARN("loopClosureThread -->> Dont find loopSubMapCur %d in keyframeInSubmapIndex !", loopSubMapCur);
			return false;
		}

        float X, Y, Z, ROLL, PITCH, YAW;
		Eigen::Affine3f curSubMap2KeyTrans = key2CurSubMapTrans.inverse();
		// Eigen::Affine3f tWrong = pclPointToAffine3f(subMapInfo[loopSubMapCur]->submap_pose_6D_optimized);
        Eigen::Affine3f tCorrect = correctionLidarFrame * key2PreSubMapTrans * curSubMap2KeyTrans;  // pre-multiplying -> successive rotation about a fixed frame
        
		pcl::getTranslationAndEulerAngles(tCorrect, X, Y, Z, ROLL, PITCH, YAW);
        gtsam::Pose3 poseFrom = Pose3(Rot3::RzRyRx(ROLL, PITCH, YAW), Point3(X, Y, Z));
		// gtsam::Pose3 poseTo = pclPointTogtsamPose3(subMapInfo[bestMatched]->submap_pose_6D_optimized);
        gtsam::Pose3 poseTo = Pose3(Rot3::RzRyRx(0.0, 0.0, 0.0), Point3(0.0, 0.0, 0.0));

		gtsam::Vector Vector6(6);
        float noiseScore = 0.001;
        // float noiseScore = bestScore*0.01;
        Vector6 << noiseScore, noiseScore, noiseScore, noiseScore, noiseScore, noiseScore;
        noiseModel::Diagonal::shared_ptr constraintNoise = noiseModel::Diagonal::Variances(Vector6);

        // Add pose constraint
        // mtx.lock();
        loopIndexQueue.push_back(make_pair(loopSubMapCur, bestMatched));
        loopPoseQueue.push_back(poseFrom.between(poseTo));
        loopNoiseQueue.push_back(constraintNoise);
        // mtx.unlock();

        // add loop constriant
        loopIndexContainer.insert(std::make_pair(loopSubMapCur, bestMatched));


		auto thisCurId = subMapInfo.find(loopSubMapCur);
		if (thisCurId != subMapInfo.end()) 
		{
			pcl::PointCloud<PointXYZIL>::Ptr tmpCloud( new pcl::PointCloud<PointXYZIL>());
			tmpCloud->clear();
			*tmpCloud += *subMapInfo[loopSubMapCur]->submap_dynamic;
			*tmpCloud += *subMapInfo[loopSubMapCur]->submap_pole;
			*tmpCloud += *subMapInfo[loopSubMapCur]->submap_ground;
			*tmpCloud += *subMapInfo[loopSubMapCur]->submap_building;
			*tmpCloud = *transformPointCloud(tmpCloud, tCorrect);
			publishLabelCloud(&pubTestCur, tmpCloud, timeLaserInfoStamp, odometryFrame);
		}
        return true;
    }



    bool detectLoopClosure(int &loopKeyCur, vector<int> &loopKeyPre,
                           vector<Eigen::Affine3f> &matched_init_transform,
                           int &bestMatched) 
    {
        pcl::PointCloud<PointXYZIL>::Ptr cureKeyframeCloud( new pcl::PointCloud<PointXYZIL>());
        pcl::PointCloud<PointXYZIL>::Ptr prevKeyframeCloud( new pcl::PointCloud<PointXYZIL>());

        auto thisCurId = keyFrameInfo.find(loopKeyCur);
        if (thisCurId != keyFrameInfo.end()) {
            loopKeyCur = (int)keyFrameInfo[loopKeyCur]->keyframe_id;
			*cureKeyframeCloud += *keyFrameInfo[loopKeyCur]->semantic_dynamic;
			*cureKeyframeCloud += *keyFrameInfo[loopKeyCur]->semantic_pole;
			*cureKeyframeCloud += *keyFrameInfo[loopKeyCur]->semantic_ground;
			*cureKeyframeCloud += *keyFrameInfo[loopKeyCur]->semantic_building;
            // *cureKeyframeCloud = *keyFrameInfo[loopKeyCur]->semantic_raw;
        } else {
            loopKeyCur = -1;
            ROS_WARN("LoopKeyCur do not find !");
            return false;
        }

        std::cout << "matching..." << std::flush;
        auto t1 = ros::Time::now();

        // ICP Settings
        static pcl::IterativeClosestPoint<PointXYZIL, PointXYZIL> reg;
        reg.setMaxCorrespondenceDistance(20);
        reg.setMaximumIterations(40);
        reg.setTransformationEpsilon(1e-4);
        reg.setEuclideanFitnessEpsilon(1e-4);
        reg.setRANSACIterations(0);

        int bestID = -1;
        double bestScore = std::numeric_limits<double>::max();
        Eigen::Affine3f correctionLidarFrame;

        for (int i = 0; i < loopKeyPre.size(); i++) 
        {

		publishLabelCloud(&pubTestCur, cureKeyframeCloud, timeLaserInfoStamp, odometryFrame);

            // Align clouds
            pcl::PointCloud<PointXYZIL>::Ptr tmpCloud( new pcl::PointCloud<PointXYZIL>());
            *tmpCloud = *transformPointCloud(cureKeyframeCloud, matched_init_transform[i]);
            reg.setInputSource(tmpCloud);

		float x, y, z, roll, pitch, yaw;
        pcl::getTranslationAndEulerAngles(matched_init_transform[i], x, y, z, roll, pitch, yaw);
		std::cout << "matched_init_transform[" << i << "]: " 
				  << x << ", " << y << ", " << z << ", " 
				  << roll << ", " << pitch << ", " << yaw << std::endl;
		publishLabelCloud(&pubTestCurLoop, tmpCloud, timeLaserInfoStamp, odometryFrame);

            auto thisPreId = keyFrameInfo.find(loopKeyPre[i]);
            if (thisPreId != keyFrameInfo.end()) {
                int PreID = (int)keyFrameInfo[loopKeyPre[i]]->keyframe_id;
                std::cout << "loopContainerHandler: loopKeyPre : " << PreID << std::endl;

                prevKeyframeCloud->clear();
				*prevKeyframeCloud += *keyFrameInfo[loopKeyPre[i]]->semantic_dynamic;
				*prevKeyframeCloud += *keyFrameInfo[loopKeyPre[i]]->semantic_pole;
				*prevKeyframeCloud += *keyFrameInfo[loopKeyPre[i]]->semantic_ground;
				*prevKeyframeCloud += *keyFrameInfo[loopKeyPre[i]]->semantic_building;
                // *prevKeyframeCloud = *keyFrameInfo[loopKeyPre[i]]->semantic_raw;
                reg.setInputTarget(prevKeyframeCloud);
        	
		publishLabelCloud(&pubTestPre, prevKeyframeCloud, timeLaserInfoStamp, odometryFrame);

                pcl::PointCloud<PointXYZIL>::Ptr unused_result( new pcl::PointCloud<PointXYZIL>());
                reg.align(*unused_result);
		publishLabelCloud(&pubTestCurICP, unused_result, timeLaserInfoStamp, odometryFrame);

                double score = reg.getFitnessScore();
                if (reg.hasConverged() == false || score > bestScore) 
                    continue;
                bestScore = score;
                bestMatched = PreID;
                bestID = i;
                correctionLidarFrame = reg.getFinalTransformation();
            } else {
                bestMatched = -1;
                ROS_WARN("loopKeyPre do not find !");
            }
        }

        if (loopKeyCur == -1 || bestMatched == -1) 
            return false;

        auto t2 = ros::Time::now();
        std::cout << " done" << std::endl;
        std::cout << "best_score: " << bestScore << "    time: " << (t2 - t1).toSec() << "[sec]" << std::endl;

        if (bestScore > historyKeyframeFitnessScore) 
        {
            std::cout << "loop not found..." << std::endl;
            return false;
        }
        std::cout << "loop found!!" << std::endl;

        float X, Y, Z, ROLL, PITCH, YAW;
        Eigen::Affine3f tCorrect = correctionLidarFrame * matched_init_transform[bestID];  // pre-multiplying -> successive rotation about a fixed frame
        pcl::getTranslationAndEulerAngles(tCorrect, X, Y, Z, ROLL, PITCH, YAW);
        gtsam::Pose3 pose = Pose3(Rot3::RzRyRx(ROLL, PITCH, YAW), Point3(X, Y, Z));
        gtsam::Vector Vector6(6);
        float noiseScore = 0.01;
        // float noiseScore = bestScore*0.01;
        Vector6 << noiseScore, noiseScore, noiseScore, noiseScore, noiseScore, noiseScore;
        noiseModel::Diagonal::shared_ptr constraintNoise = noiseModel::Diagonal::Variances(Vector6);

        // Add pose constraint
        // mtx.lock();
        loopIndexQueue.push_back(make_pair(loopKeyCur, bestMatched));
        loopPoseQueue.push_back(pose);
        loopNoiseQueue.push_back(constraintNoise);
        // mtx.unlock();

        // add loop constriant
        loopIndexContainer.insert(std::make_pair(loopKeyCur, bestMatched));

        return true;
    }




    void visualizeLoopClosureTest() 
    {
        visualization_msgs::MarkerArray markerArray;
		
		visualization_msgs::Marker markerNodeId;
        markerNodeId.header.frame_id = odometryFrame;
        markerNodeId.header.stamp = ros::Time::now();
        markerNodeId.action = visualization_msgs::Marker::ADD;
        markerNodeId.type =  visualization_msgs::Marker::TEXT_VIEW_FACING;
        markerNodeId.ns = "test_loop_nodes_id";
        markerNodeId.id = 0;
        markerNodeId.pose.orientation.w = 1;
        markerNodeId.scale.z = 0.4; 
        markerNodeId.color.r = 0; markerNodeId.color.g = 0; markerNodeId.color.b = 255;
        markerNodeId.color.a = 1;

        // loop nodes
        visualization_msgs::Marker markerNode;
        markerNode.header.frame_id = odometryFrame;
        markerNode.header.stamp = ros::Time::now();
        markerNode.action = visualization_msgs::Marker::ADD;
        markerNode.type = visualization_msgs::Marker::SPHERE_LIST;
        markerNode.ns = "loop_nodes";
        markerNode.id = 0;
        markerNode.pose.orientation.w = 1;
        markerNode.scale.x = 0.4;
        markerNode.scale.y = 0.4;
        markerNode.scale.z = 0.4;
        markerNode.color.r = 0.0;
        markerNode.color.g = 0.0;
        markerNode.color.b = 1.0;
        markerNode.color.a = 1;
        // loop edges
        visualization_msgs::Marker markerEdge;
        markerEdge.header.frame_id = odometryFrame;
        markerEdge.header.stamp = ros::Time::now();
        markerEdge.action = visualization_msgs::Marker::ADD;
        markerEdge.type = visualization_msgs::Marker::LINE_LIST;
        markerEdge.ns = "loop_edges";
        markerEdge.id = 1;
        markerEdge.pose.orientation.w = 1;
        markerEdge.scale.x = 0.3;
        markerEdge.scale.y = 0.3;
        markerEdge.scale.z = 0.3;
        markerEdge.color.r = 0.0;
        markerEdge.color.g = 0.0;
        markerEdge.color.b = 1.0;
        markerEdge.color.a = 1;


        for (auto it = loopIndexContainerTest.begin(); it != loopIndexContainerTest.end(); ++it) 
        {
            int key_cur = it->first;
            int key_pre = it->second;

            geometry_msgs::Pose pose;
            pose.position.x =  keyFramePosesIndex3D[key_cur].x;
            pose.position.y =  keyFramePosesIndex3D[key_cur].y;
            pose.position.z =  keyFramePosesIndex3D[key_cur].z + 0.15;
            int k = key_cur;
            ostringstream str;
            str << k;
            markerNodeId.id = k;
            markerNodeId.text = str.str();
            markerNodeId.pose = pose;
            markerArray.markers.push_back(markerNodeId);

            pose.position.x =  keyFramePosesIndex3D[key_pre].x;
            pose.position.y =  keyFramePosesIndex3D[key_pre].y;
            pose.position.z =  keyFramePosesIndex3D[key_pre].z + 0.15;
            k = key_pre;
            ostringstream str_pre;
            str_pre << k;
            markerNodeId.id = k;
            markerNodeId.text = str_pre.str();
            markerNodeId.pose = pose;
            markerArray.markers.push_back(markerNodeId);

            geometry_msgs::Point p;
            p.x = keyFramePosesIndex3D[key_cur].x;
            p.y = keyFramePosesIndex3D[key_cur].y;
            p.z = keyFramePosesIndex3D[key_cur].z;
            markerNode.points.push_back(p);
            markerEdge.points.push_back(p);
            p.x = keyFramePosesIndex3D[key_pre].x;
            p.y = keyFramePosesIndex3D[key_pre].y;
            p.z = keyFramePosesIndex3D[key_pre].z;
            markerNode.points.push_back(p);
            markerEdge.points.push_back(p);
        }

        markerArray.markers.push_back(markerNode);
        markerArray.markers.push_back(markerEdge);
        pubLoopConstraintEdgeTest.publish(markerArray);

		ROS_WARN("loopIndexContainerTest Size: %d !", loopIndexContainerTest.size());
		ROS_INFO("Finshed  visualizeLoopClosureTest !");

    }



    void visualizeLoopClosure() 
    {
        visualization_msgs::MarkerArray markerArray;

		visualization_msgs::Marker markerNodeId;
        markerNodeId.header.frame_id = odometryFrame;
        markerNodeId.header.stamp = ros::Time::now();
        markerNodeId.action = visualization_msgs::Marker::ADD;
        markerNodeId.type =  visualization_msgs::Marker::TEXT_VIEW_FACING;
        markerNodeId.ns = "loop_nodes_id";
        markerNodeId.id = 0;
        markerNodeId.pose.orientation.w = 1;
        markerNodeId.scale.z = 0.5; 
        markerNodeId.color.r = 0; markerNodeId.color.g = 0; markerNodeId.color.b = 255;
        markerNodeId.color.a = 1;

        // loop nodes
        visualization_msgs::Marker markerNode;
        markerNode.header.frame_id = odometryFrame;
        markerNode.header.stamp = ros::Time::now();
        markerNode.action = visualization_msgs::Marker::ADD;
        markerNode.type = visualization_msgs::Marker::SPHERE_LIST;
        markerNode.ns = "loop_nodes";
        markerNode.id = 0;
        markerNode.pose.orientation.w = 1;
        markerNode.scale.x = 0.5;
        markerNode.scale.y = 0.5;
        markerNode.scale.z = 0.5;
        markerNode.color.r = 1;
        markerNode.color.g = 0.0;
        markerNode.color.b = 0;
        markerNode.color.a = 1;
        // loop edges
        visualization_msgs::Marker markerEdge;
        markerEdge.header.frame_id = odometryFrame;
        markerEdge.header.stamp = ros::Time::now();
        markerEdge.action = visualization_msgs::Marker::ADD;
        markerEdge.type = visualization_msgs::Marker::LINE_LIST;
        markerEdge.ns = "loop_edges";
        markerEdge.id = 1;
        markerEdge.pose.orientation.w = 1;
        markerEdge.scale.x = 0.4;
        markerEdge.scale.y = 0.4;
        markerEdge.scale.z = 0.4;
        markerEdge.color.r = 1.0;
        markerEdge.color.g = 0.0;
        markerEdge.color.b = 0;
        markerEdge.color.a = 1;


        for (auto it = loopIndexContainer.begin(); it != loopIndexContainer.end(); ++it) 
        {
            int key_cur = it->first;
            int key_pre = it->second;

            geometry_msgs::Pose pose;
            pose.position.x =  subMapInfo[key_cur]->submap_pose_6D_optimized.x;
            pose.position.y =  subMapInfo[key_cur]->submap_pose_6D_optimized.y;
            pose.position.z =  subMapInfo[key_cur]->submap_pose_6D_optimized.z + 0.15;
            int k = key_cur;
            ostringstream str;
            str << k;
            markerNodeId.id = k;
            markerNodeId.text = str.str();
            markerNodeId.pose = pose;
            markerArray.markers.push_back(markerNodeId);

            pose.position.x =  subMapInfo[key_pre]->submap_pose_6D_optimized.x;
            pose.position.y =  subMapInfo[key_pre]->submap_pose_6D_optimized.y;
            pose.position.z =  subMapInfo[key_pre]->submap_pose_6D_optimized.z + 0.15;
            k = key_pre;
            ostringstream str_pre;
            str_pre << k;
            markerNodeId.id = k;
            markerNodeId.text = str_pre.str();
            markerNodeId.pose = pose;
            markerArray.markers.push_back(markerNodeId);


            geometry_msgs::Point p;
            p.x = subMapInfo[key_cur]->submap_pose_6D_optimized.x;
            p.y = subMapInfo[key_cur]->submap_pose_6D_optimized.y;
            p.z = subMapInfo[key_cur]->submap_pose_6D_optimized.z;
            markerNode.points.push_back(p);
            markerEdge.points.push_back(p);
            p.x = subMapInfo[key_pre]->submap_pose_6D_optimized.x;
            p.y = subMapInfo[key_pre]->submap_pose_6D_optimized.y;
            p.z = subMapInfo[key_pre]->submap_pose_6D_optimized.z;
            markerNode.points.push_back(p);
            markerEdge.points.push_back(p);
        }

        markerArray.markers.push_back(markerNode);
        markerArray.markers.push_back(markerEdge);
        pubLoopConstraintEdge.publish(markerArray);

		ROS_WARN("LoopIndexContainer Size: %d !", loopIndexContainer.size());
		ROS_INFO("Finshed  visualizeLoopClosure !");

    }

};










class SubMapOptmizationNode : public SubMapManager<PointXYZIL> {
 public:
    ros::Subscriber subGPS;
    ros::Subscriber subOdom;

    std::mutex gpsMtx;
    std::deque<nav_msgs::Odometry> gpsQueue;

    ros::Publisher pubCloudMap;

    ros::Publisher pubSubMapId;
    ros::Publisher pubKeyFrameId;

    ros::Publisher pubSubMapConstraintEdge;

    ros::Publisher pubLoopConstraintEdge;

    ros::Publisher pubSubMapOdometryGlobal;
    ros::Publisher pubOdometryGlobal;
	
	int curSubMapId = 0;
	int preSubMapId = 0;
	submap_Ptr curSubMapPtr;
    ros::Time timeSubMapInfoStamp;

    pcl::PointCloud<PointType>::Ptr cloudSubMapPoses3D;
    pcl::PointCloud<PointTypePose>::Ptr cloudSubMapPoses6D;
	pcl::PointCloud<PointType>::Ptr cloudKeyFramePoses3D;
    pcl::PointCloud<PointTypePose>::Ptr cloudKeyFramePoses6D;

    map<int,PointType> SubMapPoses3D;
    map<int,PointTypePose> SubMapPoses6D;
        
	float transformTobeMapped[6];
    float transPredictionMapped[6];

    Eigen::Affine3f transPointAssociateToSubMap;
    Eigen::Affine3f incrementalOdometryAffineFront;
    Eigen::Affine3f incrementalOdometryAffineBack;

	Eigen::Affine3f transBef;
	Eigen::Affine3f transBef2Aft = Eigen::Affine3f::Identity();

	pcl::PointCloud<PointXYZIL>::Ptr laserCloudCornerLast; // corner feature set from odoOptimization
    pcl::PointCloud<PointXYZIL>::Ptr laserCloudSurfLast; // surf feature set from odoOptimization
    pcl::PointCloud<PointXYZIL>::Ptr laserCloudCornerLastDS; // downsampled corner featuer set from odoOptimization
    pcl::PointCloud<PointXYZIL>::Ptr laserCloudSurfLastDS; // downsampled surf featuer set from odoOptimization
	
    pcl::PointCloud<PointXYZIL>::Ptr laserCloudCornerFromSubMap;
    pcl::PointCloud<PointXYZIL>::Ptr laserCloudSurfFromSubMap;
    
	pcl::KdTreeFLANN<PointXYZIL>::Ptr kdtreeCornerFromSubMap;
    pcl::KdTreeFLANN<PointXYZIL>::Ptr kdtreeSurfFromSubMap;

    int laserCloudCornerLastDSNum = 0;
    int laserCloudSurfLastDSNum = 0;

    pcl::PointCloud<PointXYZIL>::Ptr laserCloudOri;
    pcl::PointCloud<PointXYZIL>::Ptr coeffSel;

    std::vector<PointXYZIL> laserCloudOriCornerVec; // corner point holder for parallel computation
    std::vector<PointXYZIL> coeffSelCornerVec;
    std::vector<bool> laserCloudOriCornerFlag;
    std::vector<PointXYZIL> laserCloudOriSurfVec; // surf point holder for parallel computation
    std::vector<PointXYZIL> coeffSelSurfVec;
    std::vector<bool> laserCloudOriSurfFlag;
    
    bool isDegenerate = false;
    Eigen::Matrix<float, 6, 6> matP;


	bool aLoopIsClosed = false;

	float deltaR = 100;
	float deltaT = 100;

    // gtsam
    NonlinearFactorGraph gtSAMgraph;
    Values initialEstimate;
    Values optimizedEstimate;
    ISAM2 *isam;
    Values isamCurrentEstimate;
    Eigen::MatrixXd poseCovariance;


    vector<int> subMapContainer; // from new to old
    vector<pair<int, int>>subMapIndexContainerAll; // from new to old
    vector<pair<int, int>> subMapIndexContainer;
    vector<gtsam::Pose3> subMapPoseQueue;
    vector<gtsam::noiseModel::Diagonal::shared_ptr> subMapNoiseQueue;


	ros::Publisher pubTestPre;
    ros::Publisher pubTestCur;
    ros::Publisher pubTestPreN;
    ros::Publisher pubTestCurN;

	ros::Publisher pubLoopConstraintEdgeTest;
	map<int, PointType> keyFramePosesIndex3D;
    
    void allocateMemory() 
	{
        cloudSubMapPoses3D.reset(new pcl::PointCloud<PointType>());
        cloudSubMapPoses6D.reset(new pcl::PointCloud<PointTypePose>()); 

        cloudKeyFramePoses3D.reset(new pcl::PointCloud<PointType>());
        cloudKeyFramePoses6D.reset(new pcl::PointCloud<PointTypePose>()); 

        laserCloudCornerLast.reset(new pcl::PointCloud<PointXYZIL>()); 
        laserCloudSurfLast.reset(new pcl::PointCloud<PointXYZIL>()); 
        laserCloudCornerLastDS.reset(new pcl::PointCloud<PointXYZIL>()); 
        laserCloudSurfLastDS.reset(new pcl::PointCloud<PointXYZIL>()); 

        laserCloudCornerFromSubMap.reset(new pcl::PointCloud<PointXYZIL>());
        laserCloudSurfFromSubMap.reset(new pcl::PointCloud<PointXYZIL>());

        for (int i = 0; i < 6; ++i)
        {
            transformTobeMapped[i] = 0;
            transPredictionMapped[i]=0.0;
        }
        kdtreeCornerFromSubMap.reset(new pcl::KdTreeFLANN<PointXYZIL>());
        kdtreeSurfFromSubMap.reset(new pcl::KdTreeFLANN<PointXYZIL>());

        laserCloudOri.reset(new pcl::PointCloud<PointXYZIL>());
        coeffSel.reset(new pcl::PointCloud<PointXYZIL>());

        laserCloudOriCornerVec.resize(N_SCAN * Horizon_SCAN);
        coeffSelCornerVec.resize(N_SCAN * Horizon_SCAN);
        laserCloudOriCornerFlag.resize(N_SCAN * Horizon_SCAN);
        laserCloudOriSurfVec.resize(N_SCAN * Horizon_SCAN);
        coeffSelSurfVec.resize(N_SCAN * Horizon_SCAN);
        laserCloudOriSurfFlag.resize(N_SCAN * Horizon_SCAN);
    
        std::fill(laserCloudOriCornerFlag.begin(), laserCloudOriCornerFlag.end(), false);
        std::fill(laserCloudOriSurfFlag.begin(), laserCloudOriSurfFlag.end(), false);
        
        matP.setZero();

        ISAM2Params parameters;
        parameters.relinearizeThreshold = 0.1;
        parameters.relinearizeSkip = 1;
        isam = new ISAM2(parameters);



	}

    SubMapOptmizationNode() 
    {
        // subGPS = nh.subscribe<nav_msgs::Odometry>(gpsTopic, 2000, &SubMapOptmizationNode::gpsHandler, this, ros::TransportHints().tcpNoDelay());
		subOdom   = nh.subscribe<nav_msgs::Odometry>(odomTopic + "/lidar", 200, &SubMapOptmizationNode::odomHandler, this, ros::TransportHints().tcpNoDelay());


        pubCloudMap = nh.advertise<sensor_msgs::PointCloud2>("lis_slam/mapping/map_global", 1);
                
        pubSubMapId = nh.advertise<sensor_msgs::PointCloud2>("lis_slam/mapping/submap_id", 1);
        pubKeyFrameId = nh.advertise<sensor_msgs::PointCloud2>("lis_slam/mapping/keyframe_id", 1);

		pubSubMapConstraintEdge = nh.advertise<visualization_msgs::MarkerArray>("lis_slam/mapping/submap_constraints", 1);
        
		pubSubMapOdometryGlobal = nh.advertise<nav_msgs::Odometry> (odomTopic + "/submap", 200);
		pubOdometryGlobal = nh.advertise<nav_msgs::Odometry> (odomTopic + "/fusion", 200);
                
		allocateMemory();

		pubTestPre = nh.advertise<sensor_msgs::PointCloud2>("/op_pre", 1);
        pubTestCur = nh.advertise<sensor_msgs::PointCloud2>("/op_cur", 1);
        pubTestPreN = nh.advertise<sensor_msgs::PointCloud2>("/op_pre_n", 1);
        pubTestCurN = nh.advertise<sensor_msgs::PointCloud2>("/op_cur_n", 1);

		pubLoopConstraintEdgeTest = nh.advertise<visualization_msgs::MarkerArray>("lis_slam/mapping/loop_closure_constraints_test", 1);
		pubLoopConstraintEdge = nh.advertise<visualization_msgs::MarkerArray>("lis_slam/mapping/loop_closure_constraints", 1);
    }

    void gpsHandler(const nav_msgs::Odometry::ConstPtr &msgIn) 
    {
        std::lock_guard<std::mutex> lock(gpsMtx);
        gpsQueue.push_back(*msgIn);
    }

	void odomHandler(const nav_msgs::Odometry::ConstPtr &msgIn )
	{
		Eigen::Affine3f odomAffine = odom2affine(*msgIn);
        Eigen::Affine3f lidarOdomAffine = transBef2Aft * odomAffine;

        float x, y, z, roll, pitch, yaw;
        pcl::getTranslationAndEulerAngles(lidarOdomAffine, x, y, z, roll, pitch, yaw);

        // publish latest odometry
        nav_msgs::Odometry laserOdometry = *msgIn;
        laserOdometry.pose.pose.position.x = x;
        laserOdometry.pose.pose.position.y = y;
        laserOdometry.pose.pose.position.z = z;
        laserOdometry.pose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(roll, pitch, yaw);
        pubOdometryGlobal.publish(laserOdometry);

        static tf::TransformBroadcaster br;
        tf::Transform t_odom_to_lidar = tf::Transform(tf::createQuaternionFromRPY(roll, pitch, yaw), 
																										tf::Vector3(x, y, z));
        tf::StampedTransform trans_map_to_lidar = tf::StampedTransform(t_odom_to_lidar,  laserOdometry.header.stamp, odometryFrame, lidarFrame);
        br.sendTransform(trans_map_to_lidar);

	}
    



	
    /*****************************
     * @brief
     * @param input
     *****************************/
    void visualizeGlobalMapThread() 
	{
        ros::Rate rate(0.2); //0.1
        while (ros::ok())
		{
			updateKeyFrame();
            publishGlobalMap();

			visualizeLoopClosureTest();
			visualizeLoopClosure();

			publishCloud(&pubSubMapId, cloudSubMapPoses3D, ros::Time::now(), odometryFrame);       
			publishCloud(&pubKeyFrameId, cloudKeyFramePoses3D, ros::Time::now(), odometryFrame);       
            
			ros::spinOnce();
            rate.sleep();
        }

		if (savePCD == false)
            return;
		
		cout << "****************************************************" << endl;
        cout << "Saving map to pcd files ..." << endl;
        // create directory and remove old files;
        savePCDDirectory = std::getenv("HOME") + savePCDDirectory;
        int unused = system((std::string("exec rm -r ") + savePCDDirectory).c_str());
        unused = system((std::string("mkdir ") + savePCDDirectory).c_str());
        // save key frame transformations
        pcl::io::savePCDFileASCII(savePCDDirectory + "trajectory.pcd", *cloudSubMapPoses3D);
        pcl::io::savePCDFileASCII(savePCDDirectory + "transformations.pcd", *cloudSubMapPoses6D);
        // extract global point cloud map        
        pcl::PointCloud<PointXYZIL>::Ptr globalMapCloud(new pcl::PointCloud<PointXYZIL>());

        for (auto it = subMapInfo.begin(); it != subMapInfo.end(); it++) {
            *globalMapCloud += *transformPointCloud(it->second->submap_dynamic,  &it->second->submap_pose_6D_optimized);
            *globalMapCloud += *transformPointCloud(it->second->submap_pole,  &it->second->submap_pose_6D_optimized);
            *globalMapCloud += *transformPointCloud(it->second->submap_ground,  &it->second->submap_pose_6D_optimized);
            *globalMapCloud += *transformPointCloud(it->second->submap_building,  &it->second->submap_pose_6D_optimized);
            *globalMapCloud += *transformPointCloud(it->second->submap_outlier,  &it->second->submap_pose_6D_optimized);
            cout << "\r" << std::flush << "Processing feature cloud " << it->first << " of " << cloudSubMapPoses6D->size() << " ...";
        }

        pcl::io::savePCDFileASCII(savePCDDirectory + "cloudGlobal.pcd", *globalMapCloud);
        cout << "****************************************************" << endl;
        cout << "Saving map to pcd files completed" << endl;
	}


	void updateKeyFrame()
	{
		cloudKeyFramePoses3D->clear();
		cloudKeyFramePoses6D->clear();

		// cloudSubMapPoses3D->clear();

        for (auto it = subMapInfo.begin(); it != subMapInfo.end(); it++)
		{
			auto thisSubMap = it->second;

			Eigen::Affine3f  curSubMapAffine = pclPointToAffine3f(thisSubMap->submap_pose_6D_optimized);
			// std::cout << "Update Submap [" << thisSubMap->submap_id <<"] Keyfrmae Pose ..." << std::endl;
			// cloudKeyFramePoses3D->push_back(thisSubMap->submap_pose_3D_optimized);

			for(auto inter = thisSubMap->keyframe_poses_6D_map.begin(); inter != thisSubMap->keyframe_poses_6D_map.end(); inter++)
			{
				Eigen::Affine3f  curKeyFrameAffine = pclPointToAffine3f(inter->second);
				Eigen::Affine3f  curKeyFramePose = curSubMapAffine * curKeyFrameAffine;

				// std::cout << "Update Keyfrmae [" << inter->first <<"] Pose ..." << std::endl;
				float transform[6];
				pcl::getTranslationAndEulerAngles(curKeyFramePose, transform[3], transform[4], transform[5], 
                                                      transform[0], transform[1], transform[2]);
				PointType  point3d = trans2PointType(transform, inter->first);

				cloudKeyFramePoses3D->push_back(point3d);
				keyFramePosesIndex3D[ inter->first] = point3d;
			}
        }
	}


    void publishGlobalMap()
    {
        if (cloudSubMapPoses3D->points.empty() == true)
            return;

        pcl::PointCloud<PointXYZIL>::Ptr globalMapCloud(new pcl::PointCloud<PointXYZIL>());

        for (auto it = subMapInfo.begin(); it != subMapInfo.end(); it++) {
			auto itEnd = subMapInfo.end(); itEnd--;
			if(FINISHMAP == true || it != itEnd)
			{
				*globalMapCloud += *transformPointCloud(it->second->submap_dynamic,  &it->second->submap_pose_6D_optimized);
				*globalMapCloud += *transformPointCloud(it->second->submap_pole,  &it->second->submap_pose_6D_optimized);
				*globalMapCloud += *transformPointCloud(it->second->submap_ground,  &it->second->submap_pose_6D_optimized);
				*globalMapCloud += *transformPointCloud(it->second->submap_building,  &it->second->submap_pose_6D_optimized);
				*globalMapCloud += *transformPointCloud(it->second->submap_outlier,  &it->second->submap_pose_6D_optimized);
				cout << "\r" << std::flush << "Processing feature cloud " << it->first << " of " << cloudSubMapPoses6D->size() << " ...";
			}
        }

        publishLabelCloud(&pubCloudMap, globalMapCloud, timeSubMapInfoStamp, odometryFrame);
    }


    void visualizeLoopClosureTest() 
    {
        visualization_msgs::MarkerArray markerArray;
		
		visualization_msgs::Marker markerNodeId;
        markerNodeId.header.frame_id = odometryFrame;
        markerNodeId.header.stamp = ros::Time::now();
        markerNodeId.action = visualization_msgs::Marker::ADD;
        markerNodeId.type =  visualization_msgs::Marker::TEXT_VIEW_FACING;
        markerNodeId.ns = "test_loop_nodes_id";
        markerNodeId.id = 0;
        markerNodeId.pose.orientation.w = 1;
        markerNodeId.scale.z = 0.4; 
        markerNodeId.color.r = 0; markerNodeId.color.g = 0; markerNodeId.color.b = 255;
        markerNodeId.color.a = 1;

        // loop nodes
        visualization_msgs::Marker markerNode;
        markerNode.header.frame_id = odometryFrame;
        markerNode.header.stamp = ros::Time::now();
        markerNode.action = visualization_msgs::Marker::ADD;
        markerNode.type = visualization_msgs::Marker::SPHERE_LIST;
        markerNode.ns = "loop_nodes";
        markerNode.id = 0;
        markerNode.pose.orientation.w = 1;
        markerNode.scale.x = 0.4;
        markerNode.scale.y = 0.4;
        markerNode.scale.z = 0.4;
        markerNode.color.r = 0.0;
        markerNode.color.g = 0.0;
        markerNode.color.b = 1.0;
        markerNode.color.a = 1;
        // loop edges
        visualization_msgs::Marker markerEdge;
        markerEdge.header.frame_id = odometryFrame;
        markerEdge.header.stamp = ros::Time::now();
        markerEdge.action = visualization_msgs::Marker::ADD;
        markerEdge.type = visualization_msgs::Marker::LINE_LIST;
        markerEdge.ns = "loop_edges";
        markerEdge.id = 1;
        markerEdge.pose.orientation.w = 1;
        markerEdge.scale.x = 0.3;
        markerEdge.scale.y = 0.3;
        markerEdge.scale.z = 0.3;
        markerEdge.color.r = 0.0;
        markerEdge.color.g = 0.0;
        markerEdge.color.b = 1.0;
        markerEdge.color.a = 1;


        for (auto it = loopIndexContainerTest.begin(); it != loopIndexContainerTest.end(); ++it) 
        {
            int key_cur = it->first;
            int key_pre = it->second;

            geometry_msgs::Pose pose;
            pose.position.x =  keyFramePosesIndex3D[key_cur].x;
            pose.position.y =  keyFramePosesIndex3D[key_cur].y;
            pose.position.z =  keyFramePosesIndex3D[key_cur].z + 0.15;
            int k = key_cur;
            ostringstream str;
            str << k;
            markerNodeId.id = k;
            markerNodeId.text = str.str();
            markerNodeId.pose = pose;
            markerArray.markers.push_back(markerNodeId);

            pose.position.x =  keyFramePosesIndex3D[key_pre].x;
            pose.position.y =  keyFramePosesIndex3D[key_pre].y;
            pose.position.z =  keyFramePosesIndex3D[key_pre].z + 0.15;
            k = key_pre;
            ostringstream str_pre;
            str_pre << k;
            markerNodeId.id = k;
            markerNodeId.text = str_pre.str();
            markerNodeId.pose = pose;
            markerArray.markers.push_back(markerNodeId);

            geometry_msgs::Point p;
            p.x = keyFramePosesIndex3D[key_cur].x;
            p.y = keyFramePosesIndex3D[key_cur].y;
            p.z = keyFramePosesIndex3D[key_cur].z;
            markerNode.points.push_back(p);
            markerEdge.points.push_back(p);
            p.x = keyFramePosesIndex3D[key_pre].x;
            p.y = keyFramePosesIndex3D[key_pre].y;
            p.z = keyFramePosesIndex3D[key_pre].z;
            markerNode.points.push_back(p);
            markerEdge.points.push_back(p);
        }

        markerArray.markers.push_back(markerNode);
        markerArray.markers.push_back(markerEdge);
        pubLoopConstraintEdgeTest.publish(markerArray);

		// ROS_WARN("loopIndexContainerTest Size: %d !", loopIndexContainerTest.size());
		// ROS_INFO("Finshed  visualizeLoopClosureTest !");

    }



    void visualizeLoopClosure() 
    {
        visualization_msgs::MarkerArray markerArray;

		visualization_msgs::Marker markerNodeId;
        markerNodeId.header.frame_id = odometryFrame;
        markerNodeId.header.stamp = ros::Time::now();
        markerNodeId.action = visualization_msgs::Marker::ADD;
        markerNodeId.type =  visualization_msgs::Marker::TEXT_VIEW_FACING;
        markerNodeId.ns = "loop_nodes_id";
        markerNodeId.id = 0;
        markerNodeId.pose.orientation.w = 1;
        markerNodeId.scale.z = 0.5; 
        markerNodeId.color.r = 0; markerNodeId.color.g = 0; markerNodeId.color.b = 255;
        markerNodeId.color.a = 1;

        // loop nodes
        visualization_msgs::Marker markerNode;
        markerNode.header.frame_id = odometryFrame;
        markerNode.header.stamp = ros::Time::now();
        markerNode.action = visualization_msgs::Marker::ADD;
        markerNode.type = visualization_msgs::Marker::SPHERE_LIST;
        markerNode.ns = "loop_nodes";
        markerNode.id = 0;
        markerNode.pose.orientation.w = 1;
        markerNode.scale.x = 0.5;
        markerNode.scale.y = 0.5;
        markerNode.scale.z = 0.5;
        markerNode.color.r = 1;
        markerNode.color.g = 0.0;
        markerNode.color.b = 0;
        markerNode.color.a = 1;
        // loop edges
        visualization_msgs::Marker markerEdge;
        markerEdge.header.frame_id = odometryFrame;
        markerEdge.header.stamp = ros::Time::now();
        markerEdge.action = visualization_msgs::Marker::ADD;
        markerEdge.type = visualization_msgs::Marker::LINE_LIST;
        markerEdge.ns = "loop_edges";
        markerEdge.id = 1;
        markerEdge.pose.orientation.w = 1;
        markerEdge.scale.x = 0.4;
        markerEdge.scale.y = 0.4;
        markerEdge.scale.z = 0.4;
        markerEdge.color.r = 1.0;
        markerEdge.color.g = 0.0;
        markerEdge.color.b = 0;
        markerEdge.color.a = 1;


        for (auto it = loopIndexContainer.begin(); it != loopIndexContainer.end(); ++it) 
        {
            int key_cur = it->first;
            int key_pre = it->second;

            geometry_msgs::Pose pose;
            pose.position.x =  subMapInfo[key_cur]->submap_pose_6D_optimized.x;
            pose.position.y =  subMapInfo[key_cur]->submap_pose_6D_optimized.y;
            pose.position.z =  subMapInfo[key_cur]->submap_pose_6D_optimized.z + 0.15;
            int k = key_cur;
            ostringstream str;
            str << k;
            markerNodeId.id = k;
            markerNodeId.text = str.str();
            markerNodeId.pose = pose;
            markerArray.markers.push_back(markerNodeId);

            pose.position.x =  subMapInfo[key_pre]->submap_pose_6D_optimized.x;
            pose.position.y =  subMapInfo[key_pre]->submap_pose_6D_optimized.y;
            pose.position.z =  subMapInfo[key_pre]->submap_pose_6D_optimized.z + 0.15;
            k = key_pre;
            ostringstream str_pre;
            str_pre << k;
            markerNodeId.id = k;
            markerNodeId.text = str_pre.str();
            markerNodeId.pose = pose;
            markerArray.markers.push_back(markerNodeId);


            geometry_msgs::Point p;
            p.x = subMapInfo[key_cur]->submap_pose_6D_optimized.x;
            p.y = subMapInfo[key_cur]->submap_pose_6D_optimized.y;
            p.z = subMapInfo[key_cur]->submap_pose_6D_optimized.z;
            markerNode.points.push_back(p);
            markerEdge.points.push_back(p);
            p.x = subMapInfo[key_pre]->submap_pose_6D_optimized.x;
            p.y = subMapInfo[key_pre]->submap_pose_6D_optimized.y;
            p.z = subMapInfo[key_pre]->submap_pose_6D_optimized.z;
            markerNode.points.push_back(p);
            markerEdge.points.push_back(p);
        }

        markerArray.markers.push_back(markerNode);
        markerArray.markers.push_back(markerEdge);
        pubLoopConstraintEdge.publish(markerArray);

		// ROS_WARN("LoopIndexContainer Size: %d !", loopIndexContainer.size());
		// ROS_INFO("Finshed  visualizeLoopClosure !");

    }















    /*****************************
     * @brief
     * @param input
     *****************************/
    void subMapOptmizationThread() 
	{
        ros::Rate rate(1);
        bool subMapFirstFlag = true;

		while (ros::ok())
		{	
			
            if(subMapIndexQueue.empty() && !loopIndexQueue.empty())
			{
                ROS_WARN("Only LoopOptmization!!!");
                addLoopFactor();
                isam->update(gtSAMgraph, initialEstimate);
                isam->update();
                gtSAMgraph.resize(0);
                initialEstimate.clear();
                correctPoses();
            	visualizeSubMap();
			}

			if(!subMapIndexQueue.empty())
			{
				/* code */
				auto t1 = ros::Time::now();

				curSubMapId = subMapIndexQueue.front();
				ROS_WARN("\033[1;32m OptmizationThread -> curSubMapId: %d.\033[0m", curSubMapId);

				auto thisCurId = subMapInfo.find(curSubMapId);
        		if (thisCurId != subMapInfo.end())
				{
					if(curSubMapId != thisCurId->second->submap_id)
					{
						ROS_WARN("OptmizationThread -->> curSubMapId != submap_id!");
						continue;
					}
        			curSubMapPtr = thisCurId->second;

				}else{
					ROS_WARN("OptmizationThread -->> Dont find subMapInfo[%d]!", curSubMapId);
					continue;
				}

				subMapIndexQueue.pop_front();

                timeSubMapInfoStamp = curSubMapPtr->timeInfoStamp;

                updateInitialGuess();

                if(subMapFirstFlag)
                {
                	initialization();
                    publishOdometry();
                    subMapFirstFlag=false;
					preSubMapId = curSubMapId;
                    continue;
                }

				extractSubMapCloud();
				subMap2SubMapOptimization();
				saveSubMapAndFactor();
				correctPoses();
				publishOdometry();
				visualizeSubMap();

				preSubMapId = curSubMapId;

				ros::Time t2 = ros::Time::now();
				ROS_WARN("SubMap Optmization Time: %.3f", (t2 - t1).toSec());
				
			}
			rate.sleep();
			ros::spinOnce();
		}

		if(saveTrajectory == false) return;

		transformFusion();

	}




    bool updateInitialGuess()
	{
        // save current transformation before any processing
        incrementalOdometryAffineFront = trans2Affine3f(transformTobeMapped);
        
		static bool lastSubMapPreTransAvailable = false;
        static Eigen::Affine3f lastSubMapPreTransformation;

		Eigen::Affine3f transBack = pclPointToAffine3f(curSubMapPtr->submap_pose_6D_init);

		transBef = transBack;
		if (lastSubMapPreTransAvailable == false)
		{
			pcl::getTranslationAndEulerAngles(transBack, 
					transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5], 
					transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);
			lastSubMapPreTransformation = transBack;
			lastSubMapPreTransAvailable = true;
			return true;
		}
		else
		{
			Eigen::Affine3f transIncre = lastSubMapPreTransformation.inverse() * transBack;
			float x,y,z,roll,pitch,yaw;
			pcl::getTranslationAndEulerAngles(transIncre, x, y, z, roll, pitch, yaw);
			
			Eigen::Affine3f transTobe = trans2Affine3f(transformTobeMapped);
			Eigen::Affine3f transFinal = transTobe * transIncre;
			pcl::getTranslationAndEulerAngles(transFinal, 
					transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5], 
					transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);

			for(int i = 0; i < 6; ++i){
				transPredictionMapped[i] = transformTobeMapped[i];
			}
			// transPredictionMapped=trans2Affine3f(transformTobeMapped);
			lastSubMapPreTransformation = transBack;

			if(!useOdometryPitchPrediction)
			{
				ROS_INFO("transformTobeMapped[1] : %f",transformTobeMapped[1]);
				transformTobeMapped[1]=0.0;
			}
			return true;
		}

	}

    void initialization()
	{
        PointType thisPose3D;
        PointTypePose thisPose6D;

        thisPose3D.x = transformTobeMapped[3];
        thisPose3D.y = transformTobeMapped[4];
        thisPose3D.z = transformTobeMapped[5];
        thisPose3D.intensity = curSubMapId; // this can be used as index
        cloudSubMapPoses3D->push_back(thisPose3D);

        thisPose6D.x = thisPose3D.x;
        thisPose6D.y = thisPose3D.y;
        thisPose6D.z = thisPose3D.z;
        thisPose6D.intensity = thisPose3D.intensity ; // this can be used as index
        thisPose6D.roll  = transformTobeMapped[0];
        thisPose6D.pitch = transformTobeMapped[1];
        thisPose6D.yaw   = transformTobeMapped[2];
        thisPose6D.time = timeSubMapInfoStamp.toSec();
        cloudSubMapPoses6D->push_back(thisPose6D);

        SubMapPoses3D[curSubMapId]=thisPose3D;
        SubMapPoses6D[curSubMapId]=thisPose6D;


        noiseModel::Diagonal::shared_ptr priorNoise = noiseModel::Diagonal::Variances((Vector(6) << 1e-2, 1e-2, M_PI*M_PI, 1e8, 1e8, 1e8).finished()); // rad*rad, meter*meter
        gtSAMgraph.add(PriorFactor<Pose3>(curSubMapId, trans2gtsamPose(transformTobeMapped), priorNoise));
        initialEstimate.insert(curSubMapId, trans2gtsamPose(transformTobeMapped));

        for(int i=0;i<6;i++)
        {
            std::cout<<"First : transformTobeMapped["<<i<<"] : "<<transformTobeMapped[i]<<std::endl;
        }		

	}


    void extractSubMapCloud()
    {
		laserCloudCornerFromSubMap->clear();
        laserCloudSurfFromSubMap->clear();

		auto thisCurId = subMapInfo.find(preSubMapId);
		if (thisCurId != subMapInfo.end()){
			submap_Ptr preSubMap = subMapInfo[preSubMapId];

			ROS_WARN("\033[1;32m OptmizationThread -> preSubMapId: %d.\033[0m", preSubMapId);
			
			Eigen::Affine3f tran_map = pclPointToAffine3f(preSubMap->submap_pose_6D_optimized);
			this->transform_bbx(preSubMap->local_bound, preSubMap->local_cp, preSubMap->bound, preSubMap->cp, tran_map);
		
			tran_map = trans2Affine3f(transformTobeMapped);
			this->transform_bbx(curSubMapPtr->local_bound, curSubMapPtr->local_cp, curSubMapPtr->bound, curSubMapPtr->cp, tran_map);
			
			bounds_t bbx_intersection;
			get_intersection_bbx(curSubMapPtr->bound, preSubMap->bound, bbx_intersection, 10.0);

			laserCloudCornerFromSubMap->points.insert(laserCloudCornerFromSubMap->points.end(), 
													preSubMap->submap_pole->points.begin(), 
													preSubMap->submap_pole->points.end());
			laserCloudSurfFromSubMap->points.insert(laserCloudSurfFromSubMap->points.end(), 
													preSubMap->submap_ground->points.begin(), 
													preSubMap->submap_ground->points.end());
			laserCloudSurfFromSubMap->points.insert(laserCloudSurfFromSubMap->points.end(), 
													preSubMap->submap_building->points.begin(), 
													preSubMap->submap_building->points.end());
			laserCloudSurfFromSubMap->points.insert(laserCloudSurfFromSubMap->points.end(), 
													preSubMap->submap_dynamic->points.begin(), 
													preSubMap->submap_dynamic->points.end());

			*laserCloudCornerFromSubMap = *transformPointCloud(laserCloudCornerFromSubMap, &preSubMap->submap_pose_6D_optimized);
			*laserCloudSurfFromSubMap = *transformPointCloud(laserCloudSurfFromSubMap, &preSubMap->submap_pose_6D_optimized);
        	
			// ROS_WARN("\033[1;32m OptmizationThread -> laserCloudCornerFromSubMap: %d.\033[0m", laserCloudCornerFromSubMap->size());
			// ROS_WARN("\033[1;32m OptmizationThread -> laserCloudSurfFromSubMap: %d.\033[0m", laserCloudSurfFromSubMap->size());
		publishLabelCloud(&pubTestPre, laserCloudSurfFromSubMap, timeSubMapInfoStamp, odometryFrame);


			bbx_filter(laserCloudCornerFromSubMap, bbx_intersection);
        	bbx_filter(laserCloudSurfFromSubMap, bbx_intersection);

			// ROS_WARN("\033[1;32m OptmizationThread -> laserCloudCornerFromSubMap: %d.\033[0m", laserCloudCornerFromSubMap->size());
			// ROS_WARN("\033[1;32m OptmizationThread -> laserCloudSurfFromSubMap: %d.\033[0m", laserCloudSurfFromSubMap->size());
		publishLabelCloud(&pubTestPreN, laserCloudSurfFromSubMap, timeSubMapInfoStamp, odometryFrame);
		
			laserCloudCornerLast->clear();
			laserCloudSurfLast->clear();
			laserCloudCornerLastDS->clear();
			laserCloudSurfLastDS->clear();

			laserCloudCornerLast->points.insert(laserCloudCornerLast->points.end(), 
												curSubMapPtr->submap_pole->points.begin(), 
												curSubMapPtr->submap_pole->points.end());
			laserCloudSurfLast->points.insert(laserCloudSurfLast->points.end(), 
												curSubMapPtr->submap_dynamic->points.begin(), 
												curSubMapPtr->submap_dynamic->points.end());
			laserCloudSurfLast->points.insert(laserCloudSurfLast->points.end(), 
												curSubMapPtr->submap_ground->points.begin(), 
												curSubMapPtr->submap_ground->points.end());
			laserCloudSurfLast->points.insert(laserCloudSurfLast->points.end(), 
												curSubMapPtr->submap_building->points.begin(), 
												curSubMapPtr->submap_building->points.end());
			
			// ROS_WARN("\033[1;32m OptmizationThread -> laserCloudCornerLast: %d.\033[0m", laserCloudCornerLast->size());
			// ROS_WARN("\033[1;32m OptmizationThread -> laserCloudSurfLast: %d.\033[0m", laserCloudSurfLast->size());
		publishLabelCloud(&pubTestCur, laserCloudSurfLast, timeSubMapInfoStamp, odometryFrame);
		
			// tran_map = trans2Affine3f(transformTobeMapped);	
			// *laserCloudCornerLast = *transformPointCloud(laserCloudCornerLast, tran_map);
			// *laserCloudSurfLast = *transformPointCloud(laserCloudSurfLast, tran_map);

			centerpoint_t cp;
			get_bound_cpt(bbx_intersection, cp);
			tran_map = tran_map.inverse();
			this->transform_bbx(bbx_intersection, cp, bbx_intersection, cp, tran_map);

			bbx_filter(laserCloudCornerLast, bbx_intersection);
        	bbx_filter(laserCloudSurfLast, bbx_intersection);
			
			// ROS_WARN("\033[1;32m OptmizationThread -> laserCloudCornerLast: %d.\033[0m", laserCloudCornerLast->size());
			// ROS_WARN("\033[1;32m OptmizationThread -> laserCloudSurfLast: %d.\033[0m", laserCloudSurfLast->size());
		publishLabelCloud(&pubTestCurN, laserCloudSurfLast, timeSubMapInfoStamp, odometryFrame);
		
			*laserCloudCornerLastDS = *laserCloudCornerLast;
			*laserCloudSurfLastDS = *laserCloudSurfLast;
			SubMapManager::voxel_downsample_pcl(laserCloudCornerLast, laserCloudCornerLastDS, 0.2);
			SubMapManager::voxel_downsample_pcl(laserCloudSurfLast, laserCloudSurfLastDS, 0.5);
		
		}else{
			ROS_WARN("extractSubMapCloud -->> Dont find subMapInfo[%d]!", curSubMapId - 1);
			return;
		}	
        
        laserCloudCornerLastDSNum = laserCloudCornerLastDS->points.size();
        laserCloudSurfLastDSNum = laserCloudSurfLastDS->points.size();

		ROS_WARN("\033[1;32m OptmizationThread -> laserCloudCornerLastDSNum: %d.\033[0m", laserCloudCornerLastDSNum);
		ROS_WARN("\033[1;32m OptmizationThread -> laserCloudSurfLastDSNum: %d.\033[0m", laserCloudSurfLastDSNum);

	}


    void saveSubMapAndFactor()
    {
        addOdomFactor();
        //addGPSFactor();
        addLoopFactor();

        // cout << "****************************************************" << endl;
        // gtSAMgraph.print("GTSAM Graph:\n");

        // update iSAM
        isam->update(gtSAMgraph, initialEstimate);
        isam->update();

        if (aLoopIsClosed == true)
        {
            isam->update();
            isam->update();
            isam->update();
            isam->update();
            isam->update();
        }

        gtSAMgraph.resize(0);
        initialEstimate.clear();

        //save key poses
        PointType thisPose3D;
        PointTypePose thisPose6D;
        Pose3 latestEstimate;

        isamCurrentEstimate = isam->calculateEstimate();
        latestEstimate = isamCurrentEstimate.at<Pose3>(curSubMapId);
        // cout << "****************************************************" << endl;
        // isamCurrentEstimate.print("Current estimate: ");

        thisPose3D.x = latestEstimate.translation().x();
        thisPose3D.y = latestEstimate.translation().y();
        thisPose3D.z = latestEstimate.translation().z();
        thisPose3D.intensity = curSubMapId; // this can be used as index
        cloudSubMapPoses3D->push_back(thisPose3D);

        thisPose6D.x = thisPose3D.x;
        thisPose6D.y = thisPose3D.y;
        thisPose6D.z = thisPose3D.z;
        thisPose6D.intensity = thisPose3D.intensity ; // this can be used as index
        thisPose6D.roll  = latestEstimate.rotation().roll();
        thisPose6D.pitch = latestEstimate.rotation().pitch();
        thisPose6D.yaw   = latestEstimate.rotation().yaw();
        thisPose6D.time = timeSubMapInfoStamp.toSec();
        cloudSubMapPoses6D->push_back(thisPose6D);

        SubMapPoses3D[curSubMapId]=thisPose3D;
        SubMapPoses6D[curSubMapId]=thisPose6D;

		curSubMapPtr->submap_pose_6D_optimized = thisPose6D;
		curSubMapPtr->submap_pose_3D_optimized = thisPose3D;
		// std::cout << "********** saveSubMapAndFactor **********" << std::endl;
		// std::cout << "curSubMapPtr: init [" << curSubMapPtr->submap_pose_6D_init.x << ", "
        //                                 << curSubMapPtr->submap_pose_6D_init.y << ", " 
        //                                 << curSubMapPtr->submap_pose_6D_init.z << ", " 
        //                                 << curSubMapPtr->submap_pose_6D_init.roll << ", " 
        //                                 << curSubMapPtr->submap_pose_6D_init.pitch << ", " 
        //                                 << curSubMapPtr->submap_pose_6D_init.yaw << "]" 
		// 								<< std::endl;
		// std::cout << "curSubMapPtr: optimized [" << curSubMapPtr->submap_pose_6D_optimized.x << ", "
        //                                 << curSubMapPtr->submap_pose_6D_optimized.y << ", " 
        //                                 << curSubMapPtr->submap_pose_6D_optimized.z << ", " 
        //                                 << curSubMapPtr->submap_pose_6D_optimized.roll << ", " 
        //                                 << curSubMapPtr->submap_pose_6D_optimized.pitch << ", " 
        //                                 << curSubMapPtr->submap_pose_6D_optimized.yaw << "]" 
		// 								<< std::endl;
		// subMapInfo[curSubMapId]->submap_pose_6D_optimized = thisPose6D;
		// subMapInfo[curSubMapId]->submap_pose_3D_optimized = thisPose3D;
		// std::cout << "subMapInfo: init [" << subMapInfo[curSubMapId]->submap_pose_6D_init.x << ", "
        //                                 << subMapInfo[curSubMapId]->submap_pose_6D_init.y << ", " 
        //                                 << subMapInfo[curSubMapId]->submap_pose_6D_init.z << ", " 
        //                                 << subMapInfo[curSubMapId]->submap_pose_6D_init.roll << ", " 
        //                                 << subMapInfo[curSubMapId]->submap_pose_6D_init.pitch << ", " 
        //                                 << subMapInfo[curSubMapId]->submap_pose_6D_init.yaw << "]" 
		// 								<< std::endl;
		// std::cout << "subMapInfo: optimized [" << subMapInfo[curSubMapId]->submap_pose_6D_optimized.x << ", "
        //                                 << subMapInfo[curSubMapId]->submap_pose_6D_optimized.y << ", " 
        //                                 << subMapInfo[curSubMapId]->submap_pose_6D_optimized.z << ", " 
        //                                 << subMapInfo[curSubMapId]->submap_pose_6D_optimized.roll << ", " 
        //                                 << subMapInfo[curSubMapId]->submap_pose_6D_optimized.pitch << ", " 
        //                                 << subMapInfo[curSubMapId]->submap_pose_6D_optimized.yaw << "]" 
		// 								<< std::endl;


        // cout << "****************************************************" << endl;
        // cout << "Pose covariance:" << endl;
        // cout << isam->marginalCovariance(isamCurrentEstimate.size()-1) << endl << endl;
        poseCovariance = isam->marginalCovariance(curSubMapId);

        // save updated transform
        transformTobeMapped[0] = latestEstimate.rotation().roll();
        transformTobeMapped[1] = latestEstimate.rotation().pitch();
        transformTobeMapped[2] = latestEstimate.rotation().yaw();
        transformTobeMapped[3] = latestEstimate.translation().x();
        transformTobeMapped[4] = latestEstimate.translation().y();
        transformTobeMapped[5] = latestEstimate.translation().z();
    }



    void addOdomFactor()
    {
		if (subMapIndexContainer.empty()) return;

		for (int i = 0; i < (int)subMapIndexContainer.size(); ++i)
		{
			int indexFrom = subMapIndexContainer[i].first;
			int indexTo = subMapIndexContainer[i].second;
			gtsam::Pose3 poseBetween = subMapPoseQueue[i];
			gtsam::noiseModel::Diagonal::shared_ptr noiseBetween = subMapNoiseQueue[i];
			gtSAMgraph.add(BetweenFactor<Pose3>(indexFrom, indexTo, poseBetween, noiseBetween));

			gtsam::Pose3 poseTo   = trans2gtsamPose(transformTobeMapped);
			initialEstimate.insert(indexTo, poseTo);
		}

		// int indexTo = subMapIndexContainer[0].second;
		// gtsam::Pose3 poseTo   = trans2gtsamPose(transformTobeMapped);
		// initialEstimate.insert(indexTo, poseTo);

		subMapIndexContainer.clear();
		subMapPoseQueue.clear();
		subMapNoiseQueue.clear();

		// ROS_INFO("Finshed  addOdomFactor !");
    }


    void addGPSFactor()
    {
        if (gpsQueue.empty())
        {
            ROS_WARN("GPS Date is empty ! ");
            return;
        }
            

        // wait for system initialized and settles down
        if (cloudSubMapPoses3D->points.empty())
            return;
        else
        {
            if (pointDistance(cloudSubMapPoses3D->front(), cloudSubMapPoses3D->back()) < 5.0)
                return;
        }

        // pose covariance small, no need to correct
        if (poseCovariance(3,3) < poseCovThreshold && poseCovariance(4,4) < poseCovThreshold)
            return;

        // last gps position
        static PointType lastGPSPoint;

        while (!gpsQueue.empty())
        {
            if (gpsQueue.front().header.stamp.toSec() < timeSubMapInfoStamp.toSec() - 0.2)
            {
                // message too old
                gpsQueue.pop_front();
                // ROS_INFO("GPS message too old!");
            }
            else if (gpsQueue.front().header.stamp.toSec() > timeSubMapInfoStamp.toSec() + 0.2)
            {
                // ROS_INFO("GPS message too new!");
                break;
            }
            else
            {
                nav_msgs::Odometry thisGPS = gpsQueue.front();
                gpsQueue.pop_front();

                // GPS too noisy, skip
                float noise_x = thisGPS.pose.covariance[0];
                float noise_y = thisGPS.pose.covariance[7];
                float noise_z = thisGPS.pose.covariance[14];
                if (noise_x > gpsCovThreshold || noise_y > gpsCovThreshold)
                    continue;

                float gps_x = thisGPS.pose.pose.position.x;
                float gps_y = thisGPS.pose.pose.position.y;
                float gps_z = thisGPS.pose.pose.position.z;
                if (!useGpsElevation)
                {
                    gps_z = transformTobeMapped[5];
                    noise_z = 0.01;
                }

                // GPS not properly initialized (0,0,0)
                if (abs(gps_x) < 1e-6 && abs(gps_y) < 1e-6)
                    continue;

                // Add GPS every a few meters
                PointType curGPSPoint;
                curGPSPoint.x = gps_x;
                curGPSPoint.y = gps_y;
                curGPSPoint.z = gps_z;
                if (pointDistance(curGPSPoint, lastGPSPoint) < 5.0)
                    continue;
                else
                    lastGPSPoint = curGPSPoint;

                gtsam::Vector Vector3(3);
                Vector3 << max(noise_x, 1.0f), max(noise_y, 1.0f), max(noise_z, 1.0f);
                noiseModel::Diagonal::shared_ptr gps_noise = noiseModel::Diagonal::Variances(Vector3);
                gtsam::GPSFactor gps_factor((int)cloudSubMapPoses3D->back().intensity, gtsam::Point3(gps_x, gps_y, gps_z), gps_noise);
                gtSAMgraph.add(gps_factor);

                ROS_WARN("Finshed  addGPSFactor !");
                aLoopIsClosed = true;
                break;
            }
        }
    }


    void addLoopFactor()
    {
        if (loopIndexQueue.empty())
            return;

		vector<pair<int, int>> tmpLoopIndexQueue;
		vector<gtsam::Pose3> tmpLoopPoseQueue;
		vector<gtsam::noiseModel::Diagonal::shared_ptr> tmpLoopNoiseQueue;

        for (int i = 0; i < (int)loopIndexQueue.size(); ++i)
        {
            int indexFrom = loopIndexQueue[i].first;
            int indexTo = loopIndexQueue[i].second;
            gtsam::Pose3 poseBetween = loopPoseQueue[i];
            gtsam::noiseModel::Diagonal::shared_ptr noiseBetween = loopNoiseQueue[i];

			if(indexFrom > curSubMapId){
				tmpLoopIndexQueue.push_back(loopIndexQueue[i]);
				tmpLoopPoseQueue.push_back(loopPoseQueue[i]);
				tmpLoopNoiseQueue.push_back(loopNoiseQueue[i]);
			}else{
            	gtSAMgraph.add(BetweenFactor<Pose3>(indexFrom, indexTo, poseBetween, noiseBetween));
        		aLoopIsClosed = true;
			}
        }

        loopIndexQueue.clear();
        loopPoseQueue.clear();
        loopNoiseQueue.clear();

        for (int i = 0; i < (int)tmpLoopIndexQueue.size(); ++i)
        {
			loopIndexQueue.push_back(tmpLoopIndexQueue[i]);
			loopPoseQueue.push_back(tmpLoopPoseQueue[i]);
			loopNoiseQueue.push_back(tmpLoopNoiseQueue[i]);
        }		

        ROS_WARN("Finshed  addLoopFactor !");
    }

	

    void correctPoses()
    {
        if (cloudSubMapPoses3D->points.empty())
            return;

        if (aLoopIsClosed == true)
        {
            // update key poses
            int numPoses = isamCurrentEstimate.size();
            for (int i = 0; i < numPoses; ++i)
            {
				int key = (int)cloudSubMapPoses3D->points[i].intensity;
                cloudSubMapPoses3D->points[i].x = isamCurrentEstimate.at<Pose3>(key).translation().x();
                cloudSubMapPoses3D->points[i].y = isamCurrentEstimate.at<Pose3>(key).translation().y();
                cloudSubMapPoses3D->points[i].z = isamCurrentEstimate.at<Pose3>(key).translation().z();

                cloudSubMapPoses6D->points[i].x = cloudSubMapPoses3D->points[key].x;
                cloudSubMapPoses6D->points[i].y = cloudSubMapPoses3D->points[key].y;
                cloudSubMapPoses6D->points[i].z = cloudSubMapPoses3D->points[key].z;
                cloudSubMapPoses6D->points[i].roll  = isamCurrentEstimate.at<Pose3>(key).rotation().roll();
                cloudSubMapPoses6D->points[i].pitch = isamCurrentEstimate.at<Pose3>(key).rotation().pitch();
                cloudSubMapPoses6D->points[i].yaw   = isamCurrentEstimate.at<Pose3>(key).rotation().yaw();

        		SubMapPoses3D[key] = cloudSubMapPoses3D->points[i];
        		SubMapPoses6D[key] = cloudSubMapPoses6D->points[i];
            
				auto thisCurId = subMapInfo.find(key);
        		if (thisCurId != subMapInfo.end()){
        			subMapInfo[thisCurId->first]->submap_pose_6D_optimized = cloudSubMapPoses6D->points[i];
        			subMapInfo[thisCurId->first]->submap_pose_3D_optimized = cloudSubMapPoses3D->points[i];
				}else{
					ROS_WARN("correctPoses -->> Dont find subMapInfo[%d]!", key);
					continue;
				}	
			}

            ROS_INFO("Finshed  correctPoses !");
            aLoopIsClosed = false;
        }
    }





	void icpAlignment(pcl::PointCloud<PointXYZIL>::Ptr source_pc, pcl::PointCloud<PointXYZIL>::Ptr target_pc, float transformIn[])
	{

		ROS_WARN("Source PC Size: %d, Target PC Size: %d", source_pc->points.size(), target_pc->points.size());

        std::cout << "matching..." << std::flush;
        auto t1 = ros::Time::now();

        // ICP Settings
        static pcl::IterativeClosestPoint<PointXYZIL, PointXYZIL> icp;
        icp.setMaxCorrespondenceDistance(0.2);
        icp.setMaximumIterations(50);
        icp.setTransformationEpsilon(1e-5);
        icp.setEuclideanFitnessEpsilon(1e-5);
        icp.setRANSACIterations(0);

		Eigen::Affine3f initLidarFrame = trans2Affine3f(transformIn);
		// Align clouds
		pcl::PointCloud<PointXYZIL>::Ptr tmpCloud( new pcl::PointCloud<PointXYZIL>());
		*tmpCloud = *transformPointCloud(source_pc, initLidarFrame);
		icp.setInputSource(tmpCloud);

		icp.setInputTarget(target_pc);
		pcl::PointCloud<PointXYZIL>::Ptr unused_result( new pcl::PointCloud<PointXYZIL>());
		icp.align(*unused_result);

		double score = icp.getFitnessScore();
		
		Eigen::Affine3f correctionLidarFrame;
		correctionLidarFrame = icp.getFinalTransformation();
        
		auto t2 = ros::Time::now();
        std::cout << " done" << std::endl;
        std::cout << "score: " << score << "    time: " << (t2 - t1).toSec() << "[sec]" << std::endl;
        
		float X, Y, Z, ROLL, PITCH, YAW;
        Eigen::Affine3f tCorrect = correctionLidarFrame * initLidarFrame;  // pre-multiplying -> successive rotation about a fixed frame
        pcl::getTranslationAndEulerAngles(tCorrect, X, Y, Z, ROLL, PITCH, YAW);
        
		std::cout << "TransformIn: [" << ROLL << ", " << PITCH << ", " << YAW << ", " 
                                      << X << ", " << Y << ", " << Z << "]" << std::endl;

		if(score <= 3.0) 
		{
			transformTobeMapped[2] = YAW;
			transformTobeMapped[3] = X;
			transformTobeMapped[4] = Y;
		}

	}

    void scan2SubMapOptimizationICP()
	{
		pcl::PointCloud<PointXYZIL>::Ptr sourcePC( new pcl::PointCloud<PointXYZIL>());
		pcl::PointCloud<PointXYZIL>::Ptr targetPC( new pcl::PointCloud<PointXYZIL>());

		*targetPC += *laserCloudSurfFromSubMap;
		*targetPC += *laserCloudCornerFromSubMap;
		
		*targetPC += *laserCloudSurfLast;
		*sourcePC += *laserCloudCornerLast;

		icpAlignment(sourcePC, targetPC, transformTobeMapped);
	}



    void subMap2SubMapOptimization()
    {
        if (laserCloudCornerLastDSNum > edgeFeatureMinValidNum && laserCloudSurfLastDSNum > surfFeatureMinValidNum)
        {

			int subMapCloudCornerLastDSNum = laserCloudCornerFromSubMap->points.size();
			int subMapCloudSurfLastDSNum = laserCloudSurfFromSubMap->points.size();

			ROS_WARN("\033[1;32m OptmizationThread -> subMapCloudCornerLastDSNum: %d.\033[0m", subMapCloudCornerLastDSNum);
			ROS_WARN("\033[1;32m OptmizationThread -> subMapCloudSurfLastDSNum: %d.\033[0m", subMapCloudSurfLastDSNum);

            kdtreeCornerFromSubMap->setInputCloud(laserCloudCornerFromSubMap);
            kdtreeSurfFromSubMap->setInputCloud(laserCloudSurfFromSubMap);

			int iterCount = 0;
            for (; iterCount < 30; iterCount++)   //30
            {
                laserCloudOri->clear();
                coeffSel->clear();

				if(subMapCloudCornerLastDSNum>0)
                	cornerOptimization();
				
				if(subMapCloudSurfLastDSNum>0)
                	surfOptimization();

                combineOptimizationCoeffs();

                if (LMOptimization(iterCount) == true)
                    break;          
            }

			ROS_WARN("\033[1;32m OptmizationThread -> iterCount: %d, deltaR: %f, deltaT: %f.\033[0m", iterCount, deltaR, deltaT);
			transformUpdate();

			gtsam::Pose3 poseFrom = pclPointTogtsamPose3(SubMapPoses6D[preSubMapId]);
			gtsam::Pose3 poseTo   = trans2gtsamPose(transformTobeMapped);
			noiseModel::Diagonal::shared_ptr odometryNoise = noiseModel::Diagonal::Variances((Vector(6) << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4).finished());

			// std::cout<<"*********************************************************"<<std::endl;
			// for(int i=0;i<6;i++)
			// {
			//     std::cout<<"After Optimization : transformTobeMapped["<<i<<"] : "<<transformTobeMapped[i]<<std::endl;
			// }
			// std::cout<<"*********************************************************"<<std::endl;

			subMapIndexContainerAll.push_back(make_pair(preSubMapId, curSubMapId));
			subMapIndexContainer.push_back(make_pair(preSubMapId, curSubMapId));
			subMapPoseQueue.push_back(poseFrom.between(poseTo));
			subMapNoiseQueue.push_back(odometryNoise);


        } else {
            ROS_WARN("Not enough features! Only %d edge and %d planar features available.", laserCloudCornerLastDSNum, laserCloudSurfLastDSNum);
        }
    }

    void updatePointAssociateToSubMap()
    {
        transPointAssociateToSubMap = trans2Affine3f(transformTobeMapped);
    }

    void pointAssociateToSubMap(PointXYZIL const * const pi, PointXYZIL * const po)
    {
        po->x = transPointAssociateToSubMap(0,0) * pi->x + transPointAssociateToSubMap(0,1) * pi->y + transPointAssociateToSubMap(0,2) * pi->z + transPointAssociateToSubMap(0,3);
        po->y = transPointAssociateToSubMap(1,0) * pi->x + transPointAssociateToSubMap(1,1) * pi->y + transPointAssociateToSubMap(1,2) * pi->z + transPointAssociateToSubMap(1,3);
        po->z = transPointAssociateToSubMap(2,0) * pi->x + transPointAssociateToSubMap(2,1) * pi->y + transPointAssociateToSubMap(2,2) * pi->z + transPointAssociateToSubMap(2,3);
        po->intensity = pi->intensity;
        po->label = pi->label;
    }

    void cornerOptimization()
    {
        updatePointAssociateToSubMap();

		int numSearch = 0;

        // #pragma omp for
        #pragma omp parallel for num_threads(numberOfCores)
        for (int i = 0; i < laserCloudCornerLastDSNum; i++)
        {
            PointXYZIL pointOri, pointSel, coeff;
            pointOri = laserCloudCornerLastDS->points[i];
            pointAssociateToSubMap(&pointOri, &pointSel);

			std::vector<int> pointSearchInd;
            std::vector<float> pointSearchSqDis;
            kdtreeCornerFromSubMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);


			// std::vector<int> pointSearchIndPre;
            // std::vector<float> pointSearchSqDisPre;
            // kdtreeCornerFromSubMap->nearestKSearch(pointSel, 10, pointSearchIndPre, pointSearchSqDisPre);

            // std::vector<int> pointSearchInd;
            // std::vector<float> pointSearchSqDis;

            // std::vector<int> pointSearchIndNo;
            // std::vector<float> pointSearchSqDisNo;
				
			// int labelOri = laserCloudCornerLastDS->points[i].label;
			// for(int id = 0; id < pointSearchIndPre.size(); ++id)
			// {
			// 	int labelCur = laserCloudCornerFromSubMap->points[pointSearchIndPre[id]].label;
			// 	if(pointSearchInd.size() <= 5 && labelOri == labelCur && pointSearchSqDisPre[id] < 2.0)
			// 	{
			// 		pointSearchInd.push_back(pointSearchIndPre[id]);
			// 		pointSearchSqDis.push_back(pointSearchSqDisPre[id]);
			// 	}else{
			// 		pointSearchIndNo.push_back(pointSearchIndPre[id]);
			// 		pointSearchSqDisNo.push_back(pointSearchSqDisPre[id]);
			// 	}
			// }
			// int curIndSize = 5 - pointSearchInd.size();
			// for(int id = 0; id < curIndSize; ++id){
			// 	pointSearchInd.push_back(pointSearchIndNo[id]);
			// 	pointSearchSqDis.push_back(pointSearchSqDisNo[id]);
			// }

            cv::Mat matA1(3, 3, CV_32F, cv::Scalar::all(0));
            cv::Mat matD1(1, 3, CV_32F, cv::Scalar::all(0));
            cv::Mat matV1(3, 3, CV_32F, cv::Scalar::all(0));

            // if (pointSearchSqDis.size() == 5 && pointSearchSqDis[4] < 1.0) 
            if (pointSearchSqDis.size() == 5 && pointSearchSqDis[4] < 2.0) 
			{
                float cx = 0, cy = 0, cz = 0;
                for (int j = 0; j < 5; j++) {
                    cx += laserCloudCornerFromSubMap->points[pointSearchInd[j]].x;
                    cy += laserCloudCornerFromSubMap->points[pointSearchInd[j]].y;
                    cz += laserCloudCornerFromSubMap->points[pointSearchInd[j]].z;
                }
                cx /= 5; cy /= 5;  cz /= 5;

                float a11 = 0, a12 = 0, a13 = 0, a22 = 0, a23 = 0, a33 = 0;
                for (int j = 0; j < 5; j++) {
                    float ax=0, ay=0, az=0;

                    ax = laserCloudCornerFromSubMap->points[pointSearchInd[j]].x - cx;
                    ay = laserCloudCornerFromSubMap->points[pointSearchInd[j]].y - cy;
                    az = laserCloudCornerFromSubMap->points[pointSearchInd[j]].z - cz;

                    a11 += ax * ax; a12 += ax * ay; a13 += ax * az;
                    a22 += ay * ay; a23 += ay * az;
                    a33 += az * az;
                }
                a11 /= 5; a12 /= 5; a13 /= 5; a22 /= 5; a23 /= 5; a33 /= 5;

                matA1.at<float>(0, 0) = a11; matA1.at<float>(0, 1) = a12; matA1.at<float>(0, 2) = a13;
                matA1.at<float>(1, 0) = a12; matA1.at<float>(1, 1) = a22; matA1.at<float>(1, 2) = a23;
                matA1.at<float>(2, 0) = a13; matA1.at<float>(2, 1) = a23; matA1.at<float>(2, 2) = a33;

                cv::eigen(matA1, matD1, matV1);

                if (matD1.at<float>(0, 0) > 3 * matD1.at<float>(0, 1)) {

                    float x0 = pointSel.x;
                    float y0 = pointSel.y;
                    float z0 = pointSel.z;
                    float x1 = cx + 0.1 * matV1.at<float>(0, 0);
                    float y1 = cy + 0.1 * matV1.at<float>(0, 1);
                    float z1 = cz + 0.1 * matV1.at<float>(0, 2);
                    float x2 = cx - 0.1 * matV1.at<float>(0, 0);
                    float y2 = cy - 0.1 * matV1.at<float>(0, 1);
                    float z2 = cz - 0.1 * matV1.at<float>(0, 2);

                    float a012 = sqrt(((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) * ((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) 
                                    + ((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1)) * ((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1)) 
                                    + ((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1)) * ((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1)));

                    float l12 = sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2));

                    float la = ((y1 - y2)*((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) 
                              + (z1 - z2)*((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1))) / a012 / l12;

                    float lb = -((x1 - x2)*((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) 
                               - (z1 - z2)*((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1))) / a012 / l12;

                    float lc = -((x1 - x2)*((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1)) 
                               + (y1 - y2)*((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1))) / a012 / l12;

                    float ld2 = a012 / l12;

                    float s = 1 - 0.9 * fabs(ld2);

					float w = 2.0 - LabelSorce[laserCloudCornerLastDS->points[i].label];

                    coeff.x = w * s * la;
                    coeff.y = w * s * lb;
                    coeff.z = w * s * lc;
                    coeff.intensity = w * s * ld2;

                    // coeff.x = s * la;
                    // coeff.y = s * lb;
                    // coeff.z = s * lc;
                    // coeff.intensity = w * s * ld2;

                    if (s > 0.1) {
                        laserCloudOriCornerVec[i] = pointOri;
                        coeffSelCornerVec[i] = coeff;
                        laserCloudOriCornerFlag[i] = true;

						numSearch++;
                    }
                }
            }
        }

		// ROS_WARN("Corner numSearch: [%d / %d]", numSearch, laserCloudCornerLastDSNum);

    }



    void surfOptimization()
    {
        updatePointAssociateToSubMap();

		int numSearch = 0;

        // #pragma omp for
        #pragma omp parallel for num_threads(numberOfCores)
        for (int i = 0; i < laserCloudSurfLastDSNum; i++)
        {
            PointXYZIL pointOri, pointSel, coeff;
            pointOri = laserCloudSurfLastDS->points[i];
            pointAssociateToSubMap(&pointOri, &pointSel);
			
			std::vector<int> pointSearchInd;
            std::vector<float> pointSearchSqDis;
            kdtreeSurfFromSubMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);
			
			
			// std::vector<int> pointSearchIndPre;
            // std::vector<float> pointSearchSqDisPre;
            // kdtreeSurfFromSubMap->nearestKSearch(pointSel, 10, pointSearchIndPre, pointSearchSqDisPre);

            // std::vector<int> pointSearchInd;
            // std::vector<float> pointSearchSqDis;
				
            // std::vector<int> pointSearchIndNo;
            // std::vector<float> pointSearchSqDisNo;
				
			// int labelOri = laserCloudSurfLastDS->points[i].label;
			// for(int id = 0; id < pointSearchIndPre.size(); ++id)
			// {
			// 	int labelCur = laserCloudSurfFromSubMap->points[pointSearchIndPre[id]].label;
			// 	if(pointSearchInd.size() <= 5 && labelOri == labelCur && pointSearchSqDisPre[id] < 2.0)
			// 	{
			// 		pointSearchInd.push_back(pointSearchIndPre[id]);
			// 		pointSearchSqDis.push_back(pointSearchSqDisPre[id]);
			// 	}else{
			// 		pointSearchIndNo.push_back(pointSearchIndPre[id]);
			// 		pointSearchSqDisNo.push_back(pointSearchSqDisPre[id]);

			// 	}
			// }

			// int curIndSize = 5 - pointSearchInd.size();
			// for(int id = 0; id < curIndSize; ++id){
			// 	pointSearchInd.push_back(pointSearchIndNo[id]);
			// 	pointSearchSqDis.push_back(pointSearchSqDisNo[id]);
			// }


            Eigen::Matrix<float, 5, 3> matA0;
            Eigen::Matrix<float, 5, 1> matB0;
            Eigen::Vector3f matX0;

            matA0.setZero();
            matB0.fill(-1);
            matX0.setZero();

            // if (pointSearchSqDis.size() == 5 && pointSearchSqDis[4] < 1.0) 
            if (pointSearchSqDis.size() == 5 && pointSearchSqDis[4] < 2.0) 
			{

                for (int j = 0; j < 5; j++) {
                        matA0(j, 0) = laserCloudSurfFromSubMap->points[pointSearchInd[j]].x;
                        matA0(j, 1) = laserCloudSurfFromSubMap->points[pointSearchInd[j]].y;
                        matA0(j, 2) = laserCloudSurfFromSubMap->points[pointSearchInd[j]].z;
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
                    if (fabs(pa * laserCloudSurfFromSubMap->points[pointSearchInd[j]].x +
                            pb * laserCloudSurfFromSubMap->points[pointSearchInd[j]].y +
                            pc * laserCloudSurfFromSubMap->points[pointSearchInd[j]].z + pd) > 0.2) {
                        planeValid = false;
                        break;
                    }    
                }

                if (planeValid) {
                    float pd2 = pa * pointSel.x + pb * pointSel.y + pc * pointSel.z + pd;

                    float s = 1 - 0.9 * fabs(pd2) / sqrt(sqrt(pointSel.x * pointSel.x
                            + pointSel.y * pointSel.y + pointSel.z * pointSel.z));

					float w = 2.0 - LabelSorce[laserCloudSurfLastDS->points[i].label];

                    coeff.x = w * s * pa;
                    coeff.y = w * s * pb;
                    coeff.z = w * s * pc;
                    coeff.intensity = w * s * pd2;
                    
					// coeff.x = s * pa;
                    // coeff.y = s * pb;
                    // coeff.z = s * pc;
                    // coeff.intensity = w * s * pd2;

                    if (s > 0.1) {
                        laserCloudOriSurfVec[i] = pointOri;
                        coeffSelSurfVec[i] = coeff;
                        laserCloudOriSurfFlag[i] = true;
						
						numSearch++;
                    }
                }
            }
        }

		// ROS_WARN("Surf numSearch: [%d / %d]", numSearch, laserCloudSurfLastDSNum);

    }

    void combineOptimizationCoeffs()
    {
        // combine corner coeffs
        for (int i = 0; i < laserCloudCornerLastDSNum; ++i){
            if (laserCloudOriCornerFlag[i] == true){
                laserCloudOri->push_back(laserCloudOriCornerVec[i]);
                coeffSel->push_back(coeffSelCornerVec[i]);
            }
        }
        // combine surf coeffs
        for (int i = 0; i < laserCloudSurfLastDSNum; ++i){
            if (laserCloudOriSurfFlag[i] == true){
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
        // This optimization is from the original loam_velodyne by Ji Zhang, need to cope with coordinate transformation
        // lidar <- camera      ---     camera <- lidar
        // x = z                ---     x = y
        // y = x                ---     y = z
        // z = y                ---     z = x
        // roll = yaw           ---     roll = pitch
        // pitch = roll         ---     pitch = yaw
        // yaw = pitch          ---     yaw = roll

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
        for (int i = 0; i < laserCloudSelNum; i++) {
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
            float arx = (crx*sry*srz*pointOri.x + crx*crz*sry*pointOri.y - srx*sry*pointOri.z) * coeff.x
                      + (-srx*srz*pointOri.x - crz*srx*pointOri.y - crx*pointOri.z) * coeff.y
                      + (crx*cry*srz*pointOri.x + crx*cry*crz*pointOri.y - cry*srx*pointOri.z) * coeff.z;

            float ary = ((cry*srx*srz - crz*sry)*pointOri.x 
                      + (sry*srz + cry*crz*srx)*pointOri.y + crx*cry*pointOri.z) * coeff.x
                      + ((-cry*crz - srx*sry*srz)*pointOri.x 
                      + (cry*srz - crz*srx*sry)*pointOri.y - crx*sry*pointOri.z) * coeff.z;

            float arz = ((crz*srx*sry - cry*srz)*pointOri.x + (-cry*crz-srx*sry*srz)*pointOri.y)*coeff.x
                      + (crx*crz*pointOri.x - crx*srz*pointOri.y) * coeff.y
                      + ((sry*srz + cry*crz*srx)*pointOri.x + (crz*sry-cry*srx*srz)*pointOri.y)*coeff.z;
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
            for (int i = 5; i >= 0; i--) {
                if (matE.at<float>(0, i) < eignThre[i]) {
                    for (int j = 0; j < 6; j++) {
                        matV2.at<float>(i, j) = 0;
                    }
                    isDegenerate = true;
                } else {
                    break;
                }
            }
            matP = matV.inv() * matV2;
        }

        if (isDegenerate) {
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

        deltaR = sqrt(pow(pcl::rad2deg(matX.at<float>(0, 0)), 2) +
                      pow(pcl::rad2deg(matX.at<float>(1, 0)), 2) +
                      pow(pcl::rad2deg(matX.at<float>(2, 0)), 2));
        deltaT = sqrt(pow(matX.at<float>(3, 0) * 100, 2) +
                      pow(matX.at<float>(4, 0) * 100, 2) +
                      pow(matX.at<float>(5, 0) * 100, 2));

        if (deltaR < 0.002 && deltaT < 0.02) {
            return true; // converged
        }
        return false; // keep optimizing
    }

    void transformUpdate()
    {
        transformTobeMapped[0] = constraintTransformation(transformTobeMapped[0], rotation_tollerance);
        transformTobeMapped[1] = constraintTransformation(transformTobeMapped[1], rotation_tollerance);
        transformTobeMapped[5] = constraintTransformation(transformTobeMapped[5], z_tollerance);

        incrementalOdometryAffineBack = trans2Affine3f(transformTobeMapped);

		transBef2Aft = incrementalOdometryAffineBack * transBef.inverse();
    }





    void publishOdometry()
    {
        nav_msgs::Odometry laserOdometryROS;

        laserOdometryROS.header.stamp = timeSubMapInfoStamp;
        laserOdometryROS.header.frame_id = odometryFrame;//
        laserOdometryROS.child_frame_id = "odom_mapping";
        laserOdometryROS.pose.pose.position.x = transformTobeMapped[3];
        laserOdometryROS.pose.pose.position.y = transformTobeMapped[4];
        laserOdometryROS.pose.pose.position.z = transformTobeMapped[5];
        laserOdometryROS.pose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(
				transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);

        pubSubMapOdometryGlobal.publish(laserOdometryROS);
    }
    
	void visualizeSubMap()
    {
        visualization_msgs::MarkerArray markerArray;

        visualization_msgs::Marker markerNodeId;
        markerNodeId.header.frame_id = odometryFrame;
        markerNodeId.header.stamp = ros::Time::now();
        markerNodeId.action = visualization_msgs::Marker::ADD;
        markerNodeId.type =  visualization_msgs::Marker::TEXT_VIEW_FACING;
        markerNodeId.ns = "submap_nodes_id";
        markerNodeId.id = 0;
        markerNodeId.pose.orientation.w = 1;
        markerNodeId.scale.z = 0.5; 
        markerNodeId.color.r = 0; markerNodeId.color.g = 255; markerNodeId.color.b = 0;
        markerNodeId.color.a = 1;
        // nodes
        visualization_msgs::Marker markerNode;
        markerNode.header.frame_id = odometryFrame;
        markerNode.header.stamp = timeSubMapInfoStamp;
        markerNode.action = visualization_msgs::Marker::ADD;
        markerNode.type = visualization_msgs::Marker::SPHERE_LIST;
        markerNode.ns = "submap_nodes";
        markerNode.id = 0;
        markerNode.pose.orientation.w = 1;
        markerNode.scale.x = 0.5; markerNode.scale.y = 0.5; markerNode.scale.z = 0.5; 
        markerNode.color.r = 0; markerNode.color.g = 0.8; markerNode.color.b = 1;
        markerNode.color.a = 1;
        // edges
        visualization_msgs::Marker markerEdge;
        markerEdge.header.frame_id = odometryFrame;
        markerEdge.header.stamp = timeSubMapInfoStamp;
        markerEdge.action = visualization_msgs::Marker::ADD;
        markerEdge.type = visualization_msgs::Marker::LINE_LIST;
        markerEdge.ns = "submap_edges";
        markerEdge.id = 1;
        markerEdge.pose.orientation.w = 1;
        markerEdge.scale.x = 0.4; markerEdge.scale.y = 0.4; markerEdge.scale.z = 0.4;
        markerEdge.color.r = 0.5; markerEdge.color.g = 0.8; markerEdge.color.b = 0;
        markerEdge.color.a = 1;


        for (int i = 0; i < (int)subMapIndexContainerAll.size(); ++i)
        {
            int key_cur = subMapIndexContainerAll[i].first;
            int key_pre = subMapIndexContainerAll[i].second;

            geometry_msgs::Pose pose;
            pose.position.x =  cloudSubMapPoses6D->points[key_cur].x;
            pose.position.y =  cloudSubMapPoses6D->points[key_cur].y;
            pose.position.z =  cloudSubMapPoses6D->points[key_cur].z+0.15;
            int k = key_cur;
            ostringstream str;
            str << k;
            markerNodeId.id = k;
            markerNodeId.text = str.str();
            markerNodeId.pose = pose;
            markerArray.markers.push_back(markerNodeId);

            geometry_msgs::Point p;
            
            p.x = cloudSubMapPoses6D->points[key_cur].x;
            p.y = cloudSubMapPoses6D->points[key_cur].y;
            p.z = cloudSubMapPoses6D->points[key_cur].z;
            markerNode.points.push_back(p);
            markerEdge.points.push_back(p);
            p.x = cloudSubMapPoses6D->points[key_pre].x;
            p.y = cloudSubMapPoses6D->points[key_pre].y;
            p.z = cloudSubMapPoses6D->points[key_pre].z;
            markerNode.points.push_back(p);
            markerEdge.points.push_back(p);
        }

        // markerArray.markers.push_back(markerNodeId);
        markerArray.markers.push_back(markerNode);
        markerArray.markers.push_back(markerEdge);
        pubSubMapConstraintEdge.publish(markerArray);
        // ROS_INFO("Finshed  visualizeSubMap !");
    }


	void transformFusion()
	{
		//@TODO	saveTrajectory
		cout << "****************************************************" << endl;
        cout << "Saving Trajectory to files ..." << endl;

		int init_flag = true; 
		Eigen::Matrix4f H;  
		Eigen::Matrix4f H_init; 
		Eigen::Matrix4f H_rot; 

		int pose_size = 0;

		// std::string path = "/home/wqz/paperTest/my_epsc_trajectory_noloop/kitti_test.txt"; //ADD
		std::string path = RESULT_PATH;
		cout << "RESULT_PATH: " << RESULT_PATH << endl;


		for(auto it = keyFrame2SubMapPose.begin(); it != keyFrame2SubMapPose.end(); it++)
		{
			Eigen::Affine3f curSubMapAffine;
			auto thisSubMap = subMapInfo.find(it->first);
			if(thisSubMap != subMapInfo.end()){
				curSubMapAffine = pclPointToAffine3f(thisSubMap->second->submap_pose_6D_optimized);
				std::cout << "Save Submap [" << it->first <<"] Trajectory ..." << std::endl;
			}else{
				std::cout << "Dont find Submap [" << it->first <<"] Trajectory !" << std::endl;
				continue;
			}

			for(int i = 0; i < it->second.size(); i++)
			{
				pose_size++;

				Eigen::Affine3f R = curSubMapAffine * it->second[i];

				if (init_flag == true)	
				{
					H_init << R(0, 0), R(0, 1), R(0, 2), R(0, 3),
							  R(1, 0), R(1, 1), R(1, 2), R(1, 3),
							  R(2, 0), R(2, 1), R(2, 2), R(2, 3),
							  0,       0,       0,       1;  
					init_flag = false;
				}

				// H_rot << -1, 0, 0, 0,
				//           0,-1, 0, 0,
				//           0, 0, 1, 0,	
				//           0, 0, 0, 1; 

				// H_rot <<	0, 0,-1, 0,
				//          0,-1, 0, 0,
				//          1, 0, 0, 0,	
				//          0, 0, 0, 1; 

				// H_rot << 4.276802385584e-04, -9.999672484946e-01, -8.084491683471e-03, -1.198459927713e-02,
				//         -7.210626507497e-03,  8.081198471645e-03, -9.999413164504e-01, -5.403984729748e-02,
				//          9.999738645903e-01,  4.859485810390e-04, -7.206933692422e-03, -2.921968648686e-01,
				//          0,0,0,1; 

				// H_rot << 4.276802385584e-04, -9.999672484946e-01, -8.084491683471e-03, 0,
				//         -7.210626507497e-03,  8.081198471645e-03, -9.999413164504e-01, 0,
				//          9.999738645903e-01,  4.859485810390e-04, -7.206933692422e-03, 0,
				//          0,0,0,1; 

				H_rot << 1, 0, 0, 0,
						 0, 1, 0, 0,
						 0, 0, 1, 0,
						 0, 0, 0, 1; 
					
				H << R(0, 0), R(0, 1), R(0, 2), R(0, 3),
					 R(1, 0), R(1, 1), R(1, 2), R(1, 3),
					 R(2, 0), R(2, 1), R(2, 2), R(2, 3),
					 0,       0,       0,       1;  

				H = H_rot * H_init.inverse() * H; //to get H12 = H10*H02 , 180 rot according to z axis

				std::ofstream foutC(path, std::ios::app);

				foutC.setf(std::ios::scientific, std::ios::floatfield);
				foutC.precision(6);
			
				for (int i = 0; i < 3; ++i)	
				{	 
					for (int j = 0; j < 4; ++j)
					{
						if(i == 2 && j == 3)
							foutC << H.row(i)[j] << endl ;	
						else
							foutC << H.row(i)[j] << " " ;
					}
				}
				foutC.close();
			}

		}

        cout << "****************************************************" << endl;
        cout << "Saving " << pose_size << " Trajectory to files completed" << endl;
		
	}	


};

int main(int argc, char **argv) 
{
    ros::init(argc, argv, "lis_slam");

    SubMapOdometryNode SOD;
    std::thread make_submap_process(&SubMapOdometryNode::makeSubMapThread, &SOD);
    std::thread loop_closure_process(&SubMapOdometryNode::loopClosureThread, &SOD);
    

    SubMapOptmizationNode SOP;
    std::thread visualize_map_process(&SubMapOptmizationNode::visualizeGlobalMapThread, &SOP);
    std::thread submap_optmization_process(&SubMapOptmizationNode::subMapOptmizationThread, &SOP);
    
    ROS_WARN("\033[1;32m----> SubMap Optmization Node Started.\033[0m");

	ros::MultiThreadedSpinner spinner(4);
	spinner.spin();
    // ros::spin();

    make_submap_process.join();
    loop_closure_process.join();
    
    visualize_map_process.join();
    submap_optmization_process.join();

    return 0;
}