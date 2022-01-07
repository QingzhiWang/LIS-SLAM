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

#include "epscGeneration.h"
#include "subMap.h"
#include "utility.h"
#include "common.h"

#define USING_SINGLE_TARGET false
#define USING_SUBMAP_TARGET false
#define USING_SLIDING_TARGET true
#define USING_MULTI_KEYFRAME_TARGET false

#define USING_SEMANTIC_FEATURE true
#define USING_LOAM_FEATURE false



using namespace gtsam;

using symbol_shorthand::B;  // Bias  (ax,ay,az,gx,gy,gz)
using symbol_shorthand::G;  // GPS pose
using symbol_shorthand::V;  // Vel   (xdot,ydot,zdot)
using symbol_shorthand::X;  // Pose3 (x,y,z,r,p,y)


std::mutex subMapMtx;
std::deque<int> subMapIndexQueue;
map<int, submap_Ptr> subMapInfo;

map<int, int> loopIndexContainer;  // from new to old
vector<pair<int, int>> loopIndexVec;
vector<gtsam::Pose3> loopPoseVec;
vector<gtsam::noiseModel::Diagonal::shared_ptr> loopNoiseVec;


gtsam::Pose3 pclPointTogtsamPose3(PointTypePose thisPoint) {
    return gtsam::Pose3(gtsam::Rot3::RzRyRx(double(thisPoint.roll), double(thisPoint.pitch), double(thisPoint.yaw)),
                        gtsam::Point3(double(thisPoint.x), double(thisPoint.y), double(thisPoint.z)));
}

gtsam::Pose3 trans2gtsamPose(float transformIn[]) {
    return gtsam::Pose3(gtsam::Rot3::RzRyRx(transformIn[0], transformIn[1], transformIn[2]),
                        gtsam::Point3(transformIn[3], transformIn[4], transformIn[5]));
}


class SubMapOdometryNode : public SubMapManager<PointXYZIL>
{
 public:
    ros::Subscriber subCloud;
    ros::Subscriber subIMU;
	ros::Subscriber subOdom

	ros::Publisher pubCloudRegisteredRaw;
    
    ros::Publisher pubCloudCurSubMap;
    ros::Publisher pubSubMapId;

    ros::Publisher pubKeyFrameOdometryGlobal;
    ros::Publisher pubKeyFrameOdometryIncremental;
  
    ros::Publisher pubKeyFramePoseGlobal;
    ros::Publisher pubKeyFramePath;

    ros::Publisher pubLoopConstraintEdge;
    ros::Publisher pubSCDe;
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
    std::deque<keyframe_Ptr> keyFrameQueue;
    map<int, keyframe_Ptr> keyFrameInfo;

    keyframe_Ptr currentKeyFrame = keyframe_Ptr(new keyframe_t);
    submap_Ptr currentSubMap = submap_Ptr(new submap_t);
    localMap_Ptr localMap = localMap_Ptr(new localMap_t);

    pcl::PointCloud<PointTypePose>::Ptr subMapPose6D; 
    pcl::PointCloud<PointType>::Ptr subMapPose3D;
    
    map<int,PointType> subMapPosesIndex3D;
    map<int,PointTypePose> subMapPosesIndex6D;
    
    pcl::PointCloud<PointType>::Ptr keyFramePoses3D;
    pcl::PointCloud<PointTypePose>::Ptr keyFramePoses6D;
    
    map<int,PointType> keyFramePosesIndex3D;
    map<int,PointTypePose> keyFramePosesIndex6D;

    pcl::PointCloud<PointXYZIL>::Ptr laserCloudCornerLast; // corner feature set from odoOptimization
    pcl::PointCloud<PointXYZIL>::Ptr laserCloudSurfLast; // surf feature set from odoOptimization
    pcl::PointCloud<PointXYZIL>::Ptr laserCloudCornerLastDS; // downsampled corner featuer set from odoOptimization
    pcl::PointCloud<PointXYZIL>::Ptr laserCloudSurfLastDS; // downsampled surf featuer set from odoOptimization

    pcl::PointCloud<PointXYZIL>::Ptr laserCloudCornerFromSubMap;
    pcl::PointCloud<PointXYZIL>::Ptr laserCloudSurfFromSubMap;

	std::vector<pcl::PointCloud<PointXYZIL>::Ptr> laserCloudSurfVec;
	std::vector<pcl::PointCloud<PointXYZIL>::Ptr> laserCloudCornerVec;

    pcl::KdTreeFLANN<PointTypePose>::Ptr kdtreeFromKeyPoses6D;
    pcl::KdTreeFLANN<PointTypePose>::Ptr kdtreeFromsubMapPose6D;

    int laserCloudCornerLastDSNum = 0;
    int laserCloudSurfLastDSNum = 0;

    ros::Time timeLaserInfoStamp;
    ros::Time timeSubMapInfoStamp;
    double timeLaserInfoCur = -1;

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

	const double delta_t = 0;

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

	deque<nav_msgs::Odometry> imuOdomQueue;
	
	ros::Publisher pubKeyframeIMUOdometry;
	ros::Publisher pubImuOdometry;

	ros::Publisher pubLidarPath;
	ros::Publisher pubLidarOdometry;
	ros::Publisher pubLidarIMUOdometry;
	
	
	// ---- test publisher ----  
    ros::Publisher pubTest1;
    ros::Publisher pubTest2;


    
    SubMapOdometryNode() 
    {
        subCloud = nh.subscribe<lis_slam::semantic_info>( "lis_slam/semantic_fusion/semantic_info", 200, &SubMapOdometryNode::semanticInfoHandler, this, ros::TransportHints().tcpNoDelay());
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
        pubSCDe = nh.advertise<sensor_msgs::Image>("global_descriptor", 100);
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
        

        allocateMemory();
    }

    void allocateMemory() 
    {
        subMapPose6D.reset(new pcl::PointCloud<PointTypePose>()); 
        subMapPose3D.reset(new pcl::PointCloud<PointType>());

        keyFramePoses3D.reset(new pcl::PointCloud<PointType>());
        keyFramePoses6D.reset(new pcl::PointCloud<PointTypePose>());

        laserCloudCornerLast.reset(new pcl::PointCloud<PointXYZIL>()); 
        laserCloudSurfLast.reset(new pcl::PointCloud<PointXYZIL>()); 
        laserCloudCornerLastDS.reset(new pcl::PointCloud<PointXYZIL>()); 
        laserCloudSurfLastDS.reset(new pcl::PointCloud<PointXYZIL>()); 

        laserCloudCornerFromSubMap.reset(new pcl::PointCloud<PointXYZIL>());
        laserCloudSurfFromSubMap.reset(new pcl::PointCloud<PointXYZIL>());

        kdtreeFromKeyPoses6D.reset(new pcl::KdTreeFLANN<PointTypePose>());
        kdtreeFromsubMapPose6D.reset(new pcl::KdTreeFLANN<PointTypePose>());

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
			(gtsam::Vector(6) << 1e-1, 1e-1, 1e-2, 1e-3, 1e-3, 1e-3).finished());  // rad,rad,rad,m, m, m
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
		double dt = (lastImuT_imu < 0) ? (1.0 / 500.0) : (imuTime - lastImuT_imu);
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

        float x, y, z, roll, pitch, yaw;
        pcl::getTranslationAndEulerAngles(lidarOdomAffine, x, y, z, roll, pitch, yaw);

        // publish latest odometry
        nav_msgs::Odometry laserOdometry = thisOdom;
        laserOdometry.pose.pose.position.x = x;
        laserOdometry.pose.pose.position.y = y;
        laserOdometry.pose.pose.position.z = z;
        laserOdometry.pose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(roll, pitch, yaw);
        pubOdometry.publish(laserOdometry);

		double lidarOdomTime = thisOdom.header.stamp.toSec();

		// get latest odometry (at current IMU stamp)
		while (!imuOdomQueue.empty()) {
		if (imuOdomQueue.front().header.stamp.toSec() <= lidarOdomTime)
			imuOdomQueue.pop_front();
		else
			break;
		}

		Eigen::Affine3f imuOdomAffineFront = odom2affine(imuOdomQueue.front());
		Eigen::Affine3f imuOdomAffineBack = odom2affine(imuOdomQueue.back());
		Eigen::Affine3f imuOdomAffineIncre = imuOdomAffineFront.inverse() * imuOdomAffineBack;
		Eigen::Affine3f imuOdomAffineLast = lidarOdomAffine * imuOdomAffineIncre;
		float x, y, z, roll, pitch, yaw;
		pcl::getTranslationAndEulerAngles(imuOdomAffineLast, x, y, z, roll, pitch, yaw);
		
		// publish latest odometry
		nav_msgs::Odometry laserOdometry = imuOdomQueue.back();
		laserOdometry.pose.pose.position.x = x;
		laserOdometry.pose.pose.position.y = y;
		laserOdometry.pose.pose.position.z = z;
		laserOdometry.pose.pose.orientation =
			tf::createQuaternionMsgFromRollPitchYaw(roll, pitch, yaw);
		pubLidarOdometry.publish(laserOdometry);

		// publish IMU path
		static nav_msgs::Path imuPath;
		static double last_path_time = -1;
		imuTime = imuOdomQueue.back().header.stamp.toSec();
		if (imuTime - last_path_time > 0.1) {
			last_path_time = imuTime;
			geometry_msgs::PoseStamped pose_stamped;
			pose_stamped.header.stamp = imuOdomQueue.back().header.stamp;
			pose_stamped.header.frame_id = odometryFrame;
			pose_stamped.pose = laserOdometry.pose.pose;
			imuPath.poses.push_back(pose_stamped);
			while (!imuPath.poses.empty() && imuPath.poses.front().header.stamp.toSec() < lidarOdomTime - 0.1)
				imuPath.poses.erase(imuPath.poses.begin());
			if (pubLidarPath.getNumSubscribers() != 0) {
				imuPath.header.stamp = imuOdomQueue.back().header.stamp;
				imuPath.header.frame_id = odometryFrame;
				pubLidarPath.publish(imuPath);
			}
		}
		
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
                timeLaserInfoCur = cloudInfo.header.stamp.toSec();
                
                curSubMapSize++;

                if(!keyframeInit())
                    continue;
                
                ROS_WARN("Now (keyframeInit) time %f ms", ((std::chrono::duration<float>)(std::chrono::system_clock::now() - start)).count()*1000);

                updateInitialGuess();

                if(subMapFirstFlag)
                {
                    currentCloudInit();
                    saveKeyFrames();
                    publishOdometry();
                    publishKeyFrameCloud();
                    
                    fisrt_submap(currentSubMap, currentKeyFrame);
					
					#if USING_SLIDING_TARGET
						insert_local_map(localMap, currentKeyFrame, 
										local_map_radius, max_num_pts, kept_vertex_num,
										last_frame_reliable_radius, map_based_dynamic_removal_on,
										dynamic_removal_center_radius, dynamic_dist_thre_min,
										dynamic_dist_thre_max, near_dist_thre);
					#endif

                    subMapFirstFlag=false;
                    continue;
                }
                


                #if USING_SINGLE_TARGET
                    pcl::copyPointCloud(*laserCloudCornerLast,    *laserCloudCornerFromSubMap);
                    pcl::copyPointCloud(*laserCloudSurfLast,    *laserCloudSurfFromSubMap);
                    *laserCloudCornerFromSubMap = *transformPointCloud(laserCloudCornerFromSubMap, &keyFramePoses6D->back());
                    *laserCloudSurfFromSubMap = *transformPointCloud(laserCloudSurfFromSubMap, &keyFramePoses6D->back());
                #endif

				#if USING_MULTI_KEYFRAME_TARGET
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
    				extractLocalMapCloud(currentKeyFrame, cur_pose, false);
                #endif

                #if USING_SUBMAP_TARGET    
					int target_submap_id = -1;
					int target_keyframe_id = -1;
					bool using_target_id = false;
					PointTypePose  cur_pose = trans2PointTypePose(transformTobeSubMapped, keyFrameID, timeLaserInfoCur);
					
					// subMapInfo[subMapID] = currentSubMap;
					extractSurroundingKeyFrames(cur_pose, target_submap_id, target_keyframe_id, using_target_id);
					
					auto it_ = subMapInfo.find(target_submap_id);
					if(it_ != subMapInfo.end())
					{
						extractSubMapCloud(currentKeyFrame, it_->second, cur_pose, false);
						// continue;
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

                ROS_WARN("Now (extractCloud) time %f ms", ((std::chrono::duration<float>)(std::chrono::system_clock::now() - start)).count()*1000);

                currentCloudInit();
				// scan2SubMapOptimizationICP()
                scan2SubMapOptimization();
                ROS_WARN("Now (scan2SubMapOptimization) time %f ms", ((std::chrono::duration<float>)(std::chrono::system_clock::now() - start)).count()*1000);

				IMUPreintegration();
                ROS_WARN("Now (IMUPreintegration) time %f ms", ((std::chrono::duration<float>)(std::chrono::system_clock::now() - start)).count()*1000);

                calculateTranslation();
                float accu_tran = std::max(transformCurFrame2Submap[3], transformCurFrame2Submap[4]); 
                float accu_rot = transformCurFrame2Submap[2];
                if(judge_new_submap(accu_tran, accu_rot, curSubMapSize, subMapTraMax, subMapYawMax, subMapFramesSize))
                {
                    ROS_WARN("Make %d submap  has %d  Frames !", subMapID, curSubMapSize);
                    
                    saveSubMap();
                    publishSubMapCloud();
                    
                    saveKeyFrames();
                    publishOdometry();
                    publishKeyFrameCloud();
    
                    fisrt_submap(currentSubMap, currentKeyFrame);
                    curSubMapSize = 0;
                }else{

                    saveKeyFrames();
                    update_submap(currentSubMap, currentKeyFrame, 
                                local_map_radius, max_num_pts, kept_vertex_num,
                                last_frame_reliable_radius, map_based_dynamic_removal_on,
                                dynamic_removal_center_radius, dynamic_dist_thre_min,
                                dynamic_dist_thre_max, near_dist_thre);
                    
                    publishOdometry();
                    publishKeyFrameCloud();
                    
					ROS_WARN("Current SubMap Static Cloud Size: [%d, %d] .", currentSubMap->submap_pole->points.size(), currentSubMap->submap_ground->points.size());
					pcl::PointCloud<PointXYZIL>::Ptr tmpCloud( new pcl::PointCloud<PointXYZIL>());
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
					insert_local_map(localMap, currentKeyFrame, 
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

            while (seInfoQueue.size() > 6) seInfoQueue.pop_front();

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
        
        pcl::fromROSMsg(cloudInfo.semantic_raw, *currentKeyFrame->semantic_raw);
        pcl::fromROSMsg(cloudInfo.semantic_dynamic, *currentKeyFrame->semantic_dynamic);
        pcl::fromROSMsg(cloudInfo.semantic_pole, *currentKeyFrame->semantic_pole);
        pcl::fromROSMsg(cloudInfo.semantic_ground, *currentKeyFrame->semantic_ground);
        pcl::fromROSMsg(cloudInfo.semantic_building, *currentKeyFrame->semantic_building);
        pcl::fromROSMsg(cloudInfo.semantic_outlier, *currentKeyFrame->semantic_outlier);

        pcl::fromROSMsg(cloudInfo.cloud_corner, *currentKeyFrame->cloud_corner);
        pcl::fromROSMsg(cloudInfo.cloud_surface, *currentKeyFrame->cloud_surface);     

        SubMapManager::voxel_downsample_pcl(currentKeyFrame->semantic_raw, currentKeyFrame->semantic_raw_down, 0.4);
        SubMapManager::voxel_downsample_pcl(currentKeyFrame->semantic_dynamic, currentKeyFrame->semantic_dynamic_down, 0.2);
        SubMapManager::voxel_downsample_pcl(currentKeyFrame->semantic_pole, currentKeyFrame->semantic_pole_down, 0.05);
        SubMapManager::voxel_downsample_pcl(currentKeyFrame->semantic_ground, currentKeyFrame->semantic_ground_down, 0.6);
        SubMapManager::voxel_downsample_pcl(currentKeyFrame->semantic_building, currentKeyFrame->semantic_building_down, 0.4);
        SubMapManager::voxel_downsample_pcl(currentKeyFrame->semantic_outlier, currentKeyFrame->semantic_outlier_down, 0.5);
        
        SubMapManager::voxel_downsample_pcl(currentKeyFrame->cloud_corner, currentKeyFrame->cloud_corner_down, 0.2);
        SubMapManager::voxel_downsample_pcl(currentKeyFrame->cloud_surface, currentKeyFrame->cloud_surface_down, 0.4);
        
        //calculate bbx (local)
        // this->get_cloud_bbx(currentKeyFrame->semantic_raw, currentKeyFrame->local_bound);
        this->get_cloud_bbx_cpt(currentKeyFrame->semantic_raw, currentKeyFrame->local_bound, currentKeyFrame->local_cp);

        ROS_WARN("keyFrameID: %d ,keyFrameInfo Size: %d ",keyFrameID, keyFrameInfo.size());    
		
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
		
		#if USING_SEMANTIC_FEATURE
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
        #endif

		#if USING_LOAM_FEATURE
			*laserCloudCornerLast = *trans2LabelPointCloud(currentKeyFrame->cloud_corner);
			*laserCloudCornerLastDS = *trans2LabelPointCloud(currentKeyFrame->cloud_corner_down);
			*laserCloudSurfLast = *trans2LabelPointCloud(currentKeyFrame->cloud_surface);
			*laserCloudSurfLastDS = *trans2LabelPointCloud(currentKeyFrame->cloud_surface_down);																				
        #endif	

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

                lastImuTransformation = pcl::getTransformation(0, 0, 0, cloudInfo.imuRollInit, cloudInfo.imuPitchInit, cloudInfo.imuYawInit); // save imu before return;
                return;
            }
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


    void saveKeyFrames()
    {
        keyFrameID++;
		ROS_WARN("transPredictionMapped: [%f, %f, %f, %f, %f, %f]",
                transPredictionMapped[0], transPredictionMapped[1], transPredictionMapped[2],
                transPredictionMapped[3], transPredictionMapped[4], transPredictionMapped[5]);

        ROS_WARN("transformTobeSubMapped: [%f, %f, %f, %f, %f, %f]",
                transformTobeSubMapped[0], transformTobeSubMapped[1], transformTobeSubMapped[2],
                transformTobeSubMapped[3], transformTobeSubMapped[4], transformTobeSubMapped[5]);
        
        // ROS_WARN("transformCurSubmap: [%f, %f, %f, %f, %f, %f]",
        //         transformCurSubmap[0], transformCurSubmap[1], transformCurSubmap[2],
        //         transformCurSubmap[3], transformCurSubmap[4], transformCurSubmap[5]);

        PointTypePose  point6d = trans2PointTypePose(transformTobeSubMapped, keyFrameID, timeLaserInfoCur);
        PointType  point3d = trans2PointType(transformTobeSubMapped, keyFrameID);

        keyFramePoses3D->push_back(point3d);
        keyFramePoses6D->push_back(point6d);

        keyFramePosesIndex6D[keyFrameID] = point6d;
        keyFramePosesIndex3D[keyFrameID] = point3d;

        currentKeyFrame->keyframe_id = keyFrameID;
        currentKeyFrame->submap_id = subMapID;
        currentKeyFrame->id_in_submap = curSubMapSize;
        currentKeyFrame->optimized_pose = point6d;
        
        calculateTranslation();
        point6d = trans2PointTypePose(transformCurFrame2Submap, keyFrameID, timeLaserInfoCur);
        currentKeyFrame->relative_pose = point6d;

        //calculate bbx (global)
        Eigen::Affine3f tran_map = pclPointToAffine3f(currentKeyFrame->optimized_pose);
        // this->transform_bbx(currentKeyFrame->local_bound, currentKeyFrame->bound, tran_map);
        this->transform_bbx(currentKeyFrame->local_bound, currentKeyFrame->local_cp, currentKeyFrame->bound, currentKeyFrame->cp, tran_map);


        keyframe_Ptr tmpKeyFrame;
        tmpKeyFrame = keyframe_Ptr(new keyframe_t(*currentKeyFrame, true));
        keyFrameQueue.push_back(tmpKeyFrame);
        keyFrameInfo.insert(std::make_pair(keyFrameID, tmpKeyFrame));
        
        // ROS_WARN("currentKeyFrame : relative_pose: [%f, %f, %f, %f, %f, %f]",
        //         currentKeyFrame->relative_pose.roll, currentKeyFrame->relative_pose.pitch, currentKeyFrame->relative_pose.yaw,
        //         currentKeyFrame->relative_pose.x, currentKeyFrame->relative_pose.y, currentKeyFrame->relative_pose.z);

        // ROS_WARN("currentKeyFrame : optimized_pose: [%f, %f, %f, %f, %f, %f]",
        //         currentKeyFrame->optimized_pose.roll, currentKeyFrame->optimized_pose.pitch, currentKeyFrame->optimized_pose.yaw,
        //         currentKeyFrame->optimized_pose.x, currentKeyFrame->optimized_pose.y, currentKeyFrame->optimized_pose.z);

    }


    void saveSubMap()
    {
        timeSubMapInfoStamp = timeLaserInfoStamp;
        double curSubMapTime = timeSubMapInfoStamp.toSec();
        subMapID++;

        PointTypePose  point6d = trans2PointTypePose(transformCurSubmap, subMapID, curSubMapTime);
        subMapPose6D->points.push_back(point6d);      
            
        PointType  point3d = trans2PointType(transformCurSubmap, subMapID);
        subMapPose3D->points.push_back(point3d);  

        subMapPosesIndex3D[subMapID] = point3d;
        subMapPosesIndex6D[subMapID] = point6d;
        
        transformCurSubmap[0]=transformTobeSubMapped[0];
        transformCurSubmap[1]=transformTobeSubMapped[1];
        transformCurSubmap[2]=transformTobeSubMapped[2];
        transformCurSubmap[3]=transformTobeSubMapped[3];
        transformCurSubmap[4]=transformTobeSubMapped[4];
        transformCurSubmap[5]=transformTobeSubMapped[5];

        submap_Ptr tmpSubMap;
        tmpSubMap = submap_Ptr(new submap_t(*currentSubMap, true, true));
        subMapIndexQueue.push_back(subMapID);
        subMapInfo.insert(std::make_pair(subMapID, tmpSubMap));
        // subMapInfo[subMapID] = tmpSubMap;
    }



    void extractSurroundingKeyFrames(PointTypePose &cur_pose, int &target_submap_id, 
                                     int target_keyframe_id = -1, bool using_target_id = false)
    {
        kdtreeFromsubMapPose6D.reset(new pcl::KdTreeFLANN<PointTypePose>());

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
                kdtreeFromsubMapPose6D->setInputCloud(subMapPose6D); 
                // kdtreeFromsubMapPose6D->radiusSearch(subMapPose6D->back(), (double)2.0*subMapTraMax, pointSearchInd, pointSearchSqDis);
                kdtreeFromsubMapPose6D->nearestKSearch(it_->second, 2, pointSearchInd, pointSearchSqDis);
                
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
            kdtreeFromsubMapPose6D->setInputCloud(subMapPose6D); 
            // kdtreeFromsubMapPose6D->radiusSearch(subMapPose6D->back(), (double)2.0*subMapTraMax, pointSearchInd, pointSearchSqDis);
            kdtreeFromsubMapPose6D->nearestKSearch(cur_pose, 2, pointSearchInd, pointSearchSqDis);
            
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

        // std::cout << "cur_keyframe local_bound: [" << cur_keyframe->local_bound.min_x << ", "
        //                                         << cur_keyframe->local_bound.min_y << ", " 
        //                                         << cur_keyframe->local_bound.min_z << ", " 
        //                                         << cur_keyframe->local_bound.max_x << ", " 
        //                                         << cur_keyframe->local_bound.max_y << ", " 
        //                                         << cur_keyframe->local_bound.max_z
        //                                         << "]" << std::endl;

        // std::cout << "cur_keyframe bound: [" << cur_keyframe->bound.min_x << ", "
        //                                     << cur_keyframe->bound.min_y << ", " 
        //                                     << cur_keyframe->bound.min_z << ", " 
        //                                     << cur_keyframe->bound.max_x << ", " 
        //                                     << cur_keyframe->bound.max_y << ", " 
        //                                     << cur_keyframe->bound.max_z
        //                                     << "]" << std::endl;

        Eigen::Affine3f tran_map = pclPointToAffine3f(cur_submap->submap_pose_6D_optimized);
        this->transform_bbx(cur_submap->local_bound, cur_submap->local_cp, cur_submap->bound, cur_submap->cp, tran_map);

        bounds_t bbx_intersection;
        get_intersection_bbx(cur_keyframe->bound, cur_submap->bound, bbx_intersection, 2.0);

        std::cout << "cur_submap local_bound: [" << cur_submap->local_bound.min_x << ", "
                                                << cur_submap->local_bound.min_y << ", " 
                                                << cur_submap->local_bound.min_z << ", " 
                                                << cur_submap->local_bound.max_x << ", " 
                                                << cur_submap->local_bound.max_y << ", " 
                                                << cur_submap->local_bound.max_z
                                                << "]" << std::endl;

        std::cout << "cur_submap bound: [" << cur_submap->bound.min_x << ", "
                                        << cur_submap->bound.min_y << ", " 
                                        << cur_submap->bound.min_z << ", " 
                                        << cur_submap->bound.max_x << ", " 
                                        << cur_submap->bound.max_y << ", " 
                                        << cur_submap->bound.max_z
                                        << "]" << std::endl;

        std::cout << "bbx_intersection: [" << bbx_intersection.min_x << ", "
                                           << bbx_intersection.min_y << ", " 
                                           << bbx_intersection.min_z << ", " 
                                           << bbx_intersection.max_x << ", " 
                                           << bbx_intersection.max_y << ", " 
                                           << bbx_intersection.max_z  
                                           << "]" << std::endl; 

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
        std::cout << "cloud_temp size: " << cloud_temp->points.size() << std::endl;
        publishLabelCloud(&pubTest1, cloud_temp, timeLaserInfoStamp, lidarFrame);

    }


    void extractLocalMapCloud(keyframe_Ptr &cur_keyframe, PointTypePose cur_pose, bool using_keyframe_pose = false)
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
        
		bounds_t bbx_intersection;
        get_intersection_bbx(cur_keyframe->bound, localMap->bound, bbx_intersection, 10.0);
		
		// pcl::PointCloud<PointXYZIL>::Ptr cloud_temp(new pcl::PointCloud<PointXYZIL>);
		// localMap->merge_feature_points(cloud_temp, false);
        
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
        std::cout << "cloud_temp size: " << cloud_temp->points.size() << std::endl;
        publishLabelCloud(&pubTest1, cloud_temp, timeLaserInfoStamp, odometryFrame);

	}


	void icpAlignment(pcl::PointCloud<PointXYZIL>::Ptr source_pc, pcl::PointCloud<PointXYZIL>::Ptr target_pc, float transformIn[])
	{

		ROS_WARN("Source PC Size: %d, Target PC Size: %d", source_pc->points.size(), target_pc->points.size());

        std::cout << "matching..." << std::flush;
        auto t1 = ros::Time::now();

        // ICP Settings
        static pcl::IterativeClosestPoint<PointXYZIL, PointXYZIL> icp;
        icp.setMaxCorrespondenceDistance(30);
        icp.setMaximumIterations(40);
        icp.setTransformationEpsilon(1e-6);
        icp.setEuclideanFitnessEpsilon(1e-6);
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
		// transformIn[0] = ROLL;
		// transformIn[1] = PITCH;
		// transformIn[2] = YAW;
		// transformIn[3] = X;
		// transformIn[4] = Y;
		// transformIn[5] = Z;
	}

    void scan2SubMapOptimizationICP()
	{
		pcl::PointCloud<PointXYZIL>::Ptr sourcePC( new pcl::PointCloud<PointXYZIL>());
		pcl::PointCloud<PointXYZIL>::Ptr targetPC( new pcl::PointCloud<PointXYZIL>());

		*sourcePC += *currentKeyFrame->semantic_dynamic;
		*sourcePC += *currentKeyFrame->semantic_pole;
		*sourcePC += *currentKeyFrame->semantic_ground;
		*sourcePC += *currentKeyFrame->semantic_building;

		*targetPC += *laserCloudSurfFromSubMap;
		*targetPC += *laserCloudCornerFromSubMap;

		icpAlignment(sourcePC, targetPC, transformTobeSubMapped);
	}

    void scan2SubMapOptimization()
    {
        if (laserCloudCornerLastDSNum > edgeFeatureMinValidNum && laserCloudSurfLastDSNum > surfFeatureMinValidNum)
        {
            ROS_WARN("laserCloudCornerFromSubMap: %d laserCloudSurfFromSubMap: %d .", laserCloudCornerFromSubMap->points.size(), laserCloudSurfFromSubMap->points.size());
            ROS_WARN("laserCloudCornerLastDSNum: %d laserCloudSurfLastDSNum: %d .", laserCloudCornerLastDSNum, laserCloudSurfLastDSNum);
            
            kdtreeCornerFromSubMap->setInputCloud(laserCloudCornerFromSubMap);
            kdtreeSurfFromSubMap->setInputCloud(laserCloudSurfFromSubMap);

            for (int iterCount = 0; iterCount < 30; iterCount++)   //30
            {
                laserCloudOri->clear();
                coeffSel->clear();

                cornerOptimization();

                surfOptimization();

                combineOptimizationCoeffs();

                if (LMOptimization(iterCount) == true)
                    break;          
            }

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

                    // coeff.x = w * s * la;
                    // coeff.y = w * s * lb;
                    // coeff.z = w * s * lc;
                    // coeff.intensity = w * s * ld2;

                    coeff.x = s * la;
                    coeff.y = s * lb;
                    coeff.z = s * lc;
                    coeff.intensity = w * s * ld2;

                    if (s > 0.1) {
                        laserCloudOriCornerVec[i] = pointOri;
                        coeffSelCornerVec[i] = coeff;
                        laserCloudOriCornerFlag[i] = true;

						numSearch++;
                    }
                }
            }
        }

		ROS_WARN("Corner numSearch: [%d / %d]", numSearch, laserCloudCornerLastDSNum);

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

                    // coeff.x = w * s * pa;
                    // coeff.y = w * s * pb;
                    // coeff.z = w * s * pc;
                    // coeff.intensity = w * s * pd2;
                    
					coeff.x = s * pa;
                    coeff.y = s * pb;
                    coeff.z = s * pc;
                    coeff.intensity = w * s * pd2;

                    if (s > 0.1) {
                        laserCloudOriSurfVec[i] = pointOri;
                        coeffSelSurfVec[i] = coeff;
                        laserCloudOriSurfFlag[i] = true;
						
						numSearch++;
                    }
                }
            }
        }

		ROS_WARN("Surf numSearch: [%d / %d]", numSearch, laserCloudSurfLastDSNum);

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

        float deltaR = sqrt(
                            pow(pcl::rad2deg(matX.at<float>(0, 0)), 2) +
                            pow(pcl::rad2deg(matX.at<float>(1, 0)), 2) +
                            pow(pcl::rad2deg(matX.at<float>(2, 0)), 2));
        float deltaT = sqrt(
                            pow(matX.at<float>(3, 0) * 100, 2) +
                            pow(matX.at<float>(4, 0) * 100, 2) +
                            pow(matX.at<float>(5, 0) * 100, 2));

        if (deltaR < 0.03 && deltaT < 0.03) {
            return true; // converged
        }
        return false; // keep optimizing
    }

    void transformUpdate()
    {
        // if (cloudInfo.imuAvailable == true)
        // {
        //     if (std::abs(cloudInfo.imuPitchInit) < 1.4)
        //     {
        //         double imuWeight = imuRPYWeight;
        //         tf::Quaternion imuQuaternion;
        //         tf::Quaternion transformQuaternion;
        //         double rollMid, pitchMid, yawMid;

        //         // slerp roll
        //         transformQuaternion.setRPY(transformTobeSubMapped[0], 0, 0);
        //         imuQuaternion.setRPY(cloudInfo.imuRollInit, 0, 0);
        //         tf::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid);
        //         transformTobeSubMapped[0] = rollMid;

        //         // slerp pitch
        //         transformQuaternion.setRPY(0, transformTobeSubMapped[1], 0);
        //         imuQuaternion.setRPY(0, cloudInfo.imuPitchInit, 0);
        //         tf::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid);
        //         transformTobeSubMapped[1] = pitchMid;
        //     }
        // }

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
			if (imuTime < currentCorrectionTime - delta_t) {
				double dt = (lastImuT_opt < 0) ? (1.0 / 500.0) : (imuTime - lastImuT_opt);
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
				double dt = (lastImuQT < 0) ? (1.0 / 500.0) : (imuTime - lastImuQT);

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

		ROS_WARN("timeLaserInfoStamp: %f, lastImuT_opt: %f.", timeLaserInfoStamp.toSec(), lastImuT_opt);
		
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
        static tf::TransformBroadcaster br;
        tf::Transform t_odom_to_lidar = tf::Transform(tf::createQuaternionFromRPY(transformTobeSubMapped[0], transformTobeSubMapped[1], transformTobeSubMapped[2]),
                                                      tf::Vector3(transformTobeSubMapped[3], transformTobeSubMapped[4], transformTobeSubMapped[5]));
        tf::StampedTransform trans_odom_to_lidar = tf::StampedTransform(t_odom_to_lidar, timeLaserInfoStamp, odometryFrame, lidarFrame);
        br.sendTransform(trans_odom_to_lidar);

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
				continue;
			}

			// rate.sleep();
			ros::spinOnce();
			if(!keyFrameQueue.empty())
			{
				keyframe_Ptr curKeyFramePtr = keyFrameQueue.front();

				auto t1 = ros::Time::now();

				Eigen::Affine3f curPose = pclPointToAffine3f(curKeyFramePtr->optimized_pose);
				epscGen.loopDetection(
						curKeyFramePtr->cloud_corner, curKeyFramePtr->cloud_surface,
						curKeyFramePtr->semantic_raw, curPose);

				int loopKeyCur = epscGen.current_frame_id;
				std::vector<int> loopKeyPre;
				loopKeyPre.assign(epscGen.matched_frame_id.begin(),
								  epscGen.matched_frame_id.end());
				std::vector<Eigen::Affine3f> matched_init_transform;
				matched_init_transform.assign(epscGen.matched_frame_transform.begin(),
											  epscGen.matched_frame_transform.end());

				cv_bridge::CvImage out_msg;
				out_msg.header.frame_id = lidarFrame;
				out_msg.header.stamp = curKeyFramePtr->timeInfoStamp;
				out_msg.encoding = sensor_msgs::image_encodings::RGB8;
				out_msg.image = epscGen.getLastSEPSCRGB();
				pubSCDe.publish(out_msg.toImageMsg());

				out_msg.image = epscGen.getLastEPSCRGB();
				pubEPSC.publish(out_msg.toImageMsg());

				out_msg.image = epscGen.getLastSCRGB();
				pubSC.publish(out_msg.toImageMsg()); 

				out_msg.image = epscGen.getLastISCRGB();
				pubISC.publish(out_msg.toImageMsg());   

				out_msg.image = epscGen.getLastSSCRGB();
				pubSSC.publish(out_msg.toImageMsg());

				curKeyFramePtr->global_descriptor = epscGen.getLastSEPSCMONO();
					
				ros::Time t2 = ros::Time::now();
				ROS_WARN("Detect Loop Closure Time: %.3f", (t2 - t1).toSec());
				
				std::cout << std::endl;
				std::cout << "--- loop detection ---" << std::endl;
				std::cout << "keyframe_id : " << curKeyFramePtr->keyframe_id << std::endl;
				std::cout << "loopKeyCur : " << loopKeyCur << std::endl;
				std::cout << "num_candidates: " << loopKeyPre.size() << std::endl;
				
				if (loopKeyPre.empty()) 
				{
					ROS_WARN("loopKeyPre is empty !");
					continue;
				}
				for (int i = 0; i < loopKeyPre.size(); i++) 
				{
					std::cout << "loopKeyPre [" << i << "]:" << loopKeyPre[i] << std::endl;
				}
				
				int bestMatched = -1;
				if (detectLoopClosure(loopKeyCur, loopKeyPre, matched_init_transform, bestMatched) == false)
					continue;

				visualizeLoopClosure();

				curKeyFramePtr->loop_container.push_back(bestMatched);											  

				keyFrameQueue.pop_front();
			}
			
        }
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
            *cureKeyframeCloud += *keyFrameInfo[loopKeyCur]->semantic_raw;
        } else {
            loopKeyCur = -1;
            ROS_WARN("LoopKeyCur do not find !");
            return false;
        }

        std::cout << "matching..." << std::flush;
        auto t1 = ros::Time::now();

        // ICP Settings
        static pcl::IterativeClosestPoint<PointXYZIL, PointXYZIL> icp;
        icp.setMaxCorrespondenceDistance(30);
        icp.setMaximumIterations(40);
        icp.setTransformationEpsilon(1e-6);
        icp.setEuclideanFitnessEpsilon(1e-6);
        icp.setRANSACIterations(0);

        // int bestMatched = -1;
        int bestID = -1;
        double bestScore = std::numeric_limits<double>::max();
        Eigen::Affine3f correctionLidarFrame;

        for (int i = 0; i < loopKeyPre.size(); i++) 
        {
            // Align clouds
            pcl::PointCloud<PointXYZIL>::Ptr tmpCloud( new pcl::PointCloud<PointXYZIL>());
            *tmpCloud += *transformPointCloud(cureKeyframeCloud, matched_init_transform[i]);
            icp.setInputSource(tmpCloud);

            auto thisPreId = keyFrameInfo.find(loopKeyPre[i]);
            if (thisPreId != keyFrameInfo.end()) {
                int PreID = (int)keyFrameInfo[loopKeyPre[i]]->keyframe_id;
                std::cout << "loopContainerHandler: loopKeyPre : " << PreID << std::endl;

                prevKeyframeCloud->clear();
                *tmpCloud += *keyFrameInfo[loopKeyPre[i]]->semantic_raw;
                icp.setInputTarget(prevKeyframeCloud);

                pcl::PointCloud<PointXYZIL>::Ptr unused_result( new pcl::PointCloud<PointXYZIL>());
                icp.align(*unused_result);

                double score = icp.getFitnessScore();
                if (icp.hasConverged() == false || score > bestScore) 
                    continue;
                bestScore = score;
                bestMatched = PreID;
                bestID = i;
                correctionLidarFrame = icp.getFinalTransformation();

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
        loopIndexVec.push_back(make_pair(loopKeyCur, bestMatched));
        loopPoseVec.push_back(pose);
        loopNoiseVec.push_back(constraintNoise);
        // mtx.unlock();

        // add loop constriant
        loopIndexContainer[loopKeyCur] = bestMatched;

        return true;
    }







    void visualizeLoopClosure() 
    {
        visualization_msgs::MarkerArray markerArray;
        // loop nodes
        visualization_msgs::Marker markerNode;
        markerNode.header.frame_id = odometryFrame;
        markerNode.header.stamp = ros::Time::now();
        markerNode.action = visualization_msgs::Marker::ADD;
        markerNode.type = visualization_msgs::Marker::SPHERE_LIST;
        markerNode.ns = "loop_nodes";
        markerNode.id = 0;
        markerNode.pose.orientation.w = 1;
        markerNode.scale.x = 2.1;
        markerNode.scale.y = 2.1;
        markerNode.scale.z = 2.1;
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
        markerEdge.scale.x = 2.0;
        markerEdge.scale.y = 2.0;
        markerEdge.scale.z = 2.0;
        markerEdge.color.r = 1.0;
        markerEdge.color.g = 0.0;
        markerEdge.color.b = 0;
        markerEdge.color.a = 1;

        for (auto it = loopIndexContainer.begin(); it != loopIndexContainer.end(); ++it) 
        {
            int key_cur = it->first;
            int key_pre = it->second;
            geometry_msgs::Point p;
            p.x = keyFrameInfo[key_cur]->optimized_pose.x;
            p.y = keyFrameInfo[key_cur]->optimized_pose.y;
            p.z = keyFrameInfo[key_cur]->optimized_pose.z;
            markerNode.points.push_back(p);
            markerEdge.points.push_back(p);
            p.x = keyFrameInfo[key_pre]->optimized_pose.x;
            p.y = keyFrameInfo[key_pre]->optimized_pose.y;
            p.z = keyFrameInfo[key_pre]->optimized_pose.z;
            markerNode.points.push_back(p);
            markerEdge.points.push_back(p);
        }

        markerArray.markers.push_back(markerNode);
        markerArray.markers.push_back(markerEdge);
        pubLoopConstraintEdge.publish(markerArray);
    }



};


class SubMapOptmizationNode : public SubMapManager<PointXYZIL> {
 public:
    ros::Subscriber subGPS;

    std::mutex gpsMtx;
    std::deque<nav_msgs::Odometry> gpsQueue;

    ros::Publisher pubCloudMap;

    ros::Publisher pubSubMapId;
    ros::Publisher pubSubMapConstraintEdge;

    ros::Publisher pubSubMapOdometryGlobal;
    ros::Publisher pubOdometryGlobal;
    
    
    void allocateMemory() {}

    SubMapOptmizationNode() 
    {
        subGPS = nh.subscribe<nav_msgs::Odometry>(gpsTopic, 2000, &SubMapOptmizationNode::gpsHandler, this, ros::TransportHints().tcpNoDelay());
        
        pubCloudMap = nh.advertise<sensor_msgs::PointCloud2>("lis_slam/mapping/map_global", 1);
                
        pubSubMapId = nh.advertise<sensor_msgs::PointCloud2>("lis_slam/mapping/submap_id", 1);
        pubSubMapConstraintEdge = nh.advertise<visualization_msgs::MarkerArray>("lis_slam/mapping/submap_constraints", 1);
        
		pubSubmapOdometryGlobal = nh.advertise<nav_msgs::Odometry> (odomTopic + "/submap", 200);
		pubOdometryGlobal = nh.advertise<nav_msgs::Odometry> (odomTopic + "/fusion", 200);
        

        allocateMemory();
    }

    void gpsHandler(const nav_msgs::Odometry::ConstPtr &msgIn) 
    {
        std::lock_guard<std::mutex> lock(gpsMtx);
        gpsQueue.push_back(*msgIn);
    }
    
    /*****************************
     * @brief
     * @param input
     *****************************/
    void visualizeGlobalMapThread() {}

    /*****************************
     * @brief
     * @param input
     *****************************/
    void subMapOptmizationThread() {}

};

int main(int argc, char **argv) 
{
    ros::init(argc, argv, "lis_slam");

    SubMapOdometryNode SOD;
    std::thread make_submap_process(&SubMapOdometryNode::makeSubMapThread, &SOD);
    // std::thread loop_closure_process(&SubMapOdometryNode::loopClosureThread, &SOD);
    

    // SubMapOptmizationNode SOP;
    // std::thread visualize_map_process(&SubMapOptmizationNode::visualizeGlobalMapThread, &SOP);
    // std::thread submap_optmization_process(&SubMapOptmizationNode::subMapOptmizationThread, &SOP);
    
    ROS_WARN("\033[1;32m----> SubMap Optmization Node Started.\033[0m");

	ros::MultiThreadedSpinner spinner(4);
	spinner.spin();
    // ros::spin();

    make_submap_process.join();
    // loop_closure_process.join();
    
    // visualize_map_process.join();
    // submap_optmization_process.join();

    return 0;
}