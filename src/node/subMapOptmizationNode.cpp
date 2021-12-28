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

#include "lis_slam/semantic_info.h"

#include "epscGeneration.h"
#include "subMap.h"
#include "utility.h"
#include "common.h"

using namespace gtsam;

using symbol_shorthand::B;  // Bias  (ax,ay,az,gx,gy,gz)
using symbol_shorthand::G;  // GPS pose
using symbol_shorthand::V;  // Vel   (xdot,ydot,zdot)
using symbol_shorthand::X;  // Pose3 (x,y,z,r,p,y)


std::mutex subMapMtx;
std::deque<submap_Ptr> subMapQueue;

map<int, int> loopIndexContainer;  // from new to old
vector<pair<int, int>> loopIndexQueue;
vector<gtsam::Pose3> loopPoseQueue;
vector<gtsam::noiseModel::Diagonal::shared_ptr> loopNoiseQueue;


gtsam::Pose3 pclPointTogtsamPose3(PointTypePose thisPoint) {
    return gtsam::Pose3(
        gtsam::Rot3::RzRyRx(double(thisPoint.roll), double(thisPoint.pitch),
                            double(thisPoint.yaw)),
        gtsam::Point3(double(thisPoint.x), double(thisPoint.y),
                    double(thisPoint.z)));
}

gtsam::Pose3 trans2gtsamPose(float transformIn[]) {
    return gtsam::Pose3(
        gtsam::Rot3::RzRyRx(transformIn[0], transformIn[1], transformIn[2]),
        gtsam::Point3(transformIn[3], transformIn[4], transformIn[5]));
}


class SubMapOdometryNode : public SubMapManager<PointXYZIL>{
 public:
    ros::Subscriber subCloud;
    ros::Subscriber subIMU;

    std::mutex seMtx;
    std::mutex imuMtx;
    std::deque<lis_slam::semantic_info> seInfoQueue;
    std::deque<sensor_msgs::Imu> imuQueOpt;
    std::deque<sensor_msgs::Imu> imuQueImu;
    
    int keyFrameID = 0;
    int subMapID = 0;
    std::deque<keyframe_Ptr> keyFrameQueue;
    map<int, keyframe_Ptr> keyFrameInfo;

    keyframe_Ptr currentKeyFrame(new keyframe_t());
    lis_slam::semantic_info cloudInfo;

    submap_Ptr currentSubMap(new submap_t());

    pcl::PointCloud<PointTypePose>::Ptr subMapPose6D; 
    pcl::PointCloud<PointType>::Ptr subMapPose3D;
    
    map<int,PointType> subMapPosesIndex3D;
    map<int,PointTypePose> subMapPosesIndex6D;
    
    pcl::PointCloud<PointType>::Ptr cloudKeyPoses3D;
    pcl::PointCloud<PointTypePose>::Ptr cloudKeyPoses6D;

    pcl::PointCloud<PointXYZIL>::Ptr laserCloudCornerLast; // corner feature set from odoOptimization
    pcl::PointCloud<PointXYZIL>::Ptr laserCloudSurfLast; // surf feature set from odoOptimization
    pcl::PointCloud<PointXYZIL>::Ptr laserCloudCornerLastDS; // downsampled corner featuer set from odoOptimization
    pcl::PointCloud<PointXYZIL>::Ptr laserCloudSurfLastDS; // downsampled surf featuer set from odoOptimization
    
    pcl::PointCloud<PointXYZIL>::Ptr laserCloudCornerFromSubMap;
    pcl::PointCloud<PointXYZIL>::Ptr laserCloudSurfFromSubMap;
    pcl::PointCloud<PointXYZIL>::Ptr laserCloudCornerFromSubMapDS;
    pcl::PointCloud<PointXYZIL>::Ptr laserCloudSurfFromSubMapDS;

    int laserCloudCornerLastDSNum = 0;
    int laserCloudSurfLastDSNum = 0;

    ros::Time timeLaserInfoStamp;
    ros::Time timeSubMapInfoStamp;
    double timeLaserInfoCur;

    float transformTobeSubMapped[6];
    float afterMapOptmizationPoses[6];
    float transPredictionMapped[6];

    Eigen::Affine3f transDelta;
    
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

    void allocateMemory() {
        subMapPose6D.reset(new pcl::PointCloud<PointTypePose>()); 
        subMapPose3D.reset(new pcl::PointCloud<PointType>());

        cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
        cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());

        laserCloudCornerLast.reset(new pcl::PointCloud<PointXYZIL>()); 
        laserCloudSurfLast.reset(new pcl::PointCloud<PointXYZIL>()); 
        laserCloudCornerLastDS.reset(new pcl::PointCloud<PointXYZIL>()); 
        laserCloudSurfLastDS.reset(new pcl::PointCloud<PointXYZIL>()); 

        laserCloudCornerFromSubMap.reset(new pcl::PointCloud<PointXYZIL>());
        laserCloudSurfFromSubMap.reset(new pcl::PointCloud<PointXYZIL>());
        laserCloudCornerFromSubMapDS.reset(new pcl::PointCloud<PointXYZIL>());
        laserCloudSurfFromSubMapDS.reset(new pcl::PointCloud<PointXYZIL>());
        
        for (int i = 0; i < 6; ++i)
        {
            transformTobeSubMapped[i] = 0;
            afterMapOptmizationPoses[i] = 0;
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

    }
    
    SubMapOdometryNode() {
        subCloud = nh.subscribe<lis_slam::semantic_info>( "lis_slam/semantic_fusion/semantic_info", 10, &SubMapOptmizationNode::semanticInfoHandler, this, ros::TransportHints().tcpNoDelay());
        subImu = nh.subscribe<sensor_msgs::Imu>(imuTopic, 2000, &IMUPreintegration::imuHandler, this, ros::TransportHints().tcpNoDelay());
        
        pubKeyFrameOdometryGlobal      = nh.advertise<nav_msgs::Odometry> ("lis_slam/make_submap/odometry", 1);
        pubKeyFrameOdometryIncremental = nh.advertise<nav_msgs::Odometry> ("lis_slam/make_submap/odometry_incremental", 1);
      
        pubKeyFramePoseGlobal = nh.advertise<sensor_msgs::PointCloud2>("lis_slam/make_submap/trajectory", 1);
        pubKeyFramePath = nh.advertise<nav_msgs::Path>("lis_slam/make_submap/keyframe_path", 1);

        pubCloudRegisteredRaw = nh.advertise<sensor_msgs::PointCloud2>("lis_slam/make_submap/registered_raw", 1);
        
        pubCloudCurSubMap = nh.advertise<sensor_msgs::PointCloud2>("lis_slam/make_submap/submap", 1); 
        pubSubMapId = nh.advertise<sensor_msgs::PointCloud2>("lis_slam/make_submap/submap_id", 1);

        pubLoopConstraintEdge = nh.advertise<visualization_msgs::MarkerArray>("lis_slam/make_submap/loop_closure_constraints", 1);
        pubSCDe = nh.advertise<sensor_msgs::Image>("global_descriptor", 100);
        pubEPSC = nh.advertise<sensor_msgs::Image>("global_descriptor_epsc", 100);
        pubSC = nh.advertise<sensor_msgs::Image>("global_descriptor_sc", 100);
        pubISC = nh.advertise<sensor_msgs::Image>("global_descriptor_isc", 100);
        pubSSC = nh.advertise<sensor_msgs::Image>("global_descriptor_ssc", 100);

        allocateMemory();
    }

    void semanticInfoHandler(const lis_slam::semantic_infoConstPtr &msgIn) {
        std::lock_guard<std::mutex> lock(seMtx);
        seInfoQueue.push_back(*msgIn);
    }


    void imuHandler(const sensor_msgs::Imu::ConstPtr& imu_raw) {
        std::lock_guard<std::mutex> lock(imuMtx);

        sensor_msgs::Imu thisImu = imuConverter(*imu_raw);

        imuQueOpt.push_back(thisImu);
        imuQueImu.push_back(thisImu);


    }


    /*****************************
     * @brief
     * @param input
     *****************************/
    void makeSubMapThread() {
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

        while(ros::ok()){
            if (seInfoQueue.size() > 0) {
                std::chrono::time_point<std::chrono::system_clock> start, end;
                start = std::chrono::system_clock::now();
                
                std::lock_guard<std::mutex> lock(seMtx);

                cloudInfo = seInfoQueue.front();
                seInfoQueue.pop_front();

                timeLaserInfoStamp = cloudInfo.header.stamp;
                timeLaserInfoCur = cloudInfo.header.stamp.toSec();

                if(!keyframeInit())
                    continue;

                currentCloudInit();
                updateInitialGuess();
                
                curSubMapSize++;

                if(subMapFirstFlag)
                {
                    saveKeyFrames();
                    saveSubMap();
                    fisrt_submap(currentSubMap, currentKeyFrame);
                    publishOdometry();
                    subMapFirstFlag=false;
                    continue;
                }

                extractSurroundingKeyFrames();
                scan2SubMapOptimization();

                saveKeyFrames();
                update_submap(currentSubMap, currentKeyFrame, 
                              local_map_radius, max_num_pts, kept_vertex_num,
                              last_frame_reliable_radius, map_based_dynamic_removal_on,
                              dynamic_removal_center_radius, dynamic_dist_thre_min,
                              dynamic_dist_thre_max, near_dist_thre);
                
                keyframe_Ptr tmpKeyFrame(new keyframe_t(*currentKeyFrame));
                keyFrameQueue.push_back(tmpKeyFrame);
                keyFrameInfo[keyFrameID] = tmpKeyFrame;

                publishOdometry();
                publishKeyFrameCloud();

                // bool judge_new_submap(float &accu_tran, float &accu_rot, int &accu_frame,
                //                       float max_accu_tran = 30.0, 
                //                       float max_accu_rot = 90.0, 
                //                       int max_accu_frame = 150);
                //           << "Submap division criterion is: \n"
                //           << "1. Frame Number <= " << max_accu_frame
                //           << " , 2. Translation <= " << max_accu_tran
                //           << "m , 3. Rotation <= " << max_accu_rot << " degree."
                calculateTranslation();
                float accu_tran = std::max(transformCurFrame2Submap[3], transformCurFrame2Submap[4]); 
                float accu_rot = transformCurFrame2Submap[2];
                if(judge_new_submap(accu_tran, accu_rot, curSubMapSize, subMapTraMax, subMapYawMax, subMapFramesSize))
                {
                    ROS_INFO("Make %d submap  has %d  Frames !", subMapId, curSubMapSize);
                
                    saveSubMap();

                    submap_Ptr tmpSubMap(new submap_t(*currentSubMap));
                    subMapQueue.push_back(tmpSubMap);

                    publishSubMapCloud();
              
                    fisrt_submap(currentSubMap, currentKeyFrame);

                    curSubMapSize = 0;
                }

                end = std::chrono::system_clock::now();
                std::chrono::duration<float> elapsed_seconds = end - start;
                total_frame++;
                float time_temp = elapsed_seconds.count() * 1000;
                total_time += time_temp;
                
                ROS_INFO("Average make SubMap time %f ms", total_time / total_frame);
            }

            while (seInfoQueue.size() > 6) seInfoQueue.pop_front();
        }
    }

    bool keyframeInit()
    {
        currentKeyFrame->free_all();
        currentKeyFrame->loop_container.clear();
        
        currentKeyFrame->keyframe_id = keyFrameID;
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
        
        pcl::PointCloud<PointXYZIL>::Ptr semantic_pointcloud_in(new pcl::PointCloud<PointXYZIL>());
        pcl::PointCloud<PointXYZIL>::Ptr dynamic_pointcloud_in(new pcl::PointCloud<PointXYZIL>());
        pcl::PointCloud<PointXYZIL>::Ptr static_pointcloud_in(new pcl::PointCloud<PointXYZIL>());
        pcl::PointCloud<PointXYZIL>::Ptr outlier_pointcloud_in(new pcl::PointCloud<PointXYZIL>());

        pcl::PointCloud<pcl::PointXYZI>::Ptr corner_pointcloud_in(new pcl::PointCloud<pcl::PointXYZI>());
        pcl::PointCloud<pcl::PointXYZI>::Ptr surf_pointcloud_in(new pcl::PointCloud<pcl::PointXYZI>());

        pcl::fromROSMsg(cloudInfo.cloud_semantic, *semantic_pointcloud_in);
        pcl::fromROSMsg(cloudInfo.cloud_dynamic, *dynamic_pointcloud_in);
        pcl::fromROSMsg(cloudInfo.cloud_static, *static_pointcloud_in);
        pcl::fromROSMsg(cloudInfo.cloud_outlier, *outlier_pointcloud_in);

        pcl::fromROSMsg(cloudInfo.cloud_corner, *corner_pointcloud_in);
        pcl::fromROSMsg(cloudInfo.cloud_surface, *surf_pointcloud_in);

        pcl::copyPointCloud(*semantic_pointcloud_in,    *currentKeyFrame->cloud_semantic);
        pcl::copyPointCloud(*dynamic_pointcloud_in,    *currentKeyFrame->cloud_dynamic);
        pcl::copyPointCloud(*static_pointcloud_in,    *currentKeyFrame->cloud_static);
        pcl::copyPointCloud(*outlier_pointcloud_in,    *currentKeyFrame->cloud_outlier);
        
        pcl::copyPointCloud(*corner_pointcloud_in,    *currentKeyFrame->cloud_corner);
        pcl::copyPointCloud(*surf_pointcloud_in,    *currentKeyFrame->cloud_surface);


        pcl::PointCloud<PointXYZIL>::Ptr semantic_pointcloud_DS(new pcl::PointCloud<PointXYZIL>());
        pcl::PointCloud<PointXYZIL>::Ptr dynamic_pointcloud_DS(new pcl::PointCloud<PointXYZIL>());
        pcl::PointCloud<PointXYZIL>::Ptr static_pointcloud_DS(new pcl::PointCloud<PointXYZIL>());
        pcl::PointCloud<PointXYZIL>::Ptr outlier_pointcloud_DS(new pcl::PointCloud<PointXYZIL>());
        
        pcl::PointCloud<pcl::PointXYZI>::Ptr corner_pointcloud_DS(new pcl::PointCloud<pcl::PointXYZI>());
        pcl::PointCloud<pcl::PointXYZI>::Ptr surf_pointcloud_DS(new pcl::PointCloud<pcl::PointXYZI>());
        
        voxel_downsample_pcl(semantic_pointcloud_in, semantic_pointcloud_DS, mappingSurfLeafSize);
        voxel_downsample_pcl(dynamic_pointcloud_in, dynamic_pointcloud_DS, mappingCornerLeafSize);
        voxel_downsample_pcl(static_pointcloud_in, static_pointcloud_DS, mappingCornerLeafSize);
        voxel_downsample_pcl(outlier_pointcloud_in, outlier_pointcloud_DS, mappingSurfLeafSize);
        
        voxel_downsample_pcl(corner_pointcloud_in, corner_pointcloud_DS, mappingCornerLeafSize);
        voxel_downsample_pcl(surf_pointcloud_in, surf_pointcloud_DS, mappingSurfLeafSize);

        pcl::copyPointCloud(*semantic_pointcloud_DS,    *currentKeyFrame->cloud_semantic_down);
        pcl::copyPointCloud(*static_pointcloud_DS,    *currentKeyFrame->cloud_dynamic_down);
        pcl::copyPointCloud(*outlier_pointcloud_DS,    *currentKeyFrame->cloud_static_down);
        pcl::copyPointCloud(*outlier_pointcloud_in,    *currentKeyFrame->cloud_outlier_down);
        
        pcl::copyPointCloud(*corner_pointcloud_DS,    *currentKeyFrame->cloud_corner_down);
        pcl::copyPointCloud(*surf_pointcloud_DS,    *currentKeyFrame->cloud_surface_down);
        
        ROS_WARN("keyFrameID: %d ,keyFrameInfo Size: %d ",keyFrameID, keyFrameInfo.size());    
        
        return true;
    }
    
    void currentCloudInit()
    {
        pcl::PointCloud<PointXYZI>::Ptr thisFrameCloud(new pcl::PointCloud<PointXYZI>());
        pcl::PointCloud<PointXYZI>::Ptr thisFrameCloudDS(new pcl::PointCloud<PointXYZI>());
        
        pcl::copyPointCloud(*currentKeyFrame->cloud_static, *thisFrameCloud);
        pcl::copyPointCloud(*currentKeyFrame->cloud_static_down, *thisFrameCloudDS);
        
        for (int i = 0; i < (int)thisFrameCloud->points.size(); i++) {
            auto label = thisFrameCloud->points[i].label;
            
            if (UsingLableMap[label] == 40) {
                laserCloudSurfLast->points.push_back(thisFrameCloud->points[i]);
            } else if (UsingLableMap[label] == 81) {
                laserCloudCornerLast->points.push_back(thisFrameCloud->points[i]);
            }
        }

        for (int i = 0; i < (int)thisFrameCloudDS->points.size(); i++) {
            auto label = thisFrameCloudDS->points[i].label;
            
            if (UsingLableMap[label] == 40) {
                laserCloudSurfLastDS->points.push_back(thisFrameCloudDS->points[i]);
            } else if (UsingLableMap[label] == 81) {
                laserCloudCornerLastDS->points.push_back(thisFrameCloudDS->points[i]);
            }
        }

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
            // ROS_WARN("MakeSubmap: firstTransAvailable!");
            transformTobeSubMapped[0] = cloudInfo.imuRollInit;
            transformTobeSubMapped[1] = cloudInfo.imuPitchInit;
            transformTobeSubMapped[2] = cloudInfo.imuYawInit;
        
            // if (!useImuHeadingInitialization)
            //     transformTobeSubMapped[2] = 0;

            lastImuTransformation = pcl::getTransformation(0, 0, 0, cloudInfo.imuRollInit, cloudInfo.imuPitchInit, cloudInfo.imuYawInit); // save imu before return;
            firstTransAvailable = true;
            return;
        }

        // use imu pre-integration estimation for pose guess
        static bool lastImuPreTransAvailable = false;
        static Eigen::Affine3f lastImuPreTransformation;
        if (cloudInfo.odomAvailable == true)
        {
            // ROS_WARN("MakeSubmap: cloudInfo.odomAvailable == true!");
            Eigen::Affine3f transBack = pcl::getTransformation(cloudInfo.initialGuessX,    cloudInfo.initialGuessY,     cloudInfo.initialGuessZ, 
                                                               cloudInfo.initialGuessRoll, cloudInfo.initialGuessPitch, cloudInfo.initialGuessYaw);
            if (lastImuPreTransAvailable == false)
            {
                lastImuPreTransformation = transBack;
                lastImuPreTransAvailable = true;
            } else {

                
                Eigen::Affine3f transIncre = lastImuPreTransformation.inverse() * transBack;
                Eigen::Affine3f transTobe = trans2Affine3f(transformTobeSubMapped);
                Eigen::Affine3f transFinal = transTobe * transIncre;
                pcl::getTranslationAndEulerAngles(transFinal, transformTobeSubMapped[3], transformTobeSubMapped[4], transformTobeSubMapped[5], 
                                                              transformTobeSubMapped[0], transformTobeSubMapped[1], transformTobeSubMapped[2]);



                // transPredictionMapped=trans2Affine3f(transformTobeSubMapped);
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

            // ROS_WARN("MakeSubmap: cloudInfo.imuAvailable == true!");
            Eigen::Affine3f transBack = pcl::getTransformation(0, 0, 0, cloudInfo.imuRollInit, cloudInfo.imuPitchInit, cloudInfo.imuYawInit);
           
            Eigen::Affine3f transIncre = lastImuTransformation.inverse() * transBack;

            Eigen::Affine3f transTobe = trans2Affine3f(transformTobeSubMapped);
            Eigen::Affine3f transFinal = transTobe * transIncre;
            
            pcl::getTranslationAndEulerAngles(transFinal, transformTobeSubMapped[3], transformTobeSubMapped[4], transformTobeSubMapped[5], 
                                                          transformTobeSubMapped[0], transformTobeSubMapped[1], transformTobeSubMapped[2]);
            

            // transPredictionMapped=trans2Affine3f(transformTobeSubMapped);
            for(int i=0;i<6;++i){
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
                for(int i = 0; i < 6; ++i){
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
            
            for(int i = 0; i < 6; ++i){
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

        PointTypePose  point6d;
        point6d.x = 0;
        point6d.y = 0;
        point6d.z = 0;
        point6d.intensity = keyFrameID;
        point6d.roll = transformTobeSubMapped[0];
        point6d.pitch = transformTobeSubMapped[1];
        point6d.yaw = transformTobeSubMapped[2];
        point6d.time = timeLaserInfoCur;

        PointType  point3d;
        point3d.x = 0;
        point3d.y = 0;
        point3d.z = 0;
        point3d.intensity = keyFrameID;

        cloudKeyPoses3D->push_back(point3d);
        cloudKeyPoses6D->push_back(point6d);

        currentKeyFrame->submap_id = subMapId;
        currentKeyFrame->id_in_submap = curSubMapSize;
        currentKeyFrame->optimized_pose = point6d;
        
        calculateTranslation();
        
        point6d.x = transformCurFrame2Submap[3];
        point6d.y = transformCurFrame2Submap[4];
        point6d.z = transformCurFrame2Submap[5];
        point6d.intensity = keyFrameID;
        point6d.roll = transformCurFrame2Submap[0];
        point6d.pitch = transformCurFrame2Submap[1];
        point6d.yaw = transformCurFrame2Submap[2];
        point6d.time = timeLaserInfoCur;

        currentKeyFrame->relative_pose = point6d;

        //calculate bbx (local)
        get_cloud_bbx(currentKeyFrame->cloud_semantic, currentKeyFrame->local_bound);

        //calculate bbx (global)
        Eigen::Affine3f tran_map = pclPointToAffine3f(currentKeyFrame->optimized_pose);
        transform_bbx(currentKeyFrame->bound, tran_map);

    }


    void saveSubMap()
    {
        timeSubMapInfoStamp = timeLaserInfoStamp;
        subMapId++;

        PointTypePose  point6d;
        point6d.x=transformTobeSubMapped[3];
        point6d.y=transformTobeSubMapped[4];
        point6d.z=transformTobeSubMapped[5];
        point6d.intensity=subMapId;
        point6d.roll=transformTobeSubMapped[0];
        point6d.pitch=transformTobeSubMapped[1];
        point6d.yaw=transformTobeSubMapped[2];
        point6d.time=timeLaserInfoCur;

        subMapPose6D->points.push_back(point6d);      
            
        PointType  point3d;
        point3d.x=point6d.x;
        point3d.y=point6d.y;
        point3d.z=point6d.z;
        point3d.intensity=subMapId;

        subMapPose3D->points.push_back(point3d);  

        subMapPosesIndex3D[subMapId] = point3d;
        subMapPosesIndex6D[subMapId] = point6d;

        laserCloudCornerFromSubMap.clear();
        laserCloudSurfFromSubMap.clear();
        laserCloudCornerFromSubMapDS.clear();
        laserCloudSurfFromSubMapDS.clear();
        
        transformCurSubmap[0]=transformTobeSubMapped[0];
        transformCurSubmap[1]=transformTobeSubMapped[1];
        transformCurSubmap[2]=transformTobeSubMapped[2];
        transformCurSubmap[3]=transformTobeSubMapped[3];
        transformCurSubmap[4]=transformTobeSubMapped[4];
        transformCurSubmap[5]=transformTobeSubMapped[5];
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
        tf::StampedTransform trans_odom_to_lidar = tf::StampedTransform(t_odom_to_lidar, timeLaserInfoStamp, odometryFrame, "lidar_link");
        br.sendTransform(trans_odom_to_lidar);

        // Publish odometry for ROS (incremental)
        static bool lastIncreOdomPubFlag = false;
        static nav_msgs::Odometry laserOdomIncremental; // incremental odometry msg
        static Eigen::Affine3f increOdomAffine; // incremental odometry in affine
        if (lastIncreOdomPubFlag == false)
        {
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
        // ROS_INFO("Finshed  publishOdometry !");
    }

    void publishKeyFrameCloud()
    {
        // pubCloudRegisteredRaw;
        pcl::PointCloud<PointType>::Ptr thisCornerKeyFrame(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr thisSurfKeyFrame(new pcl::PointCloud<PointType>());
        pcl::copyPointCloud(*laserCloudCornerLastDS,  *thisCornerKeyFrame);
        pcl::copyPointCloud(*laserCloudSurfLastDS,    *thisSurfKeyFrame);
        
        *thisSurfKeyFrame += *thisCornerKeyFrame;

        Eigen::Affine3f transTobe = trans2Affine3f(transformTobeSubMapped);
        *thisSurfKeyFrame = *transformPointCloud(thisSurfKeyFrame, transTobe);

        publishCloud(&pubCloudRegisteredRaw, thisSurfKeyFrame, timeLaserInfoStamp, mapFrame);

        // pubKeyFramePath;

        // pubKeyFramePoseGlobal
        publishCloud(&pubKeyFramePoseGlobal, cloudKeyPoses3D, timeLaserInfoStamp, mapFrame);
    }


    void publishSubMapCloud()
    {
        pcl::PointCloud<PointXYZIL>::Ptr cloud_raw(new pcl::PointCloud<PointXYZIL>);
        currentSubMap->merge_feature_points(cloud_raw);
        publishCloud(&pubSubMapRaw, cloud_raw, timeLaserInfoStamp, lidarFrame);
        
        publishCloud(&pubSubMapId, subMapPose3D, timeLaserInfoStamp, mapFrame);       
    }

    /*****************************
     * @brief
     * @param input
     *****************************/
    void loopClosureThread() {
        ros::Rate rate(loopClosureFrequency);
        int processID = 0;
        EPSCGeneration epscGen;

        while (ros::ok()) {
        if (loopClosureEnableFlag == false) {
            ros::spinOnce();
            continue;
        }

        // rate.sleep();
        ros::spinOnce();

        if (processID <= keyFrameID) {
            keyframe_Ptr curKeyFramePtr;
            if (keyFrameInfo.find(processID) != keyFrameInfo.end()){
                curKeyFramePtr = keyFrameInfo[processID];
                processID++;
            }
            else
            continue;

            auto t1 = ros::Time::now();
            epscGen.loopDetection(
                curKeyFramePtr->cloud_corner, curKeyFramePtr->cloud_surface,
                curKeyFramePtr->cloud_semantic, curKeyFramePtr->cloud_static,
                curKeyFramePtr->init_pose);

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
            std::cout << "processID : " << processID << std::endl;
            std::cout << "loopKeyCur : " << loopKeyCur << std::endl;
            std::cout << "num_candidates: " << loopKeyPre.size() << std::endl;
            
            if (loopKeyPre.empty()) {
            ROS_WARN("loopKeyPre is empty !");
            continue;
            }
            for (int i = 0; i < loopKeyPre.size(); i++) {
            std::cout << "loopKeyPre [" << i << "]:" << loopKeyPre[i]
                        << std::endl;
            }
            
            int bestMatched = -1;
            if (detectLoopClosure(loopKeyCur, loopKeyPre, matched_init_transform,
                                bestMatched) == false)
            continue;

            visualizeLoopClosure();

            curKeyFramePtr->loop_container.push_back(bestMatched);
            
        }
        }
    }

    bool detectLoopClosure(int &loopKeyCur, vector<int> &loopKeyPre,
                            vector<Eigen::Affine3f> &matched_init_transform,
                            int &bestMatched) {
        pcl::PointCloud<PointXYZIL>::Ptr cureKeyframeCloud(
            new pcl::PointCloud<PointXYZIL>());
        pcl::PointCloud<PointXYZIL>::Ptr prevKeyframeCloud(
            new pcl::PointCloud<PointXYZIL>());

        auto thisCurId = keyFrameInfo.find(loopKeyCur);
        if (thisCurId != keyFrameInfo.end()) {
        loopKeyCur = (int)keyFrameInfo[loopKeyCur]->keyframe_id;
        *cureKeyframeCloud += *keyFrameInfo[loopKeyCur]->cloud_static;
        } else {
        loopKeyCur = -1;
        ROS_WARN("LoopKeyCur do not find !");
        return false;
        }

        std::cout << "matching..." << std::flush;
        auto t1 = ros::Time::now();

        // ICP Settings
        static pcl::IterativeClosestPoint<PointXYZIL, PointXYZIL> icp;
        icp.setMaxCorrespondenceDistance(historyKeyframeSearchRadius * 2);
        icp.setMaximumIterations(40);
        icp.setTransformationEpsilon(1e-6);
        icp.setEuclideanFitnessEpsilon(1e-6);
        icp.setRANSACIterations(0);

        // int bestMatched = -1;
        int bestID = -1;
        double bestScore = std::numeric_limits<double>::max();
        Eigen::Affine3f correctionLidarFrame;

        for (int i = 0; i < loopKeyPre.size(); i++) {
        // Align clouds
        pcl::PointCloud<PointXYZIL>::Ptr tmpCloud(
            new pcl::PointCloud<PointXYZIL>());
        *tmpCloud +=
            *transformPointCloud(cureKeyframeCloud, matched_init_transform[i]);
        icp.setInputSource(tmpCloud);

        auto thisPreId = keyFrameInfo.find(loopKeyPre[i]);
        if (thisPreId != keyFrameInfo.end()) {
            int PreID = (int)keyFrameInfo[loopKeyPre[i]]->keyframe_id;
            std::cout << "loopContainerHandler: loopKeyPre : " << PreID
                    << std::endl;

            prevKeyframeCloud->clear();
            *tmpCloud += *keyFrameInfo[loopKeyPre[i]]->cloud_static;
            icp.setInputTarget(prevKeyframeCloud);

            pcl::PointCloud<PointXYZIL>::Ptr unused_result(
                new pcl::PointCloud<PointXYZIL>());
            icp.align(*unused_result);

            double score = icp.getFitnessScore();
            if (icp.hasConverged() == false || score > bestScore) continue;
            bestScore = score;
            bestMatched = PreID;
            bestID = i;
            correctionLidarFrame = icp.getFinalTransformation();

        } else {
            bestMatched = -1;
            ROS_WARN("loopKeyPre do not find !");
        }
        }

        if (loopKeyCur == -1 || bestMatched == -1) return false;

        auto t2 = ros::Time::now();
        std::cout << " done" << std::endl;
        std::cout << "best_score: " << bestScore
                << "    time: " << (t2 - t1).toSec() << "[sec]" << std::endl;

        if (bestScore > historyKeyframeFitnessScore) {
        std::cout << "loop not found..." << std::endl;
        return false;
        }
        std::cout << "loop found!!" << std::endl;

        float X, Y, Z, ROLL, PITCH, YAW;
        Eigen::Affine3f tCorrect =
            correctionLidarFrame *
            matched_init_transform[bestID];  // pre-multiplying -> successive
                                            // rotation about a fixed frame
        pcl::getTranslationAndEulerAngles(tCorrect, X, Y, Z, ROLL, PITCH, YAW);
        gtsam::Pose3 pose = Pose3(Rot3::RzRyRx(ROLL, PITCH, YAW), Point3(X, Y, Z));
        gtsam::Vector Vector6(6);
        float noiseScore = 0.01;
        // float noiseScore = bestScore*0.01;
        Vector6 << noiseScore, noiseScore, noiseScore, noiseScore, noiseScore,
            noiseScore;
        noiseModel::Diagonal::shared_ptr constraintNoise =
            noiseModel::Diagonal::Variances(Vector6);

        // Add pose constraint
        // mtx.lock();
        loopIndexQueue.push_back(make_pair(loopKeyCur, bestMatched));
        loopPoseQueue.push_back(pose);
        loopNoiseQueue.push_back(constraintNoise);
        // mtx.unlock();

        // add loop constriant
        loopIndexContainer[loopKeyCur] = bestMatched;

        return true;
    }

    void visualizeLoopClosure() {
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

        for (auto it = loopIndexContainer.begin(); it != loopIndexContainer.end();
            ++it) {
        int key_cur = it->first;
        int key_pre = it->second;
        geometry_msgs::Point p;
        p.x = keyFrameInfo[key_cur]->init_pose(0, 3);
        p.y = keyFrameInfo[key_cur]->init_pose(1, 3);
        p.z = keyFrameInfo[key_cur]->init_pose(2, 3);
        markerNode.points.push_back(p);
        markerEdge.points.push_back(p);
        p.x = keyFrameInfo[key_pre]->init_pose(0, 3);
        p.y = keyFrameInfo[key_pre]->init_pose(1, 3);
        p.z = keyFrameInfo[key_pre]->init_pose(2, 3);
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

    ros::Publisher pubSubMapOdometryGlobal;
    
    ros::Publisher pubSubMapConstaintEdge;
    
    void allocateMemory() {}

    SubMapOptmizationNode() {
        subGPS = nh.subscribe<nav_msgs::Odometry>(gpsTopic, 2000, &SubMapOptmizationNode::gpsHandler, this, ros::TransportHints().tcpNoDelay());
        
        pubCloudMap = nh.advertise<sensor_msgs::PointCloud2>("lis_slam/mapping/map_global", 1);
        
        pubSubMapOdometryGlobal = nh.advertise<nav_msgs::Odometry>("lis_slam/make_submap/submap_odometry", 1);
        
        pubSubMapConstraintEdge = nh.advertise<visualization_msgs::MarkerArray>("lis_slam/mapping/submap_constraints", 1);

        allocateMemory();
    }

    void gpsHandler(const nav_msgs::Odometry &msgIn) {
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

int main(int argc, char **argv) {
    ros::init(argc, argv, "lis_slam");

    SubMapOdometryNode SOD;
    std::thread make_submap_process(&SubMapOdometryNode::makeSubMapThread, &SOD);
    std::thread loop_closure_process(&SubMapOdometryNode::loopClosureThread, &SOD);
    

    SubMapOptmizationNode SOP;
    std::thread visualize_map_process(&SubMapOptmizationNode::visualizeGlobalMapThread, &SOP);
    std::thread submap_optmization_process(&SubMapOptmizationNode::subMapOptmizationThread, &SOP);
    
    ROS_INFO("\033[1;32m----> SubMap Optmization Node Started.\033[0m");

    //   ros::MultiThreadedSpinner spinner(3);
    //   spinner.spin();
    ros::spin();

    make_submap_process.join();
    loop_closure_process.join();
    
    visualize_map_process.join();
    submap_optmization_process.join();

    return 0;
}