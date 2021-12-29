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
std::deque<int> subMapIndexQueue;
map<int, submap_Ptr> subMapInfo;

map<int, int> loopIndexContainer;  // from new to old
vector<pair<int, int>> loopIndexQueue;
vector<gtsam::Pose3> loopPoseQueue;
vector<gtsam::noiseModel::Diagonal::shared_ptr> loopNoiseQueue;


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
    pcl::PointCloud<PointXYZIL>::Ptr laserCloudCornerFromSubMapDS;
    pcl::PointCloud<PointXYZIL>::Ptr laserCloudSurfFromSubMapDS;

    pcl::KdTreeFLANN<PointXYZIL>::Ptr kdtreeFromKeyPoses6D;
    pcl::KdTreeFLANN<PointXYZIL>::Ptr kdtreeFromsubMapPose6D;

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
        laserCloudCornerFromSubMapDS.reset(new pcl::PointCloud<PointXYZIL>());
        laserCloudSurfFromSubMapDS.reset(new pcl::PointCloud<PointXYZIL>());

        kdtreeFromKeyPoses6D.reset(new pcl::KdTreeFLANN<PointXYZIL>());
        kdtreeFromsubMapPose6D.reset(new pcl::KdTreeFLANN<PointXYZIL>());

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
    
    SubMapOdometryNode() 
    {
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

    void semanticInfoHandler(const lis_slam::semantic_infoConstPtr &msgIn) 
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
                
                // void extractSurroundingKeyFrames(PointTypePose &cur_pose, int &target_submap_id, 
                //                      int target_keyframe_id = -1, bool using_target_id = false)
                int target_submap_id = -1;
                int target_keyframe_id = -1;
                bool using_target_id = false;
                PointTypePose  cur_pose = trans2PointTypePose(transformTobeSubMapped, keyFrameID, timeLaserInfoCur);
                extractSurroundingKeyFrames(cur_pose, target_submap_id, target_keyframe_id, using_target_id);
                
                auto it_ = subMapInfo.find(target_submap_id);
                if(it_ == subMapInfo.end())
                {
                    ROS_WARN("Dont extract Target Submap ID from Surrounding KeyFrames!");
                    continue;
                }

                extractSubMapCloud(currentKeyFrame, it_->second, cur_pose, false);
                scan2SubMapOptimization();

                saveKeyFrames();
                update_submap(currentSubMap, currentKeyFrame, 
                              local_map_radius, max_num_pts, kept_vertex_num,
                              last_frame_reliable_radius, map_based_dynamic_removal_on,
                              dynamic_removal_center_radius, dynamic_dist_thre_min,
                              dynamic_dist_thre_max, near_dist_thre);
                
                keyframe_Ptr tmpKeyFrame(new keyframe_t(*currentKeyFrame));
                keyFrameQueue.push_back(tmpKeyFrame);
                keyFrameInfo.insert(std::make_pair(keyFrameID, tmpKeyFrame));

                publishOdometry();
                publishKeyFrameCloud();

                // bool judge_new_submap(float &accu_tran, float &accu_rot, int &accu_frame,
                //                       float max_accu_tran = 30.0, 
                //                       float max_accu_rot = 90.0, 
                //                       int max_accu_frame = 150);
                //      << "Submap division criterion is: \n"
                //      << "1. Frame Number <=  max_accu_frame"
                //      << "2. Translation <=  max_accu_tran"
                //      << "3. Rotation <=  max_accu_rot"
                calculateTranslation();
                float accu_tran = std::max(transformCurFrame2Submap[3], transformCurFrame2Submap[4]); 
                float accu_rot = transformCurFrame2Submap[2];
                if(judge_new_submap(accu_tran, accu_rot, curSubMapSize, subMapTraMax, subMapYawMax, subMapFramesSize))
                {
                    ROS_INFO("Make %d submap  has %d  Frames !", subMapId, curSubMapSize);
                
                    saveSubMap();

                    submap_Ptr tmpSubMap(new submap_t(*currentSubMap));
                    subMapIndexQueue.push_back(subMapId);
                    subMapInfo.insert(std::make_pair(subMapId, tmpSubMap));

                    publishSubMapCloud();
              
                    fisrt_submap(currentSubMap, currentKeyFrame);

                    curSubMapSize = 1;
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
        
        //calculate bbx (local)
        get_cloud_bbx(currentKeyFrame->cloud_semantic, currentKeyFrame->local_bound);

        ROS_WARN("keyFrameID: %d ,keyFrameInfo Size: %d ",keyFrameID, keyFrameInfo.size());    
        
        return true;
    }
    
    void currentCloudInit()
    {
        pcl::PointCloud<PointXYZI>::Ptr thisFrameCloud(new pcl::PointCloud<PointXYZI>());
        pcl::PointCloud<PointXYZI>::Ptr thisFrameCloudDS(new pcl::PointCloud<PointXYZI>());
        
        pcl::copyPointCloud(*currentKeyFrame->cloud_static, *thisFrameCloud);
        pcl::copyPointCloud(*currentKeyFrame->cloud_static_down, *thisFrameCloudDS);
        
        for (int i = 0; i < (int)thisFrameCloud->points.size(); i++) 
        {
            auto label = thisFrameCloud->points[i].label;
            
            if (UsingLableMap[label] == 40) {
                laserCloudSurfLast->points.push_back(thisFrameCloud->points[i]);
            } else if (UsingLableMap[label] == 81) {
                laserCloudCornerLast->points.push_back(thisFrameCloud->points[i]);
            }
        }

        for (int i = 0; i < (int)thisFrameCloudDS->points.size(); i++) 
        {
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
            if (lastImuPreTransAvailable == false) {
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

        PointTypePose  point6d = trans2PointTypePose(transformTobeSubMapped, keyFrameID, timeLaserInfoCur);
        PointType  point3d = trans2PointType(transformTobeSubMapped, keyFrameID);

        keyFramePoses3D->push_back(point3d);
        keyFramePoses6D->push_back(point6d);

        keyFramePosesIndex6D[keyFrameID] = point6d;
        keyFramePosesIndex3D[keyFrameID] = point3d;

        currentKeyFrame->submap_id = subMapId;
        currentKeyFrame->id_in_submap = curSubMapSize;
        currentKeyFrame->optimized_pose = point6d;
        
        calculateTranslation();
        point6d = trans2PointTypePose(transformCurFrame2Submap, keyFrameID, timeLaserInfoCur);
        currentKeyFrame->relative_pose = point6d;

        //calculate bbx (global)
        Eigen::Affine3f tran_map = pclPointToAffine3f(currentKeyFrame->optimized_pose);
        transform_bbx(currentKeyFrame->local_bound, currentKeyFrame->bound, tran_map);

    }


    void saveSubMap()
    {
        timeSubMapInfoStamp = timeLaserInfoStamp;
        subMapId++;

        PointTypePose  point6d = trans2PointTypePose(transformTobeSubMapped, subMapId, timeSubMapInfoStamp);
        subMapPose6D->points.push_back(point6d);      
            
        PointType  point3d = trans2PointType(transformTobeSubMapped, subMapId);
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
        publishCloud(&pubKeyFramePoseGlobal, keyFramePoses3D, timeLaserInfoStamp, mapFrame);
    }


    void publishSubMapCloud()
    {
        pcl::PointCloud<PointXYZIL>::Ptr cloud_raw(new pcl::PointCloud<PointXYZIL>);
        currentSubMap->merge_feature_points(cloud_raw);
        publishCloud(&pubSubMapRaw, cloud_raw, timeLaserInfoStamp, lidarFrame);
        
        publishCloud(&pubSubMapId, subMapPose3D, timeLaserInfoStamp, mapFrame);       
    }


    void extractSurroundingKeyFrames(PointTypePose &cur_pose, int &target_submap_id, 
                                     int target_keyframe_id = -1, bool using_target_id = false)
    {
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
        }
             
    }

    void extractSubMapCloud(keyframe_Ptr &cur_keyframe, submap_Ptr &cur_submap, 
                            PointTypePose cur_pose, bool using_keyframe_pose = false)
    {
        if(using_keyframe_pose)
        {
            Eigen::Affine3f tran_map = pclPointToAffine3f(cur_keyframe->optimized_pose);
            transform_bbx(cur_keyframe->local_bound, cur_keyframe->bound, tran_map);
        }
        else
        {
            Eigen::Affine3f tran_map = pclPointToAffine3f(cur_pose);
            transform_bbx(cur_keyframe->local_bound, cur_keyframe->bound, tran_map);
        }


        Eigen::Affine3f tran_map = pclPointToAffine3f(cur_submap->submap_pose_6D_optimized);
        transform_bbx(cur_submap->local_bound, cur_submap->bound, tran_map);

        bounds_t bbx_intersection;
        get_intersection_bbx(cur_keyframe->bound, cur_submap->bound, bbx_intersection, 2.0);

        pcl::PointCloud<PointXYZIL>::Ptr cloud_temp(new pcl::PointCloud<PointXYZIL>);
        // Use the intersection bounding box to filter the outlier points
        bbx_filter(cloud_temp, bbx_intersection);
        
        for (int i = 0; i < (int)cloud_temp->points.size(); i++) {
            auto label = cloud_temp->points[i].label;
            
            if (UsingLableMap[label] == 40) {
                laserCloudSurfFromSubMap->points.push_back(cloud_temp->points[i]);
            } else if (UsingLableMap[label] == 81) {
                laserCloudCornerFromSubMap->points.push_back(cloud_temp->points[i]);
            }
        }
    }



    void scan2SubMapOptimization()
    {
        if (laserCloudCornerLastDSNum > edgeFeatureMinValidNum && laserCloudSurfLastDSNum > surfFeatureMinValidNum)
        {
            // ROS_INFO("laserCloudCornerLastDSNum: %d laserCloudSurfLastDSNum: %d .", laserCloudCornerLastDSNum, laserCloudSurfLastDSNum);
            
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

        // #pragma omp for
        #pragma omp parallel for num_threads(numberOfCores)
        for (int i = 0; i < laserCloudCornerLastDSNum; i++)
        {
            PointXYZIL pointOri, pointSel, coeff;
            std::vector<int> pointSearchInd;
            std::vector<float> pointSearchSqDis;

            pointOri = laserCloudCornerLastDS->points[i];
            pointAssociateToSubMap(&pointOri, &pointSel);
            kdtreeCornerFromSubMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);

            cv::Mat matA1(3, 3, CV_32F, cv::Scalar::all(0));
            cv::Mat matD1(1, 3, CV_32F, cv::Scalar::all(0));
            cv::Mat matV1(3, 3, CV_32F, cv::Scalar::all(0));

            if (pointSearchSqDis[4] < 1.0) {
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

                    coeff.x = s * la;
                    coeff.y = s * lb;
                    coeff.z = s * lc;
                    coeff.intensity = s * ld2;

                    if (s > 0.1) {
                        laserCloudOriCornerVec[i] = pointOri;
                        coeffSelCornerVec[i] = coeff;
                        laserCloudOriCornerFlag[i] = true;
                    }
                }
            }
        }
    }



    void surfOptimization()
    {
        updatePointAssociateToSubMap();

        // #pragma omp for
        #pragma omp parallel for num_threads(numberOfCores)
        for (int i = 0; i < laserCloudSurfLastDSNum; i++)
        {
            PointXYZIL pointOri, pointSel, coeff;
            std::vector<int> pointSearchInd;
            std::vector<float> pointSearchSqDis;

            pointOri = laserCloudSurfLastDS->points[i];
            pointAssociateToSubMap(&pointOri, &pointSel); 
            kdtreeSurfFromSubMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);

            Eigen::Matrix<float, 5, 3> matA0;
            Eigen::Matrix<float, 5, 1> matB0;
            Eigen::Vector3f matX0;

            matA0.setZero();
            matB0.fill(-1);
            matX0.setZero();

            if (pointSearchSqDis[4] < 1.0) {
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

                    coeff.x = s * pa;
                    coeff.y = s * pb;
                    coeff.z = s * pc;
                    coeff.intensity = s * pd2;

                    if (s > 0.1) {
                        laserCloudOriSurfVec[i] = pointOri;
                        coeffSelSurfVec[i] = coeff;
                        laserCloudOriSurfFlag[i] = true;
                    }
                }
            }
        }
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

        if (processID <= keyFrameID) 
        {
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
                *tmpCloud += *keyFrameInfo[loopKeyPre[i]]->cloud_static;
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
        loopIndexQueue.push_back(make_pair(loopKeyCur, bestMatched));
        loopPoseQueue.push_back(pose);
        loopNoiseQueue.push_back(constraintNoise);
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

    SubMapOptmizationNode() 
    {
        subGPS = nh.subscribe<nav_msgs::Odometry>(gpsTopic, 2000, &SubMapOptmizationNode::gpsHandler, this, ros::TransportHints().tcpNoDelay());
        
        pubCloudMap = nh.advertise<sensor_msgs::PointCloud2>("lis_slam/mapping/map_global", 1);
        
        pubSubMapOdometryGlobal = nh.advertise<nav_msgs::Odometry>("lis_slam/make_submap/submap_odometry", 1);
        
        pubSubMapConstraintEdge = nh.advertise<visualization_msgs::MarkerArray>("lis_slam/mapping/submap_constraints", 1);

        allocateMemory();
    }

    void gpsHandler(const nav_msgs::Odometry &msgIn) 
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
    
    ROS_INFO("\033[1;32m----> SubMap Optmization Node Started.\033[0m");

    //   ros::MultiThreadedSpinner spinner(3);
    //   spinner.spin();
    ros::spin();

    make_submap_process.join();
    // loop_closure_process.join();
    
    // visualize_map_process.join();
    // submap_optmization_process.join();

    return 0;
}