
//This code partly draws on  lio_sam
// Author of  EPSC-LOAM : QZ Wang  
// Email wangqingzhi27@outlook.com

#include "utility.h"
#include "lis_slam/cloud_info.h"
#include "lis_slam/loop_container.h"
#include "lis_slam/submap.h"
#include "lis_slam/loop_info.h"

#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Symbol.h>

#include <gtsam/nonlinear/ISAM2.h>

using namespace gtsam;

using symbol_shorthand::X; // Pose3 (x,y,z,r,p,y)
using symbol_shorthand::V; // Vel   (xdot,ydot,zdot)
using symbol_shorthand::B; // Bias  (ax,ay,az,gx,gy,gz)
using symbol_shorthand::G; // GPS pose


bool UsingContainer=true;

/*
    * A point cloud type that has 6D pose info ([x,y,z,roll,pitch,yaw] intensity is time stamp)
    */
struct PointXYZIRPYT
{
    PCL_ADD_POINT4D
    PCL_ADD_INTENSITY;                  // preferred way of adding a XYZ+padding
    float roll;
    float pitch;
    float yaw;
    double time;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW   // make sure our new allocators are aligned
} EIGEN_ALIGN16;                    // enforce SSE padding for correct memory alignment

POINT_CLOUD_REGISTER_POINT_STRUCT (PointXYZIRPYT,
                                   (float, x, x) (float, y, y)
                                   (float, z, z) (float, intensity, intensity)
                                   (float, roll, roll) (float, pitch, pitch) (float, yaw, yaw)
                                   (double, time, time))

typedef PointXYZIRPYT  PointTypePose;

class mapOptimization : public ParamServer
{

public:

    // gtsam
    NonlinearFactorGraph gtSAMgraph;
    Values initialEstimate;
    Values optimizedEstimate;
    ISAM2 *isam;
    Values isamCurrentEstimate;
    Eigen::MatrixXd poseCovariance;

    ros::Publisher pubLaserCloudSurround;
    ros::Publisher pubSubMapOdometryGlobal;

    ros::Publisher pubKeyPoses;
    ros::Publisher pubPath;

    ros::Publisher pubCloudRegisteredRaw;
    ros::Publisher pubLoopConstraintEdge;
    
    ros::Publisher pubSubMapConstraintEdge;

    ros::Publisher pubSubMapInfo;

    ros::Subscriber subCloud;
    ros::Subscriber subGPS;
    ros::Subscriber subLoop;
    ros::Subscriber subLoopContainer;

    std::deque<nav_msgs::Odometry> gpsQueue;

    std::deque<lis_slam::submap> subMapQueue;
    lis_slam::submap submapInfo;
    lis_slam::submap  submapInfoPub;

    vector<pcl::PointCloud<PointType>::Ptr> cornerCloudSubMap;
    vector<pcl::PointCloud<PointType>::Ptr> surfCloudSubMap;

    pcl::PointCloud<PointType>::Ptr cloudSubMapPoses3D;
    pcl::PointCloud<PointTypePose>::Ptr cloudSubMapPoses6D;

    pcl::PointCloud<PointTypePose>::Ptr copyCloudSubMapPoses6D;

    map<int,PointType> SubMapPoses3D;
    map<int,PointTypePose> SubMapPoses6D;

    pcl::PointCloud<PointType>::Ptr laserCloudCornerLast; // corner feature set from odoOptimization
    pcl::PointCloud<PointType>::Ptr laserCloudSurfLast; // surf feature set from odoOptimization
    pcl::PointCloud<PointType>::Ptr laserCloudCornerLastDS; // downsampled corner featuer set from odoOptimization
    pcl::PointCloud<PointType>::Ptr laserCloudSurfLastDS; // downsampled surf featuer set from odoOptimization

    pcl::PointCloud<PointType>::Ptr laserCloudOri;
    pcl::PointCloud<PointType>::Ptr coeffSel;

    std::vector<PointType> laserCloudOriCornerVec; // corner point holder for parallel computation
    std::vector<PointType> coeffSelCornerVec;
    std::vector<bool> laserCloudOriCornerFlag;
    std::vector<PointType> laserCloudOriSurfVec; // surf point holder for parallel computation
    std::vector<PointType> coeffSelSurfVec;
    std::vector<bool> laserCloudOriSurfFlag;

    pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMap;
    pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMap;
    pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMapDS;
    pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMapDS;

    pcl::KdTreeFLANN<PointType>::Ptr kdtreeCornerFromMap;
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurfFromMap;

    pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurroundingKeyPoses;
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeHistoryKeyPoses;    

    pcl::VoxelGrid<PointType> downSizeFilterSubMapCorner;
    pcl::VoxelGrid<PointType> downSizeFilterSubMapSurf;
    pcl::VoxelGrid<PointType> downSizeFilterSubMap;

    
    ros::Time timeLaserInfoStamp;
    double timeLaserInfoCur;

    bool subMapFirstFlag=true;

    float transformTobeMapped[6];
    float transformBefMapped[6];
    Eigen::Affine3f transPredictionMapped;

    std::mutex mtx;
    std::mutex mtxLoopInfo;

    bool isDegenerate = false;
    Eigen::Matrix<float, 6, 6> matP;

    int laserCloudCornerFromMapDSNum = 0;
    int laserCloudSurfFromMapDSNum = 0;
    int laserCloudCornerLastDSNum = 0;
    int laserCloudSurfLastDSNum = 0;

    bool aLoopIsClosed = false;
    map<int, int> loopIndexContainer; // from new to old
    vector<pair<int, int>> loopIndexQueue;
    vector<gtsam::Pose3> loopPoseQueue;
    vector<gtsam::noiseModel::Diagonal::shared_ptr> loopNoiseQueue;
    // deque<std_msgs::Float64MultiArray> loopInfoVec;
    deque<lis_slam::loop_info> loopInfoVec;
    deque<lis_slam::loop_container> loopContainerVec;


    vector<int> subMapContainer; // from new to old
    vector<pair<int, int>>subMapIndexContainerAll; // from new to old
    vector<pair<int, int>> subMapIndexQueue;
    vector<gtsam::Pose3> subMapPoseQueue;
    vector<gtsam::noiseModel::Diagonal::shared_ptr> subMapNoiseQueue;


    Eigen::Affine3f transPointAssociateToMap;
    Eigen::Affine3f incrementalOdometryAffineFront;
    Eigen::Affine3f incrementalOdometryAffineBack;

    nav_msgs::Path globalPath;

    bool icpTest=false;

    ros::Publisher pubIcpOptmizationCloud;




    mapOptimization()
    {
        ISAM2Params parameters;
        parameters.relinearizeThreshold = 0.1;
        parameters.relinearizeSkip = 1;
        isam = new ISAM2(parameters);

        pubKeyPoses                 = nh.advertise<sensor_msgs::PointCloud2>("lis_slam/mapping/trajectory", 1);
        pubLaserCloudSurround       = nh.advertise<sensor_msgs::PointCloud2>("lis_slam/mapping/map_global", 1);
        pubSubMapOdometryGlobal      = nh.advertise<nav_msgs::Odometry> ("lis_slam/mapping/odometry", 1);
        pubPath                     = nh.advertise<nav_msgs::Path>("lis_slam/mapping/path", 1);

        subCloud = nh.subscribe<lis_slam::submap>("lis_slam/submap/submap_info", 10, &mapOptimization::laserCloudInfoHandler, this, ros::TransportHints().tcpNoDelay());
        subGPS   = nh.subscribe<nav_msgs::Odometry> (gpsTopic, 200, &mapOptimization::gpsHandler, this, ros::TransportHints().tcpNoDelay());
        
        subLoop  = nh.subscribe<lis_slam::loop_info>("lio_loop/loop_closure_detection", 10, &mapOptimization::loopInfoHandler, this, ros::TransportHints().tcpNoDelay());
        subLoopContainer   = nh.subscribe<lis_slam::loop_container>("lio_loop/loop_container", 10, &mapOptimization::loopContainerHandler, this, ros::TransportHints().tcpNoDelay());

        pubCloudRegisteredRaw = nh.advertise<sensor_msgs::PointCloud2>("lis_slam/mapping/cloud_registered_raw", 1);

        pubLoopConstraintEdge = nh.advertise<visualization_msgs::MarkerArray>("/lis_slam/mapping/loop_closure_constraints", 1);
        pubSubMapConstraintEdge = nh.advertise<visualization_msgs::MarkerArray>("/lis_slam/mapping/submap_constraints", 1);
        
        pubSubMapInfo = nh.advertise<lis_slam::submap>("lis_slam/mapping/submap_info", 1);

        pubIcpOptmizationCloud = nh.advertise<sensor_msgs::PointCloud2>("lis_slam/mapping/icp_optmization_cloud", 1);

        downSizeFilterSubMapCorner.setLeafSize(subMapCornerLeafSize, subMapCornerLeafSize, subMapCornerLeafSize);
        downSizeFilterSubMapSurf.setLeafSize(subMapSurfLeafSize, subMapSurfLeafSize, subMapSurfLeafSize);
        downSizeFilterSubMap.setLeafSize(subMapLeafSize, subMapLeafSize, subMapLeafSize);


        allocateMemory();

    }

    void allocateMemory()
    {

        cloudSubMapPoses3D.reset(new pcl::PointCloud<PointType>());
        cloudSubMapPoses6D.reset(new pcl::PointCloud<PointTypePose>());

        copyCloudSubMapPoses6D.reset(new pcl::PointCloud<PointTypePose>());

        kdtreeSurroundingKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());
        kdtreeHistoryKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());


        laserCloudCornerLast.reset(new pcl::PointCloud<PointType>()); // corner feature set from odoOptimization
        laserCloudSurfLast.reset(new pcl::PointCloud<PointType>()); // surf feature set from odoOptimization
        laserCloudCornerLastDS.reset(new pcl::PointCloud<PointType>()); // downsampled corner featuer set from odoOptimization
        laserCloudSurfLastDS.reset(new pcl::PointCloud<PointType>()); // downsampled surf featuer set from odoOptimization

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

        laserCloudCornerFromMap.reset(new pcl::PointCloud<PointType>());
        laserCloudSurfFromMap.reset(new pcl::PointCloud<PointType>());
        laserCloudCornerFromMapDS.reset(new pcl::PointCloud<PointType>());
        laserCloudSurfFromMapDS.reset(new pcl::PointCloud<PointType>());

        kdtreeCornerFromMap.reset(new pcl::KdTreeFLANN<PointType>());
        kdtreeSurfFromMap.reset(new pcl::KdTreeFLANN<PointType>());

        for (int i = 0; i < 6; ++i){
            transformTobeMapped[i] = 0;
            transformBefMapped[i] = 0;
        }

        matP.setZero();



    }

    void laserCloudInfoHandler(const lis_slam::submapConstPtr& msgIn)
    {

        std::lock_guard<std::mutex> lock(mtx);
        subMapQueue.push_back(*msgIn);

        while (subMapQueue.size() > 6)
            subMapQueue.pop_front();

    }



    void gpsHandler(const nav_msgs::Odometry::ConstPtr& gpsMsg)
    {
        gpsQueue.push_back(*gpsMsg);

    }








    void pointAssociateToMap(PointType const * const pi, PointType * const po)
    {
        po->x = transPointAssociateToMap(0,0) * pi->x + transPointAssociateToMap(0,1) * pi->y + transPointAssociateToMap(0,2) * pi->z + transPointAssociateToMap(0,3);
        po->y = transPointAssociateToMap(1,0) * pi->x + transPointAssociateToMap(1,1) * pi->y + transPointAssociateToMap(1,2) * pi->z + transPointAssociateToMap(1,3);
        po->z = transPointAssociateToMap(2,0) * pi->x + transPointAssociateToMap(2,1) * pi->y + transPointAssociateToMap(2,2) * pi->z + transPointAssociateToMap(2,3);
        po->intensity = pi->intensity;
    }

    pcl::PointCloud<PointType>::Ptr transformPointCloud(pcl::PointCloud<PointType>::Ptr cloudIn, PointTypePose* transformIn)
    {
        pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

        PointType *pointFrom;

        int cloudSize = cloudIn->size();
        cloudOut->resize(cloudSize);

        Eigen::Affine3f transCur = pcl::getTransformation(transformIn->x, transformIn->y, transformIn->z, transformIn->roll, transformIn->pitch, transformIn->yaw);
        
        // #pragma omp parallel for num_threads(numberOfCores)
        #pragma omp for
        for (int i = 0; i < cloudSize; ++i)
        {
            pointFrom = &cloudIn->points[i];
            cloudOut->points[i].x = transCur(0,0) * pointFrom->x + transCur(0,1) * pointFrom->y + transCur(0,2) * pointFrom->z + transCur(0,3);
            cloudOut->points[i].y = transCur(1,0) * pointFrom->x + transCur(1,1) * pointFrom->y + transCur(1,2) * pointFrom->z + transCur(1,3);
            cloudOut->points[i].z = transCur(2,0) * pointFrom->x + transCur(2,1) * pointFrom->y + transCur(2,2) * pointFrom->z + transCur(2,3);
            cloudOut->points[i].intensity = pointFrom->intensity;
        }
        return cloudOut;
    }


    gtsam::Pose3 pclPointTogtsamPose3(PointTypePose thisPoint)
    {
        return gtsam::Pose3(gtsam::Rot3::RzRyRx(double(thisPoint.roll), double(thisPoint.pitch), double(thisPoint.yaw)),
                                  gtsam::Point3(double(thisPoint.x),    double(thisPoint.y),     double(thisPoint.z)));
    }

    gtsam::Pose3 trans2gtsamPose(float transformIn[])
    {
        return gtsam::Pose3(gtsam::Rot3::RzRyRx(transformIn[0], transformIn[1], transformIn[2]), 
                                  gtsam::Point3(transformIn[3], transformIn[4], transformIn[5]));
    }

    Eigen::Affine3f pclPointToAffine3f(PointTypePose thisPoint)
    { 
        return pcl::getTransformation(thisPoint.x, thisPoint.y, thisPoint.z, thisPoint.roll, thisPoint.pitch, thisPoint.yaw);
    }

    Eigen::Affine3f trans2Affine3f(float transformIn[])
    {
        return pcl::getTransformation(transformIn[3], transformIn[4], transformIn[5], transformIn[0], transformIn[1], transformIn[2]);
    }

    PointTypePose trans2PointTypePose(float transformIn[])
    {
        PointTypePose thisPose6D;
        thisPose6D.x = transformIn[3];
        thisPose6D.y = transformIn[4];
        thisPose6D.z = transformIn[5];
        thisPose6D.roll  = transformIn[0];
        thisPose6D.pitch = transformIn[1];
        thisPose6D.yaw   = transformIn[2];
        return thisPose6D;
    }











    void visualizeGlobalMapThread()
    {
        ros::Rate rate(0.1);
        while (ros::ok()){
            // ROS_WARN("visualizeGlobalMapThread !");
            publishGlobalMap1();
            // ros::spinOnce();
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
        pcl::PointCloud<PointType>::Ptr globalCornerCloud(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalCornerCloudDS(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalSurfCloud(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalSurfCloudDS(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalMapCloud(new pcl::PointCloud<PointType>());

        for (int i = 0; i < (int)cloudSubMapPoses3D->size(); i++) {
            *globalCornerCloud += *transformPointCloud(cornerCloudSubMap[i],  &cloudSubMapPoses6D->points[i]);
            *globalSurfCloud   += *transformPointCloud(surfCloudSubMap[i],    &cloudSubMapPoses6D->points[i]);
            cout << "\r" << std::flush << "Processing feature cloud " << i << " of " << cloudSubMapPoses6D->size() << " ...";
        }

        // down-sample and save corner cloud
        downSizeFilterSubMapCorner.setInputCloud(globalCornerCloud);
        downSizeFilterSubMapCorner.filter(*globalCornerCloudDS);
        pcl::io::savePCDFileASCII(savePCDDirectory + "cloudCorner.pcd", *globalCornerCloudDS);
        // down-sample and save surf cloud
        downSizeFilterSubMapSurf.setInputCloud(globalSurfCloud);
        downSizeFilterSubMapSurf.filter(*globalSurfCloudDS);
        pcl::io::savePCDFileASCII(savePCDDirectory + "cloudSurf.pcd", *globalSurfCloudDS);
        // down-sample and save global point cloud map
        // *globalMapCloud += *globalCornerCloud;
        // *globalMapCloud += *globalSurfCloud;
        *globalMapCloud += *globalCornerCloudDS;
        *globalMapCloud += *globalSurfCloudDS;
        pcl::io::savePCDFileASCII(savePCDDirectory + "cloudGlobal.pcd", *globalMapCloud);
        cout << "****************************************************" << endl;
        cout << "Saving map to pcd files completed" << endl;
    }

    void publishGlobalMap()
    {
        if (pubLaserCloudSurround.getNumSubscribers() == 0)
            return;

        if (cloudSubMapPoses3D->points.empty() == true)
            return;

        pcl::KdTreeFLANN<PointType>::Ptr kdtreeGlobalMap(new pcl::KdTreeFLANN<PointType>());;
        pcl::PointCloud<PointType>::Ptr globalMapKeyPoses(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalMapKeyPosesDS(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalMapKeyFrames(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalMapKeyFramesDS(new pcl::PointCloud<PointType>());

        // kd-tree to find near key frames to visualize
        std::vector<int> pointSearchIndGlobalMap;
        std::vector<float> pointSearchSqDisGlobalMap;
        // search near key frames to visualize
        // mtx.lock();
        kdtreeGlobalMap->setInputCloud(cloudSubMapPoses3D);
        kdtreeGlobalMap->radiusSearch(cloudSubMapPoses3D->back(), globalMapVisualizationSearchRadius, pointSearchIndGlobalMap, pointSearchSqDisGlobalMap, 0);
        // mtx.unlock();

        for (int i = 0; i < (int)pointSearchIndGlobalMap.size(); ++i)
            globalMapKeyPoses->push_back(cloudSubMapPoses3D->points[pointSearchIndGlobalMap[i]]);
        // downsample near selected key frames
        pcl::VoxelGrid<PointType> downSizeFilterGlobalMapKeyPoses; // for global map visualization
        downSizeFilterGlobalMapKeyPoses.setLeafSize(globalMapVisualizationPoseDensity, globalMapVisualizationPoseDensity, globalMapVisualizationPoseDensity); // for global map visualization
        downSizeFilterGlobalMapKeyPoses.setInputCloud(globalMapKeyPoses);
        downSizeFilterGlobalMapKeyPoses.filter(*globalMapKeyPosesDS);

        // extract visualized and downsampled key frames
        for (int i = 0; i < (int)globalMapKeyPosesDS->size(); ++i){
            if (pointDistance(globalMapKeyPosesDS->points[i], cloudSubMapPoses3D->back()) > globalMapVisualizationSearchRadius)
                continue;
            int thisKeyInd = (int)globalMapKeyPosesDS->points[i].intensity;

            *globalMapKeyFrames += *transformPointCloud(cornerCloudSubMap[thisKeyInd],  &cloudSubMapPoses6D->points[thisKeyInd]);
            *globalMapKeyFrames += *transformPointCloud(cornerCloudSubMap[thisKeyInd],    &cloudSubMapPoses6D->points[thisKeyInd]);
        }
        // downsample visualized points
        pcl::VoxelGrid<PointType> downSizeFilterGlobalMapKeyFrames; // for global map visualization
        downSizeFilterGlobalMapKeyFrames.setLeafSize(globalMapVisualizationLeafSize, globalMapVisualizationLeafSize, globalMapVisualizationLeafSize); // for global map visualization
        downSizeFilterGlobalMapKeyFrames.setInputCloud(globalMapKeyFrames);
        downSizeFilterGlobalMapKeyFrames.filter(*globalMapKeyFramesDS);
        publishCloud(&pubLaserCloudSurround, globalMapKeyFramesDS, timeLaserInfoStamp, odometryFrame);
    }


    void publishGlobalMap1()
    {
        if (pubLaserCloudSurround.getNumSubscribers() == 0)
            return;

        if (cloudSubMapPoses3D->points.empty() == true)
            return;

        pcl::PointCloud<PointTypePose>::Ptr globalMapKeyPoses(new pcl::PointCloud<PointTypePose>());
        *globalMapKeyPoses=*cloudSubMapPoses6D;

        pcl::PointCloud<PointType>::Ptr globalCornerCloud(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalCornerCloudDS(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalSurfCloud(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalSurfCloudDS(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalMapCloud(new pcl::PointCloud<PointType>());

        int subMapSize=globalMapKeyPoses->size();
        for (int i = 0; i < subMapSize; i++) {
            *globalCornerCloud += *transformPointCloud(cornerCloudSubMap[i],  &globalMapKeyPoses->points[i]);
            *globalSurfCloud   += *transformPointCloud(surfCloudSubMap[i],    &globalMapKeyPoses->points[i]);
            cout << "\r" << std::flush << "Processing feature cloud " << i << " of " << globalMapKeyPoses->size() << " ...";
        }

        // // down-sample and save corner cloud
        // downSizeFilterSubMapCorner.setInputCloud(globalCornerCloud);
        // downSizeFilterSubMapCorner.filter(*globalCornerCloudDS);
        // // down-sample and save surf cloud
        // downSizeFilterSubMapSurf.setInputCloud(globalSurfCloud);
        // downSizeFilterSubMapSurf.filter(*globalSurfCloudDS);
        
        // down-sample and save global point cloud map
        *globalMapCloud += *globalCornerCloud;
        *globalMapCloud += *globalSurfCloud;
        // *globalMapCloud += *globalCornerCloudDS;
        // *globalMapCloud += *globalSurfCloudDS;

        publishCloud(&pubLaserCloudSurround, globalMapCloud, timeLaserInfoStamp, mapFrame);
    }







    void loopInfoHandler(const lis_slam::loop_info::ConstPtr& loopMsg)
    {
        std::lock_guard<std::mutex> lock(mtxLoopInfo);

        loopInfoVec.push_back(*loopMsg);
        

        // if (loopClosureEnableFlag == false)
        //     return;

        // while(loopInfoVec.empty()==false)
        // {
        //     if (detectLoopClosureExternal() == false)
        //         continue;

        //     visualizeLoopClosure();
        // }
        
    }

    
    void loopContainerHandler(const lis_slam::loop_container::ConstPtr& loopMsg)
    {
        std::lock_guard<std::mutex> lock(mtxLoopInfo);

        loopContainerVec.push_back(*loopMsg);

        //test
        // while(loopContainerVec.empty()==false){

        //         int loopKeyCur = -1;
        //         int loopKeyPre = -1;

        //         ROS_WARN("loopContainerHandler  test !");
        //         lis_slam::loop_container loopInfo;
        //         loopInfo=loopContainerVec.front();

        //         auto thisCurId=SubMapPoses6D.find(loopInfo.loopKeyCur);
        //         if(thisCurId!=SubMapPoses6D.end())
        //         {
        //             loopKeyCur=(int)thisCurId->second.intensity;
        //             std::cout<<"loopContainerHandler: loopKeyCur : "<< loopKeyCur<<std::endl;
        //         }else
        //         {
        //             loopKeyCur=-1;
        //             ROS_WARN("LoopKeyCur do not find !");
        //         }
        //         std::cout<<"loopContainerHandler: loopKeyPreSize : "<< loopInfo.loopKeyPre.size()<<std::endl;
        //         for(int i=0;i<loopInfo.loopKeyPre.size();i++){
        //             auto thisPreId=SubMapPoses6D.find(loopInfo.loopKeyPre[i]);
        //             if(thisPreId!=SubMapPoses6D.end())
        //             {
        //                 loopKeyPre=(int)thisPreId->second.intensity;
        //                 std::cout<<"loopContainerHandler: loopKeyPre : "<< loopKeyPre<<std::endl;
        //             }else
        //             {
        //                 loopKeyPre=-1;
        //                 ROS_WARN("LoopKeyPre do not find !");
        //             }
        //         }

        //         loopContainerVec.pop_front();

        //     }

    
        
    }



    void loopClosureThread()
    {
        ros::Rate rate(loopClosureFrequency);

        while (ros::ok()){

            if (loopClosureEnableFlag == false){
                ros::spinOnce();
                continue;
            }
            

            if(UsingContainer){

                    while(loopContainerVec.empty()==false){
                        ROS_WARN("loopClosureThread :  detectLoopClosureExternalContainer !");
                        if (detectLoopClosureExternalContainer() == false)
                                continue;
                    }

                    visualizeLoopClosure();
                        
            }else{
                    while(loopInfoVec.empty()==false) {
                            ROS_WARN("loopClosureThread :  detectLoopClosureExternal !");
                            if (detectLoopClosureExternal() == false)
                                continue;
                    }
                     visualizeLoopClosure();
            }

             rate.sleep();
            ros::spinOnce();
        }


    }



    bool detectLoopClosureExternalContainer(){
                
                int loopKeyCur = -1;
                int loopKeyPre = -1;
                lis_slam::loop_container loopInfo;
                loopInfo=loopContainerVec.front();

                

                *copyCloudSubMapPoses6D = *cloudSubMapPoses6D;

                pcl::PointCloud<PointType>::Ptr cureKeyframeCloud(new pcl::PointCloud<PointType>());
                pcl::PointCloud<PointType>::Ptr prevKeyframeCloud(new pcl::PointCloud<PointType>());

                auto thisCurId=SubMapPoses6D.find(loopInfo.loopKeyCur);
                if(thisCurId!=SubMapPoses6D.end())
                {
                    loopKeyCur=(int)thisCurId->second.intensity;
                    std::cout<<"loopContainerHandler: loopKeyCur : "<< loopKeyCur<<std::endl;

                    *cureKeyframeCloud+=*transformPointCloud(cornerCloudSubMap[loopKeyCur], &copyCloudSubMapPoses6D->points[loopKeyCur]);
                    *cureKeyframeCloud+=*transformPointCloud(surfCloudSubMap[loopKeyCur], &copyCloudSubMapPoses6D->points[loopKeyCur]);
                    
                }else
                {
                    loopKeyCur=-1;
                    ROS_WARN("LoopKeyCur do not find !");
                }



                // if (cureKeyframeCloud->size() < 300 )
                //     return  false;


                std::cout << "matching..." << std::flush;
                auto t1 = ros::Time::now();

                // ICP Settings
                static pcl::IterativeClosestPoint<PointType, PointType> icp;
                icp.setMaxCorrespondenceDistance(historyKeyframeSearchRadius*2);
                icp.setMaximumIterations(40);
                icp.setTransformationEpsilon(1e-6);
                icp.setEuclideanFitnessEpsilon(1e-6);
                icp.setRANSACIterations(0);

                // Align clouds
                icp.setInputSource(cureKeyframeCloud);
                
                int bestMatched=-1;
                double bestScore = std::numeric_limits<double>::max();
                Eigen::Affine3f correctionLidarFrame;


                std::cout<<"loopContainerHandler: loopKeyPreSize : "<< loopInfo.loopKeyPre.size()<<std::endl;
                for(int i=0;i<loopInfo.loopKeyPre.size();i++){
                    auto thisPreId=SubMapPoses6D.find(loopInfo.loopKeyPre[i]);
                    if(thisPreId!=SubMapPoses6D.end())
                    {
                        loopKeyPre=(int)thisPreId->second.intensity;
                        std::cout<<"loopContainerHandler: loopKeyPre : "<< loopKeyPre<<std::endl;
                        
                        prevKeyframeCloud->clear();

                        findNearSubMapsforLoop(true,prevKeyframeCloud, loopKeyPre, 9);
                        findNearSubMapsforLoop(false,prevKeyframeCloud, loopKeyPre, 7);
                        if (prevKeyframeCloud->size() < 1000)
                            continue;

                        icp.setInputTarget(prevKeyframeCloud);
                        pcl::PointCloud<PointType>::Ptr unused_result(new pcl::PointCloud<PointType>());
                        icp.align(*unused_result);

                        double score = icp.getFitnessScore();
                        if (icp.hasConverged() == false || score > bestScore)
                            continue;
                        bestScore = score;
                        bestMatched = loopKeyPre;
                        correctionLidarFrame = icp.getFinalTransformation();

                    }else
                    {
                        bestMatched=-1;
                        ROS_WARN("LoopKeyPre do not find !");
                    }
                }

                loopContainerVec.pop_front();

                if(loopKeyCur==-1||bestMatched==-1)
                    return false;

                auto t2 = ros::Time::now();
                std::cout << " done" << std::endl;
                std::cout << "best_score: " <<  bestScore << "    time: " <<  (t2 - t1).toSec() << "[sec]" << std::endl;

                if(bestScore > historyKeyframeFitnessScore) {
                    std::cout << "loop not found..." << std::endl;
                    return  false;
                }
                std::cout << "loop found!!" << std::endl;


                float X, Y, Z, ROLL, PITCH, YAW;

                // transform from world origin to wrong pose
                Eigen::Affine3f tWrong = pclPointToAffine3f(copyCloudSubMapPoses6D->points[loopKeyCur]);
                // transform from world origin to corrected pose
                Eigen::Affine3f tCorrect = correctionLidarFrame * tWrong;// pre-multiplying -> successive rotation about a fixed frame
                pcl::getTranslationAndEulerAngles (tCorrect, X, Y, Z, ROLL, PITCH, YAW);
                gtsam::Pose3 poseFrom = Pose3(Rot3::RzRyRx(ROLL, PITCH, YAW), Point3(X, Y, Z));
                gtsam::Pose3 poseTo = pclPointTogtsamPose3(copyCloudSubMapPoses6D->points[bestMatched]);
                gtsam::Vector Vector6(6);

                float noiseScore = 0.01;
                // float noiseScore = bestScore*0.01;
                Vector6 << noiseScore, noiseScore, noiseScore, noiseScore, noiseScore, noiseScore;
                noiseModel::Diagonal::shared_ptr constraintNoise = noiseModel::Diagonal::Variances(Vector6);
            
                // Add pose constraint
                // mtx.lock();
                loopIndexQueue.push_back(make_pair(loopKeyCur, bestMatched));
                loopPoseQueue.push_back(poseFrom.between(poseTo));
                loopNoiseQueue.push_back(constraintNoise);
                // mtx.unlock();

                // // add loop constriant
                loopIndexContainer[loopKeyCur] = bestMatched;

                return true;


    }


    bool detectLoopClosureExternal()
    {
        ROS_WARN("Start  detectLoopClosureExternal !");
        int loopKeyCur = -1;
        int loopKeyPre = -1;

        float x, y, z, roll, pitch, yaw,noise;

        if (loopInfoVec.empty())
            return false;

        lis_slam::loop_info loopInfo;
        loopInfo=loopInfoVec.front();


        auto thisCurId=SubMapPoses6D.find(loopInfo.loopKeyCur);
        if(thisCurId!=SubMapPoses6D.end())
        {
            loopKeyCur=(int)thisCurId->second.intensity;
        }else
        {
            loopKeyCur=-1;
            ROS_WARN("LoopKeyCur do not find !");
        }

        auto thisPreId=SubMapPoses6D.find(loopInfo.loopKeyPre);
        if(thisPreId!=SubMapPoses6D.end())
        {
            loopKeyPre=(int)thisPreId->second.intensity;
        }else
        {
            loopKeyPre=-1;
            ROS_WARN("LoopKeyPre do not find !");
        }

        if(loopKeyCur==-1||loopKeyPre==-1)
            return false;
        
        std::cout<<"detectLoopClosureExternal: loopKeyCur : "<< loopKeyCur<<std::endl;
        std::cout<<"detectLoopClosureExternal: loopKeyPre : "<< loopKeyPre<<std::endl;


        x = loopInfo.loopPoseX ;
        y = loopInfo.loopPoseY ;
        z = loopInfo.loopPoseZ ;
        roll = loopInfo.loopPoseRoll ;
        pitch = loopInfo.loopPosePitch ;
        yaw = loopInfo.loopPoseYaw ;

        noise = loopInfo.loopNoise ;

        loopInfoVec.pop_front();

        Eigen::Affine3f correctionLidarFrame = pcl::getTransformation(x, y, z, roll, pitch, yaw);

        float X, Y, Z, ROLL, PITCH, YAW;

        // transform from world origin to wrong pose
        Eigen::Affine3f tWrong = pclPointToAffine3f(cloudSubMapPoses6D->points[loopKeyCur]);
        // transform from world origin to corrected pose
        Eigen::Affine3f tCorrect = correctionLidarFrame * tWrong;// pre-multiplying -> successive rotation about a fixed frame
        pcl::getTranslationAndEulerAngles (tCorrect, X, Y, Z, ROLL, PITCH, YAW);
        gtsam::Pose3 poseFrom = Pose3(Rot3::RzRyRx(ROLL, PITCH, YAW), Point3(X, Y, Z));
        gtsam::Pose3 poseTo = pclPointTogtsamPose3(cloudSubMapPoses6D->points[loopKeyPre]);
        gtsam::Vector Vector6(6);

        float noiseScore = noise;
        Vector6 << noiseScore, noiseScore, noiseScore, noiseScore, noiseScore, noiseScore;
        noiseModel::Diagonal::shared_ptr constraintNoise = noiseModel::Diagonal::Variances(Vector6);
       
        // Add pose constraint
        // mtx.lock();
        loopIndexQueue.push_back(make_pair(loopKeyCur, loopKeyPre));
        loopPoseQueue.push_back(poseFrom.between(poseTo));
        loopNoiseQueue.push_back(constraintNoise);
        // mtx.unlock();

        // // add loop constriant
        loopIndexContainer[loopKeyCur] = loopKeyPre;

        ROS_WARN("End  detectLoopClosureExternal !");
        return true;
    }


    void visualizeLoopClosure()
    {
        visualization_msgs::MarkerArray markerArray;
        // loop nodes
        visualization_msgs::Marker markerNode;
        markerNode.header.frame_id = odometryFrame;
        markerNode.header.stamp = timeLaserInfoStamp;
        markerNode.action = visualization_msgs::Marker::ADD;
        markerNode.type = visualization_msgs::Marker::SPHERE_LIST;
        markerNode.ns = "loop_nodes";
        markerNode.id = 0;
        markerNode.pose.orientation.w = 1;
        markerNode.scale.x = 2.1; markerNode.scale.y = 2.1; markerNode.scale.z = 2.1; 
        markerNode.color.r = 1; markerNode.color.g = 0.0; markerNode.color.b = 0;
        markerNode.color.a = 1;
        // loop edges
        visualization_msgs::Marker markerEdge;
        markerEdge.header.frame_id = odometryFrame;
        markerEdge.header.stamp = timeLaserInfoStamp;
        markerEdge.action = visualization_msgs::Marker::ADD;
        markerEdge.type = visualization_msgs::Marker::LINE_LIST;
        markerEdge.ns = "loop_edges";
        markerEdge.id = 1;
        markerEdge.pose.orientation.w = 1;
        markerEdge.scale.x = 2.0; markerEdge.scale.y = 2.0; markerEdge.scale.z = 2.0;
        markerEdge.color.r = 1.0; markerEdge.color.g = 0.0; markerEdge.color.b = 0;
        markerEdge.color.a = 1;


        for (auto it = loopIndexContainer.begin(); it != loopIndexContainer.end(); ++it)
        {
            int key_cur = it->first;
            int key_pre = it->second;
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

        markerArray.markers.push_back(markerNode);
        markerArray.markers.push_back(markerEdge);
        pubLoopConstraintEdge.publish(markerArray);
    }








    void mapOptmizationThread()
    {
        // ros::Rate rate(0.5);
        while (ros::ok() ){
            
            // if(subMapQueue.empty() && !(loopIndexQueue.empty())){
            //     ROS_WARN("Only LoopOptmization!!!");
                
            //     //saveKeyFramesAndFactor();

            //     addLoopFactor();

            //     // update iSAM
            //     isam->update(gtSAMgraph, initialEstimate);
            //     isam->update();
            //     isam->update();
            //     isam->update();
            //     isam->update();
            //     isam->update();

            //     gtSAMgraph.resize(0);
            //     initialEstimate.clear();


            //     correctPoses();

            //     publishFrames();

            //     publishOdometry();

            //     visualizeSubMap();

            // }
            //ROS_WARN("mapOptmizationThread !");
            if(subMapQueue.empty())
            {
                continue;
            }
            // // extract time stamp
            submapInfo = subMapQueue.front();
            subMapQueue.pop_front();

            timeLaserInfoStamp = submapInfo.header.stamp;
            timeLaserInfoCur = submapInfo.header.stamp.toSec();

            // // extract info and feature cloud
            pcl::fromROSMsg(submapInfo.submap_corner,  *laserCloudCornerLast);
            pcl::fromROSMsg(submapInfo.submap_surface, *laserCloudSurfLast);

            // ROS_WARN("subMapId: %d .",submapInfo.subMapId);

            downsampleCurrentScan();

            if (subMapFirstFlag)
            {
                updateInitialGuess();

                initialization();

                publishFrames();

                publishOdometry();

                publishSubMapInfo();

                subMapFirstFlag=false;
                continue;
            }


            if(updateInitialGuess()==false)
            {
                continue;
            }

            extractContainer();

            // if(icpTest==true)
            // {
            //     icpOptimization();
            //     icpTest=false;
            // }
    
            scan2MapOptimization();
            
            saveKeyFramesAndFactor();

            correctPoses();

            publishFrames();

            publishOdometry();

            publishSubMapInfo();

            visualizeSubMap();
               
            //  rate.sleep();

            ros::spinOnce();
        }
    }


    void initialization()
    {

        PointType thisPose3D;
        PointTypePose thisPose6D;

        thisPose3D.x = transformTobeMapped[3];
        thisPose3D.y = transformTobeMapped[4];
        thisPose3D.z = transformTobeMapped[5];
        // thisPose3D.intensity = submapInfo.subMapId; // this can be used as index
        thisPose3D.intensity = cloudSubMapPoses3D->size(); // this can be used as index
        cloudSubMapPoses3D->push_back(thisPose3D);

        thisPose6D.x = thisPose3D.x;
        thisPose6D.y = thisPose3D.y;
        thisPose6D.z = thisPose3D.z;
        thisPose6D.intensity = thisPose3D.intensity ; // this can be used as index
        thisPose6D.roll  = transformTobeMapped[0];
        thisPose6D.pitch = transformTobeMapped[1];
        thisPose6D.yaw   = transformTobeMapped[2];
        thisPose6D.time = timeLaserInfoCur;
        cloudSubMapPoses6D->push_back(thisPose6D);

        SubMapPoses3D[submapInfo.subMapId]=thisPose3D;
        SubMapPoses6D[submapInfo.subMapId]=thisPose6D;

        // save all the received edge and surf points
        pcl::PointCloud<PointType>::Ptr thisCornerKeyFrame(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr thisSurfKeyFrame(new pcl::PointCloud<PointType>());
        pcl::copyPointCloud(*laserCloudCornerLastDS,  *thisCornerKeyFrame);
        pcl::copyPointCloud(*laserCloudSurfLastDS,    *thisSurfKeyFrame);

        // save key frame cloud
        cornerCloudSubMap.push_back(thisCornerKeyFrame);
        surfCloudSubMap.push_back(thisSurfKeyFrame);


        noiseModel::Diagonal::shared_ptr priorNoise = noiseModel::Diagonal::Variances((Vector(6) << 1e-2, 1e-2, M_PI*M_PI, 1e8, 1e8, 1e8).finished()); // rad*rad, meter*meter
        gtSAMgraph.add(PriorFactor<Pose3>(0, trans2gtsamPose(transformTobeMapped), priorNoise));
        initialEstimate.insert(0, trans2gtsamPose(transformTobeMapped));

        for(int i=0;i<6;i++)
        {
            std::cout<<"First : transformTobeMapped["<<i<<"] : "<<transformTobeMapped[i]<<std::endl;
        }

    }


    bool updateInitialGuess()
    {
        // save current transformation before any processing
        incrementalOdometryAffineFront = trans2Affine3f(transformTobeMapped);

        if(submapInfo.isOptmization==true)
        {
            transformBefMapped[0]=submapInfo.subMapPoseRoll;
            transformBefMapped[1]=submapInfo.subMapPosePitch;
            transformBefMapped[2]=submapInfo.subMapPoseYaw;
            transformBefMapped[3]=submapInfo.subMapPoseX;
            transformBefMapped[4]=submapInfo.subMapPoseY;
            transformBefMapped[5]=submapInfo.subMapPoseZ;

            Eigen::Affine3f transBack = trans2Affine3f(transformTobeMapped);

            transformTobeMapped[0]=submapInfo.afterOptmizationSubMapPoseRoll;
            transformTobeMapped[1]=submapInfo.afterOptmizationSubMapPosePitch;
            transformTobeMapped[2]=submapInfo.afterOptmizationSubMapPoseYaw;
            transformTobeMapped[3]=submapInfo.afterOptmizationSubMapPoseX;
            transformTobeMapped[4]=submapInfo.afterOptmizationSubMapPoseY;
            transformTobeMapped[5]=submapInfo.afterOptmizationSubMapPoseZ;

            transPredictionMapped=trans2Affine3f(transformTobeMapped);

            Eigen::Affine3f transIncre = transBack.inverse() * transPredictionMapped;

            float x,y,z,roll,pitch,yaw;
            pcl::getTranslationAndEulerAngles(transIncre, x, y, z, roll, pitch, yaw);

            // lastOptSubMapPreTransformation = transBack;

            if(abs(yaw)>subMapOptmizationYawThresh || (x*x+y*y)>subMapOptmizationDistanceThresh*subMapOptmizationDistanceThresh)
            {
                // if(abs(transformTobeMapped[1])>0.05)
                // {
                    ROS_INFO("transformTobeMapped[1] : %f",transformTobeMapped[1]);
                    transformTobeMapped[1]=0.0;
                // }
                return true;
            }

            ROS_WARN("The distance between the two submaps is too short !");
            // ROS_INFO("Finshed  updateInitialGuess  poseAvailable !");
            return false;
            
        }

        // use submap estimation for pose guess
        static bool lastSubMapPreTransAvailable = false;
        static Eigen::Affine3f lastSubMapPreTransformation;
        if (submapInfo.poseAvailable == true)
        {
            Eigen::Affine3f transBack = pcl::getTransformation(submapInfo.subMapPoseX,    submapInfo.subMapPoseY,     submapInfo.subMapPoseZ, 
                                                               submapInfo.subMapPoseRoll, submapInfo.subMapPosePitch, submapInfo.subMapPoseYaw);
            
            transformBefMapped[0]=submapInfo.subMapPoseRoll;
            transformBefMapped[1]=submapInfo.subMapPosePitch;
            transformBefMapped[2]=submapInfo.subMapPoseYaw;
            transformBefMapped[3]=submapInfo.subMapPoseX;
            transformBefMapped[4]=submapInfo.subMapPoseY;
            transformBefMapped[5]=submapInfo.subMapPoseZ;

            if (lastSubMapPreTransAvailable == false)
            {
                pcl::getTranslationAndEulerAngles(transBack, transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5], 
                                                              transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);

                lastSubMapPreTransformation = transBack;
                lastSubMapPreTransAvailable = true;
                return true;
            } else {


                Eigen::Affine3f transIncre = lastSubMapPreTransformation.inverse() * transBack;

                float x,y,z,roll,pitch,yaw;
                pcl::getTranslationAndEulerAngles(transIncre, x, y, z, roll, pitch, yaw);

                Eigen::Affine3f transTobe = trans2Affine3f(transformTobeMapped);
                Eigen::Affine3f transFinal = transTobe * transIncre;
                pcl::getTranslationAndEulerAngles(transFinal, transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5], 
                                                              transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);


                transPredictionMapped=trans2Affine3f(transformTobeMapped);

                lastSubMapPreTransformation = transBack;

                // if(abs(yaw)>subMapOptmizationYawThresh)
                // {
                //     if(!useOdometryPitchPrediction)
                //     {
                //         ROS_INFO("transformTobeMapped[1] : %f",transformTobeMapped[1]);
                //         transformTobeMapped[1]=0.0;
                //     }

                //     int id = cloudSubMapPoses6D->size()-2;

                //     Eigen::Affine3f transBe = pclPointToAffine3f(cloudSubMapPoses6D->points[id]);
                //     Eigen::Affine3f transNow= trans2Affine3f(transformTobeMapped);

                //     Eigen::Affine3f transBN = transBe.inverse() * transNow;

                //     float x1,y1,z1,roll1,pitch1,yaw1;
                //     pcl::getTranslationAndEulerAngles(transBN, x1, y1, z1, roll1, pitch1, yaw1);

                //      if(abs(yaw1)> 0.5 )
                //      {
                //          icpTest=true;
                //          ROS_WARN("icpTest == ture !");
                //      }

                //     return true;
                // }

                if(abs(yaw)>subMapOptmizationYawThresh || (x*x+y*y)>subMapOptmizationDistanceThresh*subMapOptmizationDistanceThresh)
                {
                    if(!useOdometryPitchPrediction)
                    {
                        ROS_INFO("transformTobeMapped[1] : %f",transformTobeMapped[1]);
                        transformTobeMapped[1]=0.0;
                    }

                    return true;
                }

                ROS_WARN("The distance between the two submaps is too short !");
                // ROS_INFO("Finshed  updateInitialGuess  poseAvailable !");
                return false;
            }
        }

    }


    void extractContainer()
    {
        if (cloudSubMapPoses3D->points.empty() == true)
            return; 
        
        extractNearby();

    }


    void extractNearby()
    {
        subMapContainer.clear();

        int subMapCur=cloudSubMapPoses3D->size();
        int subMapPre = cloudSubMapPoses3D->size()-1;

        subMapContainer.push_back(subMapPre);


        // std::vector<int> pointSearchInd;
        // std::vector<float> pointSearchSqDis;

        // // extract all the nearby key poses and downsample them
        // kdtreeSurroundingKeyPoses->setInputCloud(cloudSubMapPoses3D); // create kd-tree
        // kdtreeSurroundingKeyPoses->radiusSearch(cloudSubMapPoses3D->back(), 5.0, pointSearchInd, pointSearchSqDis);
        // for (int i = 0; i < (int)pointSearchInd.size(); ++i)
        // {
        //     int subMapPre=pointSearchInd[i];

        //     bool  flag=false;
        //     for(int j = 0; j<subMapContainer.size();  ++j)
        //     {
        //         if(subMapContainer[j]==subMapPre)
        //         {
        //                  flag=true;
        //                  break;
        //         }
                   
        //     }
        //     if(flag==false){
        //         subMapContainer.push_back(subMapPre);
        //     }

        //     if(subMapContainer.size()>4)
        //         break;
        // }



        // also extract some latest key frames in case the robot rotates in one position
        int numPoses = cloudSubMapPoses3D->size();
        for (int i = numPoses-1; i >= 0; --i)
        {
            // if (timeLaserInfoCur - cloudSubMapPoses6D->points[i].time < 5.0)
            // if (timeLaserInfoCur - cloudSubMapPoses6D->points[i].time < 3.0 )
            int disContainerX=abs(cloudSubMapPoses3D->back().x-cloudSubMapPoses3D->points[i].x);
            int disContainerY=abs(cloudSubMapPoses3D->back().y-cloudSubMapPoses3D->points[i].y);
            
            long long disContainer=disContainerX*disContainerX+disContainerY*disContainerY;

            if ((timeLaserInfoCur - cloudSubMapPoses6D->points[i].time < 3.0) && (disContainer < 25))
            {
                int subMapPre=i;

                bool  flag=false;
                for(int j = 0; j<subMapContainer.size();  ++j)
                {
                    if(subMapContainer[j]==subMapPre)
                    {
                            flag=true;
                            break;
                    }
                    
                }
                if(flag==false){
                    subMapContainer.push_back(subMapPre);
                }
            }
            else
                break;


            if(subMapContainer.size()>3)
                break;
        }

    }


    void downsampleCurrentScan()
    {
        // Downsample cloud from current scan
        laserCloudCornerLastDS->clear();
        downSizeFilterSubMapCorner.setInputCloud(laserCloudCornerLast);
        downSizeFilterSubMapCorner.filter(*laserCloudCornerLastDS);
        laserCloudCornerLastDSNum = laserCloudCornerLastDS->size();

        laserCloudSurfLastDS->clear();
        downSizeFilterSubMapSurf.setInputCloud(laserCloudSurfLast);
        downSizeFilterSubMapSurf.filter(*laserCloudSurfLastDS);
        laserCloudSurfLastDSNum = laserCloudSurfLastDS->size();

        // ROS_INFO("Finshed  downsampleCurrentScan !");
    }


    void scan2MapOptimization()
    {
        if (cloudSubMapPoses3D->points.empty())
            return;

        // int subMapCur=submapInfo.subMapId;
        int subMapCur=cloudSubMapPoses3D->size();

        
        if (laserCloudCornerLastDSNum > edgeFeatureMinValidNum && laserCloudSurfLastDSNum > surfFeatureMinValidNum)
        {
            ros::Time start=ros::Time::now();

            std::sort(subMapContainer.begin(), subMapContainer.end());

            int nums = subMapContainer.size(); //'z'的数量
            ROS_INFO("submap: %d has %d Container.", subMapCur, nums);
            int testCon=1;
            for(int i=0;i<nums;i++)
            {
                int subMapPre=subMapContainer[i];
                // ROS_INFO("Container %d : %d .", testCon++,subMapPre);

                // std::cout<<"*********************************************************"<<std::endl;

                // for(int i=0;i<6;i++)
                // {
                //     std::cout<<"Befor Optimization : transformTobeMapped["<<i<<"] : "<<transformTobeMapped[i]<<std::endl;
                // }

                if(cloudSubMapPoses3D->size() <= subMapOptmizationFirstSize)
                {
                    gtsam::Pose3 poseFrom = pclPointTogtsamPose3(cloudSubMapPoses6D->points[subMapPre]);
                    gtsam::Pose3 poseTo   = trans2gtsamPose(transformTobeMapped);
                    //noiseModel::Diagonal::shared_ptr odometryNoise = noiseModel::Diagonal::Variances((Vector(6) << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4).finished());
                    noiseModel::Diagonal::shared_ptr odometryNoise = noiseModel::Diagonal::Variances((Vector(6) << 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6).finished());

                    subMapIndexContainerAll.push_back(make_pair(subMapCur, subMapPre));
                    subMapIndexQueue.push_back(make_pair(subMapCur, subMapPre));
                    subMapPoseQueue.push_back(poseFrom.between(poseTo));
                    subMapNoiseQueue.push_back(odometryNoise);

                    // ROS_INFO("cloudSubMapPoses3D size < %d : scan2MapOptimization Finshed !",subMapOptmizationFirstSize); 
                }
                else
                {
                        laserCloudCornerFromMapDS->clear();
                        laserCloudSurfFromMapDS->clear();

                        findNearSubMaps(true,laserCloudCornerFromMapDS, subMapPre, 9);
                        findNearSubMaps(false,laserCloudSurfFromMapDS, subMapPre, 7);

                        // *laserCloudCornerFromMapDS = *cornerCloudSubMap[subMapPre];
                        // *laserCloudSurfFromMapDS   = *surfCloudSubMap[subMapPre];

                        laserCloudCornerFromMapDSNum=laserCloudCornerFromMapDS->size();
                        laserCloudSurfFromMapDSNum=laserCloudSurfFromMapDS->size();


                        kdtreeCornerFromMap->setInputCloud(laserCloudCornerFromMapDS);
                        kdtreeSurfFromMap->setInputCloud(laserCloudSurfFromMapDS);


                        for (int iterCount = 0; iterCount < 30; iterCount++)   //30
                        {
                            laserCloudOri->clear();
                            coeffSel->clear();
                            
                            cornerOptimizationS();


                            surfOptimizationS();


                            combineOptimizationCoeffsS();


                    
                            if (LMOptimizationS(iterCount) == true)
                                break;    

                        }

                        transformUpdate();

                        gtsam::Pose3 poseFrom = pclPointTogtsamPose3(cloudSubMapPoses6D->points[subMapPre]);
                        gtsam::Pose3 poseTo   = trans2gtsamPose(transformTobeMapped);
                        noiseModel::Diagonal::shared_ptr odometryNoise = noiseModel::Diagonal::Variances((Vector(6) << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4).finished());

                        //  std::cout<<"*********************************************************"<<std::endl;
                        // for(int i=0;i<6;i++)
                        // {
                        //     std::cout<<"After Optimization : transformTobeMapped["<<i<<"] : "<<transformTobeMapped[i]<<std::endl;
                        // }
                        // std::cout<<"*********************************************************"<<std::endl;

                        subMapIndexContainerAll.push_back(make_pair(subMapCur, subMapPre));
                        subMapIndexQueue.push_back(make_pair(subMapCur, subMapPre));
                        subMapPoseQueue.push_back(poseFrom.between(poseTo));
                        subMapNoiseQueue.push_back(odometryNoise);

                        // ROS_INFO("scan2MapOptimization Finshed !");
                }
                

            }

            subMapContainer.clear();

            ros::Time end=ros::Time::now();
            ROS_INFO("scan2MapOptimization! time: %f ", (end-start).toSec());
            

        } else {
            ROS_WARN("Not enough features! Only %d edge and %d planar features available.", laserCloudCornerLastDSNum, laserCloudSurfLastDSNum);
        }
    }


void findNearSubMapsforLoop(const bool isCorner,pcl::PointCloud<PointType>::Ptr& nearKeyframes, const int& key, const int& searchNum)
    {
        pcl::PointCloud<PointTypePose>::Ptr testCloudSubMapPoses6D(new pcl::PointCloud<PointTypePose>());
        *testCloudSubMapPoses6D=*cloudSubMapPoses6D;
        // extract near keyframes
        nearKeyframes->clear();
        int cloudSize = cloudSubMapPoses3D->size();
        for (int i = -searchNum; i <= searchNum; ++i)
        {
            int keyNear = key + i;
            if (keyNear < 0 || keyNear >= cloudSize )
                continue;

            if(isCorner)
            {
                *nearKeyframes += *transformPointCloud(cornerCloudSubMap[keyNear], &testCloudSubMapPoses6D->points[keyNear]);
            }else
            {
                *nearKeyframes += *transformPointCloud(surfCloudSubMap[keyNear],   &testCloudSubMapPoses6D->points[keyNear]);
            }
            
        }

        if (nearKeyframes->empty())
            return;

        // downsample near keyframes
        pcl::PointCloud<PointType>::Ptr cloud_temp(new pcl::PointCloud<PointType>());
        downSizeFilterSubMap.setInputCloud(nearKeyframes);
        downSizeFilterSubMap.filter(*cloud_temp);
        *nearKeyframes = *cloud_temp;
    }


    void findNearSubMaps(const bool isCorner,pcl::PointCloud<PointType>::Ptr& nearKeyframes, const int& key, const int& searchNum)
    {

        pcl::PointCloud<PointTypePose>::Ptr testCloudSubMapPoses6D(new pcl::PointCloud<PointTypePose>());
        *testCloudSubMapPoses6D=*cloudSubMapPoses6D;
        // extract near keyframes
        nearKeyframes->clear();
        int cloudSize = testCloudSubMapPoses6D->size();
        for (int i = -searchNum; i <= searchNum; ++i)
        {
            int keyNear = key + i;
            if (keyNear < 0 || keyNear >= cloudSize )
                continue;

            if(isCorner)
            {
                *nearKeyframes += *transformPointCloud(cornerCloudSubMap[keyNear], &testCloudSubMapPoses6D->points[keyNear]);
            }else
            {
                *nearKeyframes += *transformPointCloud(surfCloudSubMap[keyNear],   &testCloudSubMapPoses6D->points[keyNear]);
            }
            
        }

        if (nearKeyframes->empty())
            return;

        // downsample near keyframes
        pcl::PointCloud<PointType>::Ptr cloud_temp(new pcl::PointCloud<PointType>());
        downSizeFilterSubMap.setInputCloud(nearKeyframes);
        downSizeFilterSubMap.filter(*cloud_temp);
        *nearKeyframes = *cloud_temp;
    }

    //使用PCL库中的ICP匹配方法进行配准，测试

    void icpOptimization()
    {

        int subMapCur=cloudSubMapPoses3D->size();
       
        if (laserCloudCornerLastDSNum > edgeFeatureMinValidNum && laserCloudSurfLastDSNum > surfFeatureMinValidNum)
        {

                int subMapPre=subMapCur-1;

                pcl::PointCloud<PointType>::Ptr cornerCloud(new pcl::PointCloud<PointType>());
                pcl::PointCloud<PointType>::Ptr surfCloud(new pcl::PointCloud<PointType>());

                findNearSubMaps(true,cornerCloud, subMapPre, 7);
                findNearSubMaps(false,surfCloud, subMapPre, 5);

                pcl::PointCloud<PointType>::Ptr targetCloud(new pcl::PointCloud<PointType>());
                *targetCloud+= *surfCloud;
                *targetCloud+= *cornerCloud;

                pcl::PointCloud<PointType>::Ptr sourceCloud(new pcl::PointCloud<PointType>());
                *sourceCloud+=*laserCloudCornerLastDS;
                *sourceCloud+=*laserCloudSurfLastDS;


                std::cout << "optmization matching..." << std::flush;
                auto t1 = ros::Time::now();

                // ICP Settings
                static pcl::IterativeClosestPoint<PointType, PointType> icp;
                icp.setMaxCorrespondenceDistance(100);
                icp.setMaximumIterations(100);
                icp.setTransformationEpsilon(1e-6);
                icp.setEuclideanFitnessEpsilon(1e-6);
                icp.setRANSACIterations(0);

                // Align clouds
                icp.setInputSource(sourceCloud);
                icp.setInputTarget(targetCloud);
                pcl::PointCloud<PointType>::Ptr unused_result(new pcl::PointCloud<PointType>());
                icp.align(*unused_result);

                double score = icp.getFitnessScore();
                
                auto t2 = ros::Time::now();
                
                std::cout << "score: " << boost::format("%.3f") % score << "    time: " << boost::format("%.3f") % (t2 - t1).toSec() << "[sec]" << std::endl;

                std::cout << " done" << std::endl;
                if(icp.hasConverged() == false || score  >3.0) {
                    std::cout << "optmization not found..." << std::endl;
                    return;
                }
            
                std::cout << "optmization found!!" << std::endl;
                std::cout << "score: " << boost::format("%.3f") % score << "    time: " << boost::format("%.3f") % (t2 - t1).toSec() << "[sec]" << std::endl;

                // Get pose transformation
                float x, y, z, roll, pitch, yaw;
                Eigen::Affine3f correctionLidarFrame;
                correctionLidarFrame = icp.getFinalTransformation();
                // transform from world origin to wrong pose
                Eigen::Affine3f tWrong = trans2Affine3f(transformTobeMapped);
                // transform from world origin to corrected pose
                Eigen::Affine3f tCorrect = correctionLidarFrame * tWrong;// pre-multiplying -> successive rotation about a fixed frame
                pcl::getTranslationAndEulerAngles (tCorrect, x, y, z, roll, pitch, yaw);

                transformTobeMapped[0] = roll;
                transformTobeMapped[1] = pitch;
                transformTobeMapped[2] = yaw;
                transformTobeMapped[3] = x;
                transformTobeMapped[4] = y;
                transformTobeMapped[5] = z;


                pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());
                pcl::fromROSMsg(submapInfo.submap_raw, *cloudOut);

                PointTypePose thisPose6D = trans2PointTypePose(transformTobeMapped);
                *cloudOut = *transformPointCloud(cloudOut,  &thisPose6D);
                publishCloud(&pubIcpOptmizationCloud, cloudOut, timeLaserInfoStamp, odometryFrame);


                // transformUpdate();

                // tWrong = trans2Affine3f(transformTobeMapped);
                // // transform from world origin to corrected pose
                // tCorrect = correctionLidarFrame * tWrong;// pre-multiplying -> successive rotation about a fixed frame
                // pcl::getTranslationAndEulerAngles (tCorrect, x, y, z, roll, pitch, yaw);
                // gtsam::Pose3 poseTo = Pose3(Rot3::RzRyRx(roll, pitch, yaw), Point3(x, y, z));
                // gtsam::Pose3 poseFrom = pclPointTogtsamPose3(cloudSubMapPoses6D->points[subMapPre]);
            
                // gtsam::Vector Vector6(6);
                // float noiseScore = score;
                // Vector6 << noiseScore, noiseScore, noiseScore, noiseScore, noiseScore, noiseScore;
                // noiseModel::Diagonal::shared_ptr odometryNoise = noiseModel::Diagonal::Variances(Vector6);

                // subMapIndexContainerAll.push_back(make_pair(subMapCur, subMapPre));
                // subMapIndexQueue.push_back(make_pair(subMapCur, subMapPre));
                // subMapPoseQueue.push_back(poseFrom.between(poseTo));
                // subMapNoiseQueue.push_back(odometryNoise);


                ROS_INFO("icpOptimization Finshed !");


        } else {
            ROS_WARN("Not enough features! Only %d edge and %d planar features available.", laserCloudCornerLastDSNum, laserCloudSurfLastDSNum);
        }
    }





    void transformUpdate()
    {
        Eigen::Affine3f odometrySub = pcl::getTransformation(submapInfo.subMapPoseX,    submapInfo.subMapPoseY,     submapInfo.subMapPoseZ, 
                                                               submapInfo.subMapPoseRoll, submapInfo.subMapPosePitch, submapInfo.subMapPoseYaw);
        Eigen::Affine3f afterOptmization = pcl::getTransformation(transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5], 
                                                            transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);
        
        Eigen::Affine3f transBetween = transPredictionMapped.inverse() * afterOptmization;
        float x, y, z, roll, pitch, yaw;
        pcl::getTranslationAndEulerAngles(transBetween, x, y, z, roll, pitch, yaw);

        float transPreX, transPreY, transPreZ, transPreRoll, transPrePitch, transPreYaw;
        pcl::getTranslationAndEulerAngles(transPredictionMapped, transPreX, transPreY, transPreZ, transPreRoll, transPrePitch, transPreYaw);

        // ROS_INFO("scan2MapOptimization transformUpdate : x : %f",x);
        // ROS_INFO("scan2MapOptimization transformUpdate : y : %f",y);
        // ROS_INFO("scan2MapOptimization transformUpdate : z : %f",z);
        // ROS_INFO("scan2MapOptimization transformUpdate : roll : %f",roll);
        // ROS_INFO("scan2MapOptimization transformUpdate : pitch : %f",pitch);
        // ROS_INFO("scan2MapOptimization transformUpdate : yaw : %f",yaw);


        // transformTobeMapped[0] = constraintTransformation(roll,odometerAndOptimizedAngleDifference,transformTobeMapped[0],transPreRoll);
        // transformTobeMapped[1] = constraintTransformation(pitch,odometerAndOptimizedAngleDifference,transformTobeMapped[1],transPrePitch);
        transformTobeMapped[2] = constraintTransformation(yaw,odometerAndOptimizedAngleDifference,transformTobeMapped[2],transPreYaw);
        // transformTobeMapped[3] = constraintTransformation(x,odometerAndOptimizedDistanceDifference,transformTobeMapped[3],transPreX);
        // transformTobeMapped[4] = constraintTransformation(y,odometerAndOptimizedDistanceDifference,transformTobeMapped[4],transPreY);
        // transformTobeMapped[5] = constraintTransformation(z,odometerAndOptimizedDistanceDifference,transformTobeMapped[5],transPreZ);

                
        incrementalOdometryAffineBack = trans2Affine3f(transformTobeMapped);
    }

    float constraintTransformation(float value, float limit, float now, float pre)
    {

        if(fabs(value)>limit)
        {
            ROS_WARN("Adding is too big !");
            return pre;
        }else 
        {
            return now;
        }

    }


    void updatePointAssociateToMap()
    {
            transPointAssociateToMap=trans2Affine3f(transformTobeMapped);
    }

    void cornerOptimizationS()
    {
        updatePointAssociateToMap();

        #pragma omp parallel for num_threads(numberOfCores)
        for (int i = 0; i < laserCloudCornerLastDSNum; i++)
        {
            PointType pointOriS, pointSelS, coeffs;
            std::vector<int> pointSearchIndS;
            std::vector<float> pointSearchSqDisS;

            pointOriS = laserCloudCornerLastDS->points[i];
            pointAssociateToMap(&pointOriS, &pointSelS);

            // if (!pcl_isfinite (pointSelS.x) || !pcl_isfinite (pointSelS.y) || !pcl_isfinite (pointSelS.z)){
            //         // ROS_INFO("cornerOptimizationS :This Point is  pcl_isfinite !");
            //         // cornerIsFinite++;
            //         continue;
            // }

            kdtreeCornerFromMap->nearestKSearch(pointSelS, 5, pointSearchIndS, pointSearchSqDisS);

            cv::Mat matA1S(3, 3, CV_32F, cv::Scalar::all(0));
            cv::Mat matD1S(1, 3, CV_32F, cv::Scalar::all(0));
            cv::Mat matV1S(3, 3, CV_32F, cv::Scalar::all(0));
                    
            if (pointSearchSqDisS[4] < 1.0) {
                float cx = 0, cy = 0, cz = 0;
                for (int j = 0; j < 5; j++) {
                    cx += laserCloudCornerFromMapDS->points[pointSearchIndS[j]].x;
                    cy += laserCloudCornerFromMapDS->points[pointSearchIndS[j]].y;
                    cz += laserCloudCornerFromMapDS->points[pointSearchIndS[j]].z;
                }
                cx /= 5; cy /= 5;  cz /= 5;

                float a11 = 0, a12 = 0, a13 = 0, a22 = 0, a23 = 0, a33 = 0;
                for (int j = 0; j < 5; j++) {
                    float ax = laserCloudCornerFromMapDS->points[pointSearchIndS[j]].x - cx;
                    float ay = laserCloudCornerFromMapDS->points[pointSearchIndS[j]].y - cy;
                    float az = laserCloudCornerFromMapDS->points[pointSearchIndS[j]].z - cz;

                    a11 += ax * ax; a12 += ax * ay; a13 += ax * az;
                    a22 += ay * ay; a23 += ay * az;
                    a33 += az * az;
                }
                a11 /= 5; a12 /= 5; a13 /= 5; a22 /= 5; a23 /= 5; a33 /= 5;

                matA1S.at<float>(0, 0) = a11; matA1S.at<float>(0, 1) = a12; matA1S.at<float>(0, 2) = a13;
                matA1S.at<float>(1, 0) = a12; matA1S.at<float>(1, 1) = a22; matA1S.at<float>(1, 2) = a23;
                matA1S.at<float>(2, 0) = a13; matA1S.at<float>(2, 1) = a23; matA1S.at<float>(2, 2) = a33;

                cv::eigen(matA1S, matD1S, matV1S);

                if (matD1S.at<float>(0, 0) > 3 * matD1S.at<float>(0, 1)) {

                    float x0 = pointSelS.x;
                    float y0 = pointSelS.y;
                    float z0 = pointSelS.z;
                    float x1 = cx + 0.1 * matV1S.at<float>(0, 0);
                    float y1 = cy + 0.1 * matV1S.at<float>(0, 1);
                    float z1 = cz + 0.1 * matV1S.at<float>(0, 2);
                    float x2 = cx - 0.1 * matV1S.at<float>(0, 0);
                    float y2 = cy - 0.1 * matV1S.at<float>(0, 1);
                    float z2 = cz - 0.1 * matV1S.at<float>(0, 2);

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

                    coeffs.x = s * la;
                    coeffs.y = s * lb;
                    coeffs.z = s * lc;
                    coeffs.intensity = s * ld2;

                    if (s > 0.1) {
                        laserCloudOriCornerVec[i] = pointOriS;
                        coeffSelCornerVec[i] = coeffs;
                        laserCloudOriCornerFlag[i] = true;
                    }
                }
            }
        }
    }


    void surfOptimizationS()
    {
        updatePointAssociateToMap();

        #pragma omp parallel for num_threads(numberOfCores)
        for (int i = 0; i < laserCloudSurfLastDSNum; i++)
        {
            PointType pointOriS, pointSelS, coeffs;
            std::vector<int> pointSearchIndS;
            std::vector<float> pointSearchSqDisS;

            pointOriS = laserCloudSurfLastDS->points[i];
            pointAssociateToMap(&pointOriS, &pointSelS); 

            // if (!pcl_isfinite (pointSelS.x) || !pcl_isfinite (pointSelS.y) || !pcl_isfinite (pointSelS.z)){
            //         // ROS_INFO("surfOptimizationS: This Point is  pcl_isfinite !");
            //         // surfIsFinite++;
            //         continue;
            // }

            kdtreeSurfFromMap->nearestKSearch(pointSelS, 5, pointSearchIndS, pointSearchSqDisS);

            // Eigen::Matrix<float, 5, 3> matA0S;
            // Eigen::Matrix<float, 5, 1> matB0S;
            // Eigen::Vector3f matX0S;

            // matA0S.setZero();
            // matB0S.fill(-1);
            // matX0S.setZero();

            cv::Mat matA0S(5, 3, CV_32F, cv::Scalar::all(0));
            cv::Mat matB0S (5, 1, CV_32F, cv::Scalar::all(-1));
            cv::Mat matX0S(3, 1, CV_32F, cv::Scalar::all(0));

            
            if (pointSearchSqDisS[4] < 1.0) {
                for (int j = 0; j < 5; j++) {
                    // matA0S(j, 0) = laserCloudSurfFromMapDS->points[pointSearchIndS[j]].x;
                    // matA0S(j, 1) = laserCloudSurfFromMapDS->points[pointSearchIndS[j]].y;
                    // matA0S(j, 2) = laserCloudSurfFromMapDS->points[pointSearchIndS[j]].z;

                    matA0S.at<float>(j, 0) = laserCloudSurfFromMapDS->points[pointSearchIndS[j]].x;
                    matA0S.at<float>(j, 1) = laserCloudSurfFromMapDS->points[pointSearchIndS[j]].y;
                    matA0S.at<float>(j, 2) = laserCloudSurfFromMapDS->points[pointSearchIndS[j]].z;
                }

                // matX0S = matA0S.colPivHouseholderQr().solve(matB0S);

                cv::solve(matA0S, matB0S, matX0S, cv::DECOMP_QR);

                // float pa = matX0S(0, 0);
                // float pb = matX0S(1, 0);
                // float pc = matX0S(2, 0);
                // float pd = 1;

                float pa = matX0S.at<float>(0, 0);
                float pb = matX0S.at<float>(1, 0);
                float pc = matX0S.at<float>(2, 0);
                float pd = 1;

                float ps = sqrt(pa * pa + pb * pb + pc * pc);
                pa /= ps; pb /= ps; pc /= ps; pd /= ps;

                bool planeValid = true;
                for (int j = 0; j < 5; j++) {
                    if (fabs(pa * laserCloudSurfFromMapDS->points[pointSearchIndS[j]].x +
                             pb * laserCloudSurfFromMapDS->points[pointSearchIndS[j]].y +
                             pc * laserCloudSurfFromMapDS->points[pointSearchIndS[j]].z + pd) > 0.2) {
                        planeValid = false;
                        break;
                    }
                }

                if (planeValid) {
                    float pd2 = pa * pointSelS.x + pb * pointSelS.y + pc * pointSelS.z + pd;

                    float s = 1 - 0.9 * fabs(pd2) / sqrt(sqrt(pointSelS.x * pointSelS.x
                            + pointSelS.y * pointSelS.y + pointSelS.z * pointSelS.z));

                    coeffs.x = s * pa;
                    coeffs.y = s * pb;
                    coeffs.z = s * pc;
                    coeffs.intensity = s * pd2;

                    if (s > 0.1) {
                        laserCloudOriSurfVec[i] = pointOriS;
                        coeffSelSurfVec[i] = coeffs;
                        laserCloudOriSurfFlag[i] = true;
                    }
                }

            }
        }
    }

    void combineOptimizationCoeffsS()
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

    bool LMOptimizationS(int iterCount)
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

        PointType pointOriS, coeffs;
        #pragma omp parallel for num_threads(numberOfCores)
        for (int i = 0; i < laserCloudSelNum; i++) {
            // lidar -> camera
            pointOriS.x = laserCloudOri->points[i].y;
            pointOriS.y = laserCloudOri->points[i].z;
            pointOriS.z = laserCloudOri->points[i].x;
            // lidar -> camera
            coeffs.x = coeffSel->points[i].y;
            coeffs.y = coeffSel->points[i].z;
            coeffs.z = coeffSel->points[i].x;
            coeffs.intensity = coeffSel->points[i].intensity;
            // in camera
            float arx = (crx*sry*srz*pointOriS.x + crx*crz*sry*pointOriS.y - srx*sry*pointOriS.z) * coeffs.x
                      + (-srx*srz*pointOriS.x - crz*srx*pointOriS.y - crx*pointOriS.z) * coeffs.y
                      + (crx*cry*srz*pointOriS.x + crx*cry*crz*pointOriS.y - cry*srx*pointOriS.z) * coeffs.z;

            float ary = ((cry*srx*srz - crz*sry)*pointOriS.x 
                      + (sry*srz + cry*crz*srx)*pointOriS.y + crx*cry*pointOriS.z) * coeffs.x
                      + ((-cry*crz - srx*sry*srz)*pointOriS.x 
                      + (cry*srz - crz*srx*sry)*pointOriS.y - crx*sry*pointOriS.z) * coeffs.z;

            float arz = ((crz*srx*sry - cry*srz)*pointOriS.x + (-cry*crz-srx*sry*srz)*pointOriS.y)*coeffs.x
                      + (crx*crz*pointOriS.x - crx*srz*pointOriS.y) * coeffs.y
                      + ((sry*srz + cry*crz*srx)*pointOriS.x + (crz*sry-cry*srx*srz)*pointOriS.y)*coeffs.z;
            // lidar -> camera
            matA.at<float>(i, 0) = arz;
            matA.at<float>(i, 1) = arx;
            matA.at<float>(i, 2) = ary;
            matA.at<float>(i, 3) = coeffs.z;
            matA.at<float>(i, 4) = coeffs.x;
            matA.at<float>(i, 5) = coeffs.y;
            matB.at<float>(i, 0) = -coeffs.intensity;
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

        float deltaR = sqrt(
                            pow(pcl::rad2deg(matX.at<float>(0, 0)), 2) +
                            pow(pcl::rad2deg(matX.at<float>(1, 0)), 2) +
                            pow(pcl::rad2deg(matX.at<float>(2, 0)), 2));
        float deltaT = sqrt(
                            pow(matX.at<float>(3, 0) * 100, 2) +
                            pow(matX.at<float>(4, 0) * 100, 2) +
                            pow(matX.at<float>(5, 0) * 100, 2));

        if (deltaR < 0.005 && deltaT < 0.005) {
            return true; // converged
        }
        return false; // keep optimizing
    }


    bool saveFrame()
    {
        if (cloudSubMapPoses3D->points.empty())
            return true;

        Eigen::Affine3f transStart = pclPointToAffine3f(cloudSubMapPoses6D->back());
        Eigen::Affine3f transFinal = pcl::getTransformation(transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5], 
                                                            transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);
        Eigen::Affine3f transBetween = transStart.inverse() * transFinal;
        float x, y, z, roll, pitch, yaw;
        pcl::getTranslationAndEulerAngles(transBetween, x, y, z, roll, pitch, yaw);

        if (abs(roll)  < surroundingkeyframeAddingAngleThreshold &&
            abs(pitch) < surroundingkeyframeAddingAngleThreshold && 
            abs(yaw)   < surroundingkeyframeAddingAngleThreshold &&
            sqrt(x*x + y*y + z*z) < surroundingkeyframeAddingDistThreshold)
        {
            ROS_WARN("saveFrame :  surrounding keyframe Adding is too small !");
            return false;
        }
            
        return true;
    }





    void saveKeyFramesAndFactor()
    {
        // if (saveFrame() == false){
            
        //         subMapIndexQueue.clear();
        //         subMapPoseQueue.clear();
        //         subMapNoiseQueue.clear();
        //         subMapIndexContainerAll.pop_back();
        //         return;
        // }

        // odom factor
        addOdomFactor();

        // gps factor
        //addGPSFactor();

        // loop factor
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
        latestEstimate = isamCurrentEstimate.at<Pose3>(isamCurrentEstimate.size()-1);
        // cout << "****************************************************" << endl;
        // isamCurrentEstimate.print("Current estimate: ");

        thisPose3D.x = latestEstimate.translation().x();
        thisPose3D.y = latestEstimate.translation().y();
        thisPose3D.z = latestEstimate.translation().z();
        thisPose3D.intensity = cloudSubMapPoses3D->size(); // this can be used as index
        cloudSubMapPoses3D->push_back(thisPose3D);

        thisPose6D.x = thisPose3D.x;
        thisPose6D.y = thisPose3D.y;
        thisPose6D.z = thisPose3D.z;
        thisPose6D.intensity = thisPose3D.intensity ; // this can be used as index
        thisPose6D.roll  = latestEstimate.rotation().roll();
        thisPose6D.pitch = latestEstimate.rotation().pitch();
        thisPose6D.yaw   = latestEstimate.rotation().yaw();
        thisPose6D.time = timeLaserInfoCur;
        cloudSubMapPoses6D->push_back(thisPose6D);

        SubMapPoses3D[submapInfo.subMapId]=thisPose3D;
        SubMapPoses6D[submapInfo.subMapId]=thisPose6D;

        // cout << "****************************************************" << endl;
        // cout << "Pose covariance:" << endl;
        // cout << isam->marginalCovariance(isamCurrentEstimate.size()-1) << endl << endl;
        poseCovariance = isam->marginalCovariance(isamCurrentEstimate.size()-1);

        // save updated transform
        transformTobeMapped[0] = latestEstimate.rotation().roll();
        transformTobeMapped[1] = latestEstimate.rotation().pitch();
        transformTobeMapped[2] = latestEstimate.rotation().yaw();
        transformTobeMapped[3] = latestEstimate.translation().x();
        transformTobeMapped[4] = latestEstimate.translation().y();
        transformTobeMapped[5] = latestEstimate.translation().z();

        // save all the received edge and surf points
        pcl::PointCloud<PointType>::Ptr thisCornerKeyFrame(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr thisSurfKeyFrame(new pcl::PointCloud<PointType>());
        pcl::copyPointCloud(*laserCloudCornerLastDS,  *thisCornerKeyFrame);
        pcl::copyPointCloud(*laserCloudSurfLastDS,    *thisSurfKeyFrame);

        // save key frame cloud
        cornerCloudSubMap.push_back(thisCornerKeyFrame);
        surfCloudSubMap.push_back(thisSurfKeyFrame);

        // save path for visualization
        updatePath(thisPose6D);

    }



    void addOdomFactor()
    {
            if (subMapIndexQueue.empty())
            {
                return ;
            }
                

            for (int i = 0; i < (int)subMapIndexQueue.size(); ++i)
            {
                int indexTo = subMapIndexQueue[i].first;
                int indexFrom = subMapIndexQueue[i].second;
                gtsam::Pose3 poseBetween = subMapPoseQueue[i];
                gtsam::noiseModel::Diagonal::shared_ptr noiseBetween = subMapNoiseQueue[i];
                gtSAMgraph.add(BetweenFactor<Pose3>(indexFrom, indexTo, poseBetween, noiseBetween));

                // gtsam::Pose3 poseTo   = trans2gtsamPose(transformTobeMapped);
                // initialEstimate.insert(indexTo, poseTo);
            }

            int indexTo = subMapIndexQueue[0].first;
            gtsam::Pose3 poseTo   = trans2gtsamPose(transformTobeMapped);
            initialEstimate.insert(indexTo, poseTo);

            subMapIndexQueue.clear();
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
            if (gpsQueue.front().header.stamp.toSec() < timeLaserInfoCur - 0.2)
            {
                // message too old
                gpsQueue.pop_front();
                // ROS_INFO("GPS message too old!");
            }
            else if (gpsQueue.front().header.stamp.toSec() > timeLaserInfoCur + 0.2)
            {
                // message too new
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
                gtsam::GPSFactor gps_factor(cloudSubMapPoses3D->size(), gtsam::Point3(gps_x, gps_y, gps_z), gps_noise);
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

        for (int i = 0; i < (int)loopIndexQueue.size(); ++i)
        {
            int indexFrom = loopIndexQueue[i].first;
            int indexTo = loopIndexQueue[i].second;
            gtsam::Pose3 poseBetween = loopPoseQueue[i];
            gtsam::noiseModel::Diagonal::shared_ptr noiseBetween = loopNoiseQueue[i];
            gtSAMgraph.add(BetweenFactor<Pose3>(indexFrom, indexTo, poseBetween, noiseBetween));
        }

        loopIndexQueue.clear();
        loopPoseQueue.clear();
        loopNoiseQueue.clear();
        aLoopIsClosed = true;

        ROS_WARN("Finshed  addLoopFactor !");
    }

    void updatePath(const PointTypePose& pose_in)
    {
        geometry_msgs::PoseStamped pose_stamped;
        pose_stamped.header.stamp = ros::Time().fromSec(pose_in.time);
        pose_stamped.header.frame_id = odometryFrame;
        pose_stamped.pose.position.x = pose_in.x;
        pose_stamped.pose.position.y = pose_in.y;
        pose_stamped.pose.position.z = pose_in.z;
        tf::Quaternion q = tf::createQuaternionFromRPY(pose_in.roll, pose_in.pitch, pose_in.yaw);
        pose_stamped.pose.orientation.x = q.x();
        pose_stamped.pose.orientation.y = q.y();
        pose_stamped.pose.orientation.z = q.z();
        pose_stamped.pose.orientation.w = q.w();

        globalPath.poses.push_back(pose_stamped);

    }


    void correctPoses()
    {
        if (cloudSubMapPoses3D->points.empty())
            return;

        if (aLoopIsClosed == true)
        {
            // clear path
            globalPath.poses.clear();
            // update key poses
            int numPoses = isamCurrentEstimate.size();
            for (int i = 0; i < numPoses; ++i)
            {
                cloudSubMapPoses3D->points[i].x = isamCurrentEstimate.at<Pose3>(i).translation().x();
                cloudSubMapPoses3D->points[i].y = isamCurrentEstimate.at<Pose3>(i).translation().y();
                cloudSubMapPoses3D->points[i].z = isamCurrentEstimate.at<Pose3>(i).translation().z();

                cloudSubMapPoses6D->points[i].x = cloudSubMapPoses3D->points[i].x;
                cloudSubMapPoses6D->points[i].y = cloudSubMapPoses3D->points[i].y;
                cloudSubMapPoses6D->points[i].z = cloudSubMapPoses3D->points[i].z;
                cloudSubMapPoses6D->points[i].roll  = isamCurrentEstimate.at<Pose3>(i).rotation().roll();
                cloudSubMapPoses6D->points[i].pitch = isamCurrentEstimate.at<Pose3>(i).rotation().pitch();
                cloudSubMapPoses6D->points[i].yaw   = isamCurrentEstimate.at<Pose3>(i).rotation().yaw();

                updatePath(cloudSubMapPoses6D->points[i]);
            }

            ROS_INFO("Finshed  correctPoses !");
            aLoopIsClosed = false;
        }
    }




    void publishFrames()
    {
        if (cloudSubMapPoses3D->points.empty())
            return;
        // publish key poses
        publishCloud(&pubKeyPoses, cloudSubMapPoses3D, timeLaserInfoStamp, odometryFrame);

        // publish registered high-res raw cloud
        if (pubCloudRegisteredRaw.getNumSubscribers() != 0)
        {
            pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());
            pcl::fromROSMsg(submapInfo.submap_raw, *cloudOut);

            PointTypePose thisPose6D = trans2PointTypePose(transformTobeMapped);
            *cloudOut = *transformPointCloud(cloudOut,  &thisPose6D);
            // submapInfoPub.submap_raw=publishCloud(&pubCloudRegisteredRaw, cloudOut, timeLaserInfoStamp, odometryFrame);
            publishCloud(&pubCloudRegisteredRaw, cloudOut, timeLaserInfoStamp, odometryFrame);
        }
        // publish path
        if (pubPath.getNumSubscribers() != 0)
        {
            globalPath.header.stamp = timeLaserInfoStamp;
            globalPath.header.frame_id = odometryFrame;
            pubPath.publish(globalPath);
        }

        // ROS_INFO("Finshed  publishFrames !");
    }

    void publishOdometry()
    {
        nav_msgs::Odometry laserOdometryROS;
        laserOdometryROS.header.stamp = timeLaserInfoStamp;
        laserOdometryROS.header.frame_id = odometryFrame;
        laserOdometryROS.child_frame_id = "odom_mapping";
        laserOdometryROS.pose.pose.position.x = transformTobeMapped[3];
        laserOdometryROS.pose.pose.position.y = transformTobeMapped[4];
        laserOdometryROS.pose.pose.position.z = transformTobeMapped[5];
        laserOdometryROS.pose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);
        laserOdometryROS.twist.twist.angular.x = transformBefMapped[0];
        laserOdometryROS.twist.twist.angular.y = transformBefMapped[1];
        laserOdometryROS.twist.twist.angular.z = transformBefMapped[2];
        laserOdometryROS.twist.twist.linear.x = transformBefMapped[3];
        laserOdometryROS.twist.twist.linear.y = transformBefMapped[4];
        laserOdometryROS.twist.twist.linear.z = transformBefMapped[5];
        pubSubMapOdometryGlobal.publish(laserOdometryROS);
    }



    void visualizeSubMap()
    {
        visualization_msgs::MarkerArray markerArray;

        visualization_msgs::Marker markerNodeId;
        markerNodeId.header.frame_id = odometryFrame;
        markerNodeId.header.stamp = timeLaserInfoStamp;
        markerNodeId.action = visualization_msgs::Marker::ADD;
        markerNodeId.type =  visualization_msgs::Marker::TEXT_VIEW_FACING;
        markerNodeId.ns = "submap_nodes_id";
        markerNodeId.id = 0;
        markerNodeId.pose.orientation.w = 1;
        markerNodeId.scale.z = 0.2; 
        markerNodeId.color.r = 0; markerNodeId.color.g = 0; markerNodeId.color.b = 255;
        markerNodeId.color.a = 1;
        // loop nodes
        visualization_msgs::Marker markerNode;
        markerNode.header.frame_id = odometryFrame;
        markerNode.header.stamp = timeLaserInfoStamp;
        markerNode.action = visualization_msgs::Marker::ADD;
        markerNode.type = visualization_msgs::Marker::SPHERE_LIST;
        markerNode.ns = "submap_nodes";
        markerNode.id = 0;
        markerNode.pose.orientation.w = 1;
        markerNode.scale.x = 0.2; markerNode.scale.y = 0.2; markerNode.scale.z = 0.2; 
        markerNode.color.r = 0; markerNode.color.g = 0.8; markerNode.color.b = 1;
        markerNode.color.a = 1;
        // loop edges
        visualization_msgs::Marker markerEdge;
        markerEdge.header.frame_id = odometryFrame;
        markerEdge.header.stamp = timeLaserInfoStamp;
        markerEdge.action = visualization_msgs::Marker::ADD;
        markerEdge.type = visualization_msgs::Marker::LINE_LIST;
        markerEdge.ns = "submap_edges";
        markerEdge.id = 1;
        markerEdge.pose.orientation.w = 1;
        markerEdge.scale.x = 0.1; markerEdge.scale.y = 0.1; markerEdge.scale.z = 0.1;
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
            int k=key_cur;
            ostringstream str;
            str<<k;
            markerNodeId.id = k;
            markerNodeId.text=str.str();
            markerNodeId.pose=pose;
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


    void publishSubMapInfo()
    {
            submapInfoPub.header.stamp=timeLaserInfoStamp;

            submapInfoPub.subMapPoseX=submapInfo.subMapPoseX;
            submapInfoPub.subMapPoseY=submapInfo.subMapPoseY;
            submapInfoPub.subMapPoseZ=submapInfo.subMapPoseZ;
            submapInfoPub.subMapPoseRoll=submapInfo.subMapPoseRoll;
            submapInfoPub.subMapPosePitch=submapInfo.subMapPosePitch;
            submapInfoPub.subMapPoseYaw=submapInfo.subMapPoseYaw;

            submapInfoPub.afterOptmizationSubMapPoseX=transformTobeMapped[3];
            submapInfoPub.afterOptmizationSubMapPoseY=transformTobeMapped[4];
            submapInfoPub.afterOptmizationSubMapPoseZ=transformTobeMapped[5];
            submapInfoPub.afterOptmizationSubMapPoseRoll=transformTobeMapped[0];
            submapInfoPub.afterOptmizationSubMapPosePitch=transformTobeMapped[1];
            submapInfoPub.afterOptmizationSubMapPoseYaw=transformTobeMapped[2];

            submapInfoPub.poseAvailable=true;
            submapInfoPub.isOptmization=true;

            // submapInfoPub.subMapId=cloudSubMapPoses3D->size()-1;
            submapInfoPub.subMapId=submapInfo.subMapId;


            submapInfoPub.submap_raw=submapInfo.submap_raw;
            submapInfoPub.submap_corner=submapInfo.submap_corner;
            submapInfoPub.submap_surface=submapInfo.submap_surface;
     
            pubSubMapInfo.publish(submapInfoPub);
            // ROS_INFO("Finshed  publishSubMapInfo !");
    }



};



int main(int argc, char** argv)
{
    ros::init(argc, argv, "lis_slam");

    mapOptimization MO;

    ROS_INFO("\033[1;32m----> Map Optimization Started.\033[0m");
    
    std::thread loopthread(&mapOptimization::loopClosureThread, &MO);
    std::thread optmizationThread(&mapOptimization::mapOptmizationThread, &MO);
    std::thread visualizeMapThread(&mapOptimization::visualizeGlobalMapThread, &MO);



    //ros::spin();

    loopthread.join();
    optmizationThread.join();
    visualizeMapThread.join();


    // MO.visualizeGlobalMapThread();

    return 0;
}
