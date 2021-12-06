// Author of  EPSC-LOAM : QZ Wang
// Email wangqingzhi27@outlook.com

//#include "odomEstimation.h"

#include "utility.h"
#include "lis_slam/cloud_info.h"

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


class OdomEstimationNode : public ParamServer
{

public:

    ros::Subscriber subCloudInfo;
    ros::Publisher pubKeyFrameInfo;    
    ros::Publisher pubKeyFrameId;

    ros::Publisher pubLaserOdometryIncremental;
    ros::Publisher pubLaserOdometryGlobal;


    lis_slam::cloud_info cloudInfo;
    lis_slam::cloud_info keyFrameInfo;

    std::mutex mtx;

    std::deque<lis_slam::cloud_info> cloudInfoQueue;

    bool FirstFlag=true;

    uint64  subMapId=0;

    pcl::PointCloud<PointTypePose>::Ptr keyFramePose6D; 
    pcl::PointCloud<PointType>::Ptr keyFramePose3D;

    map<int,PointType> keyFramePosesIndex3D;
    map<int,PointTypePose> keyFramePosesIndex6D;

    pcl::PointCloud<PointType>::Ptr laserCloudCornerLast; // corner feature set from odoOptimization
    pcl::PointCloud<PointType>::Ptr laserCloudSurfLast; // surf feature set from odoOptimization
    pcl::PointCloud<PointType>::Ptr laserCloudCornerLastDS; // downsampled corner featuer set from odoOptimization
    pcl::PointCloud<PointType>::Ptr laserCloudSurfLastDS; // downsampled surf featuer set from odoOptimization
    

    pcl::VoxelGrid<PointType> downSizeFilterCorner;
    pcl::VoxelGrid<PointType> downSizeFilterSurf;

    int laserCloudCornerLastDSNum = 0;
    int laserCloudSurfLastDSNum = 0;


    pcl::PointCloud<PointType>::Ptr laserCloudOri;
    pcl::PointCloud<PointType>::Ptr coeffSel;

    std::vector<PointType> laserCloudOriCornerVec; // corner point holder for parallel computation
    std::vector<PointType> coeffSelCornerVec;
    std::vector<bool> laserCloudOriCornerFlag;
    std::vector<PointType> laserCloudOriSurfVec; // surf point holder for parallel computation
    std::vector<PointType> coeffSelSurfVec;
    std::vector<bool> laserCloudOriSurfFlag;


    ros::Time timeLaserInfoStamp;
    ros::Time timeKeyFrameInfoStamp;
    double timeLaserInfoCur;

    float transformTobeSubMapped[6];

    float afterMapOptmizationPoses[6];
    Eigen::Affine3f transDelta;

    float transformCurFrame2PriFrame[6];
    float transformPriFrame[6];

    float transPredictionMapped[6];

    Eigen::Affine3f transPointAssociateToSubMap;
    Eigen::Affine3f incrementalOdometryAffineFront;
    Eigen::Affine3f incrementalOdometryAffineBack;


    bool isDegenerate = false;
    Eigen::Matrix<float, 6, 6> matP;

    bool isMapOptmization=false;



    pcl::PointCloud<PointType>::Ptr cloudKeyPoses3D;
    pcl::PointCloud<PointTypePose>::Ptr cloudKeyPoses6D;
    vector<pcl::PointCloud<PointType>::Ptr> cornerCloudKeyFrames;
    vector<pcl::PointCloud<PointType>::Ptr> surfCloudKeyFrames;

    map<int, pair<pcl::PointCloud<PointType>, pcl::PointCloud<PointType>>> laserCloudMapContainer;
    pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMap;
    pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMap;
    pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMapDS;
    pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMapDS;

    pcl::KdTreeFLANN<PointType>::Ptr kdtreeCornerFromMap;
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurfFromMap;

    pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurroundingKeyPoses;

    pcl::VoxelGrid<PointType> downSizeFilterSurroundingKeyPoses; // for surrounding key poses of scan-to-map optimization

    int laserCloudCornerFromMapDSNum = 0;
    int laserCloudSurfFromMapDSNum = 0;

    OdomEstimationNode() 
    {
        subCloudInfo = nh.subscribe<lis_slam::cloud_info>("lis_slam/laser_process/cloud_info", 1, &OdomEstimationNode::laserCloudInfoHandler, this);

        pubKeyFrameInfo = nh.advertise<lis_slam::submap>("lis_slam/odom_estimation/cloud_info", 1);

        pubLaserOdometryGlobal      = nh.advertise<nav_msgs::Odometry> ("lis_slam/odom_estimation/odometry", 1);
        pubLaserOdometryIncremental = nh.advertise<nav_msgs::Odometry> ("lis_slam/odom_estimation/odometry_incremental", 1);
       
        pubKeyFrameId = nh.advertise<sensor_msgs::PointCloud2>("lis_slam/odom_estimation/keyframe_id", 1);

        downSizeFilterCorner.setLeafSize(mappingCornerLeafSize, mappingCornerLeafSize, mappingCornerLeafSize);
        downSizeFilterSurf.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize);

        allocateMemory();

        downSizeFilterSurroundingKeyPoses.setLeafSize(surroundingKeyframeDensity, surroundingKeyframeDensity, surroundingKeyframeDensity); // for surrounding key poses of scan-to-map optimization

    }

    void allocateMemory()
    {
    
        keyFramePose6D.reset(new pcl::PointCloud<PointTypePose>()); 
        keyFramePose3D.reset(new pcl::PointCloud<PointType>()); 

        laserCloudCornerLast.reset(new pcl::PointCloud<PointType>()); 
        laserCloudSurfLast.reset(new pcl::PointCloud<PointType>()); 
        laserCloudCornerLastDS.reset(new pcl::PointCloud<PointType>()); 
        laserCloudSurfLastDS.reset(new pcl::PointCloud<PointType>()); 

        laserCloudCornerFromSubMap.reset(new pcl::PointCloud<PointType>());
        laserCloudSurfFromSubMap.reset(new pcl::PointCloud<PointType>());
        laserCloudCornerFromSubMapDS.reset(new pcl::PointCloud<PointType>());
        laserCloudSurfFromSubMapDS.reset(new pcl::PointCloud<PointType>());

        kdtreeCornerFromSubMap.reset(new pcl::KdTreeFLANN<PointType>());
        kdtreeSurfFromSubMap.reset(new pcl::KdTreeFLANN<PointType>());

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

        for (int i = 0; i < 6; ++i){
            transformTobeSubMapped[i] = 0;
            transformCurFrame2PriFrame[i] = 0;
            transformPriFrame[i] = 0;

            transPredictionMapped[i]=0.0;

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



    void laserCloudInfoHandler(const lis_slam::cloud_infoConstPtr& msgIn)
    {
        std::lock_guard<std::mutex> lock(mtx);
        cloudInfoQueue.push_back(*msgIn);

    }


    void OdomEstimationNodeThread(){
        
        while(ros::ok()){

            if(cloudInfoQueue.empty())
            {
                continue;
            }

            static ros::Time startTime=ros::Time::now();
            ros::Time frameStartTime=ros::Time::now();

            // extract info and feature cloud
            cloudInfo = cloudInfoQueue.front();
            cloudInfoQueue.pop_front();
            // extract time stamp
            timeLaserInfoStamp = cloudInfo.header.stamp;
            timeLaserInfoCur = cloudInfo.header.stamp.toSec();

            
            pcl::fromROSMsg(cloudInfo.cloud_corner,  *laserCloudCornerLast);
            pcl::fromROSMsg(cloudInfo.cloud_surface, *laserCloudSurfLast);
            //新增
            pcl::fromROSMsg(cloudInfo.cloud_deskewed, *laserCloudRawLast);

            // Downsample cloud from current scan
            downsampleCurrentScan();
            
            if(subMapFirstFlag)
            {
                updateInitialGuess();
                firstMakeSubMap();
                publishOdometry();
                subMapFirstFlag=false;
                continue;
            }

            updateInitialGuess();

            extractSurroundingKeyFrames();

            scan2SubMapOptimization();

            saveKeyFrames();

            publishOdometry();
            

            calculateTranslation();

            subMapYawSum+=abs(transformCurFrame2PriFrame[2]);
            if(abs(transformCurFrame2PriFrame[2])>=instertMiniYaw ||abs(transformCurFrame2PriFrame[3])>=instertMiniDistance || abs(transformCurFrame2PriFrame[4])>=instertMiniDistance)
            // if((transformCurFrame2PriFrame[3]*transformCurFrame2PriFrame[3]+transformCurFrame2PriFrame[4]*transformCurFrame2PriFrame[4])>=instertMiniDistance*instertMiniDistance)
            {
                insterSubMap();

                curSubMapSize++;

                // ros::Time frameEndTime=ros::Time::now();
                // ROS_INFO("make %d frame time: %.3f", curSubMapSize,  (frameEndTime - frameStartTime).toSec());                   
            }

            if(curSubMapSize>=(int)(subMapFramesSize/3))
            {

                ros::Time submapEndTime=ros::Time::now();
                float OdomEstimationNodeTime=(submapEndTime - submapStartTime).toSec();
                // if(subMapYawSum>=subMapYawMax||curSubMapSize>=subMapFramesSize)
                if(subMapYawSum>=subMapYawMax||curSubMapSize>=subMapFramesSize||OdomEstimationNodeTime>=subMapMaxTime)
                {
                    subMapYawSum=0;
                    ROS_INFO("Make %d submap  has %d  Frames !", subMapId, curSubMapSize);

                    publishSubMapInfo();
                    publishCloud(&pubSubMapId, subMapPose3D, timeLaserInfoStamp, mapFrame);

                    updateSubMap();

                                    
                    ros::Time submapEndTime=ros::Time::now();
                    ROS_INFO("Make %d submap time: %.3f", subMapId-1,  (submapEndTime - submapStartTime).toSec());
                    submapStartTime=ros::Time::now();
                    
                }
                
            }



        }





    }

    void pointAssociateToSubMap(PointType const * const pi, PointType * const po)
    {
        po->x = transPointAssociateToSubMap(0,0) * pi->x + transPointAssociateToSubMap(0,1) * pi->y + transPointAssociateToSubMap(0,2) * pi->z + transPointAssociateToSubMap(0,3);
        po->y = transPointAssociateToSubMap(1,0) * pi->x + transPointAssociateToSubMap(1,1) * pi->y + transPointAssociateToSubMap(1,2) * pi->z + transPointAssociateToSubMap(1,3);
        po->z = transPointAssociateToSubMap(2,0) * pi->x + transPointAssociateToSubMap(2,1) * pi->y + transPointAssociateToSubMap(2,2) * pi->z + transPointAssociateToSubMap(2,3);
        po->intensity = pi->intensity;
    }


    pcl::PointCloud<PointType>::Ptr transformPointCloud(pcl::PointCloud<PointType>::Ptr cloudIn, PointTypePose* transformIn)
    {
        pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

        PointType *pointFrom;

        int cloudSize = cloudIn->size();
        cloudOut->resize(cloudSize);

        Eigen::Affine3f transCur = pcl::getTransformation(transformIn->x, transformIn->y, transformIn->z, transformIn->roll, transformIn->pitch, transformIn->yaw);
        
        #pragma omp parallel for num_threads(numberOfCores)
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

    pcl::PointCloud<PointType>::Ptr transformPointCloud(pcl::PointCloud<PointType>::Ptr cloudIn, Eigen::Affine3f transformIn)
    {
        pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

        PointType *pointFrom;

        int cloudSize = cloudIn->size();
        cloudOut->resize(cloudSize);

        #pragma omp parallel for num_threads(numberOfCores)
        // #pragma omp for
        for (int i = 0; i < cloudSize; ++i)
        {
            pointFrom = &cloudIn->points[i];
            cloudOut->points[i].x = transformIn(0,0) * pointFrom->x + transformIn(0,1) * pointFrom->y + transformIn(0,2) * pointFrom->z + transformIn(0,3);
            cloudOut->points[i].y = transformIn(1,0) * pointFrom->x + transformIn(1,1) * pointFrom->y + transformIn(1,2) * pointFrom->z + transformIn(1,3);
            cloudOut->points[i].z = transformIn(2,0) * pointFrom->x + transformIn(2,1) * pointFrom->y + transformIn(2,2) * pointFrom->z + transformIn(2,3);
            cloudOut->points[i].intensity = pointFrom->intensity;
        }
        return cloudOut;
    }

    Eigen::Affine3f pclPointToAffine3f(PointTypePose thisPoint)
    { 
        return pcl::getTransformation(thisPoint.x, thisPoint.y, thisPoint.z, thisPoint.roll, thisPoint.pitch, thisPoint.yaw);
    }

    Eigen::Affine3f trans2Affine3f(float transformIn[])
    {
        return pcl::getTransformation(transformIn[3], transformIn[4], transformIn[5], transformIn[0], transformIn[1], transformIn[2]);
    }







    void firstMakeSubMap()
    {
        timeSubMapInfoStamp=timeLaserInfoStamp;
        PointTypePose  point6d;
        point6d.x=0;
        point6d.y=0;
        point6d.z=0;
        point6d.intensity=subMapId;
        point6d.roll=transformTobeSubMapped[0];
        point6d.pitch=transformTobeSubMapped[1];
        point6d.yaw=transformTobeSubMapped[2];
        point6d.time=timeLaserInfoCur;

        subMapPose6D->points.push_back(point6d);   

        PointType  point3d;
        point3d.x=0;
        point3d.y=0;
        point3d.z=0;
        point3d.intensity=subMapId;

        subMapPose3D->points.push_back(point3d);   

        subMapPosesIndex3D[subMapId]=point3d;
        subMapPosesIndex6D[subMapId]=point6d;


        laserCloudRawFromSubMapDS->clear();
        laserCloudCornerFromSubMapDS->clear();
        laserCloudSurfFromSubMapDS->clear();


        *laserCloudRawFromSubMapDS += *laserCloudRawLastDS;
        *laserCloudCornerFromSubMapDS += *laserCloudCornerLastDS;
        *laserCloudSurfFromSubMapDS += *laserCloudSurfLastDS;
        
        // laserCloudCornerFromSubMapDSNum= laserCloudCornerFromSubMapDS->size();  
        // laserCloudSurfFromSubMapDSNum= laserCloudSurfFromSubMapDS->size(); 

        laserCloudCornerFromPreSubMapDS->clear();
        laserCloudSurfFromPreSubMapDS->clear();
        *laserCloudCornerFromPreSubMapDS =*laserCloudCornerFromSubMapDS;
        *laserCloudSurfFromPreSubMapDS =*laserCloudSurfFromSubMapDS;


        cloudKeyPoses3D->push_back(point3d);
        cloudKeyPoses6D->push_back(point6d);
        // save all the received edge and surf points
        pcl::PointCloud<PointType>::Ptr thisCornerKeyFrame(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr thisSurfKeyFrame(new pcl::PointCloud<PointType>());
        pcl::copyPointCloud(*laserCloudCornerLastDS,  *thisCornerKeyFrame);
        pcl::copyPointCloud(*laserCloudSurfLastDS,    *thisSurfKeyFrame);

        // save key frame cloud
        cornerCloudKeyFrames.push_back(thisCornerKeyFrame);
        surfCloudKeyFrames.push_back(thisSurfKeyFrame);
    }

    void downsampleCurrentScan()
    {
        // Downsample cloud from current scan
        laserCloudCornerLastDS->clear();
        downSizeFilterCorner.setInputCloud(laserCloudCornerLast);
        downSizeFilterCorner.filter(*laserCloudCornerLastDS);
        laserCloudCornerLastDSNum = laserCloudCornerLastDS->size();

        laserCloudSurfLastDS->clear();
        downSizeFilterSurf.setInputCloud(laserCloudSurfLast);
        downSizeFilterSurf.filter(*laserCloudSurfLastDS);
        laserCloudSurfLastDSNum = laserCloudSurfLastDS->size();  

        laserCloudRawLastDS->clear();
        downSizeFilterSurf.setInputCloud(laserCloudRawLast);
        downSizeFilterSurf.filter(*laserCloudRawLastDS);
    }
        


    void calculateTranslation()
    {
        Eigen::Affine3f transBack = trans2Affine3f(transformPriFrame);
        Eigen::Affine3f transTobe = trans2Affine3f(transformTobeSubMapped);

        Eigen::Affine3f transIncre = transBack.inverse() * transTobe;

        pcl::getTranslationAndEulerAngles(transIncre, transformCurFrame2PriFrame[3], transformCurFrame2PriFrame[4], transformCurFrame2PriFrame[5], 
                                                            transformCurFrame2PriFrame[0], transformCurFrame2PriFrame[1], transformCurFrame2PriFrame[2]);

    }


    void updateInitialGuess()
    {
        // save current transformation before any processing
        incrementalOdometryAffineFront = trans2Affine3f(transformTobeSubMapped);

        static Eigen::Affine3f lastImuTransformation;
        // initialization
        static bool firstTransAvailable = false;
        if (firstTransAvailable==false)
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
                if (first==false)
                {
                    for(int i=0;i<6;++i){
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
                
                for(int i=0;i<6;++i){
                    lastTransformTobeSubMapped[i]=transformTobeSubMapped[i];
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

    void saveKeyFrames()
    {
        PointType thisPose3D;
        thisPose3D.x = transformTobeSubMapped[3];
        thisPose3D.y = transformTobeSubMapped[4];
        thisPose3D.z = transformTobeSubMapped[5];
        thisPose3D.intensity = cloudKeyPoses3D->size(); // this can be used as index
        cloudKeyPoses3D->push_back(thisPose3D);

        PointTypePose thisPose6D;   
        thisPose6D.x = thisPose3D.x;
        thisPose6D.y = thisPose3D.y;
        thisPose6D.z = thisPose3D.z;
        thisPose6D.intensity = thisPose3D.intensity ; // this can be used as index
        thisPose6D.roll  = transformTobeSubMapped[0];
        thisPose6D.pitch = transformTobeSubMapped[1];
        thisPose6D.yaw   = transformTobeSubMapped[2];
        thisPose6D.time = timeLaserInfoCur;
        cloudKeyPoses6D->push_back(thisPose6D);

        pcl::PointCloud<PointType>::Ptr thisCornerKeyFrame(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr thisSurfKeyFrame(new pcl::PointCloud<PointType>());
        pcl::copyPointCloud(*laserCloudCornerLastDS,  *thisCornerKeyFrame);
        pcl::copyPointCloud(*laserCloudSurfLastDS,    *thisSurfKeyFrame);

        // save key frame cloud
        cornerCloudKeyFrames.push_back(thisCornerKeyFrame);
        surfCloudKeyFrames.push_back(thisSurfKeyFrame);

    }



    void insterSubMap()
    {

        int lastId=subMapPose6D->points.size()-1;
        Eigen::Affine3f transTobe = trans2Affine3f(transformTobeSubMapped);
        Eigen::Affine3f transSubMap = pclPointToAffine3f(subMapPose6D->points[lastId]);
        // Eigen::Affine3f transSubMapIncre = transTobe.inverse() * transSubMap;
        Eigen::Affine3f transSubMapIncre = transSubMap.inverse() * transTobe;

        *laserCloudRawLastDS = *transformPointCloud(laserCloudRawLastDS,  transSubMapIncre);
        *laserCloudCornerLastDS = *transformPointCloud(laserCloudCornerLastDS,  transSubMapIncre);
        *laserCloudSurfLastDS = *transformPointCloud(laserCloudSurfLastDS,  transSubMapIncre);

        *laserCloudRawFromSubMap += *laserCloudRawLastDS;
        *laserCloudCornerFromSubMap += *laserCloudCornerLastDS;
        *laserCloudSurfFromSubMap += *laserCloudSurfLastDS;

        laserCloudCornerFromSubMapDS->clear();
        downSizeFilterCorner.setInputCloud(laserCloudCornerFromSubMap);
        downSizeFilterCorner.filter(*laserCloudCornerFromSubMapDS);        
        // laserCloudCornerFromSubMapDSNum= laserCloudCornerFromSubMapDS->size();

        laserCloudSurfFromSubMapDS->clear();
        downSizeFilterSurf.setInputCloud(laserCloudSurfFromSubMap);
        downSizeFilterSurf.filter(*laserCloudSurfFromSubMapDS);     
        // laserCloudSurfFromSubMapDSNum= laserCloudSurfFromSubMapDS->size(); 

        if(subMapId==0)
        {
            laserCloudCornerFromPreSubMapDS->clear();
            laserCloudSurfFromPreSubMapDS->clear();
            *laserCloudCornerFromPreSubMapDS =*laserCloudCornerFromSubMapDS;
            *laserCloudSurfFromPreSubMapDS =*laserCloudSurfFromSubMapDS;
        }

        transformPriFrame[0]=transformTobeSubMapped[0];
        transformPriFrame[1]=transformTobeSubMapped[1];
        transformPriFrame[2]=transformTobeSubMapped[2];
        transformPriFrame[3]=transformTobeSubMapped[3];
        transformPriFrame[4]=transformTobeSubMapped[4];
        transformPriFrame[5]=transformTobeSubMapped[5];

        
        publishCloud(&pubSubMapRawTest, laserCloudSurfFromSubMapDS, timeLaserInfoStamp, mapFrame);

    }





    void updateSubMap()
    {
        laserCloudCornerFromPreSubMapDS->clear();
        laserCloudSurfFromPreSubMapDS->clear();
        *laserCloudCornerFromPreSubMapDS =*laserCloudCornerFromSubMapDS;
        *laserCloudSurfFromPreSubMapDS =*laserCloudSurfFromSubMapDS;

        timeSubMapInfoStamp=timeLaserInfoStamp;
        subMapId++;
        curSubMapSize=0;

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

        subMapPosesIndex3D[subMapId]=point3d;
        subMapPosesIndex6D[subMapId]=point6d;

        laserCloudRawFromSubMap->clear();
        laserCloudCornerFromSubMap->clear();
        laserCloudSurfFromSubMap->clear();

    }


    void publishSubMapInfo()
    {
            submapInfo.header.stamp=timeSubMapInfoStamp;

            submapInfo.subMapId=subMapId;

            auto it_=subMapPosesIndex6D.find(subMapId);
            if(it_!=subMapPosesIndex6D.end())
            {
                submapInfo.subMapPoseX=it_->second.x;
                submapInfo.subMapPoseY=it_->second.y;
                submapInfo.subMapPoseZ=it_->second.z;
                submapInfo.subMapPoseRoll=it_->second.roll;
                submapInfo.subMapPosePitch=it_->second.pitch;
                submapInfo.subMapPoseYaw=it_->second.yaw;

                if(isMapOptmization)
                {

                    Eigen::Affine3f transBeforOptCurSubMap = pclPointToAffine3f(it_->second);
                    float afterSubMapPose6D[6];
                    Eigen::Affine3f transAfterOptCurSubMap = transDelta*transBeforOptCurSubMap;
                    pcl::getTranslationAndEulerAngles(transAfterOptCurSubMap, afterSubMapPose6D[3], afterSubMapPose6D[4], afterSubMapPose6D[5], 
                                                                afterSubMapPose6D[0], afterSubMapPose6D[1], afterSubMapPose6D[2]);

                    submapInfo.afterOptmizationSubMapPoseX=afterSubMapPose6D[3];
                    submapInfo.afterOptmizationSubMapPoseY=afterSubMapPose6D[4];
                    submapInfo.afterOptmizationSubMapPoseZ=afterSubMapPose6D[5];
                    submapInfo.afterOptmizationSubMapPoseRoll=afterSubMapPose6D[0];
                    submapInfo.afterOptmizationSubMapPosePitch=afterSubMapPose6D[1];
                    submapInfo.afterOptmizationSubMapPoseYaw=afterSubMapPose6D[2];

                    submapInfo.isOptmization=true;

                    isMapOptmization=false;
                }
                else
                {
                    submapInfo.isOptmization=false;
                }
                


            }else
            {
                ROS_WARN("Do not Find subMapPosesIndex6D : %d",subMapId);
                return;
            }

            // submapInfo.subMapPoseX=transformTobeSubMapped[3];
            // submapInfo.subMapPoseY=transformTobeSubMapped[4];
            // submapInfo.subMapPoseZ=transformTobeSubMapped[5];
            // submapInfo.subMapPoseRoll=transformTobeSubMapped[0];
            // submapInfo.subMapPosePitch=transformTobeSubMapped[1];
            // submapInfo.subMapPoseYaw=transformTobeSubMapped[2];


            submapInfo.poseAvailable=true;


            laserCloudRawFromSubMapDS->clear();
            downSizeFilterSubMap.setInputCloud(laserCloudRawFromSubMap);
            downSizeFilterSubMap.filter(*laserCloudRawFromSubMapDS);   

            submapInfo.submap_raw=publishCloud(&pubSubMapRaw, laserCloudRawFromSubMapDS, timeLaserInfoStamp, lidarFrame);;
            submapInfo.submap_corner=publishCloud(&pubSubMapCorner, laserCloudCornerFromSubMapDS, timeLaserInfoStamp, lidarFrame);;
            submapInfo.submap_surface=publishCloud(&pubSubMapSurf, laserCloudSurfFromSubMapDS, timeLaserInfoStamp, lidarFrame);;
            
            pubSubMapInfo.publish(submapInfo);
    }






    void extractSurroundingKeyFrames()
    {
        

        pcl::PointCloud<PointType>::Ptr surroundingKeyPoses(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr surroundingKeyPosesDS(new pcl::PointCloud<PointType>());
        // std::vector<int> pointSearchInd;
        // std::vector<float> pointSearchSqDis;

        // // extract all the nearby key poses and downsample them
        // kdtreeSurroundingKeyPoses->setInputCloud(cloudKeyPoses3D); // create kd-tree
        // kdtreeSurroundingKeyPoses->radiusSearch(cloudKeyPoses3D->back(), (double)surroundingKeyframeSearchRadius, pointSearchInd, pointSearchSqDis);
        // for (int i = 0; i < (int)pointSearchInd.size(); ++i)
        // {
        //     int id = pointSearchInd[i];
        //     surroundingKeyPoses->push_back(cloudKeyPoses3D->points[id]);
        // }

        // downSizeFilterSurroundingKeyPoses.setInputCloud(surroundingKeyPoses);
        // downSizeFilterSurroundingKeyPoses.filter(*surroundingKeyPosesDS);

        // also extract some latest key frames in case the robot rotates in one position
        int numPoses = cloudKeyPoses3D->size();
        for (int i = numPoses-1; i >= 0; --i)
        {
            if (timeLaserInfoCur - cloudKeyPoses6D->points[i].time < 20.0)   //10.0
                surroundingKeyPosesDS->push_back(cloudKeyPoses3D->points[i]);
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
            if (laserCloudMapContainer.find(thisKeyInd) != laserCloudMapContainer.end()) 
            {
                // transformed cloud available
                *laserCloudCornerFromMap += laserCloudMapContainer[thisKeyInd].first;
                *laserCloudSurfFromMap   += laserCloudMapContainer[thisKeyInd].second;
            } else {
                // transformed cloud not available
                pcl::PointCloud<PointType> laserCloudCornerTemp = *transformPointCloud(cornerCloudKeyFrames[thisKeyInd],  &cloudKeyPoses6D->points[thisKeyInd]);
                pcl::PointCloud<PointType> laserCloudSurfTemp = *transformPointCloud(surfCloudKeyFrames[thisKeyInd],    &cloudKeyPoses6D->points[thisKeyInd]);
                *laserCloudCornerFromMap += laserCloudCornerTemp;
                *laserCloudSurfFromMap   += laserCloudSurfTemp;
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

        if (laserCloudCornerLastDSNum > edgeFeatureMinValidNum && laserCloudSurfLastDSNum > surfFeatureMinValidNum)
        {
            // ROS_INFO("laserCloudCornerLastDSNum: %d laserCloudSurfLastDSNum: %d .", laserCloudCornerLastDSNum, laserCloudSurfLastDSNum);
            
            // kdtreeCornerFromSubMap->setInputCloud(laserCloudCornerFromPreSubMapDS);
            // kdtreeSurfFromSubMap->setInputCloud(laserCloudSurfFromPreSubMapDS);

            kdtreeCornerFromMap->setInputCloud(laserCloudCornerFromMapDS);
            kdtreeSurfFromMap->setInputCloud(laserCloudSurfFromMapDS);
        

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

    void cornerOptimization()
    {
        updatePointAssociateToSubMap();

        // #pragma omp for
        #pragma omp parallel for num_threads(numberOfCores)
        for (int i = 0; i < laserCloudCornerLastDSNum; i++)
        {
            PointType pointOri, pointSel, coeff;
            std::vector<int> pointSearchInd;
            std::vector<float> pointSearchSqDis;

            pointOri = laserCloudCornerLastDS->points[i];
            pointAssociateToSubMap(&pointOri, &pointSel);
            // kdtreeCornerFromSubMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);

            kdtreeCornerFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);


            cv::Mat matA1(3, 3, CV_32F, cv::Scalar::all(0));
            cv::Mat matD1(1, 3, CV_32F, cv::Scalar::all(0));
            cv::Mat matV1(3, 3, CV_32F, cv::Scalar::all(0));


            if (pointSearchSqDis[4] < 1.0) {
                float cx = 0, cy = 0, cz = 0;
                for (int j = 0; j < 5; j++) {

                        // cx += laserCloudCornerFromPreSubMapDS->points[pointSearchInd[j]].x;
                        // cy += laserCloudCornerFromPreSubMapDS->points[pointSearchInd[j]].y;
                        // cz += laserCloudCornerFromPreSubMapDS->points[pointSearchInd[j]].z;

                        cx += laserCloudCornerFromMapDS->points[pointSearchInd[j]].x;
                        cy += laserCloudCornerFromMapDS->points[pointSearchInd[j]].y;
                        cz += laserCloudCornerFromMapDS->points[pointSearchInd[j]].z;
     
                }
                cx /= 5; cy /= 5;  cz /= 5;

                float a11 = 0, a12 = 0, a13 = 0, a22 = 0, a23 = 0, a33 = 0;
                for (int j = 0; j < 5; j++) {
                    float ax=0, ay=0, az=0;
  
                    // ax = laserCloudCornerFromPreSubMapDS->points[pointSearchInd[j]].x - cx;
                    // ay = laserCloudCornerFromPreSubMapDS->points[pointSearchInd[j]].y - cy;
                    // az = laserCloudCornerFromPreSubMapDS->points[pointSearchInd[j]].z - cz;

                    ax = laserCloudCornerFromMapDS->points[pointSearchInd[j]].x - cx;
                    ay = laserCloudCornerFromMapDS->points[pointSearchInd[j]].y - cy;
                    az = laserCloudCornerFromMapDS->points[pointSearchInd[j]].z - cz;

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
            PointType pointOri, pointSel, coeff;
            std::vector<int> pointSearchInd;
            std::vector<float> pointSearchSqDis;

            pointOri = laserCloudSurfLastDS->points[i];
            pointAssociateToSubMap(&pointOri, &pointSel); 
            // kdtreeSurfFromSubMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);

            kdtreeSurfFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);


            Eigen::Matrix<float, 5, 3> matA0;
            Eigen::Matrix<float, 5, 1> matB0;
            Eigen::Vector3f matX0;

            matA0.setZero();
            matB0.fill(-1);
            matX0.setZero();

            if (pointSearchSqDis[4] < 1.0) {
                for (int j = 0; j < 5; j++) {

                        // matA0(j, 0) = laserCloudSurfFromPreSubMapDS->points[pointSearchInd[j]].x;
                        // matA0(j, 1) = laserCloudSurfFromPreSubMapDS->points[pointSearchInd[j]].y;
                        // matA0(j, 2) = laserCloudSurfFromPreSubMapDS->points[pointSearchInd[j]].z;

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

                        // if (fabs(pa * laserCloudSurfFromPreSubMapDS->points[pointSearchInd[j]].x +
                        //         pb * laserCloudSurfFromPreSubMapDS->points[pointSearchInd[j]].y +
                        //         pc * laserCloudSurfFromPreSubMapDS->points[pointSearchInd[j]].z + pd) > 0.2) {
                        //     planeValid = false;
                        //     break;
                        // }


                        if (fabs(pa * laserCloudSurfFromMapDS->points[pointSearchInd[j]].x +
                                pb * laserCloudSurfFromMapDS->points[pointSearchInd[j]].y +
                                pc * laserCloudSurfFromMapDS->points[pointSearchInd[j]].z + pd) > 0.2) {
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





        // Eigen::Affine3f odometrySub = pcl::getTransformation(cloudInfo.initialGuessX,    cloudInfo.initialGuessY,     cloudInfo.initialGuessZ, 
        //                                                        cloudInfo.initialGuessRoll, cloudInfo.initialGuessPitch, cloudInfo.initialGuessYaw);
        // Eigen::Affine3f afterOptmization = pcl::getTransformation(transformTobeSubMapped[3], transformTobeSubMapped[4], transformTobeSubMapped[5], 
        //                                                     transformTobeSubMapped[0], transformTobeSubMapped[1], transformTobeSubMapped[2]);
        
        // Eigen::Affine3f transBetween = transPredictionMapped.inverse() * afterOptmization;
        // float x, y, z, roll, pitch, yaw;
        // pcl::getTranslationAndEulerAngles(transBetween, x, y, z, roll, pitch, yaw);

        // float transPreX, transPreY, transPreZ, transPreRoll, transPrePitch, transPreYaw;
        // pcl::getTranslationAndEulerAngles(transPredictionMapped, transPreX, transPreY, transPreZ, transPreRoll, transPrePitch, transPreYaw);


        // ROS_INFO("scan2MapOptimization transformUpdate : transPreX : %f",transPredictionMapped[3]);
        // ROS_INFO("scan2MapOptimization transformUpdate : transPreY : %f",transPredictionMapped[4]);
        // ROS_INFO("scan2MapOptimization transformUpdate : transPreZ : %f",transPredictionMapped[5]);
        // ROS_INFO("scan2MapOptimization transformUpdate : transPreRoll : %f",transPredictionMapped[0]);
        // ROS_INFO("scan2MapOptimization transformUpdate : transPrePitch : %f",transPredictionMapped[1]);
        // ROS_INFO("scan2MapOptimization transformUpdate : transPreYaw : %f",transPredictionMapped[2]);

        // ROS_INFO("scan2MapOptimization transformUpdate : transformTobeSubMapped[0] : %f",transformTobeSubMapped[0]);
        // ROS_INFO("scan2MapOptimization transformUpdate : transformTobeSubMapped[1] : %f",transformTobeSubMapped[1]);
        // ROS_INFO("scan2MapOptimization transformUpdate : transformTobeSubMapped[2] : %f",transformTobeSubMapped[2]);
        // ROS_INFO("scan2MapOptimization transformUpdate : transformTobeSubMapped[3] : %f",transformTobeSubMapped[3]);
        // ROS_INFO("scan2MapOptimization transformUpdate : transformTobeSubMapped[4] : %f",transformTobeSubMapped[4]);
        // ROS_INFO("scan2MapOptimization transformUpdate : transformTobeSubMapped[5] : %f",transformTobeSubMapped[5]);

        // float x, y, z, roll, pitch, yaw;
        // x=transPredictionMapped[3]-transformTobeSubMapped[3];
        // y=transPredictionMapped[4]-transformTobeSubMapped[4];
        // z=transPredictionMapped[5]-transformTobeSubMapped[5];
        // roll=transPredictionMapped[0]-transformTobeSubMapped[0];
        // pitch=transPredictionMapped[1]-transformTobeSubMapped[1];
        // yaw=transPredictionMapped[2]-transformTobeSubMapped[2];

        // ROS_INFO("scan2MapOptimization transformUpdate : x : %f",x);
        // ROS_INFO("scan2MapOptimization transformUpdate : y : %f",y);
        // ROS_INFO("scan2MapOptimization transformUpdate : z : %f",z);
        // ROS_INFO("scan2MapOptimization transformUpdate : roll : %f",roll);
        // ROS_INFO("scan2MapOptimization transformUpdate : pitch : %f",pitch);
        // ROS_INFO("scan2MapOptimization transformUpdate : yaw : %f",yaw);


        // transformTobeSubMapped[0] = constraintTransformation(roll,odometerAndOptimizedAngleDifference,transformTobeSubMapped[0],transPreRoll);
        // transformTobeSubMapped[1] = constraintTransformation(pitch,odometerAndOptimizedAngleDifference,transformTobeSubMapped[1],transPrePitch);
        // transformTobeSubMapped[2] = constraintTransformation(yaw,odometerAndOptimizedAngleDifference,transformTobeSubMapped[2],transPreYaw);
        // transformTobeSubMapped[3] = constraintTransformation(x,odometerAndOptimizedDistanceDifference,transformTobeSubMapped[3],transPredictionMapped[3]);
        // transformTobeSubMapped[4] = constraintTransformation(y,odometerAndOptimizedDistanceDifference,transformTobeSubMapped[4],transPredictionMapped[4]);
        // transformTobeSubMapped[5] = constraintTransformation(z,odometerAndOptimizedDistanceDifference,transformTobeSubMapped[5],transPredictionMapped[5]);


        incrementalOdometryAffineBack = trans2Affine3f(transformTobeSubMapped);
    }

    float constraintTransformation(float value, float limit)
    {
        if (value < -limit)
            value = -limit;
        if (value > limit)
            value = limit;

        return value;
    }

    float constraintTransformation(float value, float limit, float now, float pre)
    {

        if(fabs(value)>limit)
        {
            ROS_WARN("Adding is too big !");
            return value;
        }else 
        {
            return now;
        }

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
        laserOdometryROS.pose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(transformTobeSubMapped[0], transformTobeSubMapped[1], transformTobeSubMapped[2]);
        pubLaserOdometryGlobal.publish(laserOdometryROS);
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
        pubLaserOdometryIncremental.publish(laserOdomIncremental);
        // ROS_INFO("Finshed  publishOdometry !");

    }


};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "lis_slam");

    OdomEstimationNode ODN;

    ROS_INFO("\033[1;32m----> Odom EstimationNode Started.\033[0m");

    std::thread odom_estimation_thread(&OdomEstimationNode::OdomEstimationNodeThread, &ODN);

    ros::spin();

    odom_estimation_thread.join();

    return 0;
}