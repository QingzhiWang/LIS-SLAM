
// Author of  EPSC-LOAM : QZ Wang  
// Email wangqingzhi27@outlook.com

#include "utility.h"
#include "lis_slam/cloud_info.h"

#include "lis_slam/loop_info.h"
#include "lis_slam/submap.h"

#include "Scancontext.h"
#include "epscGenerationClass.h"

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



class loopClosure : public ParamServer
{
public:
    ros::Publisher pubLoop;

    ros::Subscriber subMapInfo;

    lis_slam::submap  submap_info;

    ros::Publisher isc_pub;
    ros::Publisher epsc_pub;
    ros::Publisher sc_pub;



    pcl::PointCloud<PointTypePose>::Ptr subMapPose6D; 
    pcl::PointCloud<PointType>::Ptr subMapPose3D;

    pcl::PointCloud<PointTypePose>::Ptr copySubMapPose6D; 
    pcl::PointCloud<PointType>::Ptr copySubMapPose3D;

    pcl::PointCloud<PointType>::Ptr subMapCloudRawLast; 
    pcl::PointCloud<PointType>::Ptr subMapCloudRawLastDS; 

    pcl::PointCloud<PointType>::Ptr subMapCloudCornerLast; 
    pcl::PointCloud<PointType>::Ptr subMapCloudCornerLastDS; 

    pcl::PointCloud<PointType>::Ptr subMapCloudSurfLast; 
    pcl::PointCloud<PointType>::Ptr subMapCloudSurfLastDS; 

    vector<pcl::PointCloud<PointType>::Ptr> cornerCloudSubMap;
    vector<pcl::PointCloud<PointType>::Ptr> surfCloudSubMap;
    vector<pcl::PointCloud<PointType>::Ptr> rawCloudSubMap;

    pcl::VoxelGrid<PointType> downSizeFilterCorner;
    pcl::VoxelGrid<PointType> downSizeFilterSurf;
    pcl::VoxelGrid<PointType> downSizeFilterSubMapICP;

    pcl::KdTreeFLANN<PointType>::Ptr kdtreeCornerFromMap;
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurfFromMap;
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeHistoryKeyPoses;    

    map<int, int> loopIndexContainer; // from new to old
    vector<pair<int, int>> loopIndexQueue;
    vector<Eigen::Affine3f> loopPoseQueue;
    vector<float> loopNoiseQueue;


    ros::Time timeLaserInfoStamp;
    double timeLaserInfoCur;

    std::mutex mtx;

    //Scan-Context
    float yawDiffRad;
     // loop detector 
    SCManager scManager;


    //ISC
    int sector_width = 60;
    int ring_height = 20;
    double max_dis= 60.0;
    double scan_period= 0.1;

    EPSCGenerationClass epscGeneration;


    //test loop
    std::deque<std::pair<int,int>> epscTest;
    std::deque<std::pair<int,int>> iscTest;
    std::deque<std::pair<int,int>> scTest;
    std::deque<std::pair<int,int>> poseTest;

    int poseTestSize=0;
    int epscTestSize=0;
    int iscTestSize=0;

    int poseTestSizeafterICP=0;
    int epscTestSizeafterICP=0;
    int iscTestSizeafterICP=0;


   std::string RESULT_PATH="~/txt/myloop.txt"; //ADD

    
    loopClosure()
    {
        subMapInfo =  nh.subscribe<lis_slam::submap>("lis_slam/mapping/submap_info", 1, &loopClosure::mapInfoHandler, this);

        pubLoop  = nh.advertise<lis_slam::loop_info>("lio_loop/loop_closure_detection", 1);

        isc_pub = nh.advertise<sensor_msgs::Image>("/isc", 100);
        epsc_pub = nh.advertise<sensor_msgs::Image>("/epsc", 100);
        sc_pub = nh.advertise<sensor_msgs::Image>("/sc", 100);

        downSizeFilterCorner.setLeafSize(loopClosureCornerLeafSize, loopClosureCornerLeafSize, loopClosureCornerLeafSize);
        downSizeFilterSurf.setLeafSize(loopClosureSurfLeafSize, loopClosureSurfLeafSize, loopClosureSurfLeafSize);
        downSizeFilterSubMapICP.setLeafSize(subMapLeafSize, subMapLeafSize, subMapLeafSize);

        epscGeneration.init_param(ring_height,sector_width,max_dis);
        allocateMemory();
    }

    void allocateMemory()
    {

        subMapPose6D.reset(new pcl::PointCloud<PointTypePose>()); 
        subMapPose3D.reset(new pcl::PointCloud<PointType>()); 

        copySubMapPose6D.reset(new pcl::PointCloud<PointTypePose>()); 
        copySubMapPose3D.reset(new pcl::PointCloud<PointType>()); 

        subMapCloudRawLast.reset(new pcl::PointCloud<PointType>()); 
        subMapCloudRawLastDS.reset(new pcl::PointCloud<PointType>()); 

        subMapCloudCornerLast.reset(new pcl::PointCloud<PointType>()); 
        subMapCloudCornerLastDS.reset(new pcl::PointCloud<PointType>()); 

        subMapCloudSurfLast.reset(new pcl::PointCloud<PointType>()); 
        subMapCloudSurfLastDS.reset(new pcl::PointCloud<PointType>()); 

        kdtreeCornerFromMap.reset(new pcl::KdTreeFLANN<PointType>());
        kdtreeSurfFromMap.reset(new pcl::KdTreeFLANN<PointType>());
        kdtreeHistoryKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());


    }



     void mapInfoHandler(const lis_slam::submapConstPtr& msgIn)
    {

        timeLaserInfoStamp = msgIn->header.stamp;
        timeLaserInfoCur = msgIn->header.stamp.toSec();

        submap_info = *msgIn;

        pcl::fromROSMsg(msgIn->submap_corner,  *subMapCloudCornerLast);
        pcl::fromROSMsg(msgIn->submap_surface, *subMapCloudSurfLast);
        pcl::fromROSMsg(msgIn->submap_raw, *subMapCloudRawLast);

        std::lock_guard<std::mutex> lock(mtx);


        downsampleCurrentScan();

        saveSubMapInfo();

        //Scan-Context
        pcl::PointCloud<PointType>::Ptr thisRawCloud(new pcl::PointCloud<PointType>());
        pcl::copyPointCloud(*subMapCloudRawLastDS,  *thisRawCloud);
        scManager.makeAndSaveScancontextAndKeys(*thisRawCloud);



        if (loopClosureEnableFlag == false)
            return;

        if (subMapPose3D->points.empty() == true)
            return;

        *copySubMapPose3D = *subMapPose3D;
        *copySubMapPose6D = *subMapPose6D;

         // find keys
        int loopKeyCur;
        int loopKeyPre;

        detectLoopClosureDistanceModelOne(&loopKeyCur, &loopKeyPre);

        std::vector<int> loopKeyPreTest;

        detectLoopClosureDistanceModelFour(&loopKeyCur, &loopKeyPreTest);
        // performLoopClosureModelOne();
        // performLoopClosureModelTwo();
        // performLoopClosureModelThree();

        // performLoopClosureModelFour();

        // sendLoopMessages();

    }


    void downsampleCurrentScan()
    {
        // Downsample cloud from current scan
        subMapCloudCornerLastDS->clear();
        downSizeFilterCorner.setInputCloud(subMapCloudCornerLast);
        downSizeFilterCorner.filter(*subMapCloudCornerLastDS);

        subMapCloudSurfLastDS->clear();
        downSizeFilterSurf.setInputCloud(subMapCloudSurfLast);
        downSizeFilterSurf.filter(*subMapCloudSurfLastDS);

        subMapCloudRawLastDS->clear();
        downSizeFilterSurf.setInputCloud(subMapCloudRawLast);
        downSizeFilterSurf.filter(*subMapCloudRawLastDS);
    }


    void  saveSubMapInfo()
    {
        PointType thisPose3D;
        // thisPose3D.x = submap_info.subMapPoseX;
        // thisPose3D.y = submap_info.subMapPoseY;
        // thisPose3D.z = submap_info.subMapPoseZ;

        thisPose3D.x = submap_info.afterOptmizationSubMapPoseX;
        thisPose3D.y = submap_info.afterOptmizationSubMapPoseY;
        thisPose3D.z = submap_info.afterOptmizationSubMapPoseZ;
        // thisPose3D.intensity = subMapPose3D->size(); // this can be used as index
        thisPose3D.intensity = submap_info.subMapId; // this can be used as index
        subMapPose3D->push_back(thisPose3D);

        PointTypePose thisPose6D;   
        thisPose6D.x = thisPose3D.x;
        thisPose6D.y = thisPose3D.y;
        thisPose6D.z = thisPose3D.z;
        thisPose6D.intensity = thisPose3D.intensity ; // this can be used as index
        // thisPose6D.roll  = submap_info.subMapPoseRoll;
        // thisPose6D.pitch = submap_info.subMapPosePitch;
        // thisPose6D.yaw   = submap_info.subMapPoseYaw;

        thisPose6D.roll  = submap_info.afterOptmizationSubMapPoseRoll;
        thisPose6D.pitch = submap_info.afterOptmizationSubMapPosePitch;
        thisPose6D.yaw   = submap_info.afterOptmizationSubMapPoseYaw;
        thisPose6D.time = timeLaserInfoCur;
        subMapPose6D->push_back(thisPose6D);


        pcl::PointCloud<PointType>::Ptr thisCornerSubMap(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr thisSurfSubMap(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr thisRawSubMap(new pcl::PointCloud<PointType>());
        pcl::copyPointCloud(*subMapCloudCornerLastDS,  *thisCornerSubMap);
        pcl::copyPointCloud(*subMapCloudSurfLastDS,    *thisSurfSubMap);
        pcl::copyPointCloud(*subMapCloudRawLastDS,    *thisRawSubMap);

        // save key frame cloud
        cornerCloudSubMap.push_back(thisCornerSubMap);
        surfCloudSubMap.push_back(thisSurfSubMap);
        rawCloudSubMap.push_back(thisRawSubMap);
    }






    void sendLoopMessages()
    {

        if (loopIndexQueue.empty())
            return;

        lis_slam::loop_info loopInfo;

        loopInfo.header.stamp=timeLaserInfoStamp;

        for (int i = 0; i < (int)loopIndexQueue.size(); ++i)
        {
            loopInfo.loopKeyCur = (int)subMapPose3D->points[loopIndexQueue[i].first].intensity;
            loopInfo.loopKeyPre = (int)subMapPose3D->points[loopIndexQueue[i].second].intensity;

            Eigen::Affine3f  thisLoopPose =loopPoseQueue[i];
            float x, y, z, roll, pitch, yaw;
            pcl::getTranslationAndEulerAngles (thisLoopPose, x, y, z, roll, pitch, yaw);

            loopInfo.loopPoseX = x;
            loopInfo.loopPoseY = y;
            loopInfo.loopPoseZ = z;
            loopInfo.loopPoseRoll = roll;
            loopInfo.loopPosePitch = pitch;
            loopInfo.loopPoseYaw = yaw;

            loopInfo.loopNoise = loopNoiseQueue[i];

            pubLoop.publish(loopInfo);

        }

        loopIndexQueue.clear();
        loopPoseQueue.clear();
        loopNoiseQueue.clear();


    }




    pcl::PointCloud<PointType>::Ptr transformPointCloud(pcl::PointCloud<PointType>::Ptr cloudIn, PointTypePose* transformIn)
    {
        pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

        PointType *pointFrom;

        int cloudSize = cloudIn->size();
        cloudOut->resize(cloudSize);

        Eigen::Affine3f transCur = pcl::getTransformation(transformIn->x, transformIn->y, transformIn->z, transformIn->roll, transformIn->pitch, transformIn->yaw);
    
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



    void findNearSubMaps(const bool isCorner,pcl::PointCloud<PointType>::Ptr& nearKeyframes, const int& key, const int& searchNum)
    {
        // extract near keyframes
        nearKeyframes->clear();
        int cloudSize = subMapPose3D->size();
        for (int i = -searchNum; i <= searchNum; ++i)
        {
            int keyNear = key + i;
            if (keyNear < 0 || keyNear >= cloudSize )
                continue;

            if(isCorner)
            {
                *nearKeyframes += *transformPointCloud(cornerCloudSubMap[keyNear], &subMapPose6D->points[keyNear]);
            }else
            {
                *nearKeyframes += *transformPointCloud(surfCloudSubMap[keyNear],   &subMapPose6D->points[keyNear]);
            }
            
        }

        if (nearKeyframes->empty())
            return;

        // downsample near keyframes
        pcl::PointCloud<PointType>::Ptr cloud_temp(new pcl::PointCloud<PointType>());
        downSizeFilterSubMapICP.setInputCloud(nearKeyframes);
        downSizeFilterSubMapICP.filter(*cloud_temp);
        *nearKeyframes = *cloud_temp;
    }
















    //---------------------------------
    //model 1   原始回环检测 找最近再ICP匹配
    //---------------------------------
    void performLoopClosureModelOne()
    {
        if (subMapPose3D->points.empty() == true)
            return;

        // mtx.lock();
        *copySubMapPose3D = *subMapPose3D;
        *copySubMapPose6D = *subMapPose6D;
        // mtx.unlock();

        // find keys
        int loopKeyCur;
        int loopKeyPre;
        if (detectLoopClosureDistanceModelOne(&loopKeyCur, &loopKeyPre) == false)
            return;

        std::cout << std::endl;
        std::cout << "--- loop detection ---" << std::endl;
        std::cout << "loopKeyCur : " << loopKeyCur<<std::endl;
        std::cout << "loopKeyPre : " << loopKeyPre<<std::endl;
        std::cout << "matching" << std::flush;
        auto t1 = ros::Time::now();
        // extract cloud
        pcl::PointCloud<PointType>::Ptr cureKeyframeCloud(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr prevKeyframeCloud(new pcl::PointCloud<PointType>());

        *cureKeyframeCloud+=*transformPointCloud(cornerCloudSubMap[loopKeyCur], &copySubMapPose6D->points[loopKeyCur]);
        *cureKeyframeCloud+=*transformPointCloud(surfCloudSubMap[loopKeyCur], &copySubMapPose6D->points[loopKeyCur]);

        findNearSubMaps(true,prevKeyframeCloud, loopKeyPre, 7);
        findNearSubMaps(false,prevKeyframeCloud, loopKeyPre, 5);

        if (cureKeyframeCloud->size() < 300 || prevKeyframeCloud->size() < 1000)
            return;


        // ICP Settings
        static pcl::IterativeClosestPoint<PointType, PointType> icp;
        icp.setMaxCorrespondenceDistance(historyKeyframeSearchRadius*2);
        icp.setMaximumIterations(100);
        icp.setTransformationEpsilon(1e-6);
        icp.setEuclideanFitnessEpsilon(1e-6);
        icp.setRANSACIterations(0);

        // Align clouds
        icp.setInputSource(cureKeyframeCloud);
        icp.setInputTarget(prevKeyframeCloud);
        pcl::PointCloud<PointType>::Ptr unused_result(new pcl::PointCloud<PointType>());
        icp.align(*unused_result);
        
        auto t2 = ros::Time::now();
        std::cout << " done" << std::endl;
        std::cout << "score: " <<  icp.getFitnessScore() << "    time: " << (t2 - t1).toSec() << "[sec]" << std::endl;

        if (icp.hasConverged() == false || icp.getFitnessScore() > historyKeyframeFitnessScore)
        {
            std::cout << "loop not found..." << std::endl;
            return;
        }    
        std::cout << "loop  found..." << std::endl;


        Eigen::Affine3f correctionLidarFrame;
        correctionLidarFrame = icp.getFinalTransformation();
        float noiseScore = icp.getFitnessScore();

        // Add pose constraint
        // mtx.lock();
        loopIndexQueue.push_back(make_pair(loopKeyCur, loopKeyPre));
        loopPoseQueue.push_back(correctionLidarFrame);
        loopNoiseQueue.push_back(noiseScore);
        // mtx.unlock();

        // add loop constriant
        loopIndexContainer[loopKeyCur] = loopKeyPre;
    }

    bool detectLoopClosureDistanceModelOne(int *latestID, int *closestID)
    {
        int loopKeyCur = copySubMapPose3D->size() - 1;
        int loopKeyPre = -1;

        // check loop constraint added before
        auto it = loopIndexContainer.find(loopKeyCur);
        if (it != loopIndexContainer.end())
            return false;

        // find the closest history key frame
        std::vector<int> pointSearchIndLoop;
        std::vector<float> pointSearchSqDisLoop;
        kdtreeHistoryKeyPoses->setInputCloud(copySubMapPose3D);
        kdtreeHistoryKeyPoses->radiusSearch(copySubMapPose3D->back(), historyKeyframeSearchRadius, pointSearchIndLoop, pointSearchSqDisLoop, 0);
        
        for (int i = 0; i < (int)pointSearchIndLoop.size(); ++i)
        {
            int id = pointSearchIndLoop[i];
            if (abs(copySubMapPose6D->points[id].time - timeLaserInfoCur) > historyKeyframeSearchTimeDiff)
            {
                loopKeyPre = id;
                break;
            }
        }

        if (loopKeyPre == -1 || loopKeyCur == loopKeyPre)
            return false;

        *latestID = loopKeyCur;
        *closestID = loopKeyPre;



        poseTest.push_back(std::make_pair(loopKeyCur,loopKeyPre));

         poseTestSize++;

        return true;
    }
















   //---------------------------------
    //model 2  较model1增加约束 (HDL)
    //---------------------------------
    void performLoopClosureModelTwo()
    {

        if (subMapPose3D->points.empty() == true)
            return;

        // mtx.lock();
        *copySubMapPose3D = *subMapPose3D;
        *copySubMapPose6D = *subMapPose6D;
        // mtx.unlock();

        // find keys
        static int lastLoopIndex=0;

        int loopKeyCur;
        std::vector<int> loopKeyPre;
        loopKeyPre.clear();

        if (detectLoopClosureDistanceModelTwo(lastLoopIndex,&loopKeyCur, &loopKeyPre) == false)
            return;

        if(loopKeyPre.empty()) {
            return;
        }



        pcl::PointCloud<PointType>::Ptr cureKeyframeCloud(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr prevKeyframeCloud(new pcl::PointCloud<PointType>());

        *cureKeyframeCloud+=*transformPointCloud(cornerCloudSubMap[loopKeyCur], &copySubMapPose6D->points[loopKeyCur]);
        *cureKeyframeCloud+=*transformPointCloud(surfCloudSubMap[loopKeyCur], &copySubMapPose6D->points[loopKeyCur]);


        if (cureKeyframeCloud->size() < 300 )
            return;

        std::cout << std::endl;
        std::cout << "--- loop detection ---" << std::endl;
        std::cout << "num_candidates: " << loopKeyPre.size() << std::endl;
        std::cout << "matching" << std::flush;
        auto t1 = ros::Time::now();
        // ICP Settings
        static pcl::IterativeClosestPoint<PointType, PointType> icp;
        icp.setMaxCorrespondenceDistance(historyKeyframeSearchRadius*2);
        icp.setMaximumIterations(100);
        icp.setTransformationEpsilon(1e-6);
        icp.setEuclideanFitnessEpsilon(1e-6);
        icp.setRANSACIterations(0);

        // Align clouds
        icp.setInputSource(cureKeyframeCloud);
        
        int bestMatched;
        double bestScore = std::numeric_limits<double>::max();
        Eigen::Affine3f correctionLidarFrame;
        int loopCandidatesSize=loopKeyPre.size();
        if(loopCandidatesSize>10)
            loopCandidatesSize=10;
        for(int i=0;i<loopCandidatesSize;i++)
        {
            int id=loopKeyPre[i];
            prevKeyframeCloud->clear();

            findNearSubMaps(true,prevKeyframeCloud, id, 7);
            findNearSubMaps(false,prevKeyframeCloud, id, 5);
            if (prevKeyframeCloud->size() < 1000)
                continue;
            icp.setInputTarget(prevKeyframeCloud);
            pcl::PointCloud<PointType>::Ptr unused_result(new pcl::PointCloud<PointType>());
            icp.align(*unused_result);

            double score = icp.getFitnessScore();
            if (icp.hasConverged() == false || score > bestScore)
                continue;
            bestScore = score;
            bestMatched = id;
            correctionLidarFrame = icp.getFinalTransformation();
        }

        auto t2 = ros::Time::now();
        std::cout << " done" << std::endl;
        std::cout << "best_score: " <<  bestScore << "    time: " << (t2 - t1).toSec() << "[sec]" << std::endl;

        if(bestScore > historyKeyframeFitnessScore) {
            std::cout << "loop not found..." << std::endl;
            return;
        }
        std::cout << "loop found!!" << std::endl;


        // Add pose constraint
        // mtx.lock();
        loopIndexQueue.push_back(make_pair(loopKeyCur, bestMatched));
        loopPoseQueue.push_back(correctionLidarFrame);
        loopNoiseQueue.push_back(bestScore);
        // mtx.unlock();

        // add loop constriant
        loopIndexContainer[loopKeyCur] = bestMatched;
        //update last loop index
        lastLoopIndex=loopKeyCur;
    }

    bool detectLoopClosureDistanceModelTwo(int lastLoopIndex ,int *latestID, std::vector<int> *closestID)
    {

        int loopKeyCur = copySubMapPose3D->size() - 1;
        int loopKeyPre = -1;

        // check loop constraint added before
        auto it = loopIndexContainer.find(loopKeyCur);
        if (it != loopIndexContainer.end())
            return false;

        std::vector<int> candidates;
        std::vector<int> endCandidates;

        // too close to the last registered loop edge
        if(loopKeyCur- lastLoopIndex < distanceFromLastIndexThresh) {
             std::cout << "too close to the last registered loop ..." << std::endl;
            return false;
        }

        // find the closest history key frame
        std::vector<int> pointSearchIndLoop;
        std::vector<float> pointSearchSqDisLoop;
        kdtreeHistoryKeyPoses->setInputCloud(copySubMapPose3D);
        kdtreeHistoryKeyPoses->radiusSearch(copySubMapPose3D->back(), historyKeyframeSearchRadius, pointSearchIndLoop, pointSearchSqDisLoop, 0);
        

        for (int i = 0; i < (int)pointSearchIndLoop.size(); ++i)
        {
            int id = pointSearchIndLoop[i];
            loopKeyPre = id;

            // traveled distance between keyframes is too small
            if(loopKeyCur - loopKeyPre < accumDistanceIndexThresh) {
                 std::cout << "traveled distance between keyframes is too small ..." << std::endl;
                continue;
            }
           
            if (abs(copySubMapPose6D->points[loopKeyPre].time - timeLaserInfoCur) < historyKeyframeSearchTimeDiff) {
                std::cout << "historyKeyframeSearchTimeDiff is too small ..." << std::endl;
                continue;
            }

            //check distance between keyframes is too small
            bool _flag=false;
            for(int i=0;i<candidates.size();i++)
            {
                if(abs(loopKeyPre-candidates[i])<historyAccumDistanceIndexThresh)
                    _flag=true;
            }
            if(!_flag)
            {
                candidates.push_back(loopKeyPre);
                _flag=false;
            }
    
        }

        for(int i=0;i<candidates.size();i++){
            std::cout << "candidates [" << i<<"]:"<<candidates[i]<<std::endl;
        }

        *latestID = loopKeyCur;
        *closestID = candidates;

        return true;
    }
















 //---------------------------------
    //model 3   SCAN-CONTEXT
    //---------------------------------
    void performLoopClosureModelThree()
    {
        if (subMapPose3D->points.empty() == true)
            return;

        // mtx.lock();
        *copySubMapPose3D = *subMapPose3D;
        *copySubMapPose6D = *subMapPose6D;
        // mtx.unlock();

        // find keys
        int loopKeyCur;
        std::vector<int> loopKeyPre;

        if (detectLoopClosureDistanceModelThree(&loopKeyCur, &loopKeyPre) == false)
            return;

        std::cout << std::endl;
        std::cout << "--- loop detection ---" << std::endl;
        std::cout << "num_candidates: " << loopKeyPre.size() << std::endl;
        std::cout << "loopKeyCur : " << loopKeyCur<<std::endl;
        std::cout << "subMapPose3D : " << copySubMapPose3D->size() - 1<<std::endl;
        for(int i=0;i<loopKeyPre.size();i++){
            std::cout << "loopKeyPre [" << i<<"]:"<<loopKeyPre[i]<<std::endl;
        }
        std::cout << "matching..." << std::flush;
        auto t1 = ros::Time::now();

        pcl::PointCloud<PointType>::Ptr cureKeyframeCloud(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr prevKeyframeCloud(new pcl::PointCloud<PointType>());

        *cureKeyframeCloud+=*transformPointCloud(cornerCloudSubMap[loopKeyCur], &copySubMapPose6D->points[loopKeyCur]);
        *cureKeyframeCloud+=*transformPointCloud(surfCloudSubMap[loopKeyCur], &copySubMapPose6D->points[loopKeyCur]);


        if (cureKeyframeCloud->size() < 300 )
            return;
        // ICP Settings
        static pcl::IterativeClosestPoint<PointType, PointType> icp;
        icp.setMaxCorrespondenceDistance(historyKeyframeSearchRadius*2);
        icp.setMaximumIterations(50);
        icp.setTransformationEpsilon(1e-6);
        icp.setEuclideanFitnessEpsilon(1e-6);
        icp.setRANSACIterations(0);

        // Align clouds
        icp.setInputSource(cureKeyframeCloud);
        
        int bestMatched;
        double bestScore = std::numeric_limits<double>::max();
        Eigen::Affine3f correctionLidarFrame;
        int loopCandidatesSize=loopKeyPre.size();
        if(loopCandidatesSize>10)
            loopCandidatesSize=10;
        for(int i=0;i<loopCandidatesSize;i++)
        {
            int id=loopKeyPre[i];

            prevKeyframeCloud->clear();
            findNearSubMaps(true,prevKeyframeCloud, id, 7);
            findNearSubMaps(false,prevKeyframeCloud, id, 5);
            if (prevKeyframeCloud->size() < 1000)
                continue;

            icp.setInputTarget(prevKeyframeCloud);
            pcl::PointCloud<PointType>::Ptr unused_result(new pcl::PointCloud<PointType>());
            icp.align(*unused_result);

            double score = icp.getFitnessScore();
            if (icp.hasConverged() == false || score > bestScore)
                continue;
            bestScore = score;
            bestMatched = id;
            correctionLidarFrame = icp.getFinalTransformation();
        }

        auto t2 = ros::Time::now();
        std::cout << " done" << std::endl;
        std::cout << "best_score: " <<  bestScore << "    time: " <<  (t2 - t1).toSec() << "[sec]" << std::endl;

        if(bestScore > historyKeyframeFitnessScore) {
            std::cout << "loop not found..." << std::endl;
            return;
        }
        std::cout << "loop found!!" << std::endl;


        // Add pose constraint
        // mtx.lock();
        loopIndexQueue.push_back(make_pair(loopKeyCur, bestMatched));
        loopPoseQueue.push_back(correctionLidarFrame);
        loopNoiseQueue.push_back(bestScore);
        // mtx.unlock();

        // add loop constriant
        loopIndexContainer[loopKeyCur] = bestMatched;

    }


    bool detectLoopClosureDistanceModelThree(int *latestID, std::vector<int> *closestID)
    {
         //set param

        // int accumDistanceIndexThresh;           // traveled distance between ...
        // int historyAccumDistanceIndexThresh;     
        // int distanceFromLastIndexThresh;  // a new loop edge must far from the last one at least this distance
        
        // accumDistanceIndexThresh=10;           
        // distanceFromLastIndexThresh=5;  
        // historyAccumDistanceIndexThresh=5;

        int loopKeyCur = copySubMapPose3D->size() - 1;

        std::vector<int> loopKeyPre;
        std::vector<int> yawDiffRad;

        auto detectResult = scManager.detectLoopClosureID(); // first: nn index, second: yaw diff 
        for(int i=0;i<detectResult.size();i++){
            // if all close, reject
            if (detectResult[i].first <=0|| loopKeyCur == detectResult[i].first){ 
              continue;
            }

            bool _flag=false;
            for(int j=0;j<loopKeyPre.size();j++)
            {
                if(abs(detectResult[i].first-loopKeyPre[j])<historyAccumDistanceIndexThresh)
                    _flag=true;
            }
            if(!_flag)
            {
                loopKeyPre.push_back(detectResult[i].first);
                yawDiffRad.push_back(detectResult[i].second); // not use for v1 (because pcl icp withi initial somthing wrong...)
                _flag=false;
            }
            
        }

        if(loopKeyPre.empty())
            return false;
        
        // std::cout << "loopKeyCur : " << loopKeyCur<<std::endl;
        // std::cout << "loopKeyPre : " << loopKeyPre<<std::endl;
    
        *latestID = loopKeyCur;
        *closestID = loopKeyPre;

        

        return true;
    }









 //---------------------------------
    //model 4  Intensity SCAN-CONTEXT
    //---------------------------------
    void performLoopClosureModelFour()
    {

        *copySubMapPose3D = *subMapPose3D;
        *copySubMapPose6D = *subMapPose6D;

        // find keys
        int loopKeyCur;
        std::vector<int> loopKeyPre;

        if (detectLoopClosureDistanceModelFour(&loopKeyCur, &loopKeyPre) == false)
            return;

        std::cout << std::endl;
        std::cout << "--- loop detection ---" << std::endl;
        std::cout << "num_candidates: " << loopKeyPre.size() << std::endl;
        std::cout << "loopKeyCur : " << loopKeyCur<<std::endl;
        std::cout << "subMapPose3D : " << copySubMapPose3D->size() - 1<<std::endl;
        for(int i=0;i<loopKeyPre.size();i++){
            std::cout << "loopKeyPre [" << i<<"]:"<<loopKeyPre[i]<<std::endl;
        }
        std::cout << "matching..." << std::flush;
        auto t1 = ros::Time::now();

        pcl::PointCloud<PointType>::Ptr cureKeyframeCloud(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr prevKeyframeCloud(new pcl::PointCloud<PointType>());

        *cureKeyframeCloud+=*transformPointCloud(cornerCloudSubMap[loopKeyCur], &copySubMapPose6D->points[loopKeyCur]);
        *cureKeyframeCloud+=*transformPointCloud(surfCloudSubMap[loopKeyCur], &copySubMapPose6D->points[loopKeyCur]);



        if (cureKeyframeCloud->size() < 300 )
            return;
        // ICP Settings
        static pcl::IterativeClosestPoint<PointType, PointType> icp;
        icp.setMaxCorrespondenceDistance(historyKeyframeSearchRadius*2);
        icp.setMaximumIterations(50);
        icp.setTransformationEpsilon(1e-6);
        icp.setEuclideanFitnessEpsilon(1e-6);
        icp.setRANSACIterations(0);

        // Align clouds
        icp.setInputSource(cureKeyframeCloud);
        
        int bestMatched;
        double bestScore = std::numeric_limits<double>::max();
        Eigen::Affine3f correctionLidarFrame;
        int loopCandidatesSize=loopKeyPre.size();
        if(loopCandidatesSize>10)
            loopCandidatesSize=10;
        for(int i=0;i<loopCandidatesSize;i++)
        {
            int id=loopKeyPre[i];

            prevKeyframeCloud->clear();
            // *prevKeyframeCloud+=*transformPointCloud(cornerCloudSubMap[id], &copySubMapPose6D->points[id]);
            // *prevKeyframeCloud+=*transformPointCloud(surfCloudSubMap[id], &copySubMapPose6D->points[id]);
            findNearSubMaps(true,prevKeyframeCloud, id, 7);
            findNearSubMaps(false,prevKeyframeCloud, id, 5);
            if (prevKeyframeCloud->size() < 1000)
                continue;

            icp.setInputTarget(prevKeyframeCloud);
            pcl::PointCloud<PointType>::Ptr unused_result(new pcl::PointCloud<PointType>());
            icp.align(*unused_result);

            double score = icp.getFitnessScore();
            if (icp.hasConverged() == false || score > bestScore)
                continue;
            bestScore = score;
            bestMatched = id;
            correctionLidarFrame = icp.getFinalTransformation();
        }

        auto t2 = ros::Time::now();
        std::cout << " done" << std::endl;
        std::cout << "best_score: " <<  bestScore << "    time: " <<  (t2 - t1).toSec() << "[sec]" << std::endl;

        if(bestScore > historyKeyframeFitnessScore) {
            std::cout << "loop not found..." << std::endl;
            return;
        }
        std::cout << "loop found!!" << std::endl;


        // Add pose constraint
        // mtx.lock();
        loopIndexQueue.push_back(make_pair(loopKeyCur, bestMatched));
        loopPoseQueue.push_back(correctionLidarFrame);
        loopNoiseQueue.push_back(bestScore*0.5);
        // loopNoiseQueue.push_back(0.3);
        // mtx.unlock();

        // add loop constriant
        loopIndexContainer[loopKeyCur] = bestMatched;

    }


    bool detectLoopClosureDistanceModelFour(int *latestID, std::vector<int> *closestID)
    {

        ros::Time detectLoopStart=ros::Time::now();

        pcl::PointCloud<pcl::PointXYZI>::Ptr raw_pointcloud_in(new pcl::PointCloud<pcl::PointXYZI>());
        pcl::copyPointCloud(*subMapCloudRawLastDS,    *raw_pointcloud_in);
        pcl::PointCloud<pcl::PointXYZI>::Ptr corner_pointcloud_in(new pcl::PointCloud<pcl::PointXYZI>());
        pcl::copyPointCloud(*subMapCloudCornerLast,    *corner_pointcloud_in);
        pcl::PointCloud<pcl::PointXYZI>::Ptr surf_pointcloud_in(new pcl::PointCloud<pcl::PointXYZI>());
        pcl::copyPointCloud(*subMapCloudSurfLast,    *surf_pointcloud_in);

        ros::Time pointcloud_time = timeLaserInfoStamp;

        int loopKeyCur = copySubMapPose3D->size() - 1;
        ROS_INFO("ISC: loopKeyCur: %d",loopKeyCur);

        nav_msgs::Odometry laserOdometryROS;
        laserOdometryROS.pose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(copySubMapPose6D->points[loopKeyCur].roll, 
                                                                                                                                                                                            copySubMapPose6D->points[loopKeyCur].pitch, 
                                                                                                                                                                                            copySubMapPose6D->points[loopKeyCur].yaw);
        
        Eigen::Isometry3d odom_in = Eigen::Isometry3d::Identity();        
        odom_in.rotate(Eigen::Quaterniond(laserOdometryROS.pose.pose.orientation.w,laserOdometryROS.pose.pose.orientation.x,laserOdometryROS.pose.pose.orientation.y,laserOdometryROS.pose.pose.orientation.z));  
        odom_in.pretranslate(Eigen::Vector3d(copySubMapPose6D->points[loopKeyCur].x,copySubMapPose6D->points[loopKeyCur].y,copySubMapPose6D->points[loopKeyCur].z));

        epscGeneration.loopDetection(corner_pointcloud_in, surf_pointcloud_in, raw_pointcloud_in, odom_in);

        std::vector<int> loopKeyPre;

        int cur_id=epscGeneration.current_frame_id;
        // ROS_INFO("ISC: cur_id: %d",cur_id);

        for(int i=0;i<(int)epscGeneration.matched_frame_id.size();i++){
            loopKeyPre.push_back(epscGeneration.matched_frame_id[i]);
        }

        cv_bridge::CvImage out_msg;
        out_msg.header.frame_id  = lidarFrame; 
        out_msg.header.stamp  = pointcloud_time; 
        out_msg.encoding = sensor_msgs::image_encodings::RGB8; 
        out_msg.image    = epscGeneration.getLastISCRGB(); 
        isc_pub.publish(out_msg.toImageMsg());

        out_msg.image    = epscGeneration.getLastEPSCRGB(); 
        epsc_pub.publish(out_msg.toImageMsg());

        // out_msg.image    = epscGeneration.getLastSCRGB(); 
        // sc_pub.publish(out_msg.toImageMsg());


        ros::Time detectLoopEnd=ros::Time::now();
        ROS_WARN("Detect Loop Closure Model Four Time: %.3f",(detectLoopEnd-detectLoopStart).toSec());
        
        if(loopKeyPre.empty())
        {
            ROS_WARN("loopKeyPre is empty !");
            return false;
        }
         
         
    
        // *latestID = loopKeyCur;
        *latestID = cur_id;
        *closestID = loopKeyPre;

        epscTest.push_back(std::make_pair(cur_id,loopKeyPre[0]));

        epscTestSize++;



        std::vector<int>  iscLoopKeyPre;

        for(int i=0;i<(int)epscGeneration.isc_matched_frame_id.size();i++){
            iscLoopKeyPre.push_back(epscGeneration.isc_matched_frame_id[i]);
        }

        if(iscLoopKeyPre.empty())
        {
            ROS_WARN("loopKeyPre is empty !");
            return false;
        }

        iscTest.push_back(std::make_pair(cur_id,iscLoopKeyPre[0]));
        iscTestSize++;



        return true;
    }














    //---------------------------------
    //test  loop
    //---------------------------------
    void testLoopIcp()
    {

        while (ros::ok()){

            if(poseTest.size()!=0){
                int loopKeyCur=poseTest.front().first;
                int loopKeyPre=poseTest.front().second;

                poseTest.pop_front();

                std::cout << "pose matching" << std::flush;
                auto t1 = ros::Time::now();
                // extract cloud
                pcl::PointCloud<PointType>::Ptr cureKeyframeCloud(new pcl::PointCloud<PointType>());
                pcl::PointCloud<PointType>::Ptr prevKeyframeCloud(new pcl::PointCloud<PointType>());

                *cureKeyframeCloud+=*transformPointCloud(cornerCloudSubMap[loopKeyCur], &subMapPose6D->points[loopKeyCur]);
                *cureKeyframeCloud+=*transformPointCloud(surfCloudSubMap[loopKeyCur], &subMapPose6D->points[loopKeyCur]);

                findNearSubMaps(true,prevKeyframeCloud, loopKeyPre, 7);
                findNearSubMaps(false,prevKeyframeCloud, loopKeyPre, 5);

                if (cureKeyframeCloud->size() < 300 || prevKeyframeCloud->size() < 1000)
                    return;


                // ICP Settings
                static pcl::IterativeClosestPoint<PointType, PointType> icp;
                icp.setMaxCorrespondenceDistance(historyKeyframeSearchRadius*2);
                icp.setMaximumIterations(100);
                icp.setTransformationEpsilon(1e-6);
                icp.setEuclideanFitnessEpsilon(1e-6);
                icp.setRANSACIterations(0);

                // Align clouds
                icp.setInputSource(cureKeyframeCloud);
                icp.setInputTarget(prevKeyframeCloud);
                pcl::PointCloud<PointType>::Ptr unused_result(new pcl::PointCloud<PointType>());
                icp.align(*unused_result);
                
                auto t2 = ros::Time::now();
                std::cout << " done" << std::endl;
                std::cout << "score: " <<  icp.getFitnessScore() << "    time: " << (t2 - t1).toSec() << "[sec]" << std::endl;

                if (icp.hasConverged() == false || icp.getFitnessScore() > historyKeyframeFitnessScore)
                {
                    std::cout << "loop not found..." << std::endl;
                    continue;
                }    

                poseTestSizeafterICP++;
                std::cout << "loop  found..." << std::endl;

            }



            if(epscTest.size()!=0){


                int loopKeyCur=epscTest.front().first;
                int loopKeyPre=epscTest.front().second;

                epscTest.pop_front();

                std::cout << "Epsc matching" << std::flush;
                auto t1 = ros::Time::now();
                // extract cloud
                pcl::PointCloud<PointType>::Ptr cureKeyframeCloud(new pcl::PointCloud<PointType>());
                pcl::PointCloud<PointType>::Ptr prevKeyframeCloud(new pcl::PointCloud<PointType>());

                *cureKeyframeCloud+=*transformPointCloud(cornerCloudSubMap[loopKeyCur], &subMapPose6D->points[loopKeyCur]);
                *cureKeyframeCloud+=*transformPointCloud(surfCloudSubMap[loopKeyCur], &subMapPose6D->points[loopKeyCur]);

                findNearSubMaps(true,prevKeyframeCloud, loopKeyPre, 7);
                findNearSubMaps(false,prevKeyframeCloud, loopKeyPre, 5);

                if (cureKeyframeCloud->size() < 300 || prevKeyframeCloud->size() < 1000)
                    return;


                // ICP Settings
                static pcl::IterativeClosestPoint<PointType, PointType> icp;
                icp.setMaxCorrespondenceDistance(historyKeyframeSearchRadius*2);
                icp.setMaximumIterations(100);
                icp.setTransformationEpsilon(1e-6);
                icp.setEuclideanFitnessEpsilon(1e-6);
                icp.setRANSACIterations(0);

                // Align clouds
                icp.setInputSource(cureKeyframeCloud);
                icp.setInputTarget(prevKeyframeCloud);
                pcl::PointCloud<PointType>::Ptr unused_result(new pcl::PointCloud<PointType>());
                icp.align(*unused_result);
                
                auto t2 = ros::Time::now();
                std::cout << " done" << std::endl;
                std::cout << "score: " <<  icp.getFitnessScore() << "    time: " << (t2 - t1).toSec() << "[sec]" << std::endl;

                if (icp.hasConverged() == false || icp.getFitnessScore() > historyKeyframeFitnessScore)
                {
                    std::cout << "loop not found..." << std::endl;
                    continue;
                }    

                epscTestSizeafterICP++;
                
                std::cout << "loop  found..." << std::endl;
                
            }



            if(iscTest.size()!=0){


                int loopKeyCur=iscTest.front().first;
                int loopKeyPre=iscTest.front().second;
                iscTest.pop_front();

                std::cout << "Isc matching" << std::flush;
                auto t1 = ros::Time::now();
                // extract cloud
                pcl::PointCloud<PointType>::Ptr cureKeyframeCloud(new pcl::PointCloud<PointType>());
                pcl::PointCloud<PointType>::Ptr prevKeyframeCloud(new pcl::PointCloud<PointType>());

                *cureKeyframeCloud+=*transformPointCloud(cornerCloudSubMap[loopKeyCur], &subMapPose6D->points[loopKeyCur]);
                *cureKeyframeCloud+=*transformPointCloud(surfCloudSubMap[loopKeyCur], &subMapPose6D->points[loopKeyCur]);

                findNearSubMaps(true,prevKeyframeCloud, loopKeyPre, 7);
                findNearSubMaps(false,prevKeyframeCloud, loopKeyPre, 5);

                if (cureKeyframeCloud->size() < 300 || prevKeyframeCloud->size() < 1000)
                    return;


                // ICP Settings
                static pcl::IterativeClosestPoint<PointType, PointType> icp;
                icp.setMaxCorrespondenceDistance(historyKeyframeSearchRadius*2);
                icp.setMaximumIterations(100);
                icp.setTransformationEpsilon(1e-6);
                icp.setEuclideanFitnessEpsilon(1e-6);
                icp.setRANSACIterations(0);

                // Align clouds
                icp.setInputSource(cureKeyframeCloud);
                icp.setInputTarget(prevKeyframeCloud);
                pcl::PointCloud<PointType>::Ptr unused_result(new pcl::PointCloud<PointType>());
                icp.align(*unused_result);
                
                auto t2 = ros::Time::now();
                std::cout << " done" << std::endl;
                std::cout << "score: " <<  icp.getFitnessScore() << "    time: " << (t2 - t1).toSec() << "[sec]" << std::endl;

                if (icp.hasConverged() == false || icp.getFitnessScore() > historyKeyframeFitnessScore)
                {
                    std::cout << "loop not found..." << std::endl;
                    continue;
                }    

                iscTestSizeafterICP++;
                std::cout << "loop  found..." << std::endl;
                
            }


            // ROS_WARN("EPSC_SIZE: %d",epscTestSize);

            // ROS_WARN("Pose_SIZE: %d",poseTestSize);
            // ROS_WARN("ISC_SIZE: %d",iscTestSize);

            // ROS_WARN("EPSC_ICP_SIZE: %d",epscTestSizeafterICP);

            // ROS_WARN("Pose_ICP_SIZE: %d",poseTestSizeafterICP);
            // ROS_WARN("ISC_ICP_SIZE: %d",iscTestSizeafterICP);


        }


            std::ofstream foutC(RESULT_PATH, std::ios::app);

            foutC.setf(std::ios::scientific, std::ios::floatfield);
            foutC.precision(6);

            foutC <<epscTestSize<< endl ;
            foutC <<poseTestSize<< endl ;	
            foutC <<iscTestSize<< endl ;	

            foutC <<epscTestSizeafterICP<< endl ;	
            foutC <<poseTestSizeafterICP<< endl ;	
            foutC <<iscTestSizeafterICP<< endl ;		
        

            foutC.close();

    }








};


int main(int argc, char** argv)
{
    ros::init(argc, argv, "lis_slam");

    loopClosure LC;

    ROS_INFO("\033[1;32m----> Loop Closure Started.\033[0m");

    std::thread testLoopThread(&loopClosure::testLoopIcp, &LC);
    
    ros::spin();

    testLoopThread.join();



    return 0;
}