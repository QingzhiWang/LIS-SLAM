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
std::deque<keyframe_Ptr> subMapQueue;

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

    ros::Publisher pubCloudRegisteredRaw;
    ros::Publisher pubCloudCurSubMap;
    ros::Publisher pubCloudMap;

    ros::Publisher pubSubMapOdometryGlobal;
    ros::Publisher pubKeyFrameOdometryGlobal;

    ros::Publisher pubKeyFramePoseGlobal;
    ros::Publisher pubKeyFramePath;
    ros::Publisher pubSubMapConstraintEdge;

    ros::Publisher pubLoopConstraintEdge;
    ros::Publisher pubSCDe;
    ros::Publisher pubEPSC;
    ros::Publisher pubSC;
    ros::Publisher pubISC;
    ros::Publisher pubSSC;

    void allocateMemory() {}
    
    SubMapOdometryNode() {
        subCloud = nh.subscribe<lis_slam::semantic_info>( "lis_slam/semantic_fusion/semantic_info", 10, &SubMapOptmizationNode::semanticInfoHandler, this, ros::TransportHints().tcpNoDelay());
        subImu = nh.subscribe<sensor_msgs::Imu>(imuTopic, 2000, &IMUPreintegration::imuHandler, this, ros::TransportHints().tcpNoDelay());
        
        pubSubMapOdometryGlobal = nh.advertise<nav_msgs::Odometry>("lis_slam/mapping/submap_odometry", 1);
        pubKeyFrameOdometryGlobal = nh.advertise<nav_msgs::Odometry>("lis_slam/mapping/keyframe_odometry", 1);

        pubKeyFramePoseGlobal = nh.advertise<sensor_msgs::PointCloud2>("lis_slam/mapping/trajectory", 1);
        pubKeyFramePath = nh.advertise<nav_msgs::Path>("lis_slam/mapping/keyframe_path", 1);
        pubSubMapConstraintEdge = nh.advertise<visualization_msgs::MarkerArray>("lis_slam/mapping/submap_constraints", 1);

        pubCloudRegisteredRaw = nh.advertise<sensor_msgs::PointCloud2>("lis_slam/mapping/registered_raw", 1);
        pubCloudCurSubMap = nh.advertise<sensor_msgs::PointCloud2>("lis_slam/mapping/submap", 1);
        pubCloudMap = nh.advertise<sensor_msgs::PointCloud2>("lis_slam/mapping/map_global", 1);

        pubLoopConstraintEdge = nh.advertise<visualization_msgs::MarkerArray>("lis_slam/mapping/loop_closure_constraints", 1);
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
        int  curSubMapSize = 0;

        while(ros::ok()){
            if (seInfoQueue.size() > 0) {
                std::chrono::time_point<std::chrono::system_clock> start, end;
                start = std::chrono::system_clock::now();
                
                std::lock_guard<std::mutex> lock(seMtx);

                lis_slam::semantic_info currentInfoMsg = seInfoQueue.front();
                seInfoQueue.pop_front();

                keyframe_Ptr currentKeyFrame(new keyframe_t());

                keyframeInit(currentKeyFrame, currentInfoMsg);
                updateInitialGuess();

                if(subMapFirstFlag){
                    firstMakeSubMap();
                    publishOdometry();
                    subMapFirstFlag=false;
                    continue;
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

    void keyframeInit(keyframe_Ptr &currentKeyFrame, lis_slam::semantic_info &currentInfoMsg){
        currentKeyFrame->keyframe_id = keyFrameID;
        currentKeyFrame->timeInfoStamp = currentInfoMsg.header.stamp;
        currentKeyFrame->submap_id = -1;
        currentKeyFrame->id_in_submap = -1;

        if (currentInfoMsg.odomAvailable == true) {
            currentKeyFrame->init_pose = pcl::getTransformation(currentInfoMsg.initialGuessX, currentInfoMsg.initialGuessY, currentInfoMsg.initialGuessZ, 
                                                                currentInfoMsg.initialGuessRoll, currentInfoMsg.initialGuessPitch, currentInfoMsg.initialGuessYaw);
        } else if (currentInfoMsg.imuAvailable == true) {
            currentKeyFrame->init_pose = pcl::getTransformation(0, 0, 0, 
                                                                currentInfoMsg.imuRollInit, 
                                                                currentInfoMsg.imuPitchInit, 
                                                                currentInfoMsg.imuYawInit);
        }
        
        pcl::PointCloud<PointXYZIL>::Ptr semantic_pointcloud_in(new pcl::PointCloud<PointXYZIL>());
        pcl::PointCloud<PointXYZIL>::Ptr dynamic_pointcloud_in(new pcl::PointCloud<PointXYZIL>());
        pcl::PointCloud<PointXYZIL>::Ptr static_pointcloud_in(new pcl::PointCloud<PointXYZIL>());
        pcl::PointCloud<PointXYZIL>::Ptr outlier_pointcloud_in(new pcl::PointCloud<PointXYZIL>());

        pcl::PointCloud<pcl::PointXYZI>::Ptr corner_pointcloud_in(new pcl::PointCloud<pcl::PointXYZI>());
        pcl::PointCloud<pcl::PointXYZI>::Ptr surf_pointcloud_in(new pcl::PointCloud<pcl::PointXYZI>());

        pcl::fromROSMsg(currentInfoMsg.cloud_semantic, *semantic_pointcloud_in);
        pcl::fromROSMsg(currentInfoMsg.cloud_dynamic, *dynamic_pointcloud_in);
        pcl::fromROSMsg(currentInfoMsg.cloud_static, *static_pointcloud_in);
        pcl::fromROSMsg(currentInfoMsg.cloud_outlier, *outlier_pointcloud_in);

        pcl::fromROSMsg(currentInfoMsg.cloud_corner, *corner_pointcloud_in);
        pcl::fromROSMsg(currentInfoMsg.cloud_surface, *surf_pointcloud_in);

        currentKeyFrame->cloud_semantic = semantic_pointcloud_in;

        currentKeyFrame->cloud_dynamic = dynamic_pointcloud_in;
        currentKeyFrame->cloud_static = static_pointcloud_in;
        currentKeyFrame->cloud_outlier = outlier_pointcloud_in;

        currentKeyFrame->cloud_corner = corner_pointcloud_in;
        currentKeyFrame->cloud_surface = surf_pointcloud_in;

        // pcl::PointCloud<PointXYZIL>::Ptr semantic_pointcloud_DS(new pcl::PointCloud<PointXYZIL>());
        
        pcl::PointCloud<PointXYZIL>::Ptr dynamic_pointcloud_DS(new pcl::PointCloud<PointXYZIL>());
        pcl::PointCloud<PointXYZIL>::Ptr static_pointcloud_DS(new pcl::PointCloud<PointXYZIL>());
        pcl::PointCloud<PointXYZIL>::Ptr outlier_pointcloud_DS(new pcl::PointCloud<PointXYZIL>());

        pcl::PointCloud<pcl::PointXYZI>::Ptr corner_pointcloud_DS(new pcl::PointCloud<pcl::PointXYZI>());
        pcl::PointCloud<pcl::PointXYZI>::Ptr surf_pointcloud_DS(new pcl::PointCloud<pcl::PointXYZI>());
        
        // voxel_downsample_pcl(semantic_pointcloud_in, semantic_pointcloud_DS, mappingSurfLeafSize);
        
        voxel_downsample_pcl(dynamic_pointcloud_in, dynamic_pointcloud_DS, mappingCornerLeafSize);
        voxel_downsample_pcl(static_pointcloud_in, static_pointcloud_DS, mappingCornerLeafSize);
        voxel_downsample_pcl(outlier_pointcloud_in, outlier_pointcloud_DS, mappingSurfLeafSize);
        
        voxel_downsample_pcl(corner_pointcloud_in, corner_pointcloud_DS, mappingCornerLeafSize);
        voxel_downsample_pcl(surf_pointcloud_in, surf_pointcloud_DS, mappingSurfLeafSize);

        // currentKeyFrame->cloud_semantic_down = semantic_pointcloud_DS;

        currentKeyFrame->cloud_dynamic_down = dynamic_pointcloud_DS;
        currentKeyFrame->cloud_static_down = static_pointcloud_DS;
        currentKeyFrame->cloud_outlier_down = outlier_pointcloud_DS;

        currentKeyFrame->cloud_corner_down = corner_pointcloud_DS;
        currentKeyFrame->cloud_surface_down = surf_pointcloud_DS;

        keyFrameInfo[keyFrameID] = currentKeyFrame;
        keyFrameID++;
        
        ROS_WARN("keyFrameID: %d ,keyFrameInfo Size: %d ",keyFrameID, keyFrameInfo.size());    
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

    SubMapOptmizationNode() {
        subGPS = nh.subscribe<nav_msgs::Odometry>(gpsTopic, 2000, &SubMapOptmizationNode::gpsHandler, this, ros::TransportHints().tcpNoDelay());

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
    void makeMapThread() {}

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
    std::thread make_map_process(&SubMapOptmizationNode::makeMapThread, &SOP);
    std::thread submap_optmization_process(&SubMapOptmizationNode::subMapOptmizationThread, &SOP);
    
    ROS_INFO("\033[1;32m----> SubMap Optmization Node Started.\033[0m");

    //   ros::MultiThreadedSpinner spinner(3);
    //   spinner.spin();
    ros::spin();

    make_submap_process.join();
    loop_closure_process.join();
    
    make_map_process.join();
    submap_optmization_process.join();

    return 0;
}