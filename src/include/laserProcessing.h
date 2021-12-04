//This code partly draws on  lio_sam
// Author of  EPSC-LOAM : QZ Wang  
// Email wangqingzhi27@outlook.com

#ifndef _LASER_PROCESSING_H_
#define _LASER_PROCESSING_H_

#include "utility.h"

#include "lis_slam/cloud_info.h"

std::mutex imuLock;
std::mutex odoLock;
std::mutex cloLock;

std::deque<sensor_msgs::Imu> imuQueue;
std::deque<nav_msgs::Odometry> odomQueue;
std::deque<sensor_msgs::PointCloud2> cloudQueue;

const int queueLength = 2000;

struct smoothness_t{ 
    float value;
    size_t ind;
};

struct by_value{ 
    bool operator()(smoothness_t const &left, smoothness_t const &right) { 
        return left.value < right.value;
    }
};

class LaserProcessing  : public ParamServer
{
    public:
    	LaserProcessing() : deskewFlag(0){

            allocateMemory();
            resetParameters();

            pcl::console::setVerbosityLevel(pcl::console::L_ERROR);
        }

        void allocateMemory();
        void resetParameters();

        bool distortionRemoval();
        void featureExtraction();

        bool cachePointCloud();
        bool deskewInfo();
        void imuDeskewInfo();
        void odomDeskewInfo();
        void findRotation(double pointTime, float *rotXCur, float *rotYCur, float *rotZCur);
        void findPosition(double relTime, float *posXCur, float *posYCur, float *posZCur);
        PointXYZIRTL deskewPoint(PointXYZIRTL *point, double relTime);
        void projectPointCloud();
        void cloudExtraction();

        void calculateSmoothness();
        void markOccludedPoints();
        void extractFeatures();
        
        void assignCouldInfo();

        lis_slam::cloud_info& getCloudInfo(){ return cloudInfo; }
        void freeCloudInfoMemory(){
            cloudInfo.imuAvailable = false;
            cloudInfo.odomAvailable = false;

            cloudInfo.imuRollInit = 0.0;
            cloudInfo.imuPitchInit = 0.0;
            cloudInfo.imuPitchInit = 0.0;

            cloudInfo.imuPitchInit = 0.0;
            cloudInfo.imuPitchInit = 0.0;
            cloudInfo.imuPitchInit = 0.0;
            cloudInfo.imuPitchInit = 0.0;
            cloudInfo.imuPitchInit = 0.0;
            cloudInfo.imuPitchInit = 0.0;

            cloudInfo.cloud_deskewed.clear();
            cloudInfo.cloud_corner.clear();
            cloudInfo.cloud_surface.clear();
        }


        
    private:
        sensor_msgs::PointCloud2 currentCloudMsg;
        std_msgs::Header cloudHeader;

        lis_slam::cloud_info cloudInfo;

        pcl::PointCloud<PointXYZIRTL>::Ptr laserCloudIn;

        pcl::PointCloud<PointXYZIRTL>::Ptr fullCloud;
        pcl::PointCloud<PointXYZIRTL>::Ptr extractedCloud;

        pcl::PointCloud<PointXYZIRTL>::Ptr cornerCloud;
        pcl::PointCloud<PointXYZIRTL>::Ptr surfaceCloud;

        double *imuTime = new double[queueLength];
        double *imuRotX = new double[queueLength];
        double *imuRotY = new double[queueLength];
        double *imuRotZ = new double[queueLength];

        int imuPointerCur;
        bool firstPointFlag;

        Eigen::Affine3f transStartInverse;

        int deskewFlag;
        cv::Mat rangeMat;

        bool odomDeskewFlag;
        float odomIncreX;
        float odomIncreY;
        float odomIncreZ;

        double timeScanCur;
        double timeScanEnd;
        
        int32_t[] startRingIndex;
        int32_t[] endRingIndexnew;

        int32_t[]  pointColInd;
        float[] pointRange;

        std::vector<smoothness_t> cloudSmoothness;
        float *cloudCurvature;
        int *cloudNeighborPicked;
        int *cloudLabel;

        // pcl::VoxelGrid<PointXYZIRTL> downSizeFilter;
};




#endif // _LASER_PROCESSING_H_
