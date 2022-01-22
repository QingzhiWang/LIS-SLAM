// This code partly draws on  lio_sam
// Author of  EPSC-LOAM : QZ Wang
// Email wangqingzhi27@outlook.com

#ifndef _FEATURE_EXTRACTION_H_
#define _FEATURE_EXTRACTION_H_

#include "lis_slam/cloud_info.h"
#include "utility.h"
#include "common.h"

struct smoothness_t 
{
    float value;
    size_t ind;
};

struct by_value 
{
    bool operator()(smoothness_t const &left, smoothness_t const &right) 
    {
        return left.value < right.value;
    }
};

class FeatureExtraction : public ParamServer 
{
public:
    FeatureExtraction()
    {
        allocateMemory();
        resetParameters();
    }

    void allocateMemory();
    void resetParameters();

    void featureExtraction();
	bool cachePointCloud(); 
	void projectPointCloud();
    void cloudExtraction();
    void calculateSmoothness();
    void markOccludedPoints();
    void extractFeatures();
	
    void assignCouldInfo();

	void initCloudInfo(const lis_slam::cloud_infoPtr& msgIn)
    {
		cloudInfo = *msgIn;
	}

	lis_slam::cloud_info getCloudInfo() 
	{ 
		return cloudInfo; 
	}

private:
    std_msgs::Header cloudHeader;
    lis_slam::cloud_info cloudInfo;

	double timeScanCur;
	double timeScanEnd;

    pcl::PointCloud<PointXYZIRT>::Ptr laserCloudIn;

    pcl::PointCloud<PointXYZIRT>::Ptr cornerCloud;
    pcl::PointCloud<PointXYZIRT>::Ptr surfaceCloud;
    pcl::PointCloud<PointXYZIRT>::Ptr sharpCornerCloud;
    pcl::PointCloud<PointXYZIRT>::Ptr SharpSurfaceCloud;

    cv::Mat rangeMat;

    int32_t *startRingIndex;
    int32_t *endRingIndex;

    int32_t *pointColInd;
    float *pointRange;

    std::vector<smoothness_t> cloudSmoothness;
    float *cloudCurvature;
    int *cloudNeighborPicked;
    int *cloudLabel;

};

#endif  // _FEATURE_EXTRACTION_H_
