// This code partly draws on  lio_sam
// Author of  EPSC-LOAM : QZ Wang
// Email wangqingzhi27@outlook.com

#ifndef _LASER_PRETREATMENT_H_
#define _LASER_PRETREATMENT_H_

#include <cmath>

#include "utility.h"

using std::atan2;
using std::cos;
using std::sin;

template <typename PointT>
void removeClosedPointCloud(const pcl::PointCloud<PointT>& cloud_in,
                            pcl::PointCloud<PointT>& cloud_out, float minthres,
                            float maxthres);

class LaserPretreatment : public ParamServer {
 public:
  LaserPretreatment(){};
  void init();
  pcl::PointCloud<PointXYZIRT>& process(
      pcl::PointCloud<PointType>& laserCloudIn);

 private:
  const double scanPeriod = 0.1;
};

#endif  // _LASER_PRETREATMENT_H_