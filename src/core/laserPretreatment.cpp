// Author of  EPSC-LOAM : QZ Wang
// Email wangqingzhi27@outlook.com

#include "laserPretreatment.h"

template <typename PointT>
void removeClosedPointCloud(const pcl::PointCloud<PointT>& cloud_in,
                            pcl::PointCloud<PointT>& cloud_out, float minthres,
                            float maxthres) {
  if (&cloud_in != &cloud_out) {
    cloud_out.header = cloud_in.header;
    cloud_out.points.resize(cloud_in.points.size());
  }

  size_t j = 0;

  for (size_t i = 0; i < cloud_in.points.size(); ++i) {
    float thisRange = cloud_in.points[i].x * cloud_in.points[i].x +
                      cloud_in.points[i].y * cloud_in.points[i].y +
                      cloud_in.points[i].z * cloud_in.points[i].z;
    if (thisRange < minthres * minthres) continue;
    if (thisRange > maxthres * maxthres) continue;
    cloud_out.points[j] = cloud_in.points[i];
    j++;
  }
  if (j != cloud_in.points.size()) {
    cloud_out.points.resize(j);
  }

  cloud_out.height = 1;
  cloud_out.width = static_cast<uint32_t>(j);
  cloud_out.is_dense = true;
}

void LaserPretreatment::init() {}

pcl::PointCloud<PointXYZIRTL>& LaserPretreatment::process(
    pcl::PointCloud<PointType>& laserCloudIn) {
  std::vector<int> indices;

  pcl::removeNaNFromPointCloud(laserCloudIn, laserCloudIn, indices);
  removeClosedPointCloud(laserCloudIn, laserCloudIn, lidarMinRange,
                         lidarMaxRange);

  int cloudSize = laserCloudIn.points.size();
  float startOri = -atan2(laserCloudIn.points[0].y, laserCloudIn.points[0].x);
  float endOri = -atan2(laserCloudIn.points[cloudSize - 1].y,
                        laserCloudIn.points[cloudSize - 1].x) +
                 2 * M_PI;

  if (endOri - startOri > 3 * M_PI) {
    endOri -= 2 * M_PI;
  } else if (endOri - startOri < M_PI) {
    endOri += 2 * M_PI;
  }

  bool halfPassed = false;
  int count = cloudSize;
  PointXYZIRTL point;
  pcl::PointCloud<PointXYZIRTL> laserCloudOut;
  for (int i = 0; i < cloudSize; i++) {
    point.x = laserCloudIn.points[i].x;
    point.y = laserCloudIn.points[i].y;
    point.z = laserCloudIn.points[i].z;
    point.intensity = laserCloudIn.points[i].intensity;

    float angle = atan(point.z / sqrt(point.x * point.x + point.y * point.y)) *
                  180 / M_PI;
    int scanID = 0;

    if (N_SCAN == 16) {
      scanID = int((angle + 15) / 2 + 0.5);
      if (scanID > (N_SCAN - 1) || scanID < 0) {
        count--;
        continue;
      }
    } else if (N_SCAN == 32) {
      scanID = int((angle + 92.0 / 3.0) * 3.0 / 4.0);
      if (scanID > (N_SCAN - 1) || scanID < 0) {
        count--;
        continue;
      }
    } else if (N_SCAN == 64) {
      if (angle >= -8.83)
        scanID = int((2 - angle) * 3.0 + 0.5);
      else
        scanID = N_SCAN / 2 + int((-8.83 - angle) * 2.0 + 0.5);

      // use [0 50]  > 50 remove outlies
      if (angle > 2 || angle < -24.33 || scanID > 50 || scanID < 0) {
        count--;
        continue;
      }
    } else {
      printf("wrong scan number\n");
      ROS_BREAK();
    }

    float ori = -atan2(point.y, point.x);
    if (!halfPassed) {
      if (ori < startOri - M_PI / 2) {
        ori += 2 * M_PI;
      } else if (ori > startOri + M_PI * 3 / 2) {
        ori -= 2 * M_PI;
      }

      if (ori - startOri > M_PI) {
        halfPassed = true;
      }
    } else {
      ori += 2 * M_PI;
      if (ori < endOri - M_PI * 3 / 2) {
        ori += 2 * M_PI;
      } else if (ori > endOri + M_PI / 2) {
        ori -= 2 * M_PI;
      }
    }
    float relTime = (ori - startOri) / (endOri - startOri);
    point.ring = scanID;
    point.time = scanPeriod * relTime;
    point.label = 0;
    laserCloudOut.points.push_back(point);
  }
}
