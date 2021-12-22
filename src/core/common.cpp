// Author of  EPSC-LOAM : QZ Wang
// Email wangqingzhi27@outlook.com

#include "common.h"

sensor_msgs::PointCloud2 publishRawCloud(
    ros::Publisher *thisPub, pcl::PointCloud<PointXYZIRT>::Ptr thisCloud,
    ros::Time thisStamp, std::string thisFrame) {
  sensor_msgs::PointCloud2 tempCloud;
  pcl::toROSMsg(*thisCloud, tempCloud);
  tempCloud.header.stamp = thisStamp;
  tempCloud.header.frame_id = thisFrame;
  if (thisPub->getNumSubscribers() != 0) thisPub->publish(tempCloud);
  return tempCloud;
}

sensor_msgs::PointCloud2 publishCloud(
    ros::Publisher *thisPub, pcl::PointCloud<PointType>::Ptr thisCloud,
    ros::Time thisStamp, std::string thisFrame) {
  sensor_msgs::PointCloud2 tempCloud;
  pcl::toROSMsg(*thisCloud, tempCloud);
  tempCloud.header.stamp = thisStamp;
  tempCloud.header.frame_id = thisFrame;
  if (thisPub->getNumSubscribers() != 0) thisPub->publish(tempCloud);
  return tempCloud;
}

Eigen::Affine3f pclPointToAffine3f(PointTypePose thisPoint) {
  return pcl::getTransformation(thisPoint.x, thisPoint.y, thisPoint.z,
                                thisPoint.roll, thisPoint.pitch,
                                thisPoint.yaw);
}

Eigen::Affine3f trans2Affine3f(float transformIn[]) {
  return pcl::getTransformation(transformIn[3], transformIn[4],
                                transformIn[5], transformIn[0],
                                transformIn[1], transformIn[2]);
}


PointTypePose trans2PointTypePose(float transformIn[]) {
    PointTypePose thisPose6D;
    thisPose6D.x = transformIn[3];
    thisPose6D.y = transformIn[4];
    thisPose6D.z = transformIn[5];
    thisPose6D.roll = transformIn[0];
    thisPose6D.pitch = transformIn[1];
    thisPose6D.yaw = transformIn[2];
    return thisPose6D;
}


pcl::PointCloud<PointType>::Ptr transformPointCloud(
    pcl::PointCloud<PointType>::Ptr cloudIn, Eigen::Affine3f transformIn) {
  pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

  PointType* pointFrom;

  int cloudSize = cloudIn->size();
  cloudOut->resize(cloudSize);

  // #pragma omp parallel for num_threads(numberOfCores)
  #pragma omp for
  for (int i = 0; i < cloudSize; ++i) {
    pointFrom = &cloudIn->points[i];
    cloudOut->points[i].x =
        transformIn(0, 0) * pointFrom->x + transformIn(0, 1) * pointFrom->y +
        transformIn(0, 2) * pointFrom->z + transformIn(0, 3);
    cloudOut->points[i].y =
        transformIn(1, 0) * pointFrom->x + transformIn(1, 1) * pointFrom->y +
        transformIn(1, 2) * pointFrom->z + transformIn(1, 3);
    cloudOut->points[i].z =
        transformIn(2, 0) * pointFrom->x + transformIn(2, 1) * pointFrom->y +
        transformIn(2, 2) * pointFrom->z + transformIn(2, 3);
    cloudOut->points[i].intensity = pointFrom->intensity;
  }
  return cloudOut;
}


pcl::PointCloud<PointType>::Ptr transformPointCloud(
    pcl::PointCloud<PointType>::Ptr cloudIn, PointTypePose* transformIn) {
  pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

  PointType* pointFrom;

  int cloudSize = cloudIn->size();
  cloudOut->resize(cloudSize);

  Eigen::Affine3f transCur = pcl::getTransformation(
      transformIn->x, transformIn->y, transformIn->z, transformIn->roll,
      transformIn->pitch, transformIn->yaw);

  // #pragma omp parallel for num_threads(numberOfCores)
  #pragma omp for
  for (int i = 0; i < cloudSize; ++i) {
    pointFrom = &cloudIn->points[i];
    cloudOut->points[i].x = transCur(0, 0) * pointFrom->x +
                            transCur(0, 1) * pointFrom->y +
                            transCur(0, 2) * pointFrom->z + transCur(0, 3);
    cloudOut->points[i].y = transCur(1, 0) * pointFrom->x +
                            transCur(1, 1) * pointFrom->y +
                            transCur(1, 2) * pointFrom->z + transCur(1, 3);
    cloudOut->points[i].z = transCur(2, 0) * pointFrom->x +
                            transCur(2, 1) * pointFrom->y +
                            transCur(2, 2) * pointFrom->z + transCur(2, 3);
    cloudOut->points[i].intensity = pointFrom->intensity;
  }
  return cloudOut;
}

pcl::PointCloud<PointXYZIL>::Ptr transformPointCloud(
    pcl::PointCloud<PointXYZIL>::Ptr cloudIn, PointTypePose *transformIn) {
    pcl::PointCloud<PointXYZIL>::Ptr cloudOut(
        new pcl::PointCloud<PointXYZIL>());

    PointXYZIL *pointFrom;

    int cloudSize = cloudIn->size();
    cloudOut->resize(cloudSize);

    Eigen::Affine3f transCur = pcl::getTransformation(
        transformIn->x, transformIn->y, transformIn->z, transformIn->roll,
        transformIn->pitch, transformIn->yaw);

    // #pragma omp parallel for num_threads(numberOfCores)
    #pragma omp for
    for (int i = 0; i < cloudSize; ++i) {
      pointFrom = &cloudIn->points[i];
      cloudOut->points[i].x = transCur(0, 0) * pointFrom->x +
                              transCur(0, 1) * pointFrom->y +
                              transCur(0, 2) * pointFrom->z + transCur(0, 3);
      cloudOut->points[i].y = transCur(1, 0) * pointFrom->x +
                              transCur(1, 1) * pointFrom->y +
                              transCur(1, 2) * pointFrom->z + transCur(1, 3);
      cloudOut->points[i].z = transCur(2, 0) * pointFrom->x +
                              transCur(2, 1) * pointFrom->y +
                              transCur(2, 2) * pointFrom->z + transCur(2, 3);
      cloudOut->points[i].intensity = pointFrom->intensity;
      cloudOut->points[i].label = pointFrom->label;
    }
    return cloudOut;
}

pcl::PointCloud<PointXYZIL>::Ptr transformPointCloud(
    pcl::PointCloud<PointXYZIL>::Ptr cloudIn, Eigen::Affine3f &transCur) {
    pcl::PointCloud<PointXYZIL>::Ptr cloudOut(
        new pcl::PointCloud<PointXYZIL>());

    PointXYZIL *pointFrom;

    int cloudSize = cloudIn->size();
    cloudOut->resize(cloudSize);

    // #pragma omp parallel for num_threads(numberOfCores)
    #pragma omp for
    for (int i = 0; i < cloudSize; ++i) {
      pointFrom = &cloudIn->points[i];
      cloudOut->points[i].x = transCur(0, 0) * pointFrom->x +
                              transCur(0, 1) * pointFrom->y +
                              transCur(0, 2) * pointFrom->z + transCur(0, 3);
      cloudOut->points[i].y = transCur(1, 0) * pointFrom->x +
                              transCur(1, 1) * pointFrom->y +
                              transCur(1, 2) * pointFrom->z + transCur(1, 3);
      cloudOut->points[i].z = transCur(2, 0) * pointFrom->x +
                              transCur(2, 1) * pointFrom->y +
                              transCur(2, 2) * pointFrom->z + transCur(2, 3);
      cloudOut->points[i].intensity = pointFrom->intensity;
      cloudOut->points[i].label = pointFrom->label;
    }
    return cloudOut;
}
