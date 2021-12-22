// Author of  EPSC-LOAM : QZ Wang
// Email wangqingzhi27@outlook.com

#ifndef _SUBMAP_H_
#define _SUBMAP_H_

#include "utility.h"
#include "common.h"

#include "keyFrame.h"

struct submap_t {
  ros::Time timeInfoStamp;

  int submap_id;
  int submap_size;

  PointTypePose submap_pose_6D_init;
  PointTypePose submap_pose_6D_gt;
  PointTypePose submap_pose_6D_optimized;
  PointType submap_pose_3D_optimized;

  vector<int> keyframe_id_in_submap;
  pcl::PointCloud<PointType>::Ptr keyframe_poses_3D;
  pcl::PointCloud<PointTypePose>::Ptr keyframe_poses_6D;
  map<int,PointType> keyframe_poses_3D_map;
  map<int,PointTypePose> keyframe_poses_6D_map;

  pcl::PointCloud<PointXYZIL>::Ptr submap_semantic;
  pcl::PointCloud<PointXYZIL>::Ptr submap_dynamic;
  pcl::PointCloud<PointXYZIL>::Ptr submap_static;
  pcl::PointCloud<PointXYZIL>::Ptr submap_outlier;

  map<int, cv::Mat> keyframe_global_descriptor;



  void append_feature(const keyframe_t &in_cblock, bool append_down) {
    // pc_raw->points.insert(pc_raw->points.end(),
    // in_cblock.pc_raw->points.begin(), in_cblock.pc_raw->points.end());
    if (!append_down) {
      // if (used_feature_type[0] == '1')
      // 	pc_ground->points.insert(pc_ground->points.end(),
      // in_cblock.pc_ground->points.begin(),
      // in_cblock.pc_ground->points.end());

    } else {
    }
  }
};

#endif  // _SUBMAP_H_