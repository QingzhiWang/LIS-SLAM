// Author of  EPSC-LOAM : QZ Wang
// Email wangqingzhi27@outlook.com

#ifndef _KEY_FRAME_H_
#define _KEY_FRAME_H_

#include "utility.h"


typedef boost::shared_ptr<keyframe_t> keyframe_Ptr;

struct keyframe_t {
  ros::Time timeInfoStamp;
  int keyframe_id;
  int submap_id;
  int id_in_submap;

  Eigen::Affine3f init_pose;
  Eigen::Affine3f optimized_pose;
  Eigen::Affine3f gt_pose;
  Eigen::Affine3f relative_pose;

  cv::Mat global_descriptor;
  vector<int> loop_container;

  pcl::PointCloud<PointXYZIL>::Ptr cloud_semantic;
  pcl::PointCloud<PointXYZIL>::Ptr cloud_dynamic;
  pcl::PointCloud<PointXYZIL>::Ptr cloud_static;
  pcl::PointCloud<PointXYZIL>::Ptr cloud_outlier;

  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_corner;
  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_surface;

  pcl::PointCloud<PointXYZIL>::Ptr cloud_semantic_down;
  pcl::PointCloud<PointXYZIL>::Ptr cloud_dynamic_down;
  pcl::PointCloud<PointXYZIL>::Ptr cloud_static_down;
  pcl::PointCloud<PointXYZIL>::Ptr cloud_outlier_down;

  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_corner_down;
  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_surface_down;

  keyframe_t() {
    init();
    // default value
    init_pose.setIdentity();
    optimized_pose.setIdentity();
    gt_pose.setIdentity();
    relative_pose.setIdentity();
  }
  cloudblock_t(const keyframe_t &in_keyframe, bool clone_cloud = false) {
    init();
    clone_metadata(in_keyframe);

    if (clone_cloud) {
      // clone point cloud (instead of pointer)
      *cloud_semantic = *(in_keyframe.cloud_semantic);
      *cloud_dynamic = *(in_keyframe.cloud_dynamic);
      *cloud_static = *(in_keyframe.cloud_static);
      *cloud_outlier = *(in_keyframe.cloud_outlier);

      *cloud_corner = *(in_keyframe.cloud_corner);
      *cloud_surface = *(in_keyframe.cloud_surface);

      *cloud_semantic_down = *(in_keyframe.cloud_semantic_down);
      *cloud_dynamic_down = *(in_keyframe.cloud_dynamic_down);
      *cloud_static_down = *(in_keyframe.cloud_static_down);
      *cloud_outlier_down = *(in_keyframe.cloud_outlier_down);

      *cloud_corner_down = *(in_keyframe.cloud_corner_down);
      *cloud_surface_down = *(in_keyframe.cloud_surface_down);
    }
  }
  void init() {
    cloud_semantic = boost::make_shared<pcl::PointCloud<PointXYZIL>>();
    cloud_dynamic = boost::make_shared<pcl::PointCloud<PointXYZIL>>();
    cloud_static = boost::make_shared<pcl::PointCloud<PointXYZIL>>();
    cloud_outlier = boost::make_shared<pcl::PointCloud<PointXYZIL>>();

    cloud_corner = boost::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
    cloud_surface = boost::make_shared<pcl::PointCloud<pcl::PointXYZI>>();

    cloud_semantic_down = boost::make_shared<pcl::PointCloud<PointXYZIL>>();
    cloud_dynamic_down = boost::make_shared<pcl::PointCloud<PointXYZIL>>();
    cloud_static_down = boost::make_shared<pcl::PointCloud<PointXYZIL>>();
    cloud_outlier_down = boost::make_shared<pcl::PointCloud<PointXYZIL>>();

    cloud_corner_down = boost::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
    cloud_surface_down = boost::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
  }

  void clone_metadata(const keyframe_t &in_keyframe) {
    timeInfoStamp = in_keyframe.timeInfoStamp;
    keyframe_id = in_keyframe.keyframe_id;
    submap_id = in_keyframe.submap_id;
    id_in_submap = in_keyframe.id_in_submap;
    init_pose = in_keyframe.init_pose;
    optimized_pose = in_keyframe.optimized_pose;
    gt_pose = in_keyframe.gt_pose;
    relative_pose = in_keyframe.relative_pose;
    global_descriptor = in_keyframe.global_descriptor;
    loop_container = in_keyframe.loop_container;
  }

  void free_all() {
    cloud_semantic.reset(new pcl::PointCloud<PointXYZIL>());
    cloud_dynamic.reset(new pcl::PointCloud<PointXYZIL>());
    cloud_static.reset(new pcl::PointCloud<PointXYZIL>());
    cloud_outlier.reset(new pcl::PointCloud<PointXYZIL>());

    cloud_corner.reset(new pcl::PointCloud<pcl::PointXYZI>());
    cloud_surface.reset(new pcl::PointCloud<pcl::PointXYZI>());

    cloud_semantic_down.reset(new pcl::PointCloud<PointXYZIL>());
    cloud_dynamic_down.reset(new pcl::PointCloud<PointXYZIL>());
    cloud_static_down.reset(new pcl::PointCloud<PointXYZIL>());
    cloud_outlier_down.reset(new pcl::PointCloud<PointXYZIL>());

    cloud_corner_down.reset(new pcl::PointCloud<pcl::PointXYZI>());
    cloud_surface_down.reset(new pcl::PointCloud<pcl::PointXYZI>());
  }

  void transform_feature(const Eigen::Matrix4d &trans_mat,
                         bool transform_down = true,
                         bool transform_undown = true) {
    if (transform_undown) {
      pcl::transformPointCloudWithNormals(*cloud_semantic, *cloud_semantic,
                                          trans_mat);
      pcl::transformPointCloudWithNormals(*cloud_dynamic, *cloud_dynamic,
                                          trans_mat);
      pcl::transformPointCloudWithNormals(*cloud_static, *cloud_static,
                                          trans_mat);
      pcl::transformPointCloudWithNormals(*cloud_outlier, *cloud_outlier,
                                          trans_mat);
      // pcl::transformPointCloudWithNormals(*cloud_corner, *cloud_corner,
      // trans_mat); pcl::transformPointCloudWithNormals(*cloud_surface,
      // *cloud_surface, trans_mat);
    }
    if (transform_down) {
      pcl::transformPointCloudWithNormals(*cloud_semantic_down,
                                          *cloud_semantic_down, trans_mat);
      pcl::transformPointCloudWithNormals(*cloud_dynamic_down,
                                          *cloud_dynamic_down, trans_mat);
      pcl::transformPointCloudWithNormals(*cloud_static_down,
                                          *cloud_static_down, trans_mat);
      pcl::transformPointCloudWithNormals(*cloud_outlier_down,
                                          *cloud_outlier_down, trans_mat);
      // pcl::transformPointCloudWithNormals(*cloud_corner_down,
      // *cloud_corner_down, trans_mat);
      // pcl::transformPointCloudWithNormals(*cloud_surface_down,
      // *cloud_surface_down, trans_mat);
    }
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

#endif  // _KEY_FRAME_H_