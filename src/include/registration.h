//
// Created by meng on 2021/3/24.
//

#ifndef REGISTRATION_H
#define REGISTRATION_H

#include "common.h"

#include <pcl/registration/ndt.h>
#include <pcl/registration/gicp.h>
// #include <fast_gicp/gicp/fast_gicp.hpp>
// #include <fast_gicp/gicp/fast_gicp_st.hpp>
// #include <fast_gicp/gicp/fast_vgicp.hpp>

// #ifdef USE_VGICP_CUDA
// #include <fast_gicp/gicp/fast_vgicp_cuda.hpp>
// #endif

// #include <teaser/ply_io.h>
// #include <teaser/registration.h>
// #include <teaser/certification.h>
// #include <teaser/matcher.h>

class CloudData {
public:
    using POINT = PointXYZIL;
    using CLOUD = pcl::PointCloud<POINT>;
    using CLOUD_PTR = CLOUD::Ptr;

public:
    CloudData() : cloud_ptr(new CLOUD()) {
    }

public:
    double time = 0.0;
    CLOUD_PTR cloud_ptr;
};


//该类是对优化方法的ICP进行的实现,是从我的另一个项目中摘取的代码,
//它并不能直接使用,可能需要你做很小的改动,然后就能应用于你的项目,
//改动主要就是路径的设置,这里只是为了展示优化的ICP实现方法.
class OptimizedICPGN{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;//eigen自动内存对齐

    OptimizedICPGN(const unsigned int max_iterations, const float max_correspond_distance);

    bool SetTargetCloud(const CloudData::CLOUD_PTR &target_cloud_ptr);

    bool Match(const CloudData::CLOUD_PTR &source_cloud_ptr,
               const Eigen::Matrix4f &predict_pose,
               CloudData::CLOUD_PTR &transformed_source_cloud_ptr,
               Eigen::Matrix4f &result_pose);

    float GetFitnessScore();

    bool HasConverged();

private:
    unsigned int max_iterations_;
    float max_correspond_distance_;

    CloudData::CLOUD_PTR target_cloud_ptr_;
    CloudData::CLOUD_PTR source_cloud_ptr_;
    Eigen::Matrix4f final_transformation_;

    pcl::KdTreeFLANN<CloudData::POINT>::Ptr kdtree_flann_ptr_;//In order to search
};


pcl::Registration<PointXYZIL, PointXYZIL>::Ptr select_registration_method(const string &registration_method);

// int coarse_reg_teaser(const pcl::PointCloud<PointXYZIL>::Ptr &target_pts,
// 					  const pcl::PointCloud<PointXYZIL>::Ptr &source_pts,
// 					  Eigen::Matrix4f &tran_mat, float noise_bound, int min_inlier_num);


#endif //REGISTRATION_H
