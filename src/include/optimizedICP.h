//
// Created by meng on 2021/3/24.
//

#ifndef OPTIMIZED_ICP_H
#define OPTIMIZED_ICP_H

#include "common.h"

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


#endif //OPTIMIZED_ICP_H
