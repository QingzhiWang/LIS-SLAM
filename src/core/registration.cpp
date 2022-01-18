//
// Created by meng on 2021/3/25.
//
#include "registration.h"
#include "se3.hpp"
#include <pcl/features/normal_3d.h>

OptimizedICPGN::OptimizedICPGN(const unsigned int max_iterations, const float max_correspond_distance)
        : kdtree_flann_ptr_(new pcl::KdTreeFLANN<CloudData::POINT>) {
    max_iterations_ = max_iterations;
    max_correspond_distance_ = max_correspond_distance;
}

bool OptimizedICPGN::SetTargetCloud(const CloudData::CLOUD_PTR &target_cloud_ptr) {
    target_cloud_ptr_ = target_cloud_ptr;
    kdtree_flann_ptr_->setInputCloud(target_cloud_ptr);//构建kdtree用于全局最近邻搜索
}

bool OptimizedICPGN::Match(const CloudData::CLOUD_PTR &source_cloud_ptr, const Eigen::Matrix4f &predict_pose,
                           CloudData::CLOUD_PTR &transformed_source_cloud_ptr, Eigen::Matrix4f &result_pose) {
    source_cloud_ptr_ = source_cloud_ptr;

    CloudData::CLOUD_PTR transformed_cloud(new CloudData::CLOUD);

    Eigen::Matrix4f T = predict_pose;

    //Gauss-Newton's method solve ICP.
    for (unsigned int i = 0; i < max_iterations_; ++i) {
        pcl::transformPointCloud(*source_cloud_ptr, *transformed_cloud, T);
        Eigen::Matrix<float, 6, 6> Hessian = Eigen::Matrix<float, 6, 6>::Zero();
        Eigen::Matrix<float, 6, 1> B = Eigen::Matrix<float, 6, 1>::Zero();
		
		#pragma omp for
        for (unsigned int j = 0; j < transformed_cloud->size(); ++j) {
            const CloudData::POINT &origin_point = source_cloud_ptr->points[j];

            //删除距离为无穷点
            if (!pcl::isFinite(origin_point)) {
                continue;
            }

            const CloudData::POINT &transformed_point = transformed_cloud->at(j);
            std::vector<float> resultant_distances;
            std::vector<int> indices;
            //在目标点云中搜索距离当前点最近的一个点
            kdtree_flann_ptr_->nearestKSearch(transformed_point, 1, indices, resultant_distances);

            //舍弃那些最近点,但是距离大于最大对应点对距离
            if (resultant_distances.front() > max_correspond_distance_) {
                continue;
            }

            Eigen::Vector3f nearest_point = Eigen::Vector3f(target_cloud_ptr_->at(indices.front()).x,
                                                            target_cloud_ptr_->at(indices.front()).y,
                                                            target_cloud_ptr_->at(indices.front()).z);

            Eigen::Vector3f point_eigen(transformed_point.x, transformed_point.y, transformed_point.z);
            Eigen::Vector3f origin_point_eigen(origin_point.x, origin_point.y, origin_point.z);
            Eigen::Vector3f error = point_eigen - nearest_point;

            Eigen::Matrix<float, 3, 6> Jacobian = Eigen::Matrix<float, 3, 6>::Zero();
            //构建雅克比矩阵
            Jacobian.leftCols(3) = Eigen::Matrix3f::Identity();
            Jacobian.rightCols(3) = -T.block<3, 3>(0, 0) * Sophus::SO3f::hat(origin_point_eigen);

            //构建海森矩阵
            Hessian += Jacobian.transpose() * Jacobian;
            B += -Jacobian.transpose() * error;
        }

        if (Hessian.determinant() == 0) {
            continue;
        }

        Eigen::Matrix<float, 6, 1> delta_x = Hessian.inverse() * B;

        T.block<3, 1>(0, 3) = T.block<3, 1>(0, 3) + delta_x.head(3);
        T.block<3, 3>(0, 0) *= Sophus::SO3f::exp(delta_x.tail(3)).matrix();
    }

    final_transformation_ = T;
    result_pose = T;
    pcl::transformPointCloud(*source_cloud_ptr, *transformed_source_cloud_ptr, result_pose);

    return true;
}

//该函数用于计算匹配之后的得分,其写法与pcl中的计算icp或者ndt的方式是一致,
//他们之间的得分可以进行比较
float OptimizedICPGN::GetFitnessScore() {
    float max_range = std::numeric_limits<float>::max();
    float fitness_score = 0.0f;

    CloudData::CLOUD_PTR transformed_cloud_ptr(new CloudData::CLOUD);
    pcl::transformPointCloud(*source_cloud_ptr_, *transformed_cloud_ptr, final_transformation_);

    std::vector<int> nn_indices(1);
    std::vector<float> nn_dists(1);

    int nr = 0;

    for (unsigned int i = 0; i < transformed_cloud_ptr->size(); ++i) {
        kdtree_flann_ptr_->nearestKSearch(transformed_cloud_ptr->points[i], 1, nn_indices, nn_dists);

        if (nn_dists.front() <= max_range) {
            fitness_score += nn_dists.front();
            nr++;
        }
    }

    if (nr > 0)
        return fitness_score / static_cast<float>(nr);
    else
        return (std::numeric_limits<float>::max());
}

bool OptimizedICPGN::HasConverged() {
    ///TODO: add this function
    return true;
}



pcl::Registration<PointXYZIL, PointXYZIL>::Ptr select_registration_method(const string &registration_method)
{
	using PointT = PointXYZIL;
  
	if(registration_method == "ICP") {
		std::cout << "registration: ICP" << std::endl;
		pcl::IterativeClosestPoint<PointT, PointT>::Ptr icp(new pcl::IterativeClosestPoint<PointT, PointT>());
		icp->setTransformationEpsilon(0.01);
		icp->setMaximumIterations(50);
		icp->setMaxCorrespondenceDistance(5);
		icp->setUseReciprocalCorrespondences(false);
		return icp;
	}
	else if(registration_method == "GICP") {
		std::cout << "registration: GICP" << std::endl;
		pcl::GeneralizedIterativeClosestPoint<PointT, PointT>::Ptr icp(new pcl::GeneralizedIterativeClosestPoint<PointT, PointT>());
        icp->setMaxCorrespondenceDistance(10);
        icp->setMaximumIterations(30);
        icp->setTransformationEpsilon(1e-6);
        icp->setEuclideanFitnessEpsilon(1e-3);
        icp->setRANSACIterations(0);
		return icp;
	}
	else if(registration_method == "NDT") {
		std::cout << "registration: NDT" << std::endl;
		pcl::NormalDistributionsTransform<PointT, PointT>::Ptr ndt(new pcl::NormalDistributionsTransform<PointT, PointT>());
		ndt->setTransformationEpsilon(0.01); //为终止条件设置最小转换差异
		ndt->setStepSize(0.1); //为More-Thuente线搜索设置最大步长
		ndt->setResolution(1.0); //设置NDT网格结构的分辨率（VoxelGridCovariance）
		ndt->setMaximumIterations(35); //设置匹配迭代的最大次数
		return ndt;
	}
	// 未定义的引用 --error
	// else if(registration_method == "FAST_GICP") {
	// 	std::cout << "registration: FAST_GICP" << std::endl;
	// 	fast_gicp::FastGICP<PointT, PointT>::Ptr gicp(new fast_gicp::FastGICP<PointT, PointT>());
	// 	gicp->setNumThreads(4);
	// 	gicp->setTransformationEpsilon(0.01);
	// 	gicp->setMaximumIterations(50);
	// 	gicp->setMaxCorrespondenceDistance(5);
	// 	gicp->setCorrespondenceRandomness(20);
	// 	return gicp;
	// }
	// #ifdef USE_VGICP_CUDA
	// else if(registration_method == "FAST_VGICP_CUDA") {
	// 	std::cout << "registration: FAST_VGICP_CUDA" << std::endl;
	// 	fast_gicp::FastVGICPCuda<PointT, PointT>::Ptr vgicp(new fast_gicp::FastVGICPCuda<PointT, PointT>());
	// 	vgicp->setResolution(1.0);
	// 	vgicp->setTransformationEpsilon(0.01);
	// 	vgicp->setMaximumIterations(50);
	// 	vgicp->setCorrespondenceRandomness(20));
	// 	return vgicp;
	// }
	// #endif
	// else if(registration_method == "FAST_VGICP") {
	// 	std::cout << "registration: FAST_VGICP" << std::endl;
	// 	fast_gicp::FastVGICP<PointT, PointT>::Ptr vgicp(new fast_gicp::FastVGICP<PointT, PointT>());
	// 	vgicp->setNumThreads(4);
	// 	vgicp->setResolution(1.0);
	// 	vgicp->setTransformationEpsilon(0.01);
	// 	vgicp->setMaximumIterations(50);
	// 	vgicp->setCorrespondenceRandomness(20);
	// 	return vgicp;
	// } 
}




int coarse_reg_teaser(const pcl::PointCloud<PointXYZIL>::Ptr &target_pts,
					  const pcl::PointCloud<PointXYZIL>::Ptr &source_pts,
					  Eigen::Matrix4f &tran_mat, float noise_bound, int min_inlier_num)
{
	Eigen::Matrix4d tran_;
	int teaser_state = 0; //(failed: -1, successful[need check]: 0, successful[reliable]: 1)

	if (target_pts->points.size() <= 3)
	{
		std::cout << "TEASER: too few correspondences" << std::endl;
		return (-1);
	}

	int N = target_pts->points.size();
	int M = source_pts->points.size();

	float min_inlier_ratio = 0.01;

	Eigen::Matrix<double, 3, Eigen::Dynamic> tgt(3, N);
	Eigen::Matrix<double, 3, Eigen::Dynamic> src(3, M);
  	
	teaser::PointCloud tgt_cloud;
  	teaser::PointCloud src_cloud;
	
	#pragma omp for
	for (int i = 0; i < N; ++i)
	{
		tgt_cloud.push_back({
				static_cast<float>(target_pts->points[i].x), 
				static_cast<float>(target_pts->points[i].y),
				static_cast<float>(target_pts->points[i].z)});
		// tgt.col(i) << target_pts->points[i].x, target_pts->points[i].y, target_pts->points[i].z;
		
	}

	#pragma omp for
	for (int i = 0; i < M; ++i)
	{
		src_cloud.push_back({
				static_cast<float>(source_pts->points[i].x), 
				static_cast<float>(source_pts->points[i].y),
				static_cast<float>(source_pts->points[i].z)});
		// src.col(i) << source_pts->points[i].x, source_pts->points[i].y, source_pts->points[i].z;
	}

	std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();

	teaser::FPFHEstimation fpfh;
	auto obj_descriptors = fpfh.computeFPFHFeatures(src_cloud, 0.02, 0.04);
	auto scene_descriptors = fpfh.computeFPFHFeatures(tgt_cloud, 0.02, 0.04);

	teaser::Matcher matcher;
	auto correspondences = matcher.calculateCorrespondences(
			src_cloud, tgt_cloud, *obj_descriptors, *scene_descriptors, false, true, false, 0.95);

	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
	std::chrono::duration<double> time_used_1 = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

	std::cout << "TEASER Compute FPFH Features and Correspondences done in [" << time_used_1.count() * 1000.0 << "] ms." << std::endl;

	// Run TEASER++ registration
	// Prepare solver parameters
	teaser::RobustRegistrationSolver::Params params;
	params.noise_bound = noise_bound;
	params.cbar2 = 1.0;
	params.estimate_scaling = false;
	params.rotation_max_iterations = 100;
	params.rotation_gnc_factor = 1.4;
	// params.rotation_estimation_algorithm =
	// 	teaser::RobustRegistrationSolver::ROTATION_ESTIMATION_ALGORITHM::FGR;
	params.rotation_estimation_algorithm =
		teaser::RobustRegistrationSolver::ROTATION_ESTIMATION_ALGORITHM::GNC_TLS;
	params.use_max_clique = true;
	params.kcore_heuristic_threshold = 0.5;
	params.rotation_cost_threshold = 0.005; //1e-6

	// Solve with TEASER++
	std::cout << "----------------------------------------------------------------------------" << std::endl;
	std::cout << "Begin TEASER global coarse registration with [" << N << "] pairs of correspondence" << std::endl;
	// std::cout << "Begin TEASER global coarse registration with [" << correspondences.size() << "] pairs of correspondence" << std::endl;
	teaser::RobustRegistrationSolver solver(params);
	std::chrono::steady_clock::time_point tic = std::chrono::steady_clock::now();
	// solver.solve(src, tgt);
  	solver.solve(src_cloud, tgt_cloud, correspondences);
	std::chrono::steady_clock::time_point toc = std::chrono::steady_clock::now();
	std::chrono::duration<double> time_used = std::chrono::duration_cast<std::chrono::duration<double>>(toc - tic);

	std::cout << "TEASER global coarse registration done in [" << time_used.count() * 1000.0 << "] ms." << std::endl;

	auto solution = solver.getSolution();
	std::vector<int> inliers;
	//inliers = solver.getTranslationInliers();
	inliers = solver.getRotationInliers();

	std::cout << "[" << inliers.size() << "] inlier correspondences found." << std::endl;

	//if (solution.valid && 1.0 * inliers.size() / N >= min_inlier_ratio)
	if (solution.valid && inliers.size() >= min_inlier_num)
	{
		tran_.setIdentity();
		tran_.block<3, 3>(0, 0) = solution.rotation;
		tran_.block<3, 1>(0, 3) = solution.translation;

		std::cout << "Estimated transformation by TEASER is :\n"
				  << tran_ << std::endl;

		tran_mat = tran_.cast<float>();
		// certificate the result here
		// teaser::DRSCertifier::Params cer_params;
		// teaser::DRSCertifier certifier(cer_params);
		// auto certification_result = certifier.certify(tran_mat,src,dst, theta);

		if (inliers.size() >= 2 * min_inlier_num)
			return (1); //reliable
		else
			return (0); //need check
	}
	else
	{
		std::cout << "TEASER failed" << std::endl;
		return (-1);
	}
	return (-1);
}


