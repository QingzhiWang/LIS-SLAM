// Author of  EPSC-LOAM : QZ Wang
// Email wangqingzhi27@outlook.com

#include "subMap.h"

bool SubMapManager::update_submap(submap_Ptr local_map, keyframe_Ptr last_target_cblock,
                float local_map_radius = 80, 
                int max_num_pts = 80000, 
                int kept_vertex_num = 800,
                float last_frame_reliable_radius = 60,
                bool map_based_dynamic_removal_on = false,
                float dynamic_removal_center_radius = 30.0,
                float dynamic_dist_thre_min = 0.3,
                float dynamic_dist_thre_max = 3.0,
                float near_dist_thre = 0.03,
                bool recalculate_feature_on = false){

    std::chrono::steady_clock::time_point tic = std::chrono::steady_clock::now();

    Eigen::Affine3f tran_target_map; //from the last local_map to the target frame
    tran_target_map = pclPointToAffine3f(last_target_cblock->optimized_pose).inverse() * pclPointToAffine3f(local_map->submap_pose_6D_optimized);
    
    last_target_cblock->transform_feature(tran_target_map.inverse(), true, false);
    
    dynamic_dist_thre_max = std::max(dynamic_dist_thre_max, dynamic_dist_thre_min + 0.1);
    std::cout << "Map based filtering range(m): (0, " << near_dist_thre << "] U [" << dynamic_dist_thre_min << "," << dynamic_dist_thre_max << "]" << std::endl;

    if (map_based_dynamic_removal_on && local_map->feature_point_num > max_num_pts / 5)
    {
        map_based_dynamic_close_removal(local_map, last_target_cblock, used_feature_type, dynamic_removal_center_radius,
                                        dynamic_dist_thre_min, dynamic_dist_thre_max, near_dist_thre);

        std::cout << "Feature point number of last frame after dynamic removal: Dynamic: [" 
                  << last_target_cblock->cloud_dynamic->points.size() << " | "
                  << last_target_cblock->cloud_dynamic_down->points.size() << "]." << std::endl;
    }  
    local_map->append_feature(*last_target_cblock);
    
    local_map->submap_size++;
    local_map->keyframe_id_in_submap.push_back(last_target_cblock->keyframe_id);

    PointType thisPose3D;
    thisPose3D.x = last_target_cblock->optimized_pose.x;
    thisPose3D.y = last_target_cblock->optimized_pose.y;
    thisPose3D.z = last_target_cblock->optimized_pose.z;
    thisPose3D.intensity = last_target_cblock->keyframe_id;
    local_map->points.push_back(thisPose3D);

    local_map->keyframe_poses_6D.push_back(last_target_cblock->optimized_pose);

    local_map->keyframe_poses_3D_map.insert(std::make_pair(last_target_cblock->optimized_pose, thisPose3D));
    local_map->keyframe_poses_6D_map.insert(std::make_pair(last_target_cblock->optimized_pose, last_target_cblock->optimized_pose));

    local_map->feature_point_num = local_map->submap_dynamic->points.size() + 
                                   local_map->submap_static->points.size() + 
                                   local_map->submap_outlier->points.size();

    typename pcl::PointCloud<PointT>::Ptr cloud_raw(new pcl::PointCloud<PointT>);
    
    //calculate bbx (local)
    local_map->merge_feature_points(cloud_raw);
    get_cloud_bbx(cloud_raw, local_map->local_bound);

    //calculate bbx (global)
    Eigen::Affine3f tran_map = pclPointToAffine3f(local_map->submap_pose_6D_optimized);
    transform_bbx(local_map->bound, tran_map);

    local_map->free_tree();
    
    std::chrono::steady_clock::time_point toc = std::chrono::steady_clock::now();
    std::chrono::duration<double> time_update_local_map = std::chrono::duration_cast<std::chrono::duration<double>>(toc - tic);
    std::cout << "Update local map ([" << local_map->feature_point_num << "] points at present) done in [" << time_update_local_map.count() * 1000.0 << "] ms.\n";
    return true;
}

bool SubMapManager::map_based_dynamic_close_removal(submap_Ptr local_map, keyframe_Ptr last_target_cblock,
                                                    float center_radius, float dynamic_dist_thre_min, 
                                                    float dynamic_dist_thre_max, float near_dist_thre){
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            //dynamic + close points
            map_scan_feature_pts_distance_removal(last_target_cblock->cloud_dynamic, local_map->tree_dynamic, center_radius, dynamic_dist_thre_min, dynamic_dist_thre_max, near_dist_thre);
            map_scan_feature_pts_distance_removal(last_target_cblock->cloud_static, local_map->tree_static, center_radius, dynamic_dist_thre_min, dynamic_dist_thre_max, near_dist_thre);
        }
        
        #pragma omp section
        {
            //dynamic + close points
            map_scan_feature_pts_distance_removal(last_target_cblock->cloud_outlier, local_map->tree_outlier, center_radius, dynamic_dist_thre_min, dynamic_dist_thre_max, near_dist_thre);
        }
    }
    
    return true;
}

//keep the points meet dist ~ (near_dist_thre, dynamic_dist_thre_min) U (dynamic_dist_thre_max, +inf)
//filter the points meet dist ~ (0, near_dist_thre] U [dynamic_dist_thre_min, dynamic_dist_thre_max]
bool SubMapManager::map_scan_feature_pts_distance_removal(pcl::PointCloud<PointT>::Ptr feature_pts, 
                                                        const pcl::search::KdTree<PointT>::Ptr map_kdtree, 
                                                        float center_radius, float dynamic_dist_thre_min = FLT_MAX, 
                                                        float dynamic_dist_thre_max = FLT_MAX, float near_dist_thre = 0.0){
    if (feature_pts->points.size() <= 10)
        return false;

    pcl::PointCloud<PointT>::Ptr  feature_pts_temp(new pcl::PointCloud<PointT>);
    int i;
    
    //#pragma omp parallel for private(i) //Multi-thread

    std::vector<float> distances_square;
    std::vector<int> search_indices;
    for (i = 0; i < feature_pts->points.size(); i++)
    {
        if (feature_pts->points[i].x * feature_pts->points[i].x +
                feature_pts->points[i].y * feature_pts->points[i].y >
            center_radius * center_radius)
            feature_pts_temp->points.push_back(feature_pts->points[i]);
        else
        {
            map_kdtree->nearestKSearch(feature_pts->points[i], 1, search_indices, distances_square);                                                                                                                   //search nearest neighbor
            if ((distances_square[0] > near_dist_thre * near_dist_thre && distances_square[0] < dynamic_dist_thre_min * dynamic_dist_thre_min) || distances_square[0] > dynamic_dist_thre_max * dynamic_dist_thre_max) // the distance should not be too close to keep the map more uniform
                feature_pts_temp->points.push_back(feature_pts->points[i]);
            // else
            //     LOG(INFO) << "Filter out the point, dist [" << std::sqrt(distances_square[0]) << "].";

            std::vector<float>().swap(distances_square);
            std::vector<int>().swap(search_indices);
        }
    }
    feature_pts_temp->points.swap(feature_pts->points);

    return true;  
}


bool SubMapManager::judge_new_submap(float &accu_tran, float &accu_rot, int &accu_frame,
                                    float max_accu_tran = 30.0, float max_accu_rot = 90.0, 
                                    int max_accu_frame = 150){
    // LOG(INFO) << "Submap division criterion is: \n"
    //           << "1. Frame Number <= " << max_accu_frame
    //           << " , 2. Translation <= " << max_accu_tran
    //           << "m , 3. Rotation <= " << max_accu_rot << " degree.";

    if (accu_tran > max_accu_tran || accu_rot > max_accu_rot || accu_frame > max_accu_frame)
    {
        //recount from begining
        accu_tran = 0.0;
        accu_rot = 0.0;
        accu_frame = 0;
        return true;
    }
    else
        return false;
}





bool SubMapManager::bbx_filter(const typename pcl::PointCloud<PointT>::Ptr &cloud_in_out, bounds_t &bbx, bool delete_box = false){
    std::chrono::steady_clock::time_point tic = std::chrono::steady_clock::now();
    typename pcl::PointCloud<PointT>::Ptr cloud_temp(new pcl::PointCloud<PointT>);
    int original_pts_num = cloud_in_out->points.size();
    for (int i = 0; i < cloud_in_out->points.size(); i++)
    {
        //In the bounding box
        if (cloud_in_out->points[i].x > bbx.min_x && cloud_in_out->points[i].x < bbx.max_x &&
            cloud_in_out->points[i].y > bbx.min_y && cloud_in_out->points[i].y < bbx.max_y &&
            cloud_in_out->points[i].z > bbx.min_z && cloud_in_out->points[i].z < bbx.max_z)
        {
            if (!delete_box)
                cloud_temp->points.push_back(cloud_in_out->points[i]);
        }
        else
        {
            if (delete_box)
                cloud_temp->points.push_back(cloud_in_out->points[i]);
        }
    }
    cloud_temp->points.swap(cloud_in_out->points);

    std::chrono::steady_clock::time_point toc = std::chrono::steady_clock::now();
    std::chrono::duration<double> time_used = std::chrono::duration_cast<std::chrono::duration<double>>(toc - tic);

    // LOG(INFO) << "Box filter [ " << original_pts_num << " --> "
    // 		  << cloud_in_out->points.size() << " ] done in [" << time_used.count() << "] s";

    return 1;
}

bool SubMapManager::random_downsample(typename pcl::PointCloud<PointT>::Ptr &cloud_in_out, int downsample_ratio)
{
    typename pcl::PointCloud<PointT>::Ptr cloud_temp(new pcl::PointCloud<PointT>);

    if (downsample_ratio > 1)
    {
        for (int i = 0; i < cloud_in_out->points.size(); i++)
        {
            if (i % downsample_ratio == 0)
                cloud_temp->points.push_back(cloud_in_out->points[i]);
        }
        cloud_temp->points.swap(cloud_in_out->points);
        //LOG(INFO)<<"rand_filter : " << cloud_in_out->points.size();
        return 1;
    }
    else
        return 0;
}

//fixed number random downsampling
//when keep_number == 0, the output point cloud would be empty (in other words, the input point cloud would be cleared)
bool SubMapManager::random_downsample_pcl(typename pcl::PointCloud<PointT>::Ptr &cloud_in_out, int keep_number)
{
    if (cloud_in_out->points.size() <= keep_number)
        return false;
    else
    {
        if (keep_number == 0)
        {
            cloud_in_out.reset(new typename pcl::PointCloud<PointT>());
            return false;
        }
        else
        {
            typename pcl::PointCloud<PointT>::Ptr cloud_temp(new pcl::PointCloud<PointT>);
            pcl::RandomSample<PointT> ran_sample(true); // Extract removed indices
            ran_sample.setInputCloud(cloud_in_out);
            ran_sample.setSample(keep_number);
            ran_sample.filter(*cloud_temp);
            cloud_temp->points.swap(cloud_in_out->points);
            return true;
        }
    }
}


bool SubMapManager::voxel_downsample_pcl(typename pcl::PointCloud<PointT>::Ptr &cloud_in_out, float leaf_size)
{
    if (cloud_in_out->points.size() <= 0)
        return false;

    typename pcl::PointCloud<PointT>::Ptr cloud_temp(new pcl::PointCloud<PointT>);
    pcl::VoxelGrid<PointT> downSizeFilter;
    downSizeFilter.setLeafSize(leaf_size, leaf_size, leaf_size);
    downSizeFilter.setInputCloud(cloud_in_out);
    downSizeFilter.filter(*cloud_temp);

    cloud_temp->points.swap(cloud_in_out->points);

    return true;    
}



bool SubMapManager::voxel_downsample_pcl(typename pcl::PointCloud<PointT>::Ptr &cloud_in, typename pcl::PointCloud<PointT>::Ptr &cloud_out, float leaf_size);
{
    if (cloud_in->points.size() <= 0)
        return false;

    pcl::VoxelGrid<PointT> downSizeFilter;
    downSizeFilter.setLeafSize(leaf_size, leaf_size, leaf_size);
    downSizeFilter.setInputCloud(cloud_in);
    downSizeFilter.filter(*cloud_out);

    return true;    
}



void SubMapManager::label2RGBCloud(pcl::PointCloud<PointXYZIL>::Ptr in_cloud, 
                                   pcl::PointCloud<pcl::PointXYZRGB>::Ptr &out_cloud){
    out_cloud->points.resize(in_cloud->points.size());
    for (size_t i = 0; i < out_cloud->points.size(); i++) {
        out_cloud->points[i].x = in_cloud->points[i].x;
        out_cloud->points[i].y = in_cloud->points[i].y;
        out_cloud->points[i].z = in_cloud->points[i].z;
        out_cloud->points[i].r = std::get<0>(Argmax2RGB[in_cloud->points[i].label]);
        out_cloud->points[i].g = std::get<1>(Argmax2RGB[in_cloud->points[i].label]);
        out_cloud->points[i].b = std::get<2>(Argmax2RGB[in_cloud->points[i].label]);
    }
    out_cloud->height = 1;
    out_cloud->width = out_cloud->points.size();
} 