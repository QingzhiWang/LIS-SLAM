// Author of  EPSC-LOAM : QZ Wang
// Email wangqingzhi27@outlook.com

#ifndef _SUBMAP_H_
#define _SUBMAP_H_

#include "utility.h"
#include "common.h"

#include "keyFrame.h"

typedef Eigen::Matrix<double, 6, 1> Vector6d;
typedef Eigen::Matrix<double, 6, 6> Matrix6d;

struct centerpoint_t
{
	double x;
	double y;
	double z;
	centerpoint_t(double x = 0, double y = 0, double z = 0) : x(x), y(y), z(z) {}
};

//regular bounding box whose edges are parallel to x,y,z axises
struct bounds_t
{
	double min_x;
	double min_y;
	double min_z;
	double max_x;
	double max_y;
	double max_z;
	int type;

	bounds_t()
	{
		min_x = min_y = min_z = max_x = max_y = max_z = 0.0;
	}
	void inf_x()
	{
		min_x = -DBL_MAX;
		max_x = DBL_MAX;
	}
	void inf_y()
	{
		min_y = -DBL_MAX;
		max_y = DBL_MAX;
	}
	void inf_z()
	{
		min_z = -DBL_MAX;
		max_z = DBL_MAX;
	}
	void inf_xyz()
	{
		inf_x();
		inf_y();
		inf_z();
	}
};




//basic common functions of point cloud
template <typename PointT>
class CloudUtility
{
  public:
	//Get Center of a Point Cloud
	void get_cloud_cpt(const typename pcl::PointCloud<PointT>::Ptr &cloud, centerpoint_t &cp)
	{
		double cx = 0, cy = 0, cz = 0;
		int point_num = cloud->points.size();

		for (int i = 0; i < point_num; i++)
		{
			cx += cloud->points[i].x / point_num;
			cy += cloud->points[i].y / point_num;
			cz += cloud->points[i].z / point_num;
		}
		cp.x = cx;
		cp.y = cy;
		cp.z = cz;
	}

	//Get Bound of a Point Cloud
	void get_cloud_bbx(const typename pcl::PointCloud<PointT>::Ptr &cloud, bounds_t &bound)
	{
		double min_x = DBL_MAX;
		double min_y = DBL_MAX;
		double min_z = DBL_MAX;
		double max_x = -DBL_MAX;
		double max_y = -DBL_MAX;
		double max_z = -DBL_MAX;

		for (int i = 0; i < cloud->points.size(); i++)
		{
			if (min_x > cloud->points[i].x)
				min_x = cloud->points[i].x;
			if (min_y > cloud->points[i].y)
				min_y = cloud->points[i].y;
			if (min_z > cloud->points[i].z)
				min_z = cloud->points[i].z;
			if (max_x < cloud->points[i].x)
				max_x = cloud->points[i].x;
			if (max_y < cloud->points[i].y)
				max_y = cloud->points[i].y;
			if (max_z < cloud->points[i].z)
				max_z = cloud->points[i].z;
		}
		bound.min_x = min_x;
		bound.max_x = max_x;
		bound.min_y = min_y;
		bound.max_y = max_y;
		bound.min_z = min_z;
		bound.max_z = max_z;
	}

	//Get Bound and Center of a Point Cloud
	void get_cloud_bbx_cpt(const typename pcl::PointCloud<PointT>::Ptr &cloud, bounds_t &bound, centerpoint_t &cp)
	{
		get_cloud_bbx(cloud, bound);
		cp.x = 0.5 * (bound.min_x + bound.max_x);
		cp.y = 0.5 * (bound.min_y + bound.max_y);
		cp.z = 0.5 * (bound.min_z + bound.max_z);
	}

	void get_intersection_bbx(bounds_t &bbx_1, bounds_t &bbx_2, bounds_t &bbx_intersection, float bbx_boundary_pad = 2.0)
	{
		bbx_intersection.min_x = max_(bbx_1.min_x, bbx_2.min_x) - bbx_boundary_pad;
		bbx_intersection.min_y = max_(bbx_1.min_y, bbx_2.min_y) - bbx_boundary_pad;
		bbx_intersection.min_z = max_(bbx_1.min_z, bbx_2.min_z) - bbx_boundary_pad;
		bbx_intersection.max_x = min_(bbx_1.max_x, bbx_2.max_x) + bbx_boundary_pad;
		bbx_intersection.max_y = min_(bbx_1.max_y, bbx_2.max_y) + bbx_boundary_pad;
		bbx_intersection.max_z = min_(bbx_1.max_z, bbx_2.max_z) + bbx_boundary_pad;
	}

	void merge_bbx(std::vector<bounds_t> &bbxs, bounds_t &bbx_merged)
	{
		bbx_merged.min_x = DBL_MAX;
		bbx_merged.min_y = DBL_MAX;
		bbx_merged.min_z = DBL_MAX;
		bbx_merged.max_x = -DBL_MAX;
		bbx_merged.max_y = -DBL_MAX;
		bbx_merged.max_z = -DBL_MAX;

		for (int i = 0; i < bbxs.size(); i++)
		{
			bbx_merged.min_x = min_(bbx_merged.min_x, bbxs[i].min_x);
			bbx_merged.min_y = min_(bbx_merged.min_y, bbxs[i].min_y);
			bbx_merged.min_z = min_(bbx_merged.min_z, bbxs[i].min_z);
			bbx_merged.max_x = max_(bbx_merged.max_x, bbxs[i].max_x);
			bbx_merged.max_y = max_(bbx_merged.max_y, bbxs[i].max_y);
			bbx_merged.max_z = max_(bbx_merged.max_z, bbxs[i].max_z);
		}
	}

	//Get Bound of Subsets of a Point Cloud
	void get_sub_bbx(typename pcl::PointCloud<PointT>::Ptr &cloud, vector<int> &index, bounds_t &bound)
	{
		typename pcl::PointCloud<PointT>::Ptr temp_cloud(new pcl::PointCloud<PointT>);
		for (int i = 0; i < index.size(); i++)
		{
			temp_cloud->push_back(cloud->points[index[i]]);
		}
		get_cloud_bbx(temp_cloud, bound);
	}

	void get_ring_map(const typename pcl::PointCloud<PointT>::Ptr &cloud_in, ring_map_t &ring_map) //check it later
	{
		for (int i = 0; i < cloud_in->points.size(); i++)
		{
			PointT pt = cloud_in->points[i];
			float dist = std::sqrt(pt.x * pt.x + pt.y * pt.y + pt.z * pt.z);
			float hor_ang = std::atan2(pt.y, pt.x);
			float ver_ang = std::asin(pt.z / dist);

			float u = 0.5 * (1 - hor_ang / M_PI) * ring_map.width;
			float v = (1 - (ver_ang + ring_map.f_up) / (ring_map.f_up + ring_map.f_down)) * ring_map.height;

			ring_map.ring_array[(int)v][(int)u] = i; //save the indice of the point
		}
	}

  protected:
  private:
};



struct submap_t {
    ros::Time timeInfoStamp;

    int submap_id;
    int submap_size;

	bounds_t bound;				  //Bounding Box in geo-coordinate system
	centerpoint_t cp;			  //Center Point in geo-coordinate system

	bounds_t local_bound;				//Bounding Box in local coordinate system
	centerpoint_t local_cp;				//Center Point in local coordinate system
    
    PointTypePose submap_pose_6D_init;
    PointTypePose submap_pose_6D_gt;
    PointTypePose submap_pose_6D_optimized;
    PointType submap_pose_3D_optimized;

    vector<int> keyframe_id_in_submap;
    pcl::PointCloud<PointType>::Ptr keyframe_poses_3D;
    pcl::PointCloud<PointTypePose>::Ptr keyframe_poses_6D;
    map<int,PointType> keyframe_poses_3D_map;
    map<int,PointTypePose> keyframe_poses_6D_map;

    pcl::PointCloud<PointXYZIL>::Ptr submap_dynamic;
    pcl::PointCloud<PointXYZIL>::Ptr submap_static;
    pcl::PointCloud<PointXYZIL>::Ptr submap_outlier;

    pcl::search::KdTree<PointXYZIL>::Ptr tree_dynamic;
    pcl::search::KdTree<PointXYZIL>::Ptr tree_static;
    pcl::search::KdTree<PointXYZIL>::Ptr tree_outlier;

    int feature_point_num;

    map<int, cv::Mat> keyframe_global_descriptor;

    std::string filename;			//full path of the original point cloud file
    std::string filenmae_processed; //full path of the processed point cloud file

    Matrix6d information_matrix_to_next;

	submap_t()
	{
        submap_id = -1;
        submap_size = 0;

		init();

		information_matrix_to_next.setIdentity();

		feature_point_num = 0;
	}

	submap_t(const submap_t &in_block, bool clone_feature = false, bool clone_pose = true)
	{
		init();
		clone_metadata(in_block);

		if (clone_feature)
		{
			*submap_dynamic = *(in_block.submap_dynamic);
			*submap_static = *(in_block.submap_static);
			*submap_outlier = *(in_block.submap_outlier);
		}

        if (clone_pose)
        {
            *keyframe_poses_3D = *(in_block.keyframe_poses_3D);
            *keyframe_poses_6D = *(in_block.keyframe_poses_6D);
        }
	}

	void init()
	{
        submap_dynamic = boost::make_shared<pcl::PointCloud<PointXYZIL>>();
        submap_static = boost::make_shared<pcl::PointCloud<PointXYZIL>>();
        submap_outlier = boost::make_shared<pcl::PointCloud<PointXYZIL>>();
        
        keyframe_poses_3D.reset(new pcl::PointCloud<PointType>());
        keyframe_poses_6D.reset(new pcl::PointCloud<PointTypePose>());
		
        tree_dynamic = boost::make_shared<pcl::search::KdTree<PointXYZIL>>();
		tree_static = boost::make_shared<pcl::search::KdTree<PointXYZIL>>();
		tree_outlier = boost::make_shared<pcl::search::KdTree<PointXYZIL>>();

	}


    void clone_metadata(const submap_t &in_submap) 
    {
        timeInfoStamp = in_submap.timeInfoStamp;
        submap_id = in_submap.submap_id;
        submap_size = in_submap.submap_size;

        submap_pose_6D_init = in_submap.submap_pose_6D_init;
        submap_pose_6D_gt = in_submap.submap_pose_6D_gt;
        submap_pose_6D_optimized = in_submap.submap_pose_6D_optimized;
        submap_pose_3D_optimized = in_submap.submap_pose_3D_optimized;
        
        global_descriptor = in_submap.global_descriptor;

        keyframe_id_in_submap = in_submap.keyframe_id_in_submap;
        keyframe_poses_3D_map = in_submap.keyframe_poses_3D_map;
        keyframe_poses_6D_map = in_submap.keyframe_poses_6D_map;

        feature_point_num = in_submap.feature_point_num;
        
        keyframe_global_descriptor = in_submap.keyframe_global_descriptor;
        
        filename = in_submap.filename;
        filenmae_processed = in_submap.filenmae_processed;
        
        information_matrix_to_next = in_submap.information_matrix_to_next;
    }

	void free()
	{
		submap_dynamic.reset(new pcl::PointCloud<PointXYZIL>());
		submap_static.reset(new pcl::PointCloud<PointXYZIL>());
		submap_outlier.reset(new pcl::PointCloud<PointXYZIL>());

        keyframe_poses_3D.reset(new pcl::PointCloud<PointType>());
        keyframe_poses_6D.reset(new pcl::PointCloud<PointTypePose>());

        tree_dynamic.reset(new pcl::search::KdTree<PointXYZIL>());
		tree_static.reset(new pcl::search::KdTree<PointXYZIL>());
		tree_outlier.reset(new pcl::search::KdTree<PointXYZIL>>());
	}




    void append_feature(const keyframe_t &in_keyframe) 
    {
        submap_dynamic->points.insert(submap_dynamic->points.end(), in_keyframe.cloud_dynamic->points.begin(), in_keyframe.cloud_dynamic->points.end());
        submap_static->points.insert(submap_static->points.end(), in_keyframe.cloud_static->points.begin(), in_keyframe.cloud_static->points.end());
        submap_outlier->points.insert(submap_outlier->points.end(), in_keyframe.cloud_outlier->points.begin(), in_keyframe.cloud_outlier->points.end());
    }
  
    void merge_feature_points(pcl::PointCloud<PointXYZIL>::Ptr &pc_out)
    {
        pc_out->points.insert(pc_out->points.end(), submap_dynamic->points.begin(), submap_dynamic->points.end());
        pc_out->points.insert(pc_out->points.end(), submap_static->points.begin(), submap_static->points.end());
        pc_out->points.insert(pc_out->points.end(), submap_outlier->points.begin(), submap_outlier->points.end());
    }

    void transform_feature(const Eigen::Matrix4d &trans_mat) 
    {
        pcl::transformPointCloud(*cloud_dynamic, *cloud_dynamic, trans_mat);
        pcl::transformPointCloud(*cloud_static, *cloud_static, trans_mat);
        pcl::transformPointCloud(*cloud_outlier, *cloud_outlier, trans_mat);
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

typedef boost::shared_ptr<submap_t> submap_Ptr;



template <typename PointT>
class SubMapManager : public ParamServer, SemanticLabelParam, CloudUtility<PointT>{
public:
    bool update_local_map(submap_Ptr local_map, keyframe_Ptr last_target_cblock,
                    float local_map_radius = 80, int max_num_pts = 20000, int kept_vertex_num = 800,
                    float last_frame_reliable_radius = 60,
                    bool map_based_dynamic_removal_on = false,
                    float dynamic_removal_center_radius = 30.0,
                    float dynamic_dist_thre_min = 0.3,
                    float dynamic_dist_thre_max = 3.0,
                    float near_dist_thre = 0.03,
                    bool recalculate_feature_on = false);

    bool map_based_dynamic_close_removal(submap_Ptr local_map, keyframe_Ptr last_target_cblock,
                                    float center_radius, float dynamic_dist_thre_min, float dynamic_dist_thre_max, float near_dist_thre);

    //keep the points meet dist ~ (near_dist_thre, dynamic_dist_thre_min) U (dynamic_dist_thre_max, +inf)
    //filter the points meet dist ~ (0, near_dist_thre] U [dynamic_dist_thre_min, dynamic_dist_thre_max]
    bool map_scan_feature_pts_distance_removal(pcl::PointCloud<PointT>::Ptr feature_pts, const pcl::search::KdTree<PointT>::Ptr map_kdtree, float center_radius,
                                            float dynamic_dist_thre_min = FLT_MAX, float dynamic_dist_thre_max = FLT_MAX, float near_dist_thre = 0.0);

    bool update_cloud_vectors(pcl::PointCloud<PointT>::Ptr feature_pts, const pcl::search::KdTree<PointT>::Ptr map_kdtree,
                        float pca_radius = 1.5, int pca_k = 20,
                        int k_min = 8, float sin_low = 0.5, float sin_high = 0.5, float min_linearity = 0.0);

    bool judge_new_submap(float &accu_tran, float &accu_rot, int &accu_frame,
                    float max_accu_tran = 30.0, float max_accu_rot = 90.0, int max_accu_frame = 150);




    //Bridef: Used for the preprocessing of fine registration
    //Use the intersection bounding box to filter the outlier points (--> to speed up)
    bool get_cloud_pair_intersection(bounds_t &intersection_bbx,
                                        typename pcl::PointCloud<PointT>::Ptr &pc_dynamic_tc,
                                        typename pcl::PointCloud<PointT>::Ptr &pc_static_tc,
                                        typename pcl::PointCloud<PointT>::Ptr &pc_outlier_tc)
    {
        bbx_filter(pc_dynamic_tc, intersection_bbx);
        bbx_filter(pc_static_tc, intersection_bbx);
        bbx_filter(pc_outlier_tc, intersection_bbx);

        //LOG(INFO) << "Intersection Bounding box filtering done";
        return 1;
    }

	bool bbx_filter(const typename pcl::PointCloud<PointT>::Ptr &cloud_in_out, bounds_t &bbx, bool delete_box = false)
	{
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

    void label2RGBCloud(pcl::PointCloud<PointXYZIL>::Ptr in_cloud, pcl::PointCloud<pcl::PointXYZRGB>::Ptr &out_cloud){
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
};





#endif  // _SUBMAP_H_