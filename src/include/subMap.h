// Author of  EPSC-LOAM : QZ Wang
// Email wangqingzhi27@outlook.com

#ifndef _SUBMAP_H_
#define _SUBMAP_H_

#include "utility.h"
#include "common.h"

typedef Eigen::Matrix<double, 6, 1> Vector6d;
typedef Eigen::Matrix<double, 6, 6> Matrix6d;

struct centerpoint_t
{
	double x;
	double y;
	double z;
	centerpoint_t(double x = 0, double y = 0, double z = 0) : x(x), y(y), z(z) {}

    centerpoint_t& operator=(const centerpoint_t &in_cp)
    {
        if(this != &in_cp)
        {
            x = in_cp.x;
            y = in_cp.y;
            z = in_cp.z;

            return *this;
        }
    }
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

    bounds_t(const bounds_t &in_bound)
    {
        min_x = in_bound.min_x;
        min_y = in_bound.min_y;
        min_z = in_bound.min_z;
        max_x = in_bound.max_x;
        max_y = in_bound.max_y;
        max_z = in_bound.max_z;
    }

    bounds_t& operator=(const bounds_t &in_bound)
    {
        if(this != &in_bound)
        {
            min_x = in_bound.min_x;
            min_y = in_bound.min_y;
            min_z = in_bound.min_z;
            max_x = in_bound.max_x;
            max_y = in_bound.max_y;
            max_z = in_bound.max_z;

            return *this;
        }
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
		bbx_intersection.min_x = std::max(bbx_1.min_x, bbx_2.min_x) - bbx_boundary_pad;
		bbx_intersection.min_y = std::max(bbx_1.min_y, bbx_2.min_y) - bbx_boundary_pad;
		bbx_intersection.min_z = std::max(bbx_1.min_z, bbx_2.min_z) - bbx_boundary_pad;
		bbx_intersection.max_x = std::min(bbx_1.max_x, bbx_2.max_x) + bbx_boundary_pad;
		bbx_intersection.max_y = std::min(bbx_1.max_y, bbx_2.max_y) + bbx_boundary_pad;
		bbx_intersection.max_z = std::min(bbx_1.max_z, bbx_2.max_z) + bbx_boundary_pad;
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
			bbx_merged.min_x = std::min(bbx_merged.min_x, bbxs[i].min_x);
			bbx_merged.min_y = std::min(bbx_merged.min_y, bbxs[i].min_y);
			bbx_merged.min_z = std::min(bbx_merged.min_z, bbxs[i].min_z);
			bbx_merged.max_x = std::max(bbx_merged.max_x, bbxs[i].max_x);
			bbx_merged.max_y = std::max(bbx_merged.max_y, bbxs[i].max_y);
			bbx_merged.max_z = std::max(bbx_merged.max_z, bbxs[i].max_z);
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


    void transform_bbx(bounds_t &bound, Eigen::Affine3f &transCur)
    {
        
        bound.max_x = transCur(0, 0) * bound.max_x + transCur(0, 3);
        bound.max_y = transCur(1, 1) * bound.max_y + transCur(1, 3);
        bound.max_z = transCur(2, 2) * bound.max_z + transCur(2, 3);
        bound.min_x = transCur(0, 0) * bound.min_x + transCur(0, 3);
        bound.min_y = transCur(1, 1) * bound.min_y + transCur(1, 3);
        bound.min_z = transCur(2, 2) * bound.min_z + transCur(2, 3);  
    }

    void transform_bbx(bounds_t &bound_in, bounds_t &bound_out, Eigen::Affine3f &transCur)
    {
        
        bound_out.max_x = transCur(0, 0) * bound_in.max_x + transCur(0, 3);
        bound_out.max_y = transCur(1, 1) * bound_in.max_y + transCur(1, 3);
        bound_out.max_z = transCur(2, 2) * bound_in.max_z + transCur(2, 3);
        bound_out.min_x = transCur(0, 0) * bound_in.min_x + transCur(0, 3);
        bound_out.min_y = transCur(1, 1) * bound_in.min_y + transCur(1, 3);
        bound_out.min_z = transCur(2, 2) * bound_in.min_z + transCur(2, 3);  
    }

  protected:
  private:
};




struct keyframe_t 
{
    ros::Time timeInfoStamp;
    int keyframe_id;
    int submap_id;
    int id_in_submap;

    PointTypePose init_pose;
    PointTypePose optimized_pose;
    PointTypePose gt_pose;
    PointTypePose relative_pose;

	bounds_t bound;				  //Bounding Box in geo-coordinate system
	centerpoint_t cp;			  //Center Point in geo-coordinate system

	bounds_t local_bound;				//Bounding Box in local coordinate system
	centerpoint_t local_cp;				//Center Point in local coordinate system
    
    
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

    keyframe_t() 
    {
        keyframe_id = -1;
        submap_id = -1;
        id_in_submap = -1;

        init();
        // default value

    }

    keyframe_t(const keyframe_t &in_keyframe, bool clone_cloud = false) 
    {
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

    void init() 
    {
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

    void clone_metadata(const keyframe_t &in_keyframe) 
    {
        timeInfoStamp = in_keyframe.timeInfoStamp;
        keyframe_id = in_keyframe.keyframe_id;
        submap_id = in_keyframe.submap_id;
        id_in_submap = in_keyframe.id_in_submap;

        init_pose = in_keyframe.init_pose;
        optimized_pose = in_keyframe.optimized_pose;
        gt_pose = in_keyframe.gt_pose;
        relative_pose = in_keyframe.relative_pose;

        bound = in_keyframe.bound;
        cp = in_keyframe.cp;	
        local_bound = in_keyframe.local_bound;
        local_cp = in_keyframe.local_cp;

        global_descriptor = in_keyframe.global_descriptor;
        loop_container = in_keyframe.loop_container;
    }

    void free_all() 
    {
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

    void transform_feature(const Eigen::Affine3f &trans_mat,
                            bool transform_down = true,
                            bool transform_undown = true) 
    {
        if (transform_undown) {
            // pcl::transformPointCloud(*cloud_semantic, *cloud_semantic, trans_mat);
            pcl::transformPointCloud(*cloud_dynamic, *cloud_dynamic, trans_mat);
            pcl::transformPointCloud(*cloud_static, *cloud_static, trans_mat);
            pcl::transformPointCloud(*cloud_outlier, *cloud_outlier, trans_mat);
            // pcl::transformPointCloud(*cloud_corner, *cloud_corner, trans_mat); 
            // pcl::transformPointCloud(*cloud_surface, *cloud_surface, trans_mat);
        }
        if (transform_down) 
        {
            // pcl::transformPointCloud(*cloud_semantic_down, *cloud_semantic_down, trans_mat);
            pcl::transformPointCloud(*cloud_dynamic_down, *cloud_dynamic_down, trans_mat);
            pcl::transformPointCloud(*cloud_static_down, *cloud_static_down, trans_mat);
            pcl::transformPointCloud(*cloud_outlier_down, *cloud_outlier_down, trans_mat);
            // pcl::transformPointCloud(*cloud_corner_down, *cloud_corner_down, trans_mat);
            // pcl::transformPointCloud(*cloud_surface_down, *cloud_surface_down, trans_mat);
        }
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

typedef boost::shared_ptr<keyframe_t> keyframe_Ptr;




struct submap_t 
{
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

        bound = in_submap.bound;
        cp = in_submap.cp;	
        local_bound = in_submap.local_bound;
        local_cp = in_submap.local_cp;

        submap_pose_6D_init = in_submap.submap_pose_6D_init;
        submap_pose_6D_gt = in_submap.submap_pose_6D_gt;
        submap_pose_6D_optimized = in_submap.submap_pose_6D_optimized;
        submap_pose_3D_optimized = in_submap.submap_pose_3D_optimized;

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
        
        free_tree();
	}

	void free_tree()
	{
        tree_dynamic.reset(new pcl::search::KdTree<PointXYZIL>());
		tree_static.reset(new pcl::search::KdTree<PointXYZIL>());
		tree_outlier.reset(new pcl::search::KdTree<PointXYZIL>());
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

    void transform_feature(const Eigen::Affine3f &trans_mat) 
    {
        pcl::transformPointCloud(*submap_dynamic, *submap_dynamic, trans_mat);
        pcl::transformPointCloud(*submap_static, *submap_static, trans_mat);
        pcl::transformPointCloud(*submap_outlier, *submap_outlier, trans_mat);
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

typedef boost::shared_ptr<submap_t> submap_Ptr;



template <typename PointT>
class SubMapManager : public ParamServer, public SemanticLabelParam, public CloudUtility<PointT>
{
public:
    bool fisrt_submap(submap_Ptr &local_map, keyframe_Ptr &last_target_cblock)
    {
        local_map->free();
        
        local_map->timeInfoStamp = last_target_cblock->timeInfoStamp;    
        local_map->submap_id = last_target_cblock->submap_id;    
        local_map->submap_size = 1;    
        
        local_map->submap_pose_6D_init = last_target_cblock->optimized_pose;    
        local_map->submap_pose_6D_optimized = last_target_cblock->optimized_pose;    
        
        PointType thisPose3D;
        thisPose3D.x = last_target_cblock->optimized_pose.x;
        thisPose3D.y = last_target_cblock->optimized_pose.y;
        thisPose3D.z = last_target_cblock->optimized_pose.z;
        thisPose3D.intensity = last_target_cblock->submap_id;

        local_map->submap_pose_3D_optimized = thisPose3D;   
        
        local_map->keyframe_id_in_submap.push_back(last_target_cblock->keyframe_id);
        local_map->keyframe_poses_6D->points.push_back(last_target_cblock->optimized_pose);
        local_map->keyframe_poses_3D->points.push_back(thisPose3D);
        
        local_map->keyframe_poses_3D_map.insert(std::make_pair(last_target_cblock->keyframe_id, thisPose3D));
        local_map->keyframe_poses_6D_map.insert(std::make_pair(last_target_cblock->keyframe_id, last_target_cblock->optimized_pose));

        local_map->append_feature(*last_target_cblock);  

        local_map->feature_point_num = local_map->submap_dynamic->points.size() + 
                                    local_map->submap_static->points.size() + 
                                    local_map->submap_outlier->points.size();

        typename pcl::PointCloud<PointT>::Ptr cloud_raw(new pcl::PointCloud<PointT>);
        
        //calculate bbx (local)
        local_map->merge_feature_points(cloud_raw);
        this->get_cloud_bbx(cloud_raw, local_map->local_bound);

        //calculate bbx (global)
        Eigen::Affine3f tran_map = pclPointToAffine3f(local_map->submap_pose_6D_optimized);
        this->transform_bbx(local_map->local_bound, local_map->bound, tran_map);

        local_map->free_tree();

    }


    bool update_submap(
            submap_Ptr &local_map, keyframe_Ptr &last_target_cblock,
            float local_map_radius = 80, 
            int max_num_pts = 20000, 
            int kept_vertex_num = 800,
            float last_frame_reliable_radius = 60,
            bool map_based_dynamic_removal_on = false,
            float dynamic_removal_center_radius = 30.0,
            float dynamic_dist_thre_min = 0.3,
            float dynamic_dist_thre_max = 3.0,
            float near_dist_thre = 0.03)
    {

        std::chrono::steady_clock::time_point tic = std::chrono::steady_clock::now();

        Eigen::Affine3f tran_target_map; //from the last local_map to the target frame
        tran_target_map = pclPointToAffine3f(last_target_cblock->optimized_pose).inverse() * pclPointToAffine3f(local_map->submap_pose_6D_optimized);
        
        last_target_cblock->transform_feature(tran_target_map.inverse(), true, false);
        // typename pcl::PointCloud<PointT>::Ptr cloud_dynamic(new pcl::PointCloud<PointT>);
        // typename pcl::PointCloud<PointT>::Ptr cloud_static(new pcl::PointCloud<PointT>);
        // typename pcl::PointCloud<PointT>::Ptr cloud_outlier(new pcl::PointCloud<PointT>);
        // pcl::copyPointCloud(*last_target_cblock->cloud_dynamic, *cloud_dynamic);
        // pcl::copyPointCloud(*last_target_cblock->cloud_static, *cloud_static);
        // pcl::copyPointCloud(*last_target_cblock->cloud_outlier, *cloud_outlier);

        // Eigen::Affine3f tran_target_map_inv = tran_target_map.inverse();
        // pcl::transformPointCloud(*cloud_dynamic, *cloud_dynamic, tran_target_map_inv);
        // pcl::transformPointCloud(*cloud_static, *cloud_static, tran_target_map_inv);
        // pcl::transformPointCloud(*cloud_outlier, *cloud_outlier, tran_target_map_inv);
        
        dynamic_dist_thre_max = std::max(dynamic_dist_thre_max, (float)(dynamic_dist_thre_min + 0.1));
        std::cout << "Map based filtering range(m): (0, " << near_dist_thre << "] U [" << dynamic_dist_thre_min << "," << dynamic_dist_thre_max << "]" << std::endl;

        if (map_based_dynamic_removal_on && local_map->feature_point_num > max_num_pts / 5)
        {
            map_based_dynamic_close_removal(local_map, last_target_cblock, dynamic_removal_center_radius,
                                            dynamic_dist_thre_min, dynamic_dist_thre_max, near_dist_thre);

            std::cout << "Feature point number of last frame after dynamic removal: Dynamic: [" 
                    << last_target_cblock->cloud_dynamic->points.size() << " | "
                    << last_target_cblock->cloud_dynamic_down->points.size() << "]." << std::endl;
        }  
        local_map->append_feature(*last_target_cblock);

        last_target_cblock->transform_feature(tran_target_map, true, false);
        
        local_map->submap_size++;
        local_map->keyframe_id_in_submap.push_back(last_target_cblock->keyframe_id);

        PointType thisPose3D;
        thisPose3D.x = last_target_cblock->optimized_pose.x;
        thisPose3D.y = last_target_cblock->optimized_pose.y;
        thisPose3D.z = last_target_cblock->optimized_pose.z;
        thisPose3D.intensity = last_target_cblock->keyframe_id;
        local_map->keyframe_poses_3D->points.push_back(thisPose3D);

        local_map->keyframe_poses_6D->points.push_back(last_target_cblock->optimized_pose);

        local_map->keyframe_poses_3D_map.insert(std::make_pair(last_target_cblock->keyframe_id, thisPose3D));
        local_map->keyframe_poses_6D_map.insert(std::make_pair(last_target_cblock->keyframe_id, last_target_cblock->optimized_pose));

        local_map->feature_point_num = local_map->submap_dynamic->points.size() + 
                                    local_map->submap_static->points.size() + 
                                    local_map->submap_outlier->points.size();

        typename pcl::PointCloud<PointT>::Ptr cloud_raw(new pcl::PointCloud<PointT>);
        
        //calculate bbx (local)
        local_map->merge_feature_points(cloud_raw);
        this->get_cloud_bbx(cloud_raw, local_map->local_bound);

        //calculate bbx (global)
        Eigen::Affine3f tran_map = pclPointToAffine3f(local_map->submap_pose_6D_optimized);
        // transform_bbx(local_map->bound, tran_map);
        this->transform_bbx(local_map->local_bound, local_map->bound, tran_map);

        local_map->free_tree();
        
        std::chrono::steady_clock::time_point toc = std::chrono::steady_clock::now();
        std::chrono::duration<double> time_update_local_map = std::chrono::duration_cast<std::chrono::duration<double>>(toc - tic);
        std::cout << "Update local map ([" << local_map->feature_point_num << "] points at present) done in [" << time_update_local_map.count() * 1000.0 << "] ms.\n";
        return true;
    }



    bool map_based_dynamic_close_removal(
            submap_Ptr &local_map, keyframe_Ptr &last_target_cblock,
            float center_radius, float dynamic_dist_thre_min, 
            float dynamic_dist_thre_max, float near_dist_thre)
    {
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
    bool map_scan_feature_pts_distance_removal(
            typename pcl::PointCloud<PointT>::Ptr feature_pts, 
            const typename pcl::search::KdTree<PointT>::Ptr map_kdtree, 
            float center_radius, float dynamic_dist_thre_min = FLT_MAX, 
            float dynamic_dist_thre_max = FLT_MAX, float near_dist_thre = 0.0)
    {
        if (feature_pts->points.size() <= 10)
            return false;

        typename pcl::PointCloud<PointT>::Ptr  feature_pts_temp(new pcl::PointCloud<PointT>);
        
        //#pragma omp parallel for private(i) //Multi-thread
        std::vector<float> distances_square;
        std::vector<int> search_indices;
        for (int i = 0; i < feature_pts->points.size(); i++)
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


    bool judge_new_submap(
            float &accu_tran, float &accu_rot, int &accu_frame,
            float max_accu_tran = 30.0, float max_accu_rot = 90.0, 
            int max_accu_frame = 150)
    {
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




    bool random_downsample(typename pcl::PointCloud<PointT>::Ptr &cloud_in_out, int downsample_ratio)
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
    bool random_downsample_pcl(typename pcl::PointCloud<PointT>::Ptr &cloud_in_out, int keep_number)
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
                typename pcl::RandomSample<PointT> ran_sample(true); // Extract removed indices
                ran_sample.setInputCloud(cloud_in_out);
                ran_sample.setSample(keep_number);
                ran_sample.filter(*cloud_temp);
                cloud_temp->points.swap(cloud_in_out->points);
                return true;
            }
        }
    }


    bool voxel_downsample_pcl(typename pcl::PointCloud<PointT>::Ptr &cloud_in_out, float leaf_size)
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



    bool voxel_downsample_pcl(typename pcl::PointCloud<PointT>::Ptr &cloud_in, typename pcl::PointCloud<PointT>::Ptr &cloud_out, float leaf_size)
    {
        if (cloud_in->points.size() <= 0)
            return false;

        pcl::VoxelGrid<PointT> downSizeFilter;
        downSizeFilter.setLeafSize(leaf_size, leaf_size, leaf_size);
        downSizeFilter.setInputCloud(cloud_in);
        downSizeFilter.filter(*cloud_out);

        return true;    
    }

    bool voxel_downsample_pcl(pcl::PointCloud<PointType>::Ptr &cloud_in, pcl::PointCloud<PointType>::Ptr &cloud_out, float leaf_size)
    {
        if (cloud_in->points.size() <= 0)
            return false;

        pcl::VoxelGrid<PointType> downSizeFilter;
        downSizeFilter.setLeafSize(leaf_size, leaf_size, leaf_size);
        downSizeFilter.setInputCloud(cloud_in);
        downSizeFilter.filter(*cloud_out);

        return true;    
    }

    void label2RGBCloud(pcl::PointCloud<PointXYZIL>::Ptr in_cloud, 
                                    pcl::PointCloud<pcl::PointXYZRGB>::Ptr &out_cloud)
    {
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