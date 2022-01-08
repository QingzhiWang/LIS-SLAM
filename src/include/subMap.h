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

	void get_bound_cpt(const bounds_t &bound, centerpoint_t &cp)
	{
		cp.x = 0.5 * (bound.min_x + bound.max_x);
		cp.y = 0.5 * (bound.min_y + bound.max_y);
		cp.z = 0.5 * (bound.min_z + bound.max_z);
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
        bound_out.max_x = bound_in.max_x + transCur(0, 3);
        bound_out.max_y = bound_in.max_y + transCur(1, 3);
        bound_out.max_z = bound_in.max_z + transCur(2, 3);
        bound_out.min_x = bound_in.min_x + transCur(0, 3);
        bound_out.min_y = bound_in.min_y + transCur(1, 3);
        bound_out.min_z = bound_in.min_z + transCur(2, 3);  
    }


	void transform_bbx(bounds_t &bound_in, centerpoint_t &cp_in, bounds_t &bound_out, centerpoint_t &cp_out, Eigen::Affine3f &transCur)
    {
        cp_out.x = transCur(0, 0) * cp_in.x + transCur(0, 1) * cp_in.y + transCur(0, 2) * cp_in.z + transCur(0, 3);
        cp_out.y = transCur(1, 0) * cp_in.x + transCur(1, 1) * cp_in.y + transCur(1, 2) * cp_in.z + transCur(1, 3);
        cp_out.z = transCur(2, 0) * cp_in.x + transCur(2, 1) * cp_in.y + transCur(2, 2) * cp_in.z + transCur(2, 3);

        bound_out.max_x = bound_in.max_x - cp_in.x + cp_out.x;
        bound_out.max_y = bound_in.max_y - cp_in.y + cp_out.y;
        bound_out.max_z = bound_in.max_z - cp_in.z + cp_out.z;
        bound_out.min_x = bound_in.min_x - cp_in.x + cp_out.x;
        bound_out.min_y = bound_in.min_y - cp_in.y + cp_out.y;
        bound_out.min_z = bound_in.min_z - cp_in.z + cp_out.z; 
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

    pcl::PointCloud<PointXYZIL>::Ptr semantic_raw;
    pcl::PointCloud<PointXYZIL>::Ptr semantic_dynamic;
    pcl::PointCloud<PointXYZIL>::Ptr semantic_pole;
    pcl::PointCloud<PointXYZIL>::Ptr semantic_ground;
    pcl::PointCloud<PointXYZIL>::Ptr semantic_building;
    pcl::PointCloud<PointXYZIL>::Ptr semantic_outlier;

    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_corner;
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_surface;

    pcl::PointCloud<PointXYZIL>::Ptr semantic_raw_down;
    pcl::PointCloud<PointXYZIL>::Ptr semantic_dynamic_down;
    pcl::PointCloud<PointXYZIL>::Ptr semantic_pole_down;
    pcl::PointCloud<PointXYZIL>::Ptr semantic_ground_down;
    pcl::PointCloud<PointXYZIL>::Ptr semantic_building_down;
    pcl::PointCloud<PointXYZIL>::Ptr semantic_outlier_down;

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
            *semantic_raw = *(in_keyframe.semantic_raw);
            *semantic_dynamic = *(in_keyframe.semantic_dynamic);
            *semantic_pole = *(in_keyframe.semantic_pole);
            *semantic_ground = *(in_keyframe.semantic_ground);
            *semantic_building = *(in_keyframe.semantic_building);
            *semantic_outlier = *(in_keyframe.semantic_outlier);

            *cloud_corner = *(in_keyframe.cloud_corner);
            *cloud_surface = *(in_keyframe.cloud_surface);

            *semantic_raw_down = *(in_keyframe.semantic_raw_down);
            *semantic_dynamic_down = *(in_keyframe.semantic_dynamic_down);
            *semantic_pole_down = *(in_keyframe.semantic_pole_down);
            *semantic_ground_down = *(in_keyframe.semantic_ground_down);
            *semantic_building_down = *(in_keyframe.semantic_building_down);
            *semantic_outlier_down = *(in_keyframe.semantic_outlier_down);

            *cloud_corner_down = *(in_keyframe.cloud_corner_down);
            *cloud_surface_down = *(in_keyframe.cloud_surface_down);

        }
    }

    void init() 
    {
        semantic_raw = boost::make_shared<pcl::PointCloud<PointXYZIL>>();
        semantic_dynamic = boost::make_shared<pcl::PointCloud<PointXYZIL>>();
        semantic_pole = boost::make_shared<pcl::PointCloud<PointXYZIL>>();
        semantic_ground = boost::make_shared<pcl::PointCloud<PointXYZIL>>();
        semantic_building = boost::make_shared<pcl::PointCloud<PointXYZIL>>();
        semantic_outlier = boost::make_shared<pcl::PointCloud<PointXYZIL>>();

        cloud_corner = boost::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
        cloud_surface = boost::make_shared<pcl::PointCloud<pcl::PointXYZI>>();

        semantic_raw_down = boost::make_shared<pcl::PointCloud<PointXYZIL>>();
        semantic_dynamic_down = boost::make_shared<pcl::PointCloud<PointXYZIL>>();
        semantic_pole_down = boost::make_shared<pcl::PointCloud<PointXYZIL>>();
        semantic_ground_down = boost::make_shared<pcl::PointCloud<PointXYZIL>>();
        semantic_building_down = boost::make_shared<pcl::PointCloud<PointXYZIL>>();
        semantic_outlier_down = boost::make_shared<pcl::PointCloud<PointXYZIL>>();

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
        semantic_raw.reset(new pcl::PointCloud<PointXYZIL>());
        semantic_dynamic.reset(new pcl::PointCloud<PointXYZIL>());
        semantic_pole.reset(new pcl::PointCloud<PointXYZIL>());
        semantic_ground.reset(new pcl::PointCloud<PointXYZIL>());
        semantic_building.reset(new pcl::PointCloud<PointXYZIL>());
        semantic_outlier.reset(new pcl::PointCloud<PointXYZIL>());

        cloud_corner.reset(new pcl::PointCloud<pcl::PointXYZI>());
        cloud_surface.reset(new pcl::PointCloud<pcl::PointXYZI>());

        semantic_raw_down.reset(new pcl::PointCloud<PointXYZIL>());
        semantic_dynamic_down.reset(new pcl::PointCloud<PointXYZIL>());
        semantic_pole_down.reset(new pcl::PointCloud<PointXYZIL>());
        semantic_ground_down.reset(new pcl::PointCloud<PointXYZIL>());
        semantic_building_down.reset(new pcl::PointCloud<PointXYZIL>());
        semantic_outlier_down.reset(new pcl::PointCloud<PointXYZIL>());

        cloud_corner_down.reset(new pcl::PointCloud<pcl::PointXYZI>());
        cloud_surface_down.reset(new pcl::PointCloud<pcl::PointXYZI>());
    }
    
    // 不能正确被转换 未找到原因 但是在结构体外部使用内部操作可以完成转换
    void transform_feature(const PointTypePose *trans_mat,
                            bool transform_down = true,
                            bool transform_undown = true) 
    {
        ROS_WARN("trans_mat : relative_pose: [%f, %f, %f, %f, %f, %f]",
                trans_mat->roll, trans_mat->pitch, trans_mat->yaw,
                trans_mat->x, trans_mat->y, trans_mat->z);

        if (transform_undown) {
            // *semantic_raw = *transformPointCloud(semantic_raw, trans_mat);
            
            *this->semantic_dynamic = *transformPointCloud(semantic_dynamic, trans_mat);
            *this->semantic_pole = *transformPointCloud(semantic_pole, trans_mat);
            *this->semantic_ground = *transformPointCloud(semantic_ground, trans_mat);
            *this->semantic_building = *transformPointCloud(semantic_building, trans_mat);
            *this->semantic_outlier = *transformPointCloud(semantic_outlier, trans_mat);
            
            // *cloud_corner = *transformPointCloud(cloud_corner, trans_mat);
            // *cloud_surface = *transformPointCloud(cloud_surface, trans_mat);
        }
        if (transform_down) 
        {
            // *semantic_raw_down = *transformPointCloud(semantic_raw_down, trans_mat);
            
            *this->semantic_dynamic_down = *transformPointCloud(semantic_dynamic_down, trans_mat);
            *this->semantic_pole_down = *transformPointCloud(semantic_pole_down, trans_mat);
            *this->semantic_ground_down = *transformPointCloud(semantic_ground_down, trans_mat);
            *this->semantic_building_down = *transformPointCloud(semantic_building_down, trans_mat);
            *this->semantic_outlier_down = *transformPointCloud(semantic_outlier_down, trans_mat);
            
            // *cloud_corner_down = *transformPointCloud(cloud_corner_down, trans_mat);
            // *cloud_surface_down = *transformPointCloud(cloud_surface_down, trans_mat);
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
    pcl::PointCloud<PointXYZIL>::Ptr submap_pole;
    pcl::PointCloud<PointXYZIL>::Ptr submap_ground;
    pcl::PointCloud<PointXYZIL>::Ptr submap_building;
    pcl::PointCloud<PointXYZIL>::Ptr submap_outlier;

    pcl::search::KdTree<PointXYZIL>::Ptr tree_dynamic;
    pcl::search::KdTree<PointXYZIL>::Ptr tree_pole;
    pcl::search::KdTree<PointXYZIL>::Ptr tree_ground;
    pcl::search::KdTree<PointXYZIL>::Ptr tree_building;
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
			*submap_pole = *(in_block.submap_pole);
			*submap_ground = *(in_block.submap_ground);
			*submap_building = *(in_block.submap_building);
			*submap_outlier = *(in_block.submap_outlier);

            // pcl::copyPointCloud(*in_block.submap_dynamic,  *submap_dynamic);
            // pcl::copyPointCloud(*in_block.submap_pole,  *submap_pole);
            // pcl::copyPointCloud(*in_block.submap_ground,  *submap_ground);
            // pcl::copyPointCloud(*in_block.submap_building,  *submap_building);
            // pcl::copyPointCloud(*in_block.submap_outlier,  *submap_outlier);
		}

        if (clone_pose)
        {
            *keyframe_poses_3D = *(in_block.keyframe_poses_3D);
            *keyframe_poses_6D = *(in_block.keyframe_poses_6D);

            // pcl::copyPointCloud(*in_block.keyframe_poses_3D,  *keyframe_poses_3D);
            // pcl::copyPointCloud(*in_block.keyframe_poses_6D,  *keyframe_poses_6D);
        }
	}

	void init()
	{
        submap_dynamic = boost::make_shared<pcl::PointCloud<PointXYZIL>>();
        submap_pole = boost::make_shared<pcl::PointCloud<PointXYZIL>>();
        submap_ground = boost::make_shared<pcl::PointCloud<PointXYZIL>>();
        submap_building = boost::make_shared<pcl::PointCloud<PointXYZIL>>();
        submap_outlier = boost::make_shared<pcl::PointCloud<PointXYZIL>>();
        
        keyframe_poses_3D.reset(new pcl::PointCloud<PointType>());
        keyframe_poses_6D.reset(new pcl::PointCloud<PointTypePose>());
		
        tree_dynamic = boost::make_shared<pcl::search::KdTree<PointXYZIL>>();
		tree_pole = boost::make_shared<pcl::search::KdTree<PointXYZIL>>();
		tree_ground = boost::make_shared<pcl::search::KdTree<PointXYZIL>>();
		tree_building = boost::make_shared<pcl::search::KdTree<PointXYZIL>>();
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
		submap_pole.reset(new pcl::PointCloud<PointXYZIL>());
		submap_ground.reset(new pcl::PointCloud<PointXYZIL>());
		submap_building.reset(new pcl::PointCloud<PointXYZIL>());
		submap_outlier.reset(new pcl::PointCloud<PointXYZIL>());

        keyframe_poses_3D.reset(new pcl::PointCloud<PointType>());
        keyframe_poses_6D.reset(new pcl::PointCloud<PointTypePose>());
        
        free_tree();
	}

	void free_tree()
	{
        tree_dynamic.reset(new pcl::search::KdTree<PointXYZIL>());
		tree_pole.reset(new pcl::search::KdTree<PointXYZIL>());
		tree_ground.reset(new pcl::search::KdTree<PointXYZIL>());
		tree_building.reset(new pcl::search::KdTree<PointXYZIL>());
		tree_outlier.reset(new pcl::search::KdTree<PointXYZIL>());
	}


    void append_feature(const keyframe_t &in_keyframe, bool using_down_cloud = false) 
    {
        if(using_down_cloud){
            submap_dynamic->points.insert(submap_dynamic->points.end(), in_keyframe.semantic_dynamic_down->points.begin(), in_keyframe.semantic_dynamic_down->points.end());
            submap_pole->points.insert(submap_pole->points.end(), in_keyframe.semantic_pole_down->points.begin(), in_keyframe.semantic_pole_down->points.end());
            submap_ground->points.insert(submap_ground->points.end(), in_keyframe.semantic_ground_down->points.begin(), in_keyframe.semantic_ground_down->points.end());
            submap_building->points.insert(submap_building->points.end(), in_keyframe.semantic_building_down->points.begin(), in_keyframe.semantic_building_down->points.end());
            submap_outlier->points.insert(submap_outlier->points.end(), in_keyframe.semantic_outlier_down->points.begin(), in_keyframe.semantic_outlier_down->points.end());
        }else{
            submap_dynamic->points.insert(submap_dynamic->points.end(), in_keyframe.semantic_dynamic->points.begin(), in_keyframe.semantic_dynamic->points.end());
            submap_pole->points.insert(submap_pole->points.end(), in_keyframe.semantic_pole->points.begin(), in_keyframe.semantic_pole->points.end());
            submap_ground->points.insert(submap_ground->points.end(), in_keyframe.semantic_ground->points.begin(), in_keyframe.semantic_ground->points.end());
            submap_building->points.insert(submap_building->points.end(), in_keyframe.semantic_building->points.begin(), in_keyframe.semantic_building->points.end());
            submap_outlier->points.insert(submap_outlier->points.end(), in_keyframe.semantic_outlier->points.begin(), in_keyframe.semantic_outlier->points.end());
        }
    }


  	void append_feature(const pcl::PointCloud<PointXYZIL>::Ptr &dynamic_in, 
						const pcl::PointCloud<PointXYZIL>::Ptr &pole_in,
						const pcl::PointCloud<PointXYZIL>::Ptr &ground_in,
						const pcl::PointCloud<PointXYZIL>::Ptr &building_in,
						const pcl::PointCloud<PointXYZIL>::Ptr &outlier_in) 
    {
		submap_dynamic->points.insert(submap_dynamic->points.end(), dynamic_in->points.begin(), dynamic_in->points.end());
		submap_pole->points.insert(submap_pole->points.end(), pole_in->points.begin(), pole_in->points.end());
		submap_ground->points.insert(submap_ground->points.end(), ground_in->points.begin(), ground_in->points.end());
		submap_building->points.insert(submap_building->points.end(), building_in->points.begin(), building_in->points.end());
		submap_outlier->points.insert(submap_outlier->points.end(), outlier_in->points.begin(), outlier_in->points.end());
    }


    void merge_feature_points(pcl::PointCloud<PointXYZIL>::Ptr &pc_out, bool merge_outlier = true)
    {	
		if(merge_outlier)
		{
			pc_out->points.insert(pc_out->points.end(), submap_dynamic->points.begin(), submap_dynamic->points.end());
			pc_out->points.insert(pc_out->points.end(), submap_pole->points.begin(), submap_pole->points.end());
			pc_out->points.insert(pc_out->points.end(), submap_ground->points.begin(), submap_ground->points.end());
			pc_out->points.insert(pc_out->points.end(), submap_building->points.begin(), submap_building->points.end());
			pc_out->points.insert(pc_out->points.end(), submap_outlier->points.begin(), submap_outlier->points.end());
		}
		else
		{
			pc_out->points.insert(pc_out->points.end(), submap_dynamic->points.begin(), submap_dynamic->points.end());
			pc_out->points.insert(pc_out->points.end(), submap_pole->points.begin(), submap_pole->points.end());
			pc_out->points.insert(pc_out->points.end(), submap_ground->points.begin(), submap_ground->points.end());
			pc_out->points.insert(pc_out->points.end(), submap_building->points.begin(), submap_building->points.end());
		}
        
    }

    // 不能正确被转换 未找到原因 但是在结构体外部使用内部操作可以完成转换
    void transform_feature(const PointTypePose *trans_mat) 
    {
        *submap_dynamic = *transformPointCloud(submap_dynamic, trans_mat);
        *submap_pole = *transformPointCloud(submap_pole, trans_mat);
        *submap_ground = *transformPointCloud(submap_ground, trans_mat);
        *submap_building = *transformPointCloud(submap_building, trans_mat);
        *submap_outlier = *transformPointCloud(submap_outlier, trans_mat);      
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

typedef boost::shared_ptr<submap_t> submap_Ptr;


struct localMap_t 
{
    int feature_point_num;

    bounds_t bound;				  //Bounding Box in geo-coordinate system
    centerpoint_t cp;			  //Center Point in geo-coordinate system

    pcl::PointCloud<PointXYZIL>::Ptr submap_dynamic;
    pcl::PointCloud<PointXYZIL>::Ptr submap_pole;
    pcl::PointCloud<PointXYZIL>::Ptr submap_ground;
    pcl::PointCloud<PointXYZIL>::Ptr submap_building;
    pcl::PointCloud<PointXYZIL>::Ptr submap_outlier;

    pcl::search::KdTree<PointXYZIL>::Ptr tree_dynamic;
    pcl::search::KdTree<PointXYZIL>::Ptr tree_pole;
    pcl::search::KdTree<PointXYZIL>::Ptr tree_ground;
    pcl::search::KdTree<PointXYZIL>::Ptr tree_building;
    pcl::search::KdTree<PointXYZIL>::Ptr tree_outlier;

	localMap_t()
	{
        submap_dynamic = boost::make_shared<pcl::PointCloud<PointXYZIL>>();
        submap_pole = boost::make_shared<pcl::PointCloud<PointXYZIL>>();
        submap_ground = boost::make_shared<pcl::PointCloud<PointXYZIL>>();
        submap_building = boost::make_shared<pcl::PointCloud<PointXYZIL>>();
        submap_outlier = boost::make_shared<pcl::PointCloud<PointXYZIL>>();
		
        tree_dynamic = boost::make_shared<pcl::search::KdTree<PointXYZIL>>();
		tree_pole = boost::make_shared<pcl::search::KdTree<PointXYZIL>>();
		tree_ground = boost::make_shared<pcl::search::KdTree<PointXYZIL>>();
		tree_building = boost::make_shared<pcl::search::KdTree<PointXYZIL>>();
		tree_outlier = boost::make_shared<pcl::search::KdTree<PointXYZIL>>();
		
		feature_point_num = 0;
	}

	void free()
	{
		submap_dynamic.reset(new pcl::PointCloud<PointXYZIL>());
		submap_pole.reset(new pcl::PointCloud<PointXYZIL>());
		submap_ground.reset(new pcl::PointCloud<PointXYZIL>());
		submap_building.reset(new pcl::PointCloud<PointXYZIL>());
		submap_outlier.reset(new pcl::PointCloud<PointXYZIL>());
        
        free_tree();
	}

	void free_tree()
	{
        tree_dynamic.reset(new pcl::search::KdTree<PointXYZIL>());
		tree_pole.reset(new pcl::search::KdTree<PointXYZIL>());
		tree_ground.reset(new pcl::search::KdTree<PointXYZIL>());
		tree_building.reset(new pcl::search::KdTree<PointXYZIL>());
		tree_outlier.reset(new pcl::search::KdTree<PointXYZIL>());
	}

	void append_feature(const keyframe_t &in_keyframe, bool using_down_cloud = false) 
    {
        if(using_down_cloud){
            submap_dynamic->points.insert(submap_dynamic->points.end(), in_keyframe.semantic_dynamic_down->points.begin(), in_keyframe.semantic_dynamic_down->points.end());
            submap_pole->points.insert(submap_pole->points.end(), in_keyframe.semantic_pole_down->points.begin(), in_keyframe.semantic_pole_down->points.end());
            submap_ground->points.insert(submap_ground->points.end(), in_keyframe.semantic_ground_down->points.begin(), in_keyframe.semantic_ground_down->points.end());
            submap_building->points.insert(submap_building->points.end(), in_keyframe.semantic_building_down->points.begin(), in_keyframe.semantic_building_down->points.end());
            submap_outlier->points.insert(submap_outlier->points.end(), in_keyframe.semantic_outlier_down->points.begin(), in_keyframe.semantic_outlier_down->points.end());
        }else{
            submap_dynamic->points.insert(submap_dynamic->points.end(), in_keyframe.semantic_dynamic->points.begin(), in_keyframe.semantic_dynamic->points.end());
            submap_pole->points.insert(submap_pole->points.end(), in_keyframe.semantic_pole->points.begin(), in_keyframe.semantic_pole->points.end());
            submap_ground->points.insert(submap_ground->points.end(), in_keyframe.semantic_ground->points.begin(), in_keyframe.semantic_ground->points.end());
            submap_building->points.insert(submap_building->points.end(), in_keyframe.semantic_building->points.begin(), in_keyframe.semantic_building->points.end());
            submap_outlier->points.insert(submap_outlier->points.end(), in_keyframe.semantic_outlier->points.begin(), in_keyframe.semantic_outlier->points.end());
        }
    }

  	void append_feature(const pcl::PointCloud<PointXYZIL>::Ptr &dynamic_in, 
						const pcl::PointCloud<PointXYZIL>::Ptr &pole_in,
						const pcl::PointCloud<PointXYZIL>::Ptr &ground_in,
						const pcl::PointCloud<PointXYZIL>::Ptr &building_in,
						const pcl::PointCloud<PointXYZIL>::Ptr &outlier_in) 
    {
		submap_dynamic->points.insert(submap_dynamic->points.end(), dynamic_in->points.begin(), dynamic_in->points.end());
		submap_pole->points.insert(submap_pole->points.end(), pole_in->points.begin(), pole_in->points.end());
		submap_ground->points.insert(submap_ground->points.end(), ground_in->points.begin(), ground_in->points.end());
		submap_building->points.insert(submap_building->points.end(), building_in->points.begin(), building_in->points.end());
		submap_outlier->points.insert(submap_outlier->points.end(), outlier_in->points.begin(), outlier_in->points.end());
    }

    void merge_feature_points(pcl::PointCloud<PointXYZIL>::Ptr &pc_out, bool merge_outlier = true)
    {	
		if(merge_outlier)
		{
			pc_out->points.insert(pc_out->points.end(), submap_dynamic->points.begin(), submap_dynamic->points.end());
			pc_out->points.insert(pc_out->points.end(), submap_pole->points.begin(), submap_pole->points.end());
			pc_out->points.insert(pc_out->points.end(), submap_ground->points.begin(), submap_ground->points.end());
			pc_out->points.insert(pc_out->points.end(), submap_building->points.begin(), submap_building->points.end());
			pc_out->points.insert(pc_out->points.end(), submap_outlier->points.begin(), submap_outlier->points.end());
		}
		else
		{
			pc_out->points.insert(pc_out->points.end(), submap_dynamic->points.begin(), submap_dynamic->points.end());
			pc_out->points.insert(pc_out->points.end(), submap_pole->points.begin(), submap_pole->points.end());
			pc_out->points.insert(pc_out->points.end(), submap_ground->points.begin(), submap_ground->points.end());
			pc_out->points.insert(pc_out->points.end(), submap_building->points.begin(), submap_building->points.end());
		}
        
    }



};
typedef boost::shared_ptr<localMap_t> localMap_Ptr;


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
        thisPose3D.x = last_target_cblock->relative_pose.x;
        thisPose3D.y = last_target_cblock->relative_pose.y;
        thisPose3D.z = last_target_cblock->relative_pose.z;
        thisPose3D.intensity = last_target_cblock->submap_id;

        local_map->submap_pose_3D_optimized = thisPose3D;   
        
        local_map->keyframe_id_in_submap.push_back(last_target_cblock->keyframe_id);
        local_map->keyframe_poses_6D->points.push_back(last_target_cblock->relative_pose);
        local_map->keyframe_poses_3D->points.push_back(thisPose3D);
        
        local_map->keyframe_poses_3D_map.insert(std::make_pair(last_target_cblock->keyframe_id, thisPose3D));
        local_map->keyframe_poses_6D_map.insert(std::make_pair(last_target_cblock->keyframe_id, last_target_cblock->relative_pose));

        local_map->append_feature(*last_target_cblock, true);  

        local_map->feature_point_num = local_map->submap_dynamic->points.size() + 
                                    local_map->submap_pole->points.size() + 
                                    local_map->submap_ground->points.size() + 
                                    local_map->submap_building->points.size() + 
                                    local_map->submap_outlier->points.size();

        typename pcl::PointCloud<PointT>::Ptr cloud_raw(new pcl::PointCloud<PointT>);
        
        //calculate bbx (local)
        local_map->merge_feature_points(cloud_raw);
        this->get_cloud_bbx_cpt(cloud_raw, local_map->local_bound, local_map->local_cp);

        //calculate bbx (global)
        Eigen::Affine3f tran_map = pclPointToAffine3f(local_map->submap_pose_6D_optimized);
        this->transform_bbx(local_map->local_bound, local_map->local_cp, local_map->bound, local_map->cp, tran_map);
        

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

        // Eigen::Affine3f tran_target_map; //from the last local_map to the target frame
        // tran_target_map = pclPointToAffine3f(local_map->submap_pose_6D_optimized).inverse() * pclPointToAffine3f(last_target_cblock->optimized_pose) ;
        
        // Eigen::Affine3f tran_target_map = pclPointToAffine3f(last_target_cblock->relative_pose);
        
        // PointTypePose thisPose;
        // pcl::getTranslationAndEulerAngles(tran_target_map, thisPose.x, thisPose.y, thisPose.z, 
        //                                               thisPose.roll, thisPose.pitch, thisPose.yaw);
        
        // ROS_WARN("submap_pose_6D_optimized : [%f, %f, %f, %f, %f, %f]",
        //         local_map->submap_pose_6D_optimized.roll, local_map->submap_pose_6D_optimized.pitch, local_map->submap_pose_6D_optimized.yaw,
        //         local_map->submap_pose_6D_optimized.x, local_map->submap_pose_6D_optimized.y, local_map->submap_pose_6D_optimized.z);
        
        // ROS_WARN("thisPose : relative_pose: [%f, %f, %f, %f, %f, %f]",
        //         thisPose.roll, thisPose.pitch, thisPose.yaw,
        //         thisPose.x, thisPose.y, thisPose.z);

        // ROS_WARN("last_target_cblock : relative_pose: [%f, %f, %f, %f, %f, %f]",
        //         last_target_cblock->relative_pose.roll, last_target_cblock->relative_pose.pitch, last_target_cblock->relative_pose.yaw,
        //         last_target_cblock->relative_pose.x, last_target_cblock->relative_pose.y, last_target_cblock->relative_pose.z);
        
		typename pcl::PointCloud<PointT>::Ptr  semantic_dynamic(new pcl::PointCloud<PointT>);
        typename pcl::PointCloud<PointT>::Ptr  semantic_pole(new pcl::PointCloud<PointT>);
        typename pcl::PointCloud<PointT>::Ptr  semantic_ground(new pcl::PointCloud<PointT>);
        typename pcl::PointCloud<PointT>::Ptr  semantic_building(new pcl::PointCloud<PointT>);
        typename pcl::PointCloud<PointT>::Ptr  semantic_outlier(new pcl::PointCloud<PointT>);

		// 不能正确被转换 未找到原因 但是在下面操作可以完成转换
        // last_target_cblock->transform_feature(&last_target_cblock->relative_pose, false, true);
        *semantic_dynamic = *transformPointCloud(last_target_cblock->semantic_dynamic_down, &last_target_cblock->relative_pose);
        *semantic_pole = *transformPointCloud(last_target_cblock->semantic_pole_down, &last_target_cblock->relative_pose);
        *semantic_ground = *transformPointCloud(last_target_cblock->semantic_ground_down, &last_target_cblock->relative_pose);
        *semantic_building = *transformPointCloud(last_target_cblock->semantic_building_down, &last_target_cblock->relative_pose);
        *semantic_outlier = *transformPointCloud(last_target_cblock->semantic_outlier_down, &last_target_cblock->relative_pose);

        dynamic_dist_thre_max = std::max(dynamic_dist_thre_max, (float)(dynamic_dist_thre_min + 0.1));
        std::cout << "Map based filtering range(m): (0, " << near_dist_thre << "] U [" << dynamic_dist_thre_min << "," << dynamic_dist_thre_max << "]" << std::endl;

        if (map_based_dynamic_removal_on && local_map->feature_point_num > max_num_pts / 5)
        {
            local_map->tree_dynamic->setInputCloud(local_map->submap_dynamic);
            // local_map->tree_pole->setInputCloud(local_map->submap_pole);
            // local_map->tree_ground->setInputCloud(local_map->submap_ground);
            // local_map->tree_building->setInputCloud(local_map->submap_building);
            // local_map->tree_outlier->setInputCloud(local_map->submap_outlier);

            map_scan_feature_pts_distance_removal(semantic_dynamic, local_map->tree_dynamic, 
                                                  dynamic_removal_center_radius, dynamic_dist_thre_min, dynamic_dist_thre_max, near_dist_thre);
            // map_scan_feature_pts_distance_removal(semantic_pole, local_map->tree_pole, 
            //                                       dynamic_removal_center_radius, dynamic_dist_thre_min, dynamic_dist_thre_max, near_dist_thre);
			// map_scan_feature_pts_distance_removal(semantic_ground, local_map->tree_ground, 
            //                                       dynamic_removal_center_radius, dynamic_dist_thre_min, dynamic_dist_thre_max, near_dist_thre);
			// map_scan_feature_pts_distance_removal(semantic_building, local_map->tree_building, 
            //                                       dynamic_removal_center_radius, dynamic_dist_thre_min, dynamic_dist_thre_max, near_dist_thre);
			// map_scan_feature_pts_distance_removal(semantic_outlier, local_map->tree_outlier, 
            //                                       dynamic_removal_center_radius, dynamic_dist_thre_min, dynamic_dist_thre_max, near_dist_thre);

            // #pragma omp parallel sections
            // {
            //     #pragma omp section
            //     {
            //         //dynamic + close points
            //         map_scan_feature_pts_distance_removal(semantic_dynamic, local_map->tree_dynamic, center_radius, dynamic_dist_thre_min, dynamic_dist_thre_max, near_dist_thre);
            //     }
            //     #pragma omp section
            //     {
            //         //dynamic + close points
            //         map_scan_feature_pts_distance_removal(semantic_pole, local_map->tree_pole, center_radius, dynamic_dist_thre_min, dynamic_dist_thre_max, near_dist_thre);
            //         map_scan_feature_pts_distance_removal(semantic_ground, local_map->tree_ground, center_radius, dynamic_dist_thre_min, dynamic_dist_thre_max, near_dist_thre);
            //         map_scan_feature_pts_distance_removal(semantic_building, local_map->tree_building, center_radius, dynamic_dist_thre_min, dynamic_dist_thre_max, near_dist_thre);
            //     }        
            //     #pragma omp section
            //     {
            //         //dynamic + close points
            //         map_scan_feature_pts_distance_removal(semantic_outlier, local_map->tree_outlier, center_radius, dynamic_dist_thre_min, dynamic_dist_thre_max, near_dist_thre);
            //     }
            // }

            // std::cout << "Feature point number of last frame after dynamic removal (Ori | Removal): " << std::endl 
            //           <<  "Dynamic: [" << last_target_cblock->semantic_dynamic_down->points.size() << " | " << semantic_dynamic->points.size() << "]." << std::endl 
            //           <<  "Pole: [" << last_target_cblock->semantic_pole_down->points.size() << " | " << semantic_pole->points.size() << "]." << std::endl 
            //           <<  "Ground: [" << last_target_cblock->semantic_ground_down->points.size() << " | " << semantic_ground->points.size() << "]." << std::endl 
            //           <<  "Building: [" << last_target_cblock->semantic_building_down->points.size() << " | " << semantic_building->points.size() << "]." << std::endl 
            //           <<  "Outlier: [" << last_target_cblock->semantic_outlier_down->points.size() << " | " << semantic_outlier->points.size() << "]." << std::endl 
            //           << std::endl;
        }  
        // local_map->append_feature(*last_target_cblock, true);
        local_map->append_feature(semantic_dynamic, semantic_pole, semantic_ground, semantic_building, semantic_outlier);
        
        local_map->submap_size++;
        local_map->keyframe_id_in_submap.push_back(last_target_cblock->keyframe_id);

        PointType thisPose3D;
        thisPose3D.x = last_target_cblock->relative_pose.x;
        thisPose3D.y = last_target_cblock->relative_pose.y;
        thisPose3D.z = last_target_cblock->relative_pose.z;
        thisPose3D.intensity = last_target_cblock->keyframe_id;
        local_map->keyframe_poses_3D->points.push_back(thisPose3D);

        local_map->keyframe_poses_6D->points.push_back(last_target_cblock->relative_pose);

        local_map->keyframe_poses_3D_map.insert(std::make_pair(last_target_cblock->keyframe_id, thisPose3D));
        local_map->keyframe_poses_6D_map.insert(std::make_pair(last_target_cblock->keyframe_id, last_target_cblock->relative_pose));

        local_map->feature_point_num = local_map->submap_dynamic->points.size() + 
                                       local_map->submap_pole->points.size() + 
                                       local_map->submap_ground->points.size() + 
                                       local_map->submap_building->points.size() + 
                                       local_map->submap_outlier->points.size();

        typename pcl::PointCloud<PointT>::Ptr cloud_raw(new pcl::PointCloud<PointT>);
        
        //calculate bbx (local)
        local_map->merge_feature_points(cloud_raw);
        this->get_cloud_bbx_cpt(cloud_raw, local_map->local_bound, local_map->local_cp);


        //calculate bbx (global)
        Eigen::Affine3f tran_map = pclPointToAffine3f(local_map->submap_pose_6D_optimized);
        this->transform_bbx(local_map->local_bound, local_map->local_cp, local_map->bound, local_map->cp, tran_map);

        local_map->free_tree();
        
        std::chrono::steady_clock::time_point toc = std::chrono::steady_clock::now();
        std::chrono::duration<double> time_update_local_map = std::chrono::duration_cast<std::chrono::duration<double>>(toc - tic);
        std::cout << "Update submap ([" << local_map->feature_point_num << "] points at present) done in [" << time_update_local_map.count() * 1000.0 << "] ms.\n";
        return true;
    }




    bool insert_local_map(
            localMap_Ptr &local_map, keyframe_Ptr &last_target_cblock,
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

        typename pcl::PointCloud<PointT>::Ptr  semantic_dynamic(new pcl::PointCloud<PointT>);
        typename pcl::PointCloud<PointT>::Ptr  semantic_pole(new pcl::PointCloud<PointT>);
        typename pcl::PointCloud<PointT>::Ptr  semantic_ground(new pcl::PointCloud<PointT>);
        typename pcl::PointCloud<PointT>::Ptr  semantic_building(new pcl::PointCloud<PointT>);
        typename pcl::PointCloud<PointT>::Ptr  semantic_outlier(new pcl::PointCloud<PointT>);

        // *semantic_dynamic = *transformPointCloud(last_target_cblock->semantic_dynamic_down, &last_target_cblock->optimized_pose);
        // *semantic_pole = *transformPointCloud(last_target_cblock->semantic_pole_down, &last_target_cblock->optimized_pose);
        // *semantic_ground = *transformPointCloud(last_target_cblock->semantic_ground_down, &last_target_cblock->optimized_pose);
        // *semantic_building = *transformPointCloud(last_target_cblock->semantic_building_down, &last_target_cblock->optimized_pose);
        // // *semantic_outlier = *transformPointCloud(last_target_cblock->semantic_outlier_down, &last_target_cblock->optimized_pose);
        
		*semantic_dynamic = *transformPointCloud(last_target_cblock->semantic_dynamic, &last_target_cblock->optimized_pose);
        *semantic_pole = *transformPointCloud(last_target_cblock->semantic_pole, &last_target_cblock->optimized_pose);
        *semantic_ground = *transformPointCloud(last_target_cblock->semantic_ground, &last_target_cblock->optimized_pose);
        *semantic_building = *transformPointCloud(last_target_cblock->semantic_building, &last_target_cblock->optimized_pose);
        // *semantic_outlier = *transformPointCloud(last_target_cblock->semantic_outlier, &last_target_cblock->optimized_pose);
       
		dynamic_dist_thre_max = std::max(dynamic_dist_thre_max, (float)(dynamic_dist_thre_min + 0.1));
        if (map_based_dynamic_removal_on && local_map->feature_point_num > max_num_pts / 5)
        {
            local_map->tree_dynamic->setInputCloud(local_map->submap_dynamic);
            // local_map->tree_pole->setInputCloud(local_map->submap_pole);
            // local_map->tree_ground->setInputCloud(local_map->submap_ground);
            // local_map->tree_building->setInputCloud(local_map->submap_building);
            // local_map->tree_outlier->setInputCloud(local_map->submap_outlier);

            map_scan_feature_pts_distance_removal(semantic_dynamic, local_map->tree_dynamic, 
                                                  dynamic_removal_center_radius, dynamic_dist_thre_min, dynamic_dist_thre_max, near_dist_thre);
			// map_scan_feature_pts_distance_removal(semantic_pole, local_map->tree_pole, 
            //                                       dynamic_removal_center_radius, dynamic_dist_thre_min, dynamic_dist_thre_max, near_dist_thre);
			// map_scan_feature_pts_distance_removal(semantic_ground, local_map->tree_ground, 
            //                                       dynamic_removal_center_radius, dynamic_dist_thre_min, dynamic_dist_thre_max, near_dist_thre);
			// map_scan_feature_pts_distance_removal(semantic_building, local_map->tree_building, 
            //                                       dynamic_removal_center_radius, dynamic_dist_thre_min, dynamic_dist_thre_max, near_dist_thre);
			// map_scan_feature_pts_distance_removal(semantic_outlier, local_map->tree_outlier, 
            //                                       dynamic_removal_center_radius, dynamic_dist_thre_min, dynamic_dist_thre_max, near_dist_thre);

        }  
        local_map->append_feature(semantic_dynamic, semantic_pole, semantic_ground, semantic_building, semantic_outlier);

        local_map->feature_point_num = local_map->submap_dynamic->points.size() + 
                                       local_map->submap_pole->points.size() + 
                                       local_map->submap_ground->points.size() + 
                                       local_map->submap_building->points.size() + 
                                       local_map->submap_outlier->points.size();

        typename pcl::PointCloud<PointT>::Ptr cloud_raw(new pcl::PointCloud<PointT>);
        
        //calculate bbx 
        local_map->merge_feature_points(cloud_raw);
        this->get_cloud_bbx_cpt(cloud_raw, local_map->bound, local_map->cp);


        local_map->free_tree();
        
        std::chrono::steady_clock::time_point toc = std::chrono::steady_clock::now();
        std::chrono::duration<double> time_update_local_map = std::chrono::duration_cast<std::chrono::duration<double>>(toc - tic);
        std::cout << "Update localMap ([" << local_map->feature_point_num << "] points at present) done in [" << time_update_local_map.count() * 1000.0 << "] ms.\n";
        return true;
    }



    //keep the points meet dist ~ (near_dist_thre, dynamic_dist_thre_min) U (dynamic_dist_thre_max, +inf)
    //filter the points meet dist ~ (0, near_dist_thre] U [dynamic_dist_thre_min, dynamic_dist_thre_max]
    bool map_scan_feature_pts_distance_removal(
            typename pcl::PointCloud<PointT>::Ptr feature_pts, const typename pcl::search::KdTree<PointT>::Ptr map_kdtree, 
            float center_radius, float dynamic_dist_thre_min = FLT_MAX, float dynamic_dist_thre_max = FLT_MAX, float near_dist_thre = 0.0)
    {
        // std::cout << "map_scan_feature_pts_distance_removal start!" << std::endl;

        if (feature_pts->points.size() <= 10)
            return false;

        typename pcl::PointCloud<PointT>::Ptr  feature_pts_temp(new pcl::PointCloud<PointT>);
        
        std::vector<float> distances_square;
        std::vector<int> search_indices;
        // #pragma omp parallel for //Multi-thread
        for (int i = 0; i < feature_pts->points.size(); i++)
        {
            if (feature_pts->points[i].x * feature_pts->points[i].x + feature_pts->points[i].y * feature_pts->points[i].y > center_radius * center_radius)
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

        // std::cout << "map_scan_feature_pts_distance_removal end!" << std::endl;

        return true;  
    }            


    bool judge_new_submap(
            float &accu_tran, float &accu_rot, int &accu_frame,
            float max_accu_tran = 30.0, float max_accu_rot = 90.0, int max_accu_frame = 150)
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