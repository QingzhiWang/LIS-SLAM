//This code partly draws on ISCLOAM
// Author of  EPSC-LOAM : QZ Wang  
// Email wangqingzhi27@outlook.com


#ifndef _EPSC_GENERATION_CLASS_H_
#define _EPSC_GENERATION_CLASS_H_

#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <cmath>


//PCL
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>

//opencv
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

//opencv
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>

//ros
#include <ros/ros.h>

//pcl
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/crop_box.h>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Geometry>


//IF TRAVELLED DISTANCE IS LESS THAN THIS VALUE, SKIP FOR PLACE RECOGNTION
#define SKIP_NEIBOUR_DISTANCE 20.0    //20.0
//how much error will odom generate per frame 
#define INFLATION_COVARIANCE 0.03  //0.03

//define threshold for loop closure detection
#define GEOMETRY_THRESHOLD 0.55 //0.67  0.57(--)
#define INTENSITY_THRESHOLD 0.7900  //0.91

typedef cv::Mat EPSCDescriptor; 


class EPSCGenerationClass
{
    public:
        EPSCGenerationClass();
        void init_param(int rings_in, int sectors_in, double max_dis_in);

        EPSCDescriptor getLastISCMONO(void);
        EPSCDescriptor getLastISCRGB(void);

        EPSCDescriptor getLastEPSCMONO(void);
        EPSCDescriptor getLastEPSCRGB(void);

        EPSCDescriptor getLastSCRGB(void);

        void loopDetection(const pcl::PointCloud<pcl::PointXYZI>::Ptr& corner_pc, const pcl::PointCloud<pcl::PointXYZI>::Ptr& surf_pc,const pcl::PointCloud<pcl::PointXYZI>::Ptr& raw_pc,Eigen::Isometry3d& odom);
        
        int current_frame_id;
        std::vector<int> matched_frame_id;

        std::vector<int> isc_matched_frame_id;
        std::vector<int> sc_matched_frame_id;
        std::vector<int> pos_matched_frame_id;


    private:
        int rings = 20;
        int sectors = 90;
        double ring_step=0.0;
        double sector_step=0.0;
        double max_dis = 60; 

        std::vector<cv::Vec3b> color_projection;

        
        pcl::PointCloud<pcl::PointXYZI>::Ptr current_point_cloud;
        pcl::PointCloud<pcl::PointXYZI>::Ptr test_pc;

        std::vector<Eigen::Vector3d> pos_arr;
        std::vector<double> travel_distance_arr;
        std::vector<EPSCDescriptor> isc_arr;


        pcl::PointCloud<pcl::PointXYZI>::Ptr corner_point_cloud;
        pcl::PointCloud<pcl::PointXYZI>::Ptr surf_point_cloud;
        std::vector<EPSCDescriptor> epsc_arr;

        std::vector<EPSCDescriptor> sc_arr;

    private:
        void init_color(void);
        void print_param(void);
        bool is_loop_pair(EPSCDescriptor& desc1, EPSCDescriptor& desc2, double& geo_score, double& inten_score);
        EPSCDescriptor calculate_isc(const pcl::PointCloud<pcl::PointXYZI>::Ptr filtered_pointcloud);
        double calculate_geometry_dis(const EPSCDescriptor& desc1, const EPSCDescriptor& desc2, int& angle);
        double calculate_intensity_dis(const EPSCDescriptor& desc1, const EPSCDescriptor& desc2, int& angle);
        void ground_filter(const pcl::PointCloud<pcl::PointXYZI>::Ptr& pc_in, pcl::PointCloud<pcl::PointXYZI>::Ptr& pc_out);

        EPSCDescriptor calculate_epsc(const pcl::PointCloud<pcl::PointXYZI>::Ptr filtered_corner_pointcloud,const pcl::PointCloud<pcl::PointXYZI>::Ptr filtered_surf_pointcloud);
        EPSCDescriptor calculate_sc(const pcl::PointCloud<pcl::PointXYZI>::Ptr filtered_pointcloud);
};




#endif // _ISC_GENERATION_CLASS_H_

