// This code partly draws on ISCLOAM SSC
// Author of  EPSC-LOAM : QZ Wang
// Email wangqingzhi27@outlook.com

#ifndef _EPSC_GENERATION_H_
#define _EPSC_GENERATION_H_

#include "utility.h"
#include "common.h"

// IF TRAVELLED DISTANCE IS LESS THAN THIS VALUE, SKIP FOR PLACE RECOGNTION
#define SKIP_NEIBOUR_DISTANCE 20.0  // 20.0
// how much error will odom generate per frame
#define INFLATION_COVARIANCE 0.015  // 0.03

// define threshold for loop closure detection
// #define GEOMETRY_THRESHOLD 0.55     // 0.67  0.57(--)
// #define INTENSITY_THRESHOLD 0.7900  // 0.91
#define DISTANCE_THRESHOLD 0.7900  
#define LABEL_THRESHOLD 0.7900  

#define LIDAR_HEIGHT 5.0


class EPSCGeneration : public SemanticLabelParam {
private:  
	std::vector<int> order_vec = {0,  0,  0,  0,  0,  0,  0,  0, 0,  10,
									11, 12, 13, 15, 16, 14, 17, 9, 18, 19};
	bool UsingISCFlag = false;
	bool UsingSCFlag = false;
	bool UsingPoseFlag = false;
	bool UsingSEPSCFlag = true;
	bool UsingEPSCFlag = true;
	bool UsingSSCFlag = false;

	double max_dis = 70;
	double min_dis = 3;

	int rings = 20;
	int sectors = 90; //180
	double ring_step = (max_dis - min_dis) / rings;
	double sector_step = 2 * M_PI / sectors;

	int sectors_range = 360;
	//   bool rotate = false;
	//   bool occlusion = false;
	//   bool remap = true;

	std::vector<cv::Vec3b> color_projection;

	std::vector<cv::Mat> ProjectArr;

	std::vector<cv::Mat> ISCArr;
	std::vector<cv::Mat> SCArr;
	std::vector<cv::Mat> EPSCArr;
	std::vector<cv::Mat> SEPSCArr;
	std::vector<cv::Mat> SSCArr;
	std::vector<cv::Mat> myDescriptorArr;

	std::vector<double> travelDistanceArr;
	std::vector<Eigen::Vector3d> posArr;


	bool show = false;
	std::shared_ptr<pcl::visualization::CloudViewer> viewer;

	void init_color(void);
	void print_param(void);
	void init_param();

	template <typename PointT>
	void groundFilter(const typename pcl::PointCloud<PointT>::Ptr &pc_in,
									typename pcl::PointCloud<PointT>::Ptr &pc_out) {
		pcl::PassThrough<PointT> pass;
		pass.setInputCloud(pc_in);
		pass.setFilterFieldName("z");
		pass.setFilterLimits(-5.0, 30.0);
		pass.filter(*pc_out);
	}

	pcl::PointCloud<pcl::PointXYZRGB>::Ptr getColorCloud(pcl::PointCloud<PointXYZIL>::Ptr& cloud_in);
	cv::Mat getColorImage(cv::Mat& desc);

	cv::Mat project(const pcl::PointCloud<PointXYZIL>::Ptr filtered_pointcloud);

	void globalICP(cv::Mat& isc_dis1, cv::Mat& isc_dis2, double& angle,float& diff_x, float& diff_y);
	Eigen::Affine3f globalICP(cv::Mat& ssc_dis1, cv::Mat& ssc_dis2);

	cv::Mat calculateSC(const pcl::PointCloud<PointXYZIL>::Ptr filtered_pointcloud);
	cv::Mat calculateISC(const pcl::PointCloud<PointXYZIL>::Ptr filtered_pointcloud);
	cv::Mat calculateEPSC(const pcl::PointCloud<pcl::PointXYZI>::Ptr filtered_corner_pointcloud,
							const pcl::PointCloud<pcl::PointXYZI>::Ptr filtered_surf_pointcloud);
	cv::Mat calculateSEPSC(const pcl::PointCloud<PointXYZIL>::Ptr filtered_pointcloud);
	cv::Mat calculateSSC(const pcl::PointCloud<PointXYZIL>::Ptr filtered_pointcloud);

	double calculateLabelSim(cv::Mat& desc1, cv::Mat& desc2);
	double calculateDistance(const cv::Mat& desc1, const cv::Mat& desc2, double& angle);


	double getScore(pcl::PointCloud<PointXYZIL>::Ptr cloud1,
					pcl::PointCloud<PointXYZIL>::Ptr cloud2,
					Eigen::Matrix4f& trans);
	double getScore(pcl::PointCloud<PointXYZIL>::Ptr cloud1,
					pcl::PointCloud<PointXYZIL>::Ptr cloud2, double& angle,
					float& diff_x, float& diff_y);



	double getScore(pcl::PointCloud<pcl::PointXYZL>::Ptr cloud1,
					pcl::PointCloud<pcl::PointXYZL>::Ptr cloud2, double& angle,
					float& diff_x, float& diff_y);

	double getScore(pcl::PointCloud<pcl::PointXYZL>::Ptr cloud1,
					pcl::PointCloud<pcl::PointXYZL>::Ptr cloud2, 
					Eigen::Matrix4f& trans);


public:
	int current_frame_id;
	std::vector<int> matched_frame_id;
	std::vector<Eigen::Affine3f> matched_frame_transform;

	EPSCGeneration() { init_param(); };

	cv::Mat getLastSEPSCRGB(void);
	cv::Mat getLastSCRGB(void);
	cv::Mat getLastISCRGB(void);
	cv::Mat getLastEPSCRGB(void);
	cv::Mat getLastSSCRGB(void);

	cv::Mat getLastSEPSCMONO(void) {
		if (UsingSEPSCFlag)  return SEPSCArr.back();
	}
	cv::Mat getLastEPSCMONO(void) {
		if (UsingEPSCFlag) return EPSCArr.back();
	}
	cv::Mat getLastSCMONO(void) {
		if (UsingSCFlag)  return SCArr.back();
	}
	cv::Mat getLastISCMONO(void) {
		if (UsingISCFlag)  return ISCArr.back();
	}
	cv::Mat getLastSSCMONO(void) {
		if (UsingSSCFlag)  return SSCArr.back();
	}


	void loopDetection(const pcl::PointCloud<pcl::PointXYZI>::Ptr& corner_pc,
						const pcl::PointCloud<pcl::PointXYZI>::Ptr& surf_pc,
						const pcl::PointCloud<PointXYZIL>::Ptr& semantic_pc,
						Eigen::Affine3f& odom);
};

#endif  // _ISC_GENERATION_H_