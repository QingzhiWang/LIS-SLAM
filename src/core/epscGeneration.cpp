
#include "epscGeneration.h"

// #define INTEGER_INTENSITY

void EPSCGeneration::init_param() 
{
    print_param();
    init_color();

    if (show) viewer.reset(new pcl::visualization::CloudViewer("viewer"));
}

void EPSCGeneration::init_color(void) 
{
	for (int i = 0; i < 1; i++) {  // RGB format
		color_projection.push_back(cv::Vec3b(0, i * 16, 255));
	}
	for (int i = 0; i < 15; i++) {  // RGB format
		color_projection.push_back(cv::Vec3b(0, i * 16, 255));
	}
	for (int i = 0; i < 16; i++) {  // RGB format
		color_projection.push_back(cv::Vec3b(0, 255, 255 - i * 16));
	}
	for (int i = 0; i < 32; i++) {  // RGB format
		color_projection.push_back(cv::Vec3b(i * 32, 255, 0));
	}
	for (int i = 0; i < 16; i++) {  // RGB format
		color_projection.push_back(cv::Vec3b(255, 255 - i * 16, 0));
	}
	for (int i = 0; i < 64; i++) {  // RGB format
		color_projection.push_back(cv::Vec3b(i * 4, 255, 0));
	}
	for (int i = 0; i < 64; i++) {  // RGB format
		color_projection.push_back(cv::Vec3b(255, 255 - i * 4, 0));
	}
	for (int i = 0; i < 64; i++) {  // RGB format
		color_projection.push_back(cv::Vec3b(255, i * 4, i * 4));
	}
}

void EPSCGeneration::print_param() 
{
	std::cout << "The EPSC parameters are:" << std::endl;
	std::cout << "number of rings:\t" << rings << std::endl;
	std::cout << "number of sectors:\t" << sectors << std::endl;
	std::cout << "maximum distance:\t" << max_dis << std::endl;
}




cv::Mat EPSCGeneration::getColorImage(cv::Mat &desc) 
{
	cv::Mat out = cv::Mat::zeros(desc.size(), CV_8UC3);
	for (int i = 0; i < desc.rows; ++i) {
		for (int j = 0; j < desc.cols; ++j) {
			out.at<cv::Vec3b>(i, j)[0] = std::get<2>(Argmax2RGB[(int)desc.at<uchar>(i, j)]);
			out.at<cv::Vec3b>(i, j)[1] = std::get<1>(Argmax2RGB[(int)desc.at<uchar>(i, j)]);
			out.at<cv::Vec3b>(i, j)[2] = std::get<0>(Argmax2RGB[(int)desc.at<uchar>(i, j)]);
		}
	}
	return out;
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr EPSCGeneration::getColorCloud(pcl::PointCloud<PointXYZIL>::Ptr &cloud_in) 
{
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr outcloud(new pcl::PointCloud<pcl::PointXYZRGB>);
	outcloud->points.resize(cloud_in->points.size());
	for (size_t i = 0; i < outcloud->points.size(); i++) 
	{
		outcloud->points[i].x = cloud_in->points[i].x;
		outcloud->points[i].y = cloud_in->points[i].y;
		outcloud->points[i].z = cloud_in->points[i].z;
		outcloud->points[i].r = std::get<0>(Argmax2RGB[cloud_in->points[i].label]);
		outcloud->points[i].g = std::get<1>(Argmax2RGB[cloud_in->points[i].label]);
		outcloud->points[i].b = std::get<2>(Argmax2RGB[cloud_in->points[i].label]);
	}
	outcloud->height = 1;
	outcloud->width = outcloud->points.size();
	return outcloud;
}

cv::Mat EPSCGeneration::project(pcl::PointCloud<PointXYZIL>::Ptr filtered_pointcloud) 
{
	float sectors_range = 360.;
	float step = 2. * M_PI / sectors_range;
	cv::Mat ssc_dis = cv::Mat::zeros(cv::Size(sectors_range, 1), CV_32FC4);
	for (uint i = 0; i < filtered_pointcloud->points.size(); i++) 
	{
		auto label = filtered_pointcloud->points[i].label;
		if (label == 13 || label == 14 || label == 16 || label == 18 || label == 19) 
		{
			float distance = std::sqrt(
				filtered_pointcloud->points[i].x * filtered_pointcloud->points[i].x +
				filtered_pointcloud->points[i].y * filtered_pointcloud->points[i].y);
			if (distance < 1e-2) 
				continue;
			
			float angle = M_PI + std::atan2(filtered_pointcloud->points[i].y,
											filtered_pointcloud->points[i].x);
									
			int sector_id = std::floor(angle / step);
			if (sector_id >= sectors_range || sector_id < 0) continue;

			// if (order_vec[label] > order_vec[ssc_dis.at<cv::Vec4f>(0, sector_id)[3]]) {
			// 	ssc_dis.at<cv::Vec4f>(0, sector_id)[0] = distance;
			// 	// ssc_dis.at<cv::Vec4f>(0, sector_id)[0] = 1;
			// 	ssc_dis.at<cv::Vec4f>(0, sector_id)[1] = filtered_pointcloud->points[i].x;
			// 	ssc_dis.at<cv::Vec4f>(0, sector_id)[2] = filtered_pointcloud->points[i].y;
			// 	ssc_dis.at<cv::Vec4f>(0, sector_id)[3] = label;
			// }
			ssc_dis.at<cv::Vec4f>(0, sector_id)[0]++;
			ssc_dis.at<cv::Vec4f>(0, sector_id)[1] = filtered_pointcloud->points[i].x;
			ssc_dis.at<cv::Vec4f>(0, sector_id)[2] = filtered_pointcloud->points[i].y;
			ssc_dis.at<cv::Vec4f>(0, sector_id)[3] = label;
		}
	}
	return ssc_dis;
}

void EPSCGeneration::globalICP(cv::Mat &ssc_dis1, cv::Mat &ssc_dis2,
                               double &angle, float &diff_x, float &diff_y) 
{
	double similarity = 100000;
	int sectors = ssc_dis1.cols;
	for (int i = 0; i < sectors; ++i) 
	{
		float dis_count = 0;
		for (int j = 0; j < sectors; ++j) 
		{
			int new_col = j + i >= sectors ? j + i - sectors : j + i;
			cv::Vec4f vec1 = ssc_dis1.at<cv::Vec4f>(0, j);
			cv::Vec4f vec2 = ssc_dis2.at<cv::Vec4f>(0, new_col);
			// if(vec1[3]==vec2[3]){
			dis_count += fabs(vec1[0] - vec2[0]);
			// }
		}
		if (dis_count < similarity) 
		{
			similarity = dis_count;
			angle = i;
		}
	}
	int angle_o = angle;
	angle = M_PI * (360. - angle * 360. / sectors) / 180.;
	auto cs = cos(angle);
	auto sn = sin(angle);
	auto temp_dis1 = ssc_dis1.clone();
	auto temp_dis2 = ssc_dis2.clone();
	for (int i = 0; i < sectors; ++i) 
	{
		temp_dis2.at<cv::Vec4f>(0, i)[1] = ssc_dis2.at<cv::Vec4f>(0, i)[1] * cs -
										ssc_dis2.at<cv::Vec4f>(0, i)[2] * sn;
		temp_dis2.at<cv::Vec4f>(0, i)[2] = ssc_dis2.at<cv::Vec4f>(0, i)[1] * sn +
										ssc_dis2.at<cv::Vec4f>(0, i)[2] * cs;
	}

	for (int i = 0; i < 100; ++i) 
	{
		float dx = 0, dy = 0;
		int diff_count = 1;
		for (int j = 0; j < sectors; ++j) 
		{
			cv::Vec4f vec1 = temp_dis1.at<cv::Vec4f>(0, j);
			if (vec1[0] <= 0) {
				continue;
			}
			int min_id = -1;
			float min_dis = 1000000.;
			for (int k = j + angle_o - 10; k < j + angle_o + 10; ++k) 
			{
				cv::Vec4f vec_temp;
				int temp_id = k;
				if (k < 0) {
					temp_id = k + sectors;
				} else if (k >= sectors) {
					temp_id = k - sectors;
				}
				vec_temp = temp_dis2.at<cv::Vec4f>(0, temp_id);
				if (vec_temp[0] <= 0) {
					continue;
				}
				float temp_dis = (vec1[1] - vec_temp[1]) * (vec1[1] - vec_temp[1]) +
								(vec1[2] - vec_temp[2]) * (vec1[2] - vec_temp[2]);
				
				if (temp_dis < min_dis) {
					min_dis = temp_dis;
					min_id = temp_id;
				}
			}
			if (min_id < 0) {
				continue;
			}
			cv::Vec4f vec2 = temp_dis2.at<cv::Vec4f>(0, min_id);
			if (fabs(vec1[1] - vec2[1]) < 3 && fabs(vec1[2] - vec2[2]) < 3) 
			{
				dx += vec1[1] - vec2[1];
				dy += vec1[2] - vec2[2];
				diff_count++;
			}
		}
		dx = 1. * dx / diff_count;
		dy = 1. * dy / diff_count;

		pcl::PointCloud<pcl::PointXYZRGB>::Ptr temp_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

		for (int j = 0; j < sectors; ++j) 
		{
			if (temp_dis2.at<cv::Vec4f>(0, j)[0] != 0) 
			{
				temp_dis2.at<cv::Vec4f>(0, j)[1] += dx;
				temp_dis2.at<cv::Vec4f>(0, j)[2] += dy;
				if (show) 
				{
					pcl::PointXYZRGB p;
					p.x = temp_dis2.at<cv::Vec4f>(0, j)[1];
					p.y = temp_dis2.at<cv::Vec4f>(0, j)[2];
					p.z = 0;
					p.r = std::get<0>(Argmax2RGB[(int)temp_dis2.at<cv::Vec4f>(0, j)[3]]);
					p.g = std::get<1>(Argmax2RGB[(int)temp_dis2.at<cv::Vec4f>(0, j)[3]]);
					p.b = std::get<2>(Argmax2RGB[(int)temp_dis2.at<cv::Vec4f>(0, j)[3]]);
					temp_cloud->points.emplace_back(p);
				}
			}

			if (show && temp_dis1.at<cv::Vec4f>(0, j)[0] != 0) 
			{
				pcl::PointXYZRGB p;
				p.x = temp_dis1.at<cv::Vec4f>(0, j)[1];
				p.y = temp_dis1.at<cv::Vec4f>(0, j)[2];
				p.z = 0;
				p.r = std::get<0>(Argmax2RGB[(int)temp_dis1.at<cv::Vec4f>(0, j)[3]]);
				p.g = std::get<1>(Argmax2RGB[(int)temp_dis1.at<cv::Vec4f>(0, j)[3]]);
				p.b = std::get<2>(Argmax2RGB[(int)temp_dis1.at<cv::Vec4f>(0, j)[3]]);
				temp_cloud->points.emplace_back(p);
			}
		}
		if (show) 
		{
			temp_cloud->height = 1;
			temp_cloud->width = temp_cloud->points.size();
			viewer->showCloud(temp_cloud);
			usleep(1000000);
		}

		diff_x += dx;
		diff_y += dy;
		if (show) {
			std::cout << i << " diff " << diff_x << " " << diff_y << " " << dx << " " << dy << std::endl;
		}
		if (fabs(dx) < 1e-5 && fabs(dy) < 1e-5) {
			break;
		}
	}
}

Eigen::Affine3f EPSCGeneration::globalICP(cv::Mat &ssc_dis1, cv::Mat &ssc_dis2, float &yaw_diff) 
{
	float sectors_range = 360.;
	float step = 2. * M_PI / sectors_range;
	double similarity = 100000;
	// ROS_INFO("globalICP: yaw_diff: %f", yaw_diff);

	float angle = yaw_diff;
	if (angle >= 2. * M_PI) angle = angle - 2. * M_PI;
	if (angle < 0) angle = angle + 2. * M_PI;
	// ROS_INFO("globalICP: init angle: %f", angle);

	int tmp_id = std::floor(angle / step);
	int sectors = ssc_dis1.cols;

	// for (int i = tmp_id - sectors / 2; i < tmp_id + sectors / 2; ++i) 
	for (int i = tmp_id - 30; i < tmp_id + 30; ++i) 
	{
		float dis_count = 0;
		for (int j = 0; j < sectors; ++j) 
		{
			// int new_col = j + i >= sectors ? j + i - sectors : j + i;
			int new_col = j + i;
			if (new_col >= sectors) new_col = new_col - sectors;
			if (new_col < 0) new_col = new_col + sectors;
			cv::Vec4f vec1 = ssc_dis1.at<cv::Vec4f>(0, j);
			cv::Vec4f vec2 = ssc_dis2.at<cv::Vec4f>(0, new_col);
			// if(vec1[3]==vec2[3]){
			dis_count += fabs(vec1[0] - vec2[0]);
			// }
		}
		if (dis_count < similarity) 
		{
			similarity = dis_count;
			angle = i;
		}
	}

	// ROS_INFO("globalICP: sector_id: %f", angle);

	// angle = M_PI * (360. - angle * 360. / sectors) / 180.;
	angle = angle * step;
	// ROS_INFO("globalICP: ssc angle: %f", angle);

	auto cs = cos(angle);
	auto sn = sin(angle);

	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud1(new pcl::PointCloud<pcl::PointXYZ>), cloud2(new pcl::PointCloud<pcl::PointXYZ>);
	for (int i = 0; i < sectors; ++i) 
	{
		if (ssc_dis1.at<cv::Vec4f>(0, i)[3] > 0) {
			cloud1->push_back(pcl::PointXYZ(ssc_dis1.at<cv::Vec4f>(0, i)[1], ssc_dis1.at<cv::Vec4f>(0, i)[2], 0.));
		}
		if (ssc_dis2.at<cv::Vec4f>(0, i)[3] > 0) 
		{
			float tpx = ssc_dis2.at<cv::Vec4f>(0, i)[1] * cs -
						ssc_dis2.at<cv::Vec4f>(0, i)[2] * sn;
			float tpy = ssc_dis2.at<cv::Vec4f>(0, i)[1] * sn +
						ssc_dis2.at<cv::Vec4f>(0, i)[2] * cs;
			cloud2->push_back(pcl::PointXYZ(tpx, tpy, 0.));
		}
	}

	pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
	icp.setInputSource(cloud2);
	icp.setInputTarget(cloud1);
	pcl::PointCloud<pcl::PointXYZ> Final;
	icp.align(Final);
	
	Eigen::Affine3f trans;
	trans = icp.getFinalTransformation();
	Eigen::Affine3f trans1 = Eigen::Affine3f::Identity();
	trans1.rotate(Eigen::AngleAxisf(angle, Eigen::Vector3f::UnitZ()));


	float diff_x, diff_y, diff_z;
	float ROLL, PITCH, YAW;
	pcl::getTranslationAndEulerAngles(trans * trans1, diff_x, diff_y, diff_z, ROLL, PITCH, YAW);

	// ROS_INFO("globalICP: ICP angle: %f", YAW);

	if (show)
	{
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr temp_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

		for (int j = 0; j < sectors; ++j) 
		{
			if (ssc_dis1.at<cv::Vec4f>(0, j)[0] != 0) 
			{
				pcl::PointXYZRGB p;
				p.x = ssc_dis1.at<cv::Vec4f>(0, j)[1];
				p.y = ssc_dis1.at<cv::Vec4f>(0, j)[2];
				p.z = 0;
				// p.r = std::get<0>(Argmax2RGB[(int)ssc_dis1.at<cv::Vec4f>(0, j)[3]]);
				// p.g = std::get<1>(Argmax2RGB[(int)ssc_dis1.at<cv::Vec4f>(0, j)[3]]);
				// p.b = std::get<2>(Argmax2RGB[(int)ssc_dis1.at<cv::Vec4f>(0, j)[3]]);
				p.r = 255;
				p.g = 0;
				p.b = 0;
				temp_cloud->points.emplace_back(p);
			}

			if (ssc_dis2.at<cv::Vec4f>(0, j)[0] != 0) 
			{
				pcl::PointXYZRGB p;
				p.x = ssc_dis2.at<cv::Vec4f>(0, j)[1] * cs - ssc_dis2.at<cv::Vec4f>(0, j)[2] * sn;
				p.y = ssc_dis2.at<cv::Vec4f>(0, j)[1] * sn + ssc_dis2.at<cv::Vec4f>(0, j)[2] * cs;
				p.z = 1;
				// p.r = std::get<0>(Argmax2RGB[(int)ssc_dis2.at<cv::Vec4f>(0, j)[3]]);
				// p.g = std::get<1>(Argmax2RGB[(int)ssc_dis2.at<cv::Vec4f>(0, j)[3]]);
				// p.b = std::get<2>(Argmax2RGB[(int)ssc_dis2.at<cv::Vec4f>(0, j)[3]]);
				p.r = 0;
				p.g = 255;
				p.b = 0;
				temp_cloud->points.emplace_back(p);
			}

			if (ssc_dis2.at<cv::Vec4f>(0, j)[0] != 0) 
			{
				pcl::PointXYZRGB p;
				p.x = ssc_dis2.at<cv::Vec4f>(0, j)[1] * cos(YAW) - ssc_dis2.at<cv::Vec4f>(0, j)[2] * sin(YAW) + diff_x;
				p.y = ssc_dis2.at<cv::Vec4f>(0, j)[1] * sin(YAW) + ssc_dis2.at<cv::Vec4f>(0, j)[2] * cos(YAW) + diff_y;
				p.z = 2;
				// p.r = std::get<0>(Argmax2RGB[(int)ssc_dis2.at<cv::Vec4f>(0, j)[3]]);
				// p.g = std::get<1>(Argmax2RGB[(int)ssc_dis2.at<cv::Vec4f>(0, j)[3]]);
				// p.b = std::get<2>(Argmax2RGB[(int)ssc_dis2.at<cv::Vec4f>(0, j)[3]]);
				p.r = 0;
				p.g = 0;
				p.b = 255;
				temp_cloud->points.emplace_back(p);
			}

		}
	
		temp_cloud->height = 1;
		temp_cloud->width = temp_cloud->points.size();
		viewer->showCloud(temp_cloud);
		usleep(100);
	
	}
		
	return trans * trans1;

}

cv::Mat EPSCGeneration::calculateSC(const pcl::PointCloud<PointXYZIL>::Ptr filtered_pointcloud) 
{
	// const char NO_POINT = -100;
	cv::Mat sc = cv::Mat::zeros(cv::Size(sectors, rings), CV_16S);
	// cv::Mat sc(sectors, rings, CV_8U, cv::Scalar(NO_POINT));

	// for ( int row_idx = 0; row_idx < sc.rows; row_idx++ )
	// 	for ( int col_idx = 0; col_idx < sc.cols; col_idx++ )
	// 		sc.at<char>(row_idx, col_idx) = NO_POINT;

	for (int i = 0; i < (int)filtered_pointcloud->points.size(); i++) 
	{
		double distance = std::sqrt(
			filtered_pointcloud->points[i].x * filtered_pointcloud->points[i].x +
			filtered_pointcloud->points[i].y * filtered_pointcloud->points[i].y);
		if (distance >= max_dis || distance < min_dis) continue;
		int ring_id = std::floor((distance - min_dis) / ring_step);
		double angle = M_PI + std::atan2(filtered_pointcloud->points[i].y,filtered_pointcloud->points[i].x);
		int sector_id = std::floor(angle / sector_step);
		if (ring_id >= rings || ring_id < 0) continue;
		if (sector_id >= sectors || sector_id < 0) continue;

		// int z_temp = (int)(filtered_pointcloud->points[i].z) + LIDAR_HEIGHT;
		int  z_temp = (int)(100.0 * (filtered_pointcloud->points[i].z + LIDAR_HEIGHT)  / 8.0);

		// if(z_temp > 255) z_temp = 255;
		// if(z_temp < 0) z_temp = 0;

		if (sc.at<char>(ring_id, sector_id) < z_temp){
			sc.at<char>(ring_id, sector_id) = z_temp;
			//   std::cout << "z_temp.at(" << ring_id << ", " << sector_id << "): " << z_temp << "\t";
			//   std::cout << "sc.at(" << ring_id << ", " << sector_id << "): " << (int)sc.at<unsigned char>(ring_id, sector_id) << "---";
		}
	}

	// reset no points to zero (for cosine dist later)
	// for ( int row_idx = 0; row_idx < sc.rows; row_idx++ )
	// 	for ( int col_idx = 0; col_idx < sc.cols; col_idx++ )
	// 		if( sc.at<char>(row_idx, col_idx) == NO_POINT )
	// 			sc.at<char>(row_idx, col_idx) = 0;

	return sc;
}

cv::Mat EPSCGeneration::calculateISC(const pcl::PointCloud<PointXYZIL>::Ptr filtered_pointcloud) 
{
	cv::Mat isc = cv::Mat::zeros(cv::Size(sectors, rings), CV_8U);

	for (int i = 0; i < (int)filtered_pointcloud->points.size(); i++) {
		ROS_WARN_ONCE(
			"intensity is %f, if intensity showed here is integer format between " 
			"1-255, please uncomment #define INTEGER_INTENSITY "
			"in EPSCGenerationClass.cpp and recompile",
			(double)filtered_pointcloud->points[i].intensity);
		double distance = std::sqrt(
			filtered_pointcloud->points[i].x * filtered_pointcloud->points[i].x +
			filtered_pointcloud->points[i].y * filtered_pointcloud->points[i].y);
		if (distance >= max_dis || distance < min_dis) continue;
		int ring_id = std::floor((distance - min_dis) / ring_step);
		double angle = M_PI + std::atan2(filtered_pointcloud->points[i].y,filtered_pointcloud->points[i].x);
		int sector_id = std::floor(angle / sector_step);
		if (ring_id >= rings || ring_id < 0) continue;
		if (sector_id >= sectors || sector_id < 0) continue;
	#ifndef INTEGER_INTENSITY
		int intensity_temp = (int)(255 * filtered_pointcloud->points[i].intensity);
	#else
		int intensity_temp = (int)(filtered_pointcloud->points[i].intensity);
	#endif
		if (isc.at<unsigned char>(ring_id, sector_id) < intensity_temp)
			isc.at<unsigned char>(ring_id, sector_id) = intensity_temp;
	}

	return isc;
}

cv::Mat EPSCGeneration::calculateEPSC(
    const pcl::PointCloud<pcl::PointXYZI>::Ptr filtered_corner_pointcloud,
    const pcl::PointCloud<pcl::PointXYZI>::Ptr filtered_surf_pointcloud) 
{
	cv::Mat esc = cv::Mat::zeros(cv::Size(sectors, rings), CV_8U);
	cv::Mat psc = cv::Mat::zeros(cv::Size(sectors, rings), CV_8U);
	cv::Mat epsc = cv::Mat::zeros(cv::Size(sectors, rings), CV_8U);

	for (int i = 0; i < (int)filtered_corner_pointcloud->points.size(); i++) 
	{
		double distance = std::sqrt(
			filtered_corner_pointcloud->points[i].x * filtered_corner_pointcloud->points[i].x +
			filtered_corner_pointcloud->points[i].y * filtered_corner_pointcloud->points[i].y);
		if (distance >= max_dis || distance < min_dis) continue;
		int ring_id = std::floor((distance - min_dis) / ring_step);
		double angle = M_PI + std::atan2(filtered_corner_pointcloud->points[i].y,filtered_corner_pointcloud->points[i].x);
		int sector_id = std::floor(angle / sector_step);
		if (ring_id >= rings || ring_id < 0) continue;
		if (sector_id >= sectors || sector_id < 0) continue;
		esc.at<unsigned char>(ring_id, sector_id)++;
	}

	for (int i = 0; i < (int)filtered_surf_pointcloud->points.size(); i++) 
	{
		double distance = std::sqrt(
			filtered_surf_pointcloud->points[i].x * filtered_surf_pointcloud->points[i].x +
			filtered_surf_pointcloud->points[i].y * filtered_surf_pointcloud->points[i].y);
		if (distance >= max_dis || distance < min_dis) continue;
		int ring_id = std::floor((distance - min_dis) / ring_step);
		double angle = M_PI + std::atan2(filtered_surf_pointcloud->points[i].y,filtered_surf_pointcloud->points[i].x);
		int sector_id = std::floor(angle / sector_step);
		if (ring_id >= rings || ring_id < 0) continue;
		if (sector_id >= sectors || sector_id < 0) continue;
		psc.at<unsigned char>(ring_id, sector_id)++;
	}

	for (int i = 0; i < epsc.rows; i++) {
		for (int j = 0; j < epsc.cols; j++) {
			epsc.at<unsigned char>(i, j) = 100 * psc.at<unsigned char>(i, j) / (1 + esc.at<unsigned char>(i, j));
		}
	}
	return epsc;
}

cv::Mat EPSCGeneration::calculateSEPSC(const pcl::PointCloud<PointXYZIL>::Ptr filtered_pointcloud) 
{
	cv::Mat psc = cv::Mat::zeros(cv::Size(sectors, rings), CV_8U);
	cv::Mat esc = cv::Mat::zeros(cv::Size(sectors, rings), CV_8U);
	cv::Mat epsc = cv::Mat::zeros(cv::Size(sectors, rings), CV_8U);

	for (int i = 0; i < (int)filtered_pointcloud->points.size(); i++) 
	{
		auto label = filtered_pointcloud->points[i].label;
		double distance = std::sqrt(
			filtered_pointcloud->points[i].x * filtered_pointcloud->points[i].x +
			filtered_pointcloud->points[i].y * filtered_pointcloud->points[i].y);
		if (distance >= max_dis || distance < min_dis) continue;
		int ring_id = std::floor((distance - min_dis) / ring_step);
		double angle = M_PI + std::atan2(filtered_pointcloud->points[i].y,filtered_pointcloud->points[i].x);
		int sector_id = std::floor(angle / sector_step);
	
		// std::cout << "Label: " << label << std::endl;
		// std::cout << "Distance: " << distance << "\t";
		// std::cout << "sector_id: " << sector_id <<  "  ring_id: " << ring_id << "\t";
		
		if (ring_id >= rings || ring_id < 0) continue;
		if (sector_id >= sectors || sector_id < 0) continue;
		if (UsingLableMap[label] == 40 || UsingLableMap[label] == 50) {
			psc.at<unsigned char>(ring_id, sector_id)++;
			//   std::cout << "psc.at(" << ring_id << ", " << sector_id << "): " << (int)psc.at<unsigned char>(ring_id, sector_id) << "\t";
		} else if (UsingLableMap[label] == 81) {
			esc.at<unsigned char>(ring_id, sector_id)++;
			//   std::cout << "esc.at(" << ring_id << ", " << sector_id << "): " << (int)esc.at<unsigned char>(ring_id, sector_id) << "\t";
		}
		// std::cout << std::endl;
	}

	for (int i = 0; i < epsc.rows; i++) {
		for (int j = 0; j < epsc.cols; j++) {
			epsc.at<unsigned char>(i, j) = 100 * psc.at<unsigned char>(i, j) / (1 + esc.at<unsigned char>(i, j));
			//   std::cout << "epsc.at(" << i << ", " << j << "): " << (int)epsc.at<unsigned char>(i, j) << "\t";
		}
	}
	return epsc;
}

cv::Mat EPSCGeneration::calculateSSC(const pcl::PointCloud<PointXYZIL>::Ptr filtered_pointcloud) 
{
	cv::Mat ssc = cv::Mat::zeros(cv::Size(sectors, rings), CV_8U);

	for (int i = 0; i < (int)filtered_pointcloud->points.size(); i++) 
	{
		auto label = filtered_pointcloud->points[i].label;
		if (order_vec[label] > 0) 
		{
			double distance = std::sqrt(
				filtered_pointcloud->points[i].x * filtered_pointcloud->points[i].x +
				filtered_pointcloud->points[i].y * filtered_pointcloud->points[i].y);
			if (distance >= max_dis || distance < min_dis) continue;
			int ring_id = std::floor((distance - min_dis) / ring_step);
			double angle = M_PI + std::atan2(filtered_pointcloud->points[i].y,filtered_pointcloud->points[i].x);
			int sector_id = std::floor(angle / sector_step);
			if (ring_id >= rings || ring_id < 0) continue;
			if (sector_id >= sectors || sector_id < 0) continue;
			if (order_vec[label] > order_vec[ssc.at<unsigned char>(ring_id, sector_id)]) {
				ssc.at<unsigned char>(ring_id, sector_id) = label;
			}
		}
	}
	return ssc;
}


cv::Mat EPSCGeneration::calculateFEPSC(
			const pcl::PointCloud<pcl::PointXYZI>::Ptr filtered_corner_pointcloud,
			const pcl::PointCloud<pcl::PointXYZI>::Ptr filtered_surf_pointcloud,
			const pcl::PointCloud<PointXYZIL>::Ptr filtered_pointcloud)
{
	cv::Mat sepsc = calculateSEPSC(filtered_pointcloud);
	cv::Mat epsc = calculateEPSC(filtered_corner_pointcloud, filtered_surf_pointcloud);
	cv::Mat fepsc = cv::Mat::zeros(cv::Size(sectors, rings), CV_8U);

	for (int p = 0; p < sectors; p++) {
		for (int q = 0; q < rings; q++) {
			fepsc.at<unsigned char>(q, p) = sepsc.at<unsigned char>(q, p) * 0.4 + epsc.at<unsigned char>(q, p) * 0.6;
		}
	}

	return fepsc;
}



double EPSCGeneration::calculateLabelSim(cv::Mat &desc1, cv::Mat &desc2) 
{
	double similarity = 0;
	int sectors = desc1.cols;
	int rings = desc1.rows;
	int valid_num = 0;
	for (int p = 0; p < sectors; p++) {
		for (int q = 0; q < rings; q++) {
			if (desc1.at<unsigned char>(q, p) == 0 && desc2.at<unsigned char>(q, p) == 0) {
				continue;
			}
			valid_num++;

			if (desc1.at<unsigned char>(q, p) == desc2.at<unsigned char>(q, p)) {
				similarity++;
			}
		}
	}
	// std::cout<<similarity<<std::endl;
	return similarity / valid_num;
}

double EPSCGeneration::calculateDistance(const cv::Mat &desc1,
                                         const cv::Mat &desc2, double &angle) 
{
	double difference = 1.0;
	double angle_temp = angle;
	for (int i = -10; i < 10; i++) 
	{
		int match_count = 0;
		int total_points = 0;
		for (int p = 0; p < sectors; p++) 
		{
			int new_col = p + i;
			if (new_col >= sectors) new_col = new_col - sectors;
			if (new_col < 0) new_col = new_col + sectors;
			for (int q = 0; q < rings; q++) {
				match_count += abs(desc1.at<unsigned char>(q, p) -
								desc2.at<unsigned char>(q, new_col));
				total_points++;
			}
		}
		double diff_temp = ((double)match_count) / (sectors * rings * 255);
		if (diff_temp < difference) {
			difference = diff_temp;
			angle = angle_temp + (i * sector_step);
		}
	}
	return 1 - difference;
}


void EPSCGeneration::loopDetection(
    const pcl::PointCloud<pcl::PointXYZI>::Ptr &corner_pc,
    const pcl::PointCloud<pcl::PointXYZI>::Ptr &surf_pc,
    const pcl::PointCloud<PointXYZIL>::Ptr &semantic_pc,
    Eigen::Affine3f &odom) 
{
	pcl::PointCloud<pcl::PointXYZI>::Ptr pc_filtered_corner(new pcl::PointCloud<pcl::PointXYZI>());
	pcl::PointCloud<pcl::PointXYZI>::Ptr pc_filtered_surf(new pcl::PointCloud<pcl::PointXYZI>());
	pcl::PointCloud<PointXYZIL>::Ptr pc_filtered_semantic(new pcl::PointCloud<PointXYZIL>());

	// ????????????
	// groundFilter(corner_pc, pc_filtered_corner);
	// groundFilter(surf_pc, pc_filtered_surf);
	// groundFilter(semantic_pc, pc_filtered_semantic);
	pc_filtered_corner = corner_pc;
	pc_filtered_surf = surf_pc;
	pc_filtered_semantic = semantic_pc;

	// std::cout << "--- EPSCGeneration ---" << std::endl;
	// std::cout << "pc_filtered_corner size : " << pc_filtered_corner->size() << std::endl;
	// std::cout << "pc_filtered_surf size : " << pc_filtered_surf->size() << std::endl;
	// std::cout << "pc_filtered_semantic size : " << pc_filtered_semantic->size() << std::endl;

  	// Eigen::Vector3d current_t(odom(0, 3), odom(1, 3), odom(2, 3));
	float x_t, y_t, z_t;
	float roll_t, pitch_t, yaw_t;
	pcl::getTranslationAndEulerAngles(odom, x_t, y_t, z_t, roll_t, pitch_t, yaw_t);
  	Eigen::Vector3d current_t(x_t, y_t, 0.0);

	// dont change push_back sequence
	if (travelDistanceArr.size() == 0) {
		travelDistanceArr.push_back(0);
	} else {
		double dis_temp = travelDistanceArr.back() + std::sqrt((posArr.back() - current_t).array().square().sum());
		travelDistanceArr.push_back(dis_temp);
	}

	current_frame_id = posArr.size();
	matched_frame_id.clear();
	matched_frame_transform.clear();

	int best_matched_id_isc = -1;
	double best_score_isc = 0.0;
	Eigen::Affine3f best_score_isc_transform;

	int best_matched_id_sc = -1;
	double best_score_sc = 0.0;
	Eigen::Affine3f best_score_sc_transform;

	int best_matched_id_epsc = -1;
	double best_score_epsc = 0.0;
	Eigen::Affine3f best_score_epsc_transform;

	int best_matched_id_sepsc = -1;
	double best_score_sepsc = 0.0;
	Eigen::Affine3f best_score_sepsc_transform;

	int best_matched_id_ssc = -1;
	double best_score_ssc = 0.0;
	Eigen::Affine3f best_score_ssc_transform;

	int best_matched_id_fepsc = -1;
	double best_score_fepsc = 0.0;
	Eigen::Affine3f best_score_fepsc_transform;

	double min_ditance = 1000000;
	int best_matched_id_pose = -1;
	Eigen::Affine3f best_score_pose_transform;

	cv::Mat ISC_cur, SC_cur, EPSC_cur, SEPSC_cur, SSC_cur, FEPSC_cur;


  	cv::Mat cur_dis = project(pc_filtered_semantic);
	for (int i = 0; i < (int)posArr.size(); i++) 
	{
		double delta_travel_distance = travelDistanceArr.back() - travelDistanceArr[i];
		double pos_distance = std::sqrt((posArr[i] - posArr.back()).array().square().sum());
		if (delta_travel_distance > SKIP_NEIBOUR_DISTANCE && pos_distance < delta_travel_distance * INFLATION_COVARIANCE) 
		{
			// ROS_INFO("Matched_id: %d, delta_travel_distance: %f, pos_distance : %f", i, delta_travel_distance, pos_distance);
			cv::Mat before_dis = ProjectArr[i];

			// double angle = 0;
			// float diff_x = 0;
			// float diff_y = 0;
			// globalICP(before_dis, cur_dis, angle, diff_x, diff_y);
			// if (fabs(diff_x) > 5 || fabs(diff_y) > 5) {
			// 	diff_x = 0;
			// 	diff_y = 0;
			// }
			// Eigen::Affine3f transform = Eigen::Affine3f::Identity();
			// transform.translation() << diff_x, diff_y, 0;
			// transform.rotate(Eigen::AngleAxisf(angle, Eigen::Vector3f::UnitZ()));

			float yaw_diff = yaw_t - yawArr[i];

			Eigen::Affine3f transform = globalICP(before_dis, cur_dis, yaw_diff);
			float diff_x, diff_y, diff_z;
			float ROLL, PITCH, angle;
			pcl::getTranslationAndEulerAngles(transform, diff_x, diff_y, diff_z, ROLL, PITCH, angle);

			pcl::PointCloud<PointXYZIL>::Ptr trans_cloud_semantic(new pcl::PointCloud<PointXYZIL>);
			transformPointCloud(*pc_filtered_semantic, *trans_cloud_semantic, transform);
			
			pcl::PointCloud<pcl::PointXYZI>::Ptr trans_cloud_corner(new pcl::PointCloud<pcl::PointXYZI>);
			pcl::PointCloud<pcl::PointXYZI>::Ptr trans_cloud_surf(new pcl::PointCloud<pcl::PointXYZI>);
			transformPointCloud(*pc_filtered_corner, *trans_cloud_corner, transform);
			transformPointCloud(*pc_filtered_surf, *trans_cloud_surf, transform);

			if (UsingISCFlag) 
			{
				auto desc1 = ISCArr[i];
				ISC_cur = calculateISC(trans_cloud_semantic);
				double isc_angle = angle;
				auto score = calculateDistance(desc1, ISC_cur, isc_angle);
				// std::cout << "ISC_score: " << score << std::endl;
				if (score > DISTANCE_THRESHOLD && score > best_score_isc) 
				{
					best_score_isc = score;
					best_matched_id_isc = i;

					Eigen::Affine3f isc_transform = Eigen::Affine3f::Identity();
					isc_transform.translation() << diff_x, diff_y, 0;
					isc_transform.rotate(Eigen::AngleAxisf(isc_angle, Eigen::Vector3f::UnitZ()));

					best_score_isc_transform = isc_transform;
				}
			}

			if (UsingSCFlag) 
			{
				auto desc1 = SCArr[i];
				SC_cur = calculateSC(trans_cloud_semantic);
				double sc_angle = angle;
				auto score = calculateDistance(desc1, SC_cur, sc_angle);
				// std::cout << "SC_score: " << score << std::endl;
				if (score > DISTANCE_THRESHOLD && score > best_score_sc) 
				{
					best_score_sc = score;
					best_matched_id_sc = i;

					Eigen::Affine3f sc_transform = Eigen::Affine3f::Identity();
					sc_transform.translation() << diff_x, diff_y, 0;
					sc_transform.rotate(Eigen::AngleAxisf(sc_angle, Eigen::Vector3f::UnitZ()));

					best_score_sc_transform = sc_transform;
				}
			}

			if (UsingEPSCFlag) 
			{
				auto desc1 = EPSCArr[i];
				EPSC_cur = calculateEPSC(trans_cloud_corner, trans_cloud_surf);
				// double epsc_angle = angle;
				double epsc_angle = yaw_diff;
				auto score = calculateDistance(desc1, EPSC_cur, epsc_angle);
				std::cout << "EPSC_score: " << score << std::endl;
				if (score > DISTANCE_THRESHOLD && score > best_score_epsc) 
				{
					best_score_epsc = score;
					best_matched_id_epsc = i;

					Eigen::Affine3f epsc_transform = Eigen::Affine3f::Identity();
					epsc_transform.translation() << diff_x, diff_y, 0;
					epsc_transform.rotate(Eigen::AngleAxisf(epsc_angle, Eigen::Vector3f::UnitZ()));

					best_score_epsc_transform = epsc_transform;
				}
			}

			if (UsingSEPSCFlag) 
			{
				auto desc1 = SEPSCArr[i];
				SEPSC_cur = calculateSEPSC(trans_cloud_semantic);
				double sepsc_angle = angle;
				auto score = calculateDistance(desc1, SEPSC_cur, sepsc_angle);
				// std::cout << "SEPSC_score: " << score << std::endl;
				if (score > DISTANCE_THRESHOLD && score > best_score_sepsc) 
				{
					best_score_sepsc = score;
					best_matched_id_sepsc = i;
										
					Eigen::Affine3f sepsc_transform = Eigen::Affine3f::Identity();
					sepsc_transform.translation() << diff_x, diff_y, 0;
					sepsc_transform.rotate(Eigen::AngleAxisf(sepsc_angle, Eigen::Vector3f::UnitZ()));

					best_score_sepsc_transform = sepsc_transform;
				}
			}

			if (UsingFEPSCFlag) 
			{
				auto desc1 = FEPSCArr[i];
				FEPSC_cur = calculateFEPSC(trans_cloud_corner, trans_cloud_surf, trans_cloud_semantic);
				double fepsc_angle = angle;
				auto score = calculateDistance(desc1, FEPSC_cur, fepsc_angle);
				std::cout << "FEPSC_score: " << score << std::endl;
				if (score > DISTANCE_THRESHOLD && score > best_score_fepsc) 
				{
					best_score_fepsc = score;
					best_matched_id_fepsc = i;

					Eigen::Affine3f fepsc_transform = Eigen::Affine3f::Identity();
					fepsc_transform.translation() << diff_x, diff_y, 0;
					fepsc_transform.rotate(Eigen::AngleAxisf(angle, Eigen::Vector3f::UnitZ()));

					best_score_fepsc_transform = fepsc_transform;
				}
			}

			if (UsingSSCFlag) 
			{
				auto desc1 = SSCArr[i];
				SSC_cur = calculateSSC(trans_cloud_semantic);
				auto score = calculateLabelSim(desc1, SSC_cur);
				// std::cout << "SSC_score: " << score << std::endl;
				if (score > LABEL_THRESHOLD && score > best_score_ssc) 
				{
					best_score_ssc = score;
					best_matched_id_ssc = i;
					best_score_ssc_transform = transform;
				}
			}

			if (UsingPoseFlag) 
			{
				if (pos_distance < min_ditance) 
				{
					min_ditance = pos_distance;
					best_matched_id_pose = i;
					best_score_pose_transform = transform;
				}
			}
		}
	}


	posArr.push_back(current_t);
	yawArr.push_back(yaw_t);
	ProjectArr.push_back(cur_dis);

	if (UsingISCFlag) 
	{
		ISC_cur = calculateISC(pc_filtered_semantic);
		ISCArr.push_back(ISC_cur);
		if (best_matched_id_isc != -1) 
		{
			matched_frame_id.push_back(best_matched_id_isc);
			matched_frame_transform.push_back(best_score_isc_transform);
			ROS_WARN("ISC: received loop closure candidate: current: %d, history: %d, total_score: %f",
						current_frame_id, best_matched_id_isc, best_score_isc);
		}
	}

	if (UsingSCFlag) 
	{
		SC_cur = calculateSC(pc_filtered_semantic);
		SCArr.push_back(SC_cur);
		if (best_matched_id_sc != -1) 
		{
			matched_frame_id.push_back(best_matched_id_sc);
			matched_frame_transform.push_back(best_score_sc_transform);
			ROS_WARN("SC: received loop closure candidate: current: %d, history: %d, total_score: %f",
						current_frame_id, best_matched_id_sc, best_score_sc);
		}
	}

	if (UsingEPSCFlag) 
	{
		EPSC_cur = calculateEPSC(pc_filtered_corner, pc_filtered_surf);
		EPSCArr.push_back(EPSC_cur);
		if (best_matched_id_epsc != -1) 
		{
			matched_frame_id.push_back(best_matched_id_epsc);
			matched_frame_transform.push_back(best_score_epsc_transform);
			ROS_WARN("EPSC: received loop closure candidate: current: %d, history: %d, total_score: %f",
						current_frame_id, best_matched_id_epsc, best_score_epsc);
		}
	}

	if (UsingSEPSCFlag) 
	{
		SEPSC_cur = calculateSEPSC(pc_filtered_semantic);
		SEPSCArr.push_back(SEPSC_cur);
		if (best_matched_id_sepsc != -1) 
		{
			matched_frame_id.push_back(best_matched_id_sepsc);
			matched_frame_transform.push_back(best_score_sepsc_transform);
			ROS_WARN("SEPSC: received loop closure candidate: current: %d, history: %d, total_score: %f",
						current_frame_id, best_matched_id_sepsc, best_score_sepsc);
		}
	}

	if (UsingFEPSCFlag) 
	{
		FEPSC_cur = calculateFEPSC(pc_filtered_corner, pc_filtered_surf, pc_filtered_semantic);
		FEPSCArr.push_back(FEPSC_cur);
		if (best_matched_id_fepsc != -1) 
		{
			matched_frame_id.push_back(best_matched_id_fepsc);
			matched_frame_transform.push_back(best_score_fepsc_transform);
			ROS_WARN("FEPSC: received loop closure candidate: current: %d, history: %d, total_score: %f",
						current_frame_id, best_matched_id_fepsc, best_score_fepsc);
		}
	}

	if (UsingSSCFlag) 
	{
		SSC_cur = calculateSSC(pc_filtered_semantic);
		SSCArr.push_back(SSC_cur);
		if (best_matched_id_ssc != -1) 
		{
			matched_frame_id.push_back(best_matched_id_ssc);
			matched_frame_transform.push_back(best_score_ssc_transform);
			ROS_WARN("SSC: received loop closure candidate: current: %d, history: %d, total_score: %f",
						current_frame_id, best_matched_id_ssc, best_score_ssc);
		}
	}

	if (UsingPoseFlag) 
	{
		if (best_matched_id_pose != -1) 
		{
			matched_frame_id.push_back(best_matched_id_pose);
			matched_frame_transform.push_back(best_score_pose_transform);
			ROS_WARN("POSE: received loop closure candidate: current: %d, history: %d, min_ditance: %f",
						current_frame_id, best_matched_id_pose, min_ditance);
		}
	}
}

cv::Mat EPSCGeneration::getLastEPSCRGB() 
{
	if (UsingEPSCFlag) 
	{
		cv::Mat epsc_color = cv::Mat::zeros(cv::Size(sectors, rings), CV_8UC3);
		for (int i = 0; i < EPSCArr.back().rows; i++) {
			for (int j = 0; j < EPSCArr.back().cols; j++) {
				epsc_color.at<cv::Vec3b>(i, j) = color_projection[EPSCArr.back().at<unsigned char>(i, j)];
			}
		}
		return epsc_color;
	}
}

cv::Mat EPSCGeneration::getLastSCRGB() 
{
	if (UsingSCFlag) 
	{
		cv::Mat sc_color = cv::Mat::zeros(cv::Size(sectors, rings), CV_8UC3);
		for (int i = 0; i < SCArr.back().rows; i++) {
			for (int j = 0; j < SCArr.back().cols; j++) {
				sc_color.at<cv::Vec3b>(i, j) = color_projection[SCArr.back().at<unsigned char>(i, j)];
			}
		}
		return sc_color;
	}
}

cv::Mat EPSCGeneration::getLastISCRGB() 
{
	if (UsingISCFlag) 
	{
		cv::Mat isc_color = cv::Mat::zeros(cv::Size(sectors, rings), CV_8UC3);
		for (int i = 0; i < ISCArr.back().rows; i++) {
			for (int j = 0; j < ISCArr.back().cols; j++) {
				isc_color.at<cv::Vec3b>(i, j) = color_projection[ISCArr.back().at<unsigned char>(i, j)];
			}
		}
		return isc_color;
	}
}

cv::Mat EPSCGeneration::getLastSEPSCRGB() 
{
	if (UsingSEPSCFlag) 
	{
		// std::cout << "SEPSCArr.size(): " << SEPSCArr.size() << "\t";
		// std::cout << "SEPSCArr.back().rows: " << SEPSCArr.back().rows << "\t";
		// std::cout << "SEPSCArr.back().cols: " << SEPSCArr.back().cols << "\t";
		// std::cout << std::endl;
		cv::Mat sepsc_color = cv::Mat::zeros(cv::Size(sectors, rings), CV_8UC3);
		for (int i = 0; i < SEPSCArr.back().rows; i++) {
			for (int j = 0; j < SEPSCArr.back().cols; j++) {
				sepsc_color.at<cv::Vec3b>(i, j) = color_projection[SEPSCArr.back().at<unsigned char>(i, j)];
			}
		}
		return sepsc_color;
	}
}

cv::Mat EPSCGeneration::getLastFEPSCRGB(void)
{
	if (UsingFEPSCFlag) 
	{
		cv::Mat fepsc_color = cv::Mat::zeros(cv::Size(sectors, rings), CV_8UC3);
		for (int i = 0; i < FEPSCArr.back().rows; i++) {
			for (int j = 0; j < FEPSCArr.back().cols; j++) {
				fepsc_color.at<cv::Vec3b>(i, j) = color_projection[FEPSCArr.back().at<unsigned char>(i, j)];
			}
		}
		return fepsc_color;
	}
}

cv::Mat EPSCGeneration::getLastFEPSCRGB(int id)
{
	if (UsingFEPSCFlag) 
	{
		cv::Mat fepsc_color = cv::Mat::zeros(cv::Size(sectors, rings), CV_8UC3);
		for (int i = 0; i < FEPSCArr[id].rows; i++) {
			for (int j = 0; j < FEPSCArr[id].cols; j++) {
				fepsc_color.at<cv::Vec3b>(i, j) = color_projection[FEPSCArr[id].at<unsigned char>(i, j)];
			}
		}
		return fepsc_color;
	}
}

cv::Mat EPSCGeneration::getLastSSCRGB() 
{
	if (UsingSSCFlag) 
	{
		auto color_image = getColorImage(SSCArr.back());
		return color_image;
	}
}
















































// loop closure ????????????

std::string sepsc_path = "/home/wqz/AWorkSpace/TestData/0332/sepsc.txt"; //ADD
std::string sepsc_pro_path = "/home/wqz/AWorkSpace/TestData/0332/sepsc_pro.txt"; //ADD
std::string sepsc_dis_path = "/home/wqz/AWorkSpace/TestData/0332/sepsc_dis.txt"; //ADD

std::string epsc_path = "/home/wqz/AWorkSpace/TestData/0332/epsc.txt"; //ADD
std::string epsc_geo_path = "/home/wqz/AWorkSpace/TestData/0332/epsc_geo.txt"; //ADD
std::string epsc_inten_path = "/home/wqz/AWorkSpace/TestData/0332/epsc_inten.txt"; //ADD

std::string global_path = "/home/wqz/AWorkSpace/TestData/0332/global.txt"; //ADD

std::string epsc_de_path = "/home/wqz/AWorkSpace/TestData/0332/epsc_de.txt"; //ADD
std::string sepsc_de_path = "/home/wqz/AWorkSpace/TestData/0332/sepsc_de.txt"; //ADD

std::chrono::time_point<std::chrono::system_clock> start_time, end_time;
float  test = 0;

float cur_epsc_time = 0;
float cur_sepsc_time = 0;



int rings = 20;
int sectors = 90; //180

int ID = 0;

double calculate_geometry_dis( const cv::Mat& desc1,  const cv::Mat& desc2, int& angle){
    double similarity = 0.0;

    for(int i=0;i<sectors;i++){
        int match_count=0;
        for(int p=0;p<sectors;p++){
            int new_col = p+i>=sectors?p+i-sectors:p+i;
            for(int q=0;q<rings;q++){
                if((desc1.at<unsigned char>(q,p)== true && desc2.at<unsigned char>(q,new_col)== true) || (desc1.at<unsigned char>(q,p)== false && desc2.at<unsigned char>(q,new_col)== false)){
                    match_count++;
                }

            }
        }
        if(match_count>similarity){
            similarity=match_count;
            angle = i;
        }

    }
    return similarity/(sectors*rings);
    
}
double calculate_intensity_dis( const cv::Mat& desc1, const cv::Mat& desc2, int& angle){
    double difference = 1.0;
    double angle_temp = angle;
    for(int i=angle_temp-10;i<angle_temp+10;i++){

        int match_count=0;
        int total_points=0;
        for(int p=0;p<sectors;p++){
            int new_col = p+i;
            if(new_col>=sectors)
                new_col = new_col-sectors;
            if(new_col<0)
                new_col = new_col+sectors;
            for(int q=0;q<rings;q++){
                    match_count += abs(desc1.at<unsigned char>(q,p)-desc2.at<unsigned char>(q,new_col));
                    total_points++;
            }
            
        }
        double diff_temp = ((double)match_count)/(sectors*rings*255);
        if(diff_temp<difference)
            difference=diff_temp;

    }
    return 1 - difference;
    
}
bool is_loop_pair(const cv::Mat & desc1, const cv::Mat& desc2, double& geo_score, double& inten_score){
    int angle =0;

// start_time = std::chrono::system_clock::now();
    geo_score = calculate_geometry_dis(desc1,desc2,angle);
// end_time  = std::chrono::system_clock::now();
// std::ofstream foutC_epsc_geo(epsc_geo_path, std::ios::app);
// foutC_epsc_geo.setf(std::ios::scientific, std::ios::floatfield);
// foutC_epsc_geo.precision(6);
// test = ((std::chrono::duration<float>)(end_time - start_time)).count()*1000;
// foutC_epsc_geo <<  test << endl ;	
// foutC_epsc_geo.close();



    std::cout<<"geo_score: "<<geo_score<<std::endl;
    if(geo_score>GEOMETRY_THRESHOLD)
	{


// start_time = std::chrono::system_clock::now();
        inten_score = calculate_intensity_dis(desc1,desc2,angle);
// end_time  = std::chrono::system_clock::now();
// std::ofstream foutC_epsc_inten(epsc_inten_path, std::ios::app);
// foutC_epsc_inten.setf(std::ios::scientific, std::ios::floatfield);
// foutC_epsc_inten.precision(6);
// test = ((std::chrono::duration<float>)(end_time - start_time)).count()*1000;
// foutC_epsc_inten << test << endl ;	
// foutC_epsc_inten.close();


        std::cout<<"inten_score: "<<inten_score<<std::endl;
        if(inten_score>INTENSITY_THRESHOLD)
		{
            return true;
        }
    }
    return false;
}


void EPSCGeneration::loopDetectionTest(
    const pcl::PointCloud<pcl::PointXYZI>::Ptr &corner_pc,
    const pcl::PointCloud<pcl::PointXYZI>::Ptr &surf_pc,
    const pcl::PointCloud<PointXYZIL>::Ptr &semantic_pc,
    Eigen::Affine3f &odom) 
{
	pcl::PointCloud<pcl::PointXYZI>::Ptr pc_filtered_corner(new pcl::PointCloud<pcl::PointXYZI>());
	pcl::PointCloud<pcl::PointXYZI>::Ptr pc_filtered_surf(new pcl::PointCloud<pcl::PointXYZI>());
	pcl::PointCloud<PointXYZIL>::Ptr pc_filtered_semantic(new pcl::PointCloud<PointXYZIL>());

	// ????????????
	// groundFilter(corner_pc, pc_filtered_corner);
	// groundFilter(surf_pc, pc_filtered_surf);
	// groundFilter(semantic_pc, pc_filtered_semantic);
	pc_filtered_corner = corner_pc;
	pc_filtered_surf = surf_pc;
	pc_filtered_semantic = semantic_pc;


	float x_t, y_t, z_t;
	float roll_t, pitch_t, yaw_t;
	pcl::getTranslationAndEulerAngles(odom, x_t, y_t, z_t, roll_t, pitch_t, yaw_t);
  	Eigen::Vector3d current_t(x_t, y_t, 0.0);

	// dont change push_back sequence
	if (travelDistanceArr.size() == 0) {
		travelDistanceArr.push_back(0);
	} else {
		double dis_temp = travelDistanceArr.back() + std::sqrt((posArr.back() - current_t).array().square().sum());
		travelDistanceArr.push_back(dis_temp);
	}

	current_frame_id = posArr.size();
	matched_frame_id.clear();
	matched_frame_transform.clear();


	int best_matched_id_epsc = -1;
	double best_score_epsc = 0.0;
	Eigen::Affine3f best_score_epsc_transform;


	int best_matched_id_sepsc = -1;
	double best_score_sepsc = 0.0;
	Eigen::Affine3f best_score_sepsc_transform;

	cv::Mat EPSC_cur, SEPSC_cur;

  	cv::Mat cur_dis = project(pc_filtered_semantic);

	bool  isFlag = false;
	for (int i = 0; i < (int)posArr.size(); i++) 
	{
		double delta_travel_distance = travelDistanceArr.back() - travelDistanceArr[i];
		double pos_distance = std::sqrt((posArr[i] - posArr.back()).array().square().sum());
		if (delta_travel_distance > SKIP_NEIBOUR_DISTANCE && pos_distance < delta_travel_distance * INFLATION_COVARIANCE) 
		{

			isFlag = true;

ID++;


// start_time = std::chrono::system_clock::now();
// project(pc_filtered_semantic);
// end_time  = std::chrono::system_clock::now();
// std::ofstream foutC_sepsc_pro(sepsc_pro_path, std::ios::app);
// foutC_sepsc_pro.setf(std::ios::scientific, std::ios::floatfield);
// foutC_sepsc_pro.precision(6);
// test = ((std::chrono::duration<float>)(end_time - start_time)).count()*1000;
// foutC_sepsc_pro << test << endl ;	
// foutC_sepsc_pro.close();


			// ROS_INFO("Matched_id: %d, delta_travel_distance: %f, pos_distance : %f", i, delta_travel_distance, pos_distance);
			cv::Mat before_dis = ProjectArr[i];

			float yaw_diff = yaw_t - yawArr[i];

// start_time = std::chrono::system_clock::now();
			Eigen::Affine3f transform = globalICP(before_dis, cur_dis, yaw_diff);
// end_time  = std::chrono::system_clock::now();
// std::ofstream foutC_global(global_path, std::ios::app);
// foutC_global.setf(std::ios::scientific, std::ios::floatfield);
// foutC_global.precision(6);
// test = ((std::chrono::duration<float>)(end_time - start_time)).count()*1000;
// foutC_global << test << endl ;	
// foutC_global.close();


			float diff_x, diff_y, diff_z;
			float ROLL, PITCH, angle;
			pcl::getTranslationAndEulerAngles(transform, diff_x, diff_y, diff_z, ROLL, PITCH, angle);

			pcl::PointCloud<PointXYZIL>::Ptr trans_cloud_semantic(new pcl::PointCloud<PointXYZIL>);
			transformPointCloud(*pc_filtered_semantic, *trans_cloud_semantic, transform);
			
			pcl::PointCloud<pcl::PointXYZI>::Ptr trans_cloud_corner(new pcl::PointCloud<pcl::PointXYZI>);
			pcl::PointCloud<pcl::PointXYZI>::Ptr trans_cloud_surf(new pcl::PointCloud<pcl::PointXYZI>);
			transformPointCloud(*pc_filtered_corner, *trans_cloud_corner, transform);
			transformPointCloud(*pc_filtered_surf, *trans_cloud_surf, transform);


			if (UsingEPSCFlag) 
			{
start_time = std::chrono::system_clock::now();

				auto desc1 = EPSCArr[i];
				double epsc_angle = angle;

// start_time = std::chrono::system_clock::now();
				// EPSC_cur = calculateEPSC(trans_cloud_corner, trans_cloud_surf);
// end_time  = std::chrono::system_clock::now();
// std::ofstream foutC_epsc(epsc_path, std::ios::app);
// foutC_epsc.setf(std::ios::scientific, std::ios::floatfield);
// foutC_epsc.precision(6);
// test = ((std::chrono::duration<float>)(end_time - start_time)).count()*1000;
// foutC_epsc << test << endl ;	
// foutC_epsc.close();


				// auto score = calculateDistance(desc1, EPSC_cur, epsc_angle);
				// // std::cout << "EPSC_score: " << score << std::endl;
				// if (score > DISTANCE_THRESHOLD && score > best_score_epsc) 
				// {
				// 	best_score_epsc = score;
				// 	best_matched_id_epsc = i;

				// 	Eigen::Affine3f epsc_transform = Eigen::Affine3f::Identity();
				// 	epsc_transform.translation() << diff_x, diff_y, 0;
				// 	epsc_transform.rotate(Eigen::AngleAxisf(epsc_angle, Eigen::Vector3f::UnitZ()));

				// 	best_score_epsc_transform = epsc_transform;
				// }

				EPSC_cur = calculateEPSC(pc_filtered_corner, pc_filtered_surf);
				double geo_score=0;
				double inten_score =0;

				bool isLoop = is_loop_pair(EPSC_cur, desc1, geo_score, inten_score);


				if(isLoop)
				{
					if(geo_score+inten_score>best_score_epsc){
						best_score_epsc = geo_score+inten_score;
						best_matched_id_epsc = i;

						Eigen::Affine3f epsc_transform = Eigen::Affine3f::Identity();
						epsc_transform.translation() << diff_x, diff_y, 0;
						epsc_transform.rotate(Eigen::AngleAxisf(epsc_angle, Eigen::Vector3f::UnitZ()));

						best_score_epsc_transform = epsc_transform;
					}
				}


end_time  = std::chrono::system_clock::now();
test = ((std::chrono::duration<float>)(end_time - start_time)).count()*1000;
cur_epsc_time += test;

			}

			if (UsingSEPSCFlag) 
			{

start_time = std::chrono::system_clock::now();

				auto desc1 = SEPSCArr[i];

// start_time = std::chrono::system_clock::now();
				SEPSC_cur = calculateSEPSC(trans_cloud_semantic);
// end_time  = std::chrono::system_clock::now();
// std::ofstream foutC_sepsc(sepsc_path, std::ios::app);
// foutC_sepsc.setf(std::ios::scientific, std::ios::floatfield);
// foutC_sepsc.precision(6);
// test = ((std::chrono::duration<float>)(end_time - start_time)).count()*1000;
// foutC_sepsc << test << endl ;
// foutC_sepsc.close();


				double sepsc_angle = angle;


// start_time = std::chrono::system_clock::now();
				auto score = calculateDistance(desc1, SEPSC_cur, sepsc_angle);
// end_time  = std::chrono::system_clock::now();
// std::ofstream foutC_sepsc_dis(sepsc_dis_path, std::ios::app);
// foutC_sepsc_dis.setf(std::ios::scientific, std::ios::floatfield);
// foutC_sepsc_dis.precision(6);
// test = ((std::chrono::duration<float>)(end_time - start_time)).count()*1000;
// foutC_sepsc_dis << test << endl ;
// foutC_sepsc_dis.close();


				std::cout << "SEPSC_score: " << score << std::endl;
				if (score > DISTANCE_THRESHOLD && score > best_score_sepsc) 
				{
					best_score_sepsc = score;
					best_matched_id_sepsc = i;
										
					Eigen::Affine3f sepsc_transform = Eigen::Affine3f::Identity();
					sepsc_transform.translation() << diff_x, diff_y, 0;
					sepsc_transform.rotate(Eigen::AngleAxisf(sepsc_angle, Eigen::Vector3f::UnitZ()));

					best_score_sepsc_transform = sepsc_transform;
				}
			}

end_time  = std::chrono::system_clock::now();
test = ((std::chrono::duration<float>)(end_time - start_time)).count()*1000;
cur_sepsc_time += test;

		}
	}


	posArr.push_back(current_t);
	yawArr.push_back(yaw_t);
	ProjectArr.push_back(cur_dis);

if(isFlag)
{
	std::ofstream foutC_epsc_de(epsc_de_path, std::ios::app);
	foutC_epsc_de.setf(std::ios::scientific, std::ios::floatfield);
	foutC_epsc_de.precision(6);
	foutC_epsc_de << cur_epsc_time << endl ;
	foutC_epsc_de.close();

	std::ofstream foutC_sepsc_de(sepsc_de_path, std::ios::app);
	foutC_sepsc_de.setf(std::ios::scientific, std::ios::floatfield);
	foutC_sepsc_de.precision(6);
	foutC_sepsc_de << cur_sepsc_time << endl ;
	foutC_sepsc_de.close();
}

cur_epsc_time = 0;
cur_sepsc_time = 0;



	if (UsingEPSCFlag) 
	{
		EPSC_cur = calculateEPSC(pc_filtered_corner, pc_filtered_surf);
		EPSCArr.push_back(EPSC_cur);
		if (best_matched_id_epsc != -1) 
		{
			matched_frame_id.push_back(best_matched_id_epsc);
			matched_frame_transform.push_back(best_score_epsc_transform);
			ROS_WARN("EPSC: received loop closure candidate: current: %d, history: %d, total_score: %f",
						current_frame_id, best_matched_id_epsc, best_score_epsc);
		}
	}

	if (UsingSEPSCFlag) 
	{
		SEPSC_cur = calculateSEPSC(pc_filtered_semantic);
		SEPSCArr.push_back(SEPSC_cur);
		if (best_matched_id_sepsc != -1) 
		{
			matched_frame_id.push_back(best_matched_id_sepsc);
			matched_frame_transform.push_back(best_score_sepsc_transform);
			ROS_WARN("SEPSC: received loop closure candidate: current: %d, history: %d, total_score: %f",
						current_frame_id, best_matched_id_sepsc, best_score_sepsc);
		}
	}
}	
