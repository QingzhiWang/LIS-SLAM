// Author of  EPSC-LOAM : QZ Wang
// Email wangqingzhi27@outlook.com

#include "rangenetAPI.h"

RangenetAPI::RangenetAPI(const string& params) {
  std::string model_path = params;
  std::string backend = "tensorrt";

  // initialize a network
  net = cl::make_net(model_path, backend);

  semanticRGBCloud.reset(new pcl::PointCloud<pcl::PointXYZRGB>());
  semanticLabelCloud.reset(new pcl::PointCloud<PointXYZIL>());
  semanticCloud.reset(new pcl::PointCloud<pcl::PointXYZRGBL>());
}

void RangenetAPI::infer(pcl::PointCloud<PointType>& currentCloudIn) {
  std::chrono::time_point<std::chrono::system_clock> start, end;
  start = std::chrono::system_clock::now();

  this->num_points = currentCloudIn.points.size();

  std::vector<float> values;

  for (size_t i = 0; i < num_points; i++) {
    values.push_back(currentCloudIn.points[i].x);
    values.push_back(currentCloudIn.points[i].y);
    values.push_back(currentCloudIn.points[i].z);
    values.push_back(currentCloudIn.points[i].intensity);
  }

  end = std::chrono::system_clock::now();
  std::chrono::duration<float> elapsed_seconds = end - start;
  float time_temp = elapsed_seconds.count() * 1000;
  ROS_INFO("Step 1 time %f ms", time_temp);

  start = std::chrono::system_clock::now();

  this->semantic_scan = net->infer(values, num_points);

  end = std::chrono::system_clock::now();
  elapsed_seconds = end - start;
  time_temp = elapsed_seconds.count() * 1000;
  ROS_INFO("Step 2 time %f ms", time_temp);

  start = std::chrono::system_clock::now();

  std::vector<int> label_map = net->getLabelMap();

  labels.clear();
  colors.clear();
  labels.resize(num_points);
  colors.resize(num_points);

  std::vector<float> labels_prob;
  labels_prob.resize(num_points);

  for (uint32_t i = 0; i < num_points; ++i) {
    labels_prob[i] = 0;
    for (int32_t j = 0; j < LearningSize; ++j) {
      if (labels_prob[i] <= semantic_scan[i][j]) {
        // labels[i] = label_map[j];
        labels[i] = j;
        colors[i] = cv::Vec3b(std::get<0>(Argmax2RGB[j]),
                              std::get<1>(Argmax2RGB[j]),
                              std::get<2>(Argmax2RGB[j]));
        labels_prob[i] = semantic_scan[i][j];
      }
    }
  }

  end = std::chrono::system_clock::now();
  elapsed_seconds = end - start;
  time_temp = elapsed_seconds.count() * 1000;
  ROS_INFO("Step 3 time %f ms", time_temp);

  start = std::chrono::system_clock::now();

  semanticRGBCloud->clear();
  semanticLabelCloud->clear();
  semanticCloud->clear();

  for (size_t i = 0; i < num_points; i++) {
    // pcl::PointXYZRGB p;
    
    // 剔除动态物体(假剔除，因为将静态的目标也剔除掉了) 目前仅剔除 car
    // if(color_mask[i][0] == 245 && color_mask[i][1] == 150 &&
    // color_mask[i][2] == 100)
    //     continue;

    // p.x = currentCloudIn.points[i].x;
    // p.y = currentCloudIn.points[i].y;
    // p.z = currentCloudIn.points[i].z;
    // p.b = colors[i][0];
    // p.g = colors[i][1];
    // p.r = colors[i][2];
    // semanticRGBCloud->points.push_back(p);

    PointXYZIL point;

    point.x = currentCloudIn.points[i].x;
    point.y = currentCloudIn.points[i].y;
    point.z = currentCloudIn.points[i].z;
    point.intensity = currentCloudIn.points[i].intensity;
    point.label = labels[i];
    semanticLabelCloud->points.push_back(point);

    pcl::PointXYZRGBL pointRGBL;
    pointRGBL.x = currentCloudIn.points[i].x;
    pointRGBL.y = currentCloudIn.points[i].y;
    pointRGBL.z = currentCloudIn.points[i].z;
    pointRGBL.b = colors[i][0];
    pointRGBL.g = colors[i][1];
    pointRGBL.r = colors[i][2];
    pointRGBL.label = labels[i];
    semanticCloud->points.push_back(pointRGBL);
  }

  end = std::chrono::system_clock::now();
  elapsed_seconds = end - start;
  time_temp = elapsed_seconds.count() * 1000;
  ROS_INFO("Step 4 time %f ms", time_temp);
}