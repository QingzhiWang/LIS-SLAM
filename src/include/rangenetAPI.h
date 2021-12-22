// Author of  EPSC-LOAM : QZ Wang
// Email wangqingzhi27@outlook.com

#ifndef RANGENETAPI_H_
#define RANGENETAPI_H_

#include <tuple>

#include "utility.h"
#include "common.h"

// net stuff
#include <selector.hpp>

namespace cl = rangenet::segmentation;


class RangenetAPI : public ParamServer, SemanticLabelParam {
 public:
  typedef std::tuple<u_char, u_char, u_char> semantic_color;

  RangenetAPI(const string& params);

  /** @brief      Get the point cloud **/
  std::vector<cv::Vec3f> getPointCloud(const std::vector<float>& scan) {
    return net->getPoints(scan, num_points);
  }

  /** @brief      Get the color mask **/
  std::vector<cv::Vec3b> getColorMask() {
    return net->getLabels(semantic_scan, num_points);
  }

  /** @brief      infer **/
  void infer(pcl::PointCloud<PointType>& currentCloudIn);

  /** @brief      Get semantic scan **/
  std::vector<std::vector<float>> getSemanticScan() { return semantic_scan; }

  /** @brief      Get Point Cloud With Label**/
  pcl::PointCloud<PointXYZIL>::Ptr getLabelPointCloud() {
    return semanticLabelCloud;
  }

  /** @brief      Get Point Cloud With RGB **/
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr getRGBPointCloud() {
    return semanticRGBCloud;
  }

  /** @brief      Get Point Cloud With RGB and Label **/
  pcl::PointCloud<pcl::PointXYZRGBL>::Ptr getSemanticCloud() {
    return semanticCloud;
  }

 protected:
  std::unique_ptr<cl::Net> net;

  std::vector<std::vector<float>> semantic_scan;
  uint32_t num_points;

  std::vector<uint32_t> labels;
  std::vector<cv::Vec3b> colors;

  pcl::PointCloud<PointXYZIL>::Ptr semanticLabelCloud;
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr semanticRGBCloud;
  pcl::PointCloud<pcl::PointXYZRGBL>::Ptr semanticCloud;
};

#endif /* RANGENETAPI_H_ */