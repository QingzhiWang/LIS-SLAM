// Author of  EPSC-LOAM : QZ Wang
// Email wangqingzhi27@outlook.com

#ifndef RANGENETAPI_H_
#define RANGENETAPI_H_

#include <opencv2/opencv.hpp>
#include <tuple>

#include "utility.h"

// net stuff
#include <selector.hpp>

namespace cl = rangenet::segmentation;

/** \brief A class of rangenet apis.
 *
 * \author Xieyuanli Chen
 */

class RangenetAPI {
 public:
  typedef std::tuple<u_char, u_char, u_char> semantic_color;

  RangenetAPI(){};
  RangenetAPI(const string& params) {
    std::string model_path;
    //   if (params.hasParam("model_path")) {
    //     model_path = std::string(params["model_path"]);
    //   }
    //   else{
    //     std::cerr << "No model could be found!" << std::endl;
    //   }
    model_path = params;
    std::string backend = "tensorrt";

    // initialize a network
    net = cl::make_net(model_path, backend);
  }

  /** @brief      Infer logits from LiDAR scan **/
  void infer(const std::vector<float>& scan, const uint32_t num_points) {
    this->semantic_scan = net->infer(scan, num_points);
    this->num_points = num_points;
  }

  /** @brief      Get the label map from rangenet_lib **/
  std::vector<int> getLabelMap() { return net->getLabelMap(); }

  /** @brief      Get the color map from rangenet_lib **/
  std::map<uint32_t, semantic_color> getColorMap() {
    return net->getColorMap();
  }

  /** @brief      Get the point cloud **/
  std::vector<cv::Vec3f> getPointCloud(const std::vector<float>& scan) {
    return net->getPoints(scan, num_points);
  }

  /** @brief      Get the color mask **/
  std::vector<cv::Vec3b> getColorMask() {
    return net->getLabels(semantic_scan, num_points);
  }

  /** @brief      Get semantic scan **/
  std::vector<std::vector<float>> getSemanticScan() { return semantic_scan; }

 protected:
  std::unique_ptr<cl::Net> net;
  std::vector<std::vector<float>> semantic_scan;
  uint32_t num_points;
};

#endif /* RANGENETAPI_H_ */