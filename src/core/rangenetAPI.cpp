#include "RangenetAPI.h"

RangenetAPI::RangenetAPI(const std::string& params) {
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

std::vector<std::vector<float>> RangenetAPI::infer(
    const std::vector<float>& scan, const uint32_t num_points) {
  return net->infer(scan, num_points);
}
