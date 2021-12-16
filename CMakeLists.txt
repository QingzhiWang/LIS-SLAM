cmake_minimum_required(VERSION 2.8.3)
project(lis_slam)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++17")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g -pthread")
set(CMAKE_CXX_STANDARD 14)

## Find catkin macros and libraries
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  tf
  roscpp
  rospy
  pcl_ros
  cv_bridge
  pcl_conversions
  # rangenet_lib 
  # msgs
  message_generation
  std_msgs
  sensor_msgs
  geometry_msgs
  nav_msgs
)

## System dependencies are found with CMake's conventions
find_package(OpenMP REQUIRED)
find_package(PCL REQUIRED QUIET)
find_package(OpenCV REQUIRED QUIET)
find_package(GTSAM REQUIRED QUIET)

find_package(Boost REQUIRED COMPONENTS timer)

# find_package(rangenet_lib)



# libyaml-cpp (required)
find_package (yaml-cpp)
if (yaml-cpp_FOUND)
  message("YAML Libs: ${YAML_CPP_LIBRARIES}")
  message("YAML Headers: ${YAML_CPP_INCLUDE_DIR}")
else()
  message(FATAL_ERROR "yaml-cpp NOT FOUND\n")
endif (yaml-cpp_FOUND)

# get boost (required)
find_package(Boost COMPONENTS program_options filesystem)
if(Boost_FOUND)
  message("Boost Libs: ${Boost_LIBRARIES}")
  message("Boost Headers: ${Boost_INCLUDE_DIRS}\n")
else()
  message(FATAL_ERROR "Boost NOT FOUND\n")
endif()

####################################
## TensorRT specific configuration ##
####################################
# this package libs and includes for TensorRT

# set(CMAKE_PREFIX_PATH "/home/wqz/TensorRT-5.1.5.0")
option(TENSORRT_FOUND "TensorRT installed on system?" OFF)

# first try to find cuda and tensorrt-related stuff (nvinfer, onnx, etc)
find_package(TensorRT PATHS cmake)
if (TENSORRT_FOUND)
  message("TensorRT SUCCESS!\n")
  set(TENSORRT_FOUND ON)
else()
  message("TensorRT NOT FOUND\n")
  set(TENSORRT_FOUND OFF)
endif (TENSORRT_FOUND)

# complain if no backend is installed
if(NOT TENSORRT_FOUND)
  message(FATAL_ERROR
  "TensorRT must be installed
  TENSORRT_FOUND ${TENSORRT_FOUND}\n")
endif()


add_message_files(
  DIRECTORY msg
  FILES
  cloud_info.msg
  submap.msg
  loop_info.msg
  semantic_info.msg
)

generate_messages(
  DEPENDENCIES
  geometry_msgs
  std_msgs
  nav_msgs
  sensor_msgs
)

###################################
## catkin specific configuration ##
###################################
catkin_package(
  INCLUDE_DIRS
    src/segnet 
    src/include
  LIBRARIES
    semantic_external
    semantic_lib
  CATKIN_DEPENDS 
    cv_bridge geometry_msgs pcl_conversions nav_msgs roscpp rospy sensor_msgs std_msgs tf message_generation
  DEPENDS
    YAML_CPP
    PCL
    GTSAM
  CFG_EXTRAS
    tensorrt-config.cmake
)

#########################
## Install cmake finds ##
#########################
install(DIRECTORY cmake
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})

include_directories(
  src/segnet 
	src/include
	${PCL_INCLUDE_DIRS}
	${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
	${GTSAM_INCLUDE_DIR}
  ${YAML_CPP_INCLUDE_DIR}
  # ${rangenet_lib_INCLUDE_DIRS}   ####
  ${rangenet_external_INCLUDE_DIRS}
  ${TENSORRT_INCLUDE_DIRS}   ####
)

# turn config file into flags (sets TENSORRT_FOUND and TORCH_FOUND)
configure_file(src/segnet/external.hpp.in ${PROJECT_SOURCE_DIR}/src/segnet/external.hpp @ONLY)

# Mark cpp header files for installation
install(
  DIRECTORY src/segnet
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  FILES_MATCHING PATTERN "*.hpp"
)

# make wrapper lib
add_library(semantic_external src/segnet/external.cpp)

## Declare a cpp library
# define the network library (compiling tf and trt conditionally)
set(semantic_lib_SOURCES src/segnet/net.cpp src/segnet/selector.cpp)

if(TENSORRT_FOUND)
  message("Building TensorRT")
  list(APPEND semantic_lib_SOURCES src/segnet/netTensorRT.cpp)
endif()

# make the library and link stuff to it
add_library(semantic_lib ${semantic_lib_SOURCES})
target_link_libraries(semantic_lib
    semantic_external
    ${catkin_LIBRARIES}
    ${PCL_LIBRARIES}
    ${OpenCV_LIBS}
    ${GTSAM_INCLUDE_DIR}
    ${Boost_LIBRARIES}
    ${YAML_CPP_LIBRARIES}
    ${TENSORRT_LIBRARIES}
    )


# link directories
# link_directories(
# 	src/include
#   src/segnet 
# 	${PCL_LIBRARY_DIRS}
#   ${OpenCV_LIBRARY_DIRS}
#   ${GTSAM_LIBRARY_DIRS}
#   ${semantic_lib_SOURCES} ####
#   ${TENSORRT_LIBRARIES}  ####
# )



###########
## Build ##
###########

# Laser Pretreatment  新增laser预处理
add_executable(${PROJECT_NAME}_laserPretreatment src/node/laserPretreatmentNode.cpp)
add_dependencies(${PROJECT_NAME}_laserPretreatment ${catkin_EXPORTED_TARGETS} ${PROJECT_NAME}_generate_messages_cpp)
target_link_libraries(${PROJECT_NAME}_laserPretreatment ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES})

# Laser Processing
add_executable(${PROJECT_NAME}_laserProcessing src/node/laserProcessingNode.cpp src/core/laserProcessing.cpp)
add_dependencies(${PROJECT_NAME}_laserProcessing ${catkin_EXPORTED_TARGETS} ${PROJECT_NAME}_generate_messages_cpp)
target_link_libraries(${PROJECT_NAME}_laserProcessing ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES})

# Odom Estimation
add_executable(${PROJECT_NAME}_odomEstimation src/node/odomEstimationNode.cpp)
add_dependencies(${PROJECT_NAME}_odomEstimation ${catkin_EXPORTED_TARGETS} ${PROJECT_NAME}_generate_messages_cpp)
target_compile_options(${PROJECT_NAME}_odomEstimation PRIVATE ${OpenMP_CXX_FLAGS})
target_link_libraries(${PROJECT_NAME}_odomEstimation ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES}  ${OpenMP_CXX_FLAGS} gtsam)

# Semantic Fusion
add_executable(${PROJECT_NAME}_semanticFusion src/node/semanticFusionNode.cpp src/core/rangenetAPI.cpp)
add_dependencies(${PROJECT_NAME}_semanticFusion ${catkin_EXPORTED_TARGETS} ${PROJECT_NAME}_generate_messages_cpp)
target_compile_options(${PROJECT_NAME}_semanticFusion PRIVATE ${OpenMP_CXX_FLAGS})
target_link_libraries(${PROJECT_NAME}_semanticFusion ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} ${OpenMP_CXX_FLAGS} semantic_lib)

# Loop Closure / Make SubMap / Mapping Optimization
add_executable(${PROJECT_NAME}_subMapOptmization  src/node/subMapOptmizationNode.cpp src/core/epscGeneration.cpp)
add_dependencies(${PROJECT_NAME}_subMapOptmization ${catkin_EXPORTED_TARGETS} ${PROJECT_NAME}_generate_messages_cpp)
target_compile_options(${PROJECT_NAME}_subMapOptmization PRIVATE ${OpenMP_CXX_FLAGS})
target_link_libraries(${PROJECT_NAME}_subMapOptmization ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} ${OpenMP_CXX_FLAGS} ${Boost_LIBRARIES} ${YAML_CPP_LIBRARIES} gtsam)


# IMU Preintegration
# add_executable(${PROJECT_NAME}_imuPreintegration src/imuPreintegration.cpp)
# target_link_libraries(${PROJECT_NAME}_imuPreintegration ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} gtsam)

# Transform Fusion
# add_executable(${PROJECT_NAME}_transformFusion src/transformFusion.cpp)
# target_link_libraries(${PROJECT_NAME}_transformFusion ${catkin_LIBRARIES}   )