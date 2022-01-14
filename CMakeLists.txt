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
	# msgs
	message_generation
	std_msgs
	sensor_msgs
	geometry_msgs
	nav_msgs
)

find_package(Threads REQUIRED)

## System dependencies are found with CMake's conventions
find_package(OpenMP REQUIRED)
include_directories(${OpenMP_FOUND_INCLUDE_DIRS})
message("OpenMP [OK]:" ${OpenMP_FOUND_INCLUDE_DIRS})

#Eigen (involved in pcl, not needed)
find_package(Eigen3 REQUIRED NO_MODULE QUIET)
message("Eigen3 [OK]")

find_package(PCL REQUIRED QUIET)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
message("PCL [OK]:" ${PCL_INCLUDE_DIRS})

find_package(GTSAM REQUIRED QUIET)
include_directories(${GTSAM_INCLUDE_DIR})
message("GTSAM [OK]: " ${GTSAM_INCLUDE_DIR})

find_package(OpenCV REQUIRED QUIET)
include_directories(${OpenCV_INCLUDE_DIRS})
message("OPENCV [OK]: " ${OpenCV_INCLUDE_DIRS})

#TEASER++
find_package(teaserpp REQUIRED QUIET)
include_directories(${teaserpp_INCLUDE_DIRS})
message("Teaser++ [OK]")


find_package (fast_gicp)
include_directories(${fast_gicp_INCLUDE_DIRS})
message("fast_gicp [OK]")
message("fast_gicp Libs: ${fast_gicp_LIBRARIES}")
message("fast_gicp Headers: ${fast_gicp_INCLUDE_DIRS}")


find_library(VGICP_CUDA_FOUND NAMES fast_vgicp_cuda)
message(STATUS "VGICP_CUDA_FOUND:" ${VGICP_CUDA_FOUND})
if(VGICP_CUDA_FOUND)
  	add_definitions(-DUSE_VGICP_CUDA)
endif()

# libyaml-cpp (required)
find_package (yaml-cpp)
if (yaml-cpp_FOUND)
	include_directories(${YAML_CPP_INCLUDE_DIR})
  	message("YAML Libs: ${YAML_CPP_LIBRARIES}")
  	message("YAML Headers: ${YAML_CPP_INCLUDE_DIR}")
else()
  	message(FATAL_ERROR "YAML_CPP NOT FOUND\n")
endif (yaml-cpp_FOUND)

# get boost (required)
find_package(Boost COMPONENTS timer thread program_options filesystem)
if(Boost_FOUND)
	include_directories(${Boost_INCLUDE_DIRS})
	message("Boost Libs: ${Boost_LIBRARIES}")
	message("Boost Headers: ${Boost_INCLUDE_DIRS}\n")
else()
 	message(FATAL_ERROR "Boost NOT FOUND\n")
endif()


# find_package(rangenet_lib)

#####################################
## TensorRT specific configuration ##
#####################################
# set(CMAKE_PREFIX_PATH "/home/wqz/TensorRT-5.1.5.0")
option(TENSORRT_FOUND "TensorRT installed on system?" OFF)

# first try to find cuda and tensorrt-related stuff (nvinfer, onnx, etc)
find_package(TensorRT PATHS cmake)
if (TENSORRT_FOUND)
	include_directories(${TENSORRT_INCLUDE_DIRS})
	message("TensorRT SUCCESS!\n")
	set(TENSORRT_FOUND ON)
else()
	message("TensorRT NOT FOUND\n")
	set(TENSORRT_FOUND OFF)
endif (TENSORRT_FOUND)

# complain if no backend is installed
if(NOT TENSORRT_FOUND)
	message(FATAL_ERROR "TensorRT must be installed TENSORRT_FOUND ${TENSORRT_FOUND}\n")
endif()


# include folder
include_directories(${PROJECT_SOURCE_DIR}/src/segnet)
include_directories(${PROJECT_SOURCE_DIR}/src/include)
include_directories(${PROJECT_SOURCE_DIR}/src/sophus)
include_directories(
	${catkin_INCLUDE_DIRS}
  	${rangenet_external_INCLUDE_DIRS}
)

set(DEP_LIBS ${DEP_LIBS} Threads::Threads)
set(DEP_LIBS ${DEP_LIBS} ${Boost_LIBRARIES})
set(DEP_LIBS ${DEP_LIBS} ${OpenMP_CXX_FLAGS})
set(DEP_LIBS ${DEP_LIBS} ${Eigen3_LIBS})
set(DEP_LIBS ${DEP_LIBS} ${PCL_LIBRARIES})
set(DEP_LIBS ${DEP_LIBS} ${OpenCV_LIBS})
set(DEP_LIBS ${DEP_LIBS} ${GTSAM_LIBRARIES} gtsam)
set(DEP_LIBS ${DEP_LIBS} teaserpp::teaser_registration teaserpp::teaser_features teaserpp::teaser_io ${TEASERPP_BLAS_LAPACK_LIBS})
# set(DEP_LIBS ${DEP_LIBS} ${fast_gicp_LIBRARIES})
set(DEP_LIBS ${DEP_LIBS} ${TENSORRT_LIBRARIES})
set(DEP_LIBS ${DEP_LIBS} ${YAML_CPP_LIBRARIES})


add_message_files(
	DIRECTORY msg
	FILES cloud_info.msg submap.msg semantic_info.msg
)

generate_messages(
	DEPENDENCIES geometry_msgs std_msgs nav_msgs sensor_msgs
)

###################################
## catkin specific configuration ##
###################################
catkin_package(
  	INCLUDE_DIRS src
  	LIBRARIES semantic_external semantic_lib
  	CATKIN_DEPENDS 
    	cv_bridge geometry_msgs pcl_conversions nav_msgs roscpp rospy sensor_msgs std_msgs tf message_generation
  	DEPENDS YAML_CPP
  	CFG_EXTRAS tensorrt-config.cmake
)

install(DIRECTORY cmake
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})

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
set(SEMANTIC_LIB_SOURCES src/segnet/net.cpp src/segnet/selector.cpp)

if(TENSORRT_FOUND)
  message("Building TensorRT")
  list(APPEND SEMANTIC_LIB_SOURCES src/segnet/netTensorRT.cpp)
endif()

# make the library and link stuff to it
add_library(semantic_lib ${SEMANTIC_LIB_SOURCES})
target_link_libraries(semantic_lib
    semantic_external
    ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBS} ${Boost_LIBRARIES}
    ${YAML_CPP_LIBRARIES} ${TENSORRT_LIBRARIES} ${Eigen3_LIBS}
)

###########
## Build ##
###########

# Laser Pretreatment  新增laser预处理
add_executable(${PROJECT_NAME}_laserPretreatment src/node/laserPretreatmentNode.cpp src/core/common.cpp)
add_dependencies(${PROJECT_NAME}_laserPretreatment ${catkin_EXPORTED_TARGETS} ${PROJECT_NAME}_generate_messages_cpp)
target_link_libraries(${PROJECT_NAME}_laserPretreatment ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} ${OpenMP_CXX_FLAGS})

# Laser Processing
add_executable(${PROJECT_NAME}_laserProcessing src/node/laserProcessingNode.cpp src/core/laserProcessing.cpp src/core/common.cpp)
add_dependencies(${PROJECT_NAME}_laserProcessing ${catkin_EXPORTED_TARGETS} ${PROJECT_NAME}_generate_messages_cpp)
target_link_libraries(${PROJECT_NAME}_laserProcessing ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} ${OpenMP_CXX_FLAGS})

# Odom Estimation
add_executable(${PROJECT_NAME}_odomEstimation src/node/odomEstimationNode.cpp src/core/common.cpp)
add_dependencies(${PROJECT_NAME}_odomEstimation ${catkin_EXPORTED_TARGETS} ${PROJECT_NAME}_generate_messages_cpp)
target_link_libraries(${PROJECT_NAME}_odomEstimation ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES}  ${OpenMP_CXX_FLAGS})

# Semantic Fusion
add_executable(${PROJECT_NAME}_semanticFusion src/node/semanticFusionNode.cpp src/core/rangenetAPI.cpp src/core/common.cpp)
add_dependencies(${PROJECT_NAME}_semanticFusion ${catkin_EXPORTED_TARGETS} ${PROJECT_NAME}_generate_messages_cpp)
target_link_libraries(${PROJECT_NAME}_semanticFusion ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} ${OpenMP_CXX_FLAGS} semantic_lib)

# Loop Closure / Make SubMap / Mapping Optimization
add_executable(${PROJECT_NAME}_subMapOptmization  src/node/subMapOptmizationNode.cpp src/core/epscGeneration.cpp src/core/common.cpp src/core/registration.cpp)
add_dependencies(${PROJECT_NAME}_subMapOptmization ${catkin_EXPORTED_TARGETS} ${PROJECT_NAME}_generate_messages_cpp)
# target_compile_options(${PROJECT_NAME}_subMapOptmization PRIVATE ${OpenMP_CXX_FLAGS})
target_link_libraries(${PROJECT_NAME}_subMapOptmization ${catkin_LIBRARIES} ${DEP_LIBS} fast_gicp)
