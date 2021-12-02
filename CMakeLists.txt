cmake_minimum_required(VERSION 2.8.3)
project(lis_slam)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++17")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g -pthread")
set(CMAKE_CXX_STANDARD 14)

find_package(catkin REQUIRED COMPONENTS
  tf
  roscpp
  rospy
  pcl_ros
  cv_bridge
  # pcl library
  pcl_conversions
  # rangenet++ ###
  rangenet_lib  
  # msgs
  std_msgs
  sensor_msgs
  geometry_msgs
  nav_msgs
  message_generation
)

find_package(OpenMP REQUIRED)
find_package(PCL REQUIRED QUIET)
find_package(OpenCV REQUIRED QUIET)
find_package(GTSAM REQUIRED QUIET)

# include directories
include_directories(
	include
	${PCL_INCLUDE_DIRS}
	${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
	${GTSAM_INCLUDE_DIR}
  ${rangenet_lib_INCLUDE_DIRS}   ####
  ${TENSORRT_INCLUDE_DIRS}   ####
)

# link directories
link_directories(
	include
	${PCL_LIBRARY_DIRS}
  ${OpenCV_LIBRARY_DIRS}
  ${GTSAM_LIBRARY_DIRS}
  ${rangenet_lib_LIBRARIES} ####
)

add_message_files(
  DIRECTORY msg
  FILES
  cloud_info.msg
  submap.msg
  loop_info.msg
  loop_container.msg
)



generate_messages(
  DEPENDENCIES
  geometry_msgs
  std_msgs
  nav_msgs
  sensor_msgs
)

catkin_package(
  INCLUDE_DIRS include
  DEPENDS PCL GTSAM

  CATKIN_DEPENDS 
  pcl_ros
  std_msgs
  nav_msgs
  geometry_msgs
  sensor_msgs
  message_runtime 
  message_generation
)



###########
## Build ##
###########

# Laser Pretreatment  新增laser预处理
add_executable(${PROJECT_NAME}_laserPretreatment src/laserPretreatment.cpp)
add_dependencies(${PROJECT_NAME}_laserPretreatment ${catkin_EXPORTED_TARGETS} ${PROJECT_NAME}_generate_messages_cpp)
target_link_libraries(${PROJECT_NAME}_laserPretreatment ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES})

# Laser Pretreatment  新增laser预处理
add_executable(${PROJECT_NAME}_makeSubMap src/makeSubMap.cpp)
add_dependencies(${PROJECT_NAME}_makeSubMap ${catkin_EXPORTED_TARGETS} ${PROJECT_NAME}_generate_messages_cpp)
target_compile_options(${PROJECT_NAME}_makeSubMap PRIVATE ${OpenMP_CXX_FLAGS})
target_link_libraries(${PROJECT_NAME}_makeSubMap ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES}  ${OpenMP_CXX_FLAGS} gtsam)


# Range Image Projection
add_executable(${PROJECT_NAME}_imageProjection src/imageProjection.cpp)
add_dependencies(${PROJECT_NAME}_imageProjection ${catkin_EXPORTED_TARGETS} ${PROJECT_NAME}_generate_messages_cpp)
target_link_libraries(${PROJECT_NAME}_imageProjection ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES})

# Feature Association
add_executable(${PROJECT_NAME}_featureExtraction src/featureExtraction.cpp)
add_dependencies(${PROJECT_NAME}_featureExtraction ${catkin_EXPORTED_TARGETS} ${PROJECT_NAME}_generate_messages_cpp)
target_link_libraries(${PROJECT_NAME}_featureExtraction ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES})

# Mapping Optimization
add_executable(${PROJECT_NAME}_mapOptmization src/mapOptmization.cpp)
add_dependencies(${PROJECT_NAME}_mapOptmization ${catkin_EXPORTED_TARGETS} ${PROJECT_NAME}_generate_messages_cpp)
target_compile_options(${PROJECT_NAME}_mapOptmization PRIVATE ${OpenMP_CXX_FLAGS})
target_link_libraries(${PROJECT_NAME}_mapOptmization ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} ${OpenMP_CXX_FLAGS} gtsam)

# Loop Closure
add_executable(${PROJECT_NAME}_loopClosure  src/loopClosure.cpp src/Scancontext.cpp  src/epscGenerationClass.cpp)
add_dependencies(${PROJECT_NAME}_loopClosure ${catkin_EXPORTED_TARGETS} ${PROJECT_NAME}_generate_messages_cpp)
target_link_libraries(${PROJECT_NAME}_loopClosure ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} )

# IMU Preintegration
add_executable(${PROJECT_NAME}_imuPreintegration src/imuPreintegration.cpp)
target_link_libraries(${PROJECT_NAME}_imuPreintegration ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} gtsam)

add_executable(${PROJECT_NAME}_transformFusion src/transformFusion.cpp)
target_link_libraries(${PROJECT_NAME}_transformFusion ${catkin_LIBRARIES}   )