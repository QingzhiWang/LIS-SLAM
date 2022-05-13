# LIS-SLAM

## Advanced implementation of EPSC-LOAM

通过语义信息辅助的 LiDAR/IMU 融合位姿估计方法、语义信息融合的回环检测方法和基于局部子图 (SubMap) 的全局优化方法，实现了精确、稳定的激光 SLAM 算法框架 LIS-SLAM。

- 基于几何特征的点云匹配算法（LOAM），通过引入语义信息（RangeNet++）辅助提升特征关联的稳定性，并利用语义信息为每个误差项附加相应的权重优化点云匹配；同时，为了增强处理非结构化及退化场景的能力，在其基础上实现了 LiDAR/IMU 紧耦合的融合位姿估计。

- 针对基于扫描上下文 (Scan Context) 的回环检测方法在室外非结构化环境中性能下降问题，提出两种改进的回环检测方法。 首先，通过利用边缘平面特征的统计信息构建全局描述子（EPSC）提高对噪声数据的处理能力，从而增强在非结构化环境中的性能。其次，利用语义信息实现场景快速定位，解决由于视点变化引起的匹配问题，并在完成点云初始配准的基础上通过提取语义信息融合的全局描述子（SEPSC），进一步完成场景的相似性匹配。

- 通过构建 SubMap 并将其作为优化单元，利用因子图优化方法实现全局位姿优化，一方面有效解决了室外大规模场景中后端优化计算效率下降的问题，另外通过利用 SubMap 中更加完整的特征信息实现多层级的点云匹配，进一步提升位姿估计的精度。

**Modifier:** [QZ Wang](http://www.wang.qingzhi@outlook.com)

## 1. System architecture

<p align='center'>
    <img src="./assets/doc/system.png" alt="drawing" width="800"/>
</p>

## 2. Prerequisites
### 2.1 **Ubuntu** and **ROS**
Ubuntu 64-bit 16.04 or 18.04.
ROS Kinetic or Melodic. [ROS Installation](http://wiki.ros.org/ROS/Installation)

### 2.2 **GTSAM**
Follow [GTSAM Installation](https://github.com/borglab/gtsam/releases).
  ```
  wget -O ~/Downloads/gtsam.zip https://github.com/borglab/gtsam/archive/4.0.2.zip
  cd ~/Downloads/ && unzip gtsam.zip -d ~/Downloads/
  cd ~/Downloads/gtsam-4.0.2/
  mkdir build && cd build
  cmake -DGTSAM_BUILD_WITH_MARCH_NATIVE=OFF ..
  sudo make install -j8
  ```

### 2.3 **PCL**
Follow [PCL Installation](http://www.pointclouds.org/downloads/linux.html).


### 2.4 **RangeNet++**



## 3. Build EPSC-LOAM
Clone the repository and catkin_make:

  ```
  cd ~/catkin_ws/src
  git clone https://gitee.com/QZ_Wang/epsc_laom.git
  cd ../
  catkin_make
  source ~/catkin_ws/devel/setup.bash
  ```

## 4. Prepare test data
### 4.1 Laser data
  - The conversion of laser data is provided in laserPretreatment.cpp. You only need to modify N_Scan and horizon_SCAN of your 3D Lidar in params.yaml.

### 4.2 IMU data
  - **IMU alignment**. EPSC-SAM transforms IMU raw data from the IMU frame to the Lidar frame, which follows the ROS REP-105 convention (x - forward, y - left, z - upward). To make the system function properly, the correct extrinsic transformation（"extrinsicRot" and "extrinsicRPY"） needs to be provided in "params.yaml" file. 

## 5. Your datasets
Modify related parameters in params.yawl.

  ```
  roslaunch epsc_loam run.launch
  rosbag play YOUR_DATASET_FOLDER/your-bag.bag
  ```


## 6. KITTI Example (Velodyne HDL-64)
Download [KITTI Odometry dataset](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) to YOUR_DATASET_FOLDER and convert KITTI dataset to bag file. 
Modify related parameters in params.yawl.

  ```
  roslaunch epsc_loam run.launch
  rosbag play YOUR_DATASET_FOLDER/your-bag.bag
  ```

## 7.Acknowledgements
LIS-SLAM is based on LOAM(J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time) LIO-SAM,SC-LEGO-LAOM and iscloam.


