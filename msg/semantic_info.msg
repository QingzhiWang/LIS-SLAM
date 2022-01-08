# Cloud Info
Header header 

bool imuAvailable
bool odomAvailable

# Attitude for LOAM initialization
float32 imuRollInit
float32 imuPitchInit
float32 imuYawInit

# Initial guess from imu pre-integration
float32 initialGuessX
float32 initialGuessY
float32 initialGuessZ
float32 initialGuessRoll
float32 initialGuessPitch
float32 initialGuessYaw

# Point cloud messages
sensor_msgs/PointCloud2 semantic_raw

sensor_msgs/PointCloud2 semantic_outlier
sensor_msgs/PointCloud2 semantic_dynamic

sensor_msgs/PointCloud2 semantic_pole
sensor_msgs/PointCloud2 semantic_ground
sensor_msgs/PointCloud2 semantic_building
  
# extracted corner surface feature
sensor_msgs/PointCloud2 cloud_corner    
sensor_msgs/PointCloud2 cloud_surface

sensor_msgs/PointCloud2 cloud_corner_sharp    
sensor_msgs/PointCloud2 cloud_surface_sharp