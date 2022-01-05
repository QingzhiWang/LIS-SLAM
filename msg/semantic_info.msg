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
sensor_msgs/PointCloud2 cloud_semantic 
sensor_msgs/PointCloud2 cloud_outlier
sensor_msgs/PointCloud2 cloud_dynamic
sensor_msgs/PointCloud2 cloud_static_corner
sensor_msgs/PointCloud2 cloud_static_surface
  
# extracted corner surface feature
sensor_msgs/PointCloud2 cloud_corner    
sensor_msgs/PointCloud2 cloud_surface