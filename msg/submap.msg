# Cloud Info
Header header 

int64 isOptmization
int64 poseAvailable

uint64 subMapId

float32 subMapPoseX
float32 subMapPoseY
float32 subMapPoseZ
float32 subMapPoseRoll
float32 subMapPosePitch
float32 subMapPoseYaw


float32 afterOptmizationSubMapPoseX
float32 afterOptmizationSubMapPoseY
float32 afterOptmizationSubMapPoseZ
float32 afterOptmizationSubMapPoseRoll
float32 afterOptmizationSubMapPosePitch
float32 afterOptmizationSubMapPoseYaw
# Point cloud messages
sensor_msgs/PointCloud2 submap_raw  # original cloud deskewed
sensor_msgs/PointCloud2 submap_corner    # extracted corner feature
sensor_msgs/PointCloud2 submap_surface   # extracted surface feature