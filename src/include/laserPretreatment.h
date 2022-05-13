
#ifndef _LASER_PRETREATMENT_H_
#define _LASER_PRETREATMENT_H_

#include "utility.h"
#include "common.h"

using std::atan2;
using std::cos;
using std::sin;

const double scanPeriod = 0.1;

struct PointIn {
    PCL_ADD_POINT4D;
	float i;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT(
    PointIn,
    (float, x, x)(float, y, y)(float, z, z)(float, i, i))


template <typename PointT>
void removeClosedPointCloud(const pcl::PointCloud<PointT>& cloud_in,
							pcl::PointCloud<PointT>& cloud_out,
							float minthres, float maxthres) 
{
	if (&cloud_in != &cloud_out) 
	{
		cloud_out.header = cloud_in.header;
		cloud_out.points.resize(cloud_in.points.size());
	}

	size_t j = 0;

	for (size_t i = 0; i < cloud_in.points.size(); ++i) 
	{
		float thisRange = cloud_in.points[i].x * cloud_in.points[i].x + cloud_in.points[i].y * cloud_in.points[i].y + cloud_in.points[i].z * cloud_in.points[i].z;
		if (thisRange < minthres * minthres) continue;
		if (thisRange > maxthres * maxthres) continue;
		cloud_out.points[j] = cloud_in.points[i];
		j++;
	}

	if (j != cloud_in.points.size()) {
		cloud_out.points.resize(j);
	}

	cloud_out.height = 1;
	cloud_out.width = static_cast<uint32_t>(j);
	cloud_out.is_dense = true;
}


class LaserPretreatment : public ParamServer 
{
private:

public:
	LaserPretreatment() {
	}
	~LaserPretreatment(){}

	pcl::PointCloud<PointXYZIRT>::Ptr Pretreatment(pcl::PointCloud<PointIn>::Ptr& cloudIn);
	pcl::PointCloud<PointXYZIRT>::Ptr Pretreatment(pcl::PointCloud<PointType>::Ptr& cloudIn);

};


#endif  // _FEATURE_EXTRACTION_H_