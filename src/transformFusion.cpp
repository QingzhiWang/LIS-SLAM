
// Author of  EPSC-LOAM : QZ Wang  
// Email wangqingzhi27@outlook.com


#include "utility.h"

class TransformFusion : public ParamServer
{

private:

   int init_flag=true; //ADD
   Eigen::Matrix4f H;  //ADD
   Eigen::Matrix4f H_init; //ADD
   Eigen::Matrix4f H_rot; //ADD

   Eigen::Matrix4f lastR;
   Eigen::Matrix4f lastH;

//    std::string RESULT_PATH="/home/wqz/paperTest/my_epsc_v2_sync_pitch/08_pred.txt"; //ADD


    std::mutex mtx;

    ros::Subscriber subImuOdometry;
    ros::Subscriber subSubMapOdometry;

    ros::Publisher pubOdometry;
    ros::Publisher pubLocalPath;
    ros::Publisher pubOdometryIncremental;

    Eigen::Affine3f lidarOdomAftMappedAffine;
    Eigen::Affine3f lidarOdomBefMappedAffine;
    // Eigen::Affine3f imuOdomAffine;


    tf::TransformListener tfListener;
    tf::StampedTransform lidar2Baselink;

    double lidarOdomTime = -1;
    deque<nav_msgs::Odometry> imuOdomQueue;

    float transformBefMapped[6];
    float transformAftMapped[6];

public:


    TransformFusion()
    {
        if(lidarFrame != baselinkFrame)
        {
            try
            {
                tfListener.waitForTransform(lidarFrame, baselinkFrame, ros::Time(0), ros::Duration(3.0));
                tfListener.lookupTransform(lidarFrame, baselinkFrame, ros::Time(0), lidar2Baselink);
            }
            catch (tf::TransformException ex)
            {
                ROS_ERROR("%s",ex.what());
            }
        }


        // if(useImu==true){
        //     subImuOdometry   = nh.subscribe<nav_msgs::Odometry>(odomTopic,   2000, &TransformFusion::imuOdometryHandler,   this);

        // }else{
        //     subImuOdometry   = nh.subscribe<nav_msgs::Odometry>("lis_slam/submap/odometry",   2000, &TransformFusion::imuOdometryHandler,   this);
        // }

        subImuOdometry   = nh.subscribe<nav_msgs::Odometry>("lis_slam/submap/odometry",   2000, &TransformFusion::imuOdometryHandler,   this);
        subSubMapOdometry = nh.subscribe<nav_msgs::Odometry>("lis_slam/mapping/odometry", 5, &TransformFusion::subMapOdometryHandler, this);
        

        pubOdometry   = nh.advertise<nav_msgs::Odometry>("odometry/fusion", 2000);
        pubLocalPath       = nh.advertise<nav_msgs::Path>    ("lis_slam/local/path", 1);
        pubOdometryIncremental = nh.advertise<nav_msgs::Odometry>("odometry/fusion_incremental", 2000);
    

        for (int i = 0; i < 6; ++i)
        {
            transformBefMapped[i] = 0;
            transformAftMapped[i] = 0;
        }
        lidarOdomAftMappedAffine = trans2Affine3f(transformAftMapped);
        lidarOdomBefMappedAffine = trans2Affine3f(transformBefMapped);
    
    }

    Eigen::Affine3f odom2affine(nav_msgs::Odometry odom)
    {
        double x, y, z, roll, pitch, yaw;
        x = odom.pose.pose.position.x;
        y = odom.pose.pose.position.y;
        z = odom.pose.pose.position.z;
        tf::Quaternion orientation;
        tf::quaternionMsgToTF(odom.pose.pose.orientation, orientation);
        tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);
        return pcl::getTransformation(x, y, z, roll, pitch, yaw);
    }

    Eigen::Affine3f trans2Affine3f(float transformIn[])
    {
        return pcl::getTransformation(transformIn[3], transformIn[4], transformIn[5], transformIn[0], transformIn[1], transformIn[2]);
    }


    void subMapOdometryHandler(const nav_msgs::Odometry::ConstPtr& odomMsg)
    {
        std::lock_guard<std::mutex> lock(mtx);

        double roll, pitch, yaw;
        geometry_msgs::Quaternion geoQuat = odomMsg->pose.pose.orientation;
        tf::Matrix3x3(tf::Quaternion(geoQuat.x, geoQuat.y, geoQuat.z, geoQuat.w)).getRPY(roll, pitch, yaw);

        transformAftMapped[0] = roll;
        transformAftMapped[1] = pitch;
        transformAftMapped[2] = yaw;

        transformAftMapped[3] = odomMsg->pose.pose.position.x;
        transformAftMapped[4] = odomMsg->pose.pose.position.y;
        transformAftMapped[5] = odomMsg->pose.pose.position.z;

        transformBefMapped[0] = odomMsg->twist.twist.angular.x;
        transformBefMapped[1] = odomMsg->twist.twist.angular.y;
        transformBefMapped[2] = odomMsg->twist.twist.angular.z;

        transformBefMapped[3] = odomMsg->twist.twist.linear.x;
        transformBefMapped[4] = odomMsg->twist.twist.linear.y;
        transformBefMapped[5] = odomMsg->twist.twist.linear.z;


        lidarOdomAftMappedAffine = trans2Affine3f(transformAftMapped);
        lidarOdomBefMappedAffine = trans2Affine3f(transformBefMapped);

        lidarOdomTime = odomMsg->header.stamp.toSec();
    }



    void imuOdometryHandler(const nav_msgs::Odometry::ConstPtr& odomMsg)
    {
        // static tf
        static tf::TransformBroadcaster tfMap2Odom;
        static tf::Transform map_to_odom = tf::Transform(tf::createQuaternionFromRPY(0, 0, 0), tf::Vector3(0, 0, 0));
        tfMap2Odom.sendTransform(tf::StampedTransform(map_to_odom, odomMsg->header.stamp, mapFrame, odometryFrame));

        std::lock_guard<std::mutex> lock(mtx);

        imuOdomQueue.push_back(*odomMsg);
        // get latest odometry (at current IMU stamp)
        // if (lidarOdomTime == -1)
        //     return;

        while (!imuOdomQueue.empty())
        {
            if (imuOdomQueue.front().header.stamp.toSec() <= lidarOdomTime)
            {
                // ROS_INFO("IMU Odom message too old!");
                imuOdomQueue.pop_front();
            }
            else
                break;
        }

        if(imuOdomQueue.empty()&&imuOdomQueue.back().header.stamp.toSec() <= lidarOdomTime)
        {
            ROS_WARN("IMU Odom message too old!");
            return;
        }

        Eigen::Affine3f imuOdomAffine = odom2affine(imuOdomQueue.back());
    
        Eigen::Affine3f imuOdomAffineLast = lidarOdomAftMappedAffine*lidarOdomBefMappedAffine.inverse()* imuOdomAffine;
        

        float x, y, z, roll, pitch, yaw;
        pcl::getTranslationAndEulerAngles(imuOdomAffineLast, x, y, z, roll, pitch, yaw);


        // publish latest odometry
        nav_msgs::Odometry laserOdometry = imuOdomQueue.back();
        laserOdometry.pose.pose.position.x = x;
        laserOdometry.pose.pose.position.y = y;
        laserOdometry.pose.pose.position.z = z;
        laserOdometry.pose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(roll, pitch, yaw);
        pubOdometry.publish(laserOdometry);


/////////////////////added, cout results///////////////////


	Eigen::Quaterniond q;

	q.w()=laserOdometry.pose.pose.orientation.w;
	q.x()=laserOdometry.pose.pose.orientation.x;
	q.y()=laserOdometry.pose.pose.orientation.y;
	q.z()=laserOdometry.pose.pose.orientation.z;

	Eigen::Matrix3d R = q.toRotationMatrix();

	if (init_flag==true)	
	{
        H_init<< R.row(0)[0],R.row(0)[1],R.row(0)[2],x,
                R.row(1)[0],R.row(1)[1],R.row(1)[2],y,
                R.row(2)[0],R.row(2)[1],R.row(2)[2],z,
                0,0,0,1;  

        // H_rot<<	0,0,-1,0,
        //                    0,-1,0,0,
        //                     1,0,0,0,	
        //                     0,0,0,1; 
        // H_rot<<	4.276802385584e-04, -9.999672484946e-01,-8.084491683471e-03, -1.198459927713e-02,
        //                  -7.210626507497e-03, 8.081198471645e-03,-9.999413164504e-01, -5.403984729748e-02,
        //                  9.999738645903e-01, 4.859485810390e-04, -7.206933692422e-03, -2.921968648686e-01,
        //                  0,0,0,1; 

        // H_rot<<	4.276802385584e-04, -9.999672484946e-01,-8.084491683471e-03, 0,
        //             -7.210626507497e-03, 8.081198471645e-03,-9.999413164504e-01, 0,
        //             9.999738645903e-01, 4.859485810390e-04, -7.206933692422e-03, 0,
        //             0,0,0,1; 
        H_rot<<1, 0,0, 0,
                        0, 1,0, 0,
                       0, 0, 1, 0,
                        0,0,0,1; 

        lastR<< H_init.row(0)[0],H_init.row(0)[1],H_init.row(0)[2],H_init.row(0)[3],
                H_init.row(1)[0],H_init.row(1)[1],H_init.row(1)[2],H_init.row(1)[3],
                H_init.row(2)[0],H_init.row(2)[1],H_init.row(2)[2],H_init.row(2)[3],
                0,0,0,1;
        lastH<< H_rot.row(0)[0],H_rot.row(0)[1],H_rot.row(0)[2],H_rot.row(0)[3],
                H_rot.row(1)[0],H_rot.row(1)[1],H_rot.row(1)[2],H_rot.row(1)[3],
                H_rot.row(2)[0],H_rot.row(2)[1],H_rot.row(2)[2],H_rot.row(2)[3],
                0,0,0,1;

        init_flag=false;
        std::cout<<"surf_th : "<<surfThreshold<<endl;

 	}

	// Eigen::Quaterniond q1;

	// q1.w()=0.5;
	// q1.x()=0.5;
	// q1.y()=-0.5;
	// q1.z()=0.5;

	// Eigen::Matrix3d R1 = q1.toRotationMatrix();
	// H_rot<<R1.row(0)[0],R1.row(0)[1],R1.row(0)[2],0,
    //             R1.row(1)[0],R1.row(1)[1],R1.row(1)[2],0,
    //             R1.row(2)[0],R1.row(2)[1],R1.row(2)[2],0,
    //             0,0,0,1;  

    // H_rot<<	-1,0,0,0,
    //         0,-1,0,0,
    //         0,0,1,0,	
    //             0,0,0,1; 

    // H_rot<<	1,0,0,0,
    //         0,1,0,0,
    //         0,0,1,0,	
    //             0,0,0,1; 
		
	H<<  R.row(0)[0],R.row(0)[1],R.row(0)[2],x,
	          R.row(1)[0],R.row(1)[1],R.row(1)[2],y,
     	      R.row(2)[0],R.row(2)[1],R.row(2)[2],z,
     	      0,0,0,1;  


    // H = H_rot*H_init.inverse()*H; //to get H12 = H10*H02 , 180 rot according to z axis
    
    Eigen::Matrix4f HOut;
    HOut=lastH*lastR.inverse()*H;
    lastR=H;
    lastH=HOut;
    lastR<< H.row(0)[0],H.row(0)[1],H.row(0)[2],H.row(0)[3],
            H.row(1)[0],H.row(1)[1],H.row(1)[2],H.row(1)[3],
            H.row(2)[0],H.row(2)[1],H.row(2)[2],H.row(2)[3],
            0,0,0,1;
    lastH<< HOut.row(0)[0],HOut.row(0)[1],HOut.row(0)[2],HOut.row(0)[3],
            HOut.row(1)[0],HOut.row(1)[1],HOut.row(1)[2],HOut.row(1)[3],
            HOut.row(2)[0],HOut.row(2)[1],HOut.row(2)[2],HOut.row(2)[3],
            0,0,0,1;

	std::ofstream foutC(RESULT_PATH, std::ios::app);

	foutC.setf(std::ios::scientific, std::ios::floatfield);
    foutC.precision(6);
 
	//foutC << R[0] << " "<<transformMapped[3]<<" "<< R.row(1) <<" "<<transformMapped[4] <<" "<<  R.row(2) <<" "<< transformMapped[5] << endl;
	 for (int i = 0; i < 3; ++i)	
	{	 
		for (int j = 0; j < 4; ++j)
        	{
			if(i==2 && j==3)
			{
				foutC <<HOut.row(i)[j]<< endl ;	
			}
			else
			{
				foutC <<HOut.row(i)[j]<< " " ;
			}
			
		}
	}

	foutC.close();


//////////////////////////////////////////////////        

        // publish tf
        static tf::TransformBroadcaster tfOdom2BaseLink;
        tf::Transform tCur;
        tf::poseMsgToTF(laserOdometry.pose.pose, tCur);
        if(lidarFrame != baselinkFrame)
            tCur = tCur * lidar2Baselink;
        tf::StampedTransform odom_2_baselink = tf::StampedTransform(tCur, odomMsg->header.stamp, odometryFrame, baselinkFrame);
        tfOdom2BaseLink.sendTransform(odom_2_baselink);


        // Publish odometry for ROS (incremental)
        static bool lastIncreOdomPubFlag = false;
        static nav_msgs::Odometry laserOdomIncremental; // incremental odometry msg
        static Eigen::Affine3f increOdomAffine; // incremental odometry in affine
        if (lastIncreOdomPubFlag == false)
        {
            lastIncreOdomPubFlag = true;
            laserOdomIncremental = laserOdometry;
            increOdomAffine = imuOdomAffineLast;

        } else {
            // Eigen::Affine3f affineIncre = increOdomAffine.inverse() * imuOdomAffineLast;
            Eigen::Affine3f affineIncre = increOdomAffine.inverse() * imuOdomAffine;
            
            increOdomAffine = increOdomAffine * affineIncre;
            float x, y, z, roll, pitch, yaw;
            pcl::getTranslationAndEulerAngles (increOdomAffine, x, y, z, roll, pitch, yaw);
            

            laserOdomIncremental = imuOdomQueue.back();
            laserOdomIncremental.pose.pose.position.x = x;
            laserOdomIncremental.pose.pose.position.y = y;
            laserOdomIncremental.pose.pose.position.z = z;
            laserOdomIncremental.pose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(roll, pitch, yaw);
        }
        pubOdometryIncremental.publish(laserOdomIncremental);


        // publish IMU path
        static nav_msgs::Path localPath;
        static double last_path_time = -1;
        double imuTime = imuOdomQueue.back().header.stamp.toSec();
        if (imuTime - last_path_time > 0.1)
        {
            last_path_time = imuTime;
            geometry_msgs::PoseStamped pose_stamped;
            pose_stamped.header.stamp = imuOdomQueue.back().header.stamp;
            pose_stamped.header.frame_id = odometryFrame;
            pose_stamped.pose = laserOdometry.pose.pose;
            localPath.poses.push_back(pose_stamped);
            while(!localPath.poses.empty()  && lidarOdomTime!=-1 && localPath.poses.front().header.stamp.toSec() < lidarOdomTime)
                localPath.poses.erase(localPath.poses.begin());
            if (pubLocalPath.getNumSubscribers() != 0)
            {
                localPath.header.stamp = imuOdomQueue.back().header.stamp;
                localPath.header.frame_id = odometryFrame;
                pubLocalPath.publish(localPath);
            }
        }

    }

void transformFusionThread(){
    ros::Rate rate(5);

    while(ros::ok()){

            //get latest odometry (at current IMU stamp)
            // if (lidarOdomTime == -1)
            //     continue;
             if(imuOdomQueue.empty())
                continue;

            while (!imuOdomQueue.empty())
            {
                if (imuOdomQueue.front().header.stamp.toSec() <= lidarOdomTime)
                {
                    // ROS_INFO("IMU Odom message too old!");
                    imuOdomQueue.pop_front();
                }
                else
                    break;
            }

            nav_msgs::Odometry  imuOdomLatest=imuOdomQueue.back();

            if(imuOdomQueue.empty()&&imuOdomLatest.header.stamp.toSec() <= lidarOdomTime)
            {
                ROS_WARN("IMU Odom message too old!");
                continue;
            }

            Eigen::Affine3f imuOdomAffine = odom2affine(imuOdomLatest);
        
            Eigen::Affine3f imuOdomAffineLast = lidarOdomAftMappedAffine*lidarOdomBefMappedAffine.inverse()* imuOdomAffine;
            

            float x, y, z, roll, pitch, yaw;
            pcl::getTranslationAndEulerAngles(imuOdomAffineLast, x, y, z, roll, pitch, yaw);


            // publish latest odometry
            nav_msgs::Odometry laserOdometry = imuOdomLatest;
            laserOdometry.pose.pose.position.x = x;
            laserOdometry.pose.pose.position.y = y;
            laserOdometry.pose.pose.position.z = z;
            laserOdometry.pose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(roll, pitch, yaw);
            pubOdometry.publish(laserOdometry);


    /////////////////////added, cout results///////////////////


        Eigen::Quaterniond q;

        q.w()=laserOdometry.pose.pose.orientation.w;
        q.x()=laserOdometry.pose.pose.orientation.x;
        q.y()=laserOdometry.pose.pose.orientation.y;
        q.z()=laserOdometry.pose.pose.orientation.z;

        Eigen::Matrix3d R = q.toRotationMatrix();

        if (init_flag==true)	
        {
            H_init<< R.row(0)[0],R.row(0)[1],R.row(0)[2],x,
                    R.row(1)[0],R.row(1)[1],R.row(1)[2],y,
                    R.row(2)[0],R.row(2)[1],R.row(2)[2],z,
                    0,0,0,1;  

            // H_rot<<	0,0,-1,0,
            //                    0,-1,0,0,
            //                     1,0,0,0,	
            //                     0,0,0,1; 
            // H_rot<<	4.276802385584e-04, -9.999672484946e-01,-8.084491683471e-03, -1.198459927713e-02,
            //                  -7.210626507497e-03, 8.081198471645e-03,-9.999413164504e-01, -5.403984729748e-02,
            //                  9.999738645903e-01, 4.859485810390e-04, -7.206933692422e-03, -2.921968648686e-01,
            //                  0,0,0,1; 

            // H_rot<<	4.276802385584e-04, -9.999672484946e-01,-8.084491683471e-03, 0,
            //             -7.210626507497e-03, 8.081198471645e-03,-9.999413164504e-01, 0,
            //             9.999738645903e-01, 4.859485810390e-04, -7.206933692422e-03, 0,
            //             0,0,0,1; 
            H_rot<<1, 0,0, 0,
                            0, 1,0, 0,
                        0, 0, 1, 0,
                            0,0,0,1; 

            lastR<< H_init.row(0)[0],H_init.row(0)[1],H_init.row(0)[2],H_init.row(0)[3],
                    H_init.row(1)[0],H_init.row(1)[1],H_init.row(1)[2],H_init.row(1)[3],
                    H_init.row(2)[0],H_init.row(2)[1],H_init.row(2)[2],H_init.row(2)[3],
                    0,0,0,1;
            lastH<< H_rot.row(0)[0],H_rot.row(0)[1],H_rot.row(0)[2],H_rot.row(0)[3],
                    H_rot.row(1)[0],H_rot.row(1)[1],H_rot.row(1)[2],H_rot.row(1)[3],
                    H_rot.row(2)[0],H_rot.row(2)[1],H_rot.row(2)[2],H_rot.row(2)[3],
                    0,0,0,1;

            init_flag=false;
            std::cout<<"surf_th : "<<surfThreshold<<endl;

        }


        // Eigen::Quaterniond q1;

        // q1.w()=0.5;
        // q1.x()=0.5;
        // q1.y()=-0.5;
        // q1.z()=0.5;

        // Eigen::Matrix3d R1 = q1.toRotationMatrix();
        // H_rot<<R1.row(0)[0],R1.row(0)[1],R1.row(0)[2],0,
        //             R1.row(1)[0],R1.row(1)[1],R1.row(1)[2],0,
        //             R1.row(2)[0],R1.row(2)[1],R1.row(2)[2],0,
        //             0,0,0,1;  

        // H_rot<<	-1,0,0,0,
        //         0,-1,0,0,
        //         0,0,1,0,	
        //             0,0,0,1; 

        // H_rot<<	1,0,0,0,
        //         0,1,0,0,
        //         0,0,1,0,	
        //             0,0,0,1; 
            
        H<<  R.row(0)[0],R.row(0)[1],R.row(0)[2],x,
                R.row(1)[0],R.row(1)[1],R.row(1)[2],y,
                R.row(2)[0],R.row(2)[1],R.row(2)[2],z,
                0,0,0,1;  


        // H = H_rot*H_init.inverse()*H; //to get H12 = H10*H02 , 180 rot according to z axis
        
        Eigen::Matrix4f HOut;
        HOut=lastH*lastR.inverse()*H;
        lastR=H;
        lastH=HOut;
        lastR<< H.row(0)[0],H.row(0)[1],H.row(0)[2],H.row(0)[3],
                H.row(1)[0],H.row(1)[1],H.row(1)[2],H.row(1)[3],
                H.row(2)[0],H.row(2)[1],H.row(2)[2],H.row(2)[3],
                0,0,0,1;
        lastH<< HOut.row(0)[0],HOut.row(0)[1],HOut.row(0)[2],HOut.row(0)[3],
                HOut.row(1)[0],HOut.row(1)[1],HOut.row(1)[2],HOut.row(1)[3],
                HOut.row(2)[0],HOut.row(2)[1],HOut.row(2)[2],HOut.row(2)[3],
                0,0,0,1;

        std::ofstream foutC(RESULT_PATH, std::ios::app);

        foutC.setf(std::ios::scientific, std::ios::floatfield);
        foutC.precision(6);
    
        //foutC << R[0] << " "<<transformMapped[3]<<" "<< R.row(1) <<" "<<transformMapped[4] <<" "<<  R.row(2) <<" "<< transformMapped[5] << endl;
        for (int i = 0; i < 3; ++i)	
        {	 
            for (int j = 0; j < 4; ++j)
                {
                if(i==2 && j==3)
                {
                    foutC <<HOut.row(i)[j]<< endl ;	
                }
                else
                {
                    foutC <<HOut.row(i)[j]<< " " ;
                }
                
            }
        }

        foutC.close();


    //////////////////////////////////////////////////        

        // publish tf
        static tf::TransformBroadcaster tfOdom2BaseLink;
        tf::Transform tCur;
        tf::poseMsgToTF(laserOdometry.pose.pose, tCur);
        if(lidarFrame != baselinkFrame)
            tCur = tCur * lidar2Baselink;
        tf::StampedTransform odom_2_baselink = tf::StampedTransform(tCur, imuOdomLatest.header.stamp, odometryFrame, baselinkFrame);
        tfOdom2BaseLink.sendTransform(odom_2_baselink);


        // Publish odometry for ROS (incremental)
        static bool lastIncreOdomPubFlag = false;
        static nav_msgs::Odometry laserOdomIncremental; // incremental odometry msg
        static Eigen::Affine3f increOdomAffine; // incremental odometry in affine
        if (lastIncreOdomPubFlag == false)
        {
            lastIncreOdomPubFlag = true;
            laserOdomIncremental = laserOdometry;
            increOdomAffine = imuOdomAffineLast;
        } else {
            // Eigen::Affine3f affineIncre = increOdomAffine.inverse() * imuOdomAffineLast;
            Eigen::Affine3f affineIncre = increOdomAffine.inverse() * imuOdomAffine;
            increOdomAffine = increOdomAffine * affineIncre;
            float x, y, z, roll, pitch, yaw;
            pcl::getTranslationAndEulerAngles (increOdomAffine, x, y, z, roll, pitch, yaw);
            

            laserOdomIncremental = imuOdomLatest;
            laserOdomIncremental.pose.pose.position.x = x;
            laserOdomIncremental.pose.pose.position.y = y;
            laserOdomIncremental.pose.pose.position.z = z;
            laserOdomIncremental.pose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(roll, pitch, yaw);
        }
        pubOdometryIncremental.publish(laserOdomIncremental);


        // publish IMU path
        static nav_msgs::Path localPath;
        static double last_path_time = -1;
        double imuTime = imuOdomLatest.header.stamp.toSec();
        if (imuTime - last_path_time > 0.1)
        {
            last_path_time = imuTime;
            geometry_msgs::PoseStamped pose_stamped;
            pose_stamped.header.stamp = imuOdomLatest.header.stamp;
            pose_stamped.header.frame_id = odometryFrame;
            pose_stamped.pose = laserOdometry.pose.pose;
            localPath.poses.push_back(pose_stamped);

            while(!localPath.poses.empty() && lidarOdomTime!=-1 && localPath.poses.front().header.stamp.toSec() < lidarOdomTime)
                localPath.poses.erase(localPath.poses.begin());
            if (pubLocalPath.getNumSubscribers() != 0)
            {
                localPath.header.stamp = imuOdomLatest.header.stamp;
                localPath.header.frame_id = odometryFrame;
                pubLocalPath.publish(localPath);
            }
        }


        rate.sleep();


    }

}


};




int main(int argc, char** argv)
{
    ros::init(argc, argv, "lis_slam");
    
    // OdometryFusion OF;
    TransformFusion TF;

    ROS_INFO("\033[1;32m----> Odometry  Fusion Started.\033[0m");

    // std::thread transformthread(&TransformFusion::transformFusionThread, &TF);

    ros::spin();

    // transformthread.join();
    
    return 0;
}