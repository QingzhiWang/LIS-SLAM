//This code partly draws on ISCLOAM
// Author of  EPSC-LOAM : QZ Wang  
// Email wangqingzhi27@outlook.com

#include "epscGenerationClass.h"

// #define INTEGER_INTENSITY
EPSCGenerationClass::EPSCGenerationClass()
{
    
}

void EPSCGenerationClass::init_param(int rings_in, int sectors_in, double max_dis_in){
    rings = rings_in;
    sectors = sectors_in;
    max_dis = max_dis_in;
    ring_step = max_dis/rings;
    sector_step = 2*M_PI/sectors;
    print_param();
    init_color();

    current_point_cloud = pcl::PointCloud<pcl::PointXYZI>::Ptr(new pcl::PointCloud<pcl::PointXYZI>());
    corner_point_cloud = pcl::PointCloud<pcl::PointXYZI>::Ptr(new pcl::PointCloud<pcl::PointXYZI>());
    surf_point_cloud = pcl::PointCloud<pcl::PointXYZI>::Ptr(new pcl::PointCloud<pcl::PointXYZI>());

}

void EPSCGenerationClass::init_color(void){
    for(int i=0;i<1;i++){//RGB format
        color_projection.push_back(cv::Vec3b(0,i*16,255));
    }
    for(int i=0;i<15;i++){//RGB format
        color_projection.push_back(cv::Vec3b(0,i*16,255));
    }
    for(int i=0;i<16;i++){//RGB format
        color_projection.push_back(cv::Vec3b(0,255,255-i*16));
    }
    for(int i=0;i<32;i++){//RGB format
        color_projection.push_back(cv::Vec3b(i*32,255,0));
    }
    for(int i=0;i<16;i++){//RGB format
        color_projection.push_back(cv::Vec3b(255,255-i*16,0));
    }
    for(int i=0;i<64;i++){//RGB format
        color_projection.push_back(cv::Vec3b(i*4,255,0));
    }
    for(int i=0;i<64;i++){//RGB format
        color_projection.push_back(cv::Vec3b(255,255-i*4,0));
    }
    for(int i=0;i<64;i++){//RGB format
        color_projection.push_back(cv::Vec3b(255,i*4,i*4));
    }
}

void EPSCGenerationClass::print_param(){
    std::cout << "The EPSC parameters are:"<<rings<<std::endl;
    std::cout << "number of rings:\t"<<rings<<std::endl;
    std::cout << "number of sectors:\t"<<sectors<<std::endl;
    std::cout << "maximum distance:\t"<<max_dis<<std::endl;
}

EPSCDescriptor EPSCGenerationClass::calculate_isc(const pcl::PointCloud<pcl::PointXYZI>::Ptr filtered_pointcloud){
    EPSCDescriptor isc = cv::Mat::zeros(cv::Size(sectors,rings), CV_8U);

    for(int i=0;i<(int)filtered_pointcloud->points.size();i++){
        ROS_WARN_ONCE("intensity is %f, if intensity showed here is integer format between 1-255, please uncomment #define INTEGER_INTENSITY in EPSCGenerationClass.cpp and recompile", (double) filtered_pointcloud->points[i].intensity);
        double distance = std::sqrt(filtered_pointcloud->points[i].x * filtered_pointcloud->points[i].x + filtered_pointcloud->points[i].y * filtered_pointcloud->points[i].y);
        if(distance>=max_dis)
            continue;
        double angle = M_PI + std::atan2(filtered_pointcloud->points[i].y,filtered_pointcloud->points[i].x);
        int ring_id = std::floor(distance/ring_step);
        int sector_id = std::floor(angle/sector_step);
        if(ring_id>=rings)
            continue;
        if(sector_id>=sectors)
            continue;
#ifndef INTEGER_INTENSITY
        int intensity_temp = (int) (255*filtered_pointcloud->points[i].intensity);
#else
        int intensity_temp = (int) (filtered_pointcloud->points[i].intensity);
#endif
        if(isc.at<unsigned char>(ring_id,sector_id)<intensity_temp)
            isc.at<unsigned char>(ring_id,sector_id)=intensity_temp;

    }

    return isc;

}



EPSCDescriptor EPSCGenerationClass::calculate_sc(const pcl::PointCloud<pcl::PointXYZI>::Ptr filtered_pointcloud){
    EPSCDescriptor sc = cv::Mat::zeros(cv::Size(sectors,rings), CV_8U);

    for(int i=0;i<(int)filtered_pointcloud->points.size();i++){
        ROS_WARN_ONCE("intensity is %f, if intensity showed here is integer format between 1-255, please uncomment #define INTEGER_INTENSITY in EPSCGenerationClass.cpp and recompile", (double) filtered_pointcloud->points[i].intensity);
        double distance = std::sqrt(filtered_pointcloud->points[i].x * filtered_pointcloud->points[i].x + filtered_pointcloud->points[i].y * filtered_pointcloud->points[i].y);
        if(distance>=max_dis)
            continue;
        double angle = M_PI + std::atan2(filtered_pointcloud->points[i].y,filtered_pointcloud->points[i].x);
        int ring_id = std::floor(distance/ring_step);
        int sector_id = std::floor(angle/sector_step);
        if(ring_id>=rings)
            continue;
        if(sector_id>=sectors)
            continue;

        int z_temp = (int) (filtered_pointcloud->points[i].z);

        if(sc.at<unsigned char>(ring_id,sector_id)<z_temp)
            sc.at<unsigned char>(ring_id,sector_id)=z_temp;

    }

    return sc;

}




EPSCDescriptor EPSCGenerationClass::calculate_epsc(const pcl::PointCloud<pcl::PointXYZI>::Ptr filtered_corner_pointcloud,const pcl::PointCloud<pcl::PointXYZI>::Ptr filtered_surf_pointcloud){
    
    EPSCDescriptor lsc = cv::Mat::zeros(cv::Size(sectors,rings), CV_8U);
    EPSCDescriptor psc = cv::Mat::zeros(cv::Size(sectors,rings), CV_8U);
    EPSCDescriptor lpsc = cv::Mat::zeros(cv::Size(sectors,rings), CV_8U);

    for(int i=0;i<(int)filtered_corner_pointcloud->points.size();i++){
        double distance = std::sqrt(filtered_corner_pointcloud->points[i].x * filtered_corner_pointcloud->points[i].x + filtered_corner_pointcloud->points[i].y * filtered_corner_pointcloud->points[i].y);
        if(distance>=max_dis)
            continue;
        double angle = M_PI + std::atan2(filtered_corner_pointcloud->points[i].y,filtered_corner_pointcloud->points[i].x);
        int ring_id = std::floor(distance/ring_step);
        int sector_id = std::floor(angle/sector_step);
        if(ring_id>=rings)
            continue;
        if(sector_id>=sectors)
            continue;
        lsc.at<unsigned char>(ring_id,sector_id)++;

    }

    for(int i=0;i<(int)filtered_surf_pointcloud->points.size();i++){
        double distance = std::sqrt(filtered_surf_pointcloud->points[i].x * filtered_surf_pointcloud->points[i].x + filtered_surf_pointcloud->points[i].y * filtered_surf_pointcloud->points[i].y);
        if(distance>=max_dis)
            continue;
        double angle = M_PI + std::atan2(filtered_surf_pointcloud->points[i].y,filtered_surf_pointcloud->points[i].x);
        int ring_id = std::floor(distance/ring_step);
        int sector_id = std::floor(angle/sector_step);
        if(ring_id>=rings)
            continue;
        if(sector_id>=sectors)
            continue;
        psc.at<unsigned char>(ring_id,sector_id)++;

    }

    for (int i = 0;i < lpsc.rows;i++) {
        for (int j = 0;j < lpsc.cols;j++) {

                lpsc.at<unsigned char>(i,j) =100*psc.at<unsigned char>(i,j)/(1+lsc.at<unsigned char>(i,j));

        }
    }
    return lpsc;

}


EPSCDescriptor EPSCGenerationClass::getLastISCMONO(void){
    return isc_arr.back();

}

EPSCDescriptor EPSCGenerationClass::getLastISCRGB(void){
    //EPSCDescriptor isc = isc_arr.back();
    EPSCDescriptor isc_color = cv::Mat::zeros(cv::Size(sectors,rings), CV_8UC3);
    for (int i = 0;i < isc_arr.back().rows;i++) {
        for (int j = 0;j < isc_arr.back().cols;j++) {
            isc_color.at<cv::Vec3b>(i, j) = color_projection[isc_arr.back().at<unsigned char>(i,j)];

        }
    }
    return isc_color;
}

EPSCDescriptor EPSCGenerationClass::getLastEPSCMONO()
{
    return epsc_arr.back();
}
EPSCDescriptor EPSCGenerationClass::getLastEPSCRGB()
{
    EPSCDescriptor epsc_color = cv::Mat::zeros(cv::Size(sectors,rings), CV_8UC3);
    for (int i = 0;i < epsc_arr.back().rows;i++) {
        for (int j = 0;j < epsc_arr.back().cols;j++) {
            epsc_color.at<cv::Vec3b>(i, j) = color_projection[epsc_arr.back().at<unsigned char>(i,j)];

        }
    }
    return epsc_color;
}

EPSCDescriptor EPSCGenerationClass::getLastSCRGB()
{
    EPSCDescriptor sc_color = cv::Mat::zeros(cv::Size(sectors,rings), CV_8UC3);
    for (int i = 0;i < sc_arr.back().rows;i++) {
        for (int j = 0;j < sc_arr.back().cols;j++) {
            sc_color.at<cv::Vec3b>(i, j) = color_projection[sc_arr.back().at<unsigned char>(i,j)];

        }
    }
    return sc_color;
}


void EPSCGenerationClass::loopDetection(const pcl::PointCloud<pcl::PointXYZI>::Ptr& corner_pc, 
                                                                                    const pcl::PointCloud<pcl::PointXYZI>::Ptr& surf_pc,
                                                                                    const pcl::PointCloud<pcl::PointXYZI>::Ptr& raw_pc,Eigen::Isometry3d& odom)
{
    
    pcl::PointCloud<pcl::PointXYZI>::Ptr pc_filtered_corner(new pcl::PointCloud<pcl::PointXYZI>());
    pcl::PointCloud<pcl::PointXYZI>::Ptr pc_filtered_surf(new pcl::PointCloud<pcl::PointXYZI>());
    pcl::PointCloud<pcl::PointXYZI>::Ptr pc_filtered_raw(new pcl::PointCloud<pcl::PointXYZI>());
    ground_filter(corner_pc, pc_filtered_corner);
    ground_filter(surf_pc, pc_filtered_surf);
    ground_filter(raw_pc, pc_filtered_raw);
    
    EPSCDescriptor clpsc = calculate_epsc(pc_filtered_corner, pc_filtered_surf);
    EPSCDescriptor cisc = calculate_isc(pc_filtered_raw);
    EPSCDescriptor csc = calculate_sc(pc_filtered_raw);
    
    Eigen::Vector3d current_t = odom.translation();

    //dont change push_back sequence
    if(travel_distance_arr.size()==0){
        travel_distance_arr.push_back(0);
    }else{
        double dis_temp = travel_distance_arr.back()+std::sqrt((pos_arr.back()-current_t).array().square().sum());
        travel_distance_arr.push_back(dis_temp);
    }
    pos_arr.push_back(current_t);
    epsc_arr.push_back(clpsc);
    isc_arr.push_back(cisc);
    sc_arr.push_back(csc);

    current_frame_id = pos_arr.size()-1;
    matched_frame_id.clear();
    isc_matched_frame_id.clear();
    sc_matched_frame_id.clear();
    pos_matched_frame_id.clear();
    
    //search for the near neibourgh pos
    int best_matched_id_epsc=-1;
    double best_score_epsc=0.0;
    int best_matched_id_isc=-1;
    double best_score_isc=0.0;
    int best_matched_id_sc=-1;
    double best_score_sc=0.0;

    double min_ditance=1000000;
    int best_matched_id_pose=-1;

    bool iscFlag=false;
    bool scFlag=false;
    bool poseFlag=true;

    for(int i = 0; i< (int)pos_arr.size(); i++)
    {
        double delta_travel_distance = travel_distance_arr.back()- travel_distance_arr[i];
        double pos_distance = std::sqrt((pos_arr[i]-pos_arr.back()).array().square().sum());

        if(delta_travel_distance > SKIP_NEIBOUR_DISTANCE && pos_distance<delta_travel_distance*INFLATION_COVARIANCE){

            ROS_INFO("Matched_id: %d, delta_travel_distance: %f, pos_distance : %f",i,delta_travel_distance,pos_distance);
            std::cout << "EPSC:"<< std::flush;
            double geo_score=0;
            double inten_score =0;
            if(is_loop_pair(clpsc,epsc_arr[i],geo_score,inten_score)){
                if(geo_score+inten_score>best_score_epsc){
                    best_score_epsc = geo_score+inten_score;
                    best_matched_id_epsc = i;
                }
            }

            
            if(iscFlag==true){
                // std::cout << "ISC:"<< std::flush;
                double geo_score1=0;
                double inten_score1 =0;
                if(is_loop_pair(cisc,isc_arr[i],geo_score1,inten_score1)){
                    if(geo_score1+inten_score1>best_score_isc){
                        best_score_isc = geo_score1+inten_score1;
                        best_matched_id_isc = i;
                    }

                }
            }

            if(scFlag==true){
                // std::cout << "SC:"<< std::flush;
                double geo_score2=0;
                double inten_score2 =0;
                if(is_loop_pair(csc,sc_arr[i],geo_score2,inten_score2)){
                    if(geo_score2+inten_score2>best_score_sc){
                        best_score_sc = geo_score2+inten_score2;
                        best_matched_id_sc = i;
                    }
                }
            }

            if(poseFlag==true){
                // std::cout << "Pose:"<< std::flush;
                if(pos_distance<min_ditance)
                {
                    min_ditance = pos_distance;
                    best_matched_id_pose = i;
                }
            }

        }   
    }
    if(best_matched_id_epsc!=-1){
        matched_frame_id.push_back(best_matched_id_pose);
        matched_frame_id.push_back(best_matched_id_epsc);
        ROS_WARN("LPSC: received loop closure candidate: current: %d, history %d, total_score %f",current_frame_id,best_matched_id_epsc,best_score_epsc);            
    }

    if(iscFlag==true){
        if(best_matched_id_isc!=-1){
            isc_matched_frame_id.push_back(best_matched_id_isc);
            
            // ROS_WARN("ISC: received loop closure candidate: current: %d, history %d, total_score %f",current_frame_id,best_matched_id_isc,best_score_isc);
        }
    }

    if(scFlag==true){
        if(best_matched_id_sc!=-1){
            sc_matched_frame_id.push_back(best_matched_id_sc);
            // ROS_WARN("SC: received loop closure candidate: current: %d, history %d, total_score %f",current_frame_id,best_matched_id_sc,best_score_sc);
        }
    }

    if(poseFlag==true){
        if(best_matched_id_pose!=-1){
            pos_matched_frame_id.push_back(best_matched_id_pose);
            // ROS_WARN("Pose: received loop closure candidate: current: %d, history %d, min_ditance %f",current_frame_id,best_matched_id_pose,min_ditance);
        }
    }

}

bool EPSCGenerationClass::is_loop_pair(EPSCDescriptor& desc1, EPSCDescriptor& desc2, double& geo_score, double& inten_score){
    int angle =0;
    geo_score = calculate_geometry_dis(desc1,desc2,angle);
    std::cout<<"geo_score: "<<geo_score<<std::endl;
    if(geo_score>GEOMETRY_THRESHOLD){
        inten_score = calculate_intensity_dis(desc1,desc2,angle);
        std::cout<<"inten_score: "<<inten_score<<std::endl;
        if(inten_score>INTENSITY_THRESHOLD){
            return true;
        }
    }
    return false;
}

double EPSCGenerationClass::calculate_geometry_dis(const EPSCDescriptor& desc1, const EPSCDescriptor& desc2, int& angle){
    double similarity = 0.0;

    for(int i=0;i<sectors;i++){
        int match_count=0;
        for(int p=0;p<sectors;p++){
            int new_col = p+i>=sectors?p+i-sectors:p+i;
            for(int q=0;q<rings;q++){
                if((desc1.at<unsigned char>(q,p)== true && desc2.at<unsigned char>(q,new_col)== true) || (desc1.at<unsigned char>(q,p)== false && desc2.at<unsigned char>(q,new_col)== false)){
                    match_count++;
                }

            }
        }
        if(match_count>similarity){
            similarity=match_count;
            angle = i;
        }

    }
    return similarity/(sectors*rings);
    
}
double EPSCGenerationClass::calculate_intensity_dis(const EPSCDescriptor& desc1, const EPSCDescriptor& desc2, int& angle){
    double difference = 1.0;
    double angle_temp = angle;
    for(int i=angle_temp-10;i<angle_temp+10;i++){

        int match_count=0;
        int total_points=0;
        for(int p=0;p<sectors;p++){
            int new_col = p+i;
            if(new_col>=sectors)
                new_col = new_col-sectors;
            if(new_col<0)
                new_col = new_col+sectors;
            for(int q=0;q<rings;q++){
                    match_count += abs(desc1.at<unsigned char>(q,p)-desc2.at<unsigned char>(q,new_col));
                    total_points++;
            }
            
        }
        double diff_temp = ((double)match_count)/(sectors*rings*255);
        if(diff_temp<difference)
            difference=diff_temp;

    }
    return 1 - difference;
    
}



void EPSCGenerationClass::ground_filter(const pcl::PointCloud<pcl::PointXYZI>::Ptr& pc_in, pcl::PointCloud<pcl::PointXYZI>::Ptr& pc_out){
    pcl::PassThrough<pcl::PointXYZI> pass;
    pass.setInputCloud (pc_in);
    pass.setFilterFieldName ("z");
    pass.setFilterLimits (-0.9, 30.0);
    pass.filter (*pc_out);

}