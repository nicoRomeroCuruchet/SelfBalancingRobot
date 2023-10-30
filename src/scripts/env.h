#ifndef ENV_H
#define ENV_H

#include <vector>
#include <memory>
#include <ros/ros.h>
#include <std_srvs/Empty.h>
#include <nav_msgs/Odometry.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>

class SelfBalancingRobotEnv {

public:

    // Constructor
    SelfBalancingRobotEnv(ros::NodeHandle& nh, ros::Rate& r);

    // ground truth callback
    void groundTruthBaseLinkCallback(const nav_msgs::Odometry::ConstPtr& msg);

    // step simulation
    std::shared_ptr<std::vector<float>> step(std::vector<float>& action);

private:
    
    // ros topics handle
    ros::Subscriber sub_ground_truth;
    ros::Publisher  pub_velocity;

    // service client
    ros::ServiceClient resetSimulationClient;
    std_srvs::Empty pause_req;
    std_srvs::Empty unpause_req;
    ros::ServiceClient pauseClient;
    ros::ServiceClient unpauseClient;

    // Robot state
    ros::Rate rate;
    tf2::Quaternion q;
    float angle;
    float angular_rate;
    float position_x;
    float position_y;
    float velocity_x;
    float velocity_y;
    double roll, pitch, yaw;

};

#endif // ENV_H