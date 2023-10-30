#include "env.h"
#include <std_srvs/Empty.h>

/**
 * @brief Constructor for the SelfBalancingRobotEnv class.
 * 
 * @param nh The ROS node handle.
 * @param r The ROS rate.
 */
SelfBalancingRobotEnv::SelfBalancingRobotEnv(ros::NodeHandle& nh, ros::Rate& r):rate(r)
{
    // ground truth of the base link
    sub_ground_truth = nh.subscribe("/self_balancing_robot/ground_truth/state", 1, &SelfBalancingRobotEnv::groundTruthBaseLinkCallback, this);
    // control the velocity of the robot
    pub_velocity = nh.advertise<geometry_msgs::Twist>("/self_balancing_robot/cmd_vel", 1);
    // Reset the simulation
    resetSimulationClient = nh.serviceClient<std_srvs::Empty>("/gazebo/reset_simulation");
    // Pause and unpause the simulation
    pauseClient   = nh.serviceClient<std_srvs::Empty>("/gazebo/pause_physics");
    unpauseClient = nh.serviceClient<std_srvs::Empty>("/gazebo/unpause_physics");
    // Reset the simulation
    ROS_INFO("Environment initialized");
}

/**
 * @brief Callback function for the ground truth base link odometry message.
 * 
 * This function is called whenever a new ground truth base link odometry message is received.
 * The message contains information about the robot's position and orientation in the world.
 * 
 * @param msg The incoming ground truth base link odometry message.
 */
void SelfBalancingRobotEnv::groundTruthBaseLinkCallback(const nav_msgs::Odometry::ConstPtr& msg)
{
    // Position of the robot
    position_x = msg->pose.pose.position.x;
    position_y = msg->pose.pose.position.y;
    // Linear velocity of the robot
    velocity_x = msg->twist.twist.linear.x;
    velocity_y = msg->twist.twist.linear.y;
    // Angular rate of the robot
    angular_rate = msg->twist.twist.angular.y;
    // Quaternion to Euler angles
    tf2::Quaternion _q(msg->pose.pose.orientation.x, 
                       msg->pose.pose.orientation.y, 
                       msg->pose.pose.orientation.z, 
                       msg->pose.pose.orientation.w);
    q=_q;
}

/**
 * Takes a step in the environment using the given action.
 * @param action The action to take in the environment.
 * @return A shared pointer to a vector of floats representing the new state of the environment.
 */
std::shared_ptr<std::vector<float>> SelfBalancingRobotEnv::step(std::vector<float>& action) 
{
    //unpause physics:
    unpauseClient.call(unpause_req);
    //ROS_INFO("Step function called");
    //ros::Time start_time = ros::Time::now();
    // publish velocity
    geometry_msgs::Twist vel;
    // Set the linear.x value
    vel.linear.x = action[0];
    vel.linear.y  = 0;
    vel.linear.z  = 0;
    vel.angular.x = 0;
    vel.angular.y = 0;
    vel.angular.z = 0;
    pub_velocity.publish(vel);
    // state(t) 
    ros::Duration(0.005).sleep(); // ground truth is published at 200Hz
    ros::spinOnce();
    // state(t + 1) update the robot state, now update the observation
    rate.sleep();
    //ros::Duration elapsed_time = ros::Time::now() - start_time;
    //pause physics:
    pauseClient.call(pause_req);
    //std::cout << "Time step: " << elapsed_time.toSec() <<" s"<< "\n";
    //Prepare the observation state(t+1):
    // Convert quaternion to Euler angles
    tf2::Matrix3x3 m(q);
    roll, pitch, yaw;
    m.getRPY(roll, pitch, yaw);
    angle = pitch; // TODO ojo pierdo precision roll, pitch, yaw son doubles
    // return the observation:
    auto obs = std::make_shared<std::vector<float>>();
    obs->push_back(angle);
    obs->push_back(angular_rate);
    obs->push_back(position_x);
    obs->push_back(position_y);
    obs->push_back(velocity_x);
    obs->push_back(velocity_y);
    return obs;
}

