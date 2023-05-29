import os
import gym
import math
import time 
import rospy
import torch
import numpy as np
import torch.nn as nn
from std_srvs.srv import Empty
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from gazebo_msgs.msg import ModelState
from pid_controller import PIDController
from geometry_msgs.msg import Twist, Pose
from tf.transformations import euler_from_quaternion
from gazebo_msgs.srv import SetPhysicsProperties
from gazebo_msgs.srv import SetPhysicsProperties, SetModelState,\
                            SpawnModel, DeleteModel,\
                            ApplyBodyWrenchRequest, ApplyBodyWrench


class SelfBalancingRobot(gym.Env):

    def __init__(self):

        rospy.init_node('controller_node')

        self.pub                     = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.sub_ground              = rospy.Subscriber('/self_balancing_robot/ground_truth/state', Odometry, self.ground_truth_callback)
        self.reset_simulation_client = rospy.ServiceProxy('/gazebo/reset_simulation',Empty)

        self.reward_range      = (-float('inf'), float('inf'))
        self.action_space      = gym.spaces.Box(low=-5, high=5, shape=(1,), dtype=float)
        self.observation_space = gym.spaces.Box(low=np.array([-math.pi/2, -float('inf')]), 
                                                high=np.array([math.pi/2,  float('inf')]), dtype=float)
        
        # Create the environment and stop de the robot
        self.vel=Twist()
        self.vel.linear.x = 0
        self.vel.linear.y = 0
        self.vel.linear.z = 0
        self.vel.angular.x =0
        self.vel.angular.y = 0
        self.vel.angular.z = 0
        self.pub.publish(self.vel)
        
        # refresh rate in ground truth
        self.time_interval     = 0.005
        self.current_angle     = 0
        # storage callback data
        self.angular_y         = None
        self.imu_data          = None        
        # Angle threshold expresed in radians
        self.theshold          = 0.2
        

    def ground_truth_callback(self, msg):
        
        # Angular rate of the robot 
        self.angular_y        = msg.twist.twist.angular.y
        # quaternion to euler
        self.imu_data         = msg.pose.pose.orientation
        q                     = self.imu_data 
        # roll pitch yaw
        _, self.current_angle, _ = euler_from_quaternion([q.x, 
                                                          q.y, 
                                                          q.z, 
                                                          q.w])

    def step(self, action):

        time1 = time.time()

        vel=Twist()
        vel.linear.x  = action
        vel.linear.y  = 0
        vel.linear.z  = 0
        vel.angular.x = 0
        vel.angular.y = 0
        vel.angular.z = 0
        self.pub.publish(vel)

        # Check if time interval has passed
        interval = time.time() - time1
        if(interval < self.time_interval):
            time.sleep(self.time_interval - interval)

        reward = self.get_reward()
        done = abs(self.current_angle) > self.theshold 
        if done:
            vel.linear.x  = 0
            self.pub.publish(vel)

        return  np.array([self.current_angle, self.angular_y], dtype=float), reward, done, {}

    def reset(self):

        rospy.wait_for_service('/gazebo/reset_simulation')
        self.reset_simulation_client()

        self.current_angle = 0 
        self.angular_y     = 0
        return  np.array([self.current_angle, self.angular_y], dtype=float)

    def render(self):
        pass


    def get_reward(self):

        return -200.0 if abs(self.current_angle) > self.theshold  else  2 - abs(self.current_angle) * 10