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
        self.observation_space = gym.spaces.Box(low=-math.pi/2, high=math.pi/2, dtype=float)
        #
        # Velocity message to publish
        self.vel=Twist()
        self.vel.linear.x = 0
        self.vel.linear.y = 0
        self.vel.linear.z = 0
        self.vel.angular.x =0
        self.vel.angular.y = 0
        self.vel.angular.z = 0
        self.pub.publish(self.vel)
        #
        self.time_interval     = 0.005
        self.module_velocity   = 0
        self.module_angular    = 0
        self.imu_data          = None
        self.current_angle     = None
        self.current_position  = None
        self.theshold          = 0.2
        

    def ground_truth_callback(self, msg):
        
        self.module_angular   = msg.twist.twist.angular.x**2 + msg.twist.twist.angular.y**2 + msg.twist.twist.angular.z**2
        self.module_velocity  = msg.twist.twist.linear.x**2  + msg.twist.twist.linear.y**2  + msg.twist.twist.linear.z**2  
        self.current_position = msg.pose.pose.position
        self.imu_data         = msg.pose.pose.orientation
        orientation           = self.imu_data 

        quat = [orientation.x, orientation.y, orientation.z, orientation.w]
        _, self.current_angle, _ = euler_from_quaternion(quat)


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

        reward = self.get_reward()
        #position = math.sqrt(self.current_position.x**2 + self.current_position.y**2)
        done = abs(self.current_angle) > self.theshold 

        if done:
            vel.linear.x  = 0
            self.pub.publish(vel)

        # Check if time interval has passed
        interval = time.time() - time1
        if(interval < self.time_interval):
            time.sleep(self.time_interval - interval)

        return  np.array([self.current_angle], dtype=float), reward, done, {}

    def reset(self):

        rospy.wait_for_service('/gazebo/reset_simulation')
        self.reset_simulation_client()

        #print('antes',self.module_velocity)
        #self.callback = True
        #while self.callback or (round(self.module_velocity, 3) > 0) or (round(self.module_angular, 3) > 0):
        #    pass
        #print('despues',self.module_velocity)

        self.current_angle = 0 
        return  np.array([self.current_angle], dtype=float)

    def render(self):
        pass


    def get_reward(self, error=0.10):

        """ Calculate the reward based on the current state.

        If the current state of the pendulum is outside the acceptable range,
        a negative reward is given. Otherwise, a positive reward is given.

        Returns:
            float: Reward value """

        #position = math.sqrt(self.current_position.x**2 + self.current_position.y**2)
        return -200.0 if abs(self.current_angle) > self.theshold else  2 - abs(self.current_angle) * 10