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
from geometry_msgs.msg import Twist, Pose
from tf.transformations import euler_from_quaternion
from gazebo_msgs.srv import SetPhysicsProperties
from gazebo_msgs.srv import SetPhysicsProperties, SetModelState,\
                            SpawnModel, DeleteModel,\
                            ApplyBodyWrenchRequest, ApplyBodyWrench

import rospy
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from std_msgs.msg import Float64
from gazebo_msgs.msg import LinkStates
from geometry_msgs.msg import Pose, Twist
from std_srvs.srv import Empty

from task import CartPoleTask

class CartPoleEnv(gym.Env):


    def __init__(self, task):

        # Initialize node
        rospy.init_node('cart_pole_simulation', anonymous=True)

        # State and Action spaces
        self.action_space      = gym.spaces.Box(low=-10, high=10, shape=(1,), dtype=float)
        self.observation_space = gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=(4,), dtype=float)

        # Suscriber for the link states (Pose+Twist for every link including cart and pole)
        rospy.Subscriber("/gazebo/link_states", LinkStates, self.get_cart_pose_callback)

        # Publisher for the cart controller
        self.pub_cart = rospy.Publisher('/cart_controller/command', Float64, queue_size = 10)

        # Reset simulation util
        self.reset_simulation_client = rospy.ServiceProxy('/gazebo/reset_simulation',Empty)

        # Task to accomplish
        self.task = task

        # Cart States
        self.cart_pose_x = 0
        self.cart_vel_x = 0

        # Pole States
        self.yaw_angle = 0
        self.y_angular = 0

        # Duration in seconds of every step in the episode
        self.time_interval = 0.02

    def get_cart_pose_callback(self, data):

        # Set cart state (Pose and velocity in x axis)
        ind = data.name.index('cart_pole::cart_link')
        cart_pose = data.pose[ind]
        cart_vel = data.twist[ind]

        self.cart_pose_x = cart_pose.position.x
        self.cart_vel_x = cart_vel.linear.x


        # Set pole state (y Angular velocity)
        ind_pitch = data.name.index('cart_pole::pole_link')
        pole_twist = data.twist[ind_pitch]
        self.y_angular = pole_twist.angular.y


        # Set z pose of the pole tip. Is this being used?
        ind_tip = data.name.index('cart_pole::tip_link')
        pole_tip_pose = data.pose[ind_tip]
        pole_tip_pose_z = pole_tip_pose.position.z


    def render(self):
        pass


    def step(self, action):

        # Start time of the step
        time1 = time.time()

        # Publish to the pub_cart controller
        self.pub_cart.publish(action)

        # Get new state
        self.yaw_angle += self.y_angular*self.time_interval
        robot_state = [self.cart_pose_x, self.cart_vel_x, self.yaw_angle, self.y_angular]

        reward = self.task.get_reward(robot_state)
        done = self.task.is_done(robot_state)

        # Check if time interval has passed
        time2 = time.time()
        interval = time2 - time1
        if(interval < self.time_interval):
            time.sleep(self.time_interval - interval)

        return np.array(robot_state, dtype='float16'), reward, done, {}

    def reset(self):
        
        # Reset Sim
        rospy.wait_for_service('/gazebo/reset_simulation')
        self.reset_simulation_client() 

        # Reset state
        self.cart_pose_x = 0
        self.cart_vel_x = 0
        self.yaw_angle = 0
        self.y_angular = 0  
  
        return np.array([self.cart_pose_x, self.cart_vel_x, self.yaw_angle, self.y_angular], dtype='float16')
