import time
import math
import rospy
import numpy as np
import tkinter as tk
import gymnasium as gym
from std_srvs.srv import Empty
from geometry_msgs.msg import Twist
from gazebo_msgs.srv  import SetModelState
from gazebo_msgs.msg import ModelState

import socket
import time 
import struct

"""
https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html#tips-and-tricks-when-creating-a-custom-environment

Best practices when creating a custom environment:

1) always normalize your observation space when you can, i.e., when you know the boundaries. OK
2) Normalize your action space and make it symmetric when continuous (cf potential issue below). OK
A good practice is to rescale your actions to lie in [-1, 1]. This does not limit you as you can easily rescale the action inside the environment. OK
3) start with shaped reward (i.e. informative reward) and simplified version of your problem.
4) debug with random actions to check that your environment works and follows the gym interface.

"""

class SelfBalancingRobotBaseLine(gym.Env):

    """ A reinforcement learning environment for a self-balancing robot simulation.

    This class represents an environment for training and testing the control
    of a self-balancing robot. It provides methods for taking actions, receiving
    observations, calculating rewards, and resetting the environment.

    Note:
        This class is designed to be used with the OpenAI Gym reinforcement learning framework. """

    def __init__(self, max_timesteps_per_episode,
                       threshold_angle=0.2,
                       threshold_position=0.2):

        """ Initialize the SelfBalancingRobot environment. """

        super(SelfBalancingRobotBaseLine, self).__init__()

        rospy.init_node('controller_node')

        # Create a socket
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # Connect to the server
        server_address = ('localhost', 12345)  
        self.client_socket.connect(server_address)
        

        self.pub_vel = rospy.Publisher('/self_balancing_robot/cmd_vel', Twist, queue_size=1)
        # Create the service to reset the simulation
        self.reset_simulation_client = rospy.ServiceProxy('/gazebo/reset_simulation',Empty)
       # Create the service to pause the simulation
        self.pause_simulation_client = rospy.ServiceProxy('/gazebo/pause_physics',Empty)
        # Create the service to unpause the simulation
        self.unpause_simulation_client = rospy.ServiceProxy('/gazebo/unpause_physics',Empty)

        # Set the gym environment parameters
        self.reward_range      = (-float('inf'), float('inf'))
        self.action_space      = gym.spaces.Box(low=-1, high=+1, shape=(1,), dtype=float)

        """
        Observation space: 
        angle: -pi/2, +pi/2 (-1.5, 1.5) of the robot
        angular_y: not bounded
        position_x: -1, 1
        position_y: -1, 1
        velocity_x: not bounded
        velocity_y: not bounded
        """
        self.observation_space = gym.spaces.Box(low=np.array([-math.pi/2, -float('inf'), -1, -1, -float('inf'), -float('inf')]), 
                                                high=np.array([math.pi/2, +float('inf'),  1,  1, +float('inf'), +float('inf')]), dtype=float)

        # Create the environment and stop de the robot
        self.vel=Twist()
        self.current_angle     = 0       # the pitch angle of the robot
        # storage callback data
        # Angle thresholds angle expresed in radians and position in meters
        self.threshold_angle     = abs(threshold_angle)
        self.threshold_position  = abs(threshold_position)

        self.current_step = 0
        self.max_steps = max_timesteps_per_episode
        
    
    def step(self, action):

        """ Perform a simulation step in the environment.

        Args:
            action (float): The action to take.

        Returns:
            observation (numpy.ndarray): The observation of the environment.
            reward (float): The reward obtained from the action.
            done (bool): Whether the episode is done or not.
            truncated (bool): Whether the episode is truncated or not.
            info (dict): Additional information about the step. """

        # send the action
        packed_data = struct.pack(f"<{len(action)}f", *action)
        sent = self.client_socket.send(packed_data)
        if sent == 0: 
            raise RuntimeError("socket connection broken")
        # C++ connection socket. Wait until c++ step response:
        #time.sleep(0.02)
        receivedData = self.client_socket.recv(24)  # 6 floats * 4 bytes each
        # Unpack the binary data into 6 floats
        data = struct.unpack('ffffff', receivedData)
        #rospy.loginfo('Socket sended data')
        self.current_angle = data[0]
        self.angular_y     = data[1]
        self.position_x    = data[2]
        self.position_y    = data[3]
        self.velocity_x    = data[4]
        self.velocity_y    = data[5]
        self.current_step += 1
        reward = self.get_reward()
        done = (abs(self.current_angle) > self.threshold_angle) 
        truncated = self.current_step >= self.max_steps
        # environment observation
        return  np.array([self.current_angle, 
                          self.angular_y,
                          self.position_x,
                          self.position_y,
                          self.velocity_x,    
                          self.velocity_y], dtype=float), reward, done, truncated, {}

    def reset(self, **kwargs):

        """ Reset the environment.

        Returns:
        
            observation (numpy.ndarray): The initial observation of the environment. """

        # unpause simulation
        self.unpause_simulation_client()
        # stop the robot
        for _ in range(10):
            self.vel=Twist()
            self.vel.linear.x=0
            self.vel.linear.y=0
            self.vel.linear.z=0
            self.vel.angular.x=0
            self.vel.angular.y=0
            self.vel.angular.z=0
            self.pub_vel.publish(self.vel)
            time.sleep(0.01)

        rospy.wait_for_service('/gazebo/reset_simulation')
        # reset the robot position and orientation
        self.reset_simulation_client()

        # pause simulation
        self.pause_simulation_client()
        self.current_angle = 0 
        self.angular_y     = 0
        self.position_x    = 0
        self.position_y    = 0
        self.velocity_x    = 0
        self.velocity_y    = 0
        self.current_step  = 0
        info = {}
        
        # rospy.loginfo('Environment reseted')
        return  np.array([0, 0, 0, 0, 0, 0], dtype=float), info

    def render(self):
        pass

    def get_reward(self):

        """
        Calculate the reward.

        Returns:
            reward (float): The reward value based on the state of the robot.
        """ 
        return 0.5*(1-np.sin(self.current_angle)) - (self.position_x / 0.5)  