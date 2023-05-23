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
from gazebo_msgs.srv import SetPhysicsProperties, SetModelState,\
                            SpawnModel, DeleteModel,\
                            ApplyBodyWrenchRequest, ApplyBodyWrench

class SelfBalancingRobot(gym.Env):

    def __init__(self):

        rospy.init_node('controller_node')

        self.pub               = rospy.Publisher('/self_balancing_robot/cmd_vel', Twist, queue_size=1)
        self.sub               = rospy.Subscriber('/imu', Imu, self.imu_callback)
        self.sub_ground        = rospy.Subscriber('/self_balancing_robot/ground_truth/state', Odometry, self.ground_truth_callback)
        self.pause             = rospy.ServiceProxy("/gazebo/pause_physics",Empty)
        self.unpause           = rospy.ServiceProxy("/gazebo/unpause_physics",Empty)

        #
        self.reward_range      = (-float('inf'), float('inf'))
        self.action_space      = gym.spaces.Box(low=-10, high=10, shape=(1,), dtype=float)
        self.observation_space = gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=(3,), dtype=float)
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
        self.imu_data          = None
        self.current_angle     = None
        self.current_position  = None
        self.theshold          = 0.3
        self.max_distance = 5 # radius of circle the robot is constrained to
        self.reset()

    def imu_callback(self, data):

        """ Callback function for receiving IMU data from the '/imu' topic.

        Args:

            data (Imu): IMU data message """

        self.imu_data = data

        # roll, picht, yaw
        _, self.current_angle,_ = euler_from_quaternion([self.imu_data.orientation.x,
                                                         self.imu_data.orientation.y,
                                                         self.imu_data.orientation.z,
                                                         self.imu_data.orientation.w])

    def ground_truth_callback(self, msg):
        
        self.current_position = msg.pose.pose.position


    def step(self, action):

        vel=Twist()
        vel.linear.x  = action
        vel.linear.y  = 0
        vel.linear.z  = 0
        vel.angular.x = 0
        vel.angular.y = 0
        vel.angular.z = 0
        self.pub.publish(vel)

        #reward = self.get_reward()
        done = self.is_done()

        reward = 0
        if done:
            reward = -100


        return np.array([self.current_angle, self.current_position.x, self.current_position.y], dtype = float), reward, done, {}

    def is_done(self):

        """ Boolean function that returns True if the current episode has ended. """

        max_angle_exceeded = abs(self.current_angle) > self.theshold
        max_distance_exceeded = math.sqrt(self.current_position.x**2 + self.current_position.y**2) > self.max_distance
        return max_angle_exceeded or max_distance_exceeded


    def reset(self):
        
        # Delete the model
        delete_model_service = '/gazebo/delete_model'
        rospy.wait_for_service(delete_model_service)
        try:
            delete_model = rospy.ServiceProxy(delete_model_service, DeleteModel)
            delete_model('self_balancing_robot')
        except rospy.ServiceException as e:
            rospy.loginfo(f"Failed to delete the model: {str(e)}")
        
        # Wait for a moment to allow Gazebo to remove the model
        rospy.sleep(.2)
        # Spawn the model
        try:
            current_dir = os.path.dirname(__file__)
            urdf_path = os.path.join(current_dir, '../urdf/robot.urdf')
            model_name = 'self_balancing_robot'                             # Provide a unique name for your model
            rospy.wait_for_service('/gazebo/spawn_urdf_model')

            spawn_model = rospy.ServiceProxy('/gazebo/spawn_urdf_model', SpawnModel)
            with open(urdf_path, 'r') as f:
                urdf_xml = f.read()

            pose = Pose()
            pose.position.x     = 0  # Adjust the position as needed
            pose.position.y     = 0
            pose.position.z     = 0.1447948565264787
            pose.orientation.x  = 0
            pose.orientation.y  = 0
            pose.orientation.z  = 0
            pose.orientation.w  = 0
            response = spawn_model(model_name, urdf_xml, "", pose, "world")

        except rospy.ServiceException as e:
            rospy.loginfo(f"Failed to spawn the model: {str(e)}")

        self.imu_data = None
        while self.imu_data is None:
            try:
                self.imu_data = rospy.wait_for_message('/imu', Imu, timeout=1)
                self.step(0)
            except:
                pass
       
        # roll picht yaw
        _, self.current_angle,_ = euler_from_quaternion([self.imu_data.orientation.x,
                                                         self.imu_data.orientation.y,
                                                         self.imu_data.orientation.z,
                                                         self.imu_data.orientation.w])


        self.current_position = None
        while self.current_position is None:
            try:
                self.current_position = rospy.wait_for_message('/self_balancing_robot/ground_truth/state', Odometry, timeout=1).pose.pose.position
                self.step(0)
            except:
                pass
    
        return np.array([self.current_angle, self.current_position.x, self.current_position.y], dtype = float)

    def render(self):
        pass

"""
    def get_reward(self, error=0.10):

        Calculate the reward based on the current state.

        If the current state of the pendulum is outside the acceptable range,
        a negative reward is given. Otherwise, a positive reward is given.

        Returns:
            float: Reward value

        #position = math.sqrt(self.current_position.x**2 + self.current_position.y**2)
         
        return -100.0 if abs(self.current_angle) > self.theshold else 1.0

"""