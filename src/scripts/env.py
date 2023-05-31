import gym
import time
import math
import rospy
import numpy as np
from std_srvs.srv import Empty
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Pose
from tf.transformations import euler_from_quaternion

class SelfBalancingRobot(gym.Env):

    """ A reinforcement learning environment for a self-balancing robot simulation.

    This class represents an environment for training and testing the control
    of a self-balancing robot. It provides methods for taking actions, receiving
    observations, calculating rewards, and resetting the environment.

    Note:
        This class is designed to be used with the OpenAI Gym reinforcement learning framework. """

    def __init__(self):

        """ Initialize the SelfBalancingRobot environment. """

        rospy.init_node('controller_node')

        # Create the publisher to control the velocity of the robot
        self.pub                     = rospy.Publisher('/self_balancing_robot/cmd_vel', Twist, queue_size=1)
        # Create the subscriber to get the ground truth data
        self.sub_ground              = rospy.Subscriber('/self_balancing_robot/ground_truth/state', Odometry, self.ground_truth_callback)
        # Create the service to reset the simulation
        self.reset_simulation_client = rospy.ServiceProxy('/gazebo/reset_simulation',Empty)

        # Set the gym environment parameters
        self.reward_range      = (-float('inf'), float('inf'))
        self.action_space      = gym.spaces.Box(low=-5, high=5, shape=(1,), dtype=float)
        self.observation_space = gym.spaces.Box(low=np.array([-math.pi/2, -float('inf'), -1, -1, -float('inf'), -float('inf')]), 
                                                high=np.array([math.pi/2,  float('inf'),  1,  1,  float('inf'),  float('inf')]), dtype=float)
        
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
        self.time_interval     = 0.005   # running at ~200 Hz
        self.current_angle     = 0       # the pitch angle of the robot
        # storage callback data
        self.angular_y  = 0
        self.imu_data   = None
        self.position_x = None
        self.position_y = None
        self.velocity_x = None
        self.velocity_y = None
        # Angle threshold expresed in radians
        self.theshold          = 0.3
        
    def ground_truth_callback(self, msg):

        """ Callback function for ground truth data.

        Args:
            msg (Odometry): The ground truth odometry message. """
        
        # Position of the robot
        self.position_x       = msg.pose.pose.position.x
        self.position_y       = msg.pose.pose.position.y
        # Linear velocity of the robot
        self.velocity_x       = msg.twist.twist.linear.x
        self.velocity_y       = msg.twist.twist.linear.y
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

        """ Perform a simulation step in the environment.

        Args:
            action (float): The action to take.

        Returns:
            observation (numpy.ndarray): The observation of the environment.
            reward (float): The reward obtained from the action.
            done (bool): Whether the episode is done or not.
            info (dict): Additional information about the step. """

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
        # the ground truth is published at 200 Hz. Check it in the robot.urdf file, in the sensors section.
        interval = time.time() - time1
        if(interval < self.time_interval):
            time.sleep(self.time_interval - interval)

        reward = self.get_reward()
        done = abs(self.current_angle) > self.theshold or abs(self.position_x) > 1 or abs(self.position_y) > 1 
        if done:
            vel.linear.x  = 0
            self.pub.publish(vel)

        return  np.array([self.current_angle, self.angular_y,
                          self.position_x, self.position_y,
                          self.velocity_x, self.velocity_y ], dtype=float), reward, done, {}


    def reset(self):

        """ Reset the environment.

        Returns:
            observation (numpy.ndarray): The initial observation of the environment. """

        rospy.wait_for_service('/gazebo/reset_simulation')
        self.reset_simulation_client()

        self.position_x    = 0
        self.position_y    = 0
        self.velocity_x    = 0
        self.velocity_y    = 0
        self.current_angle = 0 
        self.angular_y     = 0
        
        return  np.array([0, 0,
                          0, 0,
                          0, 0], dtype=float)

    def render(self):
        pass

    def get_reward(self):

        """
        Calculate the reward based on the current_angle, position and velocity of the robot.

        Returns:
            reward (float): The reward value based on the current angle.
        """
        # 
        done = abs(self.current_angle) > self.theshold or abs(self.position_x) > 1 or abs(self.position_y) > 1

        angle_correction    =  3.0 - abs(self.current_angle) * 10
        position_correction = -(abs(self.position_x) + abs(self.position_y))*0.3
        velocity_correction = -(abs(self.velocity_x + abs(self.velocity_y)))*0.3

        return -200.0 if done else angle_correction + position_correction + velocity_correction