import time
import math
import rospy
import numpy as np
import tkinter as tk
import gymnasium as gym
from std_srvs.srv import Empty
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from tf.transformations import euler_from_quaternion
from gazebo_msgs.srv  import ApplyBodyWrenchRequest
from gazebo_msgs.srv  import ApplyBodyWrench
from gazebo_msgs.srv  import SetModelState
from gazebo_msgs.msg import ModelState


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
                        threshold_position=0.2,
                        apply_force=True):

        """ Initialize the SelfBalancingRobot environment. """

        super(SelfBalancingRobotBaseLine, self).__init__()

        rospy.init_node('controller_node')

        # Create the publisher to control the velocity of the robot
        self.pub                     = rospy.Publisher('/self_balancing_robot/cmd_vel', Twist, queue_size=1)
        # Create the subscriber to get the ground truth data
        self.sub_ground              = rospy.Subscriber('/self_balancing_robot/ground_truth/state', Odometry, self.ground_truth_callback)
        # Create the service to reset the simulation
        self.reset_simulation_client = rospy.ServiceProxy('/gazebo/reset_simulation',Empty)
        # model state
        self.set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)


        # Set the gym environment parameters
        self.reward_range      = (-float('inf'), float('inf'))
        self.action_space      = gym.spaces.Box(low=-1, high=+1, shape=(1,), dtype=float)

        """
        Observation space: 
        current_angle: -pi/2, +pi/2 (-1.5, 1.5) of the robot
        angular_y: not bounded
        position_x: -1, 1
        position_y: -1, 1
        velocity_x: not bounded
        velocity_y: not bounded
        """


        self.observation_space = gym.spaces.Box(low=np.array([-math.pi/2, -float('inf'), -float('inf'), -1, -1, -float('inf'), -float('inf')]), 
                                                high=np.array([math.pi/2, +float('inf'), +float('inf'),  1,  1, +float('inf'), +float('inf')]), dtype=float)

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
        # Angle thresholds angle expresed in radians and position in meters
        self.threshold_angle     = abs(threshold_angle)
        self.threshold_position  = abs(threshold_position)

        self.prev_action = 0
        self.curr_action = 0
        self.aplly_force = apply_force

        self.current_step = 0
        self.max_steps = max_timesteps_per_episode
        
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
        
    
    def __apply_force_to_link__(self, link_name, force, duration):

        """ Apply a force to a link in the simulation.  
        
        Args:  
            link_name (str): The name of the link to apply the force to.
            force (list): The force to apply to the link.
            duration (float): The duration of the force.
        
        Returns:
            response (bool): Whether the force was applied or not.
        """

        rospy.wait_for_service('/gazebo/apply_body_wrench')
        apply_wrench = rospy.ServiceProxy('/gazebo/apply_body_wrench', ApplyBodyWrench)
        request = ApplyBodyWrenchRequest()
        request.body_name = link_name
        request.wrench.force.x = force[0]
        request.wrench.force.y = force[1]
        request.wrench.force.z = force[2]
        request.duration = rospy.Duration(duration)
        response = apply_wrench(request)

        return response.success

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

        # Take the action
        time1 = time.time()
        vel=Twist()
        vel.linear.x  = action
        vel.linear.y  = 0
        vel.linear.z  = 0
        vel.angular.x = 0
        vel.angular.y = 0
        vel.angular.z = 0
        self.pub.publish(vel)

        self.prev_action = self.curr_action
        self.curr_action = action

        # Check if time interval has passed
        # the ground truth is published at 200 Hz. 
        # Check it in the robot.urdf file, in the sensors section.
        interval = time.time() - time1
        if(interval < self.time_interval):
            time.sleep(self.time_interval - interval)

        reward = self.get_reward()

        # Check if the episode is done
        position_module = math.sqrt(self.position_x**2 + self.position_y**2)
        done = (abs(self.current_angle) > self.threshold_angle) or (position_module > self.threshold_position)

        self.current_step += 1
        truncated = self.current_step >= self.max_steps
        done = done or truncated

        # Apply random force to the robot every n steps
        if self.current_step % 100 == 0 and self.aplly_force:
            # calculate ramdom number between -1 and 1
            random_force = np.random.uniform(-2, 2)
            self.__apply_force_to_link__("self_balancing_robot::base_link", [random_force, 0, 0], 0.1)

        if done:
            vel.linear.x  = 0
            self.pub.publish(vel)

        diff = float(self.curr_action - self.prev_action)
        # environment observation
        return  np.array([self.current_angle, 
                          self.angular_y,
                          diff,
                          self.position_x,
                          self.position_y,
                          self.velocity_x,    
                          self.velocity_y], dtype=float), reward, done, truncated, {}

    def reset(self, **kwargs):

        """ Reset the environment.

        Returns:
        
            observation (numpy.ndarray): The initial observation of the environment. """

        rospy.wait_for_service('/gazebo/reset_simulation')
        self.reset_simulation_client()
        time.sleep(0.1)
        # Wait for the environment to reset
        #x_pos =  np.random.uniform(-self.threshold_position, self.threshold_position)
        #model_state = ModelState()
        #model_state.model_name = 'self_balancing_robot'
        #model_state.pose.position.x = x_pos
        #model_state.pose.position.y = 0
        #model_state.pose.position.z = 0.145
        #model_state.pose.orientation.x = 0
        #model_state.pose.orientation.y = 0
        #model_state.pose.orientation.z = 0
        #model_state.pose.orientation.w = 1        
        #self.set_model_state(model_state)

        self.current_angle = 0 
        self.angular_y     = 0
        self.position_x    = 0
        self.position_y    = 0
        self.velocity_x    = 0
        self.velocity_y    = 0
        self.current_step  = 0
        self.prev_action   = 0
        self.curr_action   = 0
        info = {}
        
        return  np.array([0, 0, 0, 0, 0, 0, 0], dtype=float), info

    def render(self):
        pass

    def get_reward(self):

        """
        Calculate the reward.

        Returns:
            reward (float): The reward value based on the state of the robot.
        """

        angle_correction =  self.threshold_angle  - abs(self.current_angle) 
        angle_rate       = -abs(self.angular_y)                                
        position_module  = -(self.position_x**2 + self.position_y**2)          
        velocity_module  = -math.sqrt(self.velocity_x**2 + self.velocity_y**2) 
        smoothness       = -abs(self.curr_action - self.prev_action)           

        done = (abs(self.current_angle) > self.threshold_angle) or (position_module > self.threshold_position)

        weights = [10, 0.2, .1, 0, 0.1]
        state   = [angle_correction, angle_rate, position_module, velocity_module, smoothness]
        reward  = sum([weights[i]*state[i] for i in range(len(weights))])
        
        return -200.0 if done else float(reward)
    
