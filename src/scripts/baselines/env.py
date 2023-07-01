import gymnasium as gym
import time
import math
import rospy
import numpy as np
from std_srvs.srv import Empty
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Pose
from tf.transformations import euler_from_quaternion


"""

Best practices when creating a custom environment. https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html#tips-and-tricks-when-creating-a-custom-environment

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

    def __init__(self, max_timesteps_per_episode):

        """ Initialize the SelfBalancingRobot environment. """

        super(SelfBalancingRobotBaseLine, self).__init__()
        #TODO: could it be that the initiation should be: super().__init__()  ?

        rospy.init_node('controller_node')

        # Create the publisher to control the velocity of the robot
        self.pub                     = rospy.Publisher('/self_balancing_robot/cmd_vel', Twist, queue_size=1)
        # Create the subscriber to get the ground truth data
        self.sub_ground              = rospy.Subscriber('/self_balancing_robot/ground_truth/state', Odometry, self.ground_truth_callback)
        # Create the service to reset the simulation
        self.reset_simulation_client = rospy.ServiceProxy('/gazebo/reset_simulation',Empty)

        # Set the gym environment parameters
        self.reward_range      = (-float('inf'), float('inf'))
        self.action_space      = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=float)


        """
        Observation space: 

        current_angle: -pi/2, +pi/2 (-1.5, 1.5) of the robot
        angular_y: not bounded
        position_x : -1, 1
        position_y: -1, 1
        velocity_x: not bounded
        velocity_y: not bounded
        """


        self.observation_space = gym.spaces.Box(low=np.array([-math.pi/2, -float('inf'), -1, -1, -float('inf'), -float('inf')]), 
                                                high=np.array([math.pi/2,  float('inf'),  1,  1,  float('inf'),  float('inf')]), dtype=float)

        # Create the environment and stop the robot
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
        # Angle and position thresholds Angle expresed in radians and position in meters
        self.threshold_angle     = 0.2
        self.threshold_position  = 0.1

        self.current_step = 0
        self.max_steps = max_timesteps_per_episode

        #Flag to ensure a new measure is available after stepping
        self.new_measure = False
        #Debugging-------------------------------------------------
        # Variables for visualization and debugging
        # To save episodes' duration
        self.episode_reward = []
        self.current_episode_reward = 0
        self.episode_length = []
        self.current_episode_length = 0
        self.episode_time = []
        self.initial_angle = [0.0]
        self.final_angle = []
        self.bandera = False
        self.delays = []
        self.time1 = 0
        self.done = False
        self.steps_antes_de_done = [0]
        # To see whether the 1st measure after a reset is done on time (before the 1st step):
        self.steps_before_1st_measure = [0]

        self.total_steps = 0
        self.total_measures = 0
        
        # Variables for values on each step:
        self.step_time = []
        self.step_angle = []
        self.step_action = []
        self.step_done = []

        self.measures = []
        self.measure_times = []
        self.time0 = rospy.get_time()
        #----------------------------------------------------------



    def ground_truth_callback(self, msg):

        """ Callback function for ground truth data.

        Args:
            msg (Odometry): The ground truth odometry message. """
        
        #Debugging-------------------------------------------------
        self.total_measures += 1
        if self.bandera: 
            self.delays += [rospy.get_time()]
        #----------------------------------------------------------
        
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
        
        self.new_measure = True
        #Debugging-------------------------------------------------
        self.measures += [self.current_angle]
        self.measure_times += [rospy.get_time()+self.time0]
        if self.bandera: 
            self.initial_angle += [self.current_angle]
            self.bandera = False
            self.steps_before_1st_measure += [0]
        #----------------------------------------------------------
        

    def step(self, action):

        """ Perform a simulation step in the environment.

        Args:
            action (float): The action to take.

        Returns:
            observation (numpy.ndarray): The observation of the environment.
            reward (float): The reward obtained from the action.
            done (bool): Whether the episode is done or not.
            info (dict): Additional information about the step. """

        
        # For the time being, we will not be scaling the action.
        # scaled_action = self.scale_action(action, 1.0)

        time1 = time.time()

        vel=Twist()
        vel.linear.x  = action
        vel.linear.y  = 0
        vel.linear.z  = 0
        vel.angular.x = 0
        vel.angular.y = 0
        vel.angular.z = 0
        
        self.new_measure = False
        self.pub.publish(vel)

        # Wait until a new measure is available
        while(self.new_measure == False):
            pass
        self.new_measure = False

        reward = self.get_reward()

        # Check if the episode is done
        terminated = abs(self.current_angle) > self.threshold_angle or\
                     abs(self.position_x)    > self.threshold_position or\
                     abs(self.position_y)    > self.threshold_position 

    
        self.current_step += 1
        truncated = self.current_step >= self.max_steps

        done = terminated or truncated
        
        #Debugging-------------------------------------------------
        self.total_steps += 1
        self.current_episode_reward += reward
        self.current_episode_length += 1
        self.time1 = rospy.get_time() #I will measure time using the simulation time
        
        self.step_time += [rospy.get_time()+self.time0]#[self.time1]
        self.step_angle += [self.current_angle]
        self.step_action += [action[0]]
        self.step_done += [done]
        if self.done:
            print("ERROR: The episode was done on the previous step, but it hasn't reset yet and it's makeing another step.")
            self.steps_antes_de_done[-1] += 1
        self.done = done
        if done:
            self.episode_time += [rospy.get_time()+self.time0]
            self.final_angle += [self.current_angle]
            self.episode_reward += [self.current_episode_reward]
            self.episode_length += [self.current_episode_length]
            self.current_episode_reward = 0
            self.current_episode_length = 0
        if self.bandera:
            self.steps_before_1st_measure[-1] += 1
        #----------------------------------------------------------
        
        if done:
            vel.linear.x  = 0
            self.pub.publish(vel)

        # environment observation
        return  np.array([self.current_angle, self.angular_y,
                          self.position_x,    self.position_y,
                          self.velocity_x,    self.velocity_y], dtype=float), reward, terminated, truncated, {}

    # We have normalized our action space to align with best practices, and if needed, we should re-scale it.
    # TODO: this is not yet being used.
    def scale_action(self, action, factor):
        return factor * action

    def reset(self, **kwargs):

        """ Reset the environment.

        Returns:
            observation (numpy.ndarray): The initial observation of the environment. """

        #Debugging-------------------------------------------------
        # self.initial_angle += [pitch]
        self.done = False
        self.steps_antes_de_done += [0]
        self.time1 = 0
        self.time0 += rospy.get_time()
        #----------------------------------------------------------
        
        time_final = rospy.get_time()
        rospy.wait_for_service('/gazebo/reset_simulation')
        self.reset_simulation_client()
        time_inicial = rospy.get_time()
        # Now we wait until the reset is successfully performed (and hence the time diminishes)
        while((time_final < time_inicial) or (abs(self.current_angle) >= self.threshold_angle)):
            time_inicial = rospy.get_time()

        #Debugging-------------------------------------------------
        self.bandera = True
        #----------------------------------------------------------
        self.current_angle = 0 
        self.angular_y     = 0
        self.position_x    = 0
        self.position_y    = 0
        self.velocity_x    = 0
        self.velocity_y    = 0
        
        self.current_step = 0
        info = {}

        

        return  np.array([0, 0,
                          0, 0,
                          0, 0], dtype=float), info

    def render(self):
        pass

    def get_reward(self):

        """
        Calculate the reward based on the current_angle, position and velocity of the robot.

        Returns:
            reward (float): The reward value based on the current angle.
        """
        # 
        done = abs(self.current_angle) > self.threshold_angle or\
               abs(self.position_x)    > self.threshold_position or\
               abs(self.position_y)    > self.threshold_position 

        angle_correction    =  2.0 - abs(self.current_angle) * 10
        position_correction = -(abs(self.position_x) + abs(self.position_y))*0.3
        velocity_correction = -(abs(self.velocity_x) + abs(self.velocity_y))*2

        return -200.0 if done else angle_correction + position_correction + velocity_correction