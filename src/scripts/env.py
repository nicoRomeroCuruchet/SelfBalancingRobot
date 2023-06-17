import gymnasium as gym
import time
import math
import rospy
import numpy as np
from gazebo_msgs.msg import ModelState
from tf.transformations import quaternion_from_euler, euler_from_quaternion
from std_srvs.srv import Empty
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Pose, Point, Quaternion

class SelfBalancingRobot(gym.Env):

    """ A reinforcement learning environment for a self-balancing robot simulation.

    This class represents an environment for training and testing the control
    of a self-balancing robot. It provides methods for taking actions, receiving
    observations, calculating rewards, and resetting the environment.

    Note:
        This class is designed to be used with the OpenAI Gym reinforcement learning framework. """

    def __init__(self, **kwargs):

        """ Initialize the SelfBalancingRobot environment. """

        rospy.init_node('controller_node')

        # Create the publisher to control the velocity of the robot
        self.pub                     = rospy.Publisher('/self_balancing_robot/cmd_vel', Twist, queue_size=1)
        # Create the publisher to set the state of the robot
        self.set_model_state_pub     = rospy.Publisher('/gazebo/set_model_state', ModelState, queue_size=1)
        # Create the subscriber to get the ground truth data
        self.sub_ground              = rospy.Subscriber('/self_balancing_robot/ground_truth/state', Odometry, self.ground_truth_callback)
        # Create the service to reset the simulation
        self.reset_simulation_client = rospy.ServiceProxy('/gazebo/reset_simulation',Empty)

        # Set the gym environment parameters
        self.reward_range      = (-float('inf'), float('inf'))
        # self.action_space      = gym.spaces.Box(low=-5, high=5, shape=(1,), dtype=float)
        self.action_space      = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=float)
        self.observation_space = gym.spaces.Box(low=np.array([-math.pi/2, -float('inf'), -1, -1, -float('inf'), -float('inf')]), 
                                                high=np.array([math.pi/2,  float('inf'),  1,  1,  float('inf'),  float('inf')]), dtype=float)
        
        # Create the environment and stop de the robot
        self.vel           = Twist()
        self.vel.linear.x  = 0
        self.vel.linear.y  = 0
        self.vel.linear.z  = 0
        self.vel.angular.x = 0
        self.vel.angular.y = 0
        self.vel.angular.z = 0
        self.pub.publish(self.vel)
        # refresh rate in ground truth
        self.time_interval = 1/200 #0.005   # running at ~200 Hz
        self.current_angle = 0       # the pitch angle of the robot
        # storage callback data
        self.angular_y     = 0
        self.imu_data      = None
        self.position_x    = None
        self.position_y    = None
        self.velocity_x    = None
        self.velocity_y    = None
        # Angle thresholds angle expresed in radians and position in meters
        self.threshold_angle     = 0.4 #0.4
        self.threshold_position  = 1.0
        self.current_step        = 0
        try:
            self.max_steps       = max_steps
        except:
            self.max_steps       = float('inf')
        
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
        self.time0 = time.time()

    def ground_truth_callback(self, msg):

        """ Callback function for ground truth data.

        Args:
            msg (Odometry): The ground truth odometry message. """
        
        # Position of the robot
        self.total_measures += 1
        if self.bandera: 
            self.delays += [rospy.get_time()]
            # print("Time for first measure: ", rospy.get_time())
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
        self.measures += [self.current_angle]
        self.measure_times += [time.time()-self.time0]
        if self.bandera: 
            self.initial_angle += [self.current_angle]
            self.bandera = False
            self.steps_before_1st_measure += [0]

    def step(self, action):

        """ Perform a simulation step in the environment.

        Args:
            action (float): The action to take.

        Returns:
            observation (numpy.ndarray): The observation of the environment.
            reward (float): The reward obtained from the action.
            done (bool): Whether the episode is done or not.
            info (dict): Additional information about the step. """
        
        self.current_step += 1
        self.total_steps += 1

        
        vel=Twist()
        # print(action[0])
        vel.linear.x  = 0 if self.bandera else action[0] #TODO: borrar esta prueba
        vel.linear.y  = 0
        vel.linear.z  = 0
        vel.angular.x = 0
        vel.angular.y = 0
        vel.angular.z = 0

        # Check if time interval has passed
        # the ground truth is published at 200 Hz. Check it in the robot.urdf file, in the sensors section.
        interval = rospy.get_time() - self.time1
        if(interval < self.time_interval):
            rospy.sleep(self.time_interval - interval)
        
        # while(interval < self.time_interval):
        #     interval = rospy.get_time() - self.time1
        
        # interval = time.time() - self.time1
        # if(interval < self.time_interval):
        #     time.sleep(self.time_interval - interval)
        self.time1 = rospy.get_time() #I will measure time using the simulation time
        
        self.step_time += [time.time()-self.time0]#[self.time1]
        self.step_angle += [self.current_angle]
        self.step_action += [action[0]]
        self.pub.publish(vel)


        angle = self.current_angle #This auxiliary variable prevents a change between 
                                    #the angle used for terminated and the one used for final_angle
        terminated = abs(angle) > self.threshold_angle or\
                    abs(self.position_x) > self.threshold_position or\
                    abs(self.position_y) > self.threshold_position 
        truncated = self.current_step >= self.max_steps
        done = terminated or truncated
        self.step_done += [done]
        if self.done:
            print("En el step anterior había terminado, pero no resetié todavía y ya vino otro step.")
            self.steps_antes_de_done[-1] += 1
        self.done = done #TODO: borrar después de debuggear
        if done:
            self.episode_time += [rospy.get_time()]
            self.final_angle += [angle]
            # print("Done")
            vel.linear.x  = 0
            self.pub.publish(vel)
            # print("Final time: ", rospy.get_time())
        reward = self.get_reward()
        
        self.current_episode_reward += reward
        self.current_episode_length += 1
        if done:
            self.episode_reward += [self.current_episode_reward]
            self.episode_length += [self.current_episode_length]
            self.current_episode_reward = 0
            self.current_episode_length = 0
        if self.bandera:
            self.steps_before_1st_measure[-1] += 1

        # environment observation
        return  np.array([self.current_angle, self.angular_y,
                          self.position_x,    self.position_y,
                          self.velocity_x,    self.velocity_y], dtype=float), reward, terminated, truncated, {}
    
    def set_robot_pose(self, x, y, z, roll, pitch, yaw):
        # Create the ModelState message
        model_state = ModelState()
        model_state.model_name = 'self_balancing_robot'

        # Set the pose
        model_state.pose.position = Point(x, y, z)
        model_state.pose.orientation = Quaternion(*quaternion_from_euler(roll, pitch, yaw))

        rospy.wait_for_service('/gazebo/set_model_state')
        # Publish the message
        self.set_model_state_pub.publish(model_state)
        # rospy.sleep(1)  # Wait for the pose to be set
        # #TODO: necesito esperar tanto?

    def reset(self, **kwargs):

        """ Reset the environment.

        Returns:
            observation (numpy.ndarray): The initial observation of the environment. """
        self.current_step = 0
        
        rospy.wait_for_service('/gazebo/reset_simulation')
        self.reset_simulation_client()
        
        x = 0
        y = 0
        roll = 0
        pitch = 0 #TODO np.random.uniform(low=-0.2, high=0.2, size=None)
        # self.initial_angle += [pitch]
        self.bandera = True
        self.done = False
        self.steps_antes_de_done += [0]

        # Fuerzo el valor medido para que no se quede con uno que está fuera de rango:
        self.current_angle = pitch #TODO: esto está bien?

        # print(pitch)
        yaw = 0
        # Height calculated to have the robot standing on the surface
        z = (0.1 + 0.0125) * np.cos(pitch) + 0.0325
        
        self.set_robot_pose(x, y, z, roll, pitch, yaw)

        self.time1 = 0
        
        info = {}
        return  np.array([0, 0,
                          0, 0,
                          0, 0], dtype=float), info

    def render(self):
        pass

    def close(self):
        print("Close Env")
        rospy.wait_for_service('/gazebo/reset_simulation')
        self.reset_simulation_client()

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
        velocity_correction = -(abs(self.velocity_x) + abs(self.velocity_y))*0.3

        return -200.0 if done else angle_correction + position_correction + velocity_correction