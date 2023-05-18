import os
import gym
import math
import rospy
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

    """
    Custom Gym environment for controlling an inverted pendulum model in ROS.

    The environment interacts with ROS topics '/cmd_vel' and '/imu' to send
    velocity commands and receive IMU data for the pendulum's inclination angle.

    Observation Space:
        - Type: Box
        - Shape: (1,)
        - Range: (-inf, inf)
        - Description: Current inclination angle of the pendulum

    Action Space:
        - Type: Box
        - Shape: (1,)
        - Range: [-1, 1]
        - Description: Linear velocity command to control the pendulum

    Reward Range:
        - Type: Tuple
        - Values: (-inf, inf)
        - Description: The range of possible reward values

    """

    def __init__(self):

        """ Initialize the InvertedPendulumEnv class.

        Sets up the ROS node, publishers, subscribers, and defines the observation
        and action spaces. """

        rospy.init_node('controller_node')

        self.pub               = rospy.Publisher('/self_balancing_robot/cmd_vel', Twist, queue_size=1)
        self.sub               = rospy.Subscriber('/imu', Imu, self.imu_callback)
        self.sub_ground        = rospy.Subscriber('/self_balancing_robot/ground_truth/state', Odometry, self.ground_truth_callback)
        self.reward_range      = (-float('inf'), float('inf'))
        self.action_space      = gym.spaces.Box(low=-10, high=10, shape=(1,), dtype=float)
        self.observation_space = gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=(1,), dtype=float)
        self.imu_data          = None
        self.current_state     = None
        self.position          = None
        self.reset()

    def imu_callback(self, data):

        """ Callback function for receiving IMU data from the '/imu' topic.

        Args:

            data (Imu): IMU data message """

        self.imu_data = data

        # roll, picht, yaw
        _, self.current_state,_ = euler_from_quaternion([self.imu_data.orientation.x,
                                                         self.imu_data.orientation.y,
                                                         self.imu_data.orientation.z,
                                                         self.imu_data.orientation.w])


    def ground_truth_callback(self, msg):
        
        # Handle received message
        # Access the robot's state information from 'msg' object
        self.position = msg.pose.pose.position
        #self.orientation = msg.pose.pose.orientation


    def step(self, action):

        """ Perform a step in the environment.

        Publishes the given action as a velocity command to the 'self_balancing_robot/cmd_vel' topic,
        waits for a duration, computes the reward, and returns the updated state,
        reward, done flag, and additional information.

        Args:
            action (array-like): Action to be taken

        Returns:
            tuple: Tuple containing the updated state, reward, done flag, and additional information """

        vel = Twist()
        vel.linear.x = action
        self.pub.publish(vel)
        reward = self.get_reward()
        done = False
        return self.current_state, reward, done, {}

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
        rospy.sleep(.5)
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
            pose.position.x = 0  # Adjust the position as needed
            pose.position.y = 0
            pose.position.z = 0.2
            pose.orientation.x = 0
            pose.orientation.y = 0
            pose.orientation.z = 0
            pose.orientation.w = 0
            response = spawn_model(model_name, urdf_xml, "", pose, "world")

        except rospy.ServiceException as e:
            rospy.loginfo(f"Failed to spawn the model: {str(e)}")

        rospy.sleep(0.5)
        self.imu_data = None
        while self.imu_data is None:
            try:
                self.imu_data = rospy.wait_for_message('/imu', Imu, timeout=1)
                self.step(0)
            except:
                pass
       
        # picht 
        _, self.current_state,_ = euler_from_quaternion([self.imu_data.orientation.x,
                                                         self.imu_data.orientation.y,
                                                         self.imu_data.orientation.z,
                                                         self.imu_data.orientation.w])
        return self.current_state

    def get_reward(self, error=0.10):

        """ Calculate the reward based on the current state.

        If the current state of the pendulum is outside the acceptable range,
        a negative reward is given. Otherwise, a positive reward is given.

        Returns:
            float: Reward value """

        return -1.0 if abs(self.current_state) > error else 1.0
