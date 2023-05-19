import rospy
import math
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from deep_rl import SelfBalancingRobot
from pid_controller import PIDController
from gazebo_msgs.srv import  ApplyBodyWrenchRequest, ApplyBodyWrench

def apply_force_to_link(link_name, force, duration):

    """ Apply a force to a specified link in Gazebo simulation.

    Args:
        link_name (str): Name of the link to apply the force to.
        force (list[float]): List of three float values representing the force in the x, y, and z directions.
        duration (float): Duration in seconds for which the force should be applied.

    Returns:
        None

    Raises:
        rospy.ServiceException: If there is an error while calling the 'apply_body_wrench' service. """

    rospy.wait_for_service('/gazebo/apply_body_wrench')
    try:
        apply_wrench = rospy.ServiceProxy('/gazebo/apply_body_wrench', ApplyBodyWrench)
        request = ApplyBodyWrenchRequest()
        request.body_name = link_name
        request.wrench.force.x = force[0]
        request.wrench.force.y = force[1]
        request.wrench.force.z = force[2]
        request.duration = rospy.Duration(duration)
        response = apply_wrench(request)
        if response.success:
            rospy.loginfo(f"Force applied to link '{link_name}' successfully.")
        else:
            rospy.logerr(f"Failed to apply force to link '{link_name}': {response.status_message}")
    except rospy.ServiceException as e:
        rospy.logerr(f"Failed to apply force to link '{link_name}': {e}")

def main():

    # Robot connection 
    robot = SelfBalancingRobot()
    # ~ loop rate
    rate = rospy.Rate(200)
    rospy.sleep(0.1)
    # Controller pitch
    pid_angle    = PIDController(kp=40.0, 
                                 ki=2.5e-5, 
                                 kd=1200, 
                                 setpoint=0.0,
                                 output_limit=(-5,5),
                                 integral_limit=(-1,1))

    pid_position = PIDController(kp=0.01,
                                 ki=0,
                                 kd=100, 
                                 output_limit=(-0.1,0.1),
                                 integral_limit=(-0.01,0.01),
                                 setpoint=0.0)

    apply_force_to_link('base_link', [10, 0, 0], 0.01)

    while not rospy.is_shutdown():

        # get robot position 
        robot_position     = math.sqrt(robot.position.x**2 + robot.position.y**2) if robot.position is not None else 0
        s                  =  -1.0 if robot.position is not None and robot.position.x > 0 else 1.0
        pid_angle.setpoint = s*pid_position.update(robot_position) 

        # get the pitch (current_state)
        pitch = robot.current_state if robot.current_state is not None else 0.0
        control_output     = pid_angle.update(pitch)
        # step control action
        robot.current_state, reward, done, _ = robot.step(control_output)

        if done: robot.reset()

        rate.sleep()                # Sleep to maintain the loop rate


import gym
from keras import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

if __name__ == '__main__':

    try:
        main()
        #env = selfbalancingRobot()

        #model = PPO("MlpPolicy", env, verbose=1)
        #model.learn(total_timesteps=10000)
        #mean_reward, _ = model.evaluate(env, n_eval_episodes=10)
        #print(f"Mean reward: {mean_reward:.2f} +/- {_:.2f}")
    except rospy.ROSInterruptException: 
        pass
