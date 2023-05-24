import gym
import math
import rospy
from ppo import PPO
from network import FeedForwardNN
from deep_rl import SelfBalancingRobot
from pid_controller import PIDController
from gazebo_msgs.srv import  ApplyBodyWrenchRequest, ApplyBodyWrench

def apply_force_to_link(link_name, force, duration):

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

def main_pid():

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

    # Robot connection 
    robot = SelfBalancingRobot()
    # ~ loop rate
    rate = rospy.Rate(200)
    # Controller pitch

    apply_force_to_link('base_link', [10, 0, 0], 0.01)
    while not rospy.is_shutdown():

        # get robot position 
        #robot_position     = math.sqrt(robot.position.x**2 + robot.position.y**2) if robot.position is not None else 0
        #s                  =  -1.0 if robot.position is not None and robot.position.x > 0 else 1.0
        #pid_angle.setpoint = s*pid_position.update(robot_position) 

        # get the pitch (current_state)
    
        pitch = robot.current_angle if robot.current_angle is not None else 0.0
        control_output     = pid_angle.update(pitch)
        # step control action
        robot.current_state, reward, done, _ = robot.step(control_output)

        if done: 
            robot.reset()
           

        rate.sleep()                # Sleep to maintain the loop rate




"""
    This file is the executable for running PPO. It is based on this medium article: 
    https://medium.com/@eyyu/coding-ppo-from-scratch-with-pytorch-part-1-4-613dfc1b14c8
"""

import gym
import sys
import torch

from arguments import get_args
from ppo import PPO
from network import FeedForwardNN
from eval_policy import eval_policy

def train(env, hyperparameters, actor_model, critic_model):
    """
        Trains the model.

        Parameters:
            env - the environment to train on
            hyperparameters - a dict of hyperparameters to use, defined in main
            actor_model - the actor model to load in if we want to continue training
            critic_model - the critic model to load in if we want to continue training

        Return:
            None
    """ 
    print(f"Training", flush=True)

    # Create a model for PPO.
    model = PPO(policy_class=FeedForwardNN, env=env, **hyperparameters)

    # Tries to load in an existing actor/critic model to continue training on
    if actor_model != '' and critic_model != '':
        print(f"Loading in {actor_model} and {critic_model}...", flush=True)
        model.actor.load_state_dict(torch.load(actor_model))
        model.critic.load_state_dict(torch.load(critic_model))
        print(f"Successfully loaded.", flush=True)
    elif actor_model != '' or critic_model != '': # Don't train from scratch if user accidentally forgets actor/critic model
        print(f"Error: Either specify both actor/critic models or none at all. We don't want to accidentally override anything!")
        sys.exit(0)
    else:
        print(f"Training from scratch.", flush=True)

    # Train the PPO model with a specified total timesteps
    # NOTE: You can change the total timesteps here, I put a big number just because
    # you can kill the process whenever you feel like PPO is converging
    model.learn(total_timesteps=800_000_000)

def test(env, actor_model):
    """
        Tests the model.

        Parameters:
            env - the environment to test the policy on
            actor_model - the actor model to load in

        Return:
            None
    """
    print(f"Testing {actor_model}", flush=True)

    # If the actor model is not specified, then exit
    if actor_model == '':
        print(f"Didn't specify model file. Exiting.", flush=True)
        sys.exit(0)

    # Extract out dimensions of observation and action spaces
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # Build our policy the same way we build our actor model in PPO
    policy = FeedForwardNN(obs_dim, act_dim)

    # Load in the actor model saved by the PPO algorithm
    policy.load_state_dict(torch.load(actor_model))

    # Evaluate our policy with a separate module, eval_policy, to demonstrate
    # that once we are done training the model/policy with ppo.py, we no longer need
    # ppo.py since it only contains the training algorithm. The model/policy itself exists
    # independently as a binary file that can be loaded in with torch.
    eval_policy(policy=policy, env=env, render=True)



def main(args):
    """
        The main function to run.

        Parameters:
            args - the arguments parsed from command line

        Return:
            None
    """
    # NOTE: Here's where you can set hyperparameters for PPO. I don't include them as part of
    # ArgumentParser because it's too annoying to type them every time at command line. Instead, you can change them here.
    # To see a list of hyperparameters, look in ppo.py at function _init_hyperparameters
    hyperparameters = {
                'timesteps_per_batch': 6048, 
                'max_timesteps_per_episode': 3000, 
                'gamma': 0.9, 
                'n_updates_per_iteration': 10,
                'lr': 3e-4, 
                'clip': 0.2,
                'render': False,
                'render_every_i': 10
              }

    # Creates the environment we'll be running. If you want to replace with your own
    # custom environment, note that it must inherit Gym and have both continuous
    # observation and action spaces.
    env = SelfBalancingRobot()

    # Train or test, depending on the mode specified
    if args.mode == 'train':
        train(env=env, hyperparameters=hyperparameters, actor_model=args.actor_model, critic_model=args.critic_model)
    else:
        test(env=env, actor_model=args.actor_model)

if __name__ == '__main__':
    args = get_args() # Parse arguments from command line
    main(args)