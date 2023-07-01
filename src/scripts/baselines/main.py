import sys
import time
import torch
import rospy
from std_srvs.srv import Empty
from arguments import get_args
from env import SelfBalancingRobotBaseLine
from stable_baselines3.common.env_checker import check_env

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
# from callback import PlotCallback
from debugging import plot_episodes, plot_debugging

import pdb

# 1) Run main.py
# 2) Run tensorboard --logdir ./log/
# 


def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)
    # Create the environment
    env = SelfBalancingRobotBaseLine(max_timesteps_per_episode=1000)

    try:
        # Loads already existing model
        model = PPO.load("self_balancing_robot")
        model.set_env(env)
        print("Loading existing model")
    except:
        # Model does not exist. Create a new one.
        model = PPO(MlpPolicy, 
                    env, 
                    verbose=False, 
                    device=device,
                    tensorboard_log="./log/")
        print("Creating new model")

    # # Train the agent
    # model.learn(total_timesteps=1_000, progress_bar=True)

    # # Save the model
    # model.save("self_balancing_robot")
    # print("Finished training")
    # plot_episodes(env)
    # print("Finished plot1")
    # pdb.run(plot_debugging(env))
    # plot_debugging(env)
    
    # print("Finished plot2")
    
    # Create the service to reset the simulation
    pause = rospy.ServiceProxy('/gazebo/pause_physics',Empty)
    unpause = rospy.ServiceProxy('/gazebo/unpause_physics',Empty)

    print("Evaluating the performance")
    N = 3
    for k in range(0,N):
    # k = 0
    # while True:
    #     k += 1
        print("Episode number ", k)
        obs, _ = env.reset()
        for i in range(1000):
            action, action_probs = model.predict(obs, deterministic=True)
            obs, rewards, terminated, truncated , info = env.step(action)
            done = terminated or truncated
            env.render()
            if done:
                print("Episode finished after {} timesteps".format(i+1))
                
                #Final analysis of the episode:
                pause()
                plot_debugging(env)
                unpause()

                break

if __name__ == '__main__':


    args = get_args() # Parse arguments from command line
    hyperparameters = {
    'timesteps_per_batch': 7048, 
    'max_timesteps_per_episode': 3000, 
    'gamma': 0.99, 
    'n_updates_per_iteration': 10,
    'lr': 3e-4, 
    'clip': 0.2,
    }

    main(args)