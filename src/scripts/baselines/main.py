import sys
import torch
from arguments import get_args
from env import SelfBalancingRobotBaseLine
from stable_baselines3.common.env_checker import check_env

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy

from callback import PlotCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy

# Tania
import time
import matplotlib.pyplot as plt
import numpy as np


# 1) Run main.py
# 2) Run tensorboard --logdir ./log/
# 


def main(args):
    
    env = SelfBalancingRobotBaseLine(max_timesteps_per_episode=10000)

    # Loads already existing model
    try:
        print("Loading existing model")
        model = PPO.load("self_balancing_robot")
        model.set_env(env)
    except:
    # Model does not exist. Create a new one.
        print("Creating new model")
        model = PPO(MlpPolicy, env, verbose=False, tensorboard_log="./log/")


    # Train the agent for 10000 steps callback=[plot_callback, eval_callback]
    model.learn(total_timesteps=300000, progress_bar=True)

    # Save the model
    model.save("self_balancing_robot")

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