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

    # Train the agent
    model.learn(total_timesteps=400000, progress_bar=True)

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