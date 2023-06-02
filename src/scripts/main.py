import sys
import torch
# from ppo import PPO
from arguments import get_args
# from network import FeedForwardNN
# from eval_policy import eval_policy
from env import SelfBalancingRobot
from callback import PlotCallback
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy

def main(args):

    """
    The main function to run.

    Parameters:
        args - the arguments parsed from command line

    Return:
        None
    """

    # # NOTE: Here's where you can set hyperparameters for PPO. I don't include them as part of
    # # ArgumentParser because it's too annoying to type them every time at command line. Instead, you can change them here.
    # # To see a list of hyperparameters, look in ppo.py at function _init_hyperparameters
    # hyperparameters = {
    #             'timesteps_per_batch': 7048, 
    #             'max_timesteps_per_episode': 3000, 
    #             'gamma': 0.99, 
    #             'n_updates_per_iteration': 10,
    #             'lr': 3e-4, 
    #             'clip': 0.2,
    #             'render': False,
    #             'render_every_i': 10
    #           }

    # Creates the environment we'll be running. If you want to replace with your own
    # custom environment, note that it must inherit Gym and have both continuous
    # observation and action spaces.
    env = DummyVecEnv([lambda: Monitor(SelfBalancingRobot(max_steps=10e10))])

    #TODO: borrar
    # print(env)
    # # episode_rewards = env.env_method(attr_name=get_episode_rewards, indices=0)
    # env_aux = env.envs[0]
    # print(env_aux)
    # episode_rewards = env.envs[0].get_episode_rewards()
    
    # # Es env_method o get_attr? Con ambos me tira error
    
    # print(episode_rewards)

    # env = NormalizeActionWrapper(env) #TODO

    eval_env = env #TODO Debería ser diferente?
    # Use deterministic actions for evaluation
    eval_callback = EvalCallback(eval_env, best_model_save_path="./logs/",
                                log_path="./logs/", eval_freq=500,
                                deterministic=True, render=False)
    
    plot_callback = PlotCallback(check_freq = 100)

    save_dir = "./logs/"
    verbose = 0
    try:
        model = PPO.load(f"{save_dir}/best_model.zip", env, verbose=verbose)
        print("Loaded previously saved best model")
    except:
        model = PPO(MlpPolicy, env, verbose=verbose)
        print("Created new model")

    # Train the agent for 10000 steps
    model.learn(total_timesteps=1_000, progress_bar=True, callback=[plot_callback, eval_callback])

    # # Evaluate the best trained agent
    # print("Loading best model so far")
    # model = PPO.load(f"{save_dir}/best_model.zip", env, verbose=verbose)
    # print("Evaluating best model so far")
    # mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
    # print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
    
if __name__ == '__main__':
    args = get_args() # Parse arguments from command line
    main(args)