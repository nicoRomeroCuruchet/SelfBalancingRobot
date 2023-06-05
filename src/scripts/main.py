import time
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

    # Hyperparameters:
    policy = MlpPolicy
    lr = 3e-4
    n_steps = 1000 #Number of steps to run for each environment per update
    # max_timesteps_per_episode = 3000 #(Esto es del PPO viejo TODO: borrar)
    # timesteps_per_batch = 7048 #(Esto es del PPO viejo TODO: borrar)
    batch_size = 128 #minibatch size (default=64)
    gamma = 0.99
    clip_range = 0.2
    # n_updates_per_iteration = 10 #(Esto es del PPO viejo TODO: no sé qué es)
    
    # Create the environment:
    env = SelfBalancingRobot(max_steps=3e3)
    # Wrap the environment:
    env = DummyVecEnv([lambda: Monitor(env)])
    # env = NormalizeActionWrapper(env) #TODO

    eval_env = env #TODO Debería ser diferente?
    # Use deterministic actions for evaluation
    eval_callback = EvalCallback(eval_env, best_model_save_path="./logs/",
                                log_path="./logs/", eval_freq=5*n_steps,
                                deterministic=True, render=False)
    
    plot_callback = PlotCallback(check_freq = 100)

    save_dir = "./logs/"
    verbose = 0
    try:
        model = PPO.load(f"{save_dir}/best_model.zip", policy, env, learning_rate=lr, n_steps=n_steps,
                    batch_size=batch_size, gamma=gamma, clip_range=clip_range,
                    verbose=verbose)
        print("Loaded previously saved best model")
    except:
        model = PPO(policy, env, learning_rate=lr, n_steps=n_steps,
                    batch_size=batch_size, gamma=gamma, clip_range=clip_range,
                    verbose=verbose)
        print("Created new model")

    
    # Train the agent for 10000 steps
    start = time.time()
    model.learn(total_timesteps=2_000, progress_bar=True, callback=[plot_callback, eval_callback])
    end = time.time()
    print ("Total training time: ", end - start)

    # Evaluate the best trained agent
    print("Loading best model so far")
    model = PPO.load(f"{save_dir}/best_model.zip", env, verbose=verbose)
    print("Evaluating best model so far")
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
    
if __name__ == '__main__':
    args = get_args() # Parse arguments from command line
    main(args)