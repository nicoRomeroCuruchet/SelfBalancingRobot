import time
import matplotlib.pyplot as plt
import numpy as np
# from ppo import PPO
from arguments import get_args
# from network import FeedForwardNN
# from eval_policy import eval_policy
from env import SelfBalancingRobot
from normalize_action_space import NormalizeActionWrapper
from callback import PlotCallback, DebuggingPlotCallback
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy

def plot_episodes(env):
    # episode_rewards = env.envs[0].get_episode_rewards()
    # episode_lengths = env.envs[0].get_episode_lengths()
    
    # episode_rewards = env.get_episode_rewards()
    # episode_lengths = env.get_episode_lengths()
    episode_rewards = env.episode_reward
    episode_lengths = env.episode_length
    episode_times = env.episode_time
    episode_initial_angle = env.initial_angle
    episode_final_angle = env.final_angle
    delays = env.delays
    steps_before_1st_measure = env.steps_before_1st_measure
    steps_antes_de_done = env.steps_antes_de_done
    
    # number_of_episode = np.arange(1.,len(episode_rewards)+1)
    number_of_episode = np.arange(1.,len(episode_times)+1)
    
    plt.ion()
    # fig = plt.figure(figsize=(12,4))
    fig = plt.figure(figsize=(15,9)) #(15,9)
    ax1 = fig.add_subplot(331)
    ax1.set_xlabel('N° Episode')
    ax1.set_ylabel('Episode Reward')
    ax1.set_title('Episodes Total Reward')
    ax1.plot(number_of_episode,episode_rewards)
    ax2 = fig.add_subplot(332)
    ax2.set_xlabel('N° Episode')
    ax2.set_ylabel('Episode Length')
    ax2.set_title('Episodes Length')
    ax2.plot(number_of_episode,episode_lengths)
    ax3 = fig.add_subplot(333)
    ax3.set_xlabel('N° Episode')
    ax3.set_ylabel('Episode Time')
    ax3.set_title('Episodes Times')
    ax3.plot(number_of_episode,episode_times)
    ax4 = fig.add_subplot(334)
    ax4.set_xlabel('N° Episode')
    ax4.set_ylabel('Initial Angle')
    ax4.set_title('Episodes Initial Angle')
    ax4.plot(episode_initial_angle)#number_of_episode,episode_initial_angle)
    ax4.plot(number_of_episode, env.threshold_angle*np.ones(len(number_of_episode)))
    ax4.plot(number_of_episode, -env.threshold_angle*np.ones(len(number_of_episode)))
    ax5 = fig.add_subplot(335)
    ax5.set_xlabel('N° Episode')
    ax5.set_ylabel('Final Angle')
    ax5.set_title('Episodes Final Angle')
    ax5.plot(episode_final_angle)#number_of_episode,episode_final_angle)
    ax5.plot(number_of_episode, env.threshold_angle*np.ones(len(number_of_episode)))
    ax5.plot(number_of_episode, -env.threshold_angle*np.ones(len(number_of_episode)))
    ax6 = fig.add_subplot(336)
    ax6.set_xlabel('N° Episode')
    ax6.set_ylabel('Delay')
    ax6.set_title('Delay between reset and 1st state measurement')
    ax6.plot(delays)
    ax7 = fig.add_subplot(337)
    ax7.set_xlabel('N° Episode')
    ax7.set_ylabel('N° of steps before 1st measure after reset')
    ax7.set_title('Steps before 1st measure')
    ax7.plot(steps_before_1st_measure)
    ax8 = fig.add_subplot(338)
    ax8.set_xlabel('N° Episode')
    ax8.set_ylabel('Steps antes de done')
    ax8.set_title('Steps antes de done')
    ax8.plot(steps_antes_de_done)
    plt.savefig('logs/rewards_and_lengths_plot.svg', format='svg')

    fin = 50
    step_time = env.step_time#[0:fin]
    step_angle = env.step_angle#[0:fin]
    step_action = env.step_action#[0:fin]
    step_done = env.step_done#[0:fin]
    measures = env.measures
    measure_times = env.measure_times
    plt.ion()
    fig = plt.figure(figsize=(9,9))
    ax1 = fig.add_subplot(221)
    ax1.set_xlabel('N° Step')
    ax1.set_ylabel('time')
    ax1.set_title('Step time')
    ax1.plot(step_time)
    ax2 = fig.add_subplot(222)
    # ax2.set_xlabel('N° Step')
    ax2.set_xlabel('Times')
    ax2.set_ylabel('Angle')
    ax2.set_title('Angle')
    ax2.plot(step_time, step_angle)
    ax2.plot(measure_times, measures,'--')
    # ax2.plot(step_time, step_action)
    ax3 = fig.add_subplot(223)
    ax3.set_xlabel('N° Step')
    ax3.set_ylabel('Action')
    ax3.set_title('Action')
    # ax3.plot(step_action)
    ax3.plot(step_time, step_action)
    aux = np.ones((len(measure_times),1))
    for i in range(0, len(measure_times), 2):
        aux[i] = -1
    ax3.plot(measure_times, aux)
    # ax3 = fig.add_subplot(223)
    # ax3.set_xlabel('Time')
    # ax3.set_ylabel('Angle measured')
    # ax3.set_title('Angle measured')
    # ax3.plot(measure_times, measures)
    ax4 = fig.add_subplot(224)
    ax4.set_xlabel('N° Step')
    ax4.set_ylabel('Done')
    ax4.set_title('Done')
    ax4.plot(step_time, step_done)

    plt.show(block=True)


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
    total_timesteps = 1_000
    plot_freq = 10_000
    lr = 3e-4
    # timesteps_per_batch = 7048 #(Esto es del PPO viejo, es equivalente al n_steps del PPO de SB3. TODO: borrar)
    n_steps = 500 #Number of steps to run for each environment per update
    max_timesteps_per_episode = 3000 #(Esto es del PPO viejo TODO: borrar)
    batch_size = 250 #minibatch size (default=64)
    gamma = 0.99
    clip_range = 0.2
    # n_updates_per_iteration = 10 #(Esto es del PPO viejo TODO: no sé qué es)
    
    # Create the environment:
    env = SelfBalancingRobot(max_steps=max_timesteps_per_episode)
    # Wrap the environment:
    # env = NormalizeActionWrapper(env) #TODO
    # env = Monitor(env)
    # env = DummyVecEnv([lambda: env])
    
    eval_env = env #TODO Debería ser diferente?
    # Use deterministic actions for evaluation
    eval_callback = EvalCallback(eval_env, best_model_save_path="./logs/",
                                log_path="./logs/", eval_freq=10*n_steps,#min(10*n_steps,1000),
                                deterministic=True, render=False)
    
    plot_callback = PlotCallback(check_freq = plot_freq)
    debugging_plot_callback = DebuggingPlotCallback(check_freq = plot_freq)

    save_dir = "./logs/"
    verbose = 0
    try:
        model = PPO.load(f"{save_dir}/best_model.zip", policy, env, learning_rate=lr, n_steps=n_steps,
                    batch_size=batch_size, gamma=gamma, clip_range=clip_range,
                    verbose=verbose)
        print("Loaded previously saved best model")
    except:
        try:
            model = PPO.load(f"{save_dir}/final_model2.zip", policy, env, learning_rate=lr, n_steps=n_steps,
                        batch_size=batch_size, gamma=gamma, clip_range=clip_range,
                        verbose=verbose)
            print("Loaded previously saved final model")
        except:
            model = PPO(policy, env, learning_rate=lr, n_steps=n_steps,
                        batch_size=batch_size, gamma=gamma, clip_range=clip_range,
                        verbose=verbose)
            print("Created new model")

    
    # Train the agent
    callback=[plot_callback, debugging_plot_callback, eval_callback]
    start = time.time()
    model.learn(total_timesteps=total_timesteps, progress_bar=True)#, callback=callback)
    end = time.time()
    print ("Total training time: ", end - start)

    # # Evaluate the best trained agent
    # print("Loading best model so far")
    # model = PPO.load(f"{save_dir}/best_model.zip", env, verbose=verbose)
    # print("Evaluating best model so far")
    # mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    # print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
    
    # Save final model
    model.save(f"{save_dir}/final_model")
    # print("Evaluating final model obtained")
    # mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    # print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

    print("Total steps: ", env.total_steps)
    print("Total measures: ", env.total_measures)
    plot_episodes(env)
    
    
if __name__ == '__main__':
    args = get_args() # Parse arguments from command line
    main(args)