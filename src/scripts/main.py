import sys
import torch
import numpy as np
from ppo import PPO
from arguments import get_args
import matplotlib.pyplot as plt
from network import FeedForwardNN
from env import SelfBalancingRobot
from eval_policy import eval_policy

def plot_episodes(env):
    # episode_rewards = env.envs[0].get_episode_rewards()
    # episode_lengths = env.envs[0].get_episode_lengths()
    # episode_timestamps = env.envs[0].get_episode_times()
    episode_rewards = env.episode_rewards
    episode_lengths = env.episode_lengths
    episode_times = env.episode_time
    episode_initial_angle = env.initial_angle
    episode_final_angle = env.final_angle
    delays = env.delays

    number_of_episode = np.arange(1.,len(episode_rewards)+1)
    
    plt.ion()
    # fig = plt.figure(figsize=(12,4))
    fig = plt.figure(figsize=(15,4)) #(15,9)
    ax1 = fig.add_subplot(131)
    ax1.set_xlabel('N° Episode')
    ax1.set_ylabel('Episode Reward')
    ax1.set_title('Episodes Total Reward')
    ax1.plot(number_of_episode,episode_rewards)
    ax2 = fig.add_subplot(132)
    ax2.set_xlabel('N° Episode')
    ax2.set_ylabel('Episode Length')
    ax2.set_title('Episodes Length')
    ax2.plot(number_of_episode,episode_lengths)
    ax3 = fig.add_subplot(133)
    ax3.set_xlabel('N° Episode')
    ax3.set_ylabel('Episode Time')
    ax3.set_title('Episodes Times')
    ax3.plot(number_of_episode, episode_times)
    plt.ion()
    fig = plt.figure(figsize=(15,4))
    ax4 = fig.add_subplot(131)
    ax4.set_xlabel('N° Episode')
    ax4.set_ylabel('Initial Angle')
    ax4.set_title('Episodes Initial Angle')
    ax4.plot(episode_initial_angle)#number_of_episode,episode_initial_angle)
    ax5 = fig.add_subplot(132)
    ax5.set_xlabel('N° Episode')
    ax5.set_ylabel('Final Angle')
    ax5.set_title('Episodes Final Angle')
    ax5.plot(episode_final_angle)#number_of_episode,episode_final_angle)
    ax6 = fig.add_subplot(133)
    ax6.set_xlabel('N° Episode')
    ax6.set_ylabel('Delay')
    ax6.set_title('Delay between reset and 1st state measurement')
    ax6.plot(delays)
    #plt.savefig('logs/rewards_and_lengths_plot.svg', format='svg')
    plt.show(block=True)

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
    model.learn(total_timesteps=100000)

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
                'timesteps_per_batch': 7048, 
                'max_timesteps_per_episode': 3000, 
                'gamma': 0.99, 
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

    
    plot_episodes(env)

if __name__ == '__main__':
    args = get_args() # Parse arguments from command line
    main(args)