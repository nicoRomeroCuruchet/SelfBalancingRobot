import sys
import torch
import argparse
from stable_baselines3 import PPO
from env import SelfBalancingRobotBaseLine
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback


# python3 main.py -d train -m 800 -t 100000 -o models/NEW_RAINED_PPO_MODEL -g log
# python3 main.py -d test -m 500 -o models/NN_MODEL

def main(args):

    """ This function creates the environment, loads the model and trains or tests it. """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)

    # Train the agent:
    if args.mode == 'train':
        # Create the environment
        env = SelfBalancingRobotBaseLine(max_timesteps_per_episode=args.max_timesteps_per_episode,
                                         threshold_angle=0.4,
                                         threshold_position=1)
        
        env = Monitor(env, filename="logs", allow_early_resets=True)
        vec_env = DummyVecEnv([lambda: env])

        n_steps = 400
        batch_size = 200
        lr= 1e-4

        try:
            vec_env = VecNormalize.load("balancing_robot_statistics.pkl", vec_env)
            # Loads already existing model
            model = PPO.load("balancing_robot_weights.zip",
                                env=vec_env, 
                                device=device, 
                                learning_rate=lr,
                                n_steps=n_steps ,     
                                batch_size=batch_size,
                                verbose=True)
            
            print("Loading existing model: " + args.model)


        except:
            vec_env = VecNormalize(vec_env, 
                                    norm_obs = True, 
                                    norm_reward = True)
            # Model does not exist. Create a new one.
            model = PPO("MlpPolicy", 
                    env=vec_env, 
                    device=device, 
                    learning_rate=lr,
                    n_steps=n_steps,     
                    batch_size=batch_size,
                    verbose=True,
                    tensorboard_log="./log/",)
            print("Creating new model: " + args.model + " and saving it after training")

        # CALLBACKS: Save a checkpoint every 5k steps
        checkpoint_callback = CheckpointCallback(
        save_freq=5000,
        save_path="./models/",
        name_prefix="rl_model",
        save_replay_buffer=True,
        save_vecnormalize=True)

        # Train the agent
        model.learn(total_timesteps=args.total_timesteps, progress_bar=True,callback=[checkpoint_callback])
        # Save the model
        model.save(args.model)

    # Test the agent:
    elif args.mode == 'test':
        # Create the environment
        env = SelfBalancingRobotBaseLine(max_timesteps_per_episode=args.max_timesteps_per_episode,
                                         threshold_angle=0.2,
                                         threshold_position=0.5)
         
        env = Monitor(env, filename="logs", allow_early_resets=True)
        env = DummyVecEnv([lambda: env])
        vec_env = VecNormalize.load("balancing_robot_statistics.pkl", env)

        # Loads already existing model
        model = PPO.load("balancing_robot_weights.zip", 
                         env=vec_env, 
                         device=device)
        print("Testing Policy: Loading existing model " + "balancing_robot_weights.zip")
        while True:
            total_reward = 0
            obs = vec_env.reset()
            for i in range(args.max_timesteps_per_episode):
                action, _ = model.predict(obs, deterministic=True)
                #print("Send action: ", action)
                obs, reward, done, _ = vec_env.step(action)
                total_reward += reward
                if done:
                    print("Episode finished after {} timesteps".format(i+1))
                    print("Total reward: ", total_reward)
                    break

def args_parse():

     # Create the argument parser    
    parser = argparse.ArgumentParser(description='Self balancing robot RL control')
    # Add arguments
    parser.add_argument('-d', '--mode',  type=str, help='can be train or test')
    parser.add_argument('-m', '--max_timesteps_per_episode',  type=int, help='the maximum number of timesteps per episode')
    parser.add_argument('-t', '--total_timesteps',  type=int, help='the total number of timesteps to train the agent')      
    parser.add_argument('-o', '--model', type=str, help='define the model name, it can be a new model or an existing one')     
    parser.add_argument('-g', '--log',   type=str, default='log', help='define the log directory')    
    parser.add_argument('-e', '--manual',  action="store_true", help='detailed explanation of the code')    
    # Parse arguments    
    args = parser.parse_args()
   
    if args.manual:
        print("""
              
    Self balancing robot usage manual:
              
        * To test an existing model:
            python3 main.py -d test -m 1000 -o models/NN_MODEL
        
        where the arguments are:
            
            - test: is the mode.
            - 1000: is the maximum number of timesteps per episode.
            - models/NN_MODEL: is the name of the model that will be loaded from the /models directory.
        
        * To train a new model:
            python3 main.py -d train -m 1000 -t 1000000 -o models/NEW_TRAINED_PPO_MODEL -g log
        
        where the arguments are:
              
            - train: is the mode.
            - 1000: is the maximum number of timesteps per episode.
            - 1000000: is the total number of timesteps to train the agent.      
            - models/NEW_TRAINED_PPO_MODEL: is the name of the model that will be saved in the same directory where the script is running.
            - log: is the directory where the tensorboard log will be saved.
              
        * To train from an existing model:
            python3 main.py -d train -m 1000 -t 1000000 -o models/PPO_MODEL -g /log
        
        where the arguments are:
              
            - train: is the mode.
            - 1000: is the maximum number of timesteps per episode.
            - 1000000: is the total number of timesteps to train the agent.      
            - /models/PPO_MODEL: is the path of the model that will be load to continue training.
            - /log: is the directory where the tensorboard log will be saved.

        * hit cntrl+c to stop the training or testing.
                                    
        """)
        
        sys.exit()

    return args



if __name__ == '__main__':
    args = args_parse()
    main(args)
