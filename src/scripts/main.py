import os
import sys
import torch
import argparse
import mimetypes
from icecream import ic
from loguru import logger
from stable_baselines3 import PPO
from env import SelfBalancingRobotBaseLine
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback

def trainner(args)->None:

    """ Trains the robot agent using the Proximal Policy Optimization (PPO) algorithm.

    Args:
        args: A Namespace object containing the command-line arguments.

    Returns:
        None
    """
    
    logger.add(sys.stderr, format="{time} {level} {message}", filter="my_module", level="INFO")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using {} for training".format(device))
    
     # Create the environment
    env = SelfBalancingRobotBaseLine(max_timesteps_per_episode=args.max_timesteps_per_episode,
                                     threshold_angle=0.4,
                                     threshold_position=1.0)
    logger.info("Total timesteps to learn: {}".format(args.total_timesteps), )
    # Train the agent:
   
    env = Monitor(env, filename="logs", allow_early_resets=True)
    vec_env = DummyVecEnv([lambda: env])

    try:
        directory = args.model 
        file_types = lambda x: mimetypes.guess_type(x)[0] if mimetypes.guess_type(x)[0] != None else 'application/pkl'
        models = dict([(file_types(f),str(directory+'/'+f)) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))])        
        params = {}
        hyperparams = models['text/plain'] if 'text/plain' in models else None
        with open(hyperparams, 'r') as file:
            for line in file:
                key, value = line.split('=')
                params[key.strip()] = float(value)

        lf = float(params['learning_rate'])
        n_steps = int(params['n_steps'])
        batch_size = int(params['batch_size'])

        logger.info("Loaded hyperparameters from: " + models['text/plain'])
        logger.info("Learning rate: " + str(lf))
        logger.info("n_steps: " + str(n_steps))
        logger.info("batch_size: " + str(batch_size))

    except Exception as e:
        logger.error("Error reading parameters from file")
        
    # 
    try:    
        vec_env = VecNormalize.load(models['application/pkl'], vec_env)
        # Loads already existing model
        model = PPO.load(path=models['application/zip'],
                            env=vec_env, 
                            device=device, 
                            learning_rate=lf,
                            n_steps=n_steps,     
                            batch_size=batch_size,
                            verbose=True,
                            tensorboard_log="./log/")
        
        logger.info("Loaded model from: " + args.model)
    except:  

        #defaults
        learning_rate=1e-4
        n_steps=5000
        batch_size=2500
        
        vec_env = VecNormalize(vec_env, 
                                norm_obs = True, 
                                norm_reward = True)
        # Model does not exist. Create a new one.
        model = PPO("MlpPolicy", 
                    env=vec_env, 
                    device=device, 
                    learning_rate=learning_rate,
                    n_steps=n_steps,     
                    batch_size=batch_size,
                    verbose=True,
                    tensorboard_log="./log/")
        
        logger.info("Creating new model: " + args.model + " and saving it after training")
        logger.warning("Defaults parameters:")
        logger.warning("Learning rate: " + str(learning_rate))
        logger.warning("n_steps: " + str(n_steps))
        logger.warning("batch_size: " + str(batch_size))

    # CALLBACKS: Save a checkpoint every 2k steps
    checkpoint_callback = CheckpointCallback(
    save_freq=2000,
    save_path=args.model,
    name_prefix="rl_model",
    save_replay_buffer=True,
    save_vecnormalize=True)
    # Train the agent
    model.learn(total_timesteps=args.total_timesteps, progress_bar=True,callback=[checkpoint_callback])

def tester(args)->None:

    """
    Test the agent using an already trained model and a pre-defined environment.

    Args:
        args (argparse.Namespace): Command line arguments.

    Returns:
        None
    """

    logger.add(sys.stderr, format="{time} {level} {message}", filter="my_module", level="INFO")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using {} for testing the agent".format(device))
    # Create the environment
    vec_env = Monitor(SelfBalancingRobotBaseLine(max_timesteps_per_episode=args.max_timesteps_per_episode,
                                                 threshold_angle=0.4,
                                                 threshold_position=1.0) , 
                                                 filename="logs", allow_early_resets=True)        
    vec_env = DummyVecEnv([lambda: vec_env])
    vec_env = VecNormalize.load("balancing_robot_statistics.pkl", vec_env)

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
    """
    Parses command line arguments for the self balancing robot RL control script.

    Returns:
        args (argparse.Namespace): The parsed command line arguments.
    """
    # Create the argument parser    
    parser = argparse.ArgumentParser(description='Self balancing robot RL control')
    # Add arguments
    parser.add_argument('-d', '--mode',  type=str, help='can be train or test')
    parser.add_argument('-m', '--max_timesteps_per_episode',  type=int, help='the maximum number of timesteps per episode')
    parser.add_argument('-t', '--total_timesteps',  type=int, help='the total number of timesteps to train the agent')      
    parser.add_argument('-o', '--model', type=str, help='define the model name, it can be a new model or an existing one')     
    parser.add_argument('-e', '--manual',  action="store_true", help='detailed explanation of the code') 
    #parser.add_argument('-g', '--log',   type=str, default='log', help='define the log directory')    
    # Parse arguments    
    args = parser.parse_args()

    if args.manual:
        
        print("""Self balancing robot usage manual:
              
        * To test an existing model:
            python3 main.py -d test -m 1000 -o models/NN_MODEL
        
        where the arguments are:
            
            - test: is the mode.
            - 1000: is the maximum number of timesteps per episode.
            - models/NN_MODEL: is the name of the model that will be loaded from the /models directory.
        
        * To train a new model:
            python3 main.py -d train -m 1000 -t 1000000 -o models/NEW_TRAINED_PPO_MODEL 
        
        where the arguments are:
              
            - train: is the mode.
            - 1000: is the maximum number of timesteps per episode.
            - 1000000: is the total number of timesteps to train the agent.      
            - models/NEW_TRAINED_PPO_MODEL: is the name of the model that will be saved in the same directory where the script is running.
            - log: is the directory where the tensorboard log will be saved.
              
        * To train from an existing model:
            python3 main.py -d train -m 1000 -t 1000000 -o models/PPO_MODEL
        
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
    if args.mode == 'train':
        trainner(args)
    elif args.mode == 'test':
        tester(args)
    else:
        logger.error("Invalid mode: " + args.mode)

