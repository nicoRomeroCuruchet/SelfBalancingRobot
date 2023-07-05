import gymnasium as gym
import rospy

class DebugWrapper(gym.Wrapper):

    def __init__(self, env):
        super(DebugWrapper, self).__init__(env)
        self.episode_reward = []
        self.current_episode_reward = 0
        self.episode_length = []
        self.current_episode_length = 0
        self.episode_time = []
        self.final_angle = []
        self.final_x = []
        self.final_y = []
        self.time1 = 0
        self.done = False
        self.steps_antes_de_done = [0]
        self.total_steps = 0
        self.step_time = []
        self.step_angle = []
        self.step_action = []
        self.step_done = []


        # Variables changed inside the custom enviroment:
        # (They can to be obtained through info after every step)
        self.steps_before_1st_measure = None
        self.initial_angle            = None
        self.initial_x                = None
        self.initial_y                = None
        self.delays                   = None
        self.total_measures           = None
        self.measures                 = None
        self.measure_times            = None


    def reset(self, **kwargs):
        self.done = False
        self.steps_antes_de_done += [0]
        self.time1 = 0
        
        return self.env.reset(**kwargs)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        
        self.total_steps += 1
        self.current_episode_reward += reward
        self.current_episode_length += 1

        self.step_time += [rospy.get_time()+self.time0]#[self.time1]
        self.step_angle += [self.current_angle]
        self.step_action += [action[0]]
        done = terminated or truncated
        self.step_done += [done]
        if self.done:
            print("ERROR: The episode was done on the previous step, but it hasn't reset yet and it's makeing another step.")
            self.steps_antes_de_done[-1] += 1
        
        self.done = done
        
        if done:
            self.episode_reward.append(self.current_episode_reward)
            self.episode_length.append(self.current_episode_length)
            self.episode_time += [rospy.get_time()+self.time0]
            self.final_angle += [self.current_angle]
            self.final_x     += [self.position_x]
            self.final_y     += [self.position_y]
            self.current_episode_reward = 0
            self.current_episode_length = 0
        
        if self.bandera:
            self.steps_before_1st_measure[-1] += 1
        
        # Variables changed inside the custom enviroment:
        # (They can to be obtained through info after every step)
        self.steps_before_1st_measure = info["steps_before_1st_measure"]
        self.initial_angle            = info["initial_angle"]
        self.initial_x                = info["initial_x"]
        self.initial_y                = info["initial_y"]
        self.delays                   = info["delays"]
        self.total_measures           = info["total_measures"]
        self.measures                 = info["measures"]
        self.measure_times            = info["measure_times"]

        return observation, reward, terminated, truncated, info
