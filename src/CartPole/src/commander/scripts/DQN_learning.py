#!/usr/bin/env python3

import rospy
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from std_msgs.msg import Float64
from gazebo_msgs.msg import LinkStates
from geometry_msgs.msg import Pose, Twist
from std_srvs.srv import Empty

cart_pose = Pose()
pole_pose = Pose()
pole_twist = Twist()
y_angular = 0
cart_pose_x = 0

cart_pose_x = 0
y_angular = 0
cart_vel_x = 0

reset_simulation_client = rospy.ServiceProxy('/gazebo/reset_simulation',Empty)
pub_cart = rospy.Publisher('/cart_controller/command', Float64, queue_size = 10)

class QNet(nn.Module):
	def __init__(self, num_states, dim_mid, num_actions):
		super().__init__()

		self.fc = nn.Sequential(
			nn.Linear(num_states, dim_mid),
			nn.ReLU(),
			nn.Linear(dim_mid, dim_mid),
			nn.ReLU(),
			nn.Linear(dim_mid, num_actions)
		)

	def forward(self, x):
		x = self.fc(x)
		return x

class Brain:
	def __init__(self, num_states, num_actions, gamma, r, lr):
		self.num_states = num_states
		self.num_actions = num_actions
		self.eps = 1.0  
		self.gamma = gamma
		self.r = r

		self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		print("self.device = ", self.device)
		## Q network
		self.q_net = QNet(num_states, 64, num_actions)
		self.q_net.to(self.device)
		## loss function
		self.criterion = nn.MSELoss()
		## optimization algorithm
		self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

	def updateQnet(self, obs_numpy, action, reward, next_obs_numpy):
		obs_tensor = torch.from_numpy(obs_numpy).float()
		obs_tensor.unsqueeze_(0)	
		obs_tensor = obs_tensor.to(self.device)

		next_obs_tensor = torch.from_numpy(next_obs_numpy).float()
		next_obs_tensor.unsqueeze_(0)
		next_obs_tensor = next_obs_tensor.to(self.device)

		self.optimizer.zero_grad()

		self.q_net.train()
		q = self.q_net(obs_tensor)

		with torch.no_grad():
			self.q_net.eval()	
			label = self.q_net(obs_tensor)
			next_q = self.q_net(next_obs_tensor)

			label[:, action] = reward + self.gamma*np.max(next_q.cpu().detach().numpy(), axis=1)[0]

		loss = self.criterion(q, label)
		loss.backward()
		self.optimizer.step()

	def getAction(self, obs_numpy, is_training):
		if is_training and np.random.rand() < self.eps:
			action = np.random.randint(self.num_actions)
		else:
			obs_tensor = torch.from_numpy(obs_numpy).float()
			obs_tensor.unsqueeze_(0)
			obs_tensor = obs_tensor.to(self.device)
			with torch.no_grad():
				self.q_net.eval()
				q = self.q_net(obs_tensor)
				action = np.argmax(q.cpu().detach().numpy(), axis=1)[0]

		if is_training and self.eps > 0.1:
			self.eps *= self.r
		return action

class Agent:
	def __init__(self, num_states, num_actions, gamma, r, lr):
		self.brain = Brain(num_states, num_actions, gamma, r, lr)

	def updateQnet(self, obs, action, reward, next_obs):
		self.brain.updateQnet(obs, action, reward, next_obs)

	def getAction(self, obs, is_training):
		action = self.brain.getAction(obs, is_training)
		return action

def get_cart_pose(data):
    global cart_pose_x, y_angular, cart_vel_x

    ind = data.name.index('cart_pole::cart_link')
    cart_pose = data.pose[ind]
    cart_vel = data.twist[ind]

    ind_pitch = data.name.index('cart_pole::pole_link')
    pole_twist = data.twist[ind_pitch]

    ind_tip = data.name.index('cart_pole::tip_link')
    pole_tip_pose = data.pose[ind_tip]

    cart_pose_x = cart_pose.position.x
    cart_vel_x = cart_vel.linear.x
    y_angular = pole_twist.angular.y
    pole_tip_pose_z = pole_tip_pose.position.z

def simulate(episode, is_training):
    global y_angular, cart_pose_x, cart_vel_x

    reward = 0
    yaw_angle = 0
    pole_time_height = 0
    time_interval = 0.02

    rospy.wait_for_service('/gazebo/reset_simulation')
    reset_simulation_client()

    obs = np.array([0, 0, 0, 0], dtype='float16')
    next_obs = np.array([0, 0, 0, 0], dtype='float16')
    max_step = 200
    is_done = False
    episode_reward = 0

    for step in range(max_step):
        time1 = time.time()
        yaw_angle += y_angular*time_interval

        next_obs[0] = cart_pose_x
        next_obs[1] = cart_vel_x
        next_obs[2] = yaw_angle 
        next_obs[3] = y_angular

        if(step == 0):
            next_obs[0] = 0
            next_obs[1] = 0
            next_obs[2] = 0
            next_obs[3] = 0

        action = agent.getAction(obs, is_training)

        if(abs(yaw_angle) > 0.6 or step == max_step - 1):
            is_done = True

        if is_done:
            if step < max_step - 1:
                reward = -200  
            else:
                reward = episode_reward
        else:
            reward = 6 - abs(yaw_angle)*10
        
        episode_reward += reward

        force = action*16/9 - 8

        if is_training:
            agent.updateQnet(obs, action, reward, next_obs)

        obs = np.copy(next_obs)

        pub_cart.publish(force)

        time2 = time.time()
        interval = time2 - time1
        if(interval < time_interval):
            time.sleep(time_interval - interval)

        if (is_done and is_training):
            print('Episode: {0} Finished after {1} time steps with reward {2}'.format(episode, step+1, episode_reward))
            plot_durations(step)
            break
        elif(is_done and is_training == False):
            print('Evaluation: Finished after {} time steps'.format(step+1))
            break

number_of_steps = []

def plot_durations(step):
    plt.figure(2)
    plt.clf()
    number_of_steps.append(step)
    x = np.arange(0, len(number_of_steps))
    plt.title('Training')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(x, number_of_steps)

    plt.pause(0.001)

if __name__ == '__main__':
    rospy.init_node('DQN_simulation', anonymous=True)
    rospy.Subscriber("/gazebo/link_states", LinkStates, get_cart_pose)
    gamma = 0.7
    r = 0.99
    lr = 0.001
    num_states = 4
    num_actions = 10

    agent = Agent(num_states, num_actions, gamma, r, lr)

    num_episodes = 1000
    is_training = True
    for i_episode in range(num_episodes):
        simulate(i_episode, is_training)

    x = np.arange(0, len(number_of_steps))
    plt.plot(x, number_of_steps)
    plt.show()
