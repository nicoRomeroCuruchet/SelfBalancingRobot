#!/usr/bin/env python
##### SECTION ROS #####
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Imu
from std_srvs.srv import Empty
import time


####

import os
import math
from nav_msgs.msg import Odometry
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Twist, Pose
from tf.transformations import euler_from_quaternion
from gazebo_msgs.srv import SetPhysicsProperties, SetModelState,\
                            SpawnModel, DeleteModel,\
                            ApplyBodyWrenchRequest, ApplyBodyWrench


###

##### SECTION TORCH #####
import torch
from network import FeedForwardNN
from torch.distributions import MultivariateNormal
from torch.optim import Adam
import numpy as np


# Supervised Learning - Batch vs Epoch
# We split our data into batches
# Batch = Number of examples used for one step of gradient update.
# If the batch has size = 1, then its called SGD.
# 1 Epoch = Number of times the algorithm will go through the entire dataset.

# Example:
# If we have 1000 epochs, and split our training data into 40 batches -> 
# Since we perform 1 update per batch, and 1 epoch = 40 updates, 1000 epocchs = 40.000 updates.
# https://machinelearningmastery.com/difference-between-a-batch-and-an-epoch/

class PPO:

    def __init__(self):
        self.obs_dim = 1
        self.act_dim = 1

        self.actor = FeedForwardNN(self.obs_dim, self.act_dim)
        self.critic = FeedForwardNN(self.obs_dim, 1)

        self._init_hyperparameters()

        # Multivariate normal distribution cov_matrix for choosing actions
        self.cov_var = torch.full(size=(self.act_dim,), fill_value=self.variance_coeff)
        self.cov_mat = torch.diag(self.cov_var)

        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

        ###### SECTION ROS ######

        # IMU reading. Observations. Note this is a numpy array.
        self.y_angle = np.array([0], dtype = float)

        self.pub_topic_name = '/self_balancing_robot/cmd_vel'
        self.sub_topic_name = '/imu'

        self.pub = rospy.Publisher(self.pub_topic_name, Twist, queue_size=1)
        self.sub = rospy.Subscriber(self.sub_topic_name, Imu, self.callback)

        self.reset = rospy.ServiceProxy("/gazebo/reset_simulation",Empty)
        self.pause = rospy.ServiceProxy("/gazebo/pause_physics",Empty)
        self.unpause = rospy.ServiceProxy("/gazebo/unpause_physics",Empty)

        self.delete_model_service = '/gazebo/delete_model'

        rospy.init_node('velpublisher', anonymous=True)

        # Velocity message to publish
        self.vel=Twist()
        self.vel.linear.y = 0
        self.vel.linear.z = 0
        self.vel.angular.x =0
        self.vel.angular.y = 0
        self.vel.angular.z = 0

        # Reset to initial state
        self.reset()

        # Wait for one second
        time.sleep(1)


    # Resets the Gazebo world and pauses the simulation
    #def env_reset(self):

        # Reset to Initial State
        #self.reset()

        # Pause the sim
        #self.pause()

        # Return current observation as a numpy array
        #return self.y_angle

    def env_reset(self):

        self.unpause()

        # Delete the model 
        rospy.wait_for_service(self.delete_model_service)
        try:
            delete_model = rospy.ServiceProxy(self.delete_model_service, DeleteModel)
            delete_model('self_balancing_robot')
        except rospy.ServiceException as e:
            rospy.loginfo(f"Failed to delete the model: {str(e)}")
        
        # Spawn the model
        try:
            current_dir = os.path.dirname(__file__)
            urdf_path = os.path.join(current_dir, 'urdf/robot.urdf')
            model_name = 'self_balancing_robot'                             # Provide a unique name for your model
            rospy.wait_for_service('/gazebo/spawn_urdf_model')

            spawn_model = rospy.ServiceProxy('/gazebo/spawn_urdf_model', SpawnModel)
            with open(urdf_path, 'r') as f:
                urdf_xml = f.read()

            pose = Pose()
            pose.position.x = 0  # Adjust the position as needed
            pose.position.y = 0
            pose.position.z = 0.06 + 0.09
            pose.orientation.x = 0
            pose.orientation.y = 0
            pose.orientation.z = 0
            pose.orientation.w = 0
            response = spawn_model(model_name, urdf_xml, "", pose, "world")

        except rospy.ServiceException as e:
            rospy.loginfo(f"Failed to spawn the model: {str(e)}")

        # Wait for a new value
        self.y_angle = np.array([100], dtype = float)
        while self.y_angle[0] == 100:
            try:
                rospy.wait_for_message('/imu', Imu, timeout=1)
            except:
                pass
                
        # Small force applied for the robot to move
        self.apply_force_to_link('base_link', [10, 0, 0], 0.01)

        self.pause()

        return self.y_angle


    def apply_force_to_link(self, link_name, force, duration):

        """ Apply a force to a specified link in Gazebo simulation.

        Args:
            link_name (str): Name of the link to apply the force to.
            force (list[float]): List of three float values representing the force in the x, y, and z directions.
            duration (float): Duration in seconds for which the force should be applied.

        Returns:
            None

        Raises:
            rospy.ServiceException: If there is an error while calling the 'apply_body_wrench' service. """

        rospy.wait_for_service('/gazebo/apply_body_wrench')
        try:
            apply_wrench = rospy.ServiceProxy('/gazebo/apply_body_wrench', ApplyBodyWrench)
            request = ApplyBodyWrenchRequest()
            request.body_name = link_name
            request.wrench.force.x = force[0]
            request.wrench.force.y = force[1]
            request.wrench.force.z = force[2]
            request.duration = rospy.Duration(duration)
            response = apply_wrench(request)
        except rospy.ServiceException as e:
            rospy.logerr(f"Failed to apply force to link '{link_name}': {e}")

    # Takes a numpy array of action we should take
    def env_step(self, action):

        # Unpase simulation that has been previously paused by env_reset
        self.unpause()

        # Publish to topic
        self.vel.linear.x = action[0]
        self.pub.publish(self.vel)

        # Read the new state as a numpy array
        obs = self.y_angle

        # Pause the simulation, and leaves it this way until the next step or env_reset
        self.pause()

        r, d = self.get_reward(obs)

        return obs, r, d

    def get_reward(self, obs):
        if(abs(obs[0]) > self.threshold):
            return -100, True
        return 1, False

    def callback(self, data):
        _, current_state,_ = euler_from_quaternion([data.orientation.x,
                                                 data.orientation.y,
                                                 data.orientation.z,
                                                 data.orientation.w])

        self.y_angle[0] = current_state


    def _init_hyperparameters(self):
        # Default values for hyperparameters, will need to change later.
        self.timesteps_per_batch = 2400           # timesteps per batch (Set of episodes)
        self.max_timesteps_per_episode = 1600      # timesteps per episode
        self.variance_coeff = 0.5 # Variance coeff for cov.matrix
        self.gamma = 0.95 # Discount Factor
        self.epochs = 5
        self.clip = 0.2 # As recommended by the paper
        self.lr = 0.005 # Learning rate for SGD
        self.threshold = 0.5 # Threshold for max angle of robot

    def learn(self, total_timesteps):
        t_so_far = 0
        while t_so_far < total_timesteps: # Kth iteration

            # Collect a batch of data
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout()

            # Calculate V_{phi, k} for kth iteration
            V, _ = self.evaluate(batch_obs, batch_acts)

            # Calculate advantage for kth iteration
            A_k = batch_rtgs - V.detach()

            # Normalize advantages
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            # Go through the entire training data epoch times
            for _ in range(self.epochs):

                # Get the log probs for the current epoch iteration
                # Note that curr_log_probs is not detached. It contains a computation graph.
                _, curr_log_probs = self.evaluate(batch_obs, batch_acts)

                # Calculate ratios (new/old) Logs cancel out with exp
                ratios = torch.exp(curr_log_probs - batch_log_probs)

                # Calculate surrogate losses
                surr1 = ratios * A_k
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k

                # TODO: Pick Mini-batches randomly and perform one step of ascent
                # per mini batch inside this epoch

                # Minimizing the negative loss maximizes the performance function. 
                # We then take the mean to generate a single loss as a float.
                actor_loss = (-torch.min(surr1, surr2)).mean()

                # Calculate gradients and perform backward propagation for actor 
                # network
                self.actor_optim.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.actor_optim.step()

                # Calculate V_phi and pi_theta(a_t | s_t)    
                V, _ = self.evaluate(batch_obs, batch_acts)

                critic_loss = torch.nn.MSELoss()(V, batch_rtgs)
                # Calculate gradients and perform backward propagation for critic network    
                self.critic_optim.zero_grad()    
                critic_loss.backward()    
                self.critic_optim.step()

            # Increment total timestep count so far
            t_so_far += np.sum(batch_lens)


    # Runs a set of episodes in order to collect a batch of data
    def rollout(self):
        # Batch data
        batch_obs = []             # batch observations (timesteps_per_batch, dim of observation)
        batch_acts = []            # batch actions (timesteps_per_batch, dim of action)
        batch_log_probs = []       # log probs of each action (timesteps_per_batch)
        batch_rews = []            # batch rewards (number of episodes, number of timesteps per episode)
        batch_rtgs = []            # batch rewards-to-go (timesteps_per_batch)
        batch_lens = []            # episodic lengths in batch (num of episodes)
                
        # Number of timesteps run so far for this batch 
        t = 0 

        while t < self.timesteps_per_batch:

            # Rewards this episode
            ep_rewards = []

            # Resets and pauses the simulation.
            obs = self.env_reset()
            
            done = False

            for ep_t in range(self.max_timesteps_per_episode):

                # Increment timesteps ran this batch so far
                t += 1

                batch_obs.append(obs)

                # Note: We interact with the environment
                # using NUMPY Arrays. Tensors are hidden from it.
                action, log_prob = self.get_action(obs)
                obs, rew, done = self.env_step(action)

                #if(type(obs) == tuple):
                #    obs = obs[0]

                ep_rewards.append(rew)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)

                if done:
                    break

            # We just finished an episode
            batch_lens.append(ep_t + 1)
            batch_rews.append(ep_rewards)


        # Print the AVG reward
        print("Average reward: " + str(np.mean(ep_rewards)))

        # Reshape data as tensors in the shape specified
        batch_obs = torch.tensor(batch_obs, dtype=torch.float)
        batch_acts = torch.tensor(batch_acts, dtype=torch.float)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)

        batch_rtgs = self.compute_rtgs(batch_rews)

        # Return the batch data
        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens


    # Chooses an action by:
    # 1) Query actor to get multivariate normal distribution mean
    # 2) Sample using covariance matrix
    def get_action(self, obs):

        # Query the actor network for a mean action.
        # Same thing as calling self.actor.forward(obs)
        mean = self.actor(obs)

        # Create our Multivariate Normal Distribution
        dist = MultivariateNormal(mean, self.cov_mat)

        # Sample an action from the distribution and get its log prob
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        # Return the sampled action and the log prob of that action

        # Note that I'm calling detach() since the action and log_prob  
        # are tensors with computation graphs, so I want to get rid
        # of the graph and just convert the action to numpy array before
        # passing it to the environment step function

        # log prob as tensor is fine. Our computation graph will
        # start later down the line.
        return action.detach().numpy(), log_prob.detach()

    # For each timestep, computes the sum of discounted rewards
    # Until the end of its episode.
    def compute_rtgs(self, batch_rews):
        # The rewards-to-go (rtg) per episode per batch to return.
        # The shape will be (num timesteps per episode)
        batch_rtgs = []
        # Iterate through each episode backwards to maintain same order
        # in batch_rtgs
        for ep_rews in reversed(batch_rews):
            discounted_reward = 0
            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                # Insert at the start
                batch_rtgs.insert(0, discounted_reward)
        # Convert the rewards-to-go into a tensor
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)
        return batch_rtgs

    # Evaluates the value function for each state, and returns log probs for each (s,a) 
    # WITH THE MOST RECENT actor/critic networks
    def evaluate(self, batch_obs, batch_acts):

        # Find the log prob (a|s) with the most recent actor network
        mean = self.actor(batch_obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(batch_acts)

        # Find value for every s with the most recent critic network
        # Squeeze: [[1], [2], [3]] -> [1, 2, 3]
        V = self.critic(batch_obs).squeeze()

        return V, log_probs


if __name__ == '__main__':
    try:
        model = PPO()
        model.learn(10000000)        
    except rospy.ROSInterruptException:
        pass


