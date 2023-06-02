# PPO for Self-Balancing Robot Control with ROS/Gazebo

This repository contains the implementation of the Proximal Policy Optimization (PPO) algorithm applied to control a self-balancing robot simulated in the ROS (Robot Operating System) and Gazebo simulation environment. The PPO algorithm code has been developed from scratch using PyTorch by Eric Yang Yu.

# Overview

The self-balancing robot is simulated in ROS/Gazebo, which provides an accurate simulation environment for testing and training the control algorithm. The PPO algorithm is a state-of-the-art reinforcement learning method used to train the robot to maintain balance and make appropriate movements based on sensor feedback.

# Setup 

      pip install -r requirements.txts
      git clone https://github.com/nicoRomeroCuruchet/SelfBalancingRobot.git
      cd SelfBalancingRobot
      catkin_make
      source devel/setup.bash
      roslaunch self_balancing_robot main.launch
      
      


# Getting Started
Once the project is set up, launch the robot and the enviroment.
      import gym
      roslaunch self_balancing_robot main.launch

To train from scratch:

      python3 main.py

To test model:

      python3 main.py --mode test --actor_model ppo_actor.pth

To train with existing actor/critic models:

      python3 main.py --actor_model ppo_actor.pth --critic_model ppo_critic.pth

# ROS topics: Velocity and IMU topic

- Change velocity:

      rostopic pub /self_balancing_robot/cmd_vel geometry_msgs/Twist "linear:
        x: 1.0
        y: 0.0
        z: 0.0
      angular:
        x: 0.0
        y: 0.0
        z: 0.0" -r 10
             
# Dependencies
This project depends on the following packages:

- ROS Noetic
- Gazebo 11
