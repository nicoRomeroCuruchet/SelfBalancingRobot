# PPO for Self-Balancing Robot Control with ROS/Gazebo

This repository contains the implementation of the Proximal Policy Optimization (PPO) algorithm applied to control a self-balancing robot simulated in the ROS (Robot Operating System) and Gazebo simulation environment. The PPO algorithm code has been developed from scratch using PyTorch by Eric Yang Yu.

# Overview

The self-balancing robot is simulated in ROS/Gazebo, which provides an accurate simulation environment for testing and training the control algorithm. The PPO algorithm is a state-of-the-art reinforcement learning method used to train the robot to maintain balance and make appropriate movements based on sensor feedback.

# Setup       
      git clone https://github.com/nicoRomeroCuruchet/SelfBalancingRobot.git
      cd SelfBalancingRobot
      pip install -r requirements.txt
      catkin_make
      source devel/setup.bash
      roslaunch self_balancing_robot main.launch
      
# Getting Started
Once compiled and with the robot running in the Gazebo simulator.
To train from scratch:

      python3 main.py

To test model:

      python3 main.py --mode test --actor_model ppo_actor.pth

To train with existing actor/critic trained models:

      python3 main.py --actor_model ppo_actor.pth --critic_model ppo_critic.pth
             
# Dependencies
This project depends on the following packages:

- ROS Noetic
- Gazebo 11

License
The Self Balancing Robot Simulation project is released under the MIT. Please review the license file for more details and comply with the terms when using or modifying the project.
We hope you find this Self Balancing Robot Simulation project engaging and informative. Feel free to explore the documentation, experiment with different control algorithms, and contribute to the project's ongoing development. Enjoy the journey of building and simulating a self-balancing robot with ROS and Gazebo!
