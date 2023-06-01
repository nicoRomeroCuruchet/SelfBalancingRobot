# PPO for Self-Balancing Robot

This project implements the Proximal Policy Optimization (PPO) algorithm for training a self-balancing robot using reinforcement learning. The goal is to teach the robot to maintain its balance and navigate its environment by adjusting its actions based on feedback from the sensors.

![Demo](https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExZDczMTIxZGEzZTU4ZWMyMWI5M2NjY2UwMjgzZTZiNzU5NzIwMTRjNiZlcD12MV9pbnRlcm5hbF9naWZzX2dpZklkJmN0PWc/dUppFtwYS4GfFBeRvB/giphy.gif)


# Overview

The PPO algorithm is a state-of-the-art reinforcement learning method that addresses the challenge of stable and sample-efficient policy optimization. It strikes a balance between exploration and exploitation while ensuring stable policy updates. By training the self-balancing robot using PPO, we aim to enable it to learn an optimal policy for maintaining balance and making appropriate movements.


# Features

- Simulation Environment: Utilize Gazebo, a powerful robotics simulator, to create a virtual world for the self-balancing robot.
- Control Algorithms: Implement various control algorithms, such as PID (Proportional-Integral-Derivative) or RL (Reinforcement Learning), to continuously adjust the        robot's movements and maintain its vertical balance.import gym
- Real-Time Visualization: Visualize the robot's state and control inputs in real-time using ROS tools and Gazebo's graphical interface.


# Setup 

      pip install -r requirements.txt
      mkdir catkin_ws && cd catkin_ws
      git clone https://github.com/nicoRomeroCuruchet/SelfBalancingRobot.git
      catkin_make


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
