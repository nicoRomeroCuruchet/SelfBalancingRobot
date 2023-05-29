# Self Balancing Robot

The quadratic formula is given by:

$ax^2 + bx + c = 0$

where $a$, $b$, and $c$ are coefficients.

The solutions to the equation are:

$x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}$


This project aims to develop a simulated self-balancing robot using ROS (Robot Operating System) and Gazebo. The goal is to implement control algorithms that enable the robot to maintain its balance in an inverted pendulum configuration.The Self Balancing Robot Simulation project focuses on tackling the challenge of stabilizing an inverted pendulum. By leveraging ROS and Gazebo, we create a realistic simulation environment to develop and test control algorithms for the robot's balance maintenance.



![Demo](https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExZDczMTIxZGEzZTU4ZWMyMWI5M2NjY2UwMjgzZTZiNzU5NzIwMTRjNiZlcD12MV9pbnRlcm5hbF9naWZzX2dpZklkJmN0PWc/dUppFtwYS4GfFBeRvB/giphy.gif)

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
