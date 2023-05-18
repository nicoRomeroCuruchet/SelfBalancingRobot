# Self Balancing Robot

This project aims to develop a simulated self-balancing robot using ROS (Robot Operating System) and Gazebo. The goal is to implement control algorithms that enable the robot to maintain its balance in an inverted pendulum configuration.The Self Balancing Robot Simulation project focuses on tackling the challenge of stabilizing an inverted pendulum. By leveraging ROS and Gazebo, we create a realistic simulation environment to develop and test control algorithms for the robot's balance maintenance.



![Demo](https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExZDczMTIxZGEzZTU4ZWMyMWI5M2NjY2UwMjgzZTZiNzU5NzIwMTRjNiZlcD12MV9pbnRlcm5hbF9naWZzX2dpZklkJmN0PWc/dUppFtwYS4GfFBeRvB/giphy.gif)

# Features

- Simulation Environment: Utilize Gazebo, a powerful robotics simulator, to create a virtual world for the self-balancing robot.
- Sensor Integration: Incorporate an IMU (Inertial Measurement Unit) to measure the inclination angle of the robot.
- Control Algorithms: Implement various control algorithms, such as PID (Proportional-Integral-Derivative) or RL (Reinforcement Learning), to continuously adjust the        robot's movements and maintain its vertical balance.
- Real-Time Visualization: Visualize the robot's state and control inputs in real-time using ROS tools and Gazebo's graphical interface.


# Setup 

      mkdir catkin_ws && cd catkin_ws
      git clone https://github.com/nicoRomeroCuruchet/SelfBalancingRobot.git
      catkin_make
      roslaunch self_balancing_robot main.launch


# Getting Started
Once the project is set up, you can launch the simulation environment and run the control algorithms to observe the robot's self-balancing behavior.
      
      python3 main.py

# Velocity and IMU topic

- Change velocity:

      rostopic pub /self_balancing_robot/cmd_vel geometry_msgs/Twist "linear:
        x: 1.0
        y: 0.0
        z: 0.0
      angular:
        x: 0.0
        y: 0.0
        z: 0.0" -r 10
        
- Read IMU:

        rostopic echo /imu
        
# Dependencies
This project depends on the following packages:

- ROS Noetic
- Gazebo 11
