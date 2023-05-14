# Self Balancing Robot

This project contains the code to simulate a Self Balancing Robot in ROS using the Gazebo simulator.

# Setup 

      mkdir catkin_ws && cd catkin_ws
      git clone https://github.com/nicoRomeroCuruchet/SelfBalancingRobot.git
      catkin_make
      roslaunch self_balancing_robot main.launch


# Velocity and IMU topic

- Change velocity:

      rostopic pub /cmd_vel geometry_msgs/Twist "linear:
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
