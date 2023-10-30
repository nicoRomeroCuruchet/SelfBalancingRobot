cmake_minimum_required(VERSION 3.5)

project(SelfBalancingRobotEnv)

## Find catkin and any catkin packages
find_package(catkin REQUIRED COMPONENTS roscpp 
                                        std_msgs genmsg 
                                        tf2 
                                        sensor_msgs)

add_executable(robot_env main.cpp env.cpp)
target_link_libraries(robot_env ${catkin_LIBRARIES})
include_directories(${catkin_INCLUDE_DIRS})

#set_property(TARGET robot_env PROPERTY CXX_STANDARD 17)