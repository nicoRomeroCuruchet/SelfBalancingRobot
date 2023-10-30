cmake_minimum_required(VERSION 3.5)

project(SelfBalancingRobotEnv)

## Find catkin and any catkin packages
find_package(catkin REQUIRED COMPONENTS roscpp 
                                        std_msgs genmsg 
                                        tf2 
                                        sensor_msgs)

add_executable(server_env server.cpp env.cpp)
target_link_libraries(server_env ${catkin_LIBRARIES})
include_directories(${catkin_INCLUDE_DIRS})
