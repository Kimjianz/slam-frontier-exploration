#!/bin/bash
gnome-terminal --tab -- bash -ic "export TURTLEBOT3_MODEL=burger_cam; ros2 launch turtlebot3_gazebo escape_room.launch.py; exec bash"
sleep 20
gnome-terminal --tab -- bash -ic "source ~/ros2_ws/install/setup.bash; ros2 run escape_robot scan_fixer --ros-args -p use_sim_time:=true; exec bash"
gnome-terminal --tab -- bash -ic "ros2 launch slam_toolbox online_async_launch.py use_sim_time:=True slam_params_file:=$HOME/slam_params.yaml; exec bash"
sleep 8
gnome-terminal --tab -- bash -ic "ros2 launch nav2_bringup navigation_launch.py use_sim_time:=True params_file:=$HOME/nav2_params.yaml; exec bash"
sleep 8
gnome-terminal --tab -- bash -ic "rviz2 -d \$(ros2 pkg prefix nav2_bringup)/share/nav2_bringup/rviz/nav2_default_view.rviz; exec bash"
