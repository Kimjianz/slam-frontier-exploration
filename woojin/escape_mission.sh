#!/bin/bash
gnome-terminal --tab -- bash -ic "ros2 run rqt_image_view rqt_image_view /detect/image; exec bash"
gnome-terminal --tab -- bash -ic "ros2 launch explore_lite explore.launch.py; exec bash"
sleep 3
gnome-terminal --tab -- bash -ic "source ~/ros2_ws/install/setup.bash; ros2 run escape_robot escape_node --ros-args -p use_sim_time:=true; exec bash"
