#!/usr/bin/env bash
# Only this process receives the clean environment. No .bashrc changes.
set -e
export MAZE_ESCAPE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [ ! -f /opt/ros/jazzy/setup.bash ]; then
  echo "ROS 2 Jazzy 설치를 찾지 못했어."; exit 1
fi
source /opt/ros/jazzy/setup.bash
export TURTLEBOT3_MODEL=burger
export ROS_DOMAIN_ID=26
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
export GZ_SIM_RESOURCE_PATH=/opt/ros/jazzy/share/turtlebot3_gazebo/models
export GZ_PARTITION=maze_escape
export ROS_LOG_DIR="$MAZE_ESCAPE_ROOT/logs/ros"
export PYTHONUNBUFFERED=1
cd "$MAZE_ESCAPE_ROOT"
