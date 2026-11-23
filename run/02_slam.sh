#!/usr/bin/env bash
source "$(dirname -- "$0")/env.sh"
python3 scripts/check_environment.py --stage slam
exec ros2 launch slam_toolbox online_async_launch.py use_sim_time:=true autostart:=true use_lifecycle_manager:=false slam_params_file:="$MAZE_ESCAPE_ROOT/config/slam.yaml"
