#!/usr/bin/env bash
source "$(dirname -- "$0")/env.sh"
python3 scripts/check_environment.py --stage nav
exec ros2 launch nav2_bringup navigation_launch.py use_sim_time:=true autostart:=true use_composition:=False params_file:="$MAZE_ESCAPE_ROOT/config/nav2.yaml"
