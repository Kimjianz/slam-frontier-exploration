#!/usr/bin/env bash
source "$(dirname -- "$0")/env.sh"
python3 scripts/check_environment.py --stage sim
export LIBGL_ALWAYS_SOFTWARE=1
exec ros2 launch "$MAZE_ESCAPE_ROOT/launch/sim.launch.py"
