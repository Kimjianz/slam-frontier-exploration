#!/usr/bin/env bash
source "$(dirname -- "$0")/env.sh"
export LIBGL_ALWAYS_SOFTWARE=1
exec rviz2 -d "$MAZE_ESCAPE_ROOT/config/view.rviz"
