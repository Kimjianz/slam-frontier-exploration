#!/usr/bin/env bash
source "$(dirname -- "$0")/env.sh"
python3 scripts/check_environment.py --stage mission
export RCUTILS_CONSOLE_OUTPUT_FORMAT='[{severity}] {message}'
exec python3 scripts/camera_escape.py --max-goals 40 "$@"
