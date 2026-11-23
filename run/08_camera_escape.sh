#!/usr/bin/env bash
source "$(dirname -- "$0")/env.sh"
python3 scripts/check_environment.py --stage mission
exec python3 scripts/camera_escape.py --max-goals 40
