#!/usr/bin/env bash
source "$(dirname -- "$0")/run/env.sh"
for script in setup.sh run/*.sh; do
  bash -n "$script"
done
python3 scripts/check_environment.py --stage install
python3 scripts/prepare_camera.py
python3 -m unittest discover -s tests -v
echo "준비 검사가 끝났어. 안내문 순서대로 터미널을 열어줘."
