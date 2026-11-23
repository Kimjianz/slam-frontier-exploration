#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ament_index_python.packages import get_package_share_directory
from maze_escape.camera_model import build
root = Path(__file__).resolve().parents[1]
tb3 = Path(get_package_share_directory('turtlebot3_gazebo'))
build((tb3/'models/turtlebot3_burger/model.sdf').read_text(),
      (tb3/'urdf/turtlebot3_burger.urdf').read_text(),root/'models/camera_burger')
print('프로젝트 안에 RGB-D 카메라 로봇을 준비했어. 설치된 로봇은 수정하지 않았어.')
