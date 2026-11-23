#!/bin/bash
# TurtleBot3 waffle 카메라 해상도를 1920x1080 -> 320x240 으로 낮춤 (처음 한 번만 실행)
# VM(소프트웨어 렌더링)에서 시뮬레이션 속도가 너무 느려지는 문제 해결용
MODEL=/opt/ros/jazzy/share/turtlebot3_gazebo/models/turtlebot3_waffle/model.sdf

sudo sed -i 's|<width>1920</width>|<width>320</width>|; s|<height>1080</height>|<height>240</height>|' "$MODEL"
echo "변경 결과:"
grep -n "<width>\|<height>" "$MODEL"
