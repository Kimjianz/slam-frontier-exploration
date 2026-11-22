#!/bin/bash
echo "이전 실행 종료 중..."
pkill -f "ros2 launch"; pkill -f rviz2; pkill -f rqt
pkill -f escape_robot; pkill -f explore; pkill -f frontier_node
sleep 3
pkill -9 -f "gz sim"; pkill -9 -f "/opt/ros/jazzy/lib"; pkill -9 -f "ros2_ws/install"
for i in $(seq 1 10); do
  n=$(ps aux | grep -E "gz sim|/opt/ros/jazzy/lib|ros2_ws/install" | grep -v grep | wc -l)
  [ "$n" -eq 0 ] && break
  sleep 1
done
echo "남은 프로세스: $n"
rm -f /dev/shm/fastrtps_* /dev/shm/sem.fastrtps_* /dev/shm/fast_datasharing* 2>/dev/null
ros2 daemon stop > /dev/null 2>&1; ros2 daemon start > /dev/null 2>&1

~/maze_cam.sh
echo "Nav2 준비 대기 중..."
for i in $(seq 1 30); do
  sleep 5
  if timeout 10 ros2 lifecycle get /bt_navigator 2>/dev/null | grep -q "^active"; then
    echo "Nav2 준비 완료"
    ros2 param set /slam_toolbox do_loop_closing false
    if [ "$1" == "mission" ]; then
      sleep 3
      ~/escape_mission.sh
      echo "임무 시작"
    fi
    exit 0
  fi
  if [ $i -eq 12 ]; then
    echo "Nav2 다시 활성화 시도"
    ros2 service call /lifecycle_manager_navigation/manage_nodes nav2_msgs/srv/ManageLifecycleNodes "{command: 0}" > /dev/null
  fi
done
echo "Nav2 준비 실패, 다시 실행해 주세요"
