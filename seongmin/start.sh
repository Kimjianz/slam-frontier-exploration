#!/bin/bash
# 방탈출 전체 실행 스크립트
#   ./start.sh          : 환경(Gazebo, SLAM, RViz, Nav2) + 방탈출 노드까지 실행
#   ./start.sh env      : 환경만 실행 (방탈출 노드는 직접 실행)
#
# 각 프로그램의 로그는 /tmp/escape_logs/ 에 저장됨

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
REPO_DIR=$(dirname "$SCRIPT_DIR")
LOG_DIR=/tmp/escape_logs
mkdir -p "$LOG_DIR"

export TURTLEBOT3_MODEL=waffle
export LIBGL_ALWAYS_SOFTWARE=1          # VM에서 LiDAR가 .inf로 나오는 문제 해결
source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/install/setup.bash

echo "[0/5] 이전 실행 정리"
pkill -f escape_fsm; pkill -f explore; pkill -f rviz2; pkill -f "ros2 launch"; pkill -f gz
sleep 3
ros2 daemon stop > /dev/null 2>&1
ros2 daemon start > /dev/null 2>&1
rm -f "$LOG_DIR"/*.log

cleanup() {
    echo "종료 중..."
    pkill -f escape_fsm; pkill -f explore; pkill -f rviz2; pkill -f "ros2 launch"; pkill -f gz
}
trap cleanup EXIT

echo "[1/5] Gazebo + TurtleBot3 (escape_room)"
ros2 launch "$REPO_DIR/maze_world/launch/escape_room.launch.py" > "$LOG_DIR/gazebo.log" 2>&1 &
for i in $(seq 1 90); do
    ros2 topic list 2>/dev/null | grep -q "^/scan$" && break
    sleep 2
done
if ! ros2 topic list 2>/dev/null | grep -q "^/scan$"; then
    echo "  Gazebo가 켜지지 않았습니다. 로그 마지막 부분:"
    tail -n 20 "$LOG_DIR/gazebo.log"
    exit 1
fi
sleep 5

echo "[2/5] SLAM Toolbox"
ros2 launch slam_toolbox online_async_launch.py use_sim_time:=true > "$LOG_DIR/slam.log" 2>&1 &
sleep 5

echo "[3/5] RViz"
ros2 run rviz2 rviz2 -d /opt/ros/jazzy/share/nav2_bringup/rviz/nav2_default_view.rviz \
    > "$LOG_DIR/rviz.log" 2>&1 &

echo "[4/5] Nav2 (활성화될 때까지 대기)"
ros2 launch nav2_bringup navigation_launch.py use_sim_time:=true \
    params_file:=/opt/ros/jazzy/share/turtlebot3_navigation2/param/waffle.yaml \
    > "$LOG_DIR/nav2.log" 2>&1 &
# Nav2 로그에 'Managed nodes are active'가 찍힐 때까지 기다림 (최대 5분)
NAV2_OK=0
for i in $(seq 1 150); do
    if grep -q "Managed nodes are active" "$LOG_DIR/nav2.log" 2>/dev/null; then
        NAV2_OK=1
        break
    fi
    if grep -qi "Failed to bring up\|failed to change state" "$LOG_DIR/nav2.log" 2>/dev/null; then
        break
    fi
    [ $((i % 10)) -eq 0 ] && echo "      기다리는 중... $((i * 2))초"
    sleep 2
done
if [ $NAV2_OK -eq 0 ]; then
    echo "  Nav2가 준비되지 않았습니다. Nav2 로그 마지막 부분:"
    tail -n 25 "$LOG_DIR/nav2.log"
    exit 1
fi
echo "      Nav2 준비 완료"

# 출발 직후 로봇 발밑이 지도에서 '모르는 곳'으로 남아서 explore_lite가 제자리만 맴도는 문제 방지
echo "      로봇을 앞으로 약 0.4m 이동"
ros2 topic pub --times 15 -r 5 /cmd_vel geometry_msgs/msg/TwistStamped \
    "{header: {frame_id: base_link}, twist: {linear: {x: 0.15}}}" > /dev/null
ros2 topic pub --once /cmd_vel geometry_msgs/msg/TwistStamped "{header: {frame_id: base_link}}" > /dev/null
sleep 3

if [ "$1" == "env" ]; then
    echo "환경 실행 완료. 방탈출 노드는 다른 터미널에서: ros2 run escape_robot escape_fsm"
    echo "끝내려면 Ctrl + C"
    wait
else
    echo "[5/5] 방탈출 노드 실행 (Ctrl + C 로 전체 종료)"
    ros2 run escape_robot escape_fsm
fi
