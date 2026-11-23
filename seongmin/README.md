# seongmin - LEVEL 1 자율 방탈출

TurtleBot3(waffle)가 escape_room 맵에서 사람 개입 없이 탈출하는 LEVEL 1 구현입니다.

## 동작 순서 (FSM)

```
EXPLORE_KEY ──(노란 열쇠 발견)──▶ STOP_FOR_KEY ──▶ APPROACH_KEY ──(도착)──▶ PICK_KEY (has_key = True)
                                                                              │
     ┌────────────────────────────────────────────────────────────────────────┘
     ▼
(문 위치를 알면) APPROACH_DOOR / (모르면) SEARCH_DOOR ──▶ OPEN_DOOR ──▶ PASS_DOOR
                                                                              │
     ┌────────────────────────────────────────────────────────────────────────┘
     ▼
SEARCH_EXIT ──(초록 탈출구 발견)──▶ STOP_FOR_EXIT ──▶ APPROACH_EXIT ──(도착)──▶ DONE
```

- 탐색 상태에서 20초 동안 거의 움직이지 않으면 `LOOK_AROUND`(제자리 회전) → `WANDER`(아직 안 둘러본 곳으로 이동)로 넘어갑니다.

## 구현 방식

| 기능 | 방법 |
|---|---|
| 지도 생성 | SLAM Toolbox (online async) |
| 자율 탐색 | explore_lite (노드가 직접 실행/종료) |
| 이동 | Nav2 `navigate_to_pose` 액션 |
| 객체 인식 | 카메라 영상을 HSV로 바꿔서 색으로 구분 (노랑: 열쇠, 빨강: 문, 초록: 탈출구) |
| 객체 위치 | 화면 속 색 영역의 중심 → 로봇 기준 방향 계산 → LiDAR로 그 방향 거리 측정 (못 재면 SLAM 지도에서 그 방향 첫 장애물) |
| 열쇠 획득 | 열쇠 앞 도착 시 `has_key = True`, Gazebo에서 열쇠 모델 삭제 |
| 문 열기 | `has_key`가 True일 때만 Gazebo `/world/maze/remove` 서비스로 문 모델 삭제 |

## 파일 구조

```
seongmin/
├── escape_robot/                 ROS 2 패키지
│   └── escape_robot/
│       └── escape_fsm.py         방탈출 FSM 노드
├── start.sh                      전체 실행 스크립트
├── setup_camera.sh               카메라 해상도 변경 (처음 한 번)
└── README.md
```

## 실행 방법

### 처음 한 번

```bash
# 필요한 패키지
sudo apt install -y python3-opencv

# explore_lite 설치 (~/ros2_ws/src 에 m-explore-ros2 가 없을 때)
cd ~/ros2_ws/src
git clone https://github.com/robo-friends/m-explore-ros2.git

# 카메라 해상도 낮추기
cd ~/ros2_ws/src/slam-frontier-exploration/seongmin
./setup_camera.sh

# 빌드
cd ~/ros2_ws
colcon build --packages-select explore_lite_msgs explore_lite escape_robot
```

### 실행

```bash
cd ~/ros2_ws/src/slam-frontier-exploration/seongmin
./start.sh
```

Gazebo, SLAM, RViz, Nav2를 순서대로 켠 뒤 방탈출 노드를 실행합니다. 마지막에 `탈출 성공! LEVEL 1 임무 완료`가 출력되면 성공입니다.
각 프로그램 로그는 `/tmp/escape_logs/`, explore_lite 로그는 `/tmp/explore_lite.log`에 저장됩니다.

인식 결과 확인: `ros2 run rqt_image_view rqt_image_view` → `/escape/debug_image`

## 진행하면서 겪은 문제와 해결

| 문제 | 원인 | 해결 |
|---|---|---|
| Nav2가 지도를 받지 못함 (`map message is malformed`) | VM에서 Gazebo GPU LiDAR가 전부 `.inf` 출력 → SLAM 지도가 빈 상태 | `LIBGL_ALWAYS_SOFTWARE=1`로 소프트웨어 렌더링 |
| 로봇이 Nav2 명령에 반응하지 않음 | Jazzy TurtleBot3는 `cmd_vel`을 `TwistStamped`로 받음 | TurtleBot3 전용 Nav2 파라미터(`waffle.yaml`) 사용 |
| explore_lite가 제자리 목표만 반복 | 출발 위치 발밑이 지도에서 미탐색으로 남음 | 시작 시 로봇을 조금 이동, `min_frontier_size` 1.0 |
| explore_lite 정지 신호(`explore/resume`)가 무시됨 | 목표 도착 콜백에서 바로 다음 계획을 세우는 구조 | explore_lite를 노드에서 프로세스로 실행/종료 |
| 시뮬레이션이 실제보다 약 18배 느림 | 1920x1080 카메라를 CPU로 렌더링 | 카메라 해상도 320x240으로 변경 |
| 문을 지웠는데 바로 통과 못 함 | SLAM 지도에 문이 벽으로 남아 있음 | 제자리 회전으로 재스캔 + 재시도, 실패 시 탐색으로 전환 |

## 개선할 점

- 문을 연 직후 SLAM 지도 갱신이 늦어서 `PASS_DOOR`가 몇 번 실패한 뒤 넘어감
- 색 범위와 거리 기준이 escape_room 맵에 맞춰져 있음
