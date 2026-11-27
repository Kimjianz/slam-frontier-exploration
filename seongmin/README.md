# seongmin - LEVEL 1 자율 방탈출

TurtleBot3(waffle)가 escape_room 맵에서 사람 개입 없이 탈출하는 LEVEL 1 구현입니다.

## 동작 순서 (FSM)

```
EXPLORE_KEY ──(노란 열쇠 발견)──▶ STOP_FOR_KEY ──▶ APPROACH_KEY ──(도착)──▶ PICK_KEY (has_key = True)
                                                                              │
     ┌────────────────────────────────────────────────────────────────────────┘
     ▼
(문 위치를 알면) APPROACH_DOOR / (모르면) SEARCH_DOOR ──▶ OPEN_DOOR ──▶ DRIVE_THROUGH
                                                                              │
     ┌────────────────────────────────────────────────────────────────────────┘
     ▼
SEARCH_EXIT ──(초록 탈출구 발견)──▶ STOP_FOR_EXIT ──▶ APPROACH_EXIT ──(도착)──▶ DONE
```

- 탐색 상태에서 20초 동안 거의 움직이지 않으면 `LOOK_AROUND`(제자리 회전) → `WANDER`(아직 안 둘러본 곳으로 이동)로 넘어갑니다.
- 문 직진 통과(`DRIVE_THROUGH`)가 막히면 `PASS_DOOR`(Nav2로 재시도) → `ASSIST_PASS`(탐색하며 확인) 순서로 시도하고, **로봇이 문 너머에 있는 것이 확인돼야만** `SEARCH_EXIT`로 넘어갑니다.

## 구현 방식

| 기능 | 방법 |
|---|---|
| 지도 생성 | SLAM Toolbox (online async) |
| 자율 탐색 | explore_lite (노드가 직접 실행/종료) |
| 이동 | Nav2 `navigate_to_pose` 액션 |
| 객체 인식 | 카메라 영상을 HSV로 바꿔서 색으로 구분 (노랑: 열쇠, 빨강: 문, 초록: 탈출구) |
| 객체 위치 | 화면 속 색 영역의 중심 → 로봇 기준 방향 계산 → LiDAR로 그 방향 거리 측정. 못 재면 SLAM 지도에서 그 방향 첫 장애물까지, 그것도 안 되면 **화면 속 높이와 실제 높이(열쇠 0.4m, 문 1m, 탈출구 0.6m)로 거리를 추정**해서 대략 위치로 쓰고, LiDAR로 다시 재면 그 값으로 교체 |
| 열쇠 획득 | 열쇠 앞 도착 → Gazebo에서 열쇠 모델 삭제 → **삭제 성공이 확인되면** `has_key = True` (실패 시 재시도) |
| 문 열기 | `has_key`가 True일 때만 Gazebo `/world/maze/remove` 서비스로 문 모델 삭제, 성공 확인 후 다음 단계 |
| 문 통과 | 지도에서 문 주변 벽의 방향을 구해 문 정면으로 이동 → 문을 지운 뒤 LiDAR로 뚫린 구간(문 틈)을 재서 가운데로 정렬 → LiDAR로 정면 장애물을 보며 직진 → 로봇이 문 벽면을 0.6m 넘어간 것을 위치로 확인 |
| 가려짐 검사 | 색 영역 왼쪽~오른쪽 5개 방향의 LiDAR 거리를 각각 재서, 하나라도 측정이 안 되거나 차이가 크면(열쇠·탈출구 0.35m, 문 0.8m) 벽 모서리 등에 가려진 것으로 보고 위치를 기록하지 않음 |
| 도착 판정 | Nav2로 도착하면 물체 쪽으로 몸을 돌려 다시 보고, **최근 3초 안에 카메라+LiDAR로 가까이서 직접 본 경우에만** 획득·개방·탈출 성공 처리 |

## 파일 구조

```
seongmin/
├── escape_robot/                 ROS 2 패키지
│   └── escape_robot/
│       └── escape_fsm.py         방탈출 FSM 노드
├── start.sh                      전체 실행 스크립트
├── collect_logs.sh               문제 확인용 로그 모으기
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

Gazebo, SLAM, RViz, Nav2를 순서대로 켠 뒤 방탈출 노드를 실행합니다. VM에서 느리면 `./start.sh norviz`로 RViz 없이 실행할 수 있습니다. 마지막에 `탈출 성공! LEVEL 1 임무 완료`가 출력되면 성공입니다.
각 프로그램 로그는 `/tmp/escape_logs/`, explore_lite 로그는 `/tmp/explore_lite.log`에 저장됩니다.

인식 결과 확인: `ros2 run rqt_image_view rqt_image_view` → `/escape/debug_image`

문제가 생기면 새 터미널에서 `./collect_logs.sh`를 실행하면 방탈출 노드·Nav2·explore_lite 로그를 `~/debug.txt` 하나로 모아줍니다.

## 진행하면서 겪은 문제와 해결

| 문제 | 원인 | 해결 |
|---|---|---|
| Nav2가 지도를 받지 못함 (`map message is malformed`) | VM에서 Gazebo GPU LiDAR가 전부 `.inf` 출력 → SLAM 지도가 빈 상태 | `LIBGL_ALWAYS_SOFTWARE=1`로 소프트웨어 렌더링 |
| 로봇이 Nav2 명령에 반응하지 않음 | Jazzy TurtleBot3는 `cmd_vel`을 `TwistStamped`로 받음 | TurtleBot3 전용 Nav2 파라미터(`waffle.yaml`) 사용 |
| explore_lite가 제자리 목표만 반복 | 출발 위치 발밑이 지도에서 미탐색으로 남음 | 시작 시 로봇을 조금 이동, `min_frontier_size` 1.0 |
| explore_lite 정지 신호(`explore/resume`)가 무시됨 | 목표 도착 콜백에서 바로 다음 계획을 세우는 구조 | explore_lite를 노드에서 프로세스로 실행/종료 |
| 시뮬레이션이 실제보다 약 18배 느림 | 1920x1080 카메라를 CPU로 렌더링 | 카메라 해상도 320x240으로 변경 |
| 문을 지웠는데 바로 통과 못 함 | SLAM 지도에 문이 벽으로 남아 있음 (벽으로 본 횟수보다 빈칸으로 본 횟수가 훨씬 많아야 지도가 바뀜) | 지도 대신 LiDAR로 앞을 보며 문 정면에서 직진 통과 |
| 열쇠·문 삭제 실패를 놓칠 수 있음 (피드백) | 삭제 결과를 확인하지 않고 다음 단계로 진행 | 삭제 성공일 때만 `has_key`/문 개방 처리, 실패 시 재시도 |
| 문을 통과하지 못해도 다음 단계로 넘어감 (피드백) | 재시도 횟수만 보고 넘어감 | 로봇 위치가 문 너머인지 확인한 뒤에만 탈출구 탐색 |
| 지도로 잰 위치가 물체 뒤의 벽일 수 있음 (피드백) | 물체가 아직 지도에 없으면 선이 물체를 지나침. 실제 테스트에서 가로벽 끝을 열쇠로 착각해 그 자리에서 열쇠를 '획득'한 경우가 있었음 | 지도 값은 대략 위치로만 사용, LiDAR 값이 나오면 교체, 접근 중 위치가 0.5m 넘게 바뀌면 목표 재설정, **LiDAR로 확인된 위치에 가까이 있을 때만 획득** |
| 문 앞에서 직진하다 문틀 옆 벽에 막힘 | 가까워질수록 카메라에 문 일부만 보여서 문 중심이 한쪽 끝으로 쏠림 | 문을 지운 뒤 LiDAR로 문 틈의 양 끝을 재서 중심을 다시 계산하고 가운데로 정렬 |
| 탐색 중 로봇이 목표마다 멈춰서 제자리 회전 | explore_lite가 모든 목표의 도착 방향을 동쪽(yaw 0)으로 보내고, Nav2는 도착 방향 오차를 약 15도까지만 허용 | `start.sh`에서 TurtleBot3 Nav2 설정을 복사해 `yaw_goal_tolerance`만 3.14로 바꿔 사용 (문 통과 방향은 노드가 직접 맞춤) |
| 문 근처에서 Nav2가 경로를 보내는데 로봇이 멈춤 | collision_monitor의 FootprintApproach 규칙이 벽 근처에서 속도를 0까지 줄임. 설정의 로봇 반지름(0.15m)이 실제 waffle보다 작아 벽에 붙는 경로가 나옴 | collision_monitor는 PolygonStop(0.1m 안 정지)만 사용, `robot_radius`를 0.2m로 변경 |
| 열쇠·탈출구를 엉뚱한 곳에서 '획득/도착' 처리 | 벽 모서리 옆으로 물체 일부만 보일 때, 같은 방향 LiDAR가 모서리에 맞아 모서리를 물체 위치로 기록. 도착 판정도 '기억한 위치에 도착'만 확인 | 가려짐 검사 추가, 도착 후 물체를 직접 다시 보고 판정 |
| 멀리 보이는 열쇠를 버림 (문 앞에서 둘러봐도 못 찾음) | LiDAR 범위(3.5m) 밖이고, 열쇠 뒤가 아직 지도에 없어 지도로도 거리를 못 구함 | 화면 속 높이로 거리 추정 (3.9m 실험에서 오차 약 0.1m) |
| 문 앞(벽 근처)에서 탐색이 자꾸 멈칫함 | explore_lite가 벽에 붙은 Frontier를 고르면 Nav2가 실패하고, explore_lite는 다음 재계획 시각까지 대기 | explore_lite 재계획 주기 1초(같은 목표는 다시 보내지 않음), 진전 없는 목표 포기 시간 10초 |
| 시작하자마자 둘러보기를 함 | SLAM은 0.5m 이상 움직여야 지도를 갱신하는데, 바퀴에 직접 명령한 시작 이동이 LiDAR에 잡힌 로봇 몸체를 장애물로 보고 거의 안 움직임 | 시작 이동을 Nav2 목표(앞으로 0.85m, Nav2가 0.25m 안이면 도착 처리하므로 넉넉히)로 바꾸고, 실제 이동 거리가 0.5m 미만이면 방향을 바꿔 한 번 더 |
| 전체 실행이 너무 오래 걸림 | 위의 회전·멈춤에 VM의 느린 시뮬레이션까지 겹침 | 막힘 판단 시간 단축, 회전 속도 증가, 재시도 횟수 축소, `./start.sh norviz`로 RViz 없이 실행 |

## 개선할 점

- 문을 통과한 뒤에도 SLAM 지도에는 문이 벽으로 남아 있음 (돌아올 일이 없는 Level 1에서는 문제없음)
- 색 범위와 거리 기준이 escape_room 맵에 맞춰져 있음
