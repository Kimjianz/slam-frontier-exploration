#!/usr/bin/env python3
"""
LEVEL 1 자율 방탈출 노드 (FSM)

시나리오:
  탐색 → 노란 열쇠 발견 → 열쇠로 이동 → 열쇠 획득
       → 빨간 문으로 이동 → (열쇠가 있으면) 문 열기 → 문 통과
       → 초록 탈출구 탐색 → 탈출구 도착 → 탈출 성공

필요한 것 (먼저 켜져 있어야 함):
  Gazebo(escape_room) + SLAM Toolbox + Nav2 + explore_lite

실행:
  python3 escape_fsm.py
"""

import math
import shutil
import subprocess
import glob
import os
import random
import signal
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data, QoSProfile, DurabilityPolicy, ReliabilityPolicy
from action_msgs.msg import GoalStatus
from action_msgs.srv import CancelGoal
from geometry_msgs.msg import PoseStamped, TwistStamped
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import Image, LaserScan
import tf2_ros


# ------------------------------------------------------------------
# 설정값 (필요하면 여기만 고치면 됨)
# ------------------------------------------------------------------
VERSION = '11-27 v10 (LiDAR 확정 버그 수정)'
WORLD_NAME = 'maze'          # escape_room.world 안의 <world name="maze">
KEY_MODEL = 'red_key'        # Gazebo 속 열쇠 모델 이름 (색은 노란색)
DOOR_MODEL = 'red_door'      # Gazebo 속 문 모델 이름

CAMERA_HFOV = 1.02974        # TurtleBot3 waffle 카메라 수평 화각 (rad)
PROC_WIDTH = 320             # 영상을 가로 약 320픽셀로 줄여서 처리 (VM 부담 줄이기)
MIN_BLOB_FRAC = 0.0008       # 줄인 영상 전체 픽셀 중 이 비율 이상이 그 색이면 '보인다'고 판단
MAX_DETECT_RANGE = 3.3       # 레이저로 거리를 잴 수 있는 최대 거리 (센서 최대 3.5m)
MAX_MAP_RANGE = 8.0          # 레이저로 못 재면 SLAM 지도에서 이 거리까지 선을 그어 찾음
STUCK_TIME = 12.0            # 탐색 중 이 시간(Gazebo 시간) 동안
STUCK_DIST = 0.3             # 이 거리보다 적게 움직이면 '막힘' → 제자리 한 바퀴 둘러보기

STANDOFF = {'key': 0.55, 'door': 0.75, 'exit': 0.6}   # 물체 앞 몇 m에 멈출지
PICK_DISTANCE = 1.0          # 열쇠와 이 거리 안이면 '획득'
PASS_DISTANCE = 1.2          # (예비) Nav2로 통과할 때 문 너머 몇 m 지점을 목표로 할지
DOOR_FRONT = 0.7             # 문 정면 몇 m 앞에서 멈춘 뒤 직진할지
PASS_DEPTH = 0.6             # 문 벽면을 이만큼 넘어가야 '통과 성공'
DRIVE_SPEED = 0.15           # 문 통과 직진 속도 (m/s)
DRIVE_TIMEOUT = 60.0         # 직진 통과 제한 시간 (Gazebo 시간 기준, 초)
FRONT_STOP = 0.22            # 직진 중 정면 장애물이 이보다 가까우면 정지
WARMUP_DIST = 0.85           # 시작할 때 Nav2로 움직일 거리 (Nav2는 목표 0.25m 안이면 도착으로 처리하므로 넉넉히. SLAM은 0.5m 이상 움직여야 지도 갱신)
LOOK_SPEED = 0.8             # 둘러보기 회전 속도 (rad/s)
CONFIRM_TRIES = 3            # 도착했는데 LiDAR로 위치 확인이 안 되면 다시 확인할 횟수
RANGE_SPREAD_MAX = {'key': 0.35, 'door': 0.8, 'exit': 0.35}   # 색 영역 왼쪽~오른쪽 거리 차이가 이보다 크면 '가려진 물체'로 보고 위치를 믿지 않음 (문은 크고 평평해서 비스듬히 보면 차이가 커서 넉넉하게)
RECENT_SEC = 3.0             # 도착 판정: 최근 몇 초 안에 직접 본 것만 인정
CONFIRM_RANGE = {'key': 1.4, 'door': 1.9, 'exit': 1.4}   # 도착 판정: 이 거리 안에서 직접 봤어야 함
OBJ_HEIGHT = {'key': 0.4, 'door': 1.0, 'exit': 0.6}      # 물체 실제 높이(m): LiDAR·지도로 거리를 못 재면 화면 속 높이로 대략 거리 추정

# HSV 색 범위 (OpenCV 기준: H 0~180, S 0~255, V 0~255)
COLOR_RANGES = {
    'key':  [((20, 120, 70), (38, 255, 255))],                          # 노랑
    'door': [((0, 120, 70), (10, 255, 255)), ((170, 120, 70), (180, 255, 255))],  # 빨강
    'exit': [((45, 120, 70), (75, 255, 255))],                          # 초록
}
DRAW_COLOR = {'key': (255, 255, 0), 'door': (255, 0, 0), 'exit': (0, 255, 0)}
NAME_KR = {'key': '열쇠', 'door': '문', 'exit': '탈출구'}

# explore_lite를 이 노드가 직접 켜고 끔 (정지 신호가 무시되는 문제 때문에 프로세스를 종료하는 방식)
EXPLORE_CMD = ['ros2', 'run', 'explore_lite', 'explore', '--ros-args',
               '-p', 'use_sim_time:=true', '-p', 'return_to_init:=false',
               '-p', 'robot_base_frame:=base_link', '-p', 'costmap_topic:=map',
               '-p', 'costmap_updates_topic:=map_updates', '-p', 'visualize:=true',
               '-p', 'planner_frequency:=1.0', '-p', 'progress_timeout:=10.0',
               '-p', 'potential_scale:=3.0', '-p', 'gain_scale:=1.0',
               '-p', 'transform_tolerance:=0.5', '-p', 'min_frontier_size:=1.0']
EXPLORE_LOG = '/tmp/explore_lite.log'


def yaw_to_quat(yaw):
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


def quat_to_yaw(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class EscapeFSM(Node):

    def __init__(self):
        super().__init__('escape_fsm',
                         parameter_overrides=[Parameter('use_sim_time', value=True)])

        # ---------- 통신 ----------
        self.create_subscription(Image, '/camera/image_raw', self.on_image, qos_profile_sensor_data)
        self.create_subscription(LaserScan, '/scan', self.on_scan, qos_profile_sensor_data)
        map_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(OccupancyGrid, '/map', self.on_map, map_qos)
        self.explore_proc = None
        self.cancel_cli = self.create_client(CancelGoal, '/navigate_to_pose/_action/cancel_goal')
        self.debug_pub = self.create_publisher(Image, '/escape/debug_image', 1)
        self.cmd_pub = self.create_publisher(TwistStamped, '/cmd_vel', 10)
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ---------- 센서 데이터 ----------
        self.latest_image = None
        self.latest_scan = None
        self.latest_map = None

        # ---------- 기억 (발견한 물체의 지도 좌표) ----------
        self.memory = {'key': None, 'door': None, 'exit': None}
        self.memory_src = {'key': None, 'door': None, 'exit': None}   # 'lidar'(정확) / 'map'(대략)
        self.approach_target = None          # 지금 다가가는 목표를 정할 때 쓴 물체 위치
        self.door_normal = None              # 문 벽면에 수직인 방향 (로봇 쪽 → 문 너머)
        self.key_try = 0                     # 열쇠 삭제 시도 횟수
        self.door_try = 0                    # 문 삭제 시도 횟수
        self.drive_phase = None              # 직진 통과 단계: align / forward / done / blocked
        self.warmup_start = None             # 시작 이동을 시작한 위치
        self.warmup_try = 0                  # 시작 이동 시도 횟수
        self.confirm_tries = {'key': 0, 'door': 0, 'exit': 0}
        self.last_obs = {'key': None, 'door': None, 'exit': None}   # (시각, LiDAR 거리) 마지막으로 확실히 본 기록
        self.face_name = None                # 도착 후 몸을 돌려 바라볼 물체
        self.faced_at = None                 # 다 돌아본 시각
        self.drive_t0 = 0.0                  # 직진 통과 단계 시작 시각
        self.seen_now = {'key': False, 'door': False, 'exit': False}
        self.has_key = False
        self.door_open = False

        # ---------- Nav2 목표 상태 ----------
        self.nav_state = 'idle'      # idle / active / succeeded / failed
        self.goal_handle = None
        self.nav_seq = 0             # 오래된 결과 무시용 번호

        # ---------- FSM ----------
        self.state = None
        self.state_start = self.now_sec()
        self.retry = 0
        self.door_robot_pose = None
        self.turn_step = 0
        self.explore_resent = False
        self.search_state = 'EXPLORE_KEY'   # 둘러보기가 끝나면 돌아갈 탐색 상태
        self.stuck_ref = None                # (시간, x, y) 막힘 판단 기준점
        self.looked_at = []                  # 둘러보기를 했던 위치들 (같은 곳 반복 방지)
        self.wander_fail = 0                 # 갈 곳 고르기/이동 연속 실패 횟수

        self.gz_cmd = self.find_gz()

        self.create_timer(0.25, self.process_image)   # 4Hz 영상 처리
        self.create_timer(0.5, self.tick)             # 2Hz FSM
        self.create_timer(0.1, self.drive_control)    # 10Hz 문 통과 직진 제어

        self.get_logger().info(f'escape_fsm 버전: {VERSION}')
        self.get_logger().info('Nav2 액션 서버 기다리는 중...')
        self.nav_client.wait_for_server()
        self.get_logger().info('Nav2 연결 완료. Gazebo 시계와 로봇 위치를 기다렸다가 시작합니다')
        # 시작은 tick()에서: Gazebo 시계(/clock)와 로봇 위치(TF)를 받은 뒤에 WARMUP으로 들어감

    # ================================================================
    # 유틸
    # ================================================================
    def now_sec(self):
        return self.get_clock().now().nanoseconds / 1e9

    def elapsed(self):
        return self.now_sec() - self.state_start

    def change_state(self, new_state):
        if self.state == 'DRIVE_THROUGH' or self.face_name is not None:
            self.drive(0.0, 0.0)
        self.face_name, self.faced_at = None, None
        self.get_logger().info(f'[상태 전환] {self.state} → {new_state}')
        self.state = new_state
        self.state_start = self.now_sec()
        self.retry = 0
        self.explore_resent = False
        self.on_enter(new_state)

    def robot_pose(self):
        """지도(map) 기준 로봇 위치 (x, y, yaw)"""
        try:
            t = self.tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time())
        except Exception:
            return None
        tr = t.transform.translation
        return tr.x, tr.y, quat_to_yaw(t.transform.rotation)

    def explore_running(self):
        return self.explore_proc is not None and self.explore_proc.poll() is None

    def explore(self, on):
        if on:
            if not self.explore_running():
                log = open(EXPLORE_LOG, 'a')
                self.explore_proc = subprocess.Popen(EXPLORE_CMD, stdout=log, stderr=subprocess.STDOUT,
                                                     start_new_session=True)
                self.get_logger().info(f'explore_lite 시작 (로그: {EXPLORE_LOG})')
        else:
            if self.explore_running():
                try:
                    os.killpg(self.explore_proc.pid, signal.SIGINT)
                    self.explore_proc.wait(timeout=3)
                except Exception:
                    try:
                        os.killpg(self.explore_proc.pid, signal.SIGKILL)
                    except Exception:
                        pass
                self.get_logger().info('explore_lite 종료')
            self.explore_proc = None
            self.cancel_all_nav()

    def cancel_all_nav(self):
        """Nav2에 걸려 있는 모든 목표 취소 (explore_lite가 남긴 목표 포함)"""
        if self.cancel_cli.service_is_ready():
            self.cancel_cli.call_async(CancelGoal.Request())   # 빈 요청 = 전부 취소
        self.nav_seq += 1
        self.nav_state = 'idle'

    def dist_to(self, p):
        r = self.robot_pose()
        if r is None or p is None:
            return float('inf')
        return math.hypot(p[0] - r[0], p[1] - r[1])

    def forget(self, name):
        """기억한 물체 위치가 틀렸을 수 있을 때 지우고 다시 찾기"""
        self.memory[name] = None
        self.memory_src[name] = None
        self.confirm_tries[name] = 0
        self.last_obs[name] = None

    def confirmed_near(self, name, dist):
        """최근 RECENT_SEC초 안에 카메라+LiDAR로 직접, 가까이(CONFIRM_RANGE) 본 적이 있고,
        로봇이 기억한 위치에서 dist m 안에 있나 (기억한 위치에 도착한 것만으로는 인정하지 않음)"""
        obs = self.last_obs[name]
        if obs is None or self.now_sec() - obs[0] > RECENT_SEC or obs[1] > CONFIRM_RANGE[name]:
            return False
        return self.memory_src[name] == 'lidar' and self.dist_to(self.memory[name]) < dist

    def arrived_check(self, name):
        """Nav2 도착 후: 물체 쪽으로 몸을 돌리고 0.8초 동안 다시 본 다음 판정할 준비가 됐나"""
        if self.faced_at is None:
            if self.face_name is None:
                self.face_name = name                 # drive_control()이 물체 쪽으로 회전
            return False
        return self.now_sec() - self.faced_at > 0.8

    def not_confirmed(self, name, stop_state, search_state):
        """도착은 했는데 위치가 LiDAR로 확인 안 됨 → 멈춰서 다시 보기, 여러 번 안 되면 잊고 다시 찾기"""
        self.confirm_tries[name] += 1
        if self.confirm_tries[name] <= CONFIRM_TRIES:
            self.get_logger().warn(
                f'{NAME_KR[name]} 위치가 LiDAR로 확인되지 않음 → 멈춰서 다시 확인 '
                f'({self.confirm_tries[name]}/{CONFIRM_TRIES})')
            self.change_state(stop_state)
        else:
            self.get_logger().warn(f'{NAME_KR[name]} 위치를 확인할 수 없어서 잊고 다시 찾습니다')
            self.forget(name)
            self.change_state(search_state)

    # ================================================================
    # 센서 콜백
    # ================================================================
    def on_image(self, msg):
        self.latest_image = msg

    def on_scan(self, msg):
        self.latest_scan = msg

    def on_map(self, msg):
        self.latest_map = msg

    def map_raycast(self, x, y, angle):
        """SLAM 지도에서 (x, y)부터 angle 방향으로 선을 그어 처음 만나는 장애물까지 거리"""
        m = self.latest_map
        if m is None:
            return None
        res = m.info.resolution
        ox, oy = m.info.origin.position.x, m.info.origin.position.y
        W, H = m.info.width, m.info.height
        step = res * 0.5
        d = 0.15
        while d < MAX_MAP_RANGE:
            px, py = x + d * math.cos(angle), y + d * math.sin(angle)
            i, j = int((px - ox) / res), int((py - oy) / res)
            if i < 0 or j < 0 or i >= W or j >= H:
                return None
            if m.data[j * W + i] >= 65:
                return d
            d += step
        return None

    def pick_wander_point(self):
        """지도에서 아직 둘러보지 않은 곳 중 갈 수 있는 빈 칸을 골라줌"""
        m, r = self.latest_map, self.robot_pose()
        if m is None:
            self.get_logger().warn('갈 곳 고르기 실패: 지도(/map)를 아직 못 받음')
            return None
        if r is None:
            self.get_logger().warn('갈 곳 고르기 실패: 로봇 위치(TF map->base_link)를 못 읽음')
            return None
        res = m.info.resolution
        ox, oy = m.info.origin.position.x, m.info.origin.position.y
        W, H = m.info.width, m.info.height
        free = [k for k, v in enumerate(m.data) if 0 <= v < 25]
        if not free:
            self.get_logger().warn('갈 곳 고르기 실패: 지도에 빈 칸이 없음')
            return None
        samples = random.sample(free, min(600, len(free)))

        # 조건을 점점 완화하면서 찾기: (주변 여유 거리, 최소 거리, 최대 거리)
        for clear_m, dmin, dmax in ((0.35, 1.0, 5.0), (0.25, 0.7, 6.0), (0.15, 0.5, 8.0)):
            clear = max(1, int(clear_m / res))
            best, best_score = None, -1.0
            for k in samples:
                i, j = k % W, k // W
                x, y = ox + (i + 0.5) * res, oy + (j + 0.5) * res
                d = math.hypot(x - r[0], y - r[1])
                if d < dmin or d > dmax:
                    continue
                ok = True
                for dj in (-clear, 0, clear):
                    for di in (-clear, 0, clear):
                        ii, jj = i + di, j + dj
                        if ii < 0 or jj < 0 or ii >= W or jj >= H or not (0 <= m.data[jj * W + ii] < 25):
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    continue
                score = min([math.hypot(x - p[0], y - p[1]) for p in self.looked_at] + [d])
                if score > best_score:
                    best, best_score = (x, y), score
            if best is not None:
                return best
        self.get_logger().warn(f'갈 곳 고르기 실패: 조건에 맞는 빈 칸 없음 (빈 칸 {len(free)}개 중)')
        return None

    def scan_range_at(self, angle):
        """로봇 기준 angle(rad, 왼쪽 +) 방향의 레이저 거리. 못 재면 None"""
        s = self.latest_scan
        if s is None:
            return None
        a = angle
        while a < s.angle_min:
            a += 2 * math.pi
        while a >= s.angle_min + 2 * math.pi:
            a -= 2 * math.pi
        n = len(s.ranges)
        center = int(round((a - s.angle_min) / s.angle_increment))
        vals = []
        for k in range(-3, 4):
            r = s.ranges[(center + k) % n]
            if math.isfinite(r) and s.range_min < r < s.range_max:
                vals.append(r)
        if not vals:
            return None
        return float(np.median(vals))

    def scan_single(self, angle):
        """로봇 기준 angle 방향에서 가장 가까운 광선 하나의 거리. 측정 불가면 None"""
        sc = self.latest_scan
        if sc is None:
            return None
        k = int(round((angle - sc.angle_min) / sc.angle_increment)) % len(sc.ranges)
        r = sc.ranges[k]
        if math.isfinite(r) and sc.range_min < r < sc.range_max:
            return r
        return None

    # ================================================================
    # 영상 처리: 색 찾기 → 방향 계산 → 레이저로 거리 → 지도 좌표
    # ================================================================
    def process_image(self):
        msg = self.latest_image
        if msg is None or msg.encoding not in ('rgb8', 'bgr8'):
            return
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
        img = img[:, :msg.width * 3].reshape(msg.height, msg.width, 3)
        ds = max(1, msg.width // PROC_WIDTH)
        img = img[::ds, ::ds]
        if msg.encoding == 'bgr8':
            img = img[:, :, ::-1]
        img = np.ascontiguousarray(img)          # RGB
        h, w = img.shape[:2]
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        focal = (w / 2.0) / math.tan(CAMERA_HFOV / 2.0)

        debug = img.copy()
        robot = self.robot_pose()

        for name, ranges in COLOR_RANGES.items():
            mask = np.zeros((h, w), dtype=np.uint8)
            for lo, hi in ranges:
                mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
            count = int(cv2.countNonZero(mask))
            self.seen_now[name] = False
            if count < max(20, MIN_BLOB_FRAC * h * w):
                continue
            m = cv2.moments(mask)
            cx = m['m10'] / m['m00']
            bearing = math.atan2((w / 2.0) - cx, focal)      # 왼쪽이 +
            rng = self.scan_range_at(bearing)
            src = 'lidar'
            if (rng is None or rng > MAX_DETECT_RANGE) and robot is not None:
                rng = self.map_raycast(robot[0], robot[1], robot[2] + bearing)
                src = 'map'

            ys, xs = np.nonzero(mask)
            # 가려짐 검사: 색 영역 가운데 60% 구간의 여러 방향으로 거리를 재서 차이가 크면 믿지 않음
            # (벽 모서리 옆으로 물체 일부만 보이면, 같은 방향 LiDAR는 모서리에 맞아 엉뚱한 거리가 나옴)
            x0, x1 = float(xs.min()), float(xs.max())
            probe = []
            for f in (0.2, 0.35, 0.5, 0.65, 0.8):
                b = math.atan2((w / 2.0) - (x0 + f * (x1 - x0)), focal)
                if src == 'lidar':
                    pr = self.scan_single(b)          # 중앙값 말고 그 방향 광선 하나의 실제 값
                    if pr is not None and pr > MAX_DETECT_RANGE:
                        pr = None
                else:
                    pr = self.map_raycast(robot[0], robot[1], robot[2] + b) if robot else None
                probe.append(pr)
            valid = [v for v in probe if v is not None]
            # 모든 방향이 측정돼야 하고(하나라도 '측정 불가'면 그 뒤로 뚫린 것), 거리 차이가 작아야 함
            consistent = rng is not None and len(valid) == len(probe) and \
                (max(valid) - min(valid)) <= RANGE_SPREAD_MAX[name]
            if not consistent:
                # LiDAR·지도로 못 재거나 가려졌으면: 화면 속 높이로 대략 거리 추정 (위아래가 잘리지 않았을 때만)
                y0, y1 = int(ys.min()), int(ys.max())
                if y0 > 1 and y1 < h - 2:
                    vfov = 2.0 * math.atan(math.tan(CAMERA_HFOV / 2.0) * h / w)
                    ang = (y1 - y0 + 1) / h * vfov
                    rng, src, consistent = OBJ_HEIGHT[name] / (2.0 * math.tan(ang / 2.0)), 'size', True
            cv2.rectangle(debug, (int(xs.min()), int(ys.min())), (int(xs.max()), int(ys.max())),
                          DRAW_COLOR[name], 2)
            label = (f'{name} {rng:.2f}m ({src})' if consistent else f'{name} ? (hidden)') if rng else f'{name} far'
            cv2.putText(debug, label, (int(xs.min()), max(12, int(ys.min()) - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, DRAW_COLOR[name], 1)

            if rng is None or robot is None or not consistent:
                continue
            rx, ry, ryaw = robot
            ox = rx + rng * math.cos(ryaw + bearing)
            oy = ry + rng * math.sin(ryaw + bearing)
            self.seen_now[name] = True
            if src == 'lidar':
                self.last_obs[name] = (self.now_sec(), rng)
            old, old_src = self.memory[name], self.memory_src[name]
            how = {'lidar': 'LiDAR', 'map': '지도(대략)', 'size': '크기로 추정(대략)'}[src]
            if old is None:
                self.get_logger().info(
                    f'{NAME_KR[name]} 발견! 지도 위치 ({ox:.2f}, {oy:.2f}), 거리 {rng:.2f}m [{how}]')
                self.memory[name], self.memory_src[name] = (ox, oy), src
            elif old_src in ('map', 'size') and src == 'lidar':
                # 지도로 잰 값은 물체 뒤의 벽까지 잡혔을 수 있으므로 LiDAR 값이 나오면 바로 교체
                self.get_logger().info(
                    f'{NAME_KR[name]} 위치를 LiDAR로 다시 확정: ({old[0]:.2f}, {old[1]:.2f}) → ({ox:.2f}, {oy:.2f})')
                self.memory[name], self.memory_src[name] = (ox, oy), 'lidar'
            elif old_src == 'map' and src == 'size':
                self.memory[name], self.memory_src[name] = (ox, oy), 'size'
            elif old_src == 'lidar' and src in ('map', 'size'):
                pass    # LiDAR로 확정된 위치는 대략값으로 덮어쓰지 않음
            else:       # 같은 방식으로 여러 번 본 값은 부드럽게 평균
                self.memory[name] = (0.7 * old[0] + 0.3 * ox, 0.7 * old[1] + 0.3 * oy)

        out = Image()
        out.header = msg.header
        out.height, out.width = debug.shape[:2]
        out.encoding = 'rgb8'
        out.step = out.width * 3
        out.data = debug.tobytes()
        self.debug_pub.publish(out)

    # ================================================================
    # Nav2 목표 보내기
    # ================================================================
    def go_to(self, x, y, yaw):
        self.nav_seq += 1
        seq = self.nav_seq
        goal = NavigateToPose.Goal()
        goal.pose = PoseStamped()
        goal.pose.header.frame_id = 'map'
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = float(x)
        goal.pose.pose.position.y = float(y)
        q = yaw_to_quat(yaw)
        goal.pose.pose.orientation.z = q[2]
        goal.pose.pose.orientation.w = q[3]
        self.nav_state = 'active'
        self.get_logger().info(f'Nav2 목표 전송: ({x:.2f}, {y:.2f}, {math.degrees(yaw):.0f}°)')
        fut = self.nav_client.send_goal_async(goal)
        fut.add_done_callback(lambda f: self.on_goal_response(f, seq))

    def on_goal_response(self, future, seq):
        handle = future.result()
        if seq != self.nav_seq:
            return
        if not handle.accepted:
            self.get_logger().warn('Nav2가 목표를 거절함')
            self.nav_state = 'failed'
            return
        self.goal_handle = handle
        handle.get_result_async().add_done_callback(lambda f: self.on_goal_result(f, seq))

    def on_goal_result(self, future, seq):
        if seq != self.nav_seq:
            return
        status = future.result().status
        if status == GoalStatus.STATUS_SUCCEEDED:
            self.nav_state = 'succeeded'
        else:
            self.get_logger().warn(f'Nav2 이동 실패 (status={status})')
            self.nav_state = 'failed'

    def cancel_nav(self):
        self.nav_seq += 1
        if self.goal_handle is not None:
            self.goal_handle.cancel_goal_async()
        self.nav_state = 'idle'

    def approach(self, name, standoff):
        """물체 앞 standoff 지점으로 이동 (물체를 바라보는 방향)"""
        target = self.memory[name]
        robot = self.robot_pose()
        if target is None or robot is None:
            self.nav_state = 'failed'
            return False
        dx, dy = target[0] - robot[0], target[1] - robot[1]
        d = math.hypot(dx, dy)
        yaw = math.atan2(dy, dx)
        if d <= standoff:
            gx, gy = robot[0], robot[1]
        else:
            gx = target[0] - standoff * dx / d
            gy = target[1] - standoff * dy / d
        self.approach_target = target
        self.face_name, self.faced_at = None, None
        self.go_to(gx, gy, yaw)
        return True

    def target_moved(self, name):
        """다가가는 도중 물체 위치가 0.5m 넘게 바뀌었나 (LiDAR로 다시 쟀을 때 등)"""
        t, a = self.memory[name], self.approach_target
        return t is not None and a is not None and math.hypot(t[0] - a[0], t[1] - a[1]) > 0.5

    # ================================================================
    # 문: 벽 방향 계산, 문 정면으로 가기, 통과 확인, 직진 통과
    # ================================================================
    def compute_door_normal(self):
        """문 주변 벽(문 + 양옆 칸막이)의 방향을 지도에서 구해서, 벽에 수직인 방향을 돌려줌"""
        m, d, r = self.latest_map, self.memory['door'], self.robot_pose()
        if m is None or d is None or r is None:
            return None
        res = m.info.resolution
        ox, oy = m.info.origin.position.x, m.info.origin.position.y
        W, H = m.info.width, m.info.height
        R = 1.2
        i0, i1 = int((d[0] - R - ox) / res), int((d[0] + R - ox) / res)
        j0, j1 = int((d[1] - R - oy) / res), int((d[1] + R - oy) / res)
        pts = []
        for j in range(max(0, j0), min(H, j1 + 1)):
            for i in range(max(0, i0), min(W, i1 + 1)):
                if m.data[j * W + i] >= 65:
                    x, y = ox + (i + 0.5) * res, oy + (j + 0.5) * res
                    if math.hypot(x - d[0], y - d[1]) <= R:
                        pts.append((x, y))
        if len(pts) < 15:
            return None
        P = np.array(pts)
        vals, vecs = np.linalg.eigh(np.cov((P - P.mean(axis=0)).T))
        if vals[1] < 5.0 * max(vals[0], 1e-6):      # 길쭉한 직선 모양이 아니면 믿지 않음
            return None
        tx, ty = vecs[0, 1], vecs[1, 1]             # 벽이 뻗은 방향
        nx, ny = -ty, tx                            # 벽에 수직인 방향
        if nx * (d[0] - r[0]) + ny * (d[1] - r[1]) < 0:   # 로봇 쪽 → 문 너머 방향으로 맞춤
            nx, ny = -nx, -ny
        return (nx, ny)

    def approach_door(self):
        """문 정면(벽에 수직인 선 위) DOOR_FRONT m 앞으로 이동. 벽 방향을 못 구하면 기존 방식"""
        self.door_normal = self.compute_door_normal()
        d = self.memory['door']
        if self.door_normal is None or d is None:
            self.get_logger().info('문 벽 방향을 못 구해서 로봇→문 방향으로 접근')
            return self.approach('door', STANDOFF['door'])
        nx, ny = self.door_normal
        self.approach_target = d
        self.face_name, self.faced_at = None, None
        self.get_logger().info('문 정면으로 이동')
        self.go_to(d[0] - DOOR_FRONT * nx, d[1] - DOOR_FRONT * ny, math.atan2(ny, nx))
        return True

    def door_normal_or_fallback(self):
        if self.door_normal is None:
            self.door_normal = self.compute_door_normal()
        if self.door_normal is None:                # 그래도 없으면 로봇→문 방향
            d, r = self.memory['door'], self.robot_pose()
            if d is None or r is None:
                return None
            dx, dy = d[0] - r[0], d[1] - r[1]
            n = math.hypot(dx, dy) or 1.0
            self.door_normal = (dx / n, dy / n)
        return self.door_normal

    def pass_depth(self):
        """로봇이 문 벽면을 몇 m 넘어갔나 (+면 문 너머, -면 아직 앞쪽)"""
        d, n, r = self.memory['door'], self.door_normal, self.robot_pose()
        if d is None or n is None or r is None:
            return None
        return (r[0] - d[0]) * n[0] + (r[1] - d[1]) * n[1]

    def warmup_goal(self, turn):
        """시작 이동: 현재 방향에서 turn(rad)만큼 돌린 쪽으로 WARMUP_DIST m 앞을 Nav2 목표로 보냄
        (바퀴에 직접 명령하면 LiDAR가 로봇 몸체를 장애물로 잡아 바로 멈추는 경우가 있어서 Nav2 사용)"""
        r = self.robot_pose()
        if r is None:
            self.nav_state = 'failed'
            return
        yaw = r[2] + turn
        self.go_to(r[0] + WARMUP_DIST * math.cos(yaw), r[1] + WARMUP_DIST * math.sin(yaw), yaw)

    def face_control(self):
        """도착 후 기억한 물체 위치 쪽으로 제자리 회전 (카메라로 다시 보기 위해)"""
        r, t = self.robot_pose(), self.memory[self.face_name]
        if r is None or t is None:
            self.face_name, self.faced_at = None, self.now_sec()
            return
        err = math.atan2(t[1] - r[1], t[0] - r[0]) - r[2]
        err = math.atan2(math.sin(err), math.cos(err))
        if abs(err) < 0.12:
            self.drive(0.0, 0.0)
            self.face_name, self.faced_at = None, self.now_sec()
        else:
            self.drive(0.0, max(-0.5, min(0.5, 1.5 * err)))

    def depth_str(self):
        d = self.pass_depth()
        return f'{d:.2f}m' if d is not None else '?'

    def passed_door(self):
        depth = self.pass_depth()
        return depth is not None and depth > PASS_DEPTH

    def find_door_gap(self):
        """문을 지운 뒤 LiDAR로 '뚫린 구간'을 찾아서, 문 벽면 위에서 그 구간의 가운데와 폭을 계산
        반환: (문 기준 옆 방향 보정값 m, 틈 폭 m) / 못 찾으면 None"""
        sc, r, d, n = self.latest_scan, self.robot_pose(), self.memory['door'], self.door_normal
        if sc is None or r is None or d is None or n is None:
            return None
        tx, ty = -n[1], n[0]
        wall = (d[0] - r[0]) * n[0] + (d[1] - r[1]) * n[1]      # 로봇 → 문 벽면 거리
        if wall < 0.2:
            return None
        rays = []
        for k, rng in enumerate(sc.ranges):
            a = r[2] + sc.angle_min + k * sc.angle_increment
            ux, uy = math.cos(a), math.sin(a)
            un = ux * n[0] + uy * n[1]
            if un < 0.3:                                       # 문 쪽을 향한 광선만 사용
                continue
            if not math.isfinite(rng) or rng >= sc.range_max:
                rng = sc.range_max
            hx, hy = r[0] + rng * ux, r[1] + rng * uy
            open_ = (hx - d[0]) * n[0] + (hy - d[1]) * n[1] > 0.35   # 벽면 너머까지 뚫려 있음
            lam = wall / un
            side = (r[0] + lam * ux - d[0]) * tx + (r[1] + lam * uy - d[1]) * ty
            rays.append((side, open_))
        if len(rays) < 10:
            return None
        rays.sort()
        best, cur = [], []
        for side, open_ in rays:
            if open_:
                cur.append(side)
                if len(cur) > len(best):
                    best = list(cur)
            else:
                cur = []
        if len(best) < 3:
            return None
        width = best[-1] - best[0]
        if width < 0.6 or width > 3.0:
            return None
        return (best[0] + best[-1]) / 2.0, width

    def front_min(self):
        """정면 ±20도 안에서 가장 가까운 장애물 거리"""
        s = self.latest_scan
        if s is None:
            return None
        best = None
        n = len(s.ranges)
        for k in range(n):
            a = math.atan2(math.sin(s.angle_min + k * s.angle_increment),
                           math.cos(s.angle_min + k * s.angle_increment))
            if abs(a) > math.radians(20):
                continue
            r = s.ranges[k]
            if math.isfinite(r) and s.range_min < r < s.range_max:
                best = r if best is None else min(best, r)
        return best

    def drive_control(self):
        """10Hz: WARMUP(시작 이동)과 DRIVE_THROUGH(문 통과)일 때만 바퀴를 직접 움직임"""
        if self.face_name is not None and self.state in ('APPROACH_KEY', 'APPROACH_DOOR', 'APPROACH_EXIT'):
            self.face_control()
            return
        if self.state != 'DRIVE_THROUGH' or self.drive_phase in (None, 'done', 'blocked'):
            return
        r, n, d = self.robot_pose(), self.door_normal, self.memory['door']
        if r is None or n is None or d is None:
            self.drive(0.0, 0.0)
            return
        if self.passed_door():
            self.drive(0.0, 0.0)
            self.drive_phase = 'done'
            return
        tx, ty = -n[1], n[0]
        door_yaw = math.atan2(n[1], n[0])
        wrap = lambda x: math.atan2(math.sin(x), math.cos(x))

        # 1) 문 쪽을 바라보기
        if self.drive_phase == 'face':
            err = wrap(door_yaw - r[2])
            if abs(err) < 0.06:
                self.drive(0.0, 0.0)
                self.drive_phase = 'scan'
                self.drive_t0 = self.now_sec()
            else:
                self.drive(0.0, max(-0.5, min(0.5, 1.5 * err)))
            return

        # 2) 멈춘 상태에서 LiDAR로 문 틈 재기 → 문 중심 보정
        if self.drive_phase == 'scan':
            self.drive(0.0, 0.0)
            if self.now_sec() - self.drive_t0 < 1.0:          # 새 스캔이 들어오게 잠깐 대기
                return
            gap = self.find_door_gap()
            if gap is not None:
                shift, width = gap
                self.memory['door'] = (d[0] + shift * tx, d[1] + shift * ty)
                d = self.memory['door']
                self.get_logger().info(f'LiDAR로 문 틈 확인: 폭 {width:.2f}m, 문 중심을 옆으로 {shift:+.2f}m 보정')
            else:
                self.get_logger().warn('LiDAR로 문 틈을 찾지 못함 → 기억한 문 위치로 진행')
            side = (r[0] - d[0]) * tx + (r[1] - d[1]) * ty
            self.drive_phase = 'goto' if abs(side) > 0.12 else 'align'
            if self.drive_phase == 'goto':
                self.get_logger().info(f'문 중심선에서 {side:+.2f}m 벗어나 있어서 옆으로 이동')
            return

        # 3) 문 중심선 위의 문 앞 지점으로 이동
        if self.drive_phase == 'goto':
            gx, gy = d[0] - DOOR_FRONT * n[0], d[1] - DOOR_FRONT * n[1]
            dist = math.hypot(gx - r[0], gy - r[1])
            if dist < 0.06:
                self.drive(0.0, 0.0)
                self.drive_phase = 'align'
                return
            err = wrap(math.atan2(gy - r[1], gx - r[0]) - r[2])
            if abs(err) > 0.15:
                self.drive(0.0, max(-0.5, min(0.5, 1.5 * err)))
            else:
                front = self.front_min()
                if front is not None and front < FRONT_STOP:
                    self.drive(0.0, 0.0)
                    self.drive_phase = 'align'            # 더 못 가면 그 자리에서 정렬해서 진행
                    return
                self.drive(min(0.12, 0.8 * dist), max(-0.4, min(0.4, 1.5 * err)))
            return

        # 문 중심선에서 옆으로 벗어난 만큼 방향을 살짝 틀어서 가운데로 들어가게 함
        side = (r[0] - d[0]) * tx + (r[1] - d[1]) * ty
        want = door_yaw - max(-0.35, min(0.35, 1.0 * side))
        err = wrap(want - r[2])
        if self.drive_phase == 'align':
            if abs(err) < 0.06:
                self.drive_phase = 'forward'
                self.get_logger().info('문 정면 정렬 완료 → LiDAR로 앞을 보며 직진')
            else:
                self.drive(0.0, max(-0.5, min(0.5, 1.5 * err)))
            return
        front = self.front_min()
        if front is not None and front < FRONT_STOP:
            self.drive(0.0, 0.0)
            self.drive_phase = 'blocked'
            self.get_logger().warn(f'직진 중 정면 {front:.2f}m에 장애물 → 정지')
            return
        self.drive(DRIVE_SPEED, max(-0.3, min(0.3, 1.5 * err)))

    def drive(self, vx, wz):
        """바퀴에 직접 속도 명령 (Nav2가 쉬고 있을 때만 사용)"""
        t = TwistStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'base_link'
        t.twist.linear.x = float(vx)
        t.twist.angular.z = float(wz)
        self.cmd_pub.publish(t)

    def spin(self, wz):
        """제자리 회전"""
        self.drive(0.0, wz)

    # ================================================================
    # Gazebo 모델 삭제 (열쇠 줍기 / 문 열기)
    # ================================================================
    def find_gz(self):
        path = shutil.which('gz')
        if path is None:
            cands = glob.glob('/opt/ros/*/opt/gz_tools_vendor/bin/gz')
            path = cands[0] if cands else None
        if path is None:
            self.get_logger().warn('gz 명령을 못 찾음. 문은 수동으로 지워야 할 수 있음')
        return path

    def remove_model(self, model):
        req = f'name: "{model}" type: MODEL'
        cmd = [self.gz_cmd or 'gz', 'service', '-s', f'/world/{WORLD_NAME}/remove',
               '--reqtype', 'gz.msgs.Entity', '--reptype', 'gz.msgs.Boolean',
               '--timeout', '3000', '--req', req]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=6)
            ok = 'true' in res.stdout
        except Exception as e:
            self.get_logger().error(f'모델 삭제 명령 실패: {e}')
            ok = False
        if ok:
            self.get_logger().info(f'Gazebo에서 {model} 삭제 완료')
        else:
            self.get_logger().warn(f'{model} 삭제 실패. 직접 실행해보세요:\n  {" ".join(cmd[:-1])} \'{req}\'')
        return ok

    # ================================================================
    # FSM: 상태에 들어갈 때 한 번 실행
    # ================================================================
    def on_enter(self, s):
        if s == 'WARMUP':
            r = self.robot_pose()
            self.warmup_start = (r[0], r[1]) if r else None
            self.warmup_try = 1
            self.get_logger().info(f'방탈출 시작! 탐색 전에 Nav2로 {WARMUP_DIST}m 이동합니다 (출발점 구멍 방지)')
            self.warmup_goal(0.0)
        elif s in ('EXPLORE_KEY', 'SEARCH_DOOR', 'SEARCH_EXIT'):
            self.search_state = s
            self.stuck_ref = None
            self.cancel_nav()
            self.explore(True)
        elif s == 'LOOK_AROUND':
            self.explore(False)
            self.cancel_nav()
            r = self.robot_pose()
            if r is not None:
                self.looked_at.append((r[0], r[1]))
            self.get_logger().info('제자리에서 한 바퀴 둘러봅니다')
        elif s == 'WANDER':
            p = self.pick_wander_point()
            if p is None:
                self.nav_state = 'nopoint'
            else:
                r = self.robot_pose()
                yaw = math.atan2(p[1] - r[1], p[0] - r[0]) if r else 0.0
                self.get_logger().info(f'아직 안 둘러본 곳 ({p[0]:.2f}, {p[1]:.2f})으로 이동')
                self.go_to(p[0], p[1], yaw)
        elif s in ('STOP_FOR_KEY', 'STOP_FOR_DOOR', 'STOP_FOR_EXIT'):
            self.explore(False)
        elif s == 'APPROACH_KEY':
            self.approach('key', STANDOFF['key'])
        elif s == 'APPROACH_DOOR':
            self.approach_door()
        elif s == 'APPROACH_EXIT':
            self.approach('exit', STANDOFF['exit'])
        elif s == 'PICK_KEY':
            # 피드백 반영: Gazebo에서 열쇠 삭제가 '성공'했을 때만 has_key = True
            self.cancel_nav()
            self.key_try = 0
            self.try_pick_key()
        elif s == 'OPEN_DOOR':
            self.cancel_nav()
            self.door_robot_pose = self.robot_pose()
            self.door_normal_or_fallback()
            self.door_try = 0
            if self.has_key and not self.door_open:
                self.get_logger().info('열쇠가 있으므로 문을 엽니다')
                self.try_open_door()
        elif s == 'DRIVE_THROUGH':
            self.cancel_nav()
            self.drive_phase = 'face'
            self.drive_t0 = self.now_sec()
            self.get_logger().info('문 쪽을 보고 LiDAR로 문 틈을 잰 뒤, 가운데로 맞춰서 직진합니다')
        elif s == 'PASS_DOOR':
            self.pass_door()
        elif s == 'ASSIST_PASS':
            self.get_logger().warn('직진·Nav2 통과에 실패 → 탐색으로 움직이면서 문 통과를 계속 확인합니다')
            self.cancel_nav()
            self.explore(True)
        elif s == 'DONE':
            self.cancel_nav()
            self.explore(False)
            self.get_logger().info('=' * 40)
            self.get_logger().info('탈출 성공! LEVEL 1 임무 완료')
            self.get_logger().info('=' * 40)

    def pass_door(self):
        """(예비) Nav2로 문 너머 PASS_DISTANCE m 지점까지 이동"""
        door, n = self.memory['door'], self.door_normal_or_fallback()
        if door is None or n is None:
            self.nav_state = 'failed'
            return
        self.go_to(door[0] + PASS_DISTANCE * n[0], door[1] + PASS_DISTANCE * n[1], math.atan2(n[1], n[0]))

    def try_pick_key(self):
        self.key_try += 1
        if self.remove_model(KEY_MODEL):
            self.has_key = True
            self.get_logger().info('열쇠 획득! Gazebo에서 열쇠 제거 확인 (has_key = True)')
        else:
            self.get_logger().warn(f'열쇠 제거 실패 ({self.key_try}/3) → 아직 획득하지 않은 것으로 처리')

    def try_open_door(self):
        self.door_try += 1
        self.door_open = self.remove_model(DOOR_MODEL)
        if self.door_open:
            self.get_logger().info('문 개방 확인 (Gazebo에서 문 제거 성공)')
        else:
            self.get_logger().warn(f'문 제거 실패 ({self.door_try}/3)')

    # ================================================================
    # FSM: 0.5초마다 실행 (조건 확인 → 다음 상태)
    # ================================================================
    def tick(self):
        s = self.state

        # 시작 전: Gazebo 시계와 로봇 위치를 받을 때까지 기다림
        if s is None:
            if self.now_sec() > 0.0 and self.robot_pose() is not None:
                self.change_state('WARMUP')
            return

        # explore_lite가 혼자 꺼졌으면 다시 켬
        if s in ('EXPLORE_KEY', 'SEARCH_DOOR', 'SEARCH_EXIT') and not self.explore_running():
            self.explore(True)

        # 탐색 중 막힘 감지 (explore_lite가 같은 자리만 맴돌거나 탐색이 끝나버린 경우)
        if s in ('EXPLORE_KEY', 'SEARCH_DOOR', 'SEARCH_EXIT'):
            r = self.robot_pose()
            now = self.now_sec()
            if r is not None:
                if self.stuck_ref is None or math.hypot(r[0] - self.stuck_ref[1],
                                                        r[1] - self.stuck_ref[2]) > STUCK_DIST:
                    self.stuck_ref = (now, r[0], r[1])
                elif now - self.stuck_ref[0] > STUCK_TIME:
                    self.change_state('LOOK_AROUND')
                    return

        # 둘러보기: 1.5초 대기(탐색 정지 반영) → 약 380도 회전 → 원래 탐색으로 복귀
        if s == 'LOOK_AROUND':
            want = {'EXPLORE_KEY': 'key', 'SEARCH_DOOR': 'door', 'SEARCH_EXIT': 'exit'}[self.search_state]
            nxt = {'key': 'STOP_FOR_KEY', 'door': 'STOP_FOR_DOOR', 'exit': 'STOP_FOR_EXIT'}[want]
            t = self.elapsed()
            if self.memory[want] is not None and t > 1.5:
                self.spin(0.0)
                self.change_state(nxt)
            elif t < 1.5:
                pass
            elif t < 1.5 + 6.7 / LOOK_SPEED:     # 약 380도
                self.spin(LOOK_SPEED)
            else:
                self.spin(0.0)
                self.get_logger().info('둘러봤지만 못 찾음 → 다른 곳으로 이동')
                self.change_state('WANDER')
            return

        # 다른 곳으로 이동: 가는 도중에 목표가 보이면 바로 멈추고, 도착(또는 실패)하면 다시 둘러보기
        if s == 'WANDER':
            want = {'EXPLORE_KEY': 'key', 'SEARCH_DOOR': 'door', 'SEARCH_EXIT': 'exit'}[self.search_state]
            if self.memory[want] is not None:
                nxt = {'key': 'STOP_FOR_KEY', 'door': 'STOP_FOR_DOOR', 'exit': 'STOP_FOR_EXIT'}[want]
                self.cancel_nav()
                self.change_state(nxt)
            elif self.nav_state == 'nopoint':
                # 갈 곳을 못 고르면 같은 자리에서 다시 돌지 말고 explore_lite 탐색으로 돌아감
                self.wander_fail = 0
                self.get_logger().info('갈 곳이 없어서 explore_lite 탐색으로 돌아갑니다')
                self.change_state(self.search_state)
            elif self.nav_state == 'failed':
                self.wander_fail += 1
                if self.wander_fail >= 3:
                    self.wander_fail = 0
                    self.get_logger().info('이동이 계속 실패해서 explore_lite 탐색으로 돌아갑니다')
                    self.change_state(self.search_state)
                else:
                    self.change_state('WANDER')       # 다른 곳을 다시 골라봄
            elif self.nav_state == 'succeeded' or self.elapsed() > 60.0:
                self.wander_fail = 0
                self.change_state('LOOK_AROUND')
            return

        # 0) 시작 이동: Nav2 이동이 끝나면 실제로 움직인 거리 확인 → 부족하면 방향을 바꿔 한 번 더
        if s == 'WARMUP':
            if self.nav_state in ('succeeded', 'failed') or self.elapsed() > 30.0:
                r = self.robot_pose()
                moved = math.hypot(r[0] - self.warmup_start[0], r[1] - self.warmup_start[1]) \
                    if (r and self.warmup_start) else 0.0
                if moved < 0.5 and self.warmup_try < 2:
                    self.warmup_try += 1
                    self.get_logger().info(f'시작 이동이 {moved:.2f}m뿐이라 방향을 90도 바꿔 다시 이동')
                    self.state_start = self.now_sec()
                    self.warmup_goal(math.pi / 2)
                else:
                    self.get_logger().info(f'시작 이동 완료 ({moved:.2f}m)')
                    self.change_state('EXPLORE_KEY')
            return

        # 1) 열쇠 찾기 (탐색 중 문을 보면 위치만 기억해 둠)
        if s == 'EXPLORE_KEY':
            if self.seen_now['key'] or self.memory['key'] is not None:
                self.change_state('STOP_FOR_KEY')

        # 2) 탐색 정지 후 1초 기다림 (explore_lite의 '목표 전부 취소'가 끝나도록 + 위치 다시 측정)
        elif s == 'STOP_FOR_KEY':
            if self.elapsed() > 1.0:
                self.change_state('APPROACH_KEY')

        # 3) 열쇠로 이동
        elif s == 'APPROACH_KEY':
            if self.nav_state == 'active' and self.target_moved('key'):
                self.get_logger().info('열쇠 위치가 바뀌어서 목표를 다시 잡습니다')
                self.approach('key', STANDOFF['key'])
            elif self.confirmed_near('key', PICK_DISTANCE * 0.8):
                self.change_state('PICK_KEY')
            elif self.nav_state == 'succeeded':
                # 도착하면 열쇠 쪽으로 돌아서 다시 보고, 최근에 가까이서 직접 본 경우에만 줍기
                if not self.arrived_check('key'):
                    return
                if self.confirmed_near('key', PICK_DISTANCE):
                    self.change_state('PICK_KEY')
                else:
                    self.not_confirmed('key', 'STOP_FOR_KEY', 'EXPLORE_KEY')
            elif self.nav_state == 'failed':
                if self.confirmed_near('key', PICK_DISTANCE):
                    self.change_state('PICK_KEY')
                elif self.retry < 3:
                    self.retry += 1
                    self.get_logger().info(f'열쇠 접근 재시도 {self.retry}/3')
                    self.approach('key', STANDOFF['key'] + 0.2 * self.retry)
                else:
                    self.forget('key')             # 위치가 틀렸을 수 있으니 잊고 다시 찾기
                    self.change_state('EXPLORE_KEY')

        # 4) 열쇠 획득 → 문 위치를 알면 바로 문으로, 모르면 탐색
        elif s == 'PICK_KEY':
            if self.has_key:
                if self.elapsed() > 1.0:
                    if self.memory['door'] is not None:
                        self.change_state('APPROACH_DOOR')
                    else:
                        self.change_state('SEARCH_DOOR')
            elif self.elapsed() > 2.0 * self.key_try:      # 2초 간격으로 다시 시도
                if self.key_try < 3:
                    self.try_pick_key()
                else:
                    self.get_logger().warn('열쇠 제거가 계속 실패 → 열쇠에 다시 다가가서 시도합니다')
                    self.change_state('APPROACH_KEY')

        # 5) 문 찾기
        elif s == 'SEARCH_DOOR':
            if self.seen_now['door'] or self.memory['door'] is not None:
                self.change_state('STOP_FOR_DOOR')

        elif s == 'STOP_FOR_DOOR':
            if self.elapsed() > 1.0:
                self.change_state('APPROACH_DOOR')

        # 6) 문으로 이동
        elif s == 'APPROACH_DOOR':
            if self.nav_state == 'active' and self.target_moved('door'):
                self.get_logger().info('문 위치가 바뀌어서 목표를 다시 잡습니다')
                self.approach_door()
            elif self.nav_state == 'succeeded':
                if not self.arrived_check('door'):
                    return
                if self.confirmed_near('door', 1.5):
                    self.change_state('OPEN_DOOR')
                else:
                    self.not_confirmed('door', 'STOP_FOR_DOOR', 'SEARCH_DOOR')
            elif self.nav_state == 'failed':
                if self.confirmed_near('door', 1.3):
                    self.change_state('OPEN_DOOR')
                elif self.retry < 3:
                    self.retry += 1
                    self.get_logger().info(f'문 접근 재시도 {self.retry}/3')
                    self.approach('door', STANDOFF['door'] + 0.2 * self.retry)
                else:
                    self.forget('door')
                    self.change_state('SEARCH_DOOR')

        # 7) 문 열기: 열쇠 확인 → Gazebo에서 문 삭제 (성공 확인될 때까지 재시도)
        elif s == 'OPEN_DOOR':
            if not self.has_key:
                self.get_logger().warn('열쇠가 없어서 문을 열 수 없음 → 열쇠 탐색')
                self.change_state('EXPLORE_KEY')
            elif self.door_open:
                if self.elapsed() > 1.0:
                    self.change_state('DRIVE_THROUGH')
            elif self.elapsed() > 2.0 * self.door_try:
                if self.door_try < 3:
                    self.try_open_door()
                else:
                    self.get_logger().error('문 제거가 계속 실패해서 통과할 수 없음. Gazebo 연결을 확인하세요 (5초마다 재시도)')
                    self.door_try = 0
                    self.state_start = self.now_sec() + 3.0

        # 8) 문 통과: 문 정면에서 LiDAR로 앞을 보며 직진 (SLAM 지도에 문이 남아 있어도 통과 가능)
        elif s == 'DRIVE_THROUGH':
            if self.drive_phase == 'done' or self.passed_door():
                self.get_logger().info(f'문 통과 확인 (문 벽면에서 {self.depth_str()} 넘어감)')
                self.change_state('SEARCH_EXIT')
            elif self.drive_phase == 'blocked' or self.elapsed() > DRIVE_TIMEOUT:
                self.change_state('PASS_DOOR')

        # 8-1) 예비: Nav2로 통과 시도. 통과가 '확인'돼야만 탈출구 찾기로 넘어감
        elif s == 'PASS_DOOR':
            if self.passed_door():
                self.get_logger().info(f'문 통과 확인 (문 벽면에서 {self.depth_str()} 넘어감)')
                self.change_state('SEARCH_EXIT')
            elif self.nav_state in ('succeeded', 'failed'):
                if self.retry < 2:
                    self.retry += 1
                    self.get_logger().info(f'Nav2로 문 통과 재시도 {self.retry}/2 (3초 후)')
                    self.nav_state = 'waiting'
                    self.create_timer_once(
                        3.0, lambda: self.pass_door() if self.state == 'PASS_DOOR' else None)
                else:
                    self.change_state('ASSIST_PASS')

        # 8-2) 최후 수단: 탐색으로 움직이면서 통과 여부만 계속 확인 (통과 전에는 다음 단계로 안 감)
        elif s == 'ASSIST_PASS':
            if self.passed_door():
                self.get_logger().info(f'문 통과 확인 (문 벽면에서 {self.depth_str()} 넘어감)')
                self.change_state('SEARCH_EXIT')
            elif self.elapsed() > 90.0:
                self.explore(False)
                self.change_state('APPROACH_DOOR')     # 문 정면으로 다시 가서 직진 재시도

        # 9) 탈출구 찾기
        elif s == 'SEARCH_EXIT':
            if self.seen_now['exit'] or self.memory['exit'] is not None:
                self.change_state('STOP_FOR_EXIT')

        elif s == 'STOP_FOR_EXIT':
            if self.elapsed() > 1.0:
                self.change_state('APPROACH_EXIT')

        # 10) 탈출구로 이동
        elif s == 'APPROACH_EXIT':
            if self.nav_state == 'active' and self.target_moved('exit'):
                self.get_logger().info('탈출구 위치가 바뀌어서 목표를 다시 잡습니다')
                self.approach('exit', STANDOFF['exit'])
            elif self.confirmed_near('exit', 0.9):
                self.change_state('DONE')
            elif self.nav_state == 'succeeded':
                if not self.arrived_check('exit'):
                    return
                if self.confirmed_near('exit', 1.1):
                    self.change_state('DONE')
                else:
                    self.not_confirmed('exit', 'STOP_FOR_EXIT', 'SEARCH_EXIT')
            elif self.nav_state == 'failed':
                if self.retry < 3:
                    self.retry += 1
                    self.approach('exit', STANDOFF['exit'] + 0.2 * self.retry)
                else:
                    self.forget('exit')
                    self.change_state('SEARCH_EXIT')

    def create_timer_once(self, sec, fn):
        holder = {}

        def cb():
            holder['t'].cancel()
            self.destroy_timer(holder['t'])
            fn()
        holder['t'] = self.create_timer(sec, cb)


def main():
    rclpy.init()
    node = EscapeFSM()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.explore_running():          # 노드를 끄면 explore_lite도 같이 끔
            try:
                os.killpg(node.explore_proc.pid, signal.SIGINT)
            except Exception:
                pass
        try:
            node.spin(0.0)
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
