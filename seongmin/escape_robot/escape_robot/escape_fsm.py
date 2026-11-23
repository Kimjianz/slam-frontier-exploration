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
WORLD_NAME = 'maze'          # escape_room.world 안의 <world name="maze">
KEY_MODEL = 'red_key'        # Gazebo 속 열쇠 모델 이름 (색은 노란색)
DOOR_MODEL = 'red_door'      # Gazebo 속 문 모델 이름

CAMERA_HFOV = 1.02974        # TurtleBot3 waffle 카메라 수평 화각 (rad)
PROC_WIDTH = 320             # 영상을 가로 약 320픽셀로 줄여서 처리 (VM 부담 줄이기)
MIN_BLOB_FRAC = 0.0008       # 줄인 영상 전체 픽셀 중 이 비율 이상이 그 색이면 '보인다'고 판단
MAX_DETECT_RANGE = 3.3       # 레이저로 거리를 잴 수 있는 최대 거리 (센서 최대 3.5m)
MAX_MAP_RANGE = 8.0          # 레이저로 못 재면 SLAM 지도에서 이 거리까지 선을 그어 찾음
STUCK_TIME = 20.0            # 탐색 중 이 시간 동안
STUCK_DIST = 0.3             # 이 거리보다 적게 움직이면 '막힘' → 제자리 한 바퀴 둘러보기

STANDOFF = {'key': 0.55, 'door': 0.75, 'exit': 0.6}   # 물체 앞 몇 m에 멈출지
PICK_DISTANCE = 1.0          # 열쇠와 이 거리 안이면 '획득'
PASS_DISTANCE = 1.2          # 문 너머 몇 m 지점까지 통과할지

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
               '-p', 'planner_frequency:=0.5', '-p', 'progress_timeout:=20.0',
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

        self.get_logger().info('Nav2 액션 서버 기다리는 중...')
        self.nav_client.wait_for_server()
        self.get_logger().info('Nav2 연결 완료. 방탈출 시작!')
        self.change_state('EXPLORE_KEY')

    # ================================================================
    # 유틸
    # ================================================================
    def now_sec(self):
        return self.get_clock().now().nanoseconds / 1e9

    def elapsed(self):
        return self.now_sec() - self.state_start

    def change_state(self, new_state):
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
            if (rng is None or rng > MAX_DETECT_RANGE) and robot is not None:
                rng = self.map_raycast(robot[0], robot[1], robot[2] + bearing)

            ys, xs = np.nonzero(mask)
            cv2.rectangle(debug, (int(xs.min()), int(ys.min())), (int(xs.max()), int(ys.max())),
                          DRAW_COLOR[name], 2)
            label = f'{name} {rng:.2f}m' if rng else f'{name} far'
            cv2.putText(debug, label, (int(xs.min()), max(12, int(ys.min()) - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, DRAW_COLOR[name], 1)

            if rng is None or robot is None:
                continue
            rx, ry, ryaw = robot
            ox = rx + rng * math.cos(ryaw + bearing)
            oy = ry + rng * math.sin(ryaw + bearing)
            self.seen_now[name] = True
            old = self.memory[name]
            if old is None:
                self.get_logger().info(f'{NAME_KR[name]} 발견! 지도 위치 ({ox:.2f}, {oy:.2f}), 거리 {rng:.2f}m')
                self.memory[name] = (ox, oy)
            else:   # 여러 번 본 값을 부드럽게 평균
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
        self.go_to(gx, gy, yaw)
        return True

    def spin(self, wz):
        """제자리 회전 명령 (Nav2가 쉬고 있을 때만 사용)"""
        t = TwistStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'base_link'
        t.twist.angular.z = float(wz)
        self.cmd_pub.publish(t)

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
        if s in ('EXPLORE_KEY', 'SEARCH_DOOR', 'SEARCH_EXIT'):
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
            self.approach('door', STANDOFF['door'])
        elif s == 'APPROACH_EXIT':
            self.approach('exit', STANDOFF['exit'])
        elif s == 'PICK_KEY':
            self.cancel_nav()
            self.has_key = True
            self.get_logger().info('열쇠 획득! (has_key = True)')
            self.remove_model(KEY_MODEL)
        elif s == 'OPEN_DOOR':
            self.cancel_nav()
            self.door_robot_pose = self.robot_pose()
            if self.has_key:
                self.get_logger().info('열쇠가 있으므로 문을 엽니다')
                self.door_open = self.remove_model(DOOR_MODEL)
            self.turn_step = 0
        elif s == 'PASS_DOOR':
            self.pass_door()
        elif s == 'DONE':
            self.cancel_nav()
            self.explore(False)
            self.get_logger().info('=' * 40)
            self.get_logger().info('탈출 성공! LEVEL 1 임무 완료')
            self.get_logger().info('=' * 40)

    def pass_door(self):
        door, rp = self.memory['door'], self.door_robot_pose
        if door is None or rp is None:
            return
        dx, dy = door[0] - rp[0], door[1] - rp[1]
        d = math.hypot(dx, dy) or 1.0
        yaw = math.atan2(dy, dx)
        self.go_to(door[0] + PASS_DISTANCE * dx / d, door[1] + PASS_DISTANCE * dy / d, yaw)

    # ================================================================
    # FSM: 0.5초마다 실행 (조건 확인 → 다음 상태)
    # ================================================================
    def tick(self):
        s = self.state

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
            elif t < 12.5:
                self.spin(0.6)
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

        # 1) 열쇠 찾기 (탐색 중 문을 보면 위치만 기억해 둠)
        if s == 'EXPLORE_KEY':
            if self.seen_now['key'] or self.memory['key'] is not None:
                self.change_state('STOP_FOR_KEY')

        # 2) 탐색 정지 후 2초 기다림 (explore_lite의 '목표 전부 취소'가 끝나도록 + 위치 다시 측정)
        elif s == 'STOP_FOR_KEY':
            if self.elapsed() > 2.0:
                self.change_state('APPROACH_KEY')

        # 3) 열쇠로 이동
        elif s == 'APPROACH_KEY':
            if self.nav_state == 'succeeded' or self.dist_to(self.memory['key']) < PICK_DISTANCE * 0.8:
                self.change_state('PICK_KEY')
            elif self.nav_state == 'failed':
                if self.dist_to(self.memory['key']) < PICK_DISTANCE:
                    self.change_state('PICK_KEY')
                elif self.retry < 3:
                    self.retry += 1
                    self.get_logger().info(f'열쇠 접근 재시도 {self.retry}/3')
                    self.approach('key', STANDOFF['key'] + 0.2 * self.retry)
                else:
                    self.memory['key'] = None      # 위치가 틀렸을 수 있으니 잊고 다시 찾기
                    self.change_state('EXPLORE_KEY')

        # 4) 열쇠 획득 → 문 위치를 알면 바로 문으로, 모르면 탐색
        elif s == 'PICK_KEY':
            if self.elapsed() > 1.0:
                if self.memory['door'] is not None:
                    self.change_state('APPROACH_DOOR')
                else:
                    self.change_state('SEARCH_DOOR')

        # 5) 문 찾기
        elif s == 'SEARCH_DOOR':
            if self.seen_now['door'] or self.memory['door'] is not None:
                self.change_state('STOP_FOR_DOOR')

        elif s == 'STOP_FOR_DOOR':
            if self.elapsed() > 2.0:
                self.change_state('APPROACH_DOOR')

        # 6) 문으로 이동
        elif s == 'APPROACH_DOOR':
            if self.nav_state == 'succeeded':
                self.change_state('OPEN_DOOR')
            elif self.nav_state == 'failed':
                if self.dist_to(self.memory['door']) < 1.3:
                    self.change_state('OPEN_DOOR')
                elif self.retry < 3:
                    self.retry += 1
                    self.get_logger().info(f'문 접근 재시도 {self.retry}/3')
                    self.approach('door', STANDOFF['door'] + 0.2 * self.retry)
                else:
                    self.memory['door'] = None
                    self.change_state('SEARCH_DOOR')

        # 7) 문 열기: 문 삭제 → 제자리에서 좌우로 돌아서 SLAM 지도 갱신 → 지도 반영 대기
        elif s == 'OPEN_DOOR':
            if not self.has_key:
                self.get_logger().warn('열쇠가 없어서 문을 열 수 없음 → 열쇠 탐색')
                self.change_state('EXPLORE_KEY')
                return
            if self.door_robot_pose is None:
                self.door_robot_pose = self.robot_pose()
                return
            # 문이 사라진 공간을 SLAM이 다시 스캔하도록 제자리에서 왼쪽 → 오른쪽으로 회전
            # (SLAM은 로봇이 0.5m 이동하거나 0.5rad 회전해야 지도를 갱신함)
            t = self.elapsed()
            if t < 1.0:
                pass
            elif t < 4.5:
                self.spin(0.5)            # 왼쪽으로 약 100도
            elif t < 8.0:
                self.spin(-0.5)           # 다시 오른쪽으로
            elif t < 8.6:
                self.spin(0.0)
                if self.turn_step == 0:
                    self.turn_step = 1
                    self.get_logger().info('SLAM 지도에 문이 사라진 게 반영되길 기다리는 중 (6초)')
            elif t > 14.0:
                self.change_state('PASS_DOOR')

        # 8) 문 통과 (지도가 아직 안 바뀌었으면 몇 번 재시도)
        elif s == 'PASS_DOOR':
            if self.seen_now['exit']:
                self.change_state('STOP_FOR_EXIT')
            elif self.nav_state == 'succeeded':
                self.change_state('SEARCH_EXIT')
            elif self.nav_state == 'failed':
                if self.retry < 8:
                    self.retry += 1
                    self.get_logger().info(f'문 통과 재시도 {self.retry}/8 (3초 후)')
                    self.nav_state = 'waiting'
                    self.create_timer_once(
                        3.0, lambda: self.pass_door() if self.state == 'PASS_DOOR' else None)
                else:
                    self.change_state('SEARCH_EXIT')

        # 9) 탈출구 찾기
        elif s == 'SEARCH_EXIT':
            if self.seen_now['exit'] or self.memory['exit'] is not None:
                self.change_state('STOP_FOR_EXIT')

        elif s == 'STOP_FOR_EXIT':
            if self.elapsed() > 2.0:
                self.change_state('APPROACH_EXIT')

        # 10) 탈출구로 이동
        elif s == 'APPROACH_EXIT':
            if self.nav_state == 'succeeded' or self.dist_to(self.memory['exit']) < 0.9:
                self.change_state('DONE')
            elif self.nav_state == 'failed':
                if self.retry < 3:
                    self.retry += 1
                    self.approach('exit', STANDOFF['exit'] + 0.2 * self.retry)
                else:
                    self.memory['exit'] = None
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
