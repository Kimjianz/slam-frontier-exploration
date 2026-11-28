"""레벨 1 자율 방탈출 노드

탐색(explore_lite) → 인식(카메라 + LiDAR) → 위치 기억 → Nav2로 근처 이동
→ 직접 속도 제어로 접근 → 획득·개방(Gazebo 모델 제거) → 문 통과 → 탈출구 도착
"""
import math
import subprocess
from collections import deque

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import (QoSProfile, DurabilityPolicy, ReliabilityPolicy,
                       qos_profile_sensor_data)
from rclpy.time import Time
from std_msgs.msg import Bool
from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import Image, LaserScan
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import Marker, MarkerArray
from nav2_msgs.action import NavigateToPose
from action_msgs.msg import GoalStatus, GoalStatusArray
from action_msgs.srv import CancelGoal
from tf2_ros import Buffer, TransformListener
try:
    from slam_toolbox.srv import Pause
except ImportError:
    Pause = None

# ================= 설정값 =================
WORLD = 'maze'                    # Gazebo 월드 이름 (팀 escape_room.world)
FOV = 1.4                         # 카메라 시야각 (rad), model.sdf의 horizontal_fov와 같아야 함
MIN_AREA = 50                     # 이보다 작은 색 영역은 무시 (픽셀)
POS_MAX_ANGLE = math.radians(25)  # 화면 중앙 ±25도 안에서만 위치 계산
MAX_W = 0.3                       # 모든 회전 최대 속도 (rad/s), SLAM 안정용
FRONT_STOP = 0.18                 # 앞 장애물이 이보다 가까우면 전진 정지 (m)
REAR_STOP = 0.18                  # 뒤 장애물이 이보다 가까우면 후진 정지 (m)
APPROACH_DIST = 0.5               # 획득 판정 거리 (m)
APPROACH_COUNT = 5                # 연속 확인 횟수
APPROACH_TIMEOUT = 40.0           # 목표를 이 시간 동안 못 보면 실패 (s)
APPROACH_LIMIT = 60.0             # 접근 전체 제한 시간 (s)
STANDOFF = 0.7                    # 물체에서 이만큼 떨어진 곳까지 Nav2 이동 (m)
VIEW_MAX_DIST = 1.2               # '잘 보였던 자리'가 물체에서 이보다 멀면 1순위로 안 씀 (m)
MAX_FAIL = 5                      # 같은 물체 실패 허용 횟수
RETRY_WAIT = 20.0                 # 실패 후 재시도까지 대기 (s)
IDLE_TIME = 15.0                  # 이 시간 동안 안 움직이면 정체로 판단 (s)
FRONTIER_MIN_SIZE = 10            # 이보다 작은 Frontier 묶음은 무시 (칸)
FRONTIER_MIN_DIST = 1.0           # 로봇 바로 옆 Frontier는 제외 (m)
NO_PROGRESS_TIME = 15.0           # Frontier로 가는 중 이 시간 동안 안 움직이면 포기 (s)
FRONTIER_TIMEOUT = 60.0           # Frontier 하나로 이동 제한 시간 (s)
WALL_MARGIN_CELLS = 3             # 벽에서 이 칸 수(3칸 = 15cm) 안이면 '벽에 붙은 칸'
WALL_FRONTIER_RATIO = 0.5         # Frontier 묶음 중 벽에 붙은 칸이 이 비율을 넘으면 제외
CAM_RANGE = 5.0                   # 카메라로 '봤다'고 기록할 최대 거리 (m)
CAM_UNKNOWN_MAX = 1.5             # 지도에 없는(회색) 영역으로는 이만큼만 '봤다'고 기록 (m)
CAM_RAYS = 81                     # 카메라 시야를 나눠 쏘는 광선 수
COVER_RATIO = 0.8                 # Frontier 너머를 이 비율 이상 카메라로 봤으면 제외
TRIED_R = 1.0                     # 이미 가본 Frontier 주변 제외 반경 (m)
START_IDLE = 6.0                  # 출발 후 한 번도 못 움직였으면 이 시간만 기다림 (s)
STUCK_RETRY = 3                   # Frontier가 남았는데 정체될 때 재시도 횟수
SEARCH_STEP = 1.2                 # 수색 지점 간격 (m)
SPIN_TIME = 2 * math.pi / MAX_W + 1.0   # 한 바퀴 도는 시간 (s)
BEYOND_DIST = 0.6                 # 문 너머로 들어갈 거리 (m)
BEYOND_TIMEOUT = 60.0             # 문 통과 제한 시간 (s)
UNSTUCK_DIST = 0.2                # 끼었을 때 후진 거리 (m)

TARGETS = {'key': 'yellow', 'door': 'red', 'exit': 'green'}   # 팀 미로 색 규칙
MODELS = {'key': 'red_key', 'door': 'red_door', 'exit': 'exit_marker'}   # Gazebo 모델 이름
COLORS = {'key': (255, 255, 0), 'exit': (0, 255, 0), 'door': (255, 0, 0)}
NAMES = {'key': '열쇠', 'door': '문', 'exit': '탈출구'}
HEIGHTS = {'key': 0.4, 'door': 1.0, 'exit': 0.6}   # 물체 실제 높이 (m), 카메라 거리 추정용


# 로봇 출발 위치 (Gazebo 좌표). SLAM 지도는 출발점을 (0, 0)으로 잡으므로 로그에만 더해서 표시
START_X, START_Y = -2.0, -2.5


def gz(x, y):
    return f'({x + START_X:.2f}, {y + START_Y:.2f})'


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


# 색 이름 → 해당 색 픽셀 마스크 (벽의 회청색은 어느 쪽에도 안 걸리게)
def color_mask(img, color):
    r = img[:, :, 0].astype(np.int16)
    g = img[:, :, 1].astype(np.int16)
    b = img[:, :, 2].astype(np.int16)
    if color == 'red':
        m = (r > 70) & (g < r * 0.4) & (b < r * 0.4)
    elif color == 'green':
        m = (g > 70) & (r < g * 0.4) & (b < g * 0.4)
    elif color == 'blue':
        m = (b > 70) & (r < b * 0.4) & (g < b * 0.4)
    else:  # yellow: 빨강·초록이 비슷하게 강하고 파랑이 약함
        m = ((r > 70) & (g > 70) & (abs(r - g) < 0.3 * np.maximum(r, g))
             & (b < 0.4 * np.minimum(r, g)))
    return m.astype(np.uint8)


def yaw_of(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y ** 2 + q.z ** 2))


class EscapeNode(Node):
    def __init__(self):
        super().__init__('escape_node')
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)

        # 입력
        self.create_subscription(Image, '/camera/image_raw', self.image_cb, 1)
        self.create_subscription(LaserScan, '/scan', self.scan_cb, qos_profile_sensor_data)
        self.create_subscription(OccupancyGrid, '/map', self.map_cb, latched)
        self.create_subscription(OccupancyGrid, '/global_costmap/costmap', self.cost_cb, latched)
        self.create_subscription(GoalStatusArray, '/navigate_to_pose/_action/status',
                                 self.nav_status_cb, latched)
        self.create_subscription(MarkerArray, '/explore/frontiers', self.frontiers_cb, 10)

        # 출력
        self.cmd_pub = self.create_publisher(TwistStamped, '/cmd_vel', 10)
        self.resume_pub = self.create_publisher(Bool, '/explore/resume', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/objects', 10)
        self.cov_pub = self.create_publisher(OccupancyGrid, '/camera_coverage', 1)
        self.debug_pub = self.create_publisher(Image, '/detect/image', 1)

        # Nav2, SLAM, TF
        self.nav = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.cancel_all_cli = self.create_client(CancelGoal, '/navigate_to_pose/_action/cancel_goal')
        self.pause_cli = (self.create_client(Pause, '/slam_toolbox/pause_new_measurements')
                          if Pause else None)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # 센서
        self.scan = None
        self.map = None
        self.costmap = None
        self.nav_status = None
        self.frontiers_left = None
        # 인식·기억
        self.seen = {}     # 이름: (화면 속 각도, 본 시각)
        self.obs = {}      # 이름: 최근 관측 위치들
        self.memory = {}   # 이름: (x, y) 기억한 위치
        self.view = {}     # 이름: (면적, x, y, yaw) 가장 잘 보였던 로봇 위치
        # 임무
        self.has_key = False
        self.door_open = False
        self.fails = {}
        self.retry_at = {}
        self.state = None
        self.state_start = 0.0
        self.target = None
        # 이동
        self.goals = []
        self.arrived = False
        self.had_goals = False
        self.nav_busy = False
        self.goal_handle = None
        self.seq = 0
        self.need_unstuck = False
        self.exploring = None
        self.slam_paused = False
        self.pause_since = 0.0
        self.handover_since = None
        self.idle_pos = None
        self.idle_since = 0.0
        self.stuck_retries = 0
        self.ever_moved = False
        self.frontier_clusters = []
        self.frontier_goal = None
        self.frontier_sent = False
        self.tried_frontiers = []
        self.cur_frontier = None
        self.no_frontier_count = 0
        self.explore_goal_start = 0.0
        self.prog_pos = None
        self.prog_t = 0.0
        self.last_plan = 0.0
        self.seen_cells = set()
        self.last_image_t = -100.0
        self.last_cov_pub = 0.0
        self.covered_clusters = []
        self.covered_count = 0
        self.search_pts = []
        self.searched = []
        self.search_moving = False
        self.spin_until = 0.0
        self.app_count = 0
        self.lost_since = None
        self.beyond = None
        self.unstuck_then = None
        self.unstuck_from = None
        self.last_log = 0.0
        self.create_timer(0.1, self.step)
        self.create_timer(0.5, self.update_coverage)

    # ================= 공통 =================
    def t(self):
        return self.get_clock().now().nanoseconds / 1e9

    def set_state(self, state, msg=''):
        self.state = state
        self.state_start = self.t()
        self.get_logger().info(f'[{state}] {msg}')

    def scan_cb(self, msg):
        self.scan = msg

    def map_cb(self, msg):
        self.map = msg

    def cost_cb(self, msg):
        self.costmap = msg

    def nav_status_cb(self, msg):
        self.nav_status = msg

    # explore_lite가 표시하는 Frontier 중 제외(빨강)되지 않은 것의 개수
    def frontiers_cb(self, msg):
        adds = [m for m in msg.markers if m.action == Marker.ADD
                and not (m.color.r > 0.5 and m.color.g < 0.5 and m.color.b < 0.5)]
        points = [m for m in adds if m.type == Marker.POINTS]
        self.frontiers_left = len(points) if points else len(adds)
        if points:
            self.frontier_clusters = [[(q.x, q.y) for q in m.points] for m in points if m.points]
        else:
            self.frontier_clusters = [[(m.pose.position.x, m.pose.position.y)] for m in adds]

    def robot_pose(self):
        try:
            tf = self.tf_buffer.lookup_transform('map', 'base_footprint', Time())
        except Exception:
            return None
        return tf.transform.translation.x, tf.transform.translation.y, yaw_of(tf.transform.rotation)

    def range_at(self, angle):
        s = self.scan
        if s is None:
            return None
        n = len(s.ranges)
        i = int(round((angle - s.angle_min) / s.angle_increment))
        vals = [s.ranges[(i + k) % n] for k in range(-2, 3)]
        vals = [r for r in vals if s.range_min < r < s.range_max and math.isfinite(r)]
        return float(np.median(vals)) if vals else None

    # 앞(또는 뒤) ±20도 안에 limit보다 가까운 장애물이 없는지
    def clear(self, front, limit):
        s = self.scan
        if s is None:
            return True
        for i, r in enumerate(s.ranges):
            a = s.angle_min + i * s.angle_increment
            a = math.atan2(math.sin(a), math.cos(a))
            ahead = abs(a) < 0.35 if front else abs(a) > math.pi - 0.35
            if ahead and s.range_min < r < limit:
                return False
        return True

    def lidar_to_map(self, r, angle):
        try:
            tf = self.tf_buffer.lookup_transform('map', self.scan.header.frame_id, Time())
        except Exception:
            return None
        yaw = yaw_of(tf.transform.rotation)
        return (tf.transform.translation.x + r * math.cos(yaw + angle),
                tf.transform.translation.y + r * math.sin(yaw + angle))

    def is_free(self, x, y):
        cm = self.costmap
        if cm is None:
            return True
        cx = int((x - cm.info.origin.position.x) / cm.info.resolution)
        cy = int((y - cm.info.origin.position.y) / cm.info.resolution)
        if not (0 <= cx < cm.info.width and 0 <= cy < cm.info.height):
            return False
        v = cm.data[cy * cm.info.width + cx]
        return 0 <= v < 50

    def send_vel(self, v, w):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.twist.linear.x = float(v)
        msg.twist.angular.z = float(w)
        self.cmd_pub.publish(msg)

    def stop(self):
        self.send_vel(0.0, 0.0)

    def set_explore(self, on):
        if self.exploring != on:
            self.resume_pub.publish(Bool(data=on))
            self.exploring = on
            self.get_logger().info('탐색 ' + ('재개' if on else '중단'))

    # 접근·문 통과 중에는 회전·흔들림·문 소실로 SLAM이 틀어지므로 지도 갱신 일시정지
    def set_slam_pause(self, on):
        if self.slam_paused == on or self.pause_cli is None:
            return
        if not self.pause_cli.service_is_ready():
            self.get_logger().warn('SLAM 일시정지 서비스 없음')
            return
        self.pause_cli.call_async(Pause.Request())   # 호출할 때마다 정지/재개가 바뀜
        self.slam_paused = on
        self.pause_since = self.t()
        self.get_logger().info('SLAM 지도 갱신 ' + ('일시정지' if on else '재개'))

    # Nav2(우리 목표든 explore_lite 목표든)가 아직 로봇을 움직이는 중인지
    def nav2_active(self):
        if self.nav_status is None:
            return False
        return any(st.status in (GoalStatus.STATUS_ACCEPTED, GoalStatus.STATUS_EXECUTING,
                                 GoalStatus.STATUS_CANCELING)
                   for st in self.nav_status.status_list)

    # 직접 속도 제어 전에 Nav2가 완전히 멈췄는지 확인. 안 멈췄으면 전체 취소 요청 후 대기
    def ready_for_direct(self):
        if not self.nav2_active():
            if self.handover_since is not None:
                self.get_logger().info('Nav2 정지 확인, 직접 제어 시작')
            self.handover_since = None
            return True
        if self.handover_since is None or self.t() - self.handover_since > 2.0:
            if self.handover_since is None:
                self.get_logger().info('Nav2 정지 대기 중')
            self.handover_since = self.t()
            if self.cancel_all_cli.service_is_ready():
                self.cancel_all_cli.call_async(CancelGoal.Request())   # 빈 요청 = 모든 목표 취소
        return False

    # ================= 인식 + 위치 기억 =================
    def image_cb(self, msg):
        img = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)
        img = img[:, :msg.width * 3].reshape(msg.height, msg.width, 3)
        if msg.encoding == 'bgr8':
            img = img[:, :, ::-1]
        h, w = img.shape[:2]
        f = (w / 2) / math.tan(FOV / 2)
        debug = img.copy() if self.debug_pub.get_subscription_count() > 0 else None
        self.last_image_t = self.t()
        now = self.t()

        for name, color in TARGETS.items():
            mask = color_mask(img, color)
            area = int(mask.sum())
            if area < MIN_AREA:
                continue
            x, y, bw, bh = cv2.boundingRect(mask)
            angle = math.atan2(w / 2 - (x + bw / 2), f)   # 왼쪽 +, 오른쪽 -
            # 위아래가 화면에 다 들어오면 크기로 거리 추정 (LiDAR 3.5m 밖이어도 가능)
            cam_d = f * HEIGHTS[name] / bh if (y > 2 and y + bh < h - 2 and bh > 0) else None
            self.seen[name] = (angle, now, cam_d)
            if debug is not None:
                cv2.rectangle(debug, (x, y), (x + bw, y + bh), COLORS[name], 2)
            # 화면 가장자리에 잘렸거나 옆으로 많이 벗어나면 위치 계산 제외
            if x <= 2 or x + bw >= w - 2 or abs(angle) > POS_MAX_ANGLE:
                continue
            r = self.range_at(angle)
            if r is not None and cam_d is not None and abs(r - cam_d) > 0.4 * cam_d:
                r = None
            p = self.lidar_to_map(r, angle) if r is not None else None
            if p:
                self.remember(name, *p)
                pose = self.robot_pose()
                if pose and area > self.view.get(name, (0,))[0]:
                    self.view[name] = (area, *pose)

        if debug is not None:
            out = Image()
            out.header = msg.header
            out.height, out.width = h, w
            out.encoding, out.step = 'rgb8', w * 3
            out.data = np.ascontiguousarray(debug).tobytes()
            self.debug_pub.publish(out)
        self.publish_markers()

    # 최근 50번 관측의 중앙값으로 기억 (가끔 튀는 값에 흔들리지 않게)
    def remember(self, name, x, y):
        obs = self.obs.setdefault(name, [])
        if not obs:
            self.get_logger().info(f'새로 발견: {NAMES[name]} {gz(x, y)}')
        obs.append((x, y))
        del obs[:-50]
        xs = sorted(p[0] for p in obs)
        ys = sorted(p[1] for p in obs)
        self.memory[name] = (xs[len(xs) // 2], ys[len(ys) // 2])

    def publish_markers(self):
        arr = MarkerArray()
        for name, (x, y) in self.memory.items():
            m = Marker()
            m.header.frame_id = 'map'
            m.ns, m.id, m.type = name, 0, Marker.SPHERE
            m.pose.position.x, m.pose.position.y, m.pose.position.z = x, y, 0.1
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 0.25
            r, g, b = (c / 255 for c in COLORS[name])
            m.color.r, m.color.g, m.color.b, m.color.a = r, g, b, 1.0
            arr.markers.append(m)
        self.marker_pub.publish(arr)

    # ================= 판단 =================
    def next_target(self):
        def ok(n):
            return (n in self.memory and self.fails.get(n, 0) < MAX_FAIL
                    and self.t() >= self.retry_at.get(n, 0.0))
        if not self.has_key and ok('key'):
            return 'key'
        if self.has_key and not self.door_open and ok('door'):
            return 'door'
        if self.door_open and ok('exit'):   # 레벨 1: 반드시 문을 연 뒤에만 탈출구로
            return 'exit'
        return None

    def step(self):
        if self.get_clock().now().nanoseconds == 0:   # 시뮬레이션 시간 대기
            return
        if self.state is None:
            self.enter_explore('탐색 시작')
            return
        if self.slam_paused and self.t() - self.pause_since > 20.0:
            self.get_logger().warn('SLAM 일시정지가 길어져 강제 재개')
            self.set_slam_pause(False)
        if self.state in ('EXPLORE', 'SEARCH', 'SPIN', 'BEYOND', 'FRONTIER'):
            t = self.next_target()
            if t:
                self.begin_goto(t)
                return
        getattr(self, 'do_' + self.state.lower())()

    # ================= EXPLORE (Frontier 탐색) =================
    def enter_explore(self, msg='탐색으로 복귀'):
        self.target = None
        self.idle_pos = None
        self.set_state('EXPLORE', msg)

    def do_explore(self):
        self.set_explore(False)   # explore_lite 대신 이 노드가 직접 Frontier로 보냄
        if self.nav_busy:
            pose = self.robot_pose()
            if pose and (self.prog_pos is None
                         or math.hypot(pose[0] - self.prog_pos[0], pose[1] - self.prog_pos[1]) > 0.1
                         or abs(math.atan2(math.sin(pose[2] - self.prog_pos[2]),
                                           math.cos(pose[2] - self.prog_pos[2]))) > 0.3):
                self.prog_pos = pose   # 10cm 이동 또는 약 17도 회전하면 진전 있음
                self.prog_t = self.t()
            if self.t() - self.prog_t > NO_PROGRESS_TIME:
                self.get_logger().warn(f'{NO_PROGRESS_TIME:.0f}초 동안 진전 없음, 이 Frontier 제외')
                self.cancel_nav()
                self.stop()
            elif self.t() - self.explore_goal_start > FRONTIER_TIMEOUT:
                self.get_logger().warn('Frontier 이동 시간 초과, 이 Frontier 제외')
                self.cancel_nav()
                self.stop()
            return
        if self.cur_frontier is not None:   # 직전 Frontier 이동이 끝남 (도착/실패 모두 다시 안 감)
            self.tried_frontiers.append(self.cur_frontier)
            self.cur_frontier = None
            if self.need_unstuck:
                self.need_unstuck = False
                self.begin_unstuck(lambda: self.enter_explore('탐색 재시도'))
                return
        self.need_unstuck = False
        if self.map is None or self.t() - self.last_plan < 1.0:
            return
        self.last_plan = self.t()
        pose = self.robot_pose()
        if pose is None:
            return
        clusters = self.find_frontiers()
        self.frontiers_left = len(clusters)
        goal = self.pick_frontier(clusters, pose)
        if goal is None and self.covered_clusters:
            # 카메라로 안 본 Frontier가 없으면, 카메라로 일부 본 Frontier라도 직접 가서 확인
            goal = self.pick_frontier(self.covered_clusters, pose)
        if goal is None:
            # Frontier가 없거나, 남은 것이 모두 도달 불가로 확인됨
            self.no_frontier_count += 1
            if self.no_frontier_count >= 3:
                self.no_frontier_count = 0
                if clusters:
                    self.get_logger().info(f'Frontier {len(clusters)}개가 남았지만 도달 불가 → 수색 모드')
                else:
                    self.get_logger().info(
                        f'남은 Frontier 없음 (카메라로 확인해 제외 {self.covered_count}개) → 수색 모드')
                self.tried_frontiers = []
                self.begin_search()
            return
        self.no_frontier_count = 0
        gx, gy = goal
        self.cur_frontier = goal
        self.explore_goal_start = self.t()
        self.prog_pos = None
        self.prog_t = self.t()
        self.arrived = False
        mem = ', '.join(NAMES[k] for k in self.memory) or '없음'
        self.get_logger().info(
            f'[EXPLORE] Frontier {gz(gx, gy)}로 이동 | 남은 Frontier {len(clusters)}개 (카메라 확인 제외 {self.covered_count}개) '
            f'| 열쇠 {"있음" if self.has_key else "없음"} | 기억: {mem}')
        self.send_goal(gx, gy, math.atan2(gy - pose[1], gx - pose[0]))

    # ================= 카메라 Coverage (카메라로 이미 본 곳 기록) =================
    # 지도는 LiDAR(3.5m)로만 그려지지만 카메라는 더 멀리 본다.
    # 카메라 시야 안에 들어온 칸을 기록해 두고, 너머를 이미 카메라로 본 Frontier는 건너뛴다.
    def cell_key(self, x, y):
        r = self.map.info.resolution
        return (int(math.floor(x / r)), int(math.floor(y / r)))

    def is_seen(self, x, y):
        ix, iy = self.cell_key(x, y)
        return any((ix + dx, iy + dy) in self.seen_cells for dx in (-1, 0, 1) for dy in (-1, 0, 1))

    def update_coverage(self):
        if self.map is None or self.t() - self.last_image_t > 1.0:
            return
        pose = self.robot_pose()
        if pose is None:
            return
        m = self.map
        res, w, h = m.info.resolution, m.info.width, m.info.height
        ox, oy = m.info.origin.position.x, m.info.origin.position.y
        data = m.data
        x0, y0, yaw = pose
        for k in range(CAM_RAYS):
            a = yaw - FOV / 2 + FOV * k / (CAM_RAYS - 1)
            ca, sa = math.cos(a), math.sin(a)
            dist = 0.0
            unk = 0.0
            while dist < CAM_RANGE:
                x, y = x0 + dist * ca, y0 + dist * sa
                cx, cy = int((x - ox) / res), int((y - oy) / res)
                if 0 <= cx < w and 0 <= cy < h and data[cy * w + cx] >= 50:
                    break   # 벽에 막히면 그 뒤는 못 봄
                if not (0 <= cx < w and 0 <= cy < h) or data[cy * w + cx] < 0:
                    unk += res   # 지도에 없는 곳: 안 보이는 벽이 있을 수 있음
                    if unk > CAM_UNKNOWN_MAX:
                        break
                self.seen_cells.add(self.cell_key(x, y))
                dist += res
        if self.t() - self.last_cov_pub > 1.0:
            self.last_cov_pub = self.t()
            self.publish_coverage()

    # RViz 확인용: /camera_coverage (카메라로 본 칸 = 100)
    def publish_coverage(self):
        m = self.map
        res, w, h = m.info.resolution, m.info.width, m.info.height
        bx = int(math.floor(m.info.origin.position.x / res))
        by = int(math.floor(m.info.origin.position.y / res))
        grid = np.zeros(h * w, np.int8)
        for ix, iy in self.seen_cells:
            cx, cy = ix - bx, iy - by
            if 0 <= cx < w and 0 <= cy < h:
                grid[cy * w + cx] = 100
        msg = OccupancyGrid()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.info = m.info
        msg.data = grid.tolist()
        self.cov_pub.publish(msg)

    # Frontier 칸에서 미탐색 쪽으로 0.25/0.75/1.25m 지점을 카메라가 모두 봤으면 '확인한 칸'
    # 묶음의 COVER_RATIO 이상이 확인한 칸이면 그 Frontier는 갈 필요 없음
    def cluster_covered(self, cells, unknown, res, o):
        h, w = unknown.shape
        step = max(1, len(cells) // 30)
        checked = covered = 0
        for y, x in cells[::step]:
            ux = (1 if x + 1 < w and unknown[y, x + 1] else 0) - (1 if x - 1 >= 0 and unknown[y, x - 1] else 0)
            uy = (1 if y + 1 < h and unknown[y + 1, x] else 0) - (1 if y - 1 >= 0 and unknown[y - 1, x] else 0)
            n = math.hypot(ux, uy)
            if n == 0:
                continue
            ux, uy = ux / n, uy / n
            px, py = o.x + (x + 0.5) * res, o.y + (y + 0.5) * res
            checked += 1
            if all(self.is_seen(px + ux * d, py + uy * d) for d in (0.25, 0.75, 1.25)):
                covered += 1
        return checked > 0 and covered >= COVER_RATIO * checked

    # 빈 칸 중 바로 옆(상하좌우)에 미탐색 칸이 있는 칸 = Frontier, 붙은 것끼리 묶음
    def find_frontiers(self):
        m = self.map
        h, w = m.info.height, m.info.width
        grid = np.array(m.data, dtype=np.int8).reshape(h, w)
        free = (grid >= 0) & (grid < 50)
        unknown = grid == -1
        near = np.zeros_like(unknown)
        near[1:, :] |= unknown[:-1, :]
        near[:-1, :] |= unknown[1:, :]
        near[:, 1:] |= unknown[:, :-1]
        near[:, :-1] |= unknown[:, 1:]
        frontier = free & near
        self.covered_clusters = []
        self.covered_count = 0
        # 벽(및 닫힌 문)에서 WALL_MARGIN_CELLS칸 안쪽 영역
        near_wall = grid >= 50
        for _ in range(WALL_MARGIN_CELLS):
            g = near_wall.copy()
            g[1:, :] |= near_wall[:-1, :]
            g[:-1, :] |= near_wall[1:, :]
            g[:, 1:] |= near_wall[:, :-1]
            g[:, :-1] |= near_wall[:, 1:]
            near_wall = g
        visited = np.zeros_like(frontier)
        res, o = m.info.resolution, m.info.origin.position
        clusters = []
        for r, c in zip(*np.nonzero(frontier)):
            if visited[r, c]:
                continue
            cells, q = [], deque([(r, c)])
            visited[r, c] = True
            while q:
                y, x = q.popleft()
                cells.append((y, x))
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w and frontier[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            q.append((ny, nx))
            if len(cells) < FRONTIER_MIN_SIZE:
                continue
            # 대부분 벽에 붙어 있는 묶음 = 벽 옆 작은 틈 (문 앞 서성거림 원인) → 제외
            if sum(1 for y, x in cells if near_wall[y, x]) > WALL_FRONTIER_RATIO * len(cells):
                continue
            if self.cluster_covered(cells, unknown, res, o):
                self.covered_count += 1   # 너머를 이미 카메라로 봄 → 제외하지 않고 나중 순위로
                self.covered_clusters.append([(o.x + (x + 0.5) * res, o.y + (y + 0.5) * res) for y, x in cells])
                continue
            clusters.append([(o.x + (x + 0.5) * res, o.y + (y + 0.5) * res) for y, x in cells])
        return clusters

    # 묶음마다 중심에 가까운 갈 수 있는 칸을 고르고, 가깝고 큰 묶음 우선
    def pick_frontier(self, clusters, pose):
        rx, ry = pose[0], pose[1]
        best, best_cost = None, float('inf')
        for pts in clusters:
            cx = sum(q[0] for q in pts) / len(pts)
            cy = sum(q[1] for q in pts) / len(pts)
            for x, y in sorted(pts, key=lambda q: (q[0] - cx) ** 2 + (q[1] - cy) ** 2):
                d = math.hypot(x - rx, y - ry)
                if d < FRONTIER_MIN_DIST or not self.is_free(x, y):
                    continue
                if any(math.hypot(x - tx, y - ty) < TRIED_R for tx, ty in self.tried_frontiers):
                    continue
                cost = d - 0.05 * len(pts)
                if cost < best_cost:
                    best, best_cost = (x, y), cost
                break
        return best

    # ================= GOTO (Nav2로 물체 근처까지) =================
    def candidates(self, name):
        ox, oy = self.memory[name]
        pose = self.robot_pose()
        rx, ry = (pose[0], pose[1]) if pose else (ox, oy)
        pts = []
        for k in range(12):
            a = 2 * math.pi * k / 12
            px, py = ox + STANDOFF * math.cos(a), oy + STANDOFF * math.sin(a)
            if self.is_free(px, py):
                pts.append((math.hypot(px - rx, py - ry), px, py, math.atan2(oy - py, ox - px)))
        pts.sort()
        return [(px, py, yaw) for _, px, py, yaw in pts]

    def begin_goto(self, t):
        self.cancel_nav()
        self.stop()
        self.set_slam_pause(False)
        self.set_explore(False)
        self.need_unstuck = False
        self.target = t
        self.goals = self.candidates(t)
        if t in self.view:   # 카메라에 가장 잘 보였던 위치: 물체에서 가까울 때만 1순위
            _, vx, vy, vyaw = self.view[t]
            ox, oy = self.memory[t]
            if math.hypot(vx - ox, vy - oy) <= VIEW_MAX_DIST:
                self.goals.insert(0, (vx, vy, vyaw))
            else:
                self.goals.append((vx, vy, vyaw))   # 멀리서 본 자리는 마지막 후보
        self.arrived = False
        self.had_goals = bool(self.goals)
        self.set_state('GOTO', f'{NAMES[t]} 근처로 이동 (후보 {len(self.goals)}곳)')

    def do_goto(self):
        if self.nav_busy:
            return
        if self.need_unstuck:
            self.need_unstuck = False
            self.begin_unstuck(lambda: self.resume('GOTO'))
            return
        if not self.goals:
            if self.arrived or not self.had_goals:
                self.begin_approach()
            else:
                self.fail('근처로 이동 못함')
            return
        self.send_goal(*self.goals.pop(0))

    # ================= APPROACH (카메라 + LiDAR 직접 접근) =================
    def begin_approach(self):
        self.app_count = 0
        self.lost_since = None
        self.set_state('APPROACH', f'{NAMES[self.target]} 접근')

    def do_approach(self):
        if not self.ready_for_direct():
            return
        if self.t() - self.state_start > APPROACH_LIMIT:
            self.fail('접근 시간 초과')
            return
        seen = self.seen.get(self.target)
        visible = seen is not None and self.t() - seen[1] < 0.5
        if not visible:
            self.app_count = 0
            if self.lost_since is None:
                self.lost_since = self.t()
            if self.t() - self.lost_since > APPROACH_TIMEOUT:
                self.fail('찾지 못함')
                return
            self.send_vel(0.0, 0.25)   # 제자리에서 천천히 찾기
            return
        self.lost_since = None
        angle, cam_d = seen[0], seen[2]
        d = self.range_at(angle)
        if d is not None and cam_d is not None and abs(d - cam_d) > 0.4 * cam_d:
            d = None
        drive_d = d if d is not None else cam_d
        if d is not None and d < APPROACH_DIST:
            self.stop()
            self.app_count += 1
            if self.app_count >= APPROACH_COUNT:
                self.on_reached(self.target)
            return
        self.app_count = 0
        # 작은 오차는 무시하고 약하게 회전해서 두리번거림 방지
        w = 0.0 if abs(angle) < math.radians(4) else clamp(0.8 * angle, -MAX_W, MAX_W)
        v = 0.0
        if abs(angle) < math.radians(10) and self.clear(True, FRONT_STOP):
            v = 0.08 if drive_d is None else clamp(0.5 * (drive_d - APPROACH_DIST), 0.03, 0.12)
        if self.t() - self.last_log > 3.0:
            self.last_log = self.t()
            self.get_logger().info(
                f'[APPROACH] 각도 {math.degrees(angle):+.0f}° | 거리 {-1 if d is None else d:.2f}m '
                f'| 앞 막힘 {"예" if not self.clear(True, FRONT_STOP) else "아니오"}')
        self.send_vel(v, w)

    def remove_model(self, name):
        try:
            r = subprocess.run(
                ['gz', 'service', '-s', f'/world/{WORLD}/remove',
                 '--reqtype', 'gz.msgs.Entity', '--reptype', 'gz.msgs.Boolean',
                 '--timeout', '3000', '--req', f'name: "{MODELS[name]}" type: MODEL'],
                capture_output=True, text=True, timeout=5)
            return 'true' in r.stdout
        except Exception:
            return False

    def on_reached(self, t):
        self.stop()
        if t == 'exit':
            self.set_slam_pause(False)
            self.set_explore(False)
            self.set_state('DONE', '탈출 성공!')
            return
        if t == 'door':
            self.set_slam_pause(True)   # 문이 사라지는 순간부터 통과까지만 정지
        if not self.remove_model(t):
            self.fail('Gazebo 제거 실패')
            return
        if t == 'key':
            self.set_slam_pause(False)
            self.has_key = True
            self.get_logger().info('열쇠 획득')
            self.enter_explore()
            return
        # 문: 지도에는 아직 벽으로 남아 있으므로, 문 정면 방향으로 직접 직진해서 통과
        self.door_open = True
        self.get_logger().info('문 개방')
        x, y, yaw = self.robot_pose() or (0.0, 0.0, 0.0)
        go = APPROACH_DIST + BEYOND_DIST
        self.beyond = (x + go * math.cos(yaw), y + go * math.sin(yaw))
        self.set_state('BEYOND', f'문 너머 {gz(*self.beyond)}로 직진')

    def fail(self, reason):
        t = self.target
        self.stop()
        self.set_slam_pause(False)
        self.fails[t] = self.fails.get(t, 0) + 1
        self.retry_at[t] = self.t() + RETRY_WAIT
        self.get_logger().warn(f'{NAMES[t]} {reason} ({self.fails[t]}/{MAX_FAIL})')
        self.enter_explore()

    # ================= BEYOND (문 통과) =================
    def do_beyond(self):
        if not self.ready_for_direct():
            return
        pose = self.robot_pose()
        if pose is None:
            return
        x, y, yaw = pose
        bx, by = self.beyond
        dist = math.hypot(bx - x, by - y)
        if dist < 0.15 or self.t() - self.state_start > BEYOND_TIMEOUT:
            self.stop()
            self.set_slam_pause(False)
            self.enter_explore('문 너머 탐색' if dist < 0.15 else '문 통과 시간 초과, 탐색 재개')
            return
        want = math.atan2(by - y, bx - x)
        err = math.atan2(math.sin(want - yaw), math.cos(want - yaw))
        v = 0.1 if abs(err) < 0.3 and self.clear(True, FRONT_STOP) else 0.0
        self.send_vel(v, clamp(1.0 * err, -MAX_W, MAX_W))

    # ================= UNSTUCK (끼었을 때 뒤로 빠져나오기) =================
    def begin_unstuck(self, then):
        self.set_explore(False)
        self.unstuck_then = then
        pose = self.robot_pose()
        self.unstuck_from = pose[:2] if pose else None
        self.set_state('UNSTUCK', '뒤로 빠져나오기')

    def do_unstuck(self):
        if not self.ready_for_direct():
            return
        pose = self.robot_pose()
        moved = 0.0
        if pose and self.unstuck_from:
            moved = math.hypot(pose[0] - self.unstuck_from[0], pose[1] - self.unstuck_from[1])
        if (moved > UNSTUCK_DIST or self.t() - self.state_start > 8.0
                or not self.clear(False, REAR_STOP)):
            self.stop()
            self.unstuck_then()
            return
        self.send_vel(-0.05, 0.0)

    def resume(self, state):
        self.set_state(state, '이어서 진행')

    # ================= SEARCH (Frontier 소진 후 수색) =================
    def make_search_points(self):
        m = self.map
        if m is None:
            return []
        res, w, h = m.info.resolution, m.info.width, m.info.height
        o = m.info.origin.position
        step = max(1, int(SEARCH_STEP / res))
        pts = []
        for cy in range(step // 2, h, step):
            for cx in range(step // 2, w, step):
                if not (0 <= m.data[cy * w + cx] < 50):
                    continue
                x, y = o.x + (cx + 0.5) * res, o.y + (cy + 0.5) * res
                if not self.is_free(x, y):
                    continue
                if any(math.hypot(x - sx, y - sy) < SEARCH_STEP * 0.8 for sx, sy in self.searched):
                    continue
                pts.append((x, y))
        return pts

    def begin_search(self):
        self.set_explore(False)
        self.need_unstuck = False
        self.search_pts = self.make_search_points()
        if not self.search_pts:
            self.searched = []
            self.search_pts = self.make_search_points()
        self.search_moving = False
        self.set_state('SEARCH', f'수색 시작 (지점 {len(self.search_pts)}곳)')

    def do_search(self):
        if self.nav_busy:
            return
        if self.need_unstuck:
            self.need_unstuck = False
            self.begin_unstuck(lambda: self.resume('SEARCH'))
            return
        if self.search_moving:
            self.search_moving = False
            if self.arrived:
                self.spin_until = self.t() + SPIN_TIME
                self.set_state('SPIN', '제자리 회전하며 둘러보기')
                return
        if not self.search_pts:
            self.searched = []
            self.enter_explore('수색 지점을 다 돌아서 탐색 재개')
            return
        pose = self.robot_pose()
        rx, ry = (pose[0], pose[1]) if pose else (0.0, 0.0)
        self.search_pts.sort(key=lambda p: math.hypot(p[0] - rx, p[1] - ry))
        x, y = self.search_pts.pop(0)
        self.searched.append((x, y))
        self.arrived = False
        self.search_moving = True
        self.get_logger().info(f'[SEARCH] {gz(x, y)}로 이동, 남은 지점 {len(self.search_pts)}곳')
        self.send_goal(x, y, 0.0)

    def do_spin(self):
        if not self.ready_for_direct():
            return
        if self.t() < self.spin_until:
            self.send_vel(0.0, MAX_W)
        else:
            self.stop()
            self.state = 'SEARCH'

    def do_done(self):
        self.stop()

    # ================= Nav2 =================
    def send_goal(self, x, y, yaw):
        if not self.nav.wait_for_server(timeout_sec=1.0):
            return
        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = 'map'
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = x
        goal.pose.pose.position.y = y
        goal.pose.pose.orientation.z = math.sin(yaw / 2)
        goal.pose.pose.orientation.w = math.cos(yaw / 2)
        self.seq += 1
        seq = self.seq
        self.nav_busy = True
        self.goal_handle = None
        self.nav.send_goal_async(goal).add_done_callback(lambda f: self.goal_response(f, seq))

    def goal_response(self, fut, seq):
        h = fut.result()
        if seq != self.seq:
            return
        if not h.accepted:
            self.nav_busy = False
            return
        self.goal_handle = h
        h.get_result_async().add_done_callback(lambda f: self.nav_result(f, seq))

    def nav_result(self, fut, seq):
        if seq != self.seq:
            return
        if fut.result().status == GoalStatus.STATUS_SUCCEEDED:
            self.arrived = True
            if self.state == 'GOTO':
                self.goals = []
        else:
            code = getattr(fut.result().result, 'error_code', '?')
            self.get_logger().warn(f'이동 실패 (코드 {code})')
            self.need_unstuck = True
        self.nav_busy = False

    def cancel_nav(self):
        if self.nav_busy and self.goal_handle:
            self.goal_handle.cancel_goal_async()
        self.seq += 1
        self.nav_busy = False


def main():
    rclpy.init()
    rclpy.spin(EscapeNode())


if __name__ == '__main__':
    main()
