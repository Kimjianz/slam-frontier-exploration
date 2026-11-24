import math
import subprocess

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
from action_msgs.msg import GoalStatus
from tf2_ros import Buffer, TransformListener
try:
    from slam_toolbox.srv import Pause
except ImportError:
    Pause = None

# ================= 설정값 =================
WORLD = 'default'                 # Gazebo 월드 이름
FOV = 1.4                         # 카메라 시야각 (rad)
MIN_AREA = 50                     # 이보다 작은 색 영역은 무시 (픽셀)
POS_MAX_ANGLE = math.radians(25)  # 화면 중앙 ±25도 안에서만 위치 계산
MAX_W = 0.3                       # 모든 회전 최대 속도 (rad/s), SLAM 안정용
FRONT_STOP = 0.18                 # 앞 장애물이 이보다 가까우면 전진 정지 (m)
APPROACH_DIST = 0.5               # 획득 판정 거리 (m)
APPROACH_COUNT = 5                # 연속 확인 횟수
APPROACH_TIMEOUT = 40.0           # 목표를 이 시간 동안 못 보면 실패 (s)
STANDOFF = 0.7                    # 물체에서 이만큼 떨어진 곳까지 Nav2 이동 (m)
MAX_FAIL = 5                      # 같은 물체 실패 허용 횟수
RETRY_WAIT = 20.0                 # 실패 후 재시도까지 대기 (s)
IDLE_TIME = 15.0                  # 이 시간 동안 안 움직이면 탐색 끝으로 판단 (s)
SEARCH_STEP = 1.2                 # 수색 지점 간격 (m)
SPIN_TIME = 2 * math.pi / MAX_W + 1.0   # 한 바퀴 도는 시간 (s)
BEYOND_DIST = 0.6                 # 문 너머로 들어갈 거리 (m)
BEYOND_TIMEOUT = 60.0             # 문 통과 제한 시간 (s)

TARGETS = {'key': 0, 'exit': 1, 'door': 2}   # 찾을 색 채널 (0=R, 1=G, 2=B)
COLORS = {'key': (255, 0, 0), 'exit': (0, 255, 0), 'door': (0, 0, 255)}
NAMES = {'key': '열쇠', 'door': '문', 'exit': '탈출구'}


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


class EscapeNode(Node):
    def __init__(self):
        super().__init__('escape_node')
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(Image, '/camera/image_raw', self.image_cb, 1)
        self.create_subscription(LaserScan, '/scan', self.scan_cb, qos_profile_sensor_data)
        self.create_subscription(OccupancyGrid, '/map', self.map_cb, latched)
        self.create_subscription(OccupancyGrid, '/global_costmap/costmap', self.cost_cb, latched)
        self.cmd_pub = self.create_publisher(TwistStamped, '/cmd_vel', 10)
        self.resume_pub = self.create_publisher(Bool, '/explore/resume', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/objects', 10)
        self.debug_pub = self.create_publisher(Image, '/detect/image', 1)
        self.nav = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.pause_cli = (self.create_client(Pause, '/slam_toolbox/pause_new_measurements')
                          if Pause else None)
        self.slam_paused = False
        self.need_unstuck = False
        self.unstuck_then = None
        self.unstuck_from = None

        # 센서
        self.scan = None
        self.map = None
        self.costmap = None
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
        self.exploring = None
        self.idle_pos = None
        self.idle_since = 0.0
        self.search_pts = []
        self.searched = []
        self.search_moving = False
        self.spin_until = 0.0
        self.app_count = 0
        self.lost_since = None
        self.beyond = None
        self.last_log = 0.0
        self.create_timer(0.1, self.step)

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

    def robot_pose(self):
        try:
            tf = self.tf_buffer.lookup_transform('map', 'base_footprint', Time())
        except Exception:
            return None
        q = tf.transform.rotation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y ** 2 + q.z ** 2))
        return tf.transform.translation.x, tf.transform.translation.y, yaw

    def range_at(self, angle):
        s = self.scan
        if s is None:
            return None
        n = len(s.ranges)
        i = int(round((angle - s.angle_min) / s.angle_increment))
        vals = [s.ranges[(i + k) % n] for k in range(-2, 3)]
        vals = [r for r in vals if s.range_min < r < s.range_max and math.isfinite(r)]
        return float(np.median(vals)) if vals else None

    def front_clear(self):
        s = self.scan
        if s is None:
            return True
        for i, r in enumerate(s.ranges):
            a = s.angle_min + i * s.angle_increment
            a = math.atan2(math.sin(a), math.cos(a))
            if abs(a) < 0.35 and s.range_min < r < FRONT_STOP:
                return False
        return True

    def rear_clear(self):
        s = self.scan
        if s is None:
            return True
        for i, r in enumerate(s.ranges):
            a = s.angle_min + i * s.angle_increment
            a = math.atan2(math.sin(a), math.cos(a))
            if abs(a) > math.pi - 0.5 and s.range_min < r < 0.18:
                return False
        return True

    def lidar_to_map(self, r, angle):
        try:
            tf = self.tf_buffer.lookup_transform('map', self.scan.header.frame_id, Time())
        except Exception:
            return None
        q = tf.transform.rotation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y ** 2 + q.z ** 2))
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

    # 문이 사라지는 순간 스캔이 크게 바뀌어 SLAM이 틀어지므로, 문 통과 동안 지도 갱신 일시정지
    def set_slam_pause(self, on):
        if self.slam_paused == on or self.pause_cli is None:
            return
        if not self.pause_cli.service_is_ready():
            self.get_logger().warn('SLAM 일시정지 서비스 없음')
            return
        self.pause_cli.call_async(Pause.Request())   # 호출할 때마다 정지/재개가 바뀜
        self.slam_paused = on
        self.get_logger().info('SLAM 지도 갱신 ' + ('일시정지' if on else '재개'))

    # ================= 인식 + 위치 기억 =================
    def image_cb(self, msg):
        img = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)
        img = img[:, :msg.width * 3].reshape(msg.height, msg.width, 3)
        if msg.encoding == 'bgr8':
            img = img[:, :, ::-1]
        h, w = img.shape[:2]
        f = (w / 2) / math.tan(FOV / 2)
        debug = img.copy() if self.debug_pub.get_subscription_count() > 0 else None
        now = self.t()

        for name, ch in TARGETS.items():
            c = img[:, :, ch].astype(np.int16)
            o1 = img[:, :, (ch + 1) % 3].astype(np.int16)
            o2 = img[:, :, (ch + 2) % 3].astype(np.int16)
            mask = ((c > 70) & (o1 < c * 0.4) & (o2 < c * 0.4)).astype(np.uint8)
            if int(mask.sum()) < MIN_AREA:
                continue
            x, y, bw, bh = cv2.boundingRect(mask)
            angle = math.atan2(w / 2 - (x + bw / 2), f)   # 왼쪽 +, 오른쪽 -
            self.seen[name] = (angle, now)
            if debug is not None:
                cv2.rectangle(debug, (x, y), (x + bw, y + bh), COLORS[name], 2)
            # 화면 가장자리에 잘렸거나 옆으로 많이 벗어나면 위치 계산 제외
            if x <= 2 or x + bw >= w - 2 or abs(angle) > POS_MAX_ANGLE:
                continue
            r = self.range_at(angle)
            p = self.lidar_to_map(r, angle) if r is not None else None
            if p:
                self.remember(name, *p)
                area = int(mask.sum())
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

    def remember(self, name, x, y):
        obs = self.obs.setdefault(name, [])
        if not obs:
            self.get_logger().info(f'새로 발견: {NAMES[name]} ({x:.2f}, {y:.2f})')
        obs.append((x, y))
        del obs[:-50]   # 최근 50개만, 중앙값으로 기억
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
        if ok('exit') and (self.door_open or 'door' not in self.memory):
            return 'exit'
        return None

    def step(self):
        if self.get_clock().now().nanoseconds == 0:   # 시뮬레이션 시간 대기
            return
        if self.state is None:
            self.enter_explore('탐색 시작')
            return
        if self.state in ('EXPLORE', 'SEARCH', 'SPIN', 'BEYOND'):
            t = self.next_target()
            if t:
                self.begin_goto(t)
                return
        getattr(self, 'do_' + self.state.lower())()

    # ================= EXPLORE =================
    def enter_explore(self, msg='탐색으로 복귀'):
        self.target = None
        self.idle_pos = None
        self.set_state('EXPLORE', msg)

    def do_explore(self):
        self.set_explore(True)
        pose = self.robot_pose()
        if pose:
            if self.idle_pos is None or math.hypot(pose[0] - self.idle_pos[0],
                                                   pose[1] - self.idle_pos[1]) > 0.15:
                self.idle_pos = pose[:2]
                self.idle_since = self.t()
            elif self.t() - self.idle_since > IDLE_TIME:
                self.begin_unstuck(self.begin_search)
                return
        if self.t() - self.last_log > 10.0:
            self.last_log = self.t()
            mem = ', '.join(NAMES[k] for k in self.memory) or '없음'
            self.get_logger().info(
                f'[EXPLORE] 탐색 중 | 열쇠 {"있음" if self.has_key else "없음"} | 기억: {mem}')

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
        self.target = t
        self.goals = self.candidates(t)
        if t in self.view:   # 가장 잘 보였던 위치를 1순위로
            _, vx, vy, vyaw = self.view[t]
            self.goals.insert(0, (vx, vy, vyaw))
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

    # ================= APPROACH (카메라 + LiDAR 접근) =================
    def begin_approach(self):
        self.app_count = 0
        self.lost_since = None
        self.set_slam_pause(True)   # 접근 중 흔들림으로 지도가 틀어지지 않게
        self.set_state('APPROACH', f'{NAMES[self.target]} 접근')

    def do_approach(self):
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
        angle = seen[0]
        d = self.range_at(angle)
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
        if abs(angle) < math.radians(10) and d is not None and self.front_clear():
            v = clamp(0.5 * (d - APPROACH_DIST), 0.03, 0.12)
        self.send_vel(v, w)

    def remove_model(self, name):
        try:
            r = subprocess.run(
                ['gz', 'service', '-s', f'/world/{WORLD}/remove',
                 '--reqtype', 'gz.msgs.Entity', '--reptype', 'gz.msgs.Boolean',
                 '--timeout', '3000', '--req', f'name: "{name}" type: MODEL'],
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
            self.set_slam_pause(True)
        if not self.remove_model(t):
            self.set_slam_pause(False)
            self.fail('Gazebo 제거 실패')
            return
        if t == 'key':
            self.set_slam_pause(False)
            self.has_key = True
            self.get_logger().info('열쇠 획득')
            self.enter_explore()
            return
        # 문: 지도에는 아직 벽으로 남아 있으므로 직접 직진해서 통과
        self.door_open = True
        self.get_logger().info('문 개방')
        # 접근을 마친 로봇은 문 정면을 보고 있으므로 그 방향으로 직진
        x, y, yaw = self.robot_pose() or (0.0, 0.0, 0.0)
        go = APPROACH_DIST + BEYOND_DIST
        self.beyond = (x + go * math.cos(yaw), y + go * math.sin(yaw))
        self.set_state('BEYOND', f'문 너머 ({self.beyond[0]:.1f}, {self.beyond[1]:.1f})로 직진')

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
        v = 0.1 if abs(err) < 0.3 and self.front_clear() else 0.0
        self.send_vel(v, clamp(1.0 * err, -MAX_W, MAX_W))

    # ================= UNSTUCK (벽에 끼었을 때 뒤로 빠져나오기) =================
    def begin_unstuck(self, then):
        self.set_explore(False)
        self.unstuck_then = then
        pose = self.robot_pose()
        self.unstuck_from = pose[:2] if pose else None
        self.set_state('UNSTUCK', '뒤로 빠져나오기')

    def do_unstuck(self):
        pose = self.robot_pose()
        moved = 0.0
        if pose and self.unstuck_from:
            moved = math.hypot(pose[0] - self.unstuck_from[0], pose[1] - self.unstuck_from[1])
        if moved > 0.2 or self.t() - self.state_start > 8.0 or not self.rear_clear():
            self.stop()
            self.unstuck_then()
            return
        self.send_vel(-0.05, 0.0)

    def resume(self, state):
        self.set_state(state, '이어서 진행')

    # ================= SEARCH (탐색이 멈췄을 때 수색) =================
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
        self.search_pts = self.make_search_points()
        if not self.search_pts:
            self.searched = []
            self.search_pts = self.make_search_points()
        self.search_moving = False
        self.set_state('SEARCH', f'탐색이 멈춰 수색 시작 (지점 {len(self.search_pts)}곳)')

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
        self.get_logger().info(f'[SEARCH] ({x:.1f}, {y:.1f})로 이동, 남은 지점 {len(self.search_pts)}곳')
        self.send_goal(x, y, 0.0)

    def do_spin(self):
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
