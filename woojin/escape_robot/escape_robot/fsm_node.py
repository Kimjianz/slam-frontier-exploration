import math
import subprocess

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from rclpy.time import Time
from std_msgs.msg import Bool
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import MarkerArray
from nav2_msgs.action import NavigateToPose
from action_msgs.msg import GoalStatus
from tf2_ros import Buffer, TransformListener

STANDOFF = 0.7      # 물체에서 이만큼 떨어진 곳까지 Nav2로 이동 (m)
MAX_FAIL = 5        # 같은 물체 접근 실패 허용 횟수
RETRY_TICKS = 40    # 실패 후 재시도까지 대기 (0.5초 단위 → 20초)
SPIN_TICKS = 44     # 한 바퀴 회전 (0.5초 x 28 = 14초, 0.5 rad/s)
IDLE_TICKS = 30     # 이 시간 동안 안 움직이면 탐색 끝으로 판단 (15초)
SEARCH_STEP = 1.2   # 수색 지점 간격 (m)
NAMES = {'key': '열쇠', 'door': '문', 'exit': '탈출구'}


class FSMNode(Node):
    def __init__(self):
        super().__init__('fsm_node')
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(MarkerArray, '/objects', self.objects_cb, 10)
        self.create_subscription(OccupancyGrid, '/global_costmap/costmap', self.cost_cb, qos)
        self.create_subscription(OccupancyGrid, '/map', self.map_cb, qos)
        self.resume_pub = self.create_publisher(Bool, '/explore/resume', 10)
        self.cmd_pub = self.create_publisher(TwistStamped, '/cmd_vel', 10)
        self.nav = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.objects = {}
        self.costmap = None
        self.map = None
        self.has_key = False
        self.door_open = False
        self.fails = {}
        self.retry = {}
        self.target = None
        self.goals = []
        self.arrived = False
        self.had_goals = False
        self.nav_busy = False
        self.goal_handle = None
        self.seq = 0
        self.proc = None
        self.exploring = None
        self.spin_ticks = 0
        self.spin_next = 'EXPLORE'
        self.search_pts = []
        self.search_moving = False
        self.searched = []
        self.last_pos = None
        self.idle = 0
        self.tick = 0
        self.state = 'EXPLORE'
        self.create_timer(0.5, self.step)
        self.get_logger().info('[EXPLORE] 탐색 시작')

    # ---------- 입력 ----------
    def objects_cb(self, msg):
        for m in msg.markers:
            if m.id == 0:
                self.objects[m.ns] = (m.pose.position.x, m.pose.position.y)

    def cost_cb(self, msg):
        self.costmap = msg

    def map_cb(self, msg):
        self.map = msg

    def robot_xy(self):
        try:
            t = self.tf_buffer.lookup_transform('map', 'base_footprint', Time())
            return t.transform.translation.x, t.transform.translation.y
        except Exception:
            return None

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

    # ---------- 출력 ----------
    def send_cmd(self, w):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.twist.angular.z = float(w)
        self.cmd_pub.publish(msg)

    def set_explore(self, on):
        if self.exploring != on:
            self.resume_pub.publish(Bool(data=on))
            self.exploring = on
            self.get_logger().info('탐색 ' + ('재개' if on else '중단'))

    # ---------- 판단 ----------
    def next_target(self):
        def ok(n):
            return (n in self.objects and self.fails.get(n, 0) < MAX_FAIL
                    and self.tick >= self.retry.get(n, 0))
        if not self.has_key and ok('key'):
            return 'key'
        if self.has_key and not self.door_open and ok('door'):
            return 'door'
        if ok('exit') and (self.door_open or 'door' not in self.objects):
            return 'exit'
        return None

    def candidates(self, name):
        ox, oy = self.objects[name]
        pos = self.robot_xy() or (ox, oy)
        pts = []
        for k in range(12):
            a = 2 * math.pi * k / 12
            px, py = ox + STANDOFF * math.cos(a), oy + STANDOFF * math.sin(a)
            if self.is_free(px, py):
                pts.append((math.hypot(px - pos[0], py - pos[1]), px, py))
        pts.sort()
        return [(px, py) for _, px, py in pts]

    def is_idle(self):
        pos = self.robot_xy()
        if pos is None:
            return False
        if self.last_pos and math.hypot(pos[0] - self.last_pos[0], pos[1] - self.last_pos[1]) < 0.15:
            self.idle += 1
        else:
            self.idle = 0
            self.last_pos = pos
        return self.idle >= IDLE_TICKS

    # 지도의 빈 곳을 일정 간격으로 수색 지점으로
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
                v = m.data[cy * w + cx]
                if not (0 <= v < 50):
                    continue
                x, y = o.x + (cx + 0.5) * res, o.y + (cy + 0.5) * res
                if not self.is_free(x, y):
                    continue
                if any(math.hypot(x - sx, y - sy) < SEARCH_STEP * 0.8 for sx, sy in self.searched):
                    continue
                pts.append((x, y))
        return pts

    # ---------- 상태 전환 ----------
    def enter_explore(self, msg='[EXPLORE] 탐색으로 복귀'):
        self.state = 'EXPLORE'
        self.idle = 0
        self.last_pos = None
        self.get_logger().info(msg)

    def begin_goto(self, t):
        self.cancel_nav()
        self.send_cmd(0.0)
        self.set_explore(False)
        self.target = t
        self.goals = self.candidates(t)
        self.arrived = False
        self.had_goals = bool(self.goals)
        self.state = 'GOTO'
        self.get_logger().info(f'[GOTO] {NAMES[t]} 근처로 이동 (후보 {len(self.goals)}곳)')

    def start_search(self):
        self.set_explore(False)
        self.search_pts = self.make_search_points()
        if not self.search_pts:
            self.searched = []
            self.search_pts = self.make_search_points()
        self.search_moving = False
        self.state = 'SEARCH'
        self.get_logger().info(f'[SEARCH] 탐색이 멈춰 수색 시작 (지점 {len(self.search_pts)}곳)')

    def start_spin(self, next_state):
        self.spin_ticks = SPIN_TICKS
        self.spin_next = next_state
        self.state = 'SPIN'

    def step(self):
        self.tick += 1
        s = self.state

        if s in ('EXPLORE', 'SEARCH') or (s == 'SPIN' and self.spin_next == 'SEARCH'):
            t = self.next_target()
            if t:
                self.begin_goto(t)
                return

        if s == 'EXPLORE':
            self.set_explore(True)
            if self.is_idle():
                self.start_search()
            elif self.tick % 20 == 0:
                mem = ', '.join(NAMES[k] for k in self.objects) or '없음'
                self.get_logger().info(
                    f'[EXPLORE] 탐색 중 | 열쇠 {"있음" if self.has_key else "없음"} | 기억: {mem}')

        elif s == 'SEARCH':
            if self.nav_busy:
                return
            if self.search_moving:
                self.search_moving = False
                if self.arrived:
                    self.start_spin('SEARCH')
                    return
            if not self.search_pts:
                self.searched = []
                self.enter_explore('[EXPLORE] 수색 지점을 다 돌아서 탐색 재개')
                return
            pos = self.robot_xy() or (0.0, 0.0)
            self.search_pts.sort(key=lambda p: math.hypot(p[0] - pos[0], p[1] - pos[1]))
            x, y = self.search_pts.pop(0)
            self.searched.append((x, y))
            self.arrived = False
            self.search_moving = True
            self.get_logger().info(f'[SEARCH] ({x:.1f}, {y:.1f})로 이동, 남은 지점 {len(self.search_pts)}곳')
            self.send_goal(x, y)

        elif s == 'SPIN':
            self.spin_ticks -= 1
            self.send_cmd(0.3 if self.spin_ticks > 0 else 0.0)
            if self.spin_ticks <= 0:
                if self.spin_next == 'SEARCH':
                    self.state = 'SEARCH'
                else:
                    self.enter_explore('[EXPLORE] 문 너머 탐색')

        elif s == 'GOTO':
            if self.nav_busy:
                return
            if not self.goals:
                if self.arrived or not self.had_goals:
                    self.start_approach()
                else:
                    self.get_logger().warn('근처로 이동 못함')
                    self.on_approach_done(False)
                return
            self.send_goal(*self.goals.pop(0))

        elif s == 'APPROACH':
            if self.proc.poll() is None:
                return
            self.on_approach_done(self.proc.returncode == 0)

    # ---------- Nav2 ----------
    def send_goal(self, x, y):
        if not self.nav.wait_for_server(timeout_sec=1.0):
            return
        if self.state == 'GOTO' and self.target in self.objects:
            ox, oy = self.objects[self.target]
            yaw = math.atan2(oy - y, ox - x)   # 물체를 바라보게
        else:
            yaw = 0.0
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
        self.nav_busy = False

    def cancel_nav(self):
        if self.nav_busy and self.goal_handle:
            self.goal_handle.cancel_goal_async()
        self.seq += 1
        self.nav_busy = False

    # ---------- 접근 ----------
    def start_approach(self):
        self.state = 'APPROACH'
        self.get_logger().info(f'[APPROACH] {NAMES[self.target]} 접근')
        self.proc = subprocess.Popen(
            ['ros2', 'run', 'escape_robot', 'approach_node', '--ros-args',
             '-p', f'target:={self.target}', '-p', 'use_sim_time:=true'])

    def on_approach_done(self, ok):
        t = self.target
        self.target = None
        if not ok:
            self.fails[t] = self.fails.get(t, 0) + 1
            self.retry[t] = self.tick + RETRY_TICKS
            self.get_logger().warn(f'{NAMES[t]} 접근 실패 ({self.fails[t]}/{MAX_FAIL})')
            self.enter_explore()
        elif t == 'key':
            self.has_key = True
            self.get_logger().info('열쇠 획득')
            self.enter_explore()
        elif t == 'door':
            self.door_open = True
            self.get_logger().info('문 개방')
            self.get_logger().info('[SPIN] 지도 갱신을 위해 한 바퀴 회전')
            self.start_spin('EXPLORE')
        elif t == 'exit':
            self.state = 'DONE'
            self.set_explore(False)
            self.get_logger().info('[DONE] 탈출 성공!')


def main():
    rclpy.init()
    rclpy.spin(FSMNode())


if __name__ == '__main__':
    main()
