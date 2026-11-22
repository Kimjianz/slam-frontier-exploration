import math
import subprocess

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import TwistStamped

FOV = 1.4            # 카메라 시야각 (rad)
MIN_AREA = 50        # 이보다 작은 영역은 무시 (픽셀 수)
GOAL_DIST = 0.5      # 획득 거리 (m)
GOAL_COUNT = 5       # 연속 확인 횟수
ALIGN_DEG = 10.0     # 이 각도 안이면 전진 시작 (도)
MAX_V = 0.15         # 최대 전진 속도 (m/s)
MAX_W = 0.3          # 최대 회전 속도 (rad/s)
SEARCH_TIMEOUT = 40  # 이 시간 동안 못 찾으면 실패 (s)
WORLD = 'default'

CHANNEL = {'key': 0, 'exit': 1, 'door': 2}   # 0=R, 1=G, 2=B
NAMES = {'key': '열쇠', 'door': '문', 'exit': '탈출구'}


class ApproachNode(Node):
    def __init__(self):
        super().__init__('approach_node')
        self.target = self.declare_parameter('target', 'key').value
        self.create_subscription(Image, '/camera/image_raw', self.image_cb, 10)
        self.create_subscription(LaserScan, '/scan', self.scan_cb, qos_profile_sensor_data)
        self.cmd_pub = self.create_publisher(TwistStamped, '/cmd_vel', 10)
        self.scan = None
        self.angle = None          # 화면 속 목표 방향 (rad, 왼쪽 +)
        self.seen_time = None
        self.count = 0
        self.done = False
        self.success = False
        self.start = None
        self.create_timer(0.1, self.control)
        self.get_logger().info(f'{NAMES[self.target]} 접근 시작')

    def scan_cb(self, msg):
        self.scan = msg

    def image_cb(self, msg):
        img = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)
        img = img[:, :msg.width * 3].reshape(msg.height, msg.width, 3)
        if msg.encoding == 'bgr8':
            img = img[:, :, ::-1]
        ch = CHANNEL[self.target]
        c = img[:, :, ch].astype(np.int16)
        o1 = img[:, :, (ch + 1) % 3].astype(np.int16)
        o2 = img[:, :, (ch + 2) % 3].astype(np.int16)
        mask = (c > 70) & (o1 < c * 0.4) & (o2 < c * 0.4)
        if mask.sum() < MIN_AREA:
            return
        cx = np.nonzero(mask)[1].mean()
        f = (msg.width / 2) / math.tan(FOV / 2)
        self.angle = math.atan2(msg.width / 2 - cx, f)
        self.seen_time = self.get_clock().now()

    def range_at(self, angle):
        s = self.scan
        if s is None:
            return None
        n = len(s.ranges)
        i = int(round((angle - s.angle_min) / s.angle_increment))
        vals = [s.ranges[(i + k) % n] for k in range(-2, 3)]
        vals = [r for r in vals if s.range_min < r < s.range_max and math.isfinite(r)]
        return float(np.median(vals)) if vals else None

    def send(self, v, w):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.twist.linear.x = float(v)
        msg.twist.angular.z = float(w)
        self.cmd_pub.publish(msg)

    def control(self):
        if self.done:
            return
        now = self.get_clock().now()
        if self.start is None or self.start.nanoseconds == 0:
            self.start = now
        visible = self.seen_time and (now - self.seen_time).nanoseconds < 0.5e9

        # 1. 안 보이면 제자리 회전하며 찾기
        if not visible:
            self.count = 0
            if (now - self.start).nanoseconds / 1e9 > SEARCH_TIMEOUT:
                self.get_logger().warn(f'{NAMES[self.target]}을(를) 찾지 못함, 실패')
                self.finish(False)
                return
            self.send(0.0, 0.3)
            return
        self.start = now

        # 2. 화면 중앙에 오도록 회전
        w = max(-MAX_W, min(MAX_W, 1.5 * self.angle))
        d = self.range_at(self.angle)

        # 4. 0.5m 이내 연속 확인
        if d is not None and d < GOAL_DIST:
            self.count += 1
            self.send(0.0, 0.0)
            self.get_logger().info(f'거리 {d:.2f}m, 확인 {self.count}/{GOAL_COUNT}')
            if self.count >= GOAL_COUNT:
                self.finish(True)
            return
        self.count = 0

        # 3. 중앙이면 전진 (가까울수록 느리게)
        v = 0.0
        if abs(math.degrees(self.angle)) < ALIGN_DEG and d is not None:
            v = max(0.03, min(MAX_V, 0.5 * (d - GOAL_DIST)))
        self.send(v, w)

    def finish(self, ok):
        self.done = True
        for _ in range(3):
            self.send(0.0, 0.0)
        if not ok:
            return
        if self.target == 'exit':
            self.get_logger().info('탈출구 도착, 탈출 성공')
            self.success = True
            return
        r = subprocess.run(
            ['gz', 'service', '-s', f'/world/{WORLD}/remove',
             '--reqtype', 'gz.msgs.Entity', '--reptype', 'gz.msgs.Boolean',
             '--timeout', '2000', '--req', f'name: "{self.target}" type: MODEL'],
            capture_output=True, text=True)
        ok = 'true' in r.stdout
        self.success = ok
        self.get_logger().info(
            f'{NAMES[self.target]} {"획득" if self.target == "key" else "개방"} '
            f'{"성공" if ok else "실패: " + r.stdout + r.stderr}')


def main():
    rclpy.init()
    node = ApproachNode()
    while rclpy.ok() and not node.done:
        rclpy.spin_once(node, timeout_sec=0.1)
    code = 0 if node.success else 1
    node.destroy_node()
    rclpy.shutdown()
    raise SystemExit(code)


if __name__ == '__main__':
    main()
