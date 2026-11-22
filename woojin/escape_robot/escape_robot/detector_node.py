import math

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import Image, LaserScan
from visualization_msgs.msg import Marker, MarkerArray
from tf2_ros import Buffer, TransformListener

FOV = 1.4          # 카메라 시야각 (rad)
MIN_AREA = 50      # 이보다 작은 영역은 무시 (픽셀 수)
MAX_ANGLE = 25.0   # 화면 가장자리는 부정확해서 이 각도 안쪽만 위치 계산 (도)
SAME_R = 0.5       # 기억한 위치와 이 거리 안이면 같은 물체로 봄 (m)

# 이름: (찾을 색 채널 번호, 표시 색)  채널 0=R, 1=G, 2=B
TARGETS = {
    'key':  (0, (255, 0, 0)),
    'exit': (1, (0, 255, 0)),
    'door': (2, (0, 0, 255)),
}
NAMES = {'key': '열쇠', 'door': '문', 'exit': '탈출구'}


class DetectorNode(Node):
    def __init__(self):
        super().__init__('detector_node')
        self.create_subscription(Image, '/camera/image_raw', self.image_cb, 10)
        self.create_subscription(LaserScan, '/scan', self.scan_cb, qos_profile_sensor_data)
        self.debug_pub = self.create_publisher(Image, '/detect/image', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/objects', 10)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.scan = None
        self.memory = {}   # 이름: [x, y, 관측 횟수]
        self.obs = {}      # 이름: 최근 관측 위치들
        self.last_log = self.get_clock().now()

    def scan_cb(self, msg):
        self.scan = msg

    def to_rgb(self, msg):
        img = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)
        img = img[:, :msg.width * 3].reshape(msg.height, msg.width, 3)
        if msg.encoding == 'bgr8':
            img = img[:, :, ::-1]
        return img

    # 한 색이 강하고 나머지 두 색이 약한 픽셀 찾기
    def color_mask(self, img, ch):
        c = img[:, :, ch].astype(np.int16)
        o1 = img[:, :, (ch + 1) % 3].astype(np.int16)
        o2 = img[:, :, (ch + 2) % 3].astype(np.int16)
        return ((c > 70) & (o1 < c * 0.4) & (o2 < c * 0.4)).astype(np.uint8)

    # 1. 카메라 방향으로 LiDAR 거리 읽기
    def range_at(self, angle):
        s = self.scan
        if s is None:
            return None
        n = len(s.ranges)
        i = int(round((angle - s.angle_min) / s.angle_increment))
        vals = [s.ranges[(i + k) % n] for k in range(-2, 3)]
        vals = [r for r in vals if s.range_min < r < s.range_max and math.isfinite(r)]
        return float(np.median(vals)) if vals else None

    # 2. 로봇 기준 (거리, 방향) → 지도 좌표
    def to_map(self, r, angle):
        try:
            t = self.tf_buffer.lookup_transform('map', self.scan.header.frame_id, Time())
        except Exception:
            return None
        q = t.transform.rotation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y ** 2 + q.z ** 2))
        x = t.transform.translation.x + r * math.cos(yaw + angle)
        y = t.transform.translation.y + r * math.sin(yaw + angle)
        return x, y

    # 3. 위치 기억 (여러 번 본 값의 평균)
    def remember(self, name, x, y):
        obs = self.obs.setdefault(name, [])
        if not obs:
            self.get_logger().info(f'새로 발견: {NAMES[name]} ({x:.2f}, {y:.2f})')
        obs.append((x, y))
        del obs[:-50]   # 최근 50개만 유지
        xs = sorted(p[0] for p in obs)
        ys = sorted(p[1] for p in obs)
        n = self.memory.get(name, [0, 0, 0])[2] + 1
        self.memory[name] = [xs[len(xs) // 2], ys[len(ys) // 2], n]   # 중앙값

    def image_cb(self, msg):
        img = self.to_rgb(msg)
        h, w = img.shape[:2]
        f = (w / 2) / math.tan(FOV / 2)
        debug = img.copy()

        for name, (ch, color) in TARGETS.items():
            mask = self.color_mask(img, ch)
            if int(mask.sum()) < MIN_AREA:
                continue
            x, y, bw, bh = cv2.boundingRect(mask)
            cx = x + bw / 2
            angle = math.atan2(w / 2 - cx, f)   # 왼쪽 +, 오른쪽 -
            cv2.rectangle(debug, (x, y), (x + bw, y + bh), color, 2)
            cv2.putText(debug, name, (x, max(y - 5, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            if x <= 2 or x + bw >= w - 2:   # 화면 가장자리에 잘린 물체는 제외
                continue
            if abs(math.degrees(angle)) > MAX_ANGLE:
                continue
            r = self.range_at(angle)
            if r is None:
                continue
            p = self.to_map(r, angle)
            if p:
                self.remember(name, *p)

        now = self.get_clock().now()
        if (now - self.last_log).nanoseconds > 2e9:
            if self.memory:
                self.get_logger().info(' | '.join(
                    f'{NAMES[k]} ({v[0]:.2f}, {v[1]:.2f}) {v[2]}회'
                    for k, v in self.memory.items()))
            self.last_log = now

        out = Image()
        out.header = msg.header
        out.height, out.width = h, w
        out.encoding, out.step = 'rgb8', w * 3
        out.data = np.ascontiguousarray(debug).tobytes()
        self.debug_pub.publish(out)
        self.publish_markers()

    # RViz 표시: 기억한 물체 위치에 색 공 + 이름
    def publish_markers(self):
        arr = MarkerArray()
        for i, (name, (x, y, _)) in enumerate(self.memory.items()):
            rgb = [c / 255 for c in TARGETS[name][1]]
            for j, mtype in enumerate((Marker.SPHERE, Marker.TEXT_VIEW_FACING)):
                m = Marker()
                m.header.frame_id = 'map'
                m.ns, m.id, m.type = name, j, mtype
                m.pose.position.x, m.pose.position.y = x, y
                m.pose.position.z = 0.1 if j == 0 else 0.4
                m.pose.orientation.w = 1.0
                m.scale.x = m.scale.y = m.scale.z = 0.25
                m.color.r, m.color.g, m.color.b, m.color.a = *rgb, 1.0
                if j == 1:
                    m.text = name
                    m.color.r = m.color.g = m.color.b = 1.0
                arr.markers.append(m)
        self.marker_pub.publish(arr)


def main():
    rclpy.init()
    rclpy.spin(DetectorNode())


if __name__ == '__main__':
    main()
