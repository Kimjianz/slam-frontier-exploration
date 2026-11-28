"""LiDAR가 아무것도 못 맞힌 방향(inf)을 '최대 거리 바로 앞'으로 바꿔서 /scan_fixed로 내보냄.
SLAM Toolbox는 inf 값을 버려서 벽이 먼 방향은 빈 공간으로도 칠하지 않는다.
inf를 range_max 직전 값으로 바꾸고 SLAM의 max_laser_range를 그보다 작게 두면
그 방향은 '장애물 없이 max_laser_range까지 비어 있음'으로 처리된다.
"""
import math

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan


class ScanFixer(Node):
    def __init__(self):
        super().__init__('scan_fixer')
        self.pub = self.create_publisher(LaserScan, '/scan_fixed', qos_profile_sensor_data)
        self.create_subscription(LaserScan, '/scan', self.cb, qos_profile_sensor_data)

    def cb(self, msg):
        far = msg.range_max - 0.01
        msg.ranges = [far if (math.isinf(r) and r > 0) or r >= msg.range_max else r
                      for r in msg.ranges]
        self.pub.publish(msg)


def main():
    rclpy.init()
    rclpy.spin(ScanFixer())


if __name__ == '__main__':
    main()
