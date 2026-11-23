#!/usr/bin/env python3
"""Check installed dependencies and prevent duplicate launches in domain 26."""
import argparse
import collections
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage',choices=('install','sim','slam','nav','mission'),required=True)
    args = parser.parse_args()
    try:
        import yaml
        import cv2
        import numpy
        import rclpy
        from ament_index_python.packages import get_package_share_directory
        for package in ('turtlebot3_gazebo','turtlebot3_navigation2','ros_gz_sim',
                        'ros_gz_bridge','slam_toolbox','nav2_bringup',
                        'nav2_rotation_shim_controller','rviz2'):
            path = Path(get_package_share_directory(package))
            if not str(path).startswith('/opt/ros/jazzy/'):
                raise RuntimeError('기존 덮어쓰기 경로를 사용하고 있어: '+str(path))
        tb3 = Path(get_package_share_directory('turtlebot3_gazebo'))
        for relative in ('models/turtlebot3_burger/model.sdf',
                         'params/turtlebot3_burger_bridge.yaml'):
            if not (tb3/relative).is_file():
                raise RuntimeError('로봇 파일이 없어: '+relative)
        if args.stage == 'install':
            print('설치된 ROS·로봇·회전 제어 구성 확인 완료. 기존 파일은 수정하지 않았어.')
            return 0
        rclpy.init()
        node = rclpy.create_node('maze_escape_preflight')
        end = time.monotonic()+2
        while time.monotonic() < end:
            rclpy.spin_once(node,timeout_sec=0.1)
        names = [name for name,namespace in node.get_node_names_and_namespaces()]
        counts = collections.Counter(names)
        duplicates = [name for name in ('slam_toolbox','controller_server','robot_state_publisher')
                      if counts[name] > 1]
        forbidden = {
            'sim':('robot_state_publisher','ros_gz_bridge'),
            'slam':('slam_toolbox',),
            'nav':('controller_server','bt_navigator'),
            'mission':('maze_escape_probe','maze_escape_frontiers','maze_escape_explore'),
        }[args.stage]
        running = [name for name in forbidden if name in names]
        node.destroy_node()
        rclpy.shutdown()
        if duplicates or running:
            raise RuntimeError('이미 실행 중인 프로그램이 있어: '+', '.join(sorted(set(duplicates+running))))
        print('중복 실행 검사 완료. 새 구성을 시작할게.')
        return 0
    except Exception as error:
        print('실행을 멈췄어: '+str(error),file=sys.stderr)
        print('이 오류를 해결한 뒤 같은 명령을 다시 실행해줘.',file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
