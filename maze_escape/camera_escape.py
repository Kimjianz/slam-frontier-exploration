"""Camera key acquisition and door opening on the tested frontier controller."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
import time

import cv2
import rclpy
from action_msgs.msg import GoalStatus
from nav2_msgs.action import Spin
from rclpy.action import ActionClient
from rclpy.signals import SignalHandlerOptions
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, LaserScan
from tf2_ros import TransformException

from .mission import Mission, ROOT, fingerprint
from .actuator import SceneActuator
from .vision import (ObservationGate, approach, depth_array, detect, forward_clear,
                     optical_point, rgb_array, transform_point)


def stamp(message):
    return message.header.stamp.sec+message.header.stamp.nanosec/1e9


class CameraEscape(Mission):
    def __init__(self, maximum):
        super().__init__('explore',maximum)
        self.stage = 'SEARCH_KEY'
        self.key_collected = False
        self.rgb = self.depth = self.info = self.scan_message = None
        self.last_pair = -1.0
        self.camera_problem = ''
        self.camera_wall = time.monotonic()
        self.camera_sim = 0.0
        self.observations = {}
        self.gate = ObservationGate()
        self.switch_target = None
        self.switch_deadline = 0.0
        self.wait_started = None
        self.clear_stamps = []
        self.visible_labels = set()
        self.sweep_pending = True
        self.spin_client = ActionClient(self,Spin,'/spin')
        self.worker = ThreadPoolExecutor(max_workers=1)
        self.operation = None
        self.actuator = SceneActuator(ROOT/'worlds/escape_room.world')
        self.image_pub = self.create_publisher(Image,'/maze_escape/camera/detections',self.qos)
        self.rgb_sub = self.create_subscription(Image,'/maze_escape/camera/image_raw',self.receive_rgb,self.qos)
        self.depth_sub = self.create_subscription(Image,'/maze_escape/camera/depth_raw',self.receive_depth,self.qos)
        self.info_sub = self.create_subscription(CameraInfo,'/maze_escape/camera/camera_info',
                                               lambda m:setattr(self,'info',m),self.qos)
        self.door_scan_sub = self.create_subscription(LaserScan,'/scan',
                                    lambda m:setattr(self,'scan_message',m),self.qos)
        self.emit('카메라 임무 시작: 노란 열쇠 블록 획득 후 빨간 문을 열게.')

    def receive_rgb(self, message):
        self.rgb = message
        self.observe()

    def receive_depth(self, message):
        self.depth = message
        self.observe()

    def observe(self):
        if self.rgb is None or self.depth is None or self.info is None or self.grid is None:
            return
        t = stamp(self.rgb)
        if t <= self.last_pair or abs(t-stamp(self.depth)) > .04:
            return
        if not -.2 <= self.sim()-t <= 1:
            return
        if (self.rgb.header.frame_id != 'camera_optical_frame'
                or self.depth.header.frame_id != 'camera_optical_frame'):
            self.camera_problem = '영상 좌표 이름이 camera_optical_frame과 달라.'
            return
        try:
            rgb,depth = rgb_array(self.rgb),depth_array(self.depth)
            if self.info.width != self.rgb.width or self.info.height != self.rgb.height:
                raise ValueError('카메라 보정 정보와 영상 크기가 달라.')
            detections = detect(rgb,depth)
            tr = self.tf.lookup_transform(self.grid.frame,'camera_optical_frame',
                                         Time.from_msg(self.rgb.header.stamp)).transform
            translation = (tr.translation.x,tr.translation.y,tr.translation.z)
            rotation = (tr.rotation.x,tr.rotation.y,tr.rotation.z,tr.rotation.w)
            for detection in detections:
                point = transform_point(optical_point(detection,self.info.k),translation,rotation)
                stable = self.gate.update(detection.label,t,point)
                if stable is not None:
                    self.observations[detection.label] = (stable,t,detection.depth)
                x,y,w,h = detection.box
                cv2.rectangle(rgb,(x,y),(x+w,y+h),(0,255,0),2)
                cv2.putText(rgb,detection.label+' '+format(detection.depth,'.2f')+'m',
                            (x,max(12,y-4)),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,255,0),1)
            message = Image()
            message.header = self.rgb.header
            message.height,message.width = rgb.shape[:2]
            message.encoding,message.step,message.data = 'rgb8',rgb.shape[1]*3,rgb.tobytes()
            self.image_pub.publish(message)
            self.visible_labels = {d.label for d in detections}
            self.last_pair,self.camera_sim,self.camera_wall = t,t,time.monotonic()
            self.camera_problem = ''
        except TransformException as error:
            self.camera_problem = str(error)
            return
        except (ValueError,TypeError) as error:
            self.stop('카메라 처리 오류: '+str(error),2)

    def recent(self, label):
        value = self.observations.get(label)
        return value if value is not None and -.2 <= self.sim()-value[1] <= 1 else None

    def tick(self):
        if self.done or self.stopping:
            return
        wall,sim = time.monotonic(),self.sim()
        if self.last_sim > 0 and sim < self.last_sim-.1:
            self.stop('가상 시간이 초기화됐어. 카메라 임무를 멈출게.',2)
            return
        self.update_grid()
        self.observe()
        if self.done:
            return
        robot = self.robot()
        if self.last_pair < 0:
            if wall-self.started > 45:
                self.stop('카메라 영상·깊이·보정 정보·좌표를 45초 안에 받지 못했어. '+self.camera_problem,2)
            elif wall-self.last_report > 3:
                self.last_report = wall
                self.emit('카메라 준비를 기다리는 중이야.',rgb=self.rgb is not None,
                          depth=self.depth is not None,info=self.info is not None)
            return
        if wall-self.camera_wall > 10 or sim-self.camera_sim > 2:
            self.stop('최신 RGB-D 카메라 처리가 끊겼어. 임무를 멈출게.',2)
            return
        if self.stage in ('PICKING','OPENING'):
            self.poll_operation(wall)
            return
        if self.stage == 'VERIFY_OPEN':
            self.verify_open(wall,sim)
            return
        if self.stage == 'VERIFY_PICKUP':
            if self.last_pair > self.pickup_stamp and 'key' not in self.visible_labels:
                if not self.clear_stamps or self.last_pair > self.clear_stamps[-1]:
                    self.clear_stamps.append(self.last_pair)
                if len(self.clear_stamps) >= 3:
                    self.key_collected = True
                    self.stage,self.mode,self.reached,self.failures = 'SEARCH_DOOR','explore',0,0
                    self.sweep_pending = True
                    self.state.finish('SELECT')
                    self.emit('열쇠 획득 완료. 카메라로 빨간 문을 찾을게.')
            else:
                self.clear_stamps.clear()
            if wall-self.wait_started > 25:
                self.stop('획득 요청 후 열쇠가 사라졌는지 카메라로 확인하지 못했어.',2)
            return
        if self.stage in ('VERIFY_KEY','VERIFY_DOOR'):
            self.verify_arrival(wall,robot)
            return
        if self.stage == 'SWITCHING':
            if wall > self.switch_deadline:
                self.stop('기존 목표 종료를 확인하지 못했어. 추가 이동을 보내지 않을게.',3)
            return
        label = 'key' if self.stage == 'SEARCH_KEY' else 'door' if self.stage == 'SEARCH_DOOR' else None
        observed = self.recent(label) if label else None
        if (self.state.phase == 'WAIT' and robot is not None and self.sim()>0
                and self.scan_stamp is not None and self.sim()-self.scan_stamp <= 1
                and self.navigation_ready(wall)):
            self.state.finish('SELECT')
        if label and observed is not None and robot is not None and self.state.phase != 'WAIT':
            self.emit('카메라에서 목표를 확인했어.',label=label,depth=observed[2],position=observed[0])
            self.request_switch(label,approach(robot,observed[0],.45 if label=='key' else .65))
            if self.stage == 'SWITCHING':
                return
        if label and self.state.phase == 'SELECT' and self.sweep_pending:
            self.start_sweep(wall)
            return
        if self.state.phase in ('SWEEP_REQUEST','SWEEP_ACTIVE'):
            if self.state.expired(wall):
                self.stop('카메라 둘러보기 제한 시간이 지났어.',2)
            return
        super().tick()

    def start_sweep(self, wall):
        if not self.spin_client.server_is_ready():
            self.stop('카메라 둘러보기에 필요한 Nav2 회전 서버가 준비되지 않았어.',2)
            return
        goal = Spin.Goal()
        goal.target_yaw = 2*math.pi
        goal.time_allowance.sec = 120
        generation = self.state.begin('SWEEP_REQUEST',wall,15)
        self.pending = self.spin_client.send_goal_async(goal)
        self.pending.add_done_callback(lambda f,g=generation:self.sweep_accepted(f,g))
        self.emit('카메라로 주변을 한 바퀴 둘러볼게.')

    def sweep_accepted(self, future, generation):
        if self.switch_acceptance(future,generation):
            return
        try:
            handle = future.result()
            if self.stopping or self.done or not self.state.current(generation,'SWEEP_REQUEST'):
                if handle.accepted:
                    if self.stopping or self.done:
                        self.goal_handle = handle
                        self.result_future = handle.get_result_async()
                    handle.cancel_goal_async()
                return
            if not handle.accepted:
                self.stop('카메라 둘러보기 요청이 거부됐어.',2)
                return
            self.goal_handle = handle
            self.state.phase,self.state.deadline = 'SWEEP_ACTIVE',time.monotonic()+360
            self.result_future = handle.get_result_async()
            self.result_future.add_done_callback(lambda f,g=generation:self.sweep_result(f,g))
        except Exception as error:
            self.stop('둘러보기 요청 오류: '+str(error),2)

    def sweep_result(self, future, generation):
        if self.stage == 'SWITCHING':
            self.switched_result(future,generation)
            return
        if self.done or self.stopping or not self.state.current(generation):
            return
        try:
            if future.result().status != GoalStatus.STATUS_SUCCEEDED:
                self.stop('카메라 둘러보기 회전이 성공하지 못했어.',2)
                return
            self.goal_handle = None
            self.sweep_pending = False
            self.state.finish('SELECT')
        except Exception as error:
            self.stop('둘러보기 결과 오류: '+str(error),2)

    def request_switch(self, label, target):
        self.switch_target = label,target
        if self.state.phase == 'SELECT':
            self.activate_target()
            return
        self.stage = 'SWITCHING'
        self.switch_deadline = time.monotonic()+15
        if self.state.phase in ('PLAN_RESULT','NAVIGATING','CANCELLING','SWEEP_ACTIVE'):
            self.cancel_for_switch()

    def cancel_for_switch(self):
        self.state.phase = 'CANCELLING'
        self.switch_deadline = time.monotonic()+5
        self.goal_handle.cancel_goal_async()

    def activate_target(self):
        label,target = self.switch_target
        self.stage = 'APPROACH_KEY' if label=='key' else 'APPROACH_DOOR'
        self.mode,self.reached,self.probes = 'probe',0,[target]
        self.goal_handle = None
        self.state.finish('SELECT')
        self.emit('카메라로 확인한 물체 가까이 이동할게.',label=label,target=target)

    def switch_acceptance(self, future, generation):
        if (self.stage != 'SWITCHING' or not self.state.current(generation)
                or self.done or self.stopping):
            return False
        try:
            handle = future.result()
            if not handle.accepted:
                self.activate_target()
            else:
                self.goal_handle = handle
                self.result_future = handle.get_result_async()
                self.result_future.add_done_callback(lambda f,g=generation:self.switched_result(f,g))
                self.cancel_for_switch()
        except Exception as error:
            self.stop('목표 전환 오류: '+str(error),3)
        return True

    def switched_result(self, future, generation):
        if (self.stage != 'SWITCHING' or not self.state.current(generation)
                or self.done or self.stopping):
            return
        try:
            if future.result().status not in (GoalStatus.STATUS_SUCCEEDED,GoalStatus.STATUS_CANCELED,
                                              GoalStatus.STATUS_ABORTED):
                raise ValueError('기존 목표가 종료 상태가 아니야.')
            self.activate_target()
        except Exception as error:
            self.stop('목표 종료 확인 오류: '+str(error),3)

    def plan_accepted(self, future, generation):
        if not self.switch_acceptance(future,generation):
            super().plan_accepted(future,generation)

    def nav_accepted(self, future, generation):
        if not self.switch_acceptance(future,generation):
            super().nav_accepted(future,generation)

    def plan_result(self, future, generation):
        if self.stage == 'SWITCHING':
            self.switched_result(future,generation)
        else:
            super().plan_result(future,generation)

    def nav_result(self, future, generation):
        if self.stage == 'SWITCHING':
            self.switched_result(future,generation)
            return
        if self.stage not in ('APPROACH_KEY','APPROACH_DOOR'):
            super().nav_result(future,generation)
            if not self.done and self.state.phase=='SELECT' and self.reached % 2 == 0:
                self.sweep_pending = True
            return
        if not self.state.current(generation) or self.stopping or self.done:
            return
        try:
            if future.result().status != GoalStatus.STATUS_SUCCEEDED:
                self.stop('물체로 접근하는 이동이 성공하지 못했어.',2)
                return
            self.goal_handle = None
            self.state.finish('VERIFY')
            self.stage = 'VERIFY_KEY' if self.stage=='APPROACH_KEY' else 'VERIFY_DOOR'
            self.wait_started = time.monotonic()
            self.emit('도착 후 카메라와 거리를 다시 확인할게.',stage=self.stage)
        except Exception as error:
            self.stop('도착 확인 오류: '+str(error),2)

    def verify_arrival(self, wall, robot):
        label = 'key' if self.stage=='VERIFY_KEY' else 'door'
        observed = self.recent(label)
        if observed is not None and robot is not None:
            distance = math.dist(robot[:2],observed[0][:2])
            if distance <= (.70 if label=='key' else .95):
                if label=='door' and not self.key_collected:
                    self.stop('열쇠 획득 기록이 없어 문을 열지 않을게.',2)
                    return
                if label=='door':
                    scan = self.scan_message
                    if scan is None or self.sim()-stamp(scan)>1 or forward_clear(scan):
                        if wall-self.wait_started > 20:
                            self.stop('문이 앞을 막고 있는지 라이다로 확인하지 못했어.',2)
                        return
                self.stage = 'PICKING' if label=='key' else 'OPENING'
                self.operation = self.worker.submit(self.actuator.lift,
                    'red_key' if label=='key' else 'red_door',self.key_collected)
                self.switch_deadline = wall+10
                self.emit('열쇠 획득을 요청했어.' if label=='key' else '열쇠를 사용해 문 열기를 요청했어.')
                return
        if wall-self.wait_started > 20:
            self.stop('도착 후 물체와 가까운 거리를 카메라로 재확인하지 못했어.',2)

    def poll_operation(self, wall):
        if not self.operation.done():
            if wall > self.switch_deadline:
                self.stop('가제보의 물체 조작 응답 시간이 지났어.',2)
            return
        try:
            self.operation.result()
            if self.stage == 'PICKING':
                self.stage = 'VERIFY_PICKUP'
                self.pickup_stamp = self.sim()
                self.wait_started = wall
                self.clear_stamps = []
                self.emit('열쇠 이동 요청이 승인됐어. 카메라로 획득 결과를 확인할게.')
            else:
                self.stage = 'VERIFY_OPEN'
                self.wait_started = wall
                self.open_stamp = self.sim()
                self.clear_stamps = []
                self.emit('문 이동 요청이 승인됐어. 실제로 앞이 비었는지 라이다로 확인할게.')
        except Exception as error:
            self.stop('물체 조작 실패: '+str(error),2)

    def verify_open(self, wall, sim):
        if not self.key_collected:
            self.stop('열쇠 획득 기록이 없어 문 열림 완료로 처리하지 않을게.',2)
            return
        scan = self.scan_message
        if scan is not None and self.open_stamp < stamp(scan) <= sim+.2 and sim-stamp(scan) <= 1:
            t = stamp(scan)
            if not forward_clear(scan):
                self.clear_stamps.clear()
            elif not self.clear_stamps or t > self.clear_stamps[-1]:
                self.clear_stamps.append(t)
            if len(self.clear_stamps) >= 3:
                self.stage = 'COMPLETE'
                self.emit('카메라 열쇠 인식·획득·문 열림 확인 완료!',key_collected=True,door_open=True)
                self.stop('임무 완료.',0)
                return
        if wall-self.wait_started > 30:
            self.stop('문 열림을 라이다로 확인하지 못했어. 성공으로 처리하지 않을게.',2)

    def stop(self, message, code):
        if code==0 and getattr(self,'stage','') != 'COMPLETE':
            message += ' 열쇠 획득과 문 열림은 아직 완료되지 않았어.'
            code = 2
        super().stop(message,code)

    def close(self):
        try:
            super().close()
        finally:
            self.worker.shutdown(wait=False,cancel_futures=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-goals',type=int,default=40)
    args,ros_args = parser.parse_known_args()
    if args.max_goals < 1:
        parser.error('--max-goals는 1 이상이어야 해.')
    try:
        passed = json.loads((ROOT/'logs/probe_pass.json').read_text())
        if passed['fingerprint'] != fingerprint() or passed['goals'] != 2:
            raise ValueError()
    except (OSError,KeyError,ValueError):
        raise SystemExit('먼저 05번 왕복 시험을 통과해야 해.')
    rclpy.init(args=ros_args,signal_handler_options=SignalHandlerOptions.NO)
    node = None
    try:
        node = CameraEscape(args.max_goals)
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node,timeout_sec=.1)
    except KeyboardInterrupt:
        if node is not None:
            node.emit('사용자가 카메라 임무를 멈췄어.')
    finally:
        if node is not None:
            node.close()
        if rclpy.ok():
            rclpy.shutdown()
    return node.exit_code if node is not None else 2
