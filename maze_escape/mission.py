"""One ROS node, one executor and asynchronous Nav2 actions."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import Point, PoseStamped, Twist, TwistStamped
from lifecycle_msgs.srv import GetState
from nav2_msgs.action import ComputePathToPose, NavigateToPose
from nav_msgs.msg import OccupancyGrid
from rclpy.action import ActionClient
from rclpy.clock import Clock
from rclpy.clock_type import ClockType
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.signals import SignalHandlerOptions
from rclpy.time import Time
from std_msgs.msg import ColorRGBA
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from .action_state import RequestState
from .grid import Grid, Exclusions, angle_difference, candidates, frontiers, path_acceptable, probe_goals

ROOT = Path(__file__).resolve().parents[1]


def fingerprint():
    digest = hashlib.sha256()
    for relative in ('config/nav2.yaml','config/slam.yaml','worlds/escape_room.world',
                     'maze_escape/grid.py','maze_escape/mission.py','run/env.sh',
                     'maze_escape/camera_model.py','config/camera_bridge.yaml','launch/sim.launch.py'):
        digest.update((ROOT/relative).read_bytes())
    return digest.hexdigest()


def yaw(q):
    return math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))


class Mission(Node):
    def __init__(self, mode, maximum):
        super().__init__('maze_escape_'+mode,
                         parameter_overrides=[Parameter('use_sim_time', value=True)])
        self.mode, self.maximum = mode, maximum
        self.state = RequestState()
        self.done, self.exit_code, self.stopping = False, 0, False
        self.started = time.monotonic()
        self.last_report = 0.0
        self.last_sim = 0.0
        self.map_message = self.grid = self.costmap = None
        self.map_key = None
        self.next_frontiers = 0.0
        self.goal_handle = self.pending = self.result_future = None
        self.goal = None
        self.probes = None
        self.reached = self.failures = 0
        self.exclusions = Exclusions()
        self.target, self.anchor, self.start_pose = None, None, None
        self.translation_anchor = None
        self.no_goal_since = None
        self.motion_since = self.translation_since = 0.0
        self.goal_wall = self.goal_sim = 0.0
        self.commands, self.command_types, self.command_subs = {}, {}, {}
        self.scan_stamp = None
        self.qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.map_sub = self.create_subscription(OccupancyGrid, '/map', self.receive_map, latched)
        self.cost_sub = self.create_subscription(OccupancyGrid, '/global_costmap/costmap',
                                                 self.receive_costmap, latched)
        from sensor_msgs.msg import LaserScan
        self.scan_sub = self.create_subscription(LaserScan, '/scan',
            lambda m: setattr(self,'scan_stamp',m.header.stamp.sec+m.header.stamp.nanosec/1e9), self.qos)
        self.tf = Buffer(node=self)
        self.listener = TransformListener(self.tf, self)
        self.markers = self.create_publisher(MarkerArray, '/maze_escape/frontiers', 1)
        self.plan_client = ActionClient(self, ComputePathToPose, '/compute_path_to_pose')
        self.nav_client = ActionClient(self, NavigateToPose, '/navigate_to_pose')
        self.lifecycle = {name:self.create_client(GetState, '/'+name+'/get_state')
                          for name in ('bt_navigator','controller_server')}
        self.lifecycle_values = {}
        self.lifecycle_futures = {}
        self.next_lifecycle = 0.0
        (ROOT/'logs').mkdir(exist_ok=True)
        self.log = (ROOT/'logs'/('session_'+time.strftime('%Y%m%d_%H%M%S')+'_'+mode+'.jsonl')).open('w',encoding='utf-8')
        self.wall_clock = Clock(clock_type=ClockType.STEADY_TIME)
        self.timer = self.create_timer(0.25, self.tick, clock=self.wall_clock)
        self.emit('시작', mode=mode, config=fingerprint())

    def emit(self, message, **values):
        self.get_logger().info(message + (' | '+json.dumps(values,ensure_ascii=False) if values else ''))
        self.log.write(json.dumps({'wall':time.time(),'sim':self.sim(),'state':self.state.phase,
                                   'message':message,**values},ensure_ascii=False)+'\n')
        self.log.flush()

    def sim(self):
        return self.get_clock().now().nanoseconds/1e9

    def receive_map(self, message):
        self.map_message = message

    def receive_costmap(self, message):
        try:
            self.costmap = self.to_grid(message), message.header.stamp.sec+message.header.stamp.nanosec/1e9
        except ValueError:
            self.costmap = None

    @staticmethod
    def to_grid(message):
        i = message.info
        return Grid(i.width,i.height,i.resolution,i.origin.position.x,i.origin.position.y,
                    yaw(i.origin.orientation),tuple(message.data),message.header.frame_id)

    def update_grid(self):
        m = self.map_message
        if m is None:
            return
        key = (m.info.width,m.info.height,m.info.resolution,
               m.info.origin.position.x,m.info.origin.position.y,
               yaw(m.info.origin.orientation),hash(bytes(m.data)))
        if key != self.map_key:
            self.grid = self.to_grid(m)
            self.map_key = key

    def robot(self):
        if self.grid is None:
            return None
        try:
            stamped = self.tf.lookup_transform(self.grid.frame,'base_footprint',Time())
            stamp = stamped.header.stamp.sec+stamped.header.stamp.nanosec/1e9
            if self.sim()-stamp > 2.0:
                return None
            t = stamped.transform
            return t.translation.x,t.translation.y,yaw(t.rotation)
        except TransformException:
            return None

    def refresh_commands(self):
        types = {'geometry_msgs/msg/Twist':Twist, 'geometry_msgs/msg/TwistStamped':TwistStamped}
        for topic in ('/cmd_vel_nav','/cmd_vel_smoothed','/cmd_vel'):
            published = {p.topic_type for p in self.get_publishers_info_by_topic(topic)}
            if len(published) != 1:
                continue
            name = next(iter(published))
            if name not in types or self.command_types.get(topic) == name:
                continue
            old = self.command_subs.pop(topic,None)
            if old is not None:
                self.destroy_subscription(old)
            self.command_subs[topic] = self.create_subscription(types[name],topic,
                lambda m,key=topic:self.receive_command(m,key),self.qos)
            self.command_types[topic] = name

    def receive_command(self, message, topic):
        value = getattr(message,'twist',message)
        self.commands[topic] = {'forward':value.linear.x,'rotation':value.angular.z,
                                'received_sim':self.sim()}

    def navigation_ready(self, wall):
        if wall >= self.next_lifecycle:
            self.next_lifecycle = wall+2
            self.refresh_commands()
            for name, client in self.lifecycle.items():
                future = self.lifecycle_futures.get(name)
                if future is not None and not future.done():
                    continue
                if client.service_is_ready():
                    future = client.call_async(GetState.Request())
                    self.lifecycle_futures[name] = future
                    future.add_done_callback(lambda f,key=name:self.receive_lifecycle(f,key))
        return (self.plan_client.server_is_ready() and self.nav_client.server_is_ready()
                and all(self.lifecycle_values.get(name,(0,0))[0] == 3
                        and wall-self.lifecycle_values[name][1] < 6 for name in self.lifecycle))

    def receive_lifecycle(self, future, name):
        try:
            self.lifecycle_values[name] = future.result().current_state.id,time.monotonic()
        except Exception:
            self.lifecycle_values.pop(name,None)

    def pose(self, target):
        p = PoseStamped()
        p.header.frame_id = self.grid.frame
        p.header.stamp = self.get_clock().now().to_msg()
        p.pose.position.x,p.pose.position.y = float(target[0]),float(target[1])
        p.pose.orientation.z,p.pose.orientation.w = math.sin(target[2]/2),math.cos(target[2]/2)
        return p

    def tick(self):
        if self.done or self.stopping:
            return
        wall, sim = time.monotonic(), self.sim()
        if self.last_sim > 0 and sim < self.last_sim-0.1:
            self.stop('가상 시간이 초기화됐어. 이 시험을 다시 시작해야 해.',2)
            return
        self.last_sim = sim
        try:
            self.update_grid()
        except ValueError as error:
            self.stop(str(error),2)
            return
        robot = self.robot()
        ready = self.navigation_ready(wall) if self.mode != 'frontiers' else True
        if self.state.phase == 'WAIT':
            if (self.grid is None or robot is None or sim <= 0 or not ready
                    or self.scan_stamp is None or sim-self.scan_stamp > 1.0):
                if wall-self.last_report > 3:
                    self.last_report = wall
                    self.emit('지도·로봇 좌표·레이저·자율 이동 준비를 기다리는 중이야.',
                              map=self.grid is not None,tf=robot is not None,nav=ready)
                if wall-self.started > 120:
                    self.stop('준비 조건이 120초 안에 충족되지 않았어.',2)
                return
            if self.mode == 'probe' and self.probes is None:
                self.probes = probe_goals(robot)
                self.emit('자동 이동 시험: 앞쪽 0.6m로 이동한 뒤 출발 위치로 돌아올게.',targets=self.probes)
            self.state.finish('SELECT')
        if self.mode == 'frontiers':
            if robot is None:
                return
            if wall >= self.next_frontiers:
                self.next_frontiers = wall+1
                self.select_candidates(robot,sim,draw_only=True)
            return
        if self.state.phase == 'SELECT':
            if robot is None:
                return
            if self.mode == 'probe':
                if self.reached == 2:
                    (ROOT/'logs/probe_pass.json').write_text(json.dumps(
                        {'fingerprint':fingerprint(),'time':time.time(),'goals':2}),encoding='utf-8')
                    self.stop('두 목표 자동 이동 시험에 성공했어.',0)
                    return
                target = self.probes[self.reached]
            else:
                if self.reached >= self.maximum:
                    if self.handle_search_limit(robot,sim):
                        return
                    self.stop('지정한 목표 수의 탐색 시험이 끝났어. 전체 탐색 완료를 뜻하지는 않아.',0)
                    return
                if wall < self.next_frontiers:
                    return
                self.next_frontiers = wall+1
                choices = self.select_candidates(robot,sim)
                if not choices:
                    if self.handle_no_frontier(robot,sim):
                        return
                    if self.no_goal_since is None:
                        self.no_goal_since = wall,sim
                    if wall-self.no_goal_since[0] > 120 or sim-self.no_goal_since[1] > 30:
                        self.stop('후보를 기다리는 제한 시간이 지났어. 전체 탐색 완료로 처리하지 않을게.',2)
                        return
                    if wall-self.last_report > 5:
                        self.last_report = wall
                        self.emit('사용할 목표가 없어 기다리는 중이야. 전체 지도 완성을 뜻하지는 않아.')
                    return
                self.no_goal_since = None
                choice = choices[0]
                target = choice.x,choice.y,robot[2]
            self.target = target
            if self.grid.value(self.grid.cell(*target[:2])) != 0:
                self.failed('목표 중심이 확인된 빈 공간이 아니야.')
                return
            if self.costmap is not None:
                costmap,stamp = self.costmap
                if costmap.frame == self.grid.frame and -0.5 <= sim-stamp <= 2:
                    value = costmap.value(costmap.cell(*target[:2]))
                    if value is not None and (value == -1 or value >= 99):
                        self.target = target
                        self.failed('이동용 지도에서 목표가 막혀 있어.')
                        return
            self.target,self.goal = target,self.pose(target)
            request = ComputePathToPose.Goal()
            request.goal,request.use_start = self.goal,False
            generation = self.state.begin('PLAN_REQUEST',wall,15)
            self.pending = self.plan_client.send_goal_async(request)
            self.pending.add_done_callback(lambda f,g=generation:self.plan_accepted(f,g))
            self.emit('목표까지의 경로를 확인하는 중이야.',target=target)
        elif self.state.phase in ('PLAN_REQUEST','PLAN_RESULT','NAV_REQUEST') and self.state.expired(wall):
            self.stop('요청 응답 시간이 지났어. 새 목표를 추가로 보내지 않을게.',2)
        elif self.state.phase == 'NAVIGATING':
            if robot is None:
                self.cancel('로봇 좌표를 조회하지 못했어.')
                return
            if math.hypot(robot[0]-self.translation_anchor[0],robot[1]-self.translation_anchor[1]) >= 0.04:
                self.translation_since = self.motion_since = sim
                self.translation_anchor = robot
                self.anchor = robot
            elif abs(angle_difference(robot[2],self.anchor[2])) >= 0.15:
                self.motion_since = sim
                self.anchor = robot
            if (sim-self.motion_since > 20 or sim-self.translation_since > 35
                    or sim-self.goal_sim > 90 or wall-self.goal_wall > 240):
                self.cancel('목표 이동이 정체되거나 제한 시간을 넘었어.')
                return
            if wall-self.last_report > 3:
                self.last_report = wall
                self.emit('이동 확인',goal=self.reached+1,
                    distance=round(math.hypot(robot[0]-self.target[0],robot[1]-self.target[1]),3),
                    sim_elapsed=round(sim-self.goal_sim,2),wall_elapsed=round(wall-self.goal_wall,1),
                    commands=self.commands)
        elif self.state.phase == 'CANCELLING' and self.state.expired(wall):
            self.stop('취소 완료를 확인하지 못했어. 자율 이동 터미널도 종료해야 해.',3)

    def handle_no_frontier(self, robot, sim):
        return False

    def handle_search_limit(self, robot, sim):
        return False

    def plan_accepted(self, future, generation):
        try:
            handle = future.result()
            if self.stopping or self.done or not self.state.current(generation,'PLAN_REQUEST'):
                if handle.accepted:
                    if self.stopping or self.done:
                        self.goal_handle = handle
                        self.result_future = handle.get_result_async()
                    handle.cancel_goal_async()
                return
            if not handle.accepted:
                self.failed('경로 요청이 거부됐어.')
                return
            self.goal_handle = handle
            self.state.phase = 'PLAN_RESULT'
            self.result_future = handle.get_result_async()
            self.result_future.add_done_callback(lambda f,g=generation:self.plan_result(f,g))
        except Exception as error:
            self.stop('경로 요청 오류: '+str(error),2)

    def plan_result(self, future, generation):
        if self.stopping or self.done or not self.state.current(generation,'PLAN_RESULT'):
            return
        try:
            result = future.result()
            path = result.result.path
            points = [(p.pose.position.x,p.pose.position.y) for p in path.poses]
            if (result.status != GoalStatus.STATUS_SUCCEEDED
                    or path.header.frame_id != self.grid.frame
                    or not path_acceptable(points,self.target[:2])):
                self.failed('목표에 도달하는 경로를 확인하지 못했어.')
                return
            if self.mode == 'explore':
                for point in reversed(points[:-1]):
                    if math.hypot(self.target[0]-point[0],self.target[1]-point[1]) > 0.05:
                        self.target = self.target[:2]+(math.atan2(self.target[1]-point[1],self.target[0]-point[0]),)
                        break
                self.goal = self.pose(self.target)
            request = NavigateToPose.Goal()
            request.pose = self.goal
            generation = self.state.begin('NAV_REQUEST',time.monotonic(),15)
            self.pending = self.nav_client.send_goal_async(request)
            self.pending.add_done_callback(lambda f,g=generation:self.nav_accepted(f,g))
        except Exception as error:
            self.stop('경로 결과 오류: '+str(error),2)

    def nav_accepted(self, future, generation):
        try:
            handle = future.result()
            if self.stopping or self.done or not self.state.current(generation,'NAV_REQUEST'):
                if handle.accepted:
                    if self.stopping or self.done:
                        self.goal_handle = handle
                        self.result_future = handle.get_result_async()
                    handle.cancel_goal_async()
                return
            if not handle.accepted:
                self.failed('이동 목표가 거부됐어.')
                return
            self.goal_handle = handle
            self.state.phase = 'NAVIGATING'
            self.result_future = handle.get_result_async()
            self.result_future.add_done_callback(lambda f,g=generation:self.nav_result(f,g))
            self.anchor = self.robot()
            self.translation_anchor = self.anchor
            if self.anchor is None:
                self.cancel('출발 위치를 확인하지 못했어.')
                return
            self.goal_wall,self.goal_sim = time.monotonic(),self.sim()
            self.motion_since = self.translation_since = self.goal_sim
        except Exception as error:
            self.stop('이동 요청 오류: '+str(error),2)

    def nav_result(self, future, generation):
        if not self.state.current(generation) or self.stopping:
            return
        try:
            status = future.result().status
            if self.state.phase == 'CANCELLING':
                self.failed('정체된 목표의 취소가 끝났어.')
            elif status == GoalStatus.STATUS_SUCCEEDED:
                self.reached += 1
                self.failures = 0
                self.exclusions.add(self.target[:2],self.sim(),True)
                self.state.finish('SELECT')
                self.goal_handle = None
                self.emit('목표 도착',count=self.reached)
            else:
                self.failed('자율 이동 결과가 성공이 아니야. 상태='+str(status))
        except Exception as error:
            self.stop('이동 결과 오류: '+str(error),2)

    def failed(self, reason):
        self.emit(reason)
        if self.target is not None:
            self.exclusions.add(self.target[:2],self.sim(),False)
        self.goal_handle = None
        self.failures += 1
        if self.mode == 'probe' or self.failures >= 3:
            self.stop('시험을 멈췄어. 로그를 확인하기 전에는 반복 재시도하지 않을게.',2)
        else:
            self.state.finish('SELECT')

    def cancel(self, reason):
        self.emit(reason)
        if self.goal_handle is None:
            self.stop('취소할 목표 응답을 아직 받지 못했어.',3)
            return
        self.state.phase,self.state.deadline = 'CANCELLING',time.monotonic()+5
        self.goal_handle.cancel_goal_async()

    def stop(self, message, code):
        self.emit(message)
        self.exit_code,self.done = code,True

    def select_candidates(self, robot, sim, draw_only=False):
        choices,count = candidates(self.grid,robot[:2],self.exclusions.active(sim))
        array = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        array.markers.append(clear)
        points = Marker()
        points.header.frame_id = self.grid.frame
        points.header.stamp = self.get_clock().now().to_msg()
        points.ns,points.id,points.type,points.action = 'frontiers',0,Marker.POINTS,Marker.ADD
        points.pose.orientation.w = 1.0
        points.scale.x = points.scale.y = self.grid.resolution
        points.color = ColorRGBA(r=0.1,g=0.5,b=1.0,a=1.0)
        points.points = [Point(x=x,y=y,z=0.03) for x,y in
                         (self.grid.world(cell) for cell in sorted(frontiers(self.grid)))]
        array.markers.append(points)
        for i,candidate in enumerate(choices):
            marker = Marker()
            marker.header = points.header
            marker.ns,marker.id,marker.type,marker.action = 'goals',i,Marker.SPHERE,Marker.ADD
            marker.pose.position = Point(x=candidate.x,y=candidate.y,z=0.08)
            marker.pose.orientation.w = 1.0
            marker.scale.x = marker.scale.y = marker.scale.z = 0.16
            marker.color = ColorRGBA(r=0.0,g=1.0,b=0.0 if i==0 else 1.0,a=1.0)
            array.markers.append(marker)
        self.markers.publish(array)
        if draw_only and time.monotonic()-self.last_report > 3:
            self.last_report = time.monotonic()
            self.emit('프론티어 표시',groups=count,goals=len(choices))
        return choices

    def close(self):
        self.stopping = True
        if self.pending is not None and not self.pending.done():
            end = time.monotonic()+3
            while rclpy.ok() and not self.pending.done() and time.monotonic() < end:
                rclpy.spin_once(self,timeout_sec=0.1)
            if not self.pending.done():
                self.emit('미응답 요청이 남아 있어. 자율 이동 터미널도 종료해야 해.')
        handle = self.goal_handle
        if handle is not None and handle.accepted and self.result_future is not None and not self.result_future.done():
            try:
                future = handle.cancel_goal_async()
                end = time.monotonic()+3
                while (rclpy.ok() and time.monotonic() < end
                       and (not future.done() or not self.result_future.done())):
                    rclpy.spin_once(self,timeout_sec=0.1)
                if not future.done() or not self.result_future.done():
                    self.emit('취소 완료를 확인하지 못했어. 자율 이동 터미널도 종료해야 해.')
            except Exception as error:
                self.emit('종료 시 취소 오류',error=str(error))
        self.log.close()
        self.destroy_node()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',choices=('probe','frontiers','explore'),required=True)
    parser.add_argument('--max-goals',type=int,default=10)
    args,ros_args = parser.parse_known_args()
    if args.max_goals < 1:
        parser.error('--max-goals는 1 이상이어야 해.')
    if args.mode == 'probe':
        (ROOT/'logs/probe_pass.json').unlink(missing_ok=True)
    if args.mode == 'explore':
        try:
            passed = json.loads((ROOT/'logs/probe_pass.json').read_text())
            if passed['fingerprint'] != fingerprint() or passed['goals'] != 2:
                raise ValueError()
        except (OSError,KeyError,ValueError):
            raise SystemExit('먼저 두 목표 자동 이동 시험을 통과해야 해. run/05_probe.sh를 실행해줘.')
    rclpy.init(args=ros_args,signal_handler_options=SignalHandlerOptions.NO)
    node = Mission(args.mode,args.max_goals)
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node,timeout_sec=0.1)
    except KeyboardInterrupt:
        node.emit('사용자가 시험을 멈췄어.')
    finally:
        node.close()
        if rclpy.ok():
            rclpy.shutdown()
    return node.exit_code
