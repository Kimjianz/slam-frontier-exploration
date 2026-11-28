"""Pass an observed open door with Nav2 local collision checks, then seek the exit."""
import math
import time
from action_msgs.msg import GoalStatus
from nav2_msgs.action import DriveOnHeading
from nav2_msgs.srv import ClearEntireCostmap
from .approach_planner import VerificationWindow, crossed_plane, passage_distance
from .vision import forward_clear


def scan_time(scan):
    return scan.header.stamp.sec+scan.header.stamp.nanosec/1e9


class PassageControl:
    def closed_door_in_front(self):
        scan = self.scan_message
        return (scan is not None and -.2<=self.sim()-scan_time(scan)<=1
                and not forward_clear(scan))

    def verify_open(self, wall, sim):
        if not self.key_collected:
            self.stop('열쇠 획득 기록이 없어 문 열림 완료로 처리하지 않을게.',2)
            return
        robot = self.robot()
        if robot is None:
            if self.open_window.expired(wall,sim):
                self.stop('문 통과를 준비할 로봇 좌표를 확인하지 못했어.',2)
            return
        self.cross_distance = passage_distance(robot,self.door_point)
        scan = self.scan_message
        if scan is not None and self.open_stamp < scan_time(scan) <= sim+.2 and sim-scan_time(scan)<=1:
            t = scan_time(scan)
            if not forward_clear(scan,self.cross_distance+.15):
                self.clear_stamps.clear()
            elif not self.clear_stamps or t>self.clear_stamps[-1]:
                self.clear_stamps.append(t)
            if len(self.clear_stamps)>=3:
                self.door_open = True
                self.emit('문 열림 확인 완료. 열린 문을 통과할게.',key_collected=True,door_open=True,
                          crossing_distance=round(self.cross_distance,2))
                desired = math.atan2(self.door_point[1]-robot[1],self.door_point[0]-robot[0])
                delta = math.atan2(math.sin(desired-robot[2]),math.cos(desired-robot[2]))
                if abs(delta)>.10:
                    self.stage = 'ALIGN_PASSAGE'
                    self.start_sweep(wall,delta,'passage')
                else:
                    self.begin_passage_clear()
                return
        if self.open_window.expired(wall,sim):
            self.stop('문 앞 통과 공간이 비었다는 것을 확인하지 못했어.',2)

    def begin_passage_clear(self):
        if not self.key_collected or not self.door_open:
            self.stop('열쇠와 문 열림 확인 없이 문을 통과할 수 없어.',2)
            return
        if not self.clear_local.service_is_ready():
            self.stop('문 통과 준비를 위한 이동용 지도 갱신 서비스가 준비되지 않았어.',2)
            return
        self.stage = 'CLEAR_PASSAGE'
        self.state.finish('PASSAGE')
        self.clear_future = self.clear_local.call_async(ClearEntireCostmap.Request())
        self.passage_deadline = time.monotonic()+15
        self.emit('열린 문 주변의 이동용 지도를 새 라이다 관측으로 갱신할게.')

    def passage_tick(self, wall, sim, robot):
        if self.stage=='CLEAR_PASSAGE':
            if self.clear_future.done():
                try:
                    self.clear_future.result()
                    self.stage = 'WAIT_PASSAGE'
                    self.clear_sim = sim
                    self.clear_stamps = []
                    self.passage_window = VerificationWindow(wall,sim,self.frame_counter)
                except Exception as error:
                    self.stop('문 통과용 지도 갱신 오류: '+str(error),2)
            elif wall>self.passage_deadline:
                self.stop('문 통과용 지도 갱신 응답 시간이 지났어.',2)
            return
        if self.stage=='WAIT_PASSAGE':
            scan = self.scan_message
            if robot is not None:
                self.cross_distance = passage_distance(robot,self.door_point)
            if scan is not None and self.clear_sim<scan_time(scan)<=sim+.2 and sim-scan_time(scan)<=1:
                t = scan_time(scan)
                if not forward_clear(scan,self.cross_distance+.15):
                    self.clear_stamps.clear()
                elif not self.clear_stamps or t>self.clear_stamps[-1]:
                    self.clear_stamps.append(t)
            local = self.last_local_costmap
            fresh_map = (local is not None and self.clear_sim<local[0]<=sim+.2
                         and sim-local[0]<=1.5 and wall-local[1]<5)
            if robot is not None and fresh_map and len(self.clear_stamps)>=3:
                self.start_crossing(robot,wall,sim)
            elif self.passage_window.expired(wall,sim):
                self.stop('문 통과에 필요한 최신 이동용 지도와 빈 통로를 확인하지 못했어.',2)
            return
        if self.stage=='CROSSING':
            if self.state.phase=='CANCELLING':
                if self.state.expired(wall):
                    self.stop('문 통과 목표 취소 완료를 확인하지 못했어.',3)
            elif self.state.expired(wall):
                self.stop('문 통과 요청 제한 시간이 지났어.',2)
            elif (self.state.phase=='CROSS_ACTIVE' and
                  (sim-self.cross_progress_sim>20 or wall-self.cross_progress_wall>180)):
                self.cancel('문 통과가 정체되어 취소할게.')
            return
        if self.stage=='VERIFY_CROSS':
            known = robot is not None and self.grid.value(self.grid.cell(*robot[:2]))==0
            if known and crossed_plane(robot,self.cross_start,self.door_point):
                self.crossed_door = True
                self.stage,self.mode,self.reached,self.failures = 'SEARCH_EXIT','explore',0,0
                self.sweep_pending = True
                self.no_goal_since = None
                self.state.finish('SELECT')
                self.emit('문 통과 확인 완료. 카메라로 초록색 exit_marker를 찾을게.',
                          crossed_door=True,position=robot)
            elif self.passage_window.expired(wall,sim):
                self.stop('문 반대편 위치와 지도 갱신을 확인하지 못했어.',2)

    def start_crossing(self, robot, wall, sim):
        if not self.drive_client.server_is_ready():
            self.stop('Nav2 문 통과 이동 서버가 준비되지 않았어.',2)
            return
        self.cross_start = robot
        self.cross_progress_sim,self.cross_progress_wall,self.cross_traveled = sim,wall,0.0
        goal = DriveOnHeading.Goal()
        goal.target.x = float(self.cross_distance)
        goal.speed = .06
        goal.time_allowance.sec = 120
        self.stage = 'CROSSING'
        generation = self.state.begin('CROSS_REQUEST',wall,15)
        self.pending = self.drive_client.send_goal_async(goal,
            feedback_callback=lambda f,g=generation:self.cross_feedback(f,g))
        self.pending.add_done_callback(lambda f,g=generation:self.cross_accepted(f,g))
        self.emit('라이다와 Nav2 충돌 검사를 유지하며 열린 문을 통과할게.',distance=self.cross_distance)

    def cross_feedback(self, message, generation):
        if self.done or self.stopping or not self.state.current(generation):
            return
        traveled = message.feedback.distance_traveled
        if traveled-self.cross_traveled >= .02:
            self.cross_traveled = traveled
            self.cross_progress_sim,self.cross_progress_wall = self.sim(),time.monotonic()

    def cross_accepted(self, future, generation):
        try:
            handle = future.result()
            if self.done or self.stopping or not self.state.current(generation,'CROSS_REQUEST'):
                if handle.accepted:
                    if self.done or self.stopping:
                        self.goal_handle,self.result_future = handle,handle.get_result_async()
                    handle.cancel_goal_async()
                return
            if not handle.accepted:
                self.stop('문 통과 이동 요청이 거부됐어.',2)
                return
            self.goal_handle = handle
            self.state.phase,self.state.deadline = 'CROSS_ACTIVE',time.monotonic()+600
            self.cross_progress_wall,self.cross_progress_sim = time.monotonic(),self.sim()
            self.result_future = handle.get_result_async()
            self.result_future.add_done_callback(lambda f,g=generation:self.cross_result(f,g))
        except Exception as error:
            self.stop('문 통과 요청 오류: '+str(error),2)

    def cross_result(self, future, generation):
        if self.done or self.stopping or not self.state.current(generation):
            return
        try:
            wrapped = future.result()
            if self.state.phase=='CANCELLING' or wrapped.status!=GoalStatus.STATUS_SUCCEEDED:
                self.stop('문 통과 이동이 완료되지 않았어.',2)
                return
            self.goal_handle = None
            self.stage = 'VERIFY_CROSS'
            self.state.finish('VERIFY_CROSS')
            self.passage_window = VerificationWindow(time.monotonic(),self.sim(),self.frame_counter)
        except Exception as error:
            self.stop('문 통과 결과 오류: '+str(error),2)
