"""Revisit remembered observations and inspect known space after frontiers run out."""
import math
import time
from .approach_planner import VerificationWindow
from .grid import reachable
from .visual_search import ObjectMemory, inspection_candidates, safe_view


SEARCH_LABELS = {'SEARCH_KEY':'key','SEARCH_DOOR':'door','SEARCH_EXIT':'exit'}
VISIT_STAGES = {'REVISIT_MOVE','REVISIT_SCAN','REVISIT_WAIT_POSE','REVISIT_CAMERA_WAIT'}


class VisualSearchControl:
    def init_visual_search(self, inspection_limit=16):
        self._visual_search_ready = True
        self.object_memory = ObjectMemory()
        self.inspection_limit = inspection_limit
        self.memory_tried = {label:set() for label in ('key','door','exit')}
        self.completed_views = {label:[] for label in self.memory_tried}
        self.failed_views = {label:[] for label in self.memory_tried}
        self.inspection_used = {label:0 for label in self.memory_tried}
        self.no_view_since = {label:None for label in self.memory_tried}
        self.seen_during_revisit = {label:False for label in self.memory_tried}
        self.inspection_current = None
        self.inspection_frame = None
        self.sweep_view_bins = set()
        self.sweep_frames = 0

    def record_camera_view(self, pose, sim):
        if pose is None:
            return
        sector = int((pose[2]%(2*math.pi))/(2*math.pi/8))%8
        if self.state.phase=='SWEEP_ACTIVE':
            self.sweep_view_bins.add(sector)
            self.sweep_frames += 1
            if self.stage=='REVISIT_SCAN':
                self.revisit_bins.add(sector)
                self.revisit_frames += 1

    def search_costmap(self):
        if self.costmap is None:
            return None
        grid,stamp = self.costmap
        return grid if grid.frame==self.grid.frame and -.2<=self.sim()-stamp<=2 else None

    def current_task_label(self):
        if self.stage in VISIT_STAGES:
            return self.inspection_current['label'] if self.inspection_current else None
        return SEARCH_LABELS.get(self.stage)

    def visual_search_tick(self, wall, sim, robot):
        if not getattr(self,'_visual_search_ready',False):
            self.init_visual_search()
        label = self.current_task_label()
        if label is None:
            return False
        if self.inspection_frame is None:
            self.inspection_frame = self.grid.frame
        elif self.inspection_frame!=self.grid.frame:
            self.stop('지도 좌표계가 바뀌어 기억한 관측 위치를 사용할 수 없어.',2)
            return True
        if self.stage in VISIT_STAGES:
            if (robot is not None and self.recent(label) is not None
                    and sim>=self.object_cooldown[label] and wall>=self.next_approach_plan):
                self.next_approach_plan = wall+1
                observed = self.recent(label)
                self.seen_during_revisit[label] = self.revisit_saw_target = True
                target = self.choose_approach(label,robot,observed[0])
                if target is not None:
                    if self.state.phase=='REVISIT_WAIT':
                        self.state.finish('SELECT')
                    self.emit('재방문 중 현재 카메라에서 목표를 다시 확인했어.',label=label)
                    self.request_switch(label,target)
                    return True
            if self.stage=='REVISIT_WAIT_POSE':
                if robot is not None:
                    self.begin_inspection_scan(retry=getattr(self,'revisit_retry_pending',False))
                elif self.revisit_window.expired(wall,sim):
                    self.stop('재방문 도착 후 로봇 좌표를 확인하지 못했어.',2)
                return True
            if self.stage=='REVISIT_CAMERA_WAIT':
                if self.revisit_window.enough(sim,self.frame_counter):
                    good = self.revisit_frames>=8 and len(self.revisit_bins)==8
                    if good:
                        self.finish_revisit(True,'새 위치에서 카메라 관측을 마쳤어.')
                    elif self.revisit_scan_attempts<2:
                        self.begin_inspection_scan(retry=True)
                    else:
                        self.finish_revisit(False,'회전 중 충분한 카메라 관측 방향을 확인하지 못했어.')
                elif self.revisit_window.expired(wall,sim):
                    self.finish_revisit(False,'재방문 후 새 영상 수신 대기 시간이 지났어.')
                return True
            # MOVE and SCAN keep the existing action watchdog and cancellation logic.
            return False
        if self.state.phase!='SELECT' or robot is None:
            return False
        if self.recent(label) is not None and sim>=self.object_cooldown[label]:
            return False
        return self.try_memory_visit(label,robot,sim)

    def try_memory_visit(self, label, robot, sim):
        if self.inspection_used[label]>=self.inspection_limit:
            return False
        connected = None
        costmap = self.search_costmap()
        for entry in self.object_memory.candidates(label,self.grid.frame,sim,self.memory_tried[label]):
            self.memory_tried[label].add(entry.uid)
            if connected is None:
                connected = reachable(self.grid,self.grid.cell(*robot[:2]))
            if not safe_view(self.grid,self.grid.cell(*entry.pose[:2]),connected,costmap):
                self.emit('기억한 관측 위치가 현재 이동 조건에 맞지 않아 다른 위치를 찾을게.',label=label)
                continue
            heading = math.atan2(entry.point[1]-entry.pose[1],entry.point[0]-entry.pose[0])
            self.begin_revisit(label,entry.pose[:2]+(heading,),'memory')
            self.emit('이전에 카메라로 본 물체의 관측 위치를 우선 재확인할게.',
                      label=label,object_position=entry.point,observed_at=entry.seen)
            return True
        return False

    def begin_revisit(self, label, pose, source):
        self.inspection_used[label] += 1
        self.revisit_saw_target = False
        self.inspection_current = {'label':label,'pose':tuple(pose),'source':source,
                                   'saved_frontier_count':self.reached}
        self.stage,self.mode,self.reached,self.probes = 'REVISIT_MOVE','probe',0,[tuple(pose)]
        self.goal_handle = None
        self.no_goal_since = None
        self.state.finish('SELECT')
        self.emit('카메라 관측 지점으로 재방문할게.',label=label,source=source,target=pose,
                  used=self.inspection_used[label],limit=self.inspection_limit)

    def handle_no_frontier(self, robot, sim):
        return self.schedule_known_space(robot,sim,'Frontier 후보가 없어 기존 공간을 다시 관측할게.')

    def handle_search_limit(self, robot, sim):
        return self.schedule_known_space(robot,sim,'Frontier 이동 시험 한도에 도달해 객체 재탐색으로 전환할게.')

    def schedule_known_space(self, robot, sim, reason):
        label = SEARCH_LABELS.get(self.stage)
        if label is None:
            return False
        if not getattr(self,'_visual_search_ready',False):
            self.init_visual_search()
        if self.inspection_used[label]>=self.inspection_limit:
            self.end_visual_search(label,'INSPECTION_BUDGET_EXHAUSTED','재방문 관측 한도 안에서 목표물을 찾지 못했어.')
            return True
        if self.recent(label) is None and self.try_memory_visit(label,robot,sim):
            return True
        choices,info = inspection_candidates(self.grid,robot,self.completed_views[label],
                        self.failed_views[label],self.search_costmap())
        self.emit('기존 지도 관측 후보 검사',label=label,reason=reason,**info)
        if choices:
            self.no_view_since[label] = None
            self.begin_revisit(label,choices[0],'known_space')
            return True
        wall = time.monotonic()
        if self.no_view_since[label] is None:
            self.no_view_since[label] = wall,sim
        since = self.no_view_since[label]
        if wall-since[0]>45 or sim-since[1]>3:
            code = 'OBJECT_NOT_FOUND_IN_INSPECTED_VIEWS' if self.completed_views[label] else 'OBSERVATION_VIEWPOINT_UNAVAILABLE'
            self.end_visual_search(label,code,'현재 도달 가능한 관측 후보에서 목표물을 확인하지 못했어.')
        return True

    def end_visual_search(self, label, code, reason):
        seen_before = bool(self.object_memory.views.get(label)) or self.seen_during_revisit[label]
        if code=='OBJECT_NOT_FOUND_IN_INSPECTED_VIEWS' and seen_before:
            code = 'OBJECT_NOT_RECONFIRMED_OR_UNREACHABLE'
        self.emit('객체 추가 탐색 종료',label=label,error=code,completed_views=len(self.completed_views[label]),
                  failed_views=len(self.failed_views[label]),memory_tried=len(self.memory_tried[label]),
                  used=self.inspection_used[label],seen_before=seen_before,coverage_claim=False)
        self.stop(reason+' 전체 공간에 물체가 없다고 판정한 것은 아니야.',2)

    def handle_revisit_result(self, future, generation):
        if self.stage!='REVISIT_MOVE':
            return False
        if self.done or self.stopping or not self.state.current(generation):
            return True
        try:
            if future.result().status!=4:
                self.finish_revisit(False,'관측 위치 이동이 성공하지 못했어.')
                return True
            self.goal_handle = None
            self.state.finish('SELECT')
            if self.robot() is None:
                self.stage = 'REVISIT_WAIT_POSE'
                self.revisit_retry_pending = False
                self.revisit_window = VerificationWindow(time.monotonic(),self.sim(),self.frame_counter)
            else:
                self.begin_inspection_scan()
        except Exception as error:
            self.stop('재방문 이동 결과 오류: '+str(error),2)
        return True

    def begin_inspection_scan(self, retry=False):
        pose = self.robot()
        if pose is None:
            self.stage = 'REVISIT_WAIT_POSE'
            self.revisit_retry_pending = retry
            self.revisit_window = VerificationWindow(time.monotonic(),self.sim(),self.frame_counter)
            return
        self.stage = 'REVISIT_SCAN'
        if not retry:
            self.revisit_bins,self.revisit_frames,self.revisit_scan_attempts = set(),0,0
        self.revisit_scan_attempts += 1
        self.revisit_pose = pose
        self.start_sweep(time.monotonic(),purpose='inspection')

    def inspection_rotation_finished(self):
        self.stage = 'REVISIT_CAMERA_WAIT'
        self.state.finish('REVISIT_WAIT')
        self.revisit_window = VerificationWindow(time.monotonic(),self.sim(),self.frame_counter)

    def note_search_sweep(self):
        if not getattr(self,'_visual_search_ready',False):
            return
        label = SEARCH_LABELS.get(self.stage)
        robot = self.robot()
        if label is not None and robot is not None and self.sweep_frames>=8 and len(self.sweep_view_bins)==8:
            self.completed_views[label].append(robot[:2])

    def finish_revisit(self, success, reason):
        current = self.inspection_current
        label = current['label']
        if success:
            self.completed_views[label].append(self.revisit_pose[:2])
        else:
            self.failed_views[label].append(current['pose'][:2])
        self.emit('관측 지점 재확인 결과',label=label,source=current['source'],success=success,
                  reason=reason,object_found=self.revisit_saw_target)
        self.stage,self.mode = 'SEARCH_'+label.upper(),'explore'
        self.reached = current['saved_frontier_count']
        self.failures = 0
        self.inspection_current = None
        self.goal_handle = None
        self.sweep_pending = False
        self.no_goal_since = None
        self.state.finish('SELECT')

    def failed(self, reason):
        if self.stage=='REVISIT_MOVE':
            self.finish_revisit(False,reason)
        else:
            super().failed(reason)
