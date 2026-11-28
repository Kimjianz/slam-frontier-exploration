"""Camera approach retries and reacquisition; never treat unknown cells as free."""
import math
import time
from .approach_planner import LIMITS, VerificationWindow, select_approaches


class ApproachRecovery:
    def begin_verification(self, label):
        self.stage = 'VERIFY_'+label.upper()
        self.target_label = label
        self.state.finish('VERIFY')
        self.wait_started = time.monotonic()
        self.verification = VerificationWindow(self.wait_started,self.sim(),self.frame_counter)
        self.emit('도착 후 카메라와 거리를 다시 확인할게.',stage=self.stage,
                  key_collected=self.key_collected)

    def choose_approach(self, label, robot, point):
        sim = self.sim()
        self.rejected[label] = [(p,t) for p,t in self.rejected[label] if t>sim]
        costmap = None
        if self.costmap is not None:
            candidate,stamp_value = self.costmap
            if candidate.frame==self.grid.frame and -.2<=sim-stamp_value<=2:
                costmap = candidate
        goals,info = select_approaches(self.grid,robot,point,label,
                        [p for p,t in self.rejected[label]],costmap=costmap)
        self.emit('접근 후보 검사',label=label,**info,usable=len(goals))
        if not goals:
            return None
        self.object_point = tuple(point)
        self.target_goal_kind = goals[0].kind
        self.emit('이동 가능한 접근 후보 선택',label=label,kind=goals[0].kind,
                  target=goals[0].pose,grid_route=round(goals[0].path_length,2))
        return goals[0].pose

    def failed(self, reason):
        if self.stage.startswith('APPROACH_'):
            label = self.stage.split('_',1)[1].lower()
            self.emit('접근 후보 실패. 다른 위치를 검사할게.',label=label,reason=reason,
                      target=self.target,map_value=self.grid.value(self.grid.cell(*self.target[:2])))
            self.rejected[label].append((self.target[:2],self.sim()+90))
            self.replan_counts[label] += 1
            self.goal_handle = None
            self.state.finish('SELECT')
            self.stage = 'REPLAN_'+label.upper()
            self.next_approach_plan = time.monotonic()+.5
        else:
            super().failed(reason)

    def replan_target(self, label, robot, wall):
        if robot is None or wall < self.next_approach_plan:
            return
        self.next_approach_plan = wall+1
        if self.replan_counts[label] >= 8:
            self.resume_search(label,'여러 접근 후보가 실패해서 지도를 더 관측할게.')
            return
        observed = self.recent(label)
        if observed is None:
            self.begin_reacquire(label,robot)
            return
        goal = self.choose_approach(label,robot,observed[0])
        if goal is None:
            self.begin_reacquire(label,robot)
        else:
            self.request_switch(label,goal)

    def resume_search(self, label, reason):
        self.resume_counts[label] += 1
        if self.resume_counts[label] > 3:
            self.stop('관측과 접근을 반복했지만 물체에 도달하지 못했어. '+reason,2)
            return
        self.emit('물체 주변을 다시 탐색할게.',label=label,reason=reason)
        self.stage,self.mode = 'SEARCH_'+label.upper(),'explore'
        self.reached = self.failures = 0
        self.replan_counts[label] = self.reacquire_counts[label] = 0
        self.object_cooldown[label] = self.sim()+3
        self.sweep_pending = True
        self.goal_handle = None
        self.state.finish('SELECT')

    def begin_reacquire(self, label, robot):
        attempt = self.reacquire_counts[label]
        if attempt >= 3:
            self.resume_search(label,'짧은 회전으로 다시 인식하지 못했어.')
            return
        delta = (.45,-.90,.45)[attempt]
        if attempt==0 and self.object_point is not None:
            desired = math.atan2(self.object_point[1]-robot[1],self.object_point[0]-robot[0])
            difference = math.atan2(math.sin(desired-robot[2]),math.cos(desired-robot[2]))
            if abs(difference) > .12:
                delta = max(-.75,min(.75,difference))
        self.target_label = label
        self.reacquire_counts[label] += 1
        self.stage = 'REACQUIRE'
        self.emit('카메라 재탐색을 시도할게.',label=label,attempt=attempt+1,angle=delta)
        self.start_sweep(time.monotonic(),delta,'reacquire')

    def verify_arrival(self, wall, robot):
        label = self.stage.split('_',1)[1].lower()
        if label=='door' and not self.key_collected:
            self.stop('열쇠 획득 기록이 없어 문을 열지 않을게.',2)
            return
        if label=='exit' and not (self.key_collected and self.door_open and self.crossed_door):
            self.stop('열쇠 획득·문 열림·문 통과가 확인되지 않았어.',2)
            return
        if self.verification is None:
            self.verification = VerificationWindow(wall,self.sim(),self.frame_counter)
        observed = self.recent(label)
        if observed is not None and observed[1] < self.verification.sim-.05:
            observed = None
        distance = math.dist(robot[:2],observed[0][:2]) if observed is not None and robot is not None else None
        ready = distance is not None and distance <= LIMITS[label][3]
        reason = '새 카메라 관측 대기' if observed is None else '로봇 좌표 대기' if robot is None else '거리 재확인'
        if ready and label=='door':
            ready = self.closed_door_in_front()
            reason = '문 방향과 라이다 관측 확인'
        if ready:
            self.object_point = tuple(observed[0])
            if label=='exit':
                self.stage = 'COMPLETE'
                self.emit('exit_marker 도착 확인 완료!',distance=distance,key_collected=True,
                          door_open=True,crossed_door=True,exit_reached=True)
                self.stop('미로 탈출 임무 완료.',0)
                return
            self.stage = 'PICKING' if label=='key' else 'OPENING'
            if label=='key':
                self.operation = self.worker.submit(self.actuator.collect_key)
            else:
                self.door_point = tuple(observed[0])
                self.operation = self.worker.submit(self.actuator.lift,'red_door',self.key_collected)
            self.switch_deadline = wall+10
            self.emit('열쇠 획득을 요청했어.' if label=='key' else '열쇠를 사용해 문 열기를 요청했어.')
            return
        if wall-self.last_report > 3:
            self.last_report = wall
            self.emit('물체 도착 재확인',label=label,reason=reason,distance=distance,
                      detection_age=None if observed is None else round(self.sim()-observed[1],3),
                      robot_tf=robot is not None,new_frames=self.frame_counter-self.verification.frames,
                      sim_elapsed=round(self.sim()-self.verification.sim,2),key_collected=self.key_collected)
        if self.verification.enough(self.sim(),self.frame_counter) or self.verification.expired(wall,self.sim()):
            if robot is None:
                if self.verification.expired(wall,self.sim()):
                    self.stop('로봇 좌표를 확인하지 못해 재접근할 수 없어.',2)
                return
            if observed is not None and distance is not None and distance > LIMITS[label][3]:
                if self.target is not None:
                    self.rejected[label].append((self.target[:2],self.sim()+45))
                self.stage = 'REPLAN_'+label.upper()
                self.state.finish('SELECT')
                self.next_approach_plan = wall
                self.emit('물체와 아직 멀어서 더 가까운 접근 후보를 찾을게.',label=label,distance=distance)
            else:
                self.begin_reacquire(label,robot)
