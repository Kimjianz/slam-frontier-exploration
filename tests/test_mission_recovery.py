import math
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from maze_escape.approach_planner import (VerificationWindow,crossed_plane,
                                        passage_distance,select_approaches)
from maze_escape.grid import Grid
from maze_escape.vision import detect
import test_camera as camera_tests
from test_camera import controller


def space(width=70,height=50):
    return Grid(width,height,.05,0,0,0,(0,)*(width*height))


def changed(grid, replacements):
    values = list(grid.data)
    for cell,value in replacements.items():
        values[cell[0]*grid.width+cell[1]] = value
    return Grid(grid.width,grid.height,grid.resolution,grid.origin_x,grid.origin_y,grid.origin_yaw,tuple(values))


class ApproachTests(unittest.TestCase):
    def test_blocked_original_approach_uses_a_different_known_free_goal(self):
        grid = space()
        robot,target = (.5,1.25,0),(2,1.25,.2)
        original = (1.55,1.25)
        grid = changed(grid,{grid.cell(*original):100})
        goals,info = select_approaches(grid,robot,target,'key')
        self.assertTrue(goals)
        for goal in goals:
            self.assertEqual(grid.value(grid.cell(*goal.pose[:2])),0)
            self.assertGreater(math.dist(goal.pose[:2],original),.12)
        self.assertEqual(goals[0].kind,'approach')

    def test_unknown_original_approach_is_never_sent(self):
        grid = space()
        blocked = grid.cell(1.55,1.25)
        grid = changed(grid,{blocked:-1})
        goals,_ = select_approaches(grid,(.5,1.25,0),(2,1.25,.2),'key')
        self.assertTrue(goals)
        self.assertTrue(all(grid.value(grid.cell(*g.pose[:2]))==0 for g in goals))

    def test_far_target_uses_an_observation_step_inside_mapped_space(self):
        grid = space()
        grid = changed(grid,{(r,c):-1 for r in range(grid.height) for c in range(35,grid.width)})
        robot,target = (.5,1.25,0),(3.2,1.25,.2)
        goals,_ = select_approaches(grid,robot,target,'key')
        self.assertTrue(goals)
        self.assertEqual(goals[0].kind,'observe')
        self.assertLessEqual(math.dist(goals[0].pose[:2],robot[:2]),1.0)
        self.assertLess(math.dist(goals[0].pose[:2],target[:2]),math.dist(robot[:2],target[:2]))
        self.assertEqual(grid.value(grid.cell(*goals[0].pose[:2])),0)

    def test_candidates_do_not_cross_disconnected_walls(self):
        grid = space()
        grid = changed(grid,{(r,30):100 for r in range(grid.height)})
        goals,_ = select_approaches(grid,(.5,1.25,0),(2.5,1.25,.2),'key')
        self.assertTrue(all(grid.cell(*g.pose[:2])[1]<30 for g in goals))
        self.assertFalse(any(g.kind=='approach' for g in goals))

    def test_failed_and_inflated_candidates_are_not_reused(self):
        grid = space()
        first,_ = select_approaches(grid,(.5,1.25,0),(2,1.25,.2),'key')
        old = first[0].pose[:2]
        costmap = changed(grid,{grid.cell(*old):100})
        new,_ = select_approaches(grid,(.5,1.25,0),(2,1.25,.2),'key',[old],costmap)
        self.assertTrue(new)
        self.assertTrue(all(math.dist(old,g.pose[:2])>=.18 for g in new))

    def test_unmapped_robot_cell_is_reported_without_inventing_free_cells(self):
        grid = changed(space(),{(25,10):-1})
        goals,info = select_approaches(grid,(.525,1.275,0),(2,1.25,.2),'key')
        self.assertEqual(goals,[])
        self.assertEqual(info['robot_value'],-1)

    def test_verification_uses_new_frames_and_sim_time_with_a_wall_limit(self):
        window = VerificationWindow(100,10,20)
        self.assertFalse(window.enough(10.2,21))
        self.assertFalse(window.expired(120,10.2))
        self.assertTrue(window.enough(11.2,26))
        self.assertTrue(window.expired(221,10.2))


class RecoveryTests(unittest.TestCase):
    def node(self):
        node = camera_tests.ControllerTests().node()
        node.grid = space()
        node.costmap = None
        node.next_approach_plan = 0
        return node

    def test_map_validation_failure_replans_instead_of_ending_mission(self):
        node = self.node()
        node.stage = 'APPROACH_KEY'
        node.target_label = 'key'
        node.target = (1.55,1.25,0)
        node.grid = changed(node.grid,{node.grid.cell(*node.target[:2]):100})
        node.recent = lambda label:((2,1.25,.2),10,1.5)
        node.failed('목표 중심이 확인된 빈 공간이 아니야.')
        self.assertEqual(node.stage,'REPLAN_KEY')
        self.assertEqual(node.replan_counts['key'],1)
        node.next_approach_plan = 0
        node.replan_target('key',(.5,1.25,0),time.monotonic())
        self.assertEqual(node.stage,'APPROACH_KEY')
        self.assertNotEqual(node.probes[0][:2],node.target[:2])
        self.assertEqual(node.grid.value(node.grid.cell(*node.probes[0][:2])),0)
        node.stop.assert_not_called()

    def test_nav2_path_rejection_tries_another_candidate(self):
        node = self.node()
        node.stage,node.target_label = 'APPROACH_KEY','key'
        node.target = (1.5,1.25,0)
        g = node.state.begin('PLAN_RESULT',1,15)
        future = MagicMock()
        future.result.return_value.status = 6
        future.result.return_value.result.path.poses = []
        future.result.return_value.result.path.header.frame_id = 'map'
        node.plan_result(future,g)
        self.assertEqual(node.stage,'REPLAN_KEY')
        node.stop.assert_not_called()

    def test_lost_door_after_arrival_starts_a_short_reacquisition(self):
        node = self.node()
        node.stage = 'VERIFY_DOOR'
        node.key_collected = True
        node.verification = VerificationWindow(100,10,0)
        node.sim = lambda:11.5
        node.frame_counter = 7
        node.recent = lambda label:None
        node.object_point = (1,0,.3)
        node.start_sweep = MagicMock()
        node.verify_arrival(120,(0,0,0))
        self.assertEqual(node.stage,'REACQUIRE')
        self.assertEqual(node.reacquire_counts['door'],1)
        self.assertEqual(node.start_sweep.call_args.args[2],'reacquire')
        node.stop.assert_not_called()

    def test_far_door_replans_without_relaxing_the_distance_limit(self):
        node = self.node()
        node.stage = 'VERIFY_DOOR'
        node.key_collected = True
        node.verification = VerificationWindow(100,10,0)
        node.sim = lambda:11.5
        node.frame_counter = 7
        node.recent = lambda label:((2,0,.3),11.5,2)
        node.verify_arrival(120,(0,0,0))
        self.assertEqual(node.stage,'REPLAN_DOOR')
        node.stop.assert_not_called()

    def test_reacquisition_limit_resumes_exploration_and_keeps_the_key(self):
        node = self.node()
        node.key_collected = True
        node.reacquire_counts['door'] = 3
        node.begin_reacquire('door',(0,0,0))
        self.assertEqual(node.stage,'SEARCH_DOOR')
        self.assertTrue(node.key_collected)
        node.stop.assert_not_called()

    def test_exit_cannot_complete_before_door_crossing(self):
        node = self.node()
        node.stage = 'VERIFY_EXIT'
        node.key_collected = node.door_open = True
        node.verify_arrival(10,(0,0,0))
        node.stop.assert_called_once()
        self.assertEqual(node.stop.call_args.args[1],2)

    def test_exit_is_completed_only_after_close_camera_confirmation(self):
        node = self.node()
        node.stage = 'VERIFY_EXIT'
        node.key_collected = node.door_open = node.crossed_door = True
        node.recent = lambda label:((.6,0,.3),10,.6)
        node.verify_arrival(10,(0,0,0))
        self.assertEqual(node.stage,'COMPLETE')
        self.assertEqual(node.stop.call_args.args[1],0)
        self.assertTrue(node.emit.call_args.kwargs['exit_reached'])


class PassageTests(unittest.TestCase):
    def node(self):
        node = RecoveryTests().node()
        node.key_collected = True
        node.stage = 'VERIFY_OPEN'
        node.door_point = (.65,1.25,.3)
        node.robot = lambda:(0,1.25,0)
        node.open_stamp = 10
        node.open_window = VerificationWindow(100,10,0)
        node.clear_stamps = []
        return node

    def scan(self, t, distance=3):
        return NS(header=NS(stamp=NS(sec=int(t),nanosec=int((t-int(t))*1e9))),
                  ranges=[distance]*360,angle_min=-math.pi,angle_increment=2*math.pi/360,
                  range_min=.1,range_max=3.5)

    def test_open_door_does_not_end_the_mission(self):
        node = self.node()
        node.begin_passage_clear = MagicMock()
        for t in (11,11.2,11.4):
            node.scan_message = self.scan(t)
            node.verify_open(101+t-11,t)
        self.assertTrue(node.door_open)
        self.assertNotEqual(node.stage,'COMPLETE')
        node.begin_passage_clear.assert_called_once()
        node.stop.assert_not_called()

    def test_blocked_passage_never_starts_crossing(self):
        node = self.node()
        node.begin_passage_clear = MagicMock()
        for t in (11,11.2,11.4):
            node.scan_message = self.scan(t,.7)
            node.verify_open(101+t-11,t)
        self.assertFalse(node.door_open)
        node.begin_passage_clear.assert_not_called()

    def test_crossing_waits_for_fresh_local_costmap(self):
        node = self.node()
        node.stage = 'WAIT_PASSAGE'
        node.clear_sim = 10
        node.cross_distance = 1.3
        node.last_local_costmap = (9,100)
        node.passage_window = VerificationWindow(100,10,0)
        node.start_crossing = MagicMock()
        for t in (11,11.2,11.4):
            node.scan_message = self.scan(t)
            node.passage_tick(101,t,(0,1.25,0))
        node.start_crossing.assert_not_called()
        node.last_local_costmap = (11.4,101)
        node.passage_tick(101,11.4,(0,1.25,0))
        node.start_crossing.assert_called_once()

    def test_successful_crossing_enters_exit_search_without_claiming_completion(self):
        node = self.node()
        node.door_open = True
        node.stage = 'VERIFY_CROSS'
        node.cross_start = (0,1.25,0)
        node.passage_window = VerificationWindow(100,10,0)
        node.passage_tick(101,11,(1.35,1.25,0))
        self.assertEqual(node.stage,'SEARCH_EXIT')
        self.assertTrue(node.crossed_door)
        node.stop.assert_not_called()

    def test_crossing_geometry_requires_reaching_the_far_side(self):
        self.assertAlmostEqual(passage_distance((0,0,0),(.65,0,.3)),1.3)
        self.assertFalse(crossed_plane((.4,0,0),(0,0,0),(.65,0,.3)))
        self.assertTrue(crossed_plane((1.3,0,0),(0,0,0),(.65,0,.3)))

    def test_crossing_uses_bounded_nav2_drive_on_heading(self):
        node = self.node()
        node.drive_client = MagicMock()
        node.drive_client.server_is_ready.return_value = True
        node.cross_distance = 1.3
        goal = NS(target=NS(x=0,y=0,z=0),speed=0,time_allowance=NS(sec=0))
        action = NS(Goal=lambda:goal)
        with patch.dict(node.start_crossing.__func__.__globals__,{'DriveOnHeading':action}):
            node.start_crossing((0,1.25,0),100,10)
        self.assertEqual(node.stage,'CROSSING')
        self.assertEqual(node.state.phase,'CROSS_REQUEST')
        self.assertEqual(goal.target.x,1.3)
        self.assertEqual(goal.speed,.06)
        self.assertEqual(goal.time_allowance.sec,120)
        self.assertFalse(hasattr(goal,'disable_collision_checks'))

    def test_drive_success_still_requires_crossing_position_confirmation(self):
        node = self.node()
        node.stage = 'CROSSING'
        generation = node.state.begin('CROSS_ACTIVE',100,600)
        future = MagicMock()
        future.result.return_value.status = 4
        node.cross_result(future,generation)
        self.assertEqual(node.stage,'VERIFY_CROSS')
        self.assertFalse(node.crossed_door)
        node.stop.assert_not_called()

    def test_exit_green_is_distinct_from_key_and_door_colours(self):
        rgb = np.zeros((80,120,3),np.uint8)
        rgb[20:60,5:30] = (255,255,0)
        rgb[20:60,45:70] = (255,0,0)
        rgb[20:60,85:110] = (0,255,0)
        found = detect(rgb,np.full((80,120),2,np.float32))
        self.assertEqual({d.label for d in found},{'key','door','exit'})


if __name__ == '__main__':
    unittest.main()
