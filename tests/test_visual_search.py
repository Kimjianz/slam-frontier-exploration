import math
from types import SimpleNamespace as NS
import unittest
from unittest.mock import MagicMock

from maze_escape.grid import Grid,frontiers
from maze_escape.visual_search import ObjectMemory,free_line,inspection_candidates
import test_camera as camera_tests


def room(width=72,height=60):
    return Grid(width,height,.05,0,0,0,(0,)*(width*height))


class MemoryTests(unittest.TestCase):
    def test_door_seen_before_key_is_remembered_for_later_revisit(self):
        memory = ObjectMemory()
        memory.remember('door',(2,1,.3),(.8,1,0),'map',10)
        memory.remember('key',(0,2,.2),(0,1,1),'map',40)
        found = memory.candidates('door','map',100,set())
        self.assertEqual(len(found),1)
        self.assertEqual(found[0].pose,(.8,1,0))
        self.assertEqual(found[0].point,(2,1,.3))

    def test_same_observation_site_is_updated_without_an_unbounded_queue(self):
        memory = ObjectMemory(maximum=2)
        memory.remember('door',(2,1,.3),(.8,1,0),'map',10)
        memory.remember('door',(2.05,1,.3),(.85,1,0),'map',11)
        self.assertEqual(len(memory.views['door']),1)
        memory.remember('door',(2,1,.3),(.8,2,0),'map',12)
        memory.remember('door',(2,1,.3),(.8,3,0),'map',13)
        self.assertEqual(len(memory.views['door']),2)

    def test_attempted_expired_and_other_frame_memories_are_not_reused(self):
        memory = ObjectMemory(lifetime=20)
        memory.remember('door',(2,1,.3),(.8,1,0),'map',10)
        view = memory.candidates('door','map',11,set())[0]
        self.assertEqual(memory.candidates('door','map',11,{view.uid}),[])
        self.assertEqual(memory.candidates('door','other',11,set()),[])
        self.assertEqual(memory.candidates('door','map',31,set()),[])

    def test_nonfinite_detection_is_not_stored(self):
        memory = ObjectMemory()
        memory.remember('door',(math.nan,1,.3),(.8,1,0),'map',10)
        self.assertEqual(memory.views,{})

    def test_older_frame_cannot_overwrite_a_newer_observation(self):
        memory = ObjectMemory()
        memory.remember('door',(2,1,.3),(.8,1,0),'map',20)
        memory.remember('door',(2.1,1,.3),(.9,1,0),'map',19)
        self.assertEqual(memory.views['door'][0].seen,20)
        self.assertEqual(memory.views['door'][0].pose,(.8,1,0))


class ViewpointTests(unittest.TestCase):
    def test_fully_mapped_room_without_frontiers_still_has_camera_viewpoints(self):
        grid = room()
        self.assertEqual(frontiers(grid),set())
        goals,info = inspection_candidates(grid,(.6,.6,0),completed=[(.6,.6)])
        self.assertTrue(goals)
        self.assertGreater(info['available_viewpoints'],0)
        self.assertTrue(all(grid.value(grid.cell(*g[:2]))==0 for g in goals))
        self.assertTrue(all(math.dist(g[:2],(.6,.6))>=1.02 for g in goals))

    def test_unknown_and_disconnected_areas_are_not_inspection_goals(self):
        grid = room()
        values = list(grid.data)
        for r in range(grid.height):
            values[r*grid.width+30] = 100
            for c in range(60,grid.width):
                values[r*grid.width+c] = -1
        grid = Grid(grid.width,grid.height,.05,0,0,0,tuple(values))
        goals,_ = inspection_candidates(grid,(.6,.6,0))
        self.assertTrue(goals)
        self.assertTrue(all(g[0]<1.5 for g in goals))

    def test_wall_occlusion_prevents_a_nearby_room_being_marked_observed(self):
        grid = room()
        values = list(grid.data)
        for r in range(45):
            values[r*grid.width+30] = 100
        grid = Grid(grid.width,grid.height,.05,0,0,0,tuple(values))
        visited = (1.2,1.8)
        self.assertFalse(free_line(grid,visited,(1.8,1.8)))
        goals,_ = inspection_candidates(grid,(.6,.6,0),completed=[visited],limit=50)
        self.assertTrue(any(g[0]>1.5 and math.dist(g[:2],visited)<1.02 for g in goals))

    def test_completed_viewpoints_are_not_repeated_forever(self):
        grid = room(48,48)
        completed = [(.6,.6),(.6,1.8),(1.8,.6),(1.8,1.8)]
        goals,_ = inspection_candidates(grid,(.6,.6,0),completed=completed)
        self.assertEqual(goals,[])

    def test_costmap_blocked_and_failed_viewpoints_are_filtered(self):
        grid = room()
        first,_ = inspection_candidates(grid,(.6,.6,0))
        other,_ = inspection_candidates(grid,(.6,.6,0),failed=[first[0][:2]])
        self.assertTrue(all(math.dist(g[:2],first[0][:2])>=.35 for g in other))
        cost = Grid(grid.width,grid.height,.05,0,0,0,(100,)*len(grid.data))
        self.assertEqual(inspection_candidates(grid,(.6,.6,0),costmap=cost)[0],[])


class SearchControlTests(unittest.TestCase):
    def node(self):
        node = camera_tests.ControllerTests().node()
        node.init_visual_search(4)
        node.grid,node.costmap = room(),None
        node.stage,node.mode,node.reached = 'SEARCH_DOOR','explore',5
        node.key_collected = True
        node.robot = lambda:(.6,.6,0)
        node.recent = lambda label:None
        node.state.finish('SELECT')
        node.maximum = 40
        node.sweep_pending = False
        node.no_goal_since = None
        node.next_approach_plan = 0
        return node

    def test_memory_visit_has_priority_and_does_not_open_door_from_old_data(self):
        node = self.node()
        node.object_memory.remember('door',(2,1.5,.3),(.8,1.5,0),'map',1)
        node.actuator = MagicMock()
        self.assertTrue(node.visual_search_tick(100,10,node.robot()))
        self.assertEqual(node.stage,'REVISIT_MOVE')
        self.assertEqual(node.inspection_current['source'],'memory')
        self.assertEqual(node.probes[0][:2],(.8,1.5))
        self.assertTrue(node.key_collected)
        node.actuator.lift.assert_not_called()

    def test_frontier_absence_switches_to_a_new_observation_position(self):
        node = self.node()
        node.completed_views['door'] = [(.6,.6)]
        self.assertTrue(node.handle_no_frontier(node.robot(),10))
        self.assertEqual(node.stage,'REVISIT_MOVE')
        self.assertEqual(node.inspection_current['source'],'known_space')
        self.assertGreater(math.dist(node.probes[0][:2],(.6,.6)),1.0)
        node.stop.assert_not_called()

    def test_base_frontier_stop_is_intercepted_by_camera_mission(self):
        node = self.node()
        node.last_sim,node.started,node.next_frontiers = 10,100,0
        node.update_grid = MagicMock()
        node.navigation_ready = lambda wall:True
        node.select_candidates = lambda robot,sim:[]
        camera_tests.mission.Mission.tick(node)
        self.assertEqual(node.stage,'REVISIT_MOVE')
        node.stop.assert_not_called()

    def test_frontier_goal_limit_also_preserves_an_object_search_phase(self):
        node = self.node()
        node.reached = node.maximum
        node.last_sim,node.started,node.next_frontiers = 10,100,0
        node.update_grid = MagicMock()
        node.navigation_ready = lambda wall:True
        camera_tests.mission.Mission.tick(node)
        self.assertEqual(node.stage,'REVISIT_MOVE')
        self.assertEqual(node.inspection_current['saved_frontier_count'],40)
        node.stop.assert_not_called()

    def test_failed_revisit_goal_tries_other_sites_instead_of_aborting(self):
        node = self.node()
        node.begin_revisit('door',(1.8,.6,0),'known_space')
        node.failed('Nav2가 이 관측 위치의 경로를 거절했어.')
        self.assertEqual(node.stage,'SEARCH_DOOR')
        self.assertEqual(node.reached,5)
        self.assertEqual(node.failed_views['door'],[(1.8,.6)])
        node.stop.assert_not_called()

    def test_fresh_camera_detection_during_inspection_cancels_before_new_goal(self):
        node = self.node()
        node.begin_revisit('door',(1.8,.6,0),'known_space')
        node.stage = 'REVISIT_SCAN'
        generation = node.state.begin('SWEEP_ACTIVE',100,360)
        node.recent = lambda label:((2,.6,.3),10,1.4)
        node.choose_approach = MagicMock(return_value=(1.4,.6,0))
        node.goal_handle,node.nav_client = MagicMock(),MagicMock()
        self.assertTrue(node.visual_search_tick(100,10,node.robot()))
        self.assertEqual(node.stage,'SWITCHING')
        node.goal_handle.cancel_goal_async.assert_called_once()
        node.nav_client.send_goal_async.assert_not_called()
        future = MagicMock()
        future.result.return_value.status = 5
        node.sweep_result(future,generation)
        self.assertEqual(node.stage,'APPROACH_DOOR')
        self.assertTrue(node.key_collected)

    def test_scan_requires_camera_frames_and_directions_before_marking_a_site(self):
        node = self.node()
        node.begin_revisit('door',(1.8,.6,0),'known_space')
        node.start_sweep = MagicMock()
        node.begin_inspection_scan()
        node.state.begin('SWEEP_ACTIVE',100,360)
        for i in range(8):
            node.frame_counter += 1
            node.record_camera_view((1.8,.6,(i+.1)*math.pi/4),10+i*.2)
        node.inspection_rotation_finished()
        node.frame_counter += 6
        node.sim = lambda:11.2
        node.visual_search_tick(101.2,11.2,node.robot())
        self.assertEqual(node.stage,'SEARCH_DOOR')
        self.assertEqual(node.completed_views['door'],[(.6,.6)])
        self.assertEqual(node.reached,5)
        node.stop.assert_not_called()

    def test_insufficient_scan_does_not_claim_a_completed_observation(self):
        node = self.node()
        node.begin_revisit('door',(1.8,.6,0),'known_space')
        node.start_sweep = MagicMock()
        node.begin_inspection_scan()
        node.inspection_rotation_finished()
        node.frame_counter += 6
        node.sim = lambda:11.2
        node.visual_search_tick(101.2,11.2,node.robot())
        self.assertEqual(node.stage,'REVISIT_SCAN')
        self.assertEqual(node.revisit_scan_attempts,2)
        self.assertEqual(node.completed_views['door'],[])

    def test_missing_pose_before_retry_waits_without_starting_another_spin(self):
        node = self.node()
        node.begin_revisit('door',(1.8,.6,0),'known_space')
        node.start_sweep = MagicMock()
        node.revisit_scan_attempts,node.revisit_frames,node.revisit_bins = 1,8,{0,1}
        node.robot = lambda:None
        node.begin_inspection_scan(retry=True)
        self.assertEqual(node.stage,'REVISIT_WAIT_POSE')
        node.start_sweep.assert_not_called()
        node.robot = lambda:(.6,.6,0)
        node.visual_search_tick(101,10,node.robot())
        self.assertEqual(node.stage,'REVISIT_SCAN')
        self.assertEqual(node.revisit_scan_attempts,2)

    def test_inspection_budget_is_a_failure_not_mission_success(self):
        node = self.node()
        node.inspection_used['door'] = node.inspection_limit
        node.handle_no_frontier(node.robot(),10)
        self.assertEqual(node.stop.call_args.args[1],2)
        record = [call.kwargs for call in node.emit.call_args_list if call.kwargs.get('error')]
        self.assertEqual(record[0]['error'],'INSPECTION_BUDGET_EXHAUSTED')
        self.assertFalse(record[0]['coverage_claim'])

    def test_unreachable_memory_is_not_used_as_an_approach(self):
        node = self.node()
        node.object_memory.remember('door',(2,1,.3),(-5,1,0),'map',1)
        self.assertFalse(node.try_memory_visit('door',node.robot(),10))
        self.assertEqual(node.stage,'SEARCH_DOOR')
        self.assertEqual(len(node.memory_tried['door']),1)

    def test_all_planned_views_inspected_without_detection_is_not_success(self):
        node = self.node()
        node.grid = room(48,48)
        node.completed_views['door'] = [(.6,.6),(.6,1.8),(1.8,.6),(1.8,1.8)]
        node.handle_no_frontier(node.robot(),10)
        node.stop.assert_not_called()
        node.handle_no_frontier(node.robot(),14)
        self.assertEqual(node.stop.call_args.args[1],2)
        records = [call.kwargs for call in node.emit.call_args_list if call.kwargs.get('error')]
        self.assertEqual(records[0]['error'],'OBJECT_NOT_FOUND_IN_INSPECTED_VIEWS')
        self.assertFalse(records[0]['coverage_claim'])

    def test_seen_but_unapproachable_object_is_not_logged_as_never_found(self):
        node = self.node()
        node.grid = room(48,48)
        node.completed_views['door'] = [(.6,.6),(.6,1.8),(1.8,.6),(1.8,1.8)]
        node.seen_during_revisit['door'] = True
        node.handle_no_frontier(node.robot(),10)
        node.handle_no_frontier(node.robot(),14)
        records = [call.kwargs for call in node.emit.call_args_list if call.kwargs.get('error')]
        self.assertEqual(records[0]['error'],'OBJECT_NOT_RECONFIRMED_OR_UNREACHABLE')
        self.assertTrue(records[0]['seen_before'])


if __name__=='__main__':
    unittest.main()
