import math
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET
import yaml
from maze_escape.grid import (Grid, Exclusions, angle_difference, candidates,
                         destination_clear, frontiers, groups, path_acceptable,
                         probe_goals, reachable)
from maze_escape.action_state import RequestState

ROOT = Path(__file__).resolve().parents[1]


def room():
    data = tuple(0 if 3 <= r < 17 and 3 <= c < 17 else -1
                 for r in range(20) for c in range(20))
    return Grid(20,20,0.1,0,0,0,data)


class GeometryTests(unittest.TestCase):
    def test_rotated_origin_round_trip(self):
        grid = Grid(10,10,.05,-8,4,1.2,(0,)*100)
        for cell in ((0,0),(5,3),(9,9)):
            self.assertEqual(grid.cell(*grid.world(cell)),cell)

    def test_negative_coordinate_is_outside(self):
        grid = room()
        self.assertEqual(grid.cell(-.001,0),(0,-1))
        self.assertIsNone(grid.value((0,-1)))

    def test_invalid_dimensions_rejected(self):
        with self.assertRaises(ValueError):
            Grid(3,3,.05,0,0,0,(0,))

    def test_frontier_is_free_not_unknown_or_wall(self):
        grid = Grid(3,2,1,0,0,0,(-1,0,100,-1,0,100))
        self.assertEqual(frontiers(grid),{(0,1),(1,1)})

    def test_grouping_accepts_diagonal_and_filters_specks(self):
        self.assertEqual(len(groups({(0,0),(1,1),(2,2),(9,9)},minimum=3)),1)

    def test_start_is_not_rejected_by_unknown_under_robot(self):
        grid = Grid(3,3,.05,0,0,0,(-1,0,-1,-1,0,-1,-1,0,-1))
        self.assertEqual(len(reachable(grid,(1,1))),3)

    def test_disconnected_region_not_reachable(self):
        grid = Grid(3,2,.1,0,0,0,(0,100,0,0,100,0))
        self.assertNotIn((0,2),reachable(grid,(0,0)))

    def test_destination_requires_clearance(self):
        grid = room()
        self.assertFalse(destination_clear(grid,(3,5)))
        self.assertTrue(destination_clear(grid,(10,10)))

    def test_one_frontier_can_offer_multiple_approaches(self):
        grid = room()
        choices,count = candidates(grid,grid.world((10,10)))
        self.assertEqual(count,1)
        self.assertEqual(len(choices),3)
        self.assertTrue(all(destination_clear(grid,grid.cell(c.x,c.y)) for c in choices))

    def test_failed_points_are_excluded(self):
        grid = room()
        robot = grid.world((10,10))
        choices,_ = candidates(grid,robot)
        point = choices[0].x,choices[0].y
        changed,_ = candidates(grid,robot,excluded=[point])
        self.assertTrue(changed)
        self.assertTrue(all(math.hypot(c.x-point[0],c.y-point[1]) > .4 for c in changed))

    def test_exclusions_expire_in_simulation_time(self):
        exclusions = Exclusions()
        exclusions.add((1,2),10,False)
        self.assertEqual(exclusions.active(129),[(1,2)])
        self.assertEqual(exclusions.active(130),[])

    def test_path_must_reach_goal_and_be_finite(self):
        self.assertFalse(path_acceptable([], (1,1)))
        self.assertFalse(path_acceptable([(0,0)], (1,1)))
        self.assertFalse(path_acceptable([(math.nan,0),(1,1)], (1,1)))
        self.assertTrue(path_acceptable([(0,0),(.95,1)], (1,1)))

    def test_probe_returns_to_start_with_reversed_heading(self):
        first,second = probe_goals((4,2,math.pi/2))
        self.assertAlmostEqual(first[0],4)
        self.assertAlmostEqual(first[1],2.6)
        self.assertEqual(second[:2],(4,2))
        self.assertAlmostEqual(abs(angle_difference(second[2],math.pi/2)),math.pi)


class StateTests(unittest.TestCase):
    def test_stale_request_cannot_change_new_goal(self):
        state = RequestState()
        old = state.begin('NAV_REQUEST',100,15)
        state.finish('SELECT')
        new = state.begin('NAV_REQUEST',102,15)
        self.assertFalse(state.current(old))
        self.assertTrue(state.current(new,'NAV_REQUEST'))

    def test_deadline_uses_explicit_wall_clock(self):
        state = RequestState()
        state.begin('CANCELLING',100,5)
        self.assertFalse(state.expired(104.9))
        self.assertTrue(state.expired(105))
        state.finish('SELECT')
        self.assertFalse(state.expired(999))


class ConfigurationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nav = yaml.safe_load((ROOT/'config/nav2.yaml').read_text())
        cls.slam = yaml.safe_load((ROOT/'config/slam.yaml').read_text())['slam_toolbox']['ros__parameters']

    def test_turns_are_processed_by_slam(self):
        self.assertTrue(self.slam['check_min_dist_and_heading_precisely'])
        self.assertLessEqual(self.slam['minimum_travel_heading'],.1)
        self.assertEqual(self.slam['max_laser_range'],3.5)

    def test_collision_protection_kept_with_timeout_margin(self):
        c = self.nav['collision_monitor']['ros__parameters']
        self.assertTrue(c['scan']['enabled'])
        self.assertGreater(c['scan']['source_timeout'],.2)
        self.assertTrue(c['PolygonStop']['enabled'])

    def test_velocity_limits_and_stopped_threshold_are_consistent(self):
        c = self.nav['controller_server']['ros__parameters']
        f = c['FollowPath']
        s = self.nav['velocity_smoother']['ros__parameters']
        self.assertEqual(f['max_vel_x'],f['max_speed_xy'])
        self.assertEqual(s['max_velocity'][0],f['max_vel_x'])
        self.assertEqual(s['max_velocity'][2],f['max_vel_theta'])
        self.assertLess(f['trans_stopped_velocity'],f['max_vel_x'])
        self.assertEqual(c['goal_checker']['xy_goal_tolerance'],f['xy_goal_tolerance'])

    def test_path_alignment_controller_wraps_dwb(self):
        f = self.nav['controller_server']['ros__parameters']['FollowPath']
        self.assertEqual(f['plugin'],'nav2_rotation_shim_controller::RotationShimController')
        self.assertEqual(f['primary_controller'],'dwb_core::DWBLocalPlanner')
        self.assertLess(f['angular_disengage_threshold'],f['angular_dist_threshold'])

    def test_all_parameter_sections_use_simulation_time(self):
        def visit(value):
            if isinstance(value,dict):
                for key,child in value.items():
                    if key == 'ros__parameters':
                        self.assertIs(child['use_sim_time'],True)
                    else:
                        visit(child)
        visit(self.nav)

    def test_maze_is_valid_and_probe_corridor_is_clear(self):
        root = ET.parse(ROOT/'worlds/escape_room.world')
        self.assertEqual(root.find('world').attrib['name'],'maze')
        for model in root.findall('./world/model'):
            size = model.findtext('.//collision/geometry/box/size')
            pose = model.findtext('pose')
            if size is None or pose is None:
                continue
            x,y = map(float,pose.split()[:2])
            sx,sy = map(float,size.split()[:2])
            # Entire 0.6m probe corridor plus robot radius at the initial spawn.
            overlaps = (x-sx/2-.12 < -1.4 and x+sx/2+.12 > -2.0
                        and y-sy/2-.12 < -2.5 < y+sy/2+.12)
            self.assertFalse(overlaps,model.attrib['name'])


if __name__ == '__main__':
    unittest.main()
