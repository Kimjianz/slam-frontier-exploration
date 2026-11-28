"""State integration with real geometry/logic and simulated ROS action responses."""
from concurrent.futures import Future
import math
from types import SimpleNamespace as NS
import unittest
from unittest.mock import MagicMock,patch
import test_mission_recovery as recovery


class Handle:
    accepted = True
    def __init__(self):
        self.result = Future()
    def get_result_async(self):
        return self.result
    def cancel_goal_async(self):
        future = Future()
        future.set_result(NS())
        return future


class Client:
    def __init__(self):
        self.goals = []
        self.handle = Handle()
    def server_is_ready(self):
        return True
    def send_goal_async(self, goal, **kwargs):
        self.goals.append(goal)
        future = Future()
        future.set_result(self.handle)
        return future


class FullMissionTests(unittest.TestCase):
    def test_key_removal_door_reacquisition_crossing_and_exit_sequence(self):
        node = recovery.RecoveryTests().node()
        node.grid = recovery.space(width=150)
        node.update_grid,node.observe = MagicMock(),MagicMock()
        node.started,node.last_sim,node.last_pair = 100,10,10
        node.camera_wall,node.camera_sim = 100,10
        node.no_goal_since = None
        node.visible_labels = set()
        node.clear_stamps = []
        node.wait_started = 100
        node.sweep_pending = False
        clock = [10.0]
        pose = [(.5,1.25,0)]
        memory = {}
        events = []
        node.sim = lambda:clock[0]
        node.robot = lambda:pose[0]
        node.recent = lambda label:memory.get(label)
        def completed(function,*args):
            future = Future()
            future.set_result(function(*args))
            return future
        node.worker = NS(submit=completed)
        def collect():
            events.append('key_removed')
            return 'red_key'
        def open_door(name, has_key):
            self.assertEqual(name,'red_door')
            self.assertTrue(has_key)
            events.append('door_opened')
            return name
        node.actuator = NS(collect_key=collect,lift=open_door)
        node.mode = 'explore'
        def nav_arrive(label, point):
            memory[label] = (point,clock[0],1)
            selected = node.choose_approach(label,pose[0],point)
            self.assertIsNotNone(selected)
            node.state.finish('SELECT')
            node.request_switch(label,selected)
            node.target = selected
            pose[0] = selected
            generation = node.state.begin('NAVIGATING',100,240)
            result = Future()
            result.set_result(NS(status=4))
            node.nav_result(result,generation)
            self.assertEqual(node.stage,'VERIFY_'+label.upper())

        with patch('time.monotonic',return_value=100):
            nav_arrive('key',(2,1.25,.2))
            node.verify_arrival(100,pose[0])
            self.assertEqual(node.stage,'PICKING')
            self.assertFalse(node.key_collected)
            node.poll_operation(100)
            self.assertEqual(node.stage,'VERIFY_PICKUP')
            for t in (10.2,10.4,10.6):
                clock[0] = node.last_pair = node.camera_sim = t
                node.frame_counter += 1
                if t==10.6:
                    with patch('time.monotonic',return_value=130):
                        node.camera_wall = 130
                        node.tick()
                else:
                    node.tick()
            self.assertTrue(node.key_collected)
            self.assertEqual(node.stage,'SEARCH_DOOR')

            nav_arrive('door',(3.2,1.25,.3))
            node.recent = lambda label:None if label=='door' else memory.get(label)
            clock[0] += 1.3
            node.frame_counter += 7
            node.start_sweep = MagicMock()
            node.verify_arrival(102,pose[0])
            self.assertEqual(node.stage,'REACQUIRE')
            self.assertTrue(node.key_collected)
            node.sweep_purpose = 'reacquire'
            generation = node.state.begin('SWEEP_ACTIVE',102,360)
            finished = Future()
            finished.set_result(NS(status=4))
            node.sweep_result(finished,generation)
            self.assertEqual(node.stage,'VERIFY_DOOR')
            node.recent = lambda label:memory.get(label)
            memory['door'] = ((3.2,1.25,.3),clock[0],.7)
            node.scan_message = recovery.PassageTests().scan(clock[0],.7)
            node.verify_arrival(103,pose[0])
            self.assertEqual(node.stage,'OPENING')
            node.poll_operation(103)
            self.assertEqual(node.stage,'VERIFY_OPEN')

            clear_future = Future()
            clear_future.set_result(NS())
            node.clear_local = NS(service_is_ready=lambda:True,call_async=lambda r:clear_future)
            for delta in (.2,.4,.6):
                t = node.open_stamp+delta
                node.scan_message = recovery.PassageTests().scan(t,3)
                clock[0] = t
                node.verify_open(104,t)
            self.assertTrue(node.door_open)
            self.assertEqual(node.stage,'CLEAR_PASSAGE')
            node.passage_tick(104,clock[0],pose[0])
            self.assertEqual(node.stage,'WAIT_PASSAGE')
            node.drive_client = Client()
            action = NS(Goal=lambda:NS(target=NS(x=0,y=0,z=0),speed=0,time_allowance=NS(sec=0)))
            with patch.dict(node.start_crossing.__func__.__globals__,{'DriveOnHeading':action}):
                for delta in (.2,.4,.6):
                    t = node.clear_sim+delta
                    clock[0] = t
                    node.scan_message = recovery.PassageTests().scan(t,3)
                    node.last_local_costmap = (t,105)
                    node.passage_tick(105,t,pose[0])
            self.assertEqual(node.stage,'CROSSING')
            self.assertEqual(node.state.phase,'CROSS_ACTIVE')
            goal = node.drive_client.goals[0]
            x,y,yaw = pose[0]
            pose[0] = (x+goal.target.x*math.cos(yaw),y+goal.target.x*math.sin(yaw),yaw)
            node.drive_client.handle.result.set_result(NS(status=4))
            self.assertEqual(node.stage,'VERIFY_CROSS')
            node.passage_tick(106,clock[0],pose[0])
            self.assertEqual(node.stage,'SEARCH_EXIT')
            self.assertTrue(node.crossed_door)
            node.stop.assert_not_called()

            nav_arrive('exit',(5.5,1.25,.3))
            node.verify_arrival(107,pose[0])
            self.assertEqual(node.stage,'COMPLETE')
            node.stop.assert_called_once_with('미로 탈출 임무 완료.',0)
            self.assertEqual(events,['key_removed','door_opened'])


if __name__ == '__main__':
    unittest.main()
