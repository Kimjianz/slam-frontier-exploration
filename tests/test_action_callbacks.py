"""Exercise real mission callback methods with ROS transport replaced by doubles.

These tests verify cancellation/state ordering; they do not simulate Nav2 driving.
"""
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import MagicMock, patch
from maze_escape.action_state import RequestState


def load_callbacks():
    names = ('rclpy','action_msgs.msg','geometry_msgs.msg','lifecycle_msgs.srv',
             'nav2_msgs.action','nav_msgs.msg','rclpy.action','rclpy.clock',
             'rclpy.clock_type','rclpy.node','rclpy.parameter','rclpy.qos',
             'rclpy.signals','rclpy.time','std_msgs.msg','tf2_ros','visualization_msgs.msg')
    modules = {name:MagicMock() for name in names}
    for name in names:
        parent = name.split('.')[0]
        modules.setdefault(parent,MagicMock())
    class FakeNode:
        @property
        def handle(self):
            return 'reserved ROS node handle'

        @handle.setter
        def handle(self, value):
            raise AttributeError('handle cannot be modified after node creation')
    modules['rclpy.node'].Node = FakeNode
    modules['tf2_ros'].TransformException = Exception
    path = Path(__file__).resolve().parents[1]/'maze_escape/mission.py'
    spec = importlib.util.spec_from_file_location('maze_escape._callback_test',path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules,modules):
        spec.loader.exec_module(module)
    return module


mission = load_callbacks()


class CallbackTests(unittest.TestCase):
    def make_node(self, phase):
        node = mission.Mission.__new__(mission.Mission)
        node.state = RequestState()
        generation = node.state.begin(phase,100,15)
        node.stopping = node.done = False
        node.goal_handle = None
        node.result_future = None
        node.nav_client = MagicMock()
        node.emit = MagicMock()
        node.stop = MagicMock()
        node.cancel = MagicMock()
        return node,generation

    def test_late_acceptance_is_cancelled_without_changing_current_goal(self):
        node,g = self.make_node('NAV_REQUEST')
        node.state.finish('SELECT')
        current = object()
        node.goal_handle = current
        handle = MagicMock(accepted=True)
        future = MagicMock()
        future.result.return_value = handle
        node.nav_accepted(future,g)
        handle.cancel_goal_async.assert_called_once()
        self.assertIs(node.goal_handle,current)
        self.assertEqual(node.state.phase,'SELECT')

    def test_shutdown_does_not_send_navigation_from_completed_plan(self):
        node,g = self.make_node('PLAN_RESULT')
        node.stopping = True
        future = MagicMock()
        node.plan_result(future,g)
        future.result.assert_not_called()
        node.nav_client.send_goal_async.assert_not_called()

    def test_shutdown_cancels_a_late_accepted_navigation_goal(self):
        node,g = self.make_node('NAV_REQUEST')
        node.stopping = True
        handle = MagicMock(accepted=True)
        future = MagicMock()
        future.result.return_value = handle
        node.nav_accepted(future,g)
        handle.cancel_goal_async.assert_called_once()
        self.assertIs(node.goal_handle,handle)
        self.assertIsNotNone(node.result_future)
        self.assertEqual(node.state.phase,'NAV_REQUEST')

    def test_result_is_registered_before_missing_pose_triggers_cancel(self):
        node,g = self.make_node('NAV_REQUEST')
        node.robot = lambda:None
        handle = MagicMock(accepted=True)
        future = MagicMock()
        future.result.return_value = handle
        node.nav_accepted(future,g)
        self.assertIsNotNone(node.result_future)
        node.result_future.add_done_callback.assert_called_once()
        node.cancel.assert_called_once()
        node.stop.assert_not_called()

    def test_success_callback_from_old_goal_does_not_advance_new_goal(self):
        node,g = self.make_node('NAVIGATING')
        node.reached = 0
        node.state.finish('SELECT')
        future = MagicMock()
        node.nav_result(future,g)
        future.result.assert_not_called()
        self.assertEqual(node.reached,0)


if __name__ == '__main__':
    unittest.main()
