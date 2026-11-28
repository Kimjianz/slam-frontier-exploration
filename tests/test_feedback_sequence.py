"""Reproduce the team feedback: an already mapped maze, old door view, and a key."""
from concurrent.futures import Future
from types import SimpleNamespace as NS
import time
import unittest
from unittest.mock import MagicMock
import numpy as np

from maze_escape.grid import frontiers
from maze_escape.vision import ObservationGate
import test_camera as camera_tests
import test_mission_sequence as sequence
import test_visual_search as visual_tests


class FeedbackSequenceTests(unittest.TestCase):
    def test_prior_door_view_is_revisited_and_reconfirmed_after_key_without_frontiers(self):
        node = camera_tests.ControllerTests().node()
        node.init_visual_search()
        node.grid,node.costmap = visual_tests.room(),None
        self.assertEqual(frontiers(node.grid),set())
        node.stage,node.mode,node.reached = 'SEARCH_KEY','explore',3
        node.state.finish('SELECT')
        clock,pose = [10.4],[(.6,.6,0)]
        node.sim,node.robot = lambda:clock[0],lambda:pose[0]
        node.last_pair = -1
        node.gate,node.observations = ObservationGate(),{}
        node.info = NS(width=80,height=60,k=(100,0,39.5,0,100,29.5,0,0,1))
        node.tf,node.image_pub,node.actuator = MagicMock(),MagicMock(),MagicMock()
        node.tf.lookup_transform.return_value.transform = NS(translation=NS(x=.68,y=.6,z=.2),
                                                             rotation=NS(x=-.5,y=.5,z=-.5,w=.5))
        rgb,depth = camera_tests.VisionTests().image()
        def observe(t):
            clock[0] = t
            node.rgb = camera_tests.frame(rgb,t)
            node.depth = NS(height=60,width=80,encoding='32FC1',is_bigendian=0,
                            step=80*4,data=depth.tobytes(),header=node.rgb.header)
            node.observe()
        for t in (10.4,10.6,10.8):
            observe(t)
        self.assertIn('door',node.object_memory.views)
        self.assertFalse(node.key_collected)

        # The key has been collected elsewhere; the old door view is no longer recent.
        node.key_collected = True
        node.stage = 'SEARCH_DOOR'
        pose[0],clock[0] = (1.8,1.8,0),25
        self.assertIsNone(node.recent('door'))
        node.next_approach_plan = 0
        self.assertTrue(node.visual_search_tick(time.monotonic(),25,node.robot()))
        self.assertEqual(node.stage,'REVISIT_MOVE')
        self.assertEqual(node.inspection_current['source'],'memory')
        self.assertEqual(node.probes[0][:2],(.6,.6))
        node.actuator.lift.assert_not_called()

        node.spin_client = sequence.Client()
        pose[0] = (.6,.6,0)
        generation = node.state.begin('NAVIGATING',100,240)
        arrived = Future()
        arrived.set_result(NS(status=4))
        node.nav_result(arrived,generation)
        self.assertEqual(node.stage,'REVISIT_SCAN')
        self.assertEqual(node.state.phase,'SWEEP_ACTIVE')
        old_handle = node.spin_client.handle
        old_handle.cancel_goal_async = MagicMock(wraps=old_handle.cancel_goal_async)

        # A fresh RGB-D observation, not the stored coordinates, authorizes approach.
        for t in (25.2,25.4,25.6):
            observe(t)
        self.assertIsNotNone(node.recent('door'))
        node.nav_client = MagicMock()
        node.visual_search_tick(time.monotonic(),25.6,node.robot())
        self.assertEqual(node.stage,'SWITCHING')
        old_handle.cancel_goal_async.assert_called_once()
        node.nav_client.send_goal_async.assert_not_called()
        old_handle.result.set_result(NS(status=5))
        self.assertEqual(node.stage,'APPROACH_DOOR')
        self.assertEqual(node.grid.value(node.grid.cell(*node.probes[0][:2])),0)
        self.assertTrue(node.key_collected)
        node.actuator.lift.assert_not_called()
        node.stop.assert_not_called()


if __name__=='__main__':
    unittest.main()
