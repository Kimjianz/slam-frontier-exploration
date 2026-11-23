import importlib.util
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import MagicMock, patch
import xml.etree.ElementTree as ET
import numpy as np

from maze_escape.vision import (ObservationGate,approach,depth_array,detect,forward_clear,
                               optical_point,rgb_array,transform_point)
from maze_escape.actuator import SceneActuator
from maze_escape.camera_model import build
from maze_escape.action_state import RequestState
from test_action_callbacks import mission

ROOT = Path(__file__).resolve().parents[1]


def frame(rgb, t=10):
    return NS(width=rgb.shape[1],height=rgb.shape[0],encoding='rgb8',step=rgb.shape[1]*3,
              data=rgb.tobytes(),header=NS(frame_id='camera_optical_frame',
              stamp=NS(sec=int(t),nanosec=int(round((t-int(t))*1e9)))))


def load_controller():
    names = ('rclpy','rclpy.action','rclpy.signals','rclpy.time','action_msgs.msg',
             'nav2_msgs.action','sensor_msgs.msg','tf2_ros')
    mocks = {name:MagicMock() for name in names}
    mocks.update({name.split('.')[0]:MagicMock() for name in names if '.' in name})
    mocks['maze_escape.mission'] = mission
    mocks['tf2_ros'].TransformException = Exception
    mocks['sensor_msgs.msg'].Image = NS
    spec = importlib.util.spec_from_file_location('maze_escape._camera_test',ROOT/'maze_escape/camera_escape.py')
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules,mocks):
        spec.loader.exec_module(module)
    module.GoalStatus = NS(STATUS_SUCCEEDED=4,STATUS_CANCELED=5,STATUS_ABORTED=6)
    return module


controller = load_controller()


class VisionTests(unittest.TestCase):
    def image(self):
        rgb = np.zeros((60,80,3),np.uint8)
        rgb[15:35,10:30] = (255,255,0)
        rgb[10:50,50:70] = (255,0,0)
        return rgb,np.full((60,80),2,np.float32)

    def test_yellow_key_and_red_door_are_separate(self):
        rgb,depth = self.image()
        self.assertEqual({d.label for d in detect(rgb,depth)},{'key','door'})

    def test_green_exit_and_blue_wall_are_not_keys(self):
        rgb,depth = self.image()
        rgb[15:35,10:30] = (0,255,0)
        rgb[10:50,50:70] = (100,150,200)
        self.assertEqual(detect(rgb,depth),[])

    def test_invalid_or_misaligned_depth_cannot_detect(self):
        rgb,depth = self.image()
        depth[:] = np.nan
        self.assertEqual(detect(rgb,depth),[])
        with self.assertRaises(ValueError):
            detect(rgb,depth[:20])

    def test_depth_discontinuity_is_rejected(self):
        rgb,depth = self.image()
        depth[:25] = 3.2
        self.assertEqual(detect(rgb,depth),[])

    def test_pixel_projection_and_optical_rotation(self):
        d = NS(u=40,v=30,depth=2)
        k = (100,0,40,0,100,30,0,0,1)
        p = optical_point(d,k)
        self.assertEqual(p,(0,0,2))
        world = transform_point(p,(1.08,2,.2),(-.5,.5,-.5,.5))
        np.testing.assert_allclose(world,(3.08,2,.2),atol=1e-8)

    def test_rgb_padding_and_bgr_order(self):
        msg = NS(width=2,height=1,step=8,encoding='bgr8',data=bytes([0,255,255,0,0,255,99,99]))
        self.assertEqual(rgb_array(msg).tolist(),[[[255,255,0],[255,0,0]]])

    def test_big_endian_depth_and_millimetres(self):
        values = np.array([[1,2]],dtype='>f4')
        msg = NS(width=2,height=1,step=8,encoding='32FC1',is_bigendian=1,data=values.tobytes())
        np.testing.assert_allclose(depth_array(msg),[[1,2]])
        msg.encoding,msg.step,msg.is_bigendian = '16UC1',4,0
        msg.data = np.array([[1000,2000]],dtype='<u2').tobytes()
        np.testing.assert_allclose(depth_array(msg),[[1,2]])

    def test_stable_observation_requires_distinct_nearby_samples(self):
        gate = ObservationGate()
        self.assertIsNone(gate.update('key',1,(1,2,0)))
        self.assertIsNone(gate.update('key',1,(1,2,0)))
        self.assertIsNone(gate.update('key',1.2,(1.1,2,0)))
        self.assertIsNotNone(gate.update('key',1.4,(1,2,0)))
        self.assertIsNone(gate.update('key',1.6,(9,2,0)))

    def test_approach_keeps_standoff_and_faces_object(self):
        self.assertEqual(approach((0,0,0),(2,0,0),.5),(1.5,0,0))
        self.assertEqual(approach((0,0,0),(.2,0,0),.5),(0,0,0))


class ScanTests(unittest.TestCase):
    def scan(self, distance):
        return NS(ranges=[distance]*360,angle_min=-math.pi,angle_increment=2*math.pi/360,
                  range_min=.1,range_max=3.5)

    def test_closed_and_open_door(self):
        self.assertFalse(forward_clear(self.scan(.7)))
        self.assertTrue(forward_clear(self.scan(2)))
        self.assertTrue(forward_clear(self.scan(float('inf'))))

    def test_bad_scan_cannot_confirm_open(self):
        self.assertFalse(forward_clear(self.scan(float('nan'))))
        self.assertFalse(forward_clear(self.scan(0)))


class ModelTests(unittest.TestCase):
    def test_only_sensor_and_camera_tf_are_added(self):
        source = '<sdf version="1.8"><model name="burger"><link name="base_footprint"/>' \
                 '<plugin name="drive"><wheel_radius>0.033</wheel_radius></plugin></model></sdf>'
        urdf = '<robot name="burger"><link name="base_footprint"/></robot>'
        with tempfile.TemporaryDirectory() as temporary:
            sdf_file,urdf_file = build(source,urdf,temporary)
            sdf,robot = ET.parse(sdf_file),ET.parse(urdf_file)
            self.assertEqual(sdf.findtext('.//wheel_radius'),'0.033')
            self.assertEqual(sdf.find('.//sensor').attrib['type'],'rgbd_camera')
            self.assertEqual(sdf.findtext('.//optical_frame_id'),'camera_optical_frame')
            self.assertEqual(robot.find("joint[@name='camera_mount']/parent").attrib['link'],'base_footprint')


class ActuatorTests(unittest.TestCase):
    def test_door_is_locked_until_key_collected(self):
        actuator = SceneActuator(ROOT/'worlds/escape_room.world')
        with patch('maze_escape.actuator.subprocess.run') as run:
            with self.assertRaises(ValueError):
                actuator.lift('red_door',False)
            run.assert_not_called()

    def test_service_ack_is_required_and_timeout_is_bounded(self):
        actuator = SceneActuator(ROOT/'worlds/escape_room.world')
        with patch('maze_escape.actuator.subprocess.run') as run:
            run.return_value = NS(returncode=0,stdout='data: false',stderr='')
            with self.assertRaises(RuntimeError):
                actuator.lift('red_door',True)
            run.return_value = NS(returncode=0,stdout='data: true',stderr='')
            self.assertEqual(actuator.lift('red_door',True),'red_door')
            self.assertEqual(run.call_args.kwargs['timeout'],8)


class ControllerTests(unittest.TestCase):
    def node(self):
        node = controller.CameraEscape.__new__(controller.CameraEscape)
        node.state = RequestState()
        node.state.begin('NAV_REQUEST',1,15)
        node.stage = 'SEARCH_KEY'
        node.done = node.stopping = False
        node.emit = MagicMock()
        node.stop = MagicMock()
        node.goal_handle = MagicMock()
        node.key_collected = False
        return node

    def test_target_switch_waits_for_pending_goal_before_cancel(self):
        node = self.node()
        node.request_switch('key',(1,2,0))
        self.assertEqual(node.stage,'SWITCHING')
        node.goal_handle.cancel_goal_async.assert_not_called()
        future = MagicMock()
        future.result.return_value.accepted = True
        node.switch_acceptance(future,node.state.generation)
        self.assertEqual(node.state.phase,'CANCELLING')
        self.assertEqual(node.stage,'SWITCHING')
        node.goal_handle.cancel_goal_async.assert_called_once()
        finished = MagicMock()
        finished.result.return_value.status = 5
        node.switched_result(finished,node.state.generation)
        self.assertEqual(node.stage,'APPROACH_KEY')
        self.assertEqual(node.state.phase,'SELECT')

    def test_door_cannot_complete_without_key(self):
        node = self.node()
        node.verify_open(10,10)
        node.stop.assert_called_once()
        self.assertEqual(node.stop.call_args.args[1],2)

    def test_open_service_response_does_not_complete_mission(self):
        node = self.node()
        node.stage = 'OPENING'
        node.key_collected = True
        node.operation = MagicMock()
        node.operation.done.return_value = True
        node.sim = lambda:10
        node.poll_operation(10)
        self.assertEqual(node.stage,'VERIFY_OPEN')
        node.stop.assert_not_called()

    def test_rgb_depth_tf_pipeline_produces_a_camera_observation(self):
        node = self.node()
        rgb = np.zeros((60,80,3),np.uint8)
        rgb[20:40,30:50] = (255,255,0)
        node.grid = NS(frame='map')
        node.info = NS(width=80,height=60,k=(100,0,39.5,0,100,29.5,0,0,1))
        node.last_pair = -1
        node.sim = lambda:11
        node.observations = {}
        node.gate = ObservationGate()
        node.tf = MagicMock()
        node.tf.lookup_transform.return_value.transform = NS(translation=NS(x=1.08,y=2,z=.2),
                                                              rotation=NS(x=-.5,y=.5,z=-.5,w=.5))
        node.image_pub = MagicMock()
        for t in (10.4,10.6,10.8):
            node.rgb = frame(rgb,t)
            node.depth = NS(height=60,width=80,encoding='32FC1',is_bigendian=0,
                            step=80*4,data=np.full((60,80),2,np.float32).tobytes(),header=node.rgb.header)
            node.observe()
        self.assertIn('key',node.observations)
        np.testing.assert_allclose(node.observations['key'][0],(3.08,2,.2),atol=.02)
        self.assertEqual(node.image_pub.publish.call_count,3)
        node.stop.assert_not_called()


if __name__ == '__main__':
    unittest.main()
