"""Local RGB-D sensor; installed robot physics remain unchanged."""
from pathlib import Path
import xml.etree.ElementTree as ET

def build(sdf_text, urdf_text, destination):
    destination = Path(destination)
    destination.mkdir(parents=True,exist_ok=True)
    sdf = ET.fromstring(sdf_text)
    sdf.set('version','1.9')
    link = sdf.find("./model/link[@name='base_footprint']")
    if link is None:
        raise ValueError('기본 로봇에 base_footprint 링크가 없어.')
    link.append(ET.fromstring('''<sensor name="key_rgbd" type="rgbd_camera">
      <pose>0.08 0 0.20 0 0 0</pose><always_on>true</always_on>
      <update_rate>5</update_rate><topic>/maze_escape/camera</topic>
      <camera><optical_frame_id>camera_optical_frame</optical_frame_id>
        <horizontal_fov>1.2217304764</horizontal_fov>
        <image><width>320</width><height>240</height><format>R8G8B8</format></image>
        <clip><near>0.05</near><far>4.0</far></clip>
        <depth_camera><clip><near>0.08</near><far>4.0</far></clip></depth_camera>
      </camera></sensor>'''))
    urdf = ET.fromstring(urdf_text)
    for xml in (
        '<link name="camera_link"/>',
        '<joint name="camera_mount" type="fixed"><parent link="base_footprint"/>'
        '<child link="camera_link"/><origin xyz="0.08 0 0.20" rpy="0 0 0"/></joint>',
        '<link name="camera_optical_frame"/>',
        '<joint name="camera_optical_joint" type="fixed"><parent link="camera_link"/>'
        '<child link="camera_optical_frame"/><origin xyz="0 0 0" '
        'rpy="-1.5707963267948966 0 -1.5707963267948966"/></joint>',
    ):
        urdf.append(ET.fromstring(xml))
    ET.ElementTree(sdf).write(destination/'model.sdf',encoding='utf-8',xml_declaration=True)
    ET.ElementTree(urdf).write(destination/'robot.urdf',encoding='utf-8',xml_declaration=True)
    return destination/'model.sdf',destination/'robot.urdf'
