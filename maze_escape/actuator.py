"""Bounded Gazebo commands for collection and a vertically opening door."""
import re
import subprocess
import xml.etree.ElementTree as ET


class SceneActuator:
    def __init__(self, world):
        root = ET.parse(world).getroot().find('world')
        self.world = root.attrib['name']
        self.poses = {}
        for name in ('red_key','red_door'):
            model = root.find("model[@name='"+name+"']")
            if model is None:
                raise ValueError('조작할 물체가 미로에 없어: '+name)
            pose = [float(x) for x in model.findtext('pose').split()]
            if any(abs(v)>1e-8 for v in pose[3:]):
                raise ValueError('이 문 조작기는 기본 미로의 수직 슬라이딩 방식을 사용해.')
            self.poses[name] = pose

    def collect_key(self):
        self._request('remove','gz.msgs.Entity','name: "red_key", type: MODEL','열쇠 삭제')
        return 'red_key'

    def lift(self, name, key_collected=False):
        if name != 'red_door':
            raise ValueError('위로 여는 동작은 문에만 사용할 수 있어.')
        if not key_collected:
            raise ValueError('열쇠를 획득하기 전에는 문을 열 수 없어.')
        x,y,z = self.poses[name][:3]
        request = ('name: "'+name+'", position: {x: '+str(x)+', y: '+str(y)
                   +', z: '+str(z+3.0)+'}, orientation: {w: 1.0}')
        self._request('set_pose','gz.msgs.Pose',request,'문 열기')
        return name

    def _request(self, service, request_type, request, operation):
        result = subprocess.run(['gz','service','-s','/world/'+self.world+'/'+service,
            '--reqtype',request_type,'--reptype','gz.msgs.Boolean',
            '--timeout','3000','--req',request],capture_output=True,text=True,timeout=8)
        if result.returncode != 0 or re.search(r'\bdata:\s*true\b',result.stdout) is None:
            raise RuntimeError('가제보가 '+operation+' 성공을 응답하지 않았어: '+(result.stderr+result.stdout)[-500:])
