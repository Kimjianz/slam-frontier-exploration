"""RGB-D colour observations. No simulator model positions are used here."""
from dataclasses import dataclass
import math
import cv2
import numpy as np


def rgb_array(message):
    channels = {'rgb8':3,'bgr8':3,'rgba8':4,'bgra8':4}.get(message.encoding.lower())
    if channels is None or message.step < message.width*channels:
        raise ValueError('지원하지 않는 컬러 영상 형식이야: '+message.encoding)
    raw = np.frombuffer(message.data,dtype=np.uint8).reshape(message.height,message.step)
    image = raw[:,:message.width*channels].reshape(message.height,message.width,channels)[:,:,:3]
    return image[:,:,::-1].copy() if message.encoding.lower().startswith('bgr') else image.copy()


def depth_array(message):
    encoding = message.encoding.upper()
    if encoding not in ('32FC1','16UC1'):
        raise ValueError('지원하지 않는 깊이 영상 형식이야: '+encoding)
    kind = ('>' if message.is_bigendian else '<')+('f4' if encoding=='32FC1' else 'u2')
    dtype = np.dtype(kind)
    if message.step < message.width*dtype.itemsize or message.step % dtype.itemsize:
        raise ValueError('깊이 영상의 행 크기가 올바르지 않아.')
    image = np.frombuffer(message.data,dtype=dtype).reshape(message.height,message.step//dtype.itemsize)
    return image[:,:message.width].astype(np.float32)*(0.001 if encoding=='16UC1' else 1.0)


@dataclass(frozen=True)
class Detection:
    label: str
    u: float
    v: float
    depth: float
    box: tuple
    pixels: int


def detect(rgb, depth):
    if rgb.shape[:2] != depth.shape:
        raise ValueError('컬러 영상과 깊이 영상의 크기가 달라.')
    hsv = cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
    masks = {
        'key': cv2.inRange(hsv,(20,100,60),(40,255,255)),
        'door': cv2.bitwise_or(cv2.inRange(hsv,(0,110,55),(10,255,255)),
                              cv2.inRange(hsv,(170,110,55),(179,255,255))),
        'exit': cv2.inRange(hsv,(45,100,60),(85,255,255)),
    }
    result = []
    for label,mask in masks.items():
        mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
        count,components,stats,centres = cv2.connectedComponentsWithStats(mask)
        found = []
        for i in range(1,count):
            x,y,w,h,area = map(int,stats[i])
            if area < 30 or w < 4 or h < 4 or area/(w*h) < .45:
                continue
            if label == 'key' and not .35 <= w/h <= 2.8:
                continue
            inside = cv2.erode((components==i).astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)
            values = depth[inside]
            valid = values[np.isfinite(values) & (values>.10) & (values<3.5)]
            if len(valid) < 12 or len(valid) < .6*len(values):
                continue
            distance = float(np.median(valid))
            # A depth boundary passing through the colour blob is ambiguous.
            if float(np.percentile(valid,80)-np.percentile(valid,20)) > max(.25,.2*distance):
                continue
            u,v = map(float,centres[i])
            found.append(Detection(label,u,v,distance,(x,y,w,h),area))
        if found:
            result.append(max(found,key=lambda d:d.pixels))
    return result


def optical_point(detection, k):
    fx,fy,cx,cy = k[0],k[4],k[2],k[5]
    if not all(math.isfinite(v) for v in (fx,fy,cx,cy)) or min(fx,fy) <= 0:
        raise ValueError('카메라 내부 설정이 올바르지 않아.')
    z = detection.depth
    return (detection.u-cx)*z/fx,(detection.v-cy)*z/fy,z


def transform_point(point, translation, quaternion):
    p = np.asarray(point,dtype=float)
    q = np.asarray(quaternion,dtype=float)
    norm = np.linalg.norm(q)
    if not np.isfinite(norm) or norm < 1e-8:
        raise ValueError('카메라 좌표 변환이 올바르지 않아.')
    q /= norm
    v,w = q[:3],q[3]
    return tuple(p+2*np.cross(v,np.cross(v,p)+w*p)+np.asarray(translation))


class ObservationGate:
    def __init__(self):
        self.points = {}

    def update(self, label, stamp, point):
        values = self.points.setdefault(label,[])
        if values and stamp <= values[-1][0]:
            return None
        if values and (stamp-values[-1][0] > .6 or math.dist(point[:2],values[-1][1][:2]) > .3):
            values.clear()
        values.append((stamp,point))
        values[:] = values[-5:]
        if len(values) >= 3 and values[-1][0]-values[0][0] >= .35:
            return tuple(float(x) for x in np.median([p for t,p in values],axis=0))
        return None


def approach(robot, point, standoff):
    dx,dy = point[0]-robot[0],point[1]-robot[1]
    distance = math.hypot(dx,dy)
    if distance < standoff:
        return robot[0],robot[1],math.atan2(dy,dx)
    return point[0]-standoff*dx/distance,point[1]-standoff*dy/distance,math.atan2(dy,dx)


def forward_clear(scan, clearance=1.1):
    ranges = []
    for i,value in enumerate(scan.ranges):
        angle = scan.angle_min+i*scan.angle_increment
        angle = math.atan2(math.sin(angle),math.cos(angle))
        if abs(angle) <= .12:
            if value == float('inf'):
                ranges.append(scan.range_max)
            elif math.isfinite(value) and scan.range_min <= value <= scan.range_max:
                ranges.append(value)
            else:
                return False
    return len(ranges) >= 3 and min(ranges) >= clearance
