"""Object observation memory and known-space viewpoints, independent of ROS."""
from dataclasses import dataclass
import math
from .grid import destination_clear, reachable


@dataclass
class RememberedView:
    uid: int
    label: str
    point: tuple
    pose: tuple
    frame: str
    seen: float


class ObjectMemory:
    def __init__(self, maximum=8, lifetime=900):
        self.maximum,self.lifetime = maximum,lifetime
        self.views = {}
        self.serial = 0

    def remember(self, label, point, pose, frame, sim):
        if not all(math.isfinite(v) for v in (*point,*pose,sim)):
            return
        entries = self.views.setdefault(label,[])
        for entry in entries:
            if (entry.frame==frame and math.dist(entry.pose[:2],pose[:2])<.6
                    and math.dist(entry.point[:2],point[:2])<.5):
                if sim>=entry.seen:
                    entry.point,entry.pose,entry.seen = tuple(point),tuple(pose),sim
                return
        self.serial += 1
        entries.append(RememberedView(self.serial,label,tuple(point),tuple(pose),frame,sim))
        entries.sort(key=lambda e:e.seen,reverse=True)
        del entries[self.maximum:]

    def candidates(self, label, frame, sim, attempted):
        return sorted((entry for entry in self.views.get(label,[])
                       if entry.frame==frame and entry.uid not in attempted
                       and 0<=sim-entry.seen<=self.lifetime),key=lambda entry:-entry.seen)


def free_line(grid, first, second):
    steps = max(1,math.ceil(math.dist(first[:2],second[:2])/(grid.resolution*.5)))
    return all(grid.value(grid.cell(first[0]+(second[0]-first[0])*i/steps,
                                   first[1]+(second[1]-first[1])*i/steps))==0
               for i in range(steps+1))


def safe_view(grid, cell, connected, costmap=None):
    if cell not in connected or not destination_clear(grid,cell,.12):
        return False
    if costmap is not None:
        x,y = grid.world(cell)
        cost = costmap.value(costmap.cell(x,y))
        if cost is None or cost < 0 or cost >= 99:
            return False
    return True


def inspection_candidates(grid, robot, completed=(), failed=(), costmap=None,
                          spacing=1.2, limit=12):
    connected = reachable(grid,grid.cell(*robot[:2]))
    tiles = {}
    for cell,steps in connected.items():
        x,y = grid.world(cell)
        if any(math.hypot(x-a,y-b)<.35 for a,b in failed):
            continue
        if any(math.hypot(x-a,y-b)<spacing*.85 and free_line(grid,(x,y),(a,b))
               for a,b in completed):
            continue
        tile = math.floor(x/spacing),math.floor(y/spacing)
        centrality = math.hypot(x-(tile[0]+.5)*spacing,y-(tile[1]+.5)*spacing)
        tiles.setdefault(tile,[]).append((centrality,steps,cell,x,y))
    choices = []
    for entries in tiles.values():
        for centre,steps,cell,x,y in sorted(entries):
            if safe_view(grid,cell,connected,costmap):
                heading = math.atan2(y-robot[1],x-robot[0]) if steps else robot[2]
                choices.append((steps*grid.resolution,(x,y,heading)))
                break
    choices.sort(key=lambda entry:(entry[0],entry[1]))
    return [pose for distance,pose in choices[:limit]],{
        'reachable_cells':len(connected),'available_viewpoints':len(choices),
        'completed_views':len(completed),'failed_views':len(failed)}
