"""Select reachable, known-free camera approach goals and bounded observation steps."""
from dataclasses import dataclass
import math
from .grid import angle_difference, destination_clear, reachable


LIMITS = {'key':(.38,.64,.48,.70),'door':(.55,.83,.67,.95),'exit':(.45,.72,.58,.85)}


@dataclass(frozen=True)
class ApproachGoal:
    pose: tuple
    kind: str
    path_length: float


def visible_from(grid, start, target):
    distance = math.dist(start[:2],target[:2])
    steps = max(1,math.ceil(max(0,distance-.16)/(grid.resolution/2)))
    for i in range(steps):
        ratio = (distance-.16)*i/steps/distance if distance > .16 else 0
        point = (start[0]+ratio*(target[0]-start[0]),start[1]+ratio*(target[1]-start[1]))
        value = grid.value(grid.cell(*point))
        if value is not None and value > 0:
            return False
    return True


def select_approaches(grid, robot, target, label, rejected=(), costmap=None, limit=8):
    minimum,maximum,preferred,_ = LIMITS[label]
    distances = reachable(grid,grid.cell(*robot[:2]))
    info = {'robot_cell':grid.cell(*robot[:2]),'robot_value':grid.value(grid.cell(*robot[:2])),
            'reachable_cells':len(distances),'final_candidates':0,'observation_candidates':0}
    if not distances:
        return [],info
    direction = math.atan2(robot[1]-target[1],robot[0]-target[0])
    robot_distance = math.dist(robot[:2],target[:2])
    final,progress = [],[]
    for cell,steps in distances.items():
        x,y = grid.world(cell)
        distance = math.hypot(x-target[0],y-target[1])
        near = minimum <= distance <= maximum
        closer = maximum < distance < robot_distance-.25
        if not near and not closer:
            continue
        if any(math.hypot(x-a,y-b)<.18 for a,b in rejected):
            continue
        if not destination_clear(grid,cell,.12):
            continue
        if costmap is not None:
            value = costmap.value(costmap.cell(x,y))
            if value is None or value < 0 or value >= 99:
                continue
        if not visible_from(grid,(x,y),target):
            continue
        yaw = math.atan2(target[1]-y,target[0]-x)
        goal = ApproachGoal((x,y,yaw),'approach' if near else 'observe',steps*grid.resolution)
        if near:
            side = abs(angle_difference(math.atan2(y-target[1],x-target[0]),direction))
            final.append((abs(distance-preferred)+.08*goal.path_length+.06*side,goal))
        elif .25 <= math.dist((x,y),robot[:2]) <= 1.0:
            progress.append((distance+.12*goal.path_length,goal))
    info['final_candidates'],info['observation_candidates'] = len(final),len(progress)
    chosen = []
    for score,goal in sorted(final or progress,key=lambda item:(item[0],item[1].pose)):
        if all(math.dist(goal.pose[:2],other.pose[:2]) >= .18 for other in chosen):
            chosen.append(goal)
            if len(chosen) >= limit:
                break
    return chosen,info


@dataclass
class VerificationWindow:
    wall: float
    sim: float
    frames: int

    def enough(self, sim, frames):
        return sim-self.sim >= 1 and frames-self.frames >= 6

    def expired(self, wall, sim):
        return wall-self.wall > 120 or sim-self.sim > 15


def passage_distance(robot, door):
    projection = ((door[0]-robot[0])*math.cos(robot[2])
                  +(door[1]-robot[1])*math.sin(robot[2]))
    return min(1.6,max(1.1,projection+.65))


def crossed_plane(robot, start, door):
    forward = ((robot[0]-door[0])*math.cos(start[2])
               +(robot[1]-door[1])*math.sin(start[2]))
    return math.dist(robot[:2],start[:2]) >= .75 and forward >= .20
