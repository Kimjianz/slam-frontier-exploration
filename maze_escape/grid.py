"""ROS-independent grid geometry and frontier candidate generation."""
from collections import deque
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Grid:
    width: int
    height: int
    resolution: float
    origin_x: float
    origin_y: float
    origin_yaw: float
    data: tuple
    frame: str = 'map'

    def __post_init__(self):
        if (self.width <= 0 or self.height <= 0
                or len(self.data) != self.width*self.height
                or not math.isfinite(self.resolution) or self.resolution <= 0
                or not all(math.isfinite(v) for v in
                           (self.origin_x, self.origin_y, self.origin_yaw))):
            raise ValueError('지도 크기와 좌표 정보가 올바르지 않아.')

    def inside(self, cell):
        r, c = cell
        return 0 <= r < self.height and 0 <= c < self.width

    def value(self, cell):
        return self.data[cell[0]*self.width+cell[1]] if self.inside(cell) else None

    def cell(self, x, y):
        dx, dy = x-self.origin_x, y-self.origin_y
        co, si = math.cos(self.origin_yaw), math.sin(self.origin_yaw)
        return (math.floor((-si*dx+co*dy)/self.resolution),
                math.floor((co*dx+si*dy)/self.resolution))

    def world(self, cell):
        r, c = cell
        x, y = (c+0.5)*self.resolution, (r+0.5)*self.resolution
        co, si = math.cos(self.origin_yaw), math.sin(self.origin_yaw)
        return self.origin_x+co*x-si*y, self.origin_y+si*x+co*y

    def neighbors(self, cell):
        r, c = cell
        return ((r-1,c), (r+1,c), (r,c-1), (r,c+1))


def angle_difference(a, b):
    return math.atan2(math.sin(a-b), math.cos(a-b))


def frontiers(grid):
    return {divmod(i, grid.width) for i, value in enumerate(grid.data)
            if value == 0 and any(grid.value(n) == -1
                                  for n in grid.neighbors(divmod(i, grid.width)))}


def groups(cells, minimum=5):
    remaining, result = set(cells), []
    for seed in sorted(cells):
        if seed not in remaining:
            continue
        remaining.remove(seed)
        queue, group = deque([seed]), []
        while queue:
            r, c = queue.popleft()
            group.append((r,c))
            for dr in (-1,0,1):
                for dc in (-1,0,1):
                    n = (r+dr,c+dc)
                    if n in remaining:
                        remaining.remove(n)
                        queue.append(n)
        if len(group) >= minimum:
            result.append(group)
    return result


def reachable(grid, start):
    # Connectivity is a preliminary check. Nav2 validates the robot footprint.
    if grid.value(start) != 0:
        return {}
    queue, distance = deque([start]), {start:0}
    while queue:
        cell = queue.popleft()
        for n in grid.neighbors(cell):
            if n not in distance and grid.value(n) == 0:
                distance[n] = distance[cell]+1
                queue.append(n)
    return distance


def destination_clear(grid, cell, radius=0.14):
    # Only destinations require this disk; do not reject the robot's start
    # because lidar has an unobserved minimum-range patch under the robot.
    cells = radius/grid.resolution+math.sqrt(0.5)
    extent = math.ceil(cells)
    r, c = cell
    return all(grid.value((r+dr,c+dc)) == 0
               for dr in range(-extent,extent+1)
               for dc in range(-extent,extent+1)
               if dr*dr+dc*dc <= cells*cells)


@dataclass(frozen=True)
class Candidate:
    x: float
    y: float
    distance: float
    group: int


def candidates(grid, robot, excluded=(), clearance=0.14, minimum_distance=0.40,
               approach_distance=0.60, spacing=0.40, limit=12):
    start = grid.cell(*robot)
    distance = reachable(grid, start)
    clusters = groups(frontiers(grid))
    ranked = []
    if not distance:
        return [], len(clusters)
    for number, cluster in enumerate(clusters, 1):
        seen, queue = set(cluster), deque((cell,0) for cell in sorted(cluster))
        maximum = math.ceil(approach_distance/grid.resolution)
        local = []
        while queue:
            cell, depth = queue.popleft()
            if cell in distance:
                x, y = grid.world(cell)
                if (math.hypot(x-robot[0],y-robot[1]) >= minimum_distance
                        and all(math.hypot(x-a,y-b) > spacing for a,b in excluded)
                        and destination_clear(grid, cell, clearance)):
                    local.append((depth, distance[cell], cell, x, y))
            if depth < maximum:
                for n in grid.neighbors(cell):
                    if n not in seen and grid.value(n) == 0:
                        seen.add(n)
                        queue.append((n,depth+1))
        selected = []
        for depth, steps, cell, x, y in sorted(local):
            if all(math.hypot(x-a,y-b) >= spacing for a,b in selected):
                selected.append((x,y))
                ranked.append(Candidate(x,y,steps*grid.resolution,number))
                if len(selected) == 3:
                    break
    return sorted(ranked, key=lambda p:(p.distance,p.group,p.x,p.y))[:limit], len(clusters)


def path_acceptable(points, goal, tolerance=0.10):
    return bool(points) and all(math.isfinite(v) for p in list(points)+[goal] for v in p) \
        and math.hypot(points[-1][0]-goal[0],points[-1][1]-goal[1]) <= tolerance


def probe_goals(start, distance=0.60):
    x, y, yaw = start
    return [(x+distance*math.cos(yaw), y+distance*math.sin(yaw), yaw),
            (x,y,angle_difference(yaw+math.pi,0))]


class Exclusions:
    def __init__(self):
        self.items = []

    def add(self, point, simulation_time, succeeded):
        self.items.append((point,simulation_time+(60 if succeeded else 120)))

    def active(self, simulation_time):
        self.items = [(p,end) for p,end in self.items if end > simulation_time]
        return [p for p,end in self.items]
