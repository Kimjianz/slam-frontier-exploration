"""Bounded action state, independently testable without ROS."""
from dataclasses import dataclass


@dataclass
class RequestState:
    phase: str = 'WAIT'
    generation: int = 0
    deadline: float = 0.0

    def begin(self, phase, wall_time, timeout):
        self.generation += 1
        self.phase = phase
        self.deadline = wall_time+timeout
        return self.generation

    def current(self, generation, phase=None):
        return generation == self.generation and (phase is None or phase == self.phase)

    def expired(self, wall_time):
        return self.deadline > 0 and wall_time >= self.deadline

    def finish(self, phase):
        self.generation += 1
        self.phase = phase
        self.deadline = 0.0
