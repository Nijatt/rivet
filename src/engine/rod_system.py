# src/engine/rope_system.py
import numpy as np
from dataclasses import dataclass, field

@dataclass
class ElasticEdge:
    p0: int
    p1: int
    rest_len: float
    g1: int
    ghost_rest_len: float
    edge_mid_prev_velocity: np.ndarray  # shape (3,)


@dataclass
class OrientationElement:
    edge0: int
    edge1: int
    rest_darboux: np.ndarray  # shape (3,)


@dataclass
class ElasticRod:
    particles: list 
    ghost_particles:list
    edges: list[ElasticEdge] = field(default_factory=list)
    orientation_elements: list[OrientationElement] = field(default_factory=list)

