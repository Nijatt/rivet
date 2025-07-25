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

@dataclass
class OrientationElement:
    edge0: int
    edge1: int
    rest_darboux: np.ndarray  # shape (3,)


@dataclass
class ElasticRod:
    particles: list           # list[RigidBody]
    ghost_particles:list
    edges: list[ElasticEdge] = field(default_factory=list)
    orientation_elements: list[OrientationElement] = field(default_factory=list)


# #TODO: gravity must be in z but we have unity consistent y
# class ElasticEdge:
#     def __init__(self, p0,p1,g1, edgeL,ghostL):
#         #Indices of the particles and ghost particles.
#         self.p0 = p0
#         self.p1 = p1
#         self.g1 = g1

#         self.edgeL = edgeL
#         self.ghostL = ghostL

#     def get_frame(data_container):
#         pass

# class OrientationElement:
#     def __init__(self, edge0,edge1, restDarboux):
#         #Indices of the edges.
#         self.edge0 = edge0
#         self.edge1 = edge1
#         self.restDarboux = restDarboux
