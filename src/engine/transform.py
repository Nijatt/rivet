import numpy as np
from .quaternion import quat_identity, quat_normalize, quat_mul,quat_from_axis_angle

class Transform:
    __slots__ = ("position", "rotation")
    def __init__(self, position=(0.0,0.0,0.0), rotation=None):
        self.position = np.array(position, dtype=float)
        self.rotation = quat_identity() if rotation is None else quat_normalize(np.array(rotation, dtype=float))

    def set_position(self, p):
        self.position[...] = p  # expects length 3

    def set_rotation(self, q):
        self.rotation[...] = quat_normalize(np.array(q, dtype=float))

    def rotate_about_axis(self, axis, angle):
        incr = quat_from_axis_angle(axis, angle)
        self.rotation = quat_normalize(quat_mul(self.rotation, incr))
