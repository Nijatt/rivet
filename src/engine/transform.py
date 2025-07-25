# src/engine/transform.py  (or inline in core.py)

import numpy as np
from .math import float3 
from .quaternion import (
    quat_identity, quat_normalize,
    quat_mul, quat_from_axis_angle
)

class Transform:
    __slots__ = ("position", "rotation")

    def __init__(self, position=(0.0, 0.0, 0.0), rotation=None):
        self.position = (
            position if isinstance(position, float3)
            else float3(*position)
        )
        self.rotation = (
            quat_identity()
            if rotation is None
            else quat_normalize(np.asarray(rotation, dtype=float))
        )

    def set_position(self, p):
        """p: float3 | (x,y,z)"""
        self.position = p if isinstance(p, float3) else float3(*p)

    def set_rotation(self, q):
        """q: length‑4 quaternion iterable"""
        self.rotation = quat_normalize(np.asarray(q, dtype=float))

    def translate(self, delta):
        """delta: float3 | (dx,dy,dz) — additive"""
        self.position += (
            delta if isinstance(delta, float3) else float3(*delta)
        )

    def rotate_about_axis(self, axis, angle):
        """axis: (3,) array or iterable • angle: radians"""
        incr = quat_from_axis_angle(axis, angle)
        self.rotation = quat_normalize(quat_mul(self.rotation, incr))

    def copy(self):
        return Transform(
            position=float3(self.position.x, self.position.y, self.position.z),
            rotation=self.rotation.copy(),
        )

    def __repr__(self):
        pos = self.position
        rot = ", ".join(f"{v:.4g}" for v in self.rotation)
        return f"Transform(pos={pos}, rot=[{rot}])"
