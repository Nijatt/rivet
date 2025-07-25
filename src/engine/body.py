# body.py
# body.py
import numpy as np
from .transform import Transform
from .quaternion import quat_identity, quat_normalize, quat_mul, quat_from_axis_angle
from .math import float3

class RigidBody:
    """
    Minimal rigid body for positional XPBD integration.
    - transform: committed state
    - pred_transform: predicted (temporary) state used during a substep
    """
    __slots__ = ("transform", "pred_transform", "vel", "mass", "inv_mass", "radius")

    def __init__(self, transform: Transform, mass: float = 1.0, radius: float = 0.5):
        if not isinstance(transform, Transform):
            raise TypeError("RigidBody expects a Transform instance as first argument.")

        # Committed transform
        self.transform = transform

        # Independent predicted transform (deep copy so prediction is separate)
        self.pred_transform = transform

        self.vel = np.zeros(3, dtype=float)
        self.mass = mass
        self.inv_mass = 0.0 if mass == 0 else 1.0 / mass
        self.radius = radius

    @property
    def pos(self):
        return self.transform.position

    @property
    def pred_pos(self):
        return self.pred_transform.position

    

class Particle:
    __slots__ = ("transform", "pred_transform", "vel", "mass", "inv_mass", "radius")

    def __init__(self, transform: Transform, mass: float = 1.0, radius: float = 0.5):
        if not isinstance(transform, Transform):
            raise TypeError("RigidBody expects a Transform instance as first argument.")
        self.transform = transform
        self.pred_transform  = transform
        self.vel       = np.zeros(3, dtype=float)
        self.mass      = mass
        self.inv_mass  = 0.0 if mass == 0 else 1.0 / mass
        self.radius    = radius

    @property
    def pos(self):
        return self.transform.position
    
    @property
    def pred_pos(self):
        return self.pred_transform.position
