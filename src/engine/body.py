# body.py
import numpy as np
from .transform import Transform

class RigidBody:
    """
    Rigid body with positional XPBD integration.
    Expects a preconstructed Transform instance (position: float3, rotation: quat).
    """
    __slots__ = ("transform", "pred_transform", "vel", "mass", "inv_mass", "radius")

    def __init__(self, transform: Transform, mass: float = 1.0, radius: float = 0.5):
        if not isinstance(transform, Transform):
            raise TypeError("RigidBody expects a Transform instance as first argument.")
        self.transform = transform
        # Predicted position starts as current committed transform position
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
