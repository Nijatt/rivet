# src/engine/body.py
import numpy as np

class RigidBody:
    def __init__(self, position, mass=1.0, radius=0.5):
        self.pos = np.array(position, dtype=float)
        self.pred_pos = self.pos.copy()
        self.vel = np.zeros(3)
        self.mass = mass
        self.inv_mass = 0 if mass == 0 else 1/mass
        self.radius = radius  # for simple circle collision