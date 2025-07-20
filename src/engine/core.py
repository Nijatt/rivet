# src/engine/core.py
import numpy as np

class XPBDSolver:
    def __init__(self, bodies, gravity=np.array([0.0, -9.81]), substeps=4, iters=8):
        self.bodies, self.gravity = bodies, gravity
        self.substeps, self.iters = substeps, iters

    def step(self, dt):
        h = dt / self.substeps
        for _ in range(self.substeps):
            # 1) apply gravity & predict positions
            for b in self.bodies:
                if b.inv_mass == 0: continue
                b.vel += self.gravity * h
                b.pred_pos = b.pos + b.vel * h

            # 2) solve constraints iteratively
            for _ in range(self.iters):
                for b in self.bodies:
                    # here you'd call body-body & body-plane constraints
                    # collision.resolve(b, other, ...)
                    pass

            # 3) update velocities & positions
            for b in self.bodies:
                b.vel = (b.pred_pos - b.pos) / h
                b.pos = b.pred_pos
