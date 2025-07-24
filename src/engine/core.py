# src/engine/core.py
import numpy as np

#TODO: gravity must be in z but we have unity consistent y
class XPBDSolver:
    def __init__(self, bodies, gravity=np.array([0.0, -9.81,0.0]), substeps=4, iters=8):
        self.bodies, self.gravity = bodies, gravity
        self.substeps, self.iters = substeps, iters
    
    #NOTE: we are going to define the whole solver here for the sake of simplicity
    #NOTE: the constraints will be solved here.
    


    #TODO: separate the algorithm so you can use substeps.
    def step(self, dt):
        h = dt / self.substeps

        for _ in range(self.substeps):
            # 1) apply gravity & predict positions
            for b in self.bodies:
                if b.inv_mass == 0: continue
                b.vel += self.gravity * h
                b.pred_transform.position = b.transform.position + b.vel * h

                #TODO: use ghost points here. Do  dirty check for the particles.
                # if ghost particle do particle check here.

            #NOTE: This segment of the code is an important part where we can test different type of solvers.
            #TODO: in current case test it with the GS. Then eventually move to the other system for better accurasy.

            for _ in range(self.iters):
                #TODO: solve rope physics here.
                pass

            # 3) update velocities & positions
            for b in self.bodies:
                b.vel = (b.pred_transform.position - b.transform.position) / h
                b.transform.position = b.pred_transform.position
