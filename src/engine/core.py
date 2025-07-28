# src/engine/core.py
import numpy as np
from engine.rod_utils import RodUtils

#TODO: gravity must be in z but we have unity consistent y
class PBDSolver:
    def __init__(self, elastic_rod, gravity=np.array([0.0, -9.81,0.0]), substeps=4, iters=8):
        self.elastic_rod, self.gravity = elastic_rod, gravity
        self.substeps, self.iters = substeps, iters
        self.g_norm2 = np.dot(self.gravity, self.gravity) 
    
    #NOTE: we are going to define the whole solver here for the sake of simplicity
    #NOTE: the constraints will be solved here.


    #TODO: separate the algorithm so you can use substeps.
    def step(self, dt):
        h = dt / self.substeps

        for _ in range(self.substeps):
            # 1) apply gravity & predict positions
            for b in self.elastic_rod.particles:
                if b.inv_mass == 0: continue
                b.vel += self.gravity * h
                b.pred_transform.position = b.transform.position + b.vel * h

            for e in self.elastic_rod.edges:
                p0, p1  = self.elastic_rod.particles[e.p0], self.elastic_rod.particles[e.p1]
                g1 = self.elastic_rod.ghost_particles[e.g1]
                if g1.inv_mass == 0: continue
                # midpoint velocity at current sub‑step
                #Store midpoint velocity in the edge.
                v_m_now  = 0.5 * (p0.vel + p1.vel)

                # midpoint velocity at previous sub‑step (store it on the ghost)
                v_m_old  = g1.mid_vel_prev if hasattr(g1, 'mid_vel_prev') else v_m_now

                # acceleration of midpoint
                a_m      = (v_m_now - v_m_old) / dt

                # ratio r  (Eq. r = (a_m·g)/|g|²)
                r        = np.dot(a_m, self.gravity) / self.g_norm2

                # Δv = (1 – r) Δt g   —— distribute: −1 for ghost, +½ each to end points
                dv       = (1.0 - r) * dt * self.gravity

                g1.vel       -= dv                                     # v_g  ← v_g − Δv
                p0.vel  += 0.5 * dv                               # v_{e‑1} += ½ Δv
                p1.vel  += 0.5 * dv                               # v_e     += ½ Δv

                # TODO: stash current midpoint velocity for next sub‑step
                # g1.mid_vel_prev = v_m_now
                g1.pred_transform.position = g1.transform.position + g1.vel * h


            for b in self.elastic_rod.particles:
                if b.inv_mass == 0: continue
                b.pred_transform.position = b.transform.position + b.vel * h


            #NOTE: This segment of the code is an important part where we can test different type of solvers.
            #TODO: in current case test it with the GS. Then eventually move to the other system for better accurasy.
            for _ in range(self.iters):
                for parity in (0, 1): 
                    for ei in range(parity, len(self.elastic_rod.edges), 2):

                        e   = self.elastic_rod.edges[ei]

                        i0, i1 , g1= e.p0, e.p1,e.g1
                        p0 = self.elastic_rod.particles[i0]
                        p1 = self.elastic_rod.particles[i1]
                        g1 = self.elastic_rod.particles[g1]

                        w0, w1 = p0.inv_mass, p1.inv_mass
                        if w0 == 0.0 and w1 == 0.0:
                            continue

                        #copy the position
                        x0 = p0.pred_transform.position
                        x1 = p1.pred_transform.position

                        stiffness = 0.1
                        x0, x1 = RodUtils.project_edge(x0, x1, w0, w1, e.rest_len, stiffness)



                        #Copy back.
                        p0.pred_transform.position = x0
                        p1.pred_transform.position = x1



            # 3) update velocities & positions
            for b in self.elastic_rod.particles:
                b.vel = (b.pred_transform.position - b.transform.position) / h
                b.transform.position = b.pred_transform.position

            for b in self.elastic_rod.ghost_particles:
                b.vel = (b.pred_transform.position - b.transform.position) / h
                b.transform.position = b.pred_transform.position



#TODO: gravity must be in z but we have unity consistent y
class XPBDSolver:
    def __init__(self, elastic_rod, gravity=np.array([0.0, -9.81,0.0]), substeps=4, iters=8):
        self.elastic_rod, self.gravity = elastic_rod, gravity
        self.substeps, self.iters = substeps, iters
    
    #NOTE: we are going to define the whole solver here for the sake of simplicity
    #NOTE: the constraints will be solved here.


    #TODO: separate the algorithm so you can use substeps.
    def step(self, dt):
        h = dt / self.substeps

        for _ in range(self.substeps):
            # 1) apply gravity & predict positions
            for b in self.elastic_rod.particles:
                if b.inv_mass == 0: continue
                b.vel += self.gravity * h
                b.pred_transform.position = b.transform.position + b.vel * h

                #TODO: use ghost points here. Do  dirty check for the particles.
                # if ghost particle do particle check here.

            #NOTE: This segment of the code is an important part where we can test different type of solvers.
            #TODO: in current case test it with the GS. Then eventually move to the other system for better accurasy.

            for _ in range(self.iters):
                #TODO: just solve distance 
                for e in self.elastic_rod.edges:

                    i0, i1 = e.p0, e.p1
                    p0 = self.elastic_rod.particles[i0]
                    p1 = self.elastic_rod.particles[i1]

                    w0, w1 = p0.inv_mass, p1.inv_mass
                    if w0 == 0.0 and w1 == 0.0:
                        continue

                    x0 = p0.pred_transform.position
                    x1 = p1.pred_transform.position

                    d   = x1 - x0
                    L   = np.linalg.norm(d)
                    if L < 1e-8:
                        continue

                    C   = L - e.rest_len
                    n   = d / L
                    wsum = w0 + w1
                    
                    if wsum == 0.0:
                        continue

                    corr_mag = C / wsum

                    # bail if anything is non-finite
                    if not np.isfinite(corr_mag):
                        continue

                    corr = corr_mag * n
                    if not np.all(np.isfinite(corr)):
                        continue
                    
                    stiffness = 0.03

                    if w0 != 0.0:
                        x0 += stiffness*w0 * corr
                    if w1 != 0.0:
                        x1 -= stiffness*w1 * corr

                    p0.pred_transform.position = x0
                    p1.pred_transform.position = x1


            # 3) update velocities & positions
            for b in self.elastic_rod.particles:
                b.vel = (b.pred_transform.position - b.transform.position) / h
                b.transform.position = b.pred_transform.position

#Make it casual xpbd solver.
class XPBDSolver2:
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
                #TODO: solve rope
                pass

            # 3) update velocities & positions
            for b in self.bodies:
                b.vel = (b.pred_transform.position - b.transform.position) / h
                b.transform.position = b.pred_transform.position
