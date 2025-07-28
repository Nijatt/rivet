# src/engine/core.py
import numpy as np
from .rod_utils import RodUtils

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
   
                v_m_now  = 0.5 * (p0.vel + p1.vel)
                v_m_old = e.edge_mid_prev_velocity
                a_m  = (v_m_now - v_m_old) / dt
               
                r = np.dot(a_m, self.gravity) / self.g_norm2
                dv  = (1.0 - r) * dt * self.gravity

                g1.vel -= dv 
                p0.vel += 0.5 * dv 
                p1.vel += 0.5 * dv 

                e.edge_mid_prev_velocity = v_m_now
                g1.pred_transform.position = g1.transform.position + g1.vel * h


            for b in self.elastic_rod.particles:
                if b.inv_mass == 0: continue
                b.pred_transform.position = b.transform.position + b.vel * h


            #NOTE: This segment of the code is an important part where we can test different type of solvers.
            #TODO: in current case test it with the GS. Then eventually move to the other system for better accurasy.
            for _ in range(self.iters):
                # for parity in (0, 1): 
                #     for ei in range(parity, len(self.elastic_rod.edges), 2):
                    for e in self.elastic_rod.edges:
                        # e   = self.elastic_rod.edges[ei]

                        i0, i1 , g1= e.p0, e.p1,e.g1

                        p0 = self.elastic_rod.particles[i0]
                        p1 = self.elastic_rod.particles[i1]
                        g1 = self.elastic_rod.ghost_particles[g1]
                        w0, w1, gW1= p0.inv_mass, p1.inv_mass, g1.inv_mass

                        if w0 == 0.0 and w1 == 0.0:
                            continue

                        #copy the position
                        x0 = p0.pred_transform.position
                        x1 = p1.pred_transform.position
                        xg1 =g1.pred_transform.position

                        stiffness = 0.5

                        x0, x1 = RodUtils.project_edge(x0, x1, w0, w1, e.rest_len, stiffness)
                        x0, x1,xg1 = RodUtils.project_perpendicular_bisector(x0, x1,xg1, w0, w1,gW1, stiffness)
                        x0, x1,xg1 = RodUtils.project_ghost_distance(x0, x1,xg1, w0, w1,gW1, e.ghost_rest_len, stiffness)

                        #Copy back.
                        p0.pred_transform.position = x0
                        p1.pred_transform.position = x1
                        g1.pred_transform.position = xg1

                    for oe in self.elastic_rod.orientation_elements:
                        #Get edges
                        edge0 = self.elastic_rod.edges[oe.edge0]
                        edge1 = self.elastic_rod.edges[oe.edge1]

                        i0, i1, i2, g1,g2 = edge0.p0, edge0.p1, edge1.p1, edge0.g1, edge1.g1,

                        p0 = self.elastic_rod.particles[i0]
                        p1 = self.elastic_rod.particles[i1]
                        p2 = self.elastic_rod.particles[i2]

                        g1 = self.elastic_rod.ghost_particles[g1]
                        g2 = self.elastic_rod.ghost_particles[g2]

                        w0, w1, w2, wg1, wg2 = p0.inv_mass, p1.inv_mass, p2.inv_mass, g1.inv_mass, g2.inv_mass

                        #copy the position
                        x0 = p0.pred_transform.position
                        x1 = p1.pred_transform.position
                        x2 = p2.pred_transform.position

                        xg1 =g1.pred_transform.position
                        xg2 =g2.pred_transform.position

                        rest_darboux = oe.rest_darboux

                        frame0=RodUtils.build_frame(x0,x1,xg1)
                        frame1=RodUtils.build_frame(x1,x2,xg2)

                        arclenght = 0.5*(edge0.rest_len+edge1.rest_len)

                        darboux = RodUtils.darboux(frame0,frame1,arclenght);

                        alpha = 1
                        constraint = RodUtils.bend_twist_constraint(rest_darboux,darboux,alpha);
                        
                        x = RodUtils.compute_x(frame0,frame1);

                        status0 , dA1p0, dA1p1, dA1g1, dA2p0, dA2p1, dA2g1, dA3p0, dA3p1 = RodUtils.material_frame_derivatives(x0,  x1,  xg1, frame0) 
                        status1 , dB1p1, dB1p2, dB1g2, dB2p1, dB2p2, dB2g2, dB3p1, dB3p2 = RodUtils.material_frame_derivatives(x1,  x2,  xg2, frame1) 

                        if not status0 or not status1:
                            continue  # or handle degeneracy accordingly
                        
                        dOmegaDp0 = RodUtils.derivative_omega_p0(frame1, dA1p0, dA2p0, dA3p0, x, darboux, arclenght)
                        dOmegaDp1 = RodUtils.derivative_omega_p1(frame0, frame1, dA1p1, dA2p1, dA3p1, dB1p1, dB2p1, dB3p1, x, darboux, arclenght)
                        dOmegaDp2 = RodUtils.derivative_omega_p2(frame0, dB1p2, dB2p2, dB3p2, x, darboux, arclenght)
                        dOmegaDg1 = RodUtils.derivative_omega_g1(frame1, dA1g1, dA2g1, x, darboux, arclenght)
                        dOmegaDg2 = RodUtils.derivative_omega_g2(frame0, dB1g2, dB2g2, x, darboux, arclenght)

                        efficient_mass = (
                            w0 * dOmegaDp0.T @ dOmegaDp0 +
                            w1 * dOmegaDp1.T @ dOmegaDp1 +
                            w2 * dOmegaDp2.T @ dOmegaDp2 +
                            wg1 * dOmegaDg1.T @ dOmegaDg1 +
                            wg2 * dOmegaDg2.T @ dOmegaDg2
                        )

                        # lambda_val = -np.linalg.inv(efficient_mass) @ constraint

                        print(efficient_mass)

                        # dp0 = w0 * dOmegaDp0 @ lambda_val
                        # dp1 = w1 * dOmegaDp1 @ lambda_val
                        # dp2 = w2 * dOmegaDp2 @ lambda_val
                        # dg1 = wg1 * dOmegaDg1 @ lambda_val
                        # dg2 = wg2 * dOmegaDg2 @ lambda_val


                        # p0.pred_transform.position = x0
                        # p1.pred_transform.position = x1
                        # g1.pred_transform.position = xg1




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
