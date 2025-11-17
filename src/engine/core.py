# src/engine/core.py
import numpy as np
from .rod_utils import RodUtils

class PBDSolver:
    def __init__(self, elastic_rod, gravity=np.array([0.0, -9.81,0.0]),  iters=8):
        self.elastic_rod, self.gravity = elastic_rod, gravity
        self.iters =  iters
        self.g_norm2 = np.dot(self.gravity, self.gravity) 
        self.linear_damping = 0.95;
    
    def step(self, dt):
        h = dt 

        linear_damping =  1 / (1 + dt * self.linear_damping);
        
        for b in self.elastic_rod.particles:
            if b.inv_mass == 0: continue
            b.vel*=linear_damping;
            b.vel += self.gravity * h

        for e in self.elastic_rod.edges:
            p0, p1  = self.elastic_rod.particles[e.p0], self.elastic_rod.particles[e.p1]
            g1 = self.elastic_rod.ghost_particles[e.g1]

            g1.vel*=linear_damping;
            # b.vel += self.gravity * h  #this is for testing

            if g1.inv_mass == 0: continue

            w0 = p0.inv_mass
            w1 = p1.inv_mass

            v_m_now  = 0.5 * (p0.vel + p1.vel)
            v_m_old = e.edge_mid_prev_velocity
            a_m  = (v_m_now - v_m_old) / dt
            
            r = np.dot(a_m, self.gravity) / self.g_norm2
            
            # r  = np.clip(r, 0.0, 1)

            dv  = (1.0 - r) * dt * self.gravity

            g1.vel -= dv 
            p0.vel += 0.5 * dv 
            p1.vel += 0.5 * dv 

            # s0 = w0/(w0+w1) if (w0+w1)>0 else 0.5
            # s1 = w1/(w0+w1) if (w0+w1)>0 else 0.5

            # if g1.inv_mass!=0: g1.vel -= dv
            # if w0!=0:          p0.vel += s0*dv
            # if w1!=0:          p1.vel += s1*dv

            e.edge_mid_prev_velocity = v_m_now
            g1.pred_transform.position = g1.transform.position + g1.vel * h

        for b in self.elastic_rod.particles:
            if b.inv_mass == 0: continue
            b.pred_transform.position = b.transform.position + b.vel * h


        for _ in range(self.iters):
            for parity in (0, 1):
                for ei in range(parity, len(self.elastic_rod.edges), 2):
                    e = self.elastic_rod.edges[ei]

                    i0, i1, gi = e.p0, e.p1, e.g1
                    p0 = self.elastic_rod.particles[i0]
                    p1 = self.elastic_rod.particles[i1]
                    g1 = self.elastic_rod.ghost_particles[gi]

                    w0, w1, wg = p0.inv_mass, p1.inv_mass, g1.inv_mass
                    if (w0 + w1 + wg) == 0.0:
                        continue

                    # copy predicted positions
                    x0  = p0.pred_transform.position
                    x1  = p1.pred_transform.position
                    xg1 = g1.pred_transform.position

                    # --- ORDER MATTERS: frame first, then length ---
                    # 1) Ghost–edge distance (hard lock)
                    x0, x1, xg1 = RodUtils.project_ghost_distance(
                        x0, x1, xg1, w0, w1, wg, e.ghost_rest_len, 1.0
                    )
                    # 2) Perpendicular bisector (center ghost on midpoint)
                    x0, x1, xg1 = RodUtils.project_perpendicular_bisector(
                        x0, x1, xg1, w0, w1, wg, 0.9
                    )
                    # 3) Edge length (near-hard but slightly < 1 to avoid overshoot)
                    x0, x1 = RodUtils.project_edge(
                        x0, x1, w0, w1, e.rest_len, 0.9
                    )

                    # write back
                    p0.pred_transform.position = x0
                    p1.pred_transform.position = x1
                    g1.pred_transform.position = xg1

                # ----- ORIENTATION (BEND/TWIST) ON THE SAME PARITY -----
                # Apply orientation elements whose "left" edge (edge0) has this parity
                for oe in self.elastic_rod.orientation_elements:
                    if (oe.edge0 % 2) != parity:
                        continue

                    edge0 = self.elastic_rod.edges[oe.edge0]
                    edge1 = self.elastic_rod.edges[oe.edge1]

                    i0, i1, i2 = edge0.p0, edge0.p1, edge1.p1
                    gi1, gi2   = edge0.g1, edge1.g1

                    p0 = self.elastic_rod.particles[i0]
                    p1 = self.elastic_rod.particles[i1]
                    p2 = self.elastic_rod.particles[i2]
                    g1 = self.elastic_rod.ghost_particles[gi1]
                    g2 = self.elastic_rod.ghost_particles[gi2]

                    w0, w1, w2, wg1, wg2 = p0.inv_mass, p1.inv_mass, p2.inv_mass, g1.inv_mass, g2.inv_mass

                    # positions (predicted)
                    x0  = p0.pred_transform.position
                    x1  = p1.pred_transform.position
                    x2  = p2.pred_transform.position
                    xg1 = g1.pred_transform.position
                    xg2 = g2.pred_transform.position

                    # frames and Darboux
                    frame0 = RodUtils.build_frame(x0, x1, xg1)
                    frame1 = RodUtils.build_frame(x1, x2, xg2)
                    arclength = 0.5 * (edge0.rest_len + edge1.rest_len)

                    darboux = RodUtils.darboux(frame0, frame1, arclength)

                    rest_darboux = oe.rest_darboux
                    
                    alpha = np.array([0.3, 0.3, 0.6])  # your material parameters
                    constraint = RodUtils.bend_twist_constraint(darboux, rest_darboux, alpha)

                    x = RodUtils.compute_x(frame0, frame1)

                    status0, dA1p0, dA1p1, dA1g1, dA2p0, dA2p1, dA2g1, dA3p0, dA3p1 = \
                        RodUtils.material_frame_derivatives(x0, x1, xg1, frame0)
                    status1, dB1p1, dB1p2, dB1g2, dB2p1, dB2p2, dB2g2, dB3p1, dB3p2 = \
                        RodUtils.material_frame_derivatives(x1, x2, xg2, frame1)
                    if not status0 or not status1:
                        continue

                    dOmegaDp0 = RodUtils.derivative_omega_p0(frame1, dA1p0, dA2p0, dA3p0, x, darboux, arclength)
                    dOmegaDp1 = RodUtils.derivative_omega_p1(frame0, frame1, dA1p1, dA2p1, dA3p1, dB1p1, dB2p1, dB3p1, x, darboux, arclength)
                    dOmegaDp2 = RodUtils.derivative_omega_p2(frame0, dB1p2, dB2p2, dB3p2, x, darboux, arclength)
                    dOmegaDg1 = RodUtils.derivative_omega_g1(frame1, dA1g1, dA2g1, x, darboux, arclength)
                    dOmegaDg2 = RodUtils.derivative_omega_g2(frame0, dB1g2, dB2g2, x, darboux, arclength)

                    M = (
                        w0  * (dOmegaDp0.T @ dOmegaDp0) +
                        w1  * (dOmegaDp1.T @ dOmegaDp1) +
                        w2  * (dOmegaDp2.T @ dOmegaDp2) +
                        wg1 * (dOmegaDg1.T @ dOmegaDg1) +
                        wg2 * (dOmegaDg2.T @ dOmegaDg2)
                    )

                    # regularize to avoid near-singular bursts on straight segments
                    eps = 1e-8
                    lambda_val = -np.linalg.solve(M + eps*np.eye(3), constraint)

                    # apply with a conservative per-pass stiffness
                    rot_stiffness = 0.2  # 0.2–0.4 is typical
                    p0.pred_transform.position += rot_stiffness * (w0  * (dOmegaDp0 @ lambda_val))
                    p1.pred_transform.position += rot_stiffness * (w1  * (dOmegaDp1 @ lambda_val))
                    p2.pred_transform.position += rot_stiffness * (w2  * (dOmegaDp2 @ lambda_val))
                    g1.pred_transform.position += rot_stiffness * (wg1 * (dOmegaDg1 @ lambda_val))
                    g2.pred_transform.position += rot_stiffness * (wg2 * (dOmegaDg2 @ lambda_val))


        for b in self.elastic_rod.particles:
            b.vel = (b.pred_transform.position - b.transform.position) / h
            b.transform.position = b.pred_transform.position

        for b in self.elastic_rod.ghost_particles:
            b.vel = (b.pred_transform.position - b.transform.position) / h
            b.transform.position = b.pred_transform.position

class ShukurovVelocitySolver:
    def __init__(
        self,
        particles,
        kv=0.2,
        ks=0.8,
        damping=0.05,
        gravity=np.array([0.0, -9.81, 0.0]),
        ground_y=0.0,
        restitution=0.2,
        friction=0.9,
    ):
        """
        particles: list[RigidBody]
        kv: velocity-following strength
        ks: spring/shape strength (0 = no soft shape)
        damping: simple velocity damping factor per step
        gravity: constant acceleration vector
        ground_y: height of ground plane (y = ground_y)
        restitution: bounce factor for vertical velocity
        friction: tangential damping on ground contact
        """
        self.particles = particles
        self.kv = kv
        self.ks = ks
        self.damping = damping
        self.gravity = gravity
        self.ground_y = ground_y
        self.restitution = restitution
        self.friction = friction

        # Cache initial rest distances between all pairs (soft "shape" memory)
        n = len(particles)
        self.rest_lengths = np.zeros((n, n), dtype=float)
        for i in range(n):
            pi = particles[i].transform.position
            for j in range(i + 1, n):
                pj = particles[j].transform.position
                L = np.linalg.norm(pi - pj)
                self.rest_lengths[i, j] = L
                self.rest_lengths[j, i] = L

    def step(self, dt: float):
        n = len(self.particles)
        if n == 0:
            return

        # Read positions & velocities into arrays
        pos = np.zeros((n, 3), dtype=float)
        vel = np.zeros((n, 3), dtype=float)
        inv_mass = np.zeros(n, dtype=float)

        for i, p in enumerate(self.particles):
            pos[i] = p.transform.position
            vel[i] = p.vel
            inv_mass[i] = p.inv_mass

        # Apply gravity to velocities
        for i in range(n):
            if inv_mass[i] == 0.0:
                continue
            vel[i] += self.gravity * dt

        # Pairwise distances
        d = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(i + 1, n):
                dij = np.linalg.norm(pos[i] - pos[j])
                d[i, j] = dij
                d[j, i] = dij

        new_vel = vel.copy()

        for i in range(n):
            if inv_mass[i] == 0.0:
                # Kinematic or fixed particle: skip dynamics
                continue

            v_i = vel[i]
            x_i = pos[i]

            vf = np.zeros(3, dtype=float)  # velocity-following term
            fs = np.zeros(3, dtype=float)  # spring-like soft-shape term

            for j in range(n):
                if i == j:
                    continue

                dij = d[i, j]
                if dij <= 1e-8:
                    continue

                # Distance-based weight: close strong, far weak
                w = 1.0 / (1.0 + dij)

                # Velocity-following
                vf += w * (vel[j] - v_i)

                # Soft rest-length shaping (optional; skip if ks == 0)
                if self.ks != 0.0:
                    Lij = self.rest_lengths[i, j]
                    fs += - (dij - Lij) * (x_i - pos[j]) / dij

            dv = self.kv * vf + self.ks * fs
            new_vel[i] = v_i + dv * dt

        # Global damping
        new_vel *= (1.0 - self.damping)

        # Integrate and handle ground collision
        for i, p in enumerate(self.particles):
            if inv_mass[i] == 0.0:
                continue

            p.vel = new_vel[i]
            p.transform.position = p.transform.position + new_vel[i] * dt

            # ----- simple ground collision: plane y = ground_y -----
            y = p.transform.position[1]
            r = getattr(p, "radius", 0.0)

            floor_level = self.ground_y + r
            if y < floor_level:
                # positional correction
                p.transform.position[1] = floor_level

                # velocity correction: bounce + friction
                if p.vel[1] < 0.0:
                    p.vel[1] = -p.vel[1] * self.restitution
                    p.vel[0] *= self.friction
                    p.vel[2] *= self.friction