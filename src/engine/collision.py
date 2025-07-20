# src/engine/collision.py
import numpy as np

def circle_vs_circle(a, b):
    # returns penetration depth and normal if colliding
    delta = b.pred_pos - a.pred_pos
    dist = np.linalg.norm(delta)
    min_dist = a.radius + b.radius
    if dist < min_dist and dist > 1e-6:
        n = delta / dist
        depth = min_dist - dist
        return depth, n
    return None, None

def resolve_collision(a, b, stiffness=1e4, dt=1/60):
    depth, n = circle_vs_circle(a, b)
    if depth:
        # XPBD constraint: C = (|p_a - p_b| - r)/--- ; λ correction
        # positional correction:
        w1, w2 = a.inv_mass, b.inv_mass
        λ = stiffness * depth / (w1 + w2 + 1e-6) * dt*dt
        a.pred_pos +=  w1 * λ * n
        b.pred_pos += -w2 * λ * n