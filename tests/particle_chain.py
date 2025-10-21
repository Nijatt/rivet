import os, sys, pygame, numpy as np
from pygame.locals import QUIT

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from engine.transform import Transform
from engine.body import RigidBody
from engine.opengl_renderer import OpenGLRenderer
import engine.gl_camera  # used for screen_to_world_ray

# ───────────────────────── Chain construction ─────────────────────────
ROPE_START      = np.array([1.0, 2.0, 1.0], dtype=float)
SEG_LEN         = 0.2
NUM_PARTICLES   = 10
SPHERE_RAD      = 0.1
DYNAMIC_MASS    = 0.1

# PBD params
ITERS           = 8          # projection iterations per step
GRAVITY         = np.array([0.0, -9.81, 0.0])
DAMPING         = 0.98       # simple velocity damping

# Build particles in a straight line
particles = []
for i in range(NUM_PARTICLES):
    pos = ROPE_START + np.array([i * SEG_LEN, 0.0, 0.0], dtype=float)
    particles.append(RigidBody(Transform(pos), mass=DYNAMIC_MASS, radius=SPHERE_RAD))

# Pin the first particle
particles[0].inv_mass = 0.0
particles[0].mass     = 0.0

# Make one particle heavy (instrument)
HEAVY_INDEX = NUM_PARTICLES - 1
heavy_particle_inv_mass = 0.01  # base inv-mass; effective inv-mass will be scaled in the solver
particles[HEAVY_INDEX].inv_mass = heavy_particle_inv_mass
particles[HEAVY_INDEX].mass     = (1.0 / heavy_particle_inv_mass) if heavy_particle_inv_mass > 0 else 0.0

# Distance-constraint list: (i, j, rest_length)
constraints = []
for i in range(NUM_PARTICLES - 1):
    p0 = particles[i].transform.position
    p1 = particles[i + 1].transform.position
    rest = float(np.linalg.norm(p1 - p0))
    constraints.append((i, i + 1, rest))

# Total rest length of the chain (sum of rest lengths)
TOTAL_REST_LENGTH = sum(rest for (_, _, rest) in constraints)

# ───────────────────────── helpers / visuals ─────────────────────────
AXIS_LEN = 100.0
axis_lines = [
    ((0,0,0), ( AXIS_LEN,0,0), (1,0,0)),
    ((0,0,0), (-AXIS_LEN,0,0), (0.6,0.2,0.2)),
    ((0,0,0), (0, AXIS_LEN,0), (0,1,0)),
    ((0,0,0), (0,-AXIS_LEN,0), (0.2,0.6,0.2)),
    ((0,0,0), (0,0, AXIS_LEN), (0.2,0.5,1)),
    ((0,0,0), (0,0,-AXIS_LEN), (0.1,0.3,0.7)),
]

def rope_lines():
    segs = [ (a.transform.position, b.transform.position, (1,1,1))
             for a, b in zip(particles[:-1], particles[1:]) ]
    return segs + axis_lines

def current_chain_length():
    return sum(
        float(np.linalg.norm(particles[i + 1].transform.position - particles[i].transform.position))
        for i in range(NUM_PARTICLES - 1)
    )

# ───────────────────────── Adaptive multiplier (exp + deadband + smoothing + slew) ─────────────────────────
def compute_stretch(cur_len, rest_len):
    r = cur_len / max(1e-12, rest_len)
    return max(0.0, r - 1.0)

def exp_multiplier_with_deadband(stretch, alpha_min=1e-4, k=25.0, s0=0.015):
    t = max(0.0, stretch - s0)
    return alpha_min + (1.0 - alpha_min) * np.exp(-k * t)

def slew_limit_alpha(alpha_prev, alpha_tgt, dt, rate_down=40.0, rate_up=0.5):
    if alpha_tgt < alpha_prev:
        max_step = rate_down * dt
        return max(alpha_tgt, alpha_prev - max_step)
    else:
        max_step = rate_up * dt
        return min(alpha_tgt, alpha_prev + max_step)

# Tunables
ALPHA_MIN   = 1e-4    # heaviest cap (minimum multiplier)
EXP_K       = 25.0    # falloff speed after deadband
DEADBAND_S0 = 0.015   # ignore <= 1.5% stretch
SLEW_DOWN   = 40.0    # per-second toward heavy (alpha smaller)
SLEW_UP     = 0.5     # per-second toward light (alpha larger)
TAU_DOWN    = 0.03    # s, smoothing when getting heavier (fast)
TAU_UP      = 0.15    # s, smoothing when getting lighter (slow)
DASHPOT_K   = 10.0    # along-rope damping gain

# Runtime state
alpha_state = 1.0
MASS_MULTIPLIER = 1.0
s_prev = 0.0
ANTICIPATION = 0.06   # s, look-ahead for stretch prediction
SDOT_CLAMP   = 2.0    # max stretch rate we trust (per second)

# ───────────────────────── Minimal PBD step ──────────────────────────
def pbd_step(particles, constraints, dt, iters, gravity, damping):
    # 1) save previous positions
    prev = [p.transform.position.copy() for p in particles]

    # 1a) dashpot along last-edge direction for the instrument (before prediction)
    try:
        i0, i1, _ = constraints[-1]
        x0 = particles[i0].transform.position
        x1 = particles[i1].transform.position
        n_last = x1 - x0
        n_norm = np.linalg.norm(n_last)
        if n_norm > 1e-8:
            n_last /= n_norm
            vj = particles[i1].vel
            c = min(max(DASHPOT_K * dt, 0.0), 1.0)
            particles[i1].vel = vj - c * (np.dot(vj, n_last)) * n_last
    except Exception:
        pass

    # 2) external forces & predict positions
    for p in particles:
        if p.inv_mass > 0.0:
            p.vel += gravity * dt
            p.transform.position += p.vel * dt

    # 3) project constraints (distance only)
    for _ in range(iters):
        for (i, j, rest) in constraints:
            pi = particles[i]
            pj = particles[j]

            wi_eff = pi.inv_mass
            wj_eff = pj.inv_mass

            # multiply last particle's inv_mass by adaptive multiplier (smaller -> heavier)
            if j == NUM_PARTICLES - 1:
                wj_eff = wj_eff / MASS_MULTIPLIER

            if wi_eff == 0.0 and wj_eff == 0.0:
                continue

            xi = pi.transform.position
            xj = pj.transform.position
            delta = xj - xi
            d = float(np.linalg.norm(delta))
            if d < 1e-8:
                continue

            n = delta / d
            C = d - rest
            wsum = wi_eff + wj_eff
            if wsum <= 0.0:
                continue

            corr = (C / wsum) * n
            if wi_eff > 0.0:
                pi.transform.position = xi + corr * wi_eff
            if wj_eff > 0.0:
                pj.transform.position = xj - corr * wj_eff

    # 4) update velocities (with damping)
    for p, x_prev in zip(particles, prev):
        new_v = (p.transform.position - x_prev) / dt
        p.vel = damping * new_v

# ───────────────────────── Interaction / drag ────────────────────────
selected_particle = None
drag_offset = np.zeros(3)
drag_depth = 0.0

# ───────────────────────── Rendering loop ────────────────────────────
pygame.init()
pygame.font.init()
pygame.display.set_caption("RIVET — Distance Chain (adaptive exp + dashpot + instant heavy)")

renderer = OpenGLRenderer()

# try to brighten the scene background
try:
    if hasattr(renderer, "set_clear_color"):
        renderer.set_clear_color((0.22, 0.22, 0.24, 1.0))
    else:
        from OpenGL import GL
        GL.glClearColor(0.22, 0.22, 0.24, 1.0)
except Exception:
    pass

# try to position camera reasonably to see the chain
try:
    center = ROPE_START + np.array([(NUM_PARTICLES - 1) * SEG_LEN * 0.5, 0.0, 0.0])
    if hasattr(renderer.camera, "set_look_at"):
        renderer.camera.set_look_at(
            eye=center + np.array([0.0, 0.5, 3.0]),
            target=center,
            up=np.array([0.0, 1.0, 0.0])
        )
    else:
        if hasattr(renderer.camera, "position"):
            renderer.camera.position = center + np.array([0.0, 0.5, 3.0])
        if hasattr(renderer.camera, "target"):
            renderer.camera.target = center
except Exception:
    pass

clock    = pygame.time.Clock()
dt       = 1.0 / 60.0
running  = True

counter = 0
total_frames = 50000

while running:
    for e in pygame.event.get():
        if e.type == QUIT:
            running = False
        elif e.type == pygame.MOUSEWHEEL:
            if selected_particle is not None:
                drag_depth += e.y * 0.5

    keys = pygame.key.get_pressed()
    renderer.camera.handle_input(keys, dt)

    mouse_pos = pygame.mouse.get_pos()
    mouse_buttons = pygame.mouse.get_pressed()

    w, h = renderer.w, renderer.h
    view_matrix = renderer.camera.view_matrix()
    proj_matrix = renderer.camera.projection_matrix(w / h)
    cam_pos, ray_dir = engine.gl_camera.screen_to_world_ray(
        mouse_pos[0], mouse_pos[1], [w, h], view_matrix, proj_matrix
    )

    if mouse_buttons[0]:
        if selected_particle is None:
            min_dist = float('inf')
            for p in particles:
                to_p = p.transform.position - cam_pos
                projection = np.dot(to_p, ray_dir)
                if projection < 0.0:
                    continue
                closest_point = cam_pos + projection * ray_dir
                dist = np.linalg.norm(closest_point - p.transform.position)
                if dist < p.radius and dist < min_dist and p.inv_mass > 0:
                    selected_particle = p
                    drag_offset = p.transform.position - closest_point
                    drag_depth = projection
                    min_dist = dist
        else:
            if selected_particle.inv_mass > 0:
                selected_particle.transform.position = (
                    cam_pos + drag_depth * ray_dir + drag_offset
                )
                selected_particle.vel = np.zeros(3)
    else:
        selected_particle = None

    # --- Adaptive heaviness based on total chain stretch (per-frame) ---
    cur_len_for_control = current_chain_length()
    s = compute_stretch(cur_len_for_control, TOTAL_REST_LENGTH)

    # Predictive feed-forward to reduce perceived latency
    s_dot = (s - s_prev) / max(1e-6, dt)
    s_prev = s
    s_pred = s + max(0.0, min(s_dot, SDOT_CLAMP)) * ANTICIPATION

    # Exponential with deadband, using predicted stretch
    alpha_tgt_raw = exp_multiplier_with_deadband(s_pred, alpha_min=ALPHA_MIN, k=EXP_K, s0=DEADBAND_S0)

    # Asymmetric smoothing (fast when getting heavier, slow when relaxing)
    tau = TAU_DOWN if alpha_tgt_raw < alpha_state else TAU_UP
    beta = 1.0 - np.exp(-dt / max(1e-6, tau))
    alpha_lp = (1.0 - beta) * alpha_state + beta * alpha_tgt_raw

    # Slew-limit the change
    alpha_state = slew_limit_alpha(alpha_state, alpha_lp, dt, rate_down=SLEW_DOWN, rate_up=SLEW_UP)

    MASS_MULTIPLIER = alpha_state  # smaller -> heavier

    if counter < total_frames:
        pbd_step(particles, constraints, dt, ITERS, GRAVITY, DAMPING)

    # 3D render
    renderer.render(particles, rope_lines())

    # HUD
    cur_len = cur_len_for_control
    stretch_len = cur_len - TOTAL_REST_LENGTH
    eff_inv_last = particles[HEAVY_INDEX].inv_mass * MASS_MULTIPLIER
    pygame.display.set_caption(
        f"RIVET — DistChain (exp adapt) | dt {dt:.4f} | iters {ITERS} | rest {TOTAL_REST_LENGTH:.3f} | "
        f"curr {cur_len:.3f} | ΔL {stretch_len:+.3f} | s {s:.3f} | s' {s_dot:.2f} | s* {s_pred:.3f} | "
        f"mult {MASS_MULTIPLIER:.6f} | eff_inv_last {eff_inv_last:.8f}"
    )

    clock.tick(60)
    counter += 1

pygame.quit()
