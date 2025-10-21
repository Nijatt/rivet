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
SEG_LEN         = 0.5
NUM_PARTICLES   = 6
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

# Make one particle heavy (treat as instrument you can grab)
HEAVY_INDEX = NUM_PARTICLES - 1
heavy_particle_inv_mass = 0.01  # smaller inv_mass -> heavier
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
    """Axis lines + one white segment between every consecutive pair of particles."""
    segs = [ (a.transform.position, b.transform.position, (1,1,1))
             for a, b in zip(particles[:-1], particles[1:]) ]
    return segs + axis_lines

def current_chain_length():
    """Sum of current distances between consecutive particles."""
    return sum(
        float(np.linalg.norm(particles[i + 1].transform.position - particles[i].transform.position))
        for i in range(NUM_PARTICLES - 1)
    )

# ───────────────────────── Minimal PBD step ──────────────────────────
def pbd_step(particles, constraints, dt, iters, gravity, damping):
    # 1) save previous positions
    prev = [p.transform.position.copy() for p in particles]

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
            wi = pi.inv_mass
            wj = pj.inv_mass
            if wi == 0.0 and wj == 0.0:
                continue

            xi = pi.transform.position
            xj = pj.transform.position
            delta = xj - xi
            d = float(np.linalg.norm(delta))
            if d < 1e-8:
                continue

            n = delta / d
            C = d - rest
            wsum = wi + wj
            if wsum <= 0.0:
                continue

            # classic PBD distance correction split by inverse-mass
            corr = (C / wsum) * n
            if wi > 0.0:
                pi.transform.position = xi + corr * wi
            if wj > 0.0:
                pj.transform.position = xj - corr * wj

    # 4) update velocities (with damping)
    for p, x_prev in zip(particles, prev):
        new_v = (p.transform.position - x_prev) / dt
        p.vel = damping * new_v

# ───────────────────────── Interaction / drag ────────────────────────
selected_particle = None
drag_offset = np.zeros(3)
drag_depth = 0.0  # Distance along the ray for dragging

# ───────────────────────── Rendering loop ────────────────────────────
pygame.init()
pygame.font.init()

# window title will carry live stats (works with OPENGL)
pygame.display.set_caption("RIVET — Distance Chain")

renderer = OpenGLRenderer()

# try to brighten the scene background
try:
    # if renderer exposes an API
    if hasattr(renderer, "set_clear_color"):
        renderer.set_clear_color((0.22, 0.22, 0.24, 1.0))
    else:
        # fallback: set OpenGL clear color directly
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
        # best-effort direct attributes
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

    if mouse_buttons[0]:  # left click held
        if selected_particle is None:
            # Pick nearest particle under the mouse ray within its radius
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
            # Move selected particle along the ray using updated drag_depth
            if selected_particle.inv_mass > 0:
                selected_particle.transform.position = (
                    cam_pos + drag_depth * ray_dir + drag_offset
                )
                selected_particle.vel = np.zeros(3)  # optional damping
    else:
        selected_particle = None

    if counter < total_frames:
        pbd_step(particles, constraints, dt, ITERS, GRAVITY, DAMPING)

    # 3D render (renderer likely handles buffer swapping internally)
    renderer.render(particles, rope_lines())

    # live HUD in window title (robust with OPENGL)
    cur_len = current_chain_length()
    stretch = cur_len - TOTAL_REST_LENGTH
    pygame.display.set_caption(
        f"RIVET — Distance Chain | dt {dt:.4f} | iters {ITERS} | rest {TOTAL_REST_LENGTH:.3f} | curr {cur_len:.3f} | Δ {stretch:+.3f} | heavy idx {HEAVY_INDEX} inv_m {heavy_particle_inv_mass:.3f}"
    )

    clock.tick(60)
    counter += 1

pygame.quit()
