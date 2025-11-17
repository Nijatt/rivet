import os, sys, pygame, numpy as np
from pygame.locals import QUIT

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from engine.transform import Transform
from engine.body import RigidBody
from engine.core import ShukurovVelocitySolver
from engine.opengl_renderer import OpenGLRenderer
import engine.gl_camera  # for screen_to_world_ray


# ───────────────────────── Triangle construction ─────────────────────────
ROPE_START   = np.array([1.0, 2.0, 1.0], dtype=float)
SEG_LEN      = 0.5
SPHERE_RAD   = 0.1
DYNAMIC_MASS = 0.1

particles = []

# Make a small triangle
p0 = ROPE_START + np.array([0.0,       0.0, 0.0])
p1 = ROPE_START + np.array([SEG_LEN,   0.0, 0.0])
p2 = ROPE_START + np.array([0.5*SEG_LEN, 0.8*SEG_LEN, 0.0])

for pos in (p0, p1, p2):
    particles.append(RigidBody(Transform(pos), mass=DYNAMIC_MASS, radius=SPHERE_RAD))

# Example: keep first particle fixed (optional, can remove if you want all free)
# particles[0].inv_mass = 0.0
# particles[0].mass     = 0.0


# ───────────────────────── Axis helper lines ─────────────────────────
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
    """Return axis_lines + triangle edges."""
    a, b, c = particles
    tri_segments = [
        (a.transform.position, b.transform.position, (1,1,1)),
        (b.transform.position, c.transform.position, (1,1,1)),
        (c.transform.position, a.transform.position, (1,1,1)),
    ]
    return tri_segments + axis_lines


# ───────────────────────── Text helper ─────────────────────────
def draw_text(surface, text, position, color=(255, 255, 0)):
    text_surface = font.render(text, True, color)
    surface.blit(text_surface, position)


# ───────────────────────── Shukurov solver ─────────────────────
solver = ShukurovVelocitySolver(
    particles,
    kv=1,    # velocity-following strength
    ks=1,    # soft-shape spring strength (set 0.0 to disable)
    damping=0
)


# ───────────────────────── Particle drag setup ─────────────────
selected_particle = None
drag_offset = np.zeros(3)
drag_depth = 0.0  # Distance along the ray for dragging


# ───────────────────────── Rendering loop ──────────────────────
pygame.init()
pygame.font.init()
font = pygame.font.SysFont("Arial", 18)

pygame.display.set_caption("RIVET — Shukurov Triangle")
renderer = OpenGLRenderer()
clock    = pygame.time.Clock()
dt       = 1.0 / 30.0
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
            # pick nearest particle in front of camera within a threshold
            min_dist = float("inf")
            for p in particles:
                to_particle = p.transform.position - cam_pos
                projection = np.dot(to_particle, ray_dir)
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

    if counter < total_frames:
        solver.step(dt)

    renderer.render(
        particles,
        rope_lines()
    )

    screen = pygame.display.get_surface()
    draw_text(screen, f"Timestep: {dt:.4f}", (10, 10))
    draw_text(screen, f"Particles: {len(particles)}", (10, 30))

    clock.tick(60)
    counter += 1

pygame.quit()
