import os, sys, pygame, numpy as np
from pygame.locals import QUIT

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from engine.transform import Transform
from engine.body import RigidBody
from engine.rod_system import ElasticEdge, OrientationElement, ElasticRod
from engine.core import PBDSolver
from engine.opengl_renderer import OpenGLRenderer
import engine.rod_utils
from engine.rod_generator import RodGenerator


# ───────────────────────── Rope construction ─────────────────────────
ROPE_START   = np.array([1.0, 2.0, 1.0], dtype=float)
SEG_LEN      = 0.5 
NUM_SPHERES  = 7
SPHERE_RAD   = 0.1
DYNAMIC_MASS = 0.1
GHOST_DIST_RATIO = 0.5 
GHOST_MASS_RATIO = 0.25 

particles = []
for i in range(NUM_SPHERES):
    pos  = ROPE_START + np.array([i * SEG_LEN, 0.0, 0.0])
    particles.append(RigidBody(Transform(pos), mass=DYNAMIC_MASS, radius=SPHERE_RAD))


T, N, B  = engine.rod_utils.RodUtils.frenet_frames(particles)

#Generate edges
edges = []
for i in range(NUM_SPHERES - 1):
    p0 = particles[i].transform.position
    p1 = particles[i + 1].transform.position
    rest_len = np.linalg.norm(p1 - p0)
    edges.append(ElasticEdge(i, i + 1, rest_len, 0, 0, 0))

ghost_particles = []
for i in range(len(edges)):
    e = edges[i]
    p0 = particles[e.p0].transform.position
    p1 = particles[e.p1].transform.position

    center = (p0 + p1) * 0.5
    n = N[i]

    ghost_offset = n * SEG_LEN * GHOST_DIST_RATIO
    ghost_pos = center + ghost_offset

    e.g1 = i
    e.ghost_rest_len = SEG_LEN * GHOST_DIST_RATIO

    ghost_rb = RigidBody(
        Transform(ghost_pos),
        mass=DYNAMIC_MASS * GHOST_MASS_RATIO,
        radius=SPHERE_RAD * 0.5
    )
    ghost_particles.append(ghost_rb)


#NOTE: lets do stupid check
particles[0].inv_mass = 0
particles[0].mass = 0
particles[1].inv_mass = 0
particles[1].mass = 0
ghost_particles[0].inv_mass = 0
ghost_particles[0].mass = 0


# particles[len(particles)-1].mass = 5
# particles[len(particles)-1].inv_mass = 1/5


#Generate orientation elements
orientation_elements = []
for i in range(NUM_SPHERES - 2):
    e0 = edges[i]
    e1 = edges[i + 1]
    
    p0 = particles[e0.p0].transform.position
    p1 = particles[e0.p1].transform.position
    p2 = particles[e1.p1].transform.position
    
    g1 = ghost_particles[e0.g1].transform.position
    g2 = ghost_particles[e1.g1].transform.position
    
    frame0 = engine.rod_utils.RodUtils.build_frame(p0, p1, g1)
    frame1 = engine.rod_utils.RodUtils.build_frame(p1, p2, g2)
    arclength = 0.5 * (e0.rest_len + e1.rest_len)
    restDarbouxVector = engine.rod_utils.RodUtils.darboux(frame0, frame1, arclength)
    orientation_elements.append(OrientationElement(i, i + 1, restDarbouxVector))

#Create rope container
elastic_rod = ElasticRod
elastic_rod.particles = particles
elastic_rod.ghost_particles = ghost_particles
elastic_rod.edges = edges
elastic_rod.orientation_elements = orientation_elements


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
    """Return axis_lines + one white segment between every consecutive pair."""
    segs = [ (a.transform.position, b.transform.position, (1,1,1))
             for a, b in zip(particles[:-1], particles[1:]) ]
    return segs + axis_lines

# ───────────────────────── XPBD solver ───────────────────────────────
solver = PBDSolver(
    elastic_rod,
    gravity=np.array([0.0, -9.81, 0]),
    substeps=1,
    iters=5
)

# Particle interaction / drag
selected_particle = None
drag_offset = np.zeros(3)
drag_depth = 0.0  # Distance along the ray for dragging

# ───────────────────────── Rendering loop ───────────────────────────
pygame.init()
pygame.display.set_caption("RIVET")
renderer = OpenGLRenderer()
clock    = pygame.time.Clock()
dt       = 1.0 / 60.0
running  = True

counter = 0
total_frames = 500000

while running:
    for e in pygame.event.get():
        if e.type == QUIT:
            running = False
        elif e.type == pygame.MOUSEWHEEL:
            if selected_particle is not None:
                # Adjust drag depth by scroll amount; tweak sensitivity if needed
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
            # Pick nearest particle in front of camera within a threshold
            min_dist = float('inf')
            for p in elastic_rod.particles:
                to_particle = p.transform.position - cam_pos
                projection = np.dot(to_particle, ray_dir)
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
        solver.step(dt)

    renderer.render(
        elastic_rod.particles + elastic_rod.ghost_particles,
        rope_lines()
    )
    clock.tick(60)
    counter += 1

pygame.quit()
