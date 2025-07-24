import os, sys, pygame, numpy as np
from pygame.locals import QUIT

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from engine.transform import Transform
from engine.body import RigidBody
from engine.rod_system import ElasticEdge,OrientationElement,ElasticRod
from engine.core import XPBDSolver
from engine.opengl_renderer import OpenGLRenderer

# ───────────────────────── Rope construction ─────────────────────────
ROPE_START   = np.array([1.0, 10.0, 1.0], dtype=float)   # first sphere centre
SEG_LEN      = 1.0                                      # spacing along +X
NUM_SPHERES  = 10                                       # 10 spheres → length 9
SPHERE_RAD   = 0.2
DYNAMIC_MASS = 1.0                                      # every sphere except anchor
GHOST_DIST_RATIO = 1.0 
GHOST_MASS_RATIO = 1.0 

#Generate particles
particles = []
for i in range(NUM_SPHERES):
    pos  = ROPE_START + np.array([i * SEG_LEN, 0.0, 0.0])
    particles.append(RigidBody(Transform(pos), mass=DYNAMIC_MASS, radius=SPHERE_RAD))

#NOTE: lets do stupid check
particles[0].inv_mass = 0;
particles[0].mass = 0;

#Generate edges
edges = []
for i in range(NUM_SPHERES - 1):
    p0 = particles[i].transform.position
    p1 = particles[i + 1].transform.position
    rest_len = np.linalg.norm(p1 - p0)
    edges.append(ElasticEdge(i, i + 1, rest_len,0,0))

#Generate edges
orientation_elements = []
for i in range(NUM_SPHERES - 1):
    #TODO: get frame element
    # edgeFrame0 = edges[i].get_frame();
    # edgeFrame1 = edges[i + 1].get_frame();
    restDarbouxVector = 0 #TODO: Calculate the darboux
    orientation_elements.append(OrientationElement(i, i + 1, restDarbouxVector))


#Create rope
elastic_rod = ElasticRod;
elastic_rod.particles = particles;
elastic_rod.edges = edges;
elastic_rod.orientation_elements = orientation_elements;


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
solver = XPBDSolver(
    elastic_rod,
    gravity=np.array([0.0, -9.81, 0.0]),
    substeps=1,
    iters=2
)

# ───────────────────────── Simple collision helpers ─────────────────
def collide_sphere_plane(b: RigidBody, plane_y=0.0, restitution=0.2):
    if b.inv_mass == 0:
        return
    y_bottom = b.pos[1] - b.radius
    if y_bottom < plane_y:
        penetration = plane_y - y_bottom
        b.pos[1] += penetration
        if b.vel[1] < 0:
            b.vel[1] = -b.vel[1] * restitution

def collide_sphere_sphere(a: RigidBody, b: RigidBody, restitution=0.2):
    if a.inv_mass == 0 and b.inv_mass == 0:
        return
    delta = b.pos - a.pos
    dist2 = delta.dot(delta)
    rsum  = a.radius + b.radius
    if dist2 >= rsum * rsum or dist2 == 0.0:
        return
    dist = dist2 ** 0.5
    n    = delta / dist
    pen  = rsum - dist
    w    = a.inv_mass + b.inv_mass
    if w == 0:
        return
    a.pos[:] -= n * pen * (a.inv_mass / w)
    b.pos[:] += n * pen * (b.inv_mass / w)
    relv = b.vel - a.vel
    vn   = relv.dot(n)
    if vn < 0:
        imp = -(1 + restitution) * vn / w
        a.vel -= imp * n * a.inv_mass
        b.vel += imp * n * b.inv_mass

# ───────────────────────── Rendering loop ───────────────────────────
pygame.init()
renderer = OpenGLRenderer()
clock    = pygame.time.Clock()
dt       = 1.0 / 60.0
running  = True

counter=0;

while running:
    for e in pygame.event.get():
        if e.type == QUIT:
            running = False
    # if(counter<100):
    #     solver.step(dt)
        
    solver.step(dt)
    # physics
    for b in elastic_rod.particles:
        collide_sphere_plane(b, plane_y=0.0)
    for i in range(len(elastic_rod.particles)):
        for j in range(i+1, len(elastic_rod.particles)):
            collide_sphere_sphere(elastic_rod.particles[i], elastic_rod.particles[j])

    # camera input
    keys = pygame.key.get_pressed()
    renderer.camera.handle_input(keys, dt)
    
    # draw
    renderer.render(elastic_rod.particles, rope_lines())
    clock.tick(60)
    # counter+=1;

pygame.quit()
