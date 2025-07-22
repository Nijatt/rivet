import os, sys, pygame, numpy as np
from pygame.locals import QUIT

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from engine.transform import Transform
from engine.body import RigidBody
from engine.core import XPBDSolver
from engine.opengl_renderer import OpenGLRenderer

# ───────────────────────── Rope construction ─────────────────────────
ROPE_START   = np.array([1.0, 10.0, .0], dtype=float)   # first sphere centre
SEG_LEN      = 1.0                                      # spacing along +X
NUM_SPHERES  = 10                                       # 10 spheres → length 9
SPHERE_RAD   = 0.4
DYNAMIC_MASS = 1.0                                      # every sphere except anchor

bodies = []
for i in range(NUM_SPHERES):
    pos  = ROPE_START + np.array([i * SEG_LEN, 0.0, 0.0])
    mass = 0.0 if i == 0 else DYNAMIC_MASS               # pin first sphere
    bodies.append(RigidBody(Transform(pos), mass=mass, radius=SPHERE_RAD))

# ───────────────────────── Axis helper lines ─────────────────────────
AXIS_LEN = 6.0
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
             for a, b in zip(bodies[:-1], bodies[1:]) ]
    return segs + axis_lines

# ───────────────────────── XPBD solver ───────────────────────────────
solver = XPBDSolver(
    bodies,
    gravity=np.array([0.0, -9.81, 0.0]),
    substeps=4,
    iters=4
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

while running:
    for e in pygame.event.get():
        if e.type == QUIT:
            running = False

    # physics
    solver.step(dt)
    for b in bodies:
        collide_sphere_plane(b, plane_y=0.0)
    for i in range(len(bodies)):
        for j in range(i+1, len(bodies)):
            collide_sphere_sphere(bodies[i], bodies[j])

    # camera input
    keys = pygame.key.get_pressed()
    renderer.camera.handle_input(keys, dt)

    # draw
    renderer.render(bodies, rope_lines())
    clock.tick(60)

pygame.quit()
