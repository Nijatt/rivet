import os, sys, pygame, numpy as np
from pygame.locals import QUIT

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from engine.transform import Transform
from engine.body import RigidBody
from engine.core import XPBDSolver
from engine.opengl_renderer import OpenGLRenderer

# ----------------------------
# Create bodies
# ----------------------------
bodies = [
    RigidBody(Transform((-4.0, 6.0, 9.0)), mass=1.0, radius=1.0),
    RigidBody(Transform((-1.5, 8.0, 8.0)), mass=1.0, radius=0.9),
    RigidBody(Transform(( 1.5, 7.0, 7.5)), mass=1.0, radius=1.1),
    RigidBody(Transform(( 4.0, 9.5, 9.5)), mass=1.0, radius=0.95),
    RigidBody(Transform(( 0.0,10.5, 8.5)), mass=1.0, radius=0.75),
    RigidBody(Transform(( 5.0, 8.5, 7.5)), mass=1.0, radius=0.95),
    RigidBody(Transform(( 4.0,10.5, 15.5)), mass=1.0, radius=0.75),
    # Optional static "ground anchor" sphere (mass=0) just to visualize
    RigidBody(Transform((0.0, 0.0, 8.0)), mass=0.0, radius=0.5),
]

AXIS_LEN = 5.0         

axis_lines = [
    #             start-point           end-point              RGB-color (0-1)
    ( (0, 0, 0),  ( AXIS_LEN, 0, 0),    (1.0, 0.0, 0.0) ),   # +X red
    ( (0, 0, 0),  (-AXIS_LEN, 0, 0),    (0.6, 0.2, 0.2) ),   # -X darker red
    ( (0, 0, 0),  (0,  AXIS_LEN, 0),    (0.0, 1.0, 0.0) ),   # +Y green
    ( (0, 0, 0),  (0, -AXIS_LEN, 0),    (0.2, 0.6, 0.2) ),   # -Y darker green
    ( (0, 0, 0),  (0, 0,  AXIS_LEN),    (0.2, 0.5, 1.0) ),   # +Z blue
    ( (0, 0, 0),  (0, 0, -AXIS_LEN),    (0.1, 0.3, 0.7) )    # -Z darker blue
]

# ----------------------------
# XPBD Solver
# ----------------------------
solver = XPBDSolver(bodies,
                    gravity=np.array([0.0, -9.81, 0.0]),
                    substeps=4,
                    iters=4)

# ----------------------------
# Simple collision helpers
# ----------------------------
def collide_sphere_plane(b: RigidBody, plane_y=0.0, restitution=0.2):
    """ Project sphere out of the plane y=plane_y and apply velocity response. """
    # After solver.step we use committed positions
    y_bottom = b.pos[1] - b.radius
    if y_bottom < plane_y:
        penetration = plane_y - y_bottom
        b.pos[1] += penetration  # project out
        if b.vel[1] < 0:
            b.vel[1] = -b.vel[1] * restitution

def collide_sphere_sphere(a: RigidBody, b: RigidBody, restitution=0.2):
    if a.inv_mass == 0 and b.inv_mass == 0:
        return
    delta = b.pos - a.pos
    dist2 = delta.dot(delta)
    rsum = a.radius + b.radius
    if dist2 > rsum * rsum or dist2 == 0.0:
        return
    dist = dist2 ** 0.5
    n = delta / dist
    penetration = rsum - dist
    w = (a.inv_mass + b.inv_mass)
    if w == 0:
        return
    # positional correction (XPBD style simple projection)
    a_off = -n * (penetration * (a.inv_mass / w))
    b_off =  n * (penetration * (b.inv_mass / w))
    a.pos[:] += a_off
    b.pos[:] += b_off
    # velocity response (project relative velocity along contact normal)
    relv = b.vel - a.vel
    vn = relv.dot(n)
    if vn < 0:
        imp = -(1 + restitution) * vn / w
        a.vel += -imp * n * a.inv_mass
        b.vel +=  imp * n * b.inv_mass

# ----------------------------
# Rendering
# ----------------------------
pygame.init()
renderer = OpenGLRenderer()
clock = pygame.time.Clock()

dt = 1.0 / 60.0
running = True

while running:
    for e in pygame.event.get():
        if e.type == QUIT:
            running = False

    # Physics step
    solver.step(dt)

    # Collisions (ground + pairwise spheres)
    for b in bodies:
        if b.inv_mass != 0:
            collide_sphere_plane(b, plane_y=0.0)

    # Naive O(n^2) sphere-sphere
    for i in range(len(bodies)):
        for j in range(i+1, len(bodies)):
            collide_sphere_sphere(bodies[i], bodies[j])

    keys = pygame.key.get_pressed()
    # renderer.camera.handle_input(keys, dt)
    print(renderer.camera)
    # Render
    renderer.render(bodies,axis_lines)
    clock.tick(60)

pygame.quit()


