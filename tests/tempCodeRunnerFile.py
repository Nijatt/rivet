# examples/drop_test_gl.py
import os, sys, pygame
import numpy as np

from pygame.locals import QUIT
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from engine.transform import Transform
from engine.body import RigidBody
from engine.core import XPBDSolver
from engine.collision import resolve_collision  # (optional if integrating collisions into solver)
from engine.opengl_renderer import OpenGLRenderer

# New positions (see table)
b1 = RigidBody(Transform((-2.0, 2.0, 6.0)), mass=1.0, radius=0.8)
b2 = RigidBody(Transform(( 2.0, 0.7, 3.5)), mass=0.0, radius=1.2)
b3 = RigidBody(Transform(( 0.0, 1.5, 5.0)), mass=1.0, radius=0.5)

solver   = XPBDSolver([b1, b2,b3], substeps=5, iters=5)
renderer = OpenGLRenderer()
renderer.camera.look_at_origin_medium()   # camera: (0,3,8) -> (0,1,0)

dt = 1/60
running = True

view = renderer.camera.view_matrix()
for i, b in enumerate([b1, b2,b3]):
    z_view = (view @ np.append(b.transform.position, 1.0))[2]
    print(f"body {i} view-space z = {z_view:.2f}")  # should be negative


while running:
    for e in pygame.event.get():
        if e.type == QUIT:
            running = False

    solver.step(dt)
    # resolve_collision(b1, b2)  # keep if still external
    renderer.render([b1, b2,b3])

pygame.quit()
