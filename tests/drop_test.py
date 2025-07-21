# examples/drop_test.py

import os, sys
import pygame
from pygame.locals import QUIT
import numpy as np

# allow imports from src/engine without setting PYTHONPATH
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from engine.body      import RigidBody
from engine.core      import XPBDSolver
from engine.collision import resolve_collision
from engine.renderer  import Renderer


from engine.transform import Transform
from engine.body import RigidBody

# create two circles
t1 = Transform(position=(0.0, 4.0, 3.0))
t2 = Transform(position=(0.0, 0.0, 0.0))

b1 = RigidBody(t1, mass=1.0, radius=0.5)
b2 = RigidBody(t2, mass=0.0, radius=1.0)   # static



solver   = XPBDSolver([b1, b2], substeps=5, iters=5)
renderer = Renderer()

while True:
    for evt in pygame.event.get():
        if evt.type == QUIT:
            pygame.quit()
            sys.exit()

    solver.step(1/60)
    # resolve_collision(b1, b2)
    renderer.render([b1, b2])