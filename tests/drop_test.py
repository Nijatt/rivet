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

# create two circles
b1 = RigidBody(position=[0, 4], mass=1, radius=0.5)
b2 = RigidBody(position=[0, 0], mass=0, radius=1)  # static ground

solver   = XPBDSolver([b1, b2], substeps=5, iters=5)
renderer = Renderer()

while True:
    for evt in pygame.event.get():
        if evt.type == QUIT:
            pygame.quit()
            sys.exit()

    solver.step(1/60)
    resolve_collision(b1, b2)
    renderer.render([b1, b2])