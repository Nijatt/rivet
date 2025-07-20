import numpy as np
import pygame
from pygame.locals import QUIT

class Renderer:
    def __init__(self, width=800, height=600):
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        self.clock  = pygame.time.Clock()

    def render(self, bodies):
        self.screen.fill((30, 30, 30))
        for b in bodies:
            # now np is defined
            x, y = map(int, b.pos * 50 + np.array([400, 300]))
            pygame.draw.circle(self.screen, (200, 200, 20), (x, y), int(b.radius * 50))
        pygame.display.flip()
        self.clock.tick(60)