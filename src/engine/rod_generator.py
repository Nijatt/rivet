import numpy as np

class RodGenerator:
    @staticmethod
    def generate_spiral(num_points=100, radius=1.0, pitch=0.1, turns=2.0, axis='z', start=np.array([0.0, 0.0, 0.0])):
        """
        Generate a 3D spiral curve.

        Parameters:
        - num_points: number of points in the spiral
        - radius: radius of the spiral
        - pitch: vertical distance per full rotation
        - turns: number of full 2π turns
        - axis: axis along which spiral grows ('x', 'y', or 'z')
        - start: starting offset position (np.array shape (3,))

        Returns:
        - np.array of shape (num_points, 3)
        """
        theta = np.linspace(0, 2 * np.pi * turns, num_points)
        z = pitch * theta / (2 * np.pi)

        x = radius * np.cos(theta)
        y = radius * np.sin(theta)

        if axis == 'x':
            coords = np.stack([z, x, y], axis=1)
        elif axis == 'y':
            coords = np.stack([x, z, y], axis=1)
        else:  # default: axis == 'z'
            coords = np.stack([x, y, z], axis=1)

        return coords + start
    
    @staticmethod
    def generate_straight_line(num_points=10, spacing=1.0, direction=np.array([1.0, 0.0, 0.0]), start=np.array([0.0, 0.0, 0.0])):
        """
        Generate a straight rope line.

        Parameters:
        - num_points: number of particles
        - spacing: distance between each point
        - direction: 3D unit vector direction
        - start: starting position (np.array shape (3,))

        Returns:
        - np.array of shape (num_points, 3)
        """
        direction = np.array(direction, dtype=float)
        direction /= np.linalg.norm(direction)

        positions = [start + i * spacing * direction for i in range(num_points)]
        return np.array(positions)
