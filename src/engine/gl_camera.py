# gl_camera.py
import numpy as np

def _normalize(v):
    n = np.linalg.norm(v)
    return v if n < 1e-9 else v / n

class Camera:
    def __init__(self,
                 position=(0, 3, 8),
                 target=(0, 1, 0),
                 up=(0, 1, 0),
                 fov_deg=60,
                 z_near=0.1,
                 z_far=100.0):
        self.position = np.array(position, dtype=float)
        self.target   = np.array(target,   dtype=float)
        self.up       = np.array(up,       dtype=float)
        self.fov_deg  = float(fov_deg)
        self.z_near   = float(z_near)
        self.z_far    = float(z_far)

    # -------- matrices --------
    def view_matrix(self):
        f = _normalize(self.target - self.position)     # forward
        s = _normalize(np.cross(f, self.up))            # right
        u = np.cross(s, f)                              # corrected up
        M = np.eye(4, dtype=float)
        M[0,0:3] = s
        M[1,0:3] = u
        M[2,0:3] = f
        T = np.eye(4, dtype=float)
        T[3,0:3] = -self.position   # translation in bottom row (row-major -> column-major)
        return M @ T

    def projection_matrix(self, aspect):
        f = 1.0 / np.tan(np.deg2rad(self.fov_deg) * 0.5)
        zn, zf = self.z_near, self.z_far
        return np.array([
            [f/aspect, 0, 0,                                0],
            [0,        f, 0,                                0],
            [0,        0, (zf+zn)/(zn - zf), (2*zf*zn)/(zn - zf)],
            [0,        0, -1,                               0],
        ], dtype=float)

    # -------- convenience --------
    def set_pose(self, position, target, up=(0,1,0)):
        self.position[:] = position
        self.target[:]   = target
        self.up[:]       = up

    def look_at_origin_high(self):
        self.set_pose((0, 5, 12), (0, 1, 0))

    def look_at_origin_medium(self):
        self.set_pose((0, 3, 8), (0, 1, 0))

    def look_at_origin_close(self):
        self.set_pose((0, 2, 4), (0, 1, 0))

    def orbit(self, radius, yaw, pitch=20.0):
        """Yaw & pitch in degrees."""
        py = np.deg2rad(pitch)
        yw = np.deg2rad(yaw)
        y = radius * np.sin(py)
        r_xy = radius * np.cos(py)
        x = r_xy * np.sin(yw)
        z = r_xy * np.cos(yw)
        self.set_pose((x, y, z), (0, 1, 0))
