import numpy as np
import math
import pygame

# ───────────────────────── helpers ─────────────────────────────
def _normalize(v):
    n = np.linalg.norm(v)
    return v if n < 1e-9 else v / n

def _deg2rad(d): return d * math.pi / 180.0

def screen_to_world_ray(mouse_x, mouse_y, screen_size, view_matrix, proj_matrix):
    """
    Convert screen coordinates to a world-space ray.
    """
    x = (2.0 * mouse_x) / screen_size[0] - 1.0
    y = 1.0 - (2.0 * mouse_y) / screen_size[1]  # flip y
    ray_clip = np.array([x, y, -1.0, 1.0])

    inv_proj = np.linalg.inv(proj_matrix)
    inv_view = np.linalg.inv(view_matrix)

    ray_eye = inv_proj @ ray_clip
    ray_eye = np.array([ray_eye[0], ray_eye[1], -1.0, 0.0])

    ray_world = inv_view @ ray_eye
    ray_world = ray_world[:3]
    ray_world /= np.linalg.norm(ray_world)

    cam_origin = inv_view[:3, 3]
    return cam_origin, ray_world

# ───────────────────────── Camera ──────────────────────────────
class Camera:
    def __init__(self,
                 position=(0, 3, 8),
                 target  =(0, 1, 0),
                 up      =(0, 1, 0),
                 fov_deg =60,
                 z_near  =0.1,
                 z_far   =100.0):
        self.position = np.array(position, dtype=np.float32)
        self.target   = np.array(target,   dtype=np.float32)
        self.up       = np.array(up,       dtype=np.float32)
        self.fov_deg  = float(fov_deg)
        self.z_near   = float(z_near)
        self.z_far    = float(z_far)

        # internal yaw / pitch (radians) – initialise from position->target
        dir_vec = _normalize(self.target - self.position)
        self.yaw   = math.atan2(dir_vec[0], dir_vec[2])          # around Y
        self.pitch = math.asin(dir_vec[1])                       # up/down

    # ───────── public matrices ─────────
    def view_matrix(self):
        forward = _normalize(self.target - self.position)
        right   = _normalize(np.cross(forward, self.up))
        up_corr = np.cross(right, forward)

        M = np.eye(4, dtype=np.float32)
        M[0, 0:3] = right
        M[1, 0:3] = up_corr
        M[2, 0:3] = -forward
        M[0, 3]   = -np.dot(right,   self.position)
        M[1, 3]   = -np.dot(up_corr, self.position)
        M[2, 3]   =  np.dot(forward, self.position)
        return M

    def projection_matrix(self, aspect):
        f = 1.0 / math.tan(_deg2rad(self.fov_deg) * 0.5)
        zn, zf = self.z_near, self.z_far
        return np.array([[f/aspect,0,0,0],
                         [0,f,0,0],
                         [0,0,(zf+zn)/(zn-zf),(2*zf*zn)/(zn-zf)],
                         [0,0,-1,0]], dtype=np.float32)

    # ───────── convenience presets ─────────
    def set_pose(self, position, target, up=(0,1,0)):
        self.position[:] = position
        self.target[:]   = target
        self.up[:]       = up
        # update yaw/pitch to match new direction
        dir_vec = _normalize(self.target - self.position)
        self.yaw   = math.atan2(dir_vec[0], dir_vec[2])
        self.pitch = math.asin(dir_vec[1])

    def look_at_origin_medium(self):
        self.set_pose((0,3,8),(0,1,0))

    # ───────── interactive control ─────────
    def handle_input(self, keys, dt,
                     move_speed=5.0,
                     rot_speed_deg=60.0):
        """
        keys : pygame.key.get_pressed()
        dt   : seconds since last frame
        """
        # ----- rotation first (num-pad) -----
        rot_step = _deg2rad(rot_speed_deg) * dt
        if keys[pygame.K_KP4]: self.yaw   -= rot_step
        if keys[pygame.K_KP6]: self.yaw   += rot_step
        if keys[pygame.K_KP8]: self.pitch += rot_step
        if keys[pygame.K_KP2]: self.pitch -= rot_step
        self.pitch = np.clip(self.pitch, _deg2rad(-85), _deg2rad(85))

        # recompute forward & right from new yaw/pitch
        cos_p = math.cos(self.pitch)
        forward = np.array([math.sin(self.yaw)*cos_p,
                            math.sin(self.pitch),
                            math.cos(self.yaw)*cos_p], dtype=np.float32)
        forward = _normalize(forward)
        right = _normalize(np.cross(forward, np.array([0,1,0],np.float32)))

        # ----- translation (WASD + R/F) -----
        move = np.zeros(3, dtype=np.float32)
        if keys[pygame.K_w]: move += forward
        if keys[pygame.K_s]: move -= forward
        if keys[pygame.K_d]: move += right
        if keys[pygame.K_a]: move -= right
        if keys[pygame.K_r]: move += np.array([0,1,0],np.float32)
        if keys[pygame.K_f]: move -= np.array([0,1,0],np.float32)

        if np.linalg.norm(move) > 0:
            move = _normalize(move) * move_speed * dt
            self.position += move
            self.target   += move  # keep same look direction

        # update target based on yaw/pitch
        self.target = self.position + forward
