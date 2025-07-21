# gl_utils.py
import numpy as np

def make_uv_sphere(stacks=12, slices=18):
    verts = []
    normals = []
    indices = []
    for i in range(stacks + 1):
        v = i / stacks
        theta = v * np.pi
        sin_t, cos_t = np.sin(theta), np.cos(theta)
        for j in range(slices + 1):
            u = j / slices
            phi = u * 2 * np.pi
            sin_p, cos_p = np.sin(phi), np.cos(phi)
            x = sin_t * cos_p
            y = cos_t
            z = sin_t * sin_p
            verts.append((x, y, z))
            normals.append((x, y, z))
    for i in range(stacks):
        for j in range(slices):
            a = i * (slices + 1) + j
            b = a + slices + 1
            indices += [a, b, a+1, b, b+1, a+1]
    return (np.array(verts, dtype=np.float32),
            np.array(normals, dtype=np.float32),
            np.array(indices, dtype=np.uint32))
