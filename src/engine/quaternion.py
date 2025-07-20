import numpy as np

def quat_identity():
    return np.array([0.0, 0.0, 0.0, 1.0])

def quat_normalize(q, eps=1e-12):
    n = np.linalg.norm(q)
    if n < eps:
        return quat_identity()
    return q / n

def quat_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
        aw*bw - ax*bx - ay*by - az*bz
    ])

def quat_from_axis_angle(axis, angle):
    axis = np.array(axis, dtype=float)
    axis /= (np.linalg.norm(axis) + 1e-12)
    s = np.sin(angle * 0.5)
    return np.array([axis[0]*s, axis[1]*s, axis[2]*s, np.cos(angle*0.5)])
