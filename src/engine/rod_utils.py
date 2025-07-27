import numpy as np

class RodUtils:
    def __init__(self):
        raise NotImplementedError("This class should not be instantiated.")
    
    #Testing the static class.
    @staticmethod
    def generate_frame(p0, p1, g1):
        frame = np.array([[1,0,0],[0,1,0],[0,0,1]]);
        return frame

    @staticmethod
    def frenet_frames(particles, fallback_up=np.array([0, 1, 0]), epsilon=1e-6):
        T = []
        for i in range(len(particles) - 1):
            e = particles[i+1].transform.position - particles[i].transform.position
            T.append(e / np.linalg.norm(e))

        N = []
        B = []
        for i in range(len(T) - 1):
            dT = T[i+1] - T[i]
            norm_dT = np.linalg.norm(dT)

            if norm_dT < epsilon:
                # Tangents are too similar — fallback needed
                # Pick a normal perpendicular to T[i]
                candidate = fallback_up
                if np.abs(np.dot(T[i], candidate)) > 0.99:  # nearly parallel
                    candidate = np.array([1, 0, 0])  # try another axis
                b = np.cross(T[i], candidate)
                b /= np.linalg.norm(b)
                n = np.cross(b, T[i])
            else:
                n = dT / norm_dT
                b = np.cross(T[i], n)

            N.append(n)
            B.append(b)

        return T, N, B
    
    @staticmethod
    def frenet_frame_lines(particles, T, N, B, scale=0.5):
        lines = []
        for i in range(len(T)):
            p = 0.5*(particles[i + 1].transform.position +particles[i].transform.position) # anchor the frame at end of segment i

            # Scale vectors for visibility
            t = T[i] * scale
            n = N[i] * scale if i < len(N) else np.array([0, 0, 0])
            b = B[i] * scale if i < len(B) else np.array([0, 0, 0])

            lines.append((p, p + t, (1, 0, 0)))  # Tangent → red
            lines.append((p, p + n, (0, 1, 0)))  # Normal → green
            lines.append((p, p + b, (0, 0, 1)))  # Binormal → blue

        return lines


