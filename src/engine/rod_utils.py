import numpy as np


class RodUtils:
    EPSILON8 = 1e-8
    EPSILON6 = 1e-6
    EPSILON4 = 1e-4

    def __init__(self):
        raise NotImplementedError("This class should not be instantiated.")
    # ==================================
    # REGION: Edge Constraints
    @staticmethod
    def edge_constraint(p0, p1, rest_length):
        return np.linalg.norm(p0 - p1) - rest_length

    @staticmethod
    def perpendicular_bisector_constraint(p0, p1, g1):
        left_side = g1 - 0.5 * (p1 + p0)
        right_side = p1 - p0
        return np.dot(left_side, right_side)

    @staticmethod
    def ghost_distance_constraint(p0, p1, g1, ghost_rest_length):
        return np.linalg.norm(0.5 * (p1 + p0) - g1) - ghost_rest_length

    @staticmethod
    def bend_twist_constraint(omega, rest_omega,alpha):
        return alpha*(omega - rest_omega) 
    # ==================================

    # ==================================
    # Regions: jacobians.
    @staticmethod
    def edge_jacobian(p0, p1, epsilon=EPSILON6):
        d = p0 - p1
        len_d = np.linalg.norm(d)

        if len_d < epsilon:
            grad0 = grad1 = np.zeros(3)
        else:
            d_hat = d / len_d
            grad0 = d_hat
            grad1 = -d_hat

        return grad0, grad1

    @staticmethod
    def perp_bisector_jacobian(p0, p1, g):
        grad0 = p0 - g
        grad1 = g - p1
        grad_g = p1 - p0
        return grad0, grad1, grad_g

    @staticmethod
    def ghost_distance_jacobian(p0, p1, g, epsilon=EPSILON6):
        r = g - 0.5 * (p0 + p1)
        len_r = np.linalg.norm(r)

        if len_r < epsilon:
            grad0 = grad1 = grad_g = np.zeros(3)
        else:
            r_hat = r / len_r
            grad0 = -0.5 * r_hat
            grad1 = -0.5 * r_hat
            grad_g = r_hat

        return grad0, grad1, grad_g
    # ==================================

    # ==================================
    # Region: projections
    @staticmethod
    def project_edge(p0, p1, w0, w1, rest_length, stiffness, epsilon=EPSILON6):
        constraint = RodUtils.edge_constraint(p0, p1, rest_length)

        if abs(constraint) < epsilon:
            return p0, p1

        g0, g1 = RodUtils.edge_jacobian(p0, p1)

        denom = w0 + w1

        if denom < epsilon:
            return p0, p1

        lambda_val = -constraint / denom

        if np.isnan(lambda_val):
            lambda_val = 0

        p0_updated = p0 + stiffness * w0 * lambda_val * g0
        p1_updated = p1 + stiffness * w1 * lambda_val * g1

        return p0_updated, p1_updated
    
    @staticmethod
    def project_ghost_distance(p0, p1, g, w0, w1, wG, rest_length, stiffness, epsilon=EPSILON6):
        constraint = RodUtils.ghost_distance_constraint(p0, p1, g, rest_length)
        
        if abs(constraint) < epsilon:
            return p0, p1, g

        grad0, grad1, gradG = RodUtils.ghost_distance_jacobian(p0, p1, g)
        denom = 0.25 * w0 + 0.25 * w1 + wG
        if abs(denom) < epsilon:
            return p0, p1, g

        lambda_val = -constraint / denom
        if np.isnan(lambda_val):
            lambda_val = 0

        p0 += stiffness * w0 * lambda_val * grad0
        p1 += stiffness * w1 * lambda_val * grad1
        g += stiffness * wG * lambda_val * gradG

        return p0, p1, g

    @staticmethod
    def project_perpendicular_bisector(p0, p1, g, w0, w1, wG, stiffness, epsilon=EPSILON6):
        constraint = RodUtils.perpendicular_bisector_constraint(p0, p1, g)
        if abs(constraint) < epsilon:
            return p0, p1, g

        grad0, grad1, gradG = RodUtils.perp_bisector_jacobian(p0, p1, g)

        denom = w0 * np.dot(grad0, grad0) + w1 * np.dot(grad1, grad1) + wG * np.dot(gradG, gradG)
        if abs(denom) < epsilon:
            return p0, p1, g

        lambda_val = -constraint / denom
        if np.isnan(lambda_val):
            lambda_val = 0

        p0 += stiffness * w0 * lambda_val * grad0
        p1 += stiffness * w1 * lambda_val * grad1
        g += stiffness * wG * lambda_val * gradG

        return p0, p1, g
    # ==================================

    # ==================================
    # Region: frame builders 
    @staticmethod
    def build_frame(p0: np.ndarray, p1: np.ndarray, g1: np.ndarray) -> np.ndarray:
        """
        Builds a 3x3 material frame matrix given three points.

        :param p0: Starting point vector.
        :param p1: Ending point vector.
        :param g1: Guide point vector.
        :return: 3x3 numpy array representing the frame matrix.
        """
        d3 = (p1 - p0)
        d3 /= np.linalg.norm(d3)

        d2 = np.cross(d3, g1 - p0)
        d2 /= np.linalg.norm(d2)

        d1 = np.cross(d2, d3)

        frame_matrix = np.column_stack((d1, d2, d3))

        return frame_matrix


    @staticmethod
    def frenet_frames(particles, fallback_up=np.array([0, 1, 0]), epsilon=1e-6):
        T = []
        for i in range(len(particles) - 1):
            e = particles[i+1].transform.position - particles[i].transform.position
            T.append(e / np.linalg.norm(e))

        N = []
        B = []
        for i in range(len(T)):
            # Default: try computing normal from adjacent tangents
            if 0 < i < len(T) - 1:
                dT = T[i+1] - T[i-1]
            elif i < len(T) - 1:
                dT = T[i+1] - T[i]
            elif i > 0:
                dT = T[i] - T[i-1]
            else:
                dT = np.zeros(3)

            norm_dT = np.linalg.norm(dT)

            if norm_dT < epsilon:
                # Pick a fallback normal perpendicular to T[i]
                candidate = fallback_up
                if np.abs(np.dot(T[i], candidate)) > 0.99:
                    candidate = np.array([1, 0, 0])
                b = np.cross(T[i], candidate)
                b /= np.linalg.norm(b)
                n = np.cross(b, T[i])
            else:
                n = dT / norm_dT
                b = np.cross(T[i], n)

            N.append(n)
            B.append(b)

        return T, N, B

    

    #Drawing helper methods
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
    


