"""Small SO(3)/SE(3) helpers."""
import numpy as np
from scipy.spatial.transform import Rotation


def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def so3_exp(w):
    th = np.linalg.norm(w)
    if th < 1e-10:
        return np.eye(3) + skew(w)
    k = w / th
    K = skew(k)
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def so3_log(R):
    return Rotation.from_matrix(R).as_rotvec()


def inv_T(T):
    Ti = np.eye(4)
    Ti[:3, :3] = T[:3, :3].T
    Ti[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return Ti


def quat_wxyz_to_R(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def R_from_two_vectors(a, b):
    """Rotation R with R a/|a| = b/|b|."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(a @ b)
    if np.linalg.norm(v) < 1e-12:
        return np.eye(3) if c > 0 else so3_exp(np.array([np.pi, 0, 0]))
    return so3_exp(v / np.linalg.norm(v) * np.arctan2(np.linalg.norm(v), c))
