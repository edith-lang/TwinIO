"""USBL measurement model, Eq. (15)."""
import numpy as np


def usbl_cov(p_asv, p_auv, c):
    d = p_auv - p_asv
    sl = np.linalg.norm(d)
    u = d / max(sl, 1e-9)
    uu = np.outer(u, u)
    return c["sigma_r"] ** 2 * uu + (sl * c["sigma_ang"]) ** 2 * (np.eye(3) - uu) + c["sigma_gps"] ** 2 * np.eye(3)


def usbl_measure(p_asv_true, p_auv_true, c, rng):
    """Returns (z_p, R_m, slant range). Noise on range, two angles and ASV GNSS."""
    d = p_auv_true - p_asv_true
    sl = np.linalg.norm(d)
    az, el = np.arctan2(d[1], d[0]), np.arcsin(d[2] / sl)
    sl_m = sl + rng.normal(0, c["sigma_r"])
    az_m = az + rng.normal(0, c["sigma_ang"])
    el_m = el + rng.normal(0, c["sigma_ang"])
    u = np.array([np.cos(el_m) * np.cos(az_m), np.cos(el_m) * np.sin(az_m), np.sin(el_m)])
    p_asv_meas = p_asv_true + rng.normal(0, c["sigma_gps"], 3)       # ASV knows itself from GNSS
    z = p_asv_meas + sl_m * u
    return z, usbl_cov(p_asv_meas, z, c), sl
