"""EuRoC IMU + ground truth, resampled onto the IMU clock (200 Hz)."""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

SPLITS = {
    "train": ["MH_01_easy", "MH_02_easy", "MH_04_difficult"],
    # calibration and test swap roles (2-fold): Sigma_res fitted on one, tested on the other
    "eval": ["MH_03_medium", "MH_05_difficult"],
}


@dataclass
class ImuSequence:
    name: str
    t: np.ndarray        # s, starting at 0
    gyro: np.ndarray     # N x 3  rad/s
    acc: np.ndarray      # N x 3  m/s^2
    p: np.ndarray        # N x 3  world position (GT)
    v: np.ndarray        # N x 3  world velocity (GT)
    R: np.ndarray        # N x 3 x 3  R_WB (GT)
    v_body: np.ndarray   # N x 3  label, Eq. (1): v_b = R_WB^T v_W

    @property
    def dt(self):
        return float(np.median(np.diff(self.t)))


def _csv(path):
    return np.loadtxt(path, delimiter=",", comments="#", ndmin=2)


def find_sequence_dir(root, name):
    for dp, dn, _ in os.walk(root):
        if os.path.basename(dp) == name and "mav0" in dn:
            return os.path.join(dp, "mav0")
    raise FileNotFoundError(f"{name} (…/{name}/mav0) not found under {root}")


def load_sequence(root: str, name: str) -> ImuSequence:
    m = find_sequence_dir(root, name)
    imu = _csv(os.path.join(m, "imu0", "data.csv"))
    gt = _csv(os.path.join(m, "state_groundtruth_estimate0", "data.csv"))
    ti, tg = imu[:, 0].astype(np.int64), gt[:, 0].astype(np.int64)
    keep = (ti >= tg[0]) & (ti <= tg[-1])
    ti, imu = ti[keep], imu[keep]
    tgs, tis = (tg - tg[0]) * 1e-9, (ti - tg[0]) * 1e-9
    p = np.stack([np.interp(tis, tgs, gt[:, 1 + k]) for k in range(3)], 1)
    v = np.stack([np.interp(tis, tgs, gt[:, 8 + k]) for k in range(3)], 1)
    q = gt[:, [5, 6, 7, 4]]                      # w x y z -> x y z w
    R = Slerp(tgs, Rotation.from_quat(q))(tis).as_matrix()
    v_body = np.einsum("nji,nj->ni", R, v)
    return ImuSequence(name, tis - tis[0], imu[:, 1:4].copy(), imu[:, 4:7].copy(), p, v, R, v_body)


def windows(seq: ImuSequence, T: int, stride: int):
    """Start indices of all full windows."""
    return np.arange(0, len(seq.t) - T + 1, stride)


def imu_tensor(seq: ImuSequence):
    return np.concatenate([seq.gyro, seq.acc], 1).astype(np.float32)   # u_t = [w_t, a_t]
