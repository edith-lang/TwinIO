"""Uncertainty of the learned aid (Sec. V).

Sigma_v = Sigma_s + kappa * Sigma_res                                  Eq. (4)
Sigma_s = J Sigma_imu J^T, Sigma_imu = diag(sigma_c^2 / dt)            Eq. (3)
kappa   = max(1, 2 tau_c / T_u)                                        Eq. (6)
"""
from __future__ import annotations

import hashlib
import os

import numpy as np
import torch

from .data import imu_tensor


def query_indices(seq, T, Tu):
    """IMU indices at which the aid is fused: every Tu seconds once a full window exists."""
    step = int(round(Tu / seq.dt))
    return np.arange(T - 1, len(seq.t), step)


@torch.no_grad()
def predict(model, seq, idx, T, batch=64):
    u = torch.from_numpy(imu_tensor(seq))
    out = []
    for b in range(0, len(idx), batch):
        w = torch.stack([u[i - T + 1:i + 1] for i in idx[b:b + batch]])
        out.append(model(w)[:, -1].numpy())
    return np.concatenate(out) if out else np.zeros((0, 3))


def sensor_noise_cov(model, seq, idx, T, imu_noise, dt):
    """Eq. (3): exact Jacobian of v_b(t_q) w.r.t. the raw window, three backward passes."""
    s2 = np.repeat([imu_noise["sigma_g"] ** 2, imu_noise["sigma_a"] ** 2], 3) / dt   # per-sample variance
    s2 = torch.tensor(s2, dtype=torch.float32)
    u = torch.from_numpy(imu_tensor(seq))
    out = np.zeros((len(idx), 3, 3))
    for k, i in enumerate(idx):
        w = u[i - T + 1:i + 1].unsqueeze(0).clone().requires_grad_(True)
        v = model(w)[0, -1]
        J = torch.stack([torch.autograd.grad(v[a], w, retain_graph=a < 2)[0][0] for a in range(3)])  # 3 x T x 6
        Jf = J.reshape(3, -1)
        out[k] = (Jf * s2.repeat(T)) @ Jf.T
    return out


def residual_stats(model, seqs, T, Tu, dense_s=0.1):
    """Sigma_res and tau_c from validation residuals (calibration split)."""
    res, dense = [], []
    for s in seqs:
        idx = query_indices(s, T, Tu)
        res.append(predict(model, s, idx, T) - s.v_body[idx])
        di = query_indices(s, T, dense_s)
        dense.append(predict(model, s, di, T) - s.v_body[di])
    res = np.concatenate(res)
    Sigma_res = np.cov(res.T)
    # tau_c: lag where the autocorrelation first drops below 1/e (mean over axes and sequences)
    taus = []
    for r in dense:
        r = r - r.mean(0)
        for a in range(3):
            x = r[:, a]
            ac = np.correlate(x, x, "full")[len(x) - 1:]
            ac = ac / ac[0]
            below = np.where(ac < np.exp(-1))[0]
            taus.append((below[0] if len(below) else len(ac)) * dense_s)
    return Sigma_res, float(np.mean(taus)), res


def kappa(tau_c, Tu, enabled=True):
    return max(1.0, 2.0 * tau_c / Tu) if enabled else 1.0


def nees(res, covs):
    """Mean normalised estimation error squared, should be 3 for a calibrated 3-D Gaussian."""
    return float(np.mean([r @ np.linalg.solve(C, r) for r, C in zip(res, covs)]))


class AidStream:
    """Precomputed learned-aid measurements (time, v_b, Sigma_s) for one sequence."""

    def __init__(self, model, seq, T, Tu, imu_noise, cache_dir=None, model_tag=""):
        idx = query_indices(seq, T, Tu)
        key = hashlib.md5(f"{model_tag}{seq.name}{T}{Tu}{imu_noise}".encode()).hexdigest()[:10]
        path = os.path.join(cache_dir, f"aid_{seq.name}_{key}.npz") if cache_dir else None
        if path and os.path.exists(path):
            d = np.load(path)
            self.idx, self.v, self.Sigma_s = d["idx"], d["v"], d["Sigma_s"]
            return
        self.idx = idx
        self.v = predict(model, seq, idx, T)
        self.Sigma_s = sensor_noise_cov(model, seq, idx, T, imu_noise, seq.dt)
        if path:
            os.makedirs(cache_dir, exist_ok=True)
            np.savez(path, idx=self.idx, v=self.v, Sigma_s=self.Sigma_s)
