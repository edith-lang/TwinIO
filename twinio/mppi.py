"""MPPI positioning of the ASV (Sec. VII-C, Eqs. (13)-(14)).

ASV: planar unicycle on the surface, state (x, y, psi), control (omega, v).
Stage cost  J(p) = -[Phi(p) - w_cov C(p)] + R_w omega^2 + R_v v^2
Phi(p)      = (log det P_fut - log det P_post) exp(-lambda_d d) - w_comm p_comm
P_post      = (P_fut^-1 + R_m(p)^-1)^-1
C(p)        = (1/N) sum_i (d_i / R_oper)^2
"""
from __future__ import annotations

import numpy as np


class MPPI:
    def __init__(self, c, rng):
        self.c, self.rng = c, rng
        self.U = np.zeros((c["H"], 2))        # warm-started nominal (omega, v)

    def _rollout(self, x0, U):
        c = self.c
        K, H = U.shape[0], U.shape[1]
        X = np.zeros((K, H, 3))
        x = np.repeat(x0[None], K, 0)
        for h in range(H):
            w = np.clip(U[:, h, 0], -c["w_max"], c["w_max"])
            v = np.clip(U[:, h, 1], 0.0, c["v_max"])
            x = x + c["dt"] * np.stack([v * np.cos(x[:, 2]), v * np.sin(x[:, 2]), w], 1)
            X[:, h] = x
        return X

    def plan(self, x0, z_asv, target_p, target_Ppos, auv_ps, usbl_c, loss_prob):
        c = self.c
        K, H = c["K"], c["H"]
        eps = self.rng.normal(0, 1, (K, H, 2)) * np.array([c["noise_w"], c["noise_v"]])
        U = self.U[None] + eps
        X = self._rollout(x0, U)
        pts = np.concatenate([X[..., :2], np.full((K, H, 1), z_asv)], -1).reshape(-1, 3)
        # information gain of the next fix on the target AUV
        d = target_p[None] - pts
        sl = np.linalg.norm(d, axis=1)
        u = d / sl[:, None]
        uu = u[:, :, None] * u[:, None, :]
        Rm = (usbl_c["sigma_r"] ** 2 * uu + (sl * usbl_c["sigma_ang"])[:, None, None] ** 2 * (np.eye(3) - uu)
              + usbl_c["sigma_gps"] ** 2 * np.eye(3))
        Pf = target_Ppos
        Ppost = np.linalg.inv(np.linalg.inv(Pf)[None] + np.linalg.inv(Rm))
        gain = np.linalg.slogdet(Pf)[1] - np.linalg.slogdet(Ppost)[1]
        delay = 2 * sl / usbl_c["sound_speed"] + usbl_c["modem_delay"]
        p_comm = np.clip(loss_prob + (sl / c["R_oper"]) ** 4, 0, 1)
        phi = gain * np.exp(-c["lambda_d"] * delay) - c["w_comm"] * p_comm
        cover = np.mean([(np.linalg.norm(a[None] - pts, axis=1) / c["R_oper"]) ** 2 for a in auv_ps], 0)
        J = (-(phi - c["w_cov"] * cover)).reshape(K, H).sum(1)
        J += (c["R_w"] * U[..., 0] ** 2 + c["R_v"] * U[..., 1] ** 2).sum(1)
        w = np.exp(-(J - J.min()) / c["lambda"])
        w /= w.sum()
        self.U = self.U + np.einsum("k,khc->hc", w, eps)
        u0 = self.U[0].copy()
        self.U = np.vstack([self.U[1:], self.U[-1:]])          # warm start
        return np.array([np.clip(u0[0], -c["w_max"], c["w_max"]), np.clip(u0[1], 0, c["v_max"])])

    @staticmethod
    def move(x, u, dt):
        return x + dt * np.array([u[1] * np.cos(x[2]), u[1] * np.sin(x[2]), u[0]])
