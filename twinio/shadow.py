"""ASV-side shadow of an AUV filter (Sec. VII-A) and the mission plan it runs on."""
from __future__ import annotations

from collections import deque

import numpy as np
from scipy.ndimage import uniform_filter1d

from .eskf import BA, P, TH, V, joseph, process_noise, propagate_cov
from .geometry import so3_log, skew

G = np.array([0.0, 0.0, -9.81])


class MissionPlan:
    """Known survey path, parametrised by arclength s (shared before the mission).

    For the EuRoC prototype the 'planned' path is the ground-truth flight smoothed
    over plan_smooth_s seconds; nominal orientation, turn rate and specific force
    are derived from it.
    """

    def __init__(self, t, p, R, smooth_s, dt):
        n = max(1, int(round(smooth_s / dt)))
        ps = uniform_filter1d(p, n, axis=0, mode="nearest")
        vs = np.gradient(ps, dt, axis=0)
        acc = np.gradient(vs, dt, axis=0)
        self.p, self.v, self.R = ps, vs, R
        self.f = np.einsum("nji,nj->ni", R, acc - G)                       # f_nom = R^T (a_nom - g)
        rot = np.array([so3_log(R[i].T @ R[min(i + 1, len(R) - 1)]) for i in range(len(R))]) / dt
        self.w = uniform_filter1d(rot, n, axis=0, mode="nearest")          # body turn rate
        seg = np.linalg.norm(np.diff(ps, axis=0), axis=1)
        self.s = np.concatenate([[0.0], np.cumsum(seg)])
        self.t = t
        self.V_nom = self.s[-1] / (t[-1] - t[0])

    def index(self, s):
        return int(np.clip(np.searchsorted(self.s, s), 0, len(self.s) - 1))

    def project(self, z, s_guess, window=20.0):
        """Arclength of the path point closest to z, searched near s_guess."""
        m = np.where(np.abs(self.s - s_guess) < window)[0]
        if len(m) == 0:
            m = np.arange(len(self.s))
        return float(self.s[m[np.argmin(np.linalg.norm(self.p[m] - z, axis=1))]])


class Shadow:
    """Covariance-only copy of the AUV filter, propagated with nominal inputs."""

    def __init__(self, plan: MissionPlan, P0, imu_noise, Sigma_v, buffer_s, dt):
        self.plan, self.P = plan, P0.copy()
        self.Q = process_noise(imu_noise, dt)
        self.Sigma_v = Sigma_v
        self.s = 0.0
        self.t = 0.0
        self.dt = dt
        self.buf = deque()          # (t, s, P, aided?)
        self.buffer_s = buffer_s

    def pos(self):
        return self.plan.p[self.plan.index(self.s)]

    def trace_pos(self):
        return float(np.trace(self.P[P, P]))

    def _vel_zero_innov(self, i):
        R, v = self.plan.R[i], self.plan.v[i]
        h = R.T @ v
        H = np.zeros((3, 15))
        H[:, TH] = skew(h)
        H[:, V] = R.T
        self.P, _, _ = joseph(self.P, H, self.Sigma_v)                       # z = h(x_sh): only covariance changes

    def step(self, aided: bool):
        self.buf.append((self.t, self.s, self.P.copy(), aided))
        i = self.plan.index(self.s)
        self.P = propagate_cov(self.P, self.plan.R[i], self.plan.f[i], self.plan.w[i], self.dt, self.Q)
        self.s += self.plan.V_nom * self.dt
        self.t += self.dt
        if aided:
            self._vel_zero_innov(self.plan.index(self.s))
        while self.buf and self.t - self.buf[0][0] > self.buffer_s:
            self.buf.popleft()

    def apply_fix(self, z, Rm, t_valid, t_rx):
        """Same rewind / Joseph / replay as the AUV; then reset the arclength."""
        entries = list(self.buf)
        if not entries or t_valid < entries[0][0]:
            return False
        k = max(i for i, e in enumerate(entries) if e[0] <= t_valid)
        t0, s0, P0, _ = entries[k]
        self.t, self.s, self.P = t0, s0, P0.copy()
        H = np.zeros((3, 15))
        H[:, P] = np.eye(3)
        self.P, _, _ = joseph(self.P, H, Rm)
        self.buf = deque(entries[:k])
        for (_, _, _, aided) in entries[k:]:
            self.step(aided)
        self.s = self.plan.project(z, self.s) + self.plan.V_nom * (t_rx - t_valid)
        return True

    def resync(self, auv_trace):
        """Covariance beacon (Sec. VIII-B): rescale to the AUV's reported trace."""
        self.P *= auv_trace / max(self.trace_pos(), 1e-12)
