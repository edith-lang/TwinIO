"""AUV 15-state error-state Kalman filter (Sec. VI).

Error state x~ = [dth, dv, dp, dbg, dba], R = R^ Exp(dth).
Propagation Eqs. (7)-(9); learned velocity update Eqs. (10)-(11);
delayed USBL fix by rewind-and-replay over a fixed-lag buffer (Sec. VI-C).
"""
from __future__ import annotations

from collections import deque

import numpy as np

from .geometry import skew, so3_exp

G = np.array([0.0, 0.0, -9.81])
TH, V, P, BG, BA = (slice(0, 3), slice(3, 6), slice(6, 9), slice(9, 12), slice(12, 15))


def propagate_cov(Pm, R, f, w, dt, Q):
    """Eqs. (8)-(9)."""
    F = np.eye(15)
    F[TH, TH] = so3_exp(-w * dt)
    F[TH, BG] = -np.eye(3) * dt
    F[V, TH] = -R @ skew(f) * dt
    F[V, BA] = -R * dt
    F[P, V] = np.eye(3) * dt
    return F @ Pm @ F.T + Q


def process_noise(imu, dt):
    """Q = diag(sigma_g^2/dt, sigma_a^2/dt, sigma_bg^2 dt, sigma_ba^2 dt) mapped through G (dt^2 for white terms)."""
    Q = np.zeros((15, 15))
    Q[TH, TH] = np.eye(3) * imu["sigma_g"] ** 2 * dt
    Q[V, V] = np.eye(3) * imu["sigma_a"] ** 2 * dt
    Q[BG, BG] = np.eye(3) * imu["sigma_bg"] ** 2 * dt
    Q[BA, BA] = np.eye(3) * imu["sigma_ba"] ** 2 * dt
    return Q


def joseph(Pm, H, Rn, r=None):
    S = H @ Pm @ H.T + Rn
    K = Pm @ H.T @ np.linalg.inv(S)
    IKH = np.eye(Pm.shape[0]) - K @ H
    Pn = IKH @ Pm @ IKH.T + K @ Rn @ K.T
    return 0.5 * (Pn + Pn.T), (K @ r if r is not None else None), S


class AuvESKF:
    def __init__(self, R0, v0, p0, P0, imu_noise, buffer_s=10.0, dt=0.005):
        self.R, self.v, self.p = R0.copy(), v0.copy(), p0.copy()
        self.bg, self.ba = np.zeros(3), np.zeros(3)
        self.P = P0.copy()
        self.imu = imu_noise
        self.Q = process_noise(imu_noise, dt)
        self.dt_nom = dt
        self.t = 0.0
        self.buf = deque()          # (t, state tuple, P, imu sample, vel-update or None)
        self.buffer_s = buffer_s
        self.applied_fixes = set()
        self.n_vel_updates = 0
        self.n_vel_rejected = 0

    # ---------------------------------------------------------------
    def _state(self):
        return (self.R.copy(), self.v.copy(), self.p.copy(), self.bg.copy(), self.ba.copy())

    def _set_state(self, s):
        self.R, self.v, self.p, self.bg, self.ba = (x.copy() for x in s)

    def _propagate(self, gyro, acc, dt):
        w = gyro - self.bg
        f = acc - self.ba
        R = self.R
        a_w = R @ f + G
        Q = self.Q if abs(dt - self.dt_nom) < 1e-6 else process_noise(self.imu, dt)
        self.P = propagate_cov(self.P, R, f, w, dt, Q)
        self.p = self.p + self.v * dt + 0.5 * a_w * dt * dt         # Eq. (7)
        self.v = self.v + a_w * dt
        self.R = R @ so3_exp(w * dt)
        self.t += dt

    def _inject(self, dx):
        self.R = self.R @ so3_exp(dx[TH])
        self.v += dx[V]
        self.p += dx[P]
        self.bg += dx[BG]
        self.ba += dx[BA]

    def _vel_update(self, z, Rv, gate):
        """Eq. (10): h(x) = R^T v, H = [[h]x  R^T  0 0 0]."""
        h = self.R.T @ self.v
        H = np.zeros((3, 15))
        H[:, TH] = skew(h)
        H[:, V] = self.R.T
        r = z - h
        S = H @ self.P @ H.T + Rv
        if gate > 0 and r @ np.linalg.solve(S, r) > gate:
            self.n_vel_rejected += 1
            return False
        self.P, dx, _ = joseph(self.P, H, Rv, r)
        self._inject(dx)
        self.n_vel_updates += 1
        return True

    def _pos_update(self, z, Rm):
        H = np.zeros((3, 15))
        H[:, P] = np.eye(3)
        self.P, dx, _ = joseph(self.P, H, Rm, z - self.p)
        self._inject(dx)

    # ---------------------------------------------------------------
    def step(self, gyro, acc, dt, vel_meas=None, gate=0.0):
        """One IMU step, optionally followed by a learned-velocity update."""
        self.buf.append((self.t, self._state(), self.P.copy(), (gyro, acc, dt), None))
        self._propagate(gyro, acc, dt)
        if vel_meas is not None:
            self._vel_update(*vel_meas, gate)
            t, s, P, imu, _ = self.buf[-1]
            self.buf[-1] = (t, s, P, imu, vel_meas)
        while self.buf and self.t - self.buf[0][0] > self.buffer_s:
            self.buf.popleft()

    def apply_delayed_fix(self, z, Rm, t_valid, seq_no, gate=0.0):
        """Rewind to t_valid, apply the fix, replay the buffer (Sec. VI-C)."""
        if seq_no in self.applied_fixes:
            return "duplicate"
        if not self.buf or t_valid < self.buf[0][0]:
            return "too_old"
        entries = list(self.buf)
        k = max(i for i, e in enumerate(entries) if e[0] <= t_valid)
        t0, s0, P0, _, _ = entries[k]
        self._set_state(s0)
        self.P = P0.copy()
        self.t = t0
        self._pos_update(z, Rm)
        self.buf = deque(entries[:k])
        for (_, _, _, (g, a, dt), vm) in entries[k:]:
            self.step(g, a, dt, vm, gate)
        self.applied_fixes.add(seq_no)
        return "applied"

    def pos_cov(self):
        return self.P[P, P]
