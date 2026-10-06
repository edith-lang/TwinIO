"""Cooperative TwinIO simulation on EuRoC IMU data.

Each AUV replays a held-out EuRoC flight (raw IMU, ground truth for scoring and
for simulating USBL fixes) placed in its own survey block at depth. One ASV on
the surface runs a shadow + trigger per AUV, the MPPI planner and the USBL.

Modes
  ins          IMU-only dead reckoning, no aid, no fixes
  aid          learned velocity aid, no fixes
  twinio       learned aid + shadow-triggered USBL fixes        (proposed)
  ins_trig     no aid + shadow-triggered fixes
  fixed        no aid + fixed ping schedule at the unaided interval of Eq. (16)
"""
from __future__ import annotations

import heapq

import numpy as np

from .eskf import P as PS, AuvESKF
from .mppi import MPPI
from .shadow import MissionPlan, Shadow
from .usbl import usbl_measure

MODES = ("ins", "aid", "twinio", "ins_trig", "fixed")


def static_start(seq, cfg, min_hist_s=10.0, horizon_s=60.0):
    """First IMU index after a full aid window where the vehicle is still."""
    n = int(cfg["static_window_s"] / seq.dt)
    i = int(min_hist_s / seq.dt)
    while i + n < len(seq.t) and seq.t[i] < horizon_s:
        if seq.acc[i:i + n].std(0).max() < cfg["static_acc_std"]:
            return i + n
        i += int(0.25 / seq.dt)
    return int(min_hist_s / seq.dt)


def initial_cov(f):
    return np.diag([f["init_att_std"] ** 2] * 3 + [f["init_vel_std"] ** 2] * 3 + [f["init_pos_std"] ** 2] * 3
                   + [f["init_bg_std"] ** 2] * 3 + [f["init_ba_std"] ** 2] * 3)


def filter_noise(cfg):
    """Process-noise densities used by the filter and the shadow: datasheet x q_scale.
    (Sigma_s of the learned aid always uses the datasheet values.)"""
    f, imu = cfg["filter"], cfg["imu"]
    return {k: imu[k] * f.get("q_scale_" + k.split("_", 1)[1], 1.0) for k in ("sigma_g", "sigma_a", "sigma_bg", "sigma_ba")}


class Agent:
    def __init__(self, seq, offset, aid, Sigma_res, kappa, cfg, use_aid):
        self.seq, self.off = seq, np.asarray(offset, float)
        self.i0 = static_start(seq, cfg["filter"])
        i0 = self.i0
        P0 = initial_cov(cfg["filter"])
        imu_q = filter_noise(cfg)
        self.kf = AuvESKF(seq.R[i0], seq.v[i0], seq.p[i0] + self.off, P0, imu_q,
                          cfg["coop"]["buffer_s"], seq.dt)
        self.plan = MissionPlan(seq.t[i0:], seq.p[i0:] + self.off, seq.R[i0:], cfg["coop"]["plan_smooth_s"], seq.dt)
        self.use_aid = use_aid
        self.aid = {}
        if use_aid:
            for j, v, Ss in zip(aid.idx, aid.v, aid.Sigma_s):
                self.aid[int(j)] = (v, Ss + kappa * Sigma_res)
            Sigma_v_shared = aid.Sigma_s.mean(0) + kappa * Sigma_res     # shared before the mission
        else:
            Sigma_v_shared = None
        self.shadow = Shadow(self.plan, P0, imu_q, Sigma_v_shared, cfg["coop"]["buffer_s"], seq.dt)
        self.n = len(seq.t) - i0
        self.last_fix_t = 0.0
        self.log = dict(t=[], p=[], gt=[], tr=[], tr_sh=[], nees=[])
        self.fix_times, self.pings, self.p0_after = [], 0, []

    def gt(self, k):
        return self.seq.p[self.i0 + k] + self.off


def run(cfg, agents_spec, mode, rng_seed=0, verbose=False):
    """agents_spec: list of (seq, offset, AidStream, Sigma_res, kappa). Returns metrics + logs."""
    assert mode in MODES
    c, uc, mc = cfg["coop"], cfg["usbl"], cfg["mppi"]
    rng = np.random.default_rng(rng_seed)
    use_aid = mode in ("aid", "twinio")
    fixes_on = mode in ("twinio", "ins_trig", "fixed")
    agents = [Agent(s, o, a, Sr, kp, cfg, use_aid) for (s, o, a, Sr, kp) in agents_spec]
    dt = agents[0].seq.dt
    n_steps = max(a.n for a in agents)
    asv = np.array([*cfg["scenario"]["asv_start"][:2], 0.0])          # x, y, psi
    z_asv = cfg["scenario"]["asv_start"][2]
    planner = MPPI(mc, rng)
    mppi_every = int(round(mc["dt"] / dt))
    slot_every = int(round(c["T_slot"] / dt))
    fixed_interval = cfg.get("_fixed_interval", None)
    events, seq_no, channel_free_at = [], 0, 0.0
    asv_traj, last_fix_pkt = [], [None] * len(agents)

    for k in range(n_steps):
        t = k * dt
        # ---- AUVs and their shadows -------------------------------------------------
        for a in agents:
            if k >= a.n:
                continue
            j = a.i0 + k
            meas = a.aid.get(j) if a.use_aid else None
            a.kf.step(a.seq.gyro[j], a.seq.acc[j], dt, meas, cfg["aid"].get("gate", 0.0))
            a.shadow.step(meas is not None)
            if k % 20 == 0:
                e = a.kf.p - a.gt(k)
                Pp = a.kf.pos_cov()
                a.log["t"].append(t); a.log["p"].append(a.kf.p.copy()); a.log["gt"].append(a.gt(k))
                a.log["tr"].append(np.trace(Pp)); a.log["tr_sh"].append(a.shadow.trace_pos())
                a.log["nees"].append(float(e @ np.linalg.solve(Pp, e)))
        # ---- ASV motion (MPPI) --------------------------------------------------------
        if fixes_on and k % mppi_every == 0:
            act = [a for a in agents if k < a.n]
            if act:
                tgt = max(act, key=lambda a: a.shadow.trace_pos())
                u = planner.plan(asv, z_asv, tgt.shadow.pos(), tgt.shadow.P[PS, PS],
                                 [a.shadow.pos() for a in act], uc, c["loss_prob"])
                asv = MPPI.move(asv, u, mc["dt"])
            asv_traj.append(asv.copy())
        # ---- delayed events (fix at shadow / AUV) ------------------------------------
        while events and events[0][0] <= t:
            _, _, kind, payload = heapq.heappop(events)
            if kind == "fix_auv":
                i, z, Rm, tv, sq = payload
                a = agents[i]
                tv_local = tv
                if k < a.n and a.kf.apply_delayed_fix(z, Rm, tv_local, sq, cfg["aid"].get("gate", 0.0)) == "applied":
                    a.p0_after.append(np.trace(a.kf.pos_cov()))
        # ---- channel: one exchange per slot ------------------------------------------
        if fixes_on and k % slot_every == 0 and t >= channel_free_at:
            act = [i for i, a in enumerate(agents) if k < a.n]
            if mode == "fixed":
                due = [i for i in act if t - agents[i].last_fix_t >= fixed_interval]
            else:
                due = [i for i in act if agents[i].shadow.trace_pos() > c["sigma_trig"]]
            if due:
                late = [i for i in due if t - agents[i].last_fix_t > c["T_safe"]]
                i = (max(late, key=lambda i: t - agents[i].last_fix_t) if late
                     else max(due, key=lambda i: agents[i].shadow.trace_pos()))
                a = agents[i]
                a.pings += 1
                p_asv = np.array([asv[0], asv[1], z_asv])
                r = np.linalg.norm(a.gt(k) - p_asv)
                tv = t + r / uc["sound_speed"]
                t_rx = t + 2 * r / uc["sound_speed"]
                t_a = t + 3 * r / uc["sound_speed"] + uc["modem_delay"]
                channel_free_at = t_a
                if rng.random() < c["loss_prob"]:             # ping or reply lost: nothing changes
                    continue
                z, Rm, _ = usbl_measure(p_asv, a.gt(k), uc, rng)
                if c["beacon"] and mode != "fixed":
                    rho = a.shadow.trace_pos() / max(np.trace(a.kf.pos_cov()), 1e-12)
                    lo, hi = c["beacon_band"]
                    if not lo <= rho <= hi:
                        a.shadow.resync(np.trace(a.kf.pos_cov()))
                a.shadow.apply_fix(z, Rm, tv, t_rx)
                a.last_fix_t = t
                a.fix_times.append(t)
                seq_no += 1
                pkt = (i, z, Rm, tv, seq_no)
                if c["resend_last_fix"] and last_fix_pkt[i] is not None:
                    heapq.heappush(events, (t_a, seq_no * 2 - 1, "fix_auv", last_fix_pkt[i]))
                if rng.random() >= c["loss_prob"]:            # fix packet delivered
                    heapq.heappush(events, (t_a, seq_no * 2, "fix_auv", pkt))
                last_fix_pkt[i] = pkt
    return summarise(agents, mode, cfg, np.array(asv_traj))


def summarise(agents, mode, cfg, asv_traj):
    out = dict(mode=mode, auvs=[], asv_traj=asv_traj)
    for a in agents:
        p, g = np.array(a.log["p"]), np.array(a.log["gt"])
        e = np.linalg.norm(p - g, axis=1)
        tr_sh, tr = np.array(a.log["tr_sh"]), np.array(a.log["tr"])
        dur = a.log["t"][-1]
        iv = np.diff([0.0] + a.fix_times)
        out["auvs"].append(dict(
            name=a.seq.name, duration_s=dur, ate_rmse=float(np.sqrt(np.mean(e ** 2))),
            final_err=float(e[-1]), max_err=float(e.max()),
            pings=a.pings, fixes=len(a.fix_times),
            mean_fix_interval=float(iv.mean()) if len(iv) else float("nan"),
            emission_rate_per_min=60 * a.pings / dur,
            nees_mean=float(np.mean(a.log["nees"])),
            fidelity_mean=float(np.mean(tr_sh / tr)), fidelity_min=float(np.min(tr_sh / tr)),
            fidelity_max=float(np.max(tr_sh / tr)),
            p0_mean=float(np.mean(a.p0_after)) if a.p0_after else float("nan"),
            vel_updates=a.kf.n_vel_updates,
            log=dict(t=np.array(a.log["t"]), p=p, gt=g, tr=tr, tr_sh=tr_sh)))
    return out
