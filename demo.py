"""TwinIO demo: TwinIO vs. uncorrected IMU dead reckoning on two EuRoC flights.

    python demo.py

Two AUVs replay the IMU of EuRoC MH_03 and MH_05. Each one is run twice:
  uncorrected  plain IMU dead reckoning (no learned aid, no USBL fixes)
  TwinIO       learned velocity aid + ASV shadow-triggered USBL fixes
Prints position errors and saves results/twinio_vs_uncorrected.png
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from twinio.config import load_config
from twinio.covariance import AidStream, kappa, residual_stats
from twinio.data import load_sequence
from twinio.model import VelocityNet
from twinio.sim import run

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    os.chdir(HERE)
    cfg = load_config("config.json")
    cfg["_model_tag"] = "velocity_net"
    model = VelocityNet()
    model.load_state_dict(torch.load(cfg["model_path"]))
    model.eval()
    T = int(round(cfg["train"]["window_s"] * 200))
    Tu = cfg["aid"]["Tu"]
    seqs = [load_sequence("sample_data", n) for n in cfg["scenario"]["auvs"]]

    print("Computing the learned velocity aid and its covariance (a few minutes on CPU)...")
    spec = []
    for i, s in enumerate(seqs):
        other = [x for j, x in enumerate(seqs) if j != i]          # calibrate on the other flight
        Sres, tau, _ = residual_stats(model, other, T, Tu)
        aid = AidStream(model, s, T, Tu, cfg["imu"], cfg["cache_dir"], cfg["_model_tag"])
        spec.append((s, cfg["scenario"]["offsets"][i], aid, Sres, kappa(tau, Tu, cfg["aid"]["use_kappa"])))

    results = {}
    for mode, label in (("ins", "uncorrected"), ("twinio", "TwinIO")):
        print(f"Running {label} ...")
        results[label] = run(cfg, spec, mode, rng_seed=cfg["coop"]["seed"])["auvs"]

    print("\nPosition error (m)              uncorrected        TwinIO     USBL pings")
    for a, b in zip(results["uncorrected"], results["TwinIO"]):
        print(f"  {a['name']:16s} RMSE   {a['ate_rmse']:12.2f}  {b['ate_rmse']:12.2f}")
        print(f"  {'':16s} final  {a['final_err']:12.2f}  {b['final_err']:12.2f}     {b['pings']}")

    os.makedirs("results", exist_ok=True)
    fig, ax = plt.subplots(1, len(seqs), figsize=(6 * len(seqs), 4), squeeze=False)
    for j in range(len(seqs)):
        for label, col in (("uncorrected", "C3"), ("TwinIO", "C0")):
            lg = results[label][j]["log"]
            ax[0, j].semilogy(lg["t"], np.linalg.norm(lg["p"] - lg["gt"], axis=1) + 1e-3, col, label=label)
        ax[0, j].set_title(results["TwinIO"][j]["name"])
        ax[0, j].set_xlabel("time [s]")
        ax[0, j].set_ylabel("position error [m]")
        ax[0, j].grid(alpha=.3)
    ax[0, 0].legend()
    fig.tight_layout()
    fig.savefig("results/twinio_vs_uncorrected.png", dpi=110)
    print("\nPlot saved to results/twinio_vs_uncorrected.png")


if __name__ == "__main__":
    main()
