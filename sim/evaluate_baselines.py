"""Phase 4 gate: do the hand-written baselines behave sensibly in the simulator, and does
the simulated AIMD look like the real Reno measured in the lab?

    python sim/evaluate_baselines.py

Runs aimd, fixed40 and random on the lab-matching scenarios S1-S4 (same file sizes),
prints a table next to the measured Reno from results/summary.csv (if present), and
saves results/sim_baselines.csv and results/sim_aimd_S1.png (cwnd and RTT over time).
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from sim.baselines import Aimd, FixedWindow, RandomPolicy, run_episode  # noqa: E402
from sim.link_sim import FILE_MB, scenario_params  # noqa: E402

SEEDS = range(5)


def main():
    rows, aimd_s1 = [], None
    for sc in ("S1", "S2", "S3", "S4"):
        size = FILE_MB[sc] * 1024 * 1024
        for mk in (Aimd, FixedWindow, RandomPolicy):
            for seed in SEEDS:
                pol = mk(seed) if mk is RandomPolicy else mk()
                r = run_episode(scenario_params(sc), pol, seed=seed, size_bytes=size)
                if sc == "S1" and mk is Aimd and seed == 0:
                    aimd_s1 = r["log"]
                rows.append(dict(scenario=sc, policy=pol.name, seed=seed,
                                 **{k: v for k, v in r.items() if k != "log"}))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(ROOT, "results", "sim_baselines.csv"), index=False)
    summ = df.groupby(["scenario", "policy"]).agg(
        goodput=("goodput_mbps", "mean"), goodput_std=("goodput_mbps", "std"),
        rtt_p95=("rtt_p95_ms", "mean"), loss=("loss_ratio", "mean"),
        finished=("finished", "all")).reset_index()

    real = os.path.join(ROOT, "results", "summary.csv")
    if os.path.exists(real):
        rr = pd.read_csv(real)
        # simulated aimd is compared with real Reno, simulated fixed40 with real fixed40
        rr["policy"] = rr.cc.map({"reno": "aimd", "fixed40": "fixed40"})
        rr = rr.dropna(subset=["policy"])[["scenario", "policy", "goodput_mean", "rtt_p95_mean",
                                           "loss_ratio_mean"]]
        rr.columns = ["scenario", "policy", "real_goodput", "real_rtt_p95", "real_loss"]
        summ = summ.merge(rr, on=["scenario", "policy"], how="left")
    print(summ.round(3).to_string(index=False))

    if aimd_s1:
        a = np.asarray(aimd_s1)
        fig, ax = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True)
        ax[0].plot(a[:, 0], a[:, 1] / 1000, color="#2a78d6", linewidth=1.5)
        ax[0].set_ylabel("cwnd (KB)")
        ax[1].plot(a[:, 0], a[:, 2] * 1000, color="#2a78d6", linewidth=1.5)
        ax[1].set_ylabel("RTT (ms)")
        ax[1].set_xlabel("simulated time (s)")
        ax[0].set_title("Simulated AIMD, S1 (10 Mbit/s, 40 ms, 1 BDP queue)", loc="left")
        for x in ax:
            x.spines[["top", "right"]].set_visible(False)
            x.yaxis.grid(True, color="#e4e3df")
        fig.tight_layout()
        fig.savefig(os.path.join(ROOT, "results", "sim_aimd_S1.png"), dpi=150)


if __name__ == "__main__":
    main()
