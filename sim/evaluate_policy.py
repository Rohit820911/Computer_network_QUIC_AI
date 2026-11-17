"""Evaluate a trained policy in the simulator (Phase 5 gate), against simple references.

    python sim/evaluate_policy.py --model models/ppo_v1

Policies: learned (PPO, deterministic), oracle (cwnd = BDP, knows the link: an upper
reference, not deployable), aimd (Reno-like), fixed40, random.
Three test sets:
  lab     S1-S4 exactly as in the lab (file transfers; compare with results/summary.csv)
  random  conditions drawn from the TRAINING ranges (30 s runs, no mid-run bandwidth change)
  heldout one dimension at a time pushed OUTSIDE the training ranges (bandwidth 25-40 Mbit/s,
          RTT 200-300 ms, loss 3-5%, queue 0.1-0.3 BDP): the H2 / S5 generalization tests
Metrics: util = goodput / link rate; delay = p95 RTT / base RTT; loss; and
  score = util - 0.5 * clip(delay - 1, 0, 3) - 2 * loss   (same weights as training reward).
Writes results/eval_<name>.csv and results/sim_learned_behavior_<name>.png.
READ THE BEHAVIOR PLOT: a policy can score well by exploiting the reward.
"""
import argparse
import dataclasses
import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from stable_baselines3 import PPO  # noqa: E402

from sim.baselines import Aimd, FixedWindow, ModelPolicy, RandomPolicy, run_episode  # noqa: E402
from sim.link_sim import (FILE_MB, HEADER, MAX_CWND, MIN_CWND, MSS, TRAIN,  # noqa: E402
                          LinkParams, _bdp, sample_params, scenario_params)

POLICIES = ["learned", "oracle", "aimd", "fixed40", "random"]
COLORS = {"learned": "#1baf7a", "aimd": "#2a78d6", "fixed40": "#eda100", "oracle": "#52514e"}


class Oracle:
    """cwnd = bandwidth x base RTT, re-read every interval (knows the link; upper reference)."""
    name = "oracle"

    def __init__(self, params):
        self.rtt = params.rtt

    def reset(self, sim):
        self.sim = sim
        sim.cwnd = float(np.clip(sim.bw * self.rtt, MIN_CWND, MAX_CWND))

    def act(self, cwnd, stats, state):
        return float(np.clip(self.sim.bw * self.rtt, MIN_CWND, MAX_CWND))


def make(name, params, seed, model):
    return {"learned": lambda: ModelPolicy(model), "oracle": lambda: Oracle(params),
            "aimd": Aimd, "fixed40": FixedWindow, "random": lambda: RandomPolicy(seed)}[name]()


def metrics(params, r):
    link = params.bw_mbps * MSS / (MSS + HEADER)
    util = r["goodput_mbps"] / link
    delay = r["rtt_p95_ms"] / (params.rtt * 1000)
    score = util - 0.5 * min(max(delay - 1, 0), 3) - 2 * r["loss_ratio"]
    return dict(util=util, delay=delay, loss=r["loss_ratio"], score=score)


def heldout_params(dim, rng):
    b = sample_params(rng, TRAIN)
    bw, rtt, loss = b.bw_mbps, b.rtt, b.loss
    qbdp = b.queue_bytes / _bdp(bw, rtt)
    if dim == "bandwidth 25-40 Mbit/s":
        bw = math.exp(rng.uniform(math.log(25), math.log(40)))
    elif dim == "RTT 200-300 ms":
        rtt = rng.uniform(0.2, 0.3)
    elif dim == "loss 3-5%":
        loss = rng.uniform(0.03, 0.05)
    elif dim == "queue 0.1-0.3 BDP":
        qbdp = rng.uniform(0.1, 0.3)
    return LinkParams(bw, rtt, qbdp * _bdp(bw, rtt), loss)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n-random", type=int, default=30)
    ap.add_argument("--n-held", type=int, default=15)
    ap.add_argument("--rtt-offset-ms", type=float, default=0.0,
                    help="extra delay on the RTT the policy OBSERVES (lab measured ~1.6 ms); metrics stay true")
    args = ap.parse_args()
    name = os.path.basename(args.model)
    model = PPO.load(args.model if os.path.isabs(args.model) else os.path.join(ROOT, args.model),
                     device="cpu")
    kw = dict(rtt_offset_ms=args.rtt_offset_ms, rtt_jitter_ms=0.2 if args.rtt_offset_ms > 0 else 0.0)
    tag = name + (f"_noise{args.rtt_offset_ms:g}" if args.rtt_offset_ms > 0 else "")
    rows, logs = [], {}
    rng = np.random.default_rng(1234)

    for sc in ("S1", "S2", "S3", "S4"):
        p = scenario_params(sc)
        for pol in POLICIES:
            r = run_episode(p, make(pol, p, 0, model), seed=0, size_bytes=FILE_MB[sc] * 1024 * 1024, **kw)
            logs[(sc, pol)] = r["log"]
            rows.append(dict(set="lab", case=sc, policy=pol, goodput_mbps=r["goodput_mbps"],
                             **metrics(p, r)))
    for i in range(args.n_random):
        p = dataclasses.replace(sample_params(rng, TRAIN), change_time=None, change_factor=1.0)
        for pol in POLICIES:
            r = run_episode(p, make(pol, p, i, model), seed=i, max_time=30.0, **kw)
            rows.append(dict(set="random", case="train ranges", policy=pol,
                             goodput_mbps=r["goodput_mbps"], **metrics(p, r)))
    for dim in ("bandwidth 25-40 Mbit/s", "RTT 200-300 ms", "loss 3-5%", "queue 0.1-0.3 BDP"):
        for i in range(args.n_held):
            p = heldout_params(dim, rng)
            for pol in POLICIES:
                r = run_episode(p, make(pol, p, i, model), seed=i, max_time=30.0, **kw)
                rows.append(dict(set="heldout", case=dim, policy=pol,
                                 goodput_mbps=r["goodput_mbps"], **metrics(p, r)))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(ROOT, "results", f"eval_{tag}.csv"), index=False)
    pd.set_option("display.width", 200)
    for s in ("lab", "random", "heldout"):
        d = df[df.set == s]
        t = d.groupby(["case", "policy"], sort=False)[["goodput_mbps", "util", "delay", "loss", "score"]].mean()
        t = t.reindex(POLICIES, level=1)
        print(f"\n== {s}   (util = share of link used, delay = p95 RTT / base RTT)")
        print(t.round(3).to_string())

    fig, axes = plt.subplots(2, 4, figsize=(15, 6), sharex="col", facecolor="#fcfcfb")
    for j, sc in enumerate(("S1", "S2", "S3", "S4")):
        for pol in ("aimd", "learned"):
            a = np.asarray(logs[(sc, pol)])
            axes[0, j].plot(a[:, 0], a[:, 1] / 1000, color=COLORS[pol], linewidth=1.4, label=pol)
            ok = a[:, 2] > 0  # intervals without any ACK have no RTT: leave them out of the line
            axes[1, j].plot(a[ok, 0], a[ok, 2] * 1000, color=COLORS[pol], linewidth=1.4)
        axes[0, j].set_title(sc, loc="left")
        axes[1, j].set_xlabel("simulated time (s)")
        for i in (0, 1):
            axes[i, j].spines[["top", "right"]].set_visible(False)
            axes[i, j].yaxis.grid(True, color="#e4e3df")
    axes[0, 0].set_ylabel("cwnd (KB)")
    axes[1, 0].set_ylabel("RTT (ms)")
    axes[0, 0].legend(frameon=False)
    fig.suptitle(f"Learned ({tag}) vs simulated AIMD: look for reward hacking", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "results", f"sim_learned_behavior_{tag}.png"), dpi=140)
    print(f"\nwrote results/eval_{tag}.csv and results/sim_learned_behavior_{tag}.png")


if __name__ == "__main__":
    main()
