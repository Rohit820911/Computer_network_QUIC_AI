"""Simulator predictions for S5a-S5c, written BEFORE the real runs so they cannot be adjusted.

    python experiments/s5_predict.py                    # aimd, fixed40, oracle
    python experiments/s5_predict.py --model models/ppo_v2.npz   # + your policy

Prints goodput, p95 RTT and loss for each policy. The simulator is slightly optimistic
for window-limited flows (2-7% above the lab for fixed40), so shade goodput down ~5%.
The simulator has no RTT jitter, so a v2 trained WITH noise is shown here without it:
treat its row as an upper bound.
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from sim.baselines import Aimd, FixedWindow, ModelPolicy, run_episode  # noqa: E402
from sim.link_sim import FILE_MB, MAX_CWND, MIN_CWND, scenario_params  # noqa: E402


class Oracle:
    """cwnd = link rate x base RTT (knows the link; an upper reference, not deployable)."""
    def __init__(self, params):
        self.rtt = params.rtt

    def reset(self, sim):
        self.sim = sim
        sim.cwnd = float(np.clip(sim.bw * self.rtt, MIN_CWND, MAX_CWND))

    def act(self, cwnd, stats, state):
        return float(np.clip(self.sim.bw * self.rtt, MIN_CWND, MAX_CWND))


KEYSETS = (("w1", "b1", "w2", "b2", "w3", "b3"),      # sim/export_policy.py (first version)
           ("W0", "b0", "W1", "b1", "Wa", "ba"))      # controllers/learned.py format


class AsModel:
    """Reads either .npz key naming; same forward pass as the sender (tanh, tanh, argmax)."""

    def __init__(self, path):
        z = np.load(path)
        for ks in KEYSETS:
            if all(k in z.files for k in ks):
                self.w = [z[k].astype(np.float64) for k in ks]
                return
        sys.exit(f"{path}: unknown weight names {z.files}")

    def predict(self, state, deterministic=True):
        w1, b1, w2, b2, w3, b3 = self.w
        h = np.tanh(w1 @ np.asarray(state, dtype=np.float64) + b1)
        h = np.tanh(w2 @ h + b2)
        return int(np.argmax(w3 @ h + b3)), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="models/<name>.npz")
    args = ap.parse_args()
    pols = {"aimd (reno-like)": lambda p: Aimd(), "fixed40": lambda p: FixedWindow(),
            "oracle": lambda p: Oracle(p)}
    if args.model:
        pol = AsModel(os.path.join(ROOT, args.model))
        pols["learned"] = lambda p: ModelPolicy(pol)
    print(f"{'scenario':9s} {'policy':18s} {'goodput':>8s} {'p95 RTT ms':>11s} {'loss':>7s}")
    for sc in ("S5a", "S5b", "S5c"):
        p = scenario_params(sc)
        for name, mk in pols.items():
            r = run_episode(p, mk(p), seed=0, size_bytes=FILE_MB[sc] * 1024 * 1024)
            print(f"{sc:9s} {name:18s} {r['goodput_mbps']:8.2f} {r['rtt_p95_ms']:11.1f} "
                  f"{r['loss_ratio']:7.4f}")
        print()


if __name__ == "__main__":
    main()
