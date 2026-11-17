"""Sanity baselines that run directly on the simulator (Phase 4 gate), plus a runner
that works for any policy, including the trained PPO model later (ModelPolicy).

A policy has reset(sim) and act(cwnd, stats, state) -> new cwnd in bytes, called once
per interval after the interval's statistics are known.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.features import FeatureTracker, interval_length  # noqa: E402
from sim.env import ACTIONS  # noqa: E402
from sim.noise import observe_rtt  # noqa: E402
from sim.link_sim import MAX_CWND, MIN_CWND, MSS, LinkSim  # noqa: E402


class Aimd:
    """Hand-written Reno-like rule: double each interval (slow start) until the first loss
    or until the RTT is 25% above the minimum seen (like aioquic's built-in RTT monitor
    for slow-start exit), then +1 packet per interval; halve on loss. Losses seen within
    2 intervals of a cut are ignored (one congestion event, like Reno's recovery period)."""
    name = "aimd"
    HYSTART_RATIO = 1.25
    # Real Reno reacts on every ACK, not once per RTT. This baseline therefore decides
    # 4 times per RTT (the learned agent keeps the guide's once-per-RTT rule), with the
    # per-RTT growth scaled to match: x2 per RTT in slow start, +1 packet per RTT after.
    interval_fraction = 0.25

    def reset(self, sim):
        self.slow, self.cool, self.min_rtt = True, 0, None

    def act(self, cwnd, stats, state):
        f = self.interval_fraction
        if stats.rtt_min is not None:
            self.min_rtt = stats.rtt_min if self.min_rtt is None else min(self.min_rtt, stats.rtt_min)
        if self.cool > 0:
            self.cool -= 1
            return cwnd
        if stats.lost > 0:
            self.slow, self.cool = False, int(round(2 / f))
            return cwnd * 0.5
        if (self.slow and stats.rtt_mean is not None and self.min_rtt
                and stats.rtt_mean > self.HYSTART_RATIO * self.min_rtt):
            self.slow = False
        return cwnd * 2 ** f if self.slow else cwnd + MSS * f


class FixedWindow:
    """Constant window. 40 packets is about one BDP at 10 Mbit/s and 40 ms, so it is
    tuned for S1 and expected to be poor elsewhere."""
    name = "fixed40"

    def __init__(self, packets=40):
        self.w = packets * MSS

    def reset(self, sim):
        sim.cwnd = self.w

    def act(self, cwnd, stats, state):
        return self.w


class RandomPolicy:
    name = "random"

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def reset(self, sim):
        pass

    def act(self, cwnd, stats, state):
        return cwnd * ACTIONS[int(self.rng.integers(len(ACTIONS)))]


class ModelPolicy:
    """Wraps a trained Stable Baselines3 model (inference only)."""
    name = "learned"

    def __init__(self, model):
        self.model = model

    def reset(self, sim):
        pass

    def act(self, cwnd, stats, state):
        action, _ = self.model.predict(state, deterministic=True)
        return cwnd * ACTIONS[int(action)]


def run_episode(params, policy, seed=0, size_bytes=None, max_time=120.0, trace=False,
                rtt_offset_ms=0.0, rtt_jitter_ms=0.0):
    """Run until size_bytes are delivered (like the lab file transfer) or max_time.
    Returns metrics comparable to results/summary.csv, plus an interval log.
    rtt_offset_ms / rtt_jitter_ms corrupt the RTT the POLICY observes (sim/noise.py), like the
    real stack does (measured: about 1.6 ms); the metrics always use the true values."""
    sim = LinkSim(params, seed=seed)
    sim.keep_trace = True
    tracker = FeatureTracker()
    noise_rng = np.random.default_rng(seed + 99991)
    policy.reset(sim)
    sent = lost = 0.0
    log = []
    t_done = None
    while sim.t < max_time:
        before = sim.total_acked
        frac = getattr(policy, "interval_fraction", 1.0)
        stats = sim.run_interval(max(0.005, interval_length(sim.srtt) * frac))
        rtt_mean, rtt_min = observe_rtt(stats.rtt_mean, stats.rtt_min, rtt_offset_ms / 1000,
                                        rtt_jitter_ms / 1000, noise_rng)
        tracker.update(acked_bytes=stats.acked, sent_bytes=stats.sent, lost_bytes=stats.lost,
                       rtt_mean=rtt_mean, rtt_min=rtt_min, cwnd=stats.cwnd)
        sent += stats.sent
        lost += stats.lost
        log.append((sim.t, stats.cwnd, stats.rtt_mean or 0.0, sim.queue))
        if size_bytes is not None and sim.total_acked >= size_bytes:
            frac = (size_bytes - before) / max(sim.total_acked - before, 1e-9)
            t_done = sim.t - stats.duration * (1 - frac)
            break
        sim.cwnd = float(np.clip(policy.act(sim.cwnd, stats, tracker.state()), MIN_CWND, MAX_CWND))
    t_end = t_done if t_done is not None else sim.t
    rtts = np.asarray(sim.rtt_trace) * 1000
    return dict(
        goodput_mbps=sim.total_acked * 8 / t_end / 1e6 if size_bytes is None
        else size_bytes * 8 / t_end / 1e6,
        rtt_p95_ms=float(np.percentile(rtts, 95)) if len(rtts) else float("nan"),
        loss_ratio=lost / max(sent, 1.0), seconds=t_end, finished=t_done is not None,
        log=log)
