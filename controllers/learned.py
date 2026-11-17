"""The learned congestion controller inside aioquic (Phase 6). Inference only.

Registered names:  learned            -> models/ppo_v1.npz
                   learned_<stem>     -> models/<stem>.npz   (every .npz found at import)
Create the .npz from a trained model first:  python sim/export_policy.py models/ppo_v1

How it matches the simulator (common/features.py is the single source of truth):
  * The controller accumulates, per interval, bytes sent, bytes acked, bytes lost and the RTT
    samples (mean weighted by the bytes acked since the previous sample; min).
  * An interval closes when time since its start >= interval_length(srtt) = clip(srtt, 20 ms, 400 ms),
    where srtt = 0.5*srtt + 0.5*(mean RTT of the interval), exactly as in sim/link_sim.py.
  * At the end of an interval: FeatureTracker.update(...) -> state (last 4 intervals) -> policy
    -> one of ACTIONS multiplies the window, then the window is clamped to [2, 1000] packets.
  * Known deviations from the simulator (they are part of the sim-to-real gap, not bugs):
    the first interval starts at the first RTT sample (the sim knows the true RTT up front);
    intervals close on the next callback after the deadline, not on a perfect clock; loss is what
    aioquic declares lost (can include spurious losses).
Every decision is recorded in `decision_rows` (raw stats, state, action, window) so a real run
can be compared with the simulator afterwards; transfer.py writes it next to the log.
"""
import glob
import os
import sys
from collections.abc import Iterable

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from aioquic.quic.congestion.base import QuicCongestionControl, register_congestion_control  # noqa: E402
from aioquic.quic.packet_builder import QuicSentPacket  # noqa: E402

from common.features import STATE_DIM, FeatureTracker, interval_length  # noqa: E402

# Copies of sim constants (tests/test_learned.py fails if they ever drift from the sim's).
ACTIONS = (0.8, 0.95, 1.0, 1.05, 1.25)
MSS = 1200
MIN_CWND_PKTS = 2
MAX_CWND_PKTS = 1000

DECISION_HEADER = (["t", "sent", "acked", "lost", "rtt_mean", "rtt_min", "srtt", "cwnd_before"]
                   + [f"s{i}" for i in range(STATE_DIM)] + ["action", "cwnd_after"])
_WEIGHTS = {}


def load_weights(path):
    if path not in _WEIGHTS:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found. Create it with: python sim/export_policy.py "
                                    f"{path[:-4]}")
        with np.load(path) as z:
            _WEIGHTS[path] = {k: z[k] for k in z.files}
    return _WEIGHTS[path]


class NumpyPolicy:
    def __init__(self, path):
        self.w = load_weights(path)

    def act(self, state):
        w = self.w
        x = np.asarray(state, dtype=np.float64)
        h = np.tanh(w["W0"] @ x + w["b0"])
        h = np.tanh(w["W1"] @ h + w["b1"])
        return int(np.argmax(w["Wa"] @ h + w["ba"]))


class LearnedCongestionControl(QuicCongestionControl):
    POLICY_PATH = os.path.join(ROOT, "models", "ppo_v1.npz")

    def __init__(self, *, max_datagram_size: int, policy=None) -> None:
        super().__init__(max_datagram_size=max_datagram_size)
        self.mss = max_datagram_size
        self.min_cwnd = MIN_CWND_PKTS * max_datagram_size
        self.max_cwnd = MAX_CWND_PKTS * max_datagram_size
        self.policy = policy if policy is not None else NumpyPolicy(self.POLICY_PATH)
        self.tracker = FeatureTracker()
        self.srtt = None
        self.t_start = None          # start of the current interval (set at the first RTT sample)
        self._reset_interval()
        self._acked_since_sample = 0
        self.decision_rows = []
        self.decision_header = DECISION_HEADER

    def _reset_interval(self):
        self.i_sent = self.i_acked = self.i_lost = 0
        self.i_rtt_wsum = self.i_rtt_w = 0.0
        self.i_rtt_min = None

    # ---- accounting callbacks -------------------------------------------------------------
    def on_packet_sent(self, *, packet: QuicSentPacket) -> None:
        self.bytes_in_flight += packet.sent_bytes
        self.i_sent += packet.sent_bytes

    def on_packet_acked(self, *, now: float, packet: QuicSentPacket) -> None:
        self.bytes_in_flight -= packet.sent_bytes
        self.i_acked += packet.sent_bytes
        self._acked_since_sample += packet.sent_bytes
        self._maybe_close(now)

    def on_packets_expired(self, *, packets: Iterable[QuicSentPacket]) -> None:
        for p in packets:
            self.bytes_in_flight -= p.sent_bytes

    def on_packets_lost(self, *, now: float, packets: Iterable[QuicSentPacket]) -> None:
        for p in packets:
            self.bytes_in_flight -= p.sent_bytes
            self.i_lost += p.sent_bytes
        self._maybe_close(now)

    def on_persistent_congestion(self) -> None:
        pass  # the policy decides the window; only the clamps protect against a stall

    def on_rtt_measurement(self, *, now: float, rtt: float) -> None:
        w = max(self._acked_since_sample, 1)
        self._acked_since_sample = 0
        self.i_rtt_wsum += rtt * w
        self.i_rtt_w += w
        self.i_rtt_min = rtt if self.i_rtt_min is None else min(self.i_rtt_min, rtt)
        if self.srtt is None:        # first sample: the first interval starts here
            self.srtt = rtt
            self.t_start = now
        self._maybe_close(now)

    # ---- one decision per interval ---------------------------------------------------------
    def _maybe_close(self, now):
        if self.t_start is None or now - self.t_start < interval_length(self.srtt):
            return
        rtt_mean = self.i_rtt_wsum / self.i_rtt_w if self.i_rtt_w > 0 else None
        cwnd_before = self.congestion_window
        self.tracker.update(acked_bytes=self.i_acked, sent_bytes=self.i_sent, lost_bytes=self.i_lost,
                            rtt_mean=rtt_mean, rtt_min=self.i_rtt_min, cwnd=cwnd_before)
        if rtt_mean is not None:
            self.srtt = 0.5 * self.srtt + 0.5 * rtt_mean
        state = self.tracker.state()
        action = self.policy.act(state)
        new = float(np.clip(cwnd_before * ACTIONS[action], self.min_cwnd, self.max_cwnd))
        if np.isfinite(new):
            self.congestion_window = int(new)
        self.decision_rows.append(
            [round(now, 6), self.i_sent, self.i_acked, self.i_lost,
             "" if rtt_mean is None else rtt_mean, "" if self.i_rtt_min is None else self.i_rtt_min,
             self.srtt, cwnd_before] + [float(v) for v in state] + [action, self.congestion_window])
        self.t_start = now
        self._reset_interval()


def _register():
    register_congestion_control("learned", LearnedCongestionControl)
    for path in sorted(glob.glob(os.path.join(ROOT, "models", "*.npz"))):
        stem = os.path.splitext(os.path.basename(path))[0]
        cls = type(f"Learned_{stem}", (LearnedCongestionControl,), {"POLICY_PATH": path})
        register_congestion_control(f"learned_{stem}", cls)


_register()
