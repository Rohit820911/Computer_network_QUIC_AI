"""State features, written ONCE and used by both the simulator and the real aioquic
controller. Never re-implement these anywhere else (mismatched definitions silently
destroy performance).

Once per interval the caller reports what happened in that interval:
    acked_bytes  bytes newly acknowledged
    sent_bytes   bytes sent
    lost_bytes   bytes newly declared lost
    rtt_mean     mean RTT of the samples in the interval (seconds), or None if no samples
    rtt_min      smallest RTT sample in the interval (seconds), or None
    cwnd         congestion window at the end of the interval (bytes)

Per interval the tracker produces 4 numbers, each scaled to about [0, 1]:
    0 delivery   acked_bytes / sent_bytes, clipped to [0, 2], divided by 2
                 (0.5 = balanced, below = window growing or loss, above = window shrinking)
    1 rtt        (rtt_mean / min RTT seen so far, clipped to [1, 5] - 1) / 4
                 (0 = no queueing delay, 1 = 5x the minimum RTT or worse)
    2 loss       lost / (acked + lost), divided by 0.1, clipped to [0, 1]
    3 cwnd_pos   where cwnd sits within the min..max of the last 10 intervals (0.5 if flat)
The state is the last K intervals stacked (K * 4 numbers).

Interval length rule (also shared): INTERVAL = clip(smoothed RTT, 20 ms, 400 ms).
"""
from collections import deque

import numpy as np

K_HISTORY = 4
N_FEATURES = 4
STATE_DIM = K_HISTORY * N_FEATURES
CWND_RANGE_INTERVALS = 10
RTT_RATIO_MAX = 5.0
LOSS_SCALE = 0.1
MIN_INTERVAL_S = 0.020
MAX_INTERVAL_S = 0.400

NEUTRAL = (0.5, 0.0, 0.0, 0.5)


def interval_length(srtt):
    return float(min(max(srtt, MIN_INTERVAL_S), MAX_INTERVAL_S))


class FeatureTracker:
    def __init__(self, k=K_HISTORY):
        self.k = k
        self.reset()

    def reset(self):
        self.min_rtt = None
        self.cwnds = deque(maxlen=CWND_RANGE_INTERVALS)
        self.hist = deque([NEUTRAL] * self.k, maxlen=self.k)

    def update(self, *, acked_bytes, sent_bytes, lost_bytes, rtt_mean, rtt_min, cwnd):
        if rtt_min is not None and rtt_min > 0:
            self.min_rtt = rtt_min if self.min_rtt is None else min(self.min_rtt, rtt_min)

        delivery = acked_bytes / sent_bytes if sent_bytes > 0 else 1.0
        f0 = min(max(delivery, 0.0), 2.0) / 2.0

        if rtt_mean is not None and self.min_rtt:
            ratio = rtt_mean / self.min_rtt
        else:
            ratio = 1.0
        f1 = (min(max(ratio, 1.0), RTT_RATIO_MAX) - 1.0) / (RTT_RATIO_MAX - 1.0)

        total = acked_bytes + lost_bytes
        loss = lost_bytes / total if total > 0 else 0.0
        f2 = min(max(loss / LOSS_SCALE, 0.0), 1.0)

        self.cwnds.append(cwnd)
        lo, hi = min(self.cwnds), max(self.cwnds)
        f3 = (cwnd - lo) / (hi - lo) if hi > lo else 0.5

        self.hist.append((f0, f1, f2, f3))

    def state(self):
        return np.asarray(self.hist, dtype=np.float32).reshape(-1)
