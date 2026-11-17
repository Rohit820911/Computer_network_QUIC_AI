"""How much RTT noise does the real stack add on top of the true path RTT?

    python experiments/rtt_noise_stats.py                       # logs/S1_fixed40_run*.csv
    python experiments/rtt_noise_stats.py "logs/S4_fixed40_run*.csv"

Use a controller that never builds a queue (fixed40 on S1, S2: 40 packets is about one BDP), so
every RTT above the minimum is NOT congestion but noise: serialization (about 1 ms per packet at
10 Mbit/s), Python event-loop delay, timer granularity. The learned policy reads RTT relative to
the minimum RTT, so this excess is exactly what can fool it. The simulator study (see chat) found
the learned policy v1 collapses once the excess reaches about 0.5-2 ms at 40 ms RTT.
"""
import csv
import glob
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pattern = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "logs", "S1_fixed40_run*.csv")
files = sorted(glob.glob(pattern if os.path.isabs(pattern) else os.path.join(ROOT, pattern)))
if not files:
    sys.exit(f"no files match {pattern}")
allx, mins = [], []
for f in files:
    with open(f) as fh:
        rows = list(csv.DictReader(fh))
    act = [i for i, r in enumerate(rows) if float(r["bytes_in_flight"]) > 0]
    rows = rows[: act[-1] + 1] if act else rows
    rtt = np.array([float(r["rtt_latest"]) for r in rows if float(r["rtt_latest"]) > 0]) * 1000
    # drop the start (slow start / handshake effects): the first second, or the first fifth of a short run
    t = np.array([float(r["t"]) for r in rows if float(r["rtt_latest"]) > 0])
    if len(t) == 0:
        continue
    rtt = rtt[t > min(1.0, 0.2 * t.max())]
    if len(rtt) == 0:
        continue
    mins.append(rtt.min())
    allx.append(rtt - rtt.min())
x = np.concatenate(allx)
print(f"{len(files)} runs, {len(x)} samples. minimum RTT per run: {np.mean(mins):.2f} ms")
print("excess over the run's minimum RTT (ms):  mean %.2f   median %.2f   p95 %.2f   p99 %.2f   max %.2f"
      % (x.mean(), np.median(x), np.percentile(x, 95), np.percentile(x, 99), x.max()))
print(f"mean excess as a share of the minimum RTT: {x.mean() / np.mean(mins) * 100:.1f}%")
