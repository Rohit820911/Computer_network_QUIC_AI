"""RTT observation noise, so the simulator can reproduce what the real stack does to RTT samples.

Measured in the lab (experiments/rtt_noise_stats.py on S1 fixed40, no queue): the mean RTT sits
about 1.6 ms above the minimum RTT (median 1.8, p95 2.3, max 4.1 ms) on a 40.5 ms path. That is
serialization plus Python event-loop delay, NOT congestion. A policy that reads RTT relative to the
minimum RTT is fooled by it (policy v1 collapsed to 2.5 Mbit/s on S1 in the lab).

Model: the observed interval-mean RTT = true mean + offset + small per-interval jitter; the observed
minimum stays clean (the minimum picks the luckiest sample), and the mean never falls below it.
Only the OBSERVATION is noisy: the reward is still computed from the true values.
"""


def observe_rtt(rtt_mean, rtt_min, offset_s, jitter_s, rng):
    if rtt_mean is None:
        return None, rtt_min
    m = rtt_mean + offset_s
    if jitter_s:
        m += float(rng.normal(0.0, jitter_s))
    if rtt_min is not None:
        m = max(m, rtt_min)
    return m, rtt_min
