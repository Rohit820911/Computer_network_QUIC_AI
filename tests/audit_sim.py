"""Pre-training audit of the simulator, shared features and reward (run before Phase 5).

    python tests/audit_sim.py

Every check prints PASS, WARN or FAIL. FAIL = something is wrong and must be fixed before
training. WARN = not wrong, but know it (and mention it in the report where relevant).
"""
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from common.features import STATE_DIM, FeatureTracker  # noqa: E402
from sim.env import CCEnv  # noqa: E402
from sim.link_sim import (MAX_CWND, MIN_CWND, MSS, LinkSim,  # noqa: E402
                          _bdp, sample_params, scenario_params)

counts = {"PASS": 0, "WARN": 0, "FAIL": 0}


def check(name, ok, detail="", warn_only=False):
    tag = "PASS" if ok else ("WARN" if warn_only else "FAIL")
    counts[tag] += 1
    print(f"[{tag}] {name}" + (f"  ({detail})" if detail else ""))


def t_features():
    rng = np.random.default_rng(0)
    tr = FeatureTracker()
    ok_shape = ok_range = True
    for ep in range(200):
        tr.reset()
        for _ in range(30):
            sent = rng.choice([0.0, rng.uniform(0, 5e5)])
            acked = rng.choice([0.0, rng.uniform(0, 1e6)])
            lost = rng.choice([0.0, rng.uniform(0, 1e6)])
            rtt = rng.choice([None, rng.uniform(1e-4, 2.0)])
            tr.update(acked_bytes=acked, sent_bytes=sent, lost_bytes=lost, rtt_mean=rtt,
                      rtt_min=None if rtt is None else rtt * rng.uniform(0.3, 1.0),
                      cwnd=rng.uniform(MIN_CWND, MAX_CWND))
            s = tr.state()
            ok_shape &= s.shape == (STATE_DIM,)
            ok_range &= bool(np.all(np.isfinite(s)) and s.min() >= 0 and s.max() <= 1)
    check("features: shape and finite", ok_shape)
    check("features: all values in [0,1] incl. zero-sent, no-RTT, loss>acked", ok_range)
    tr.reset()
    check("features: fresh tracker gives neutral state", np.allclose(tr.state()[:4], [0.5, 0, 0, 0.5]))


def run_policy_sim(params, seed, steps=300, cwnd_fn=None):
    sim = LinkSim(params, seed=seed)
    S = A = L = 0.0
    qmax_ok = True
    neg_inflight = False
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        if cwnd_fn:
            sim.cwnd = cwnd_fn(sim, rng)
        st = sim.run_interval(max(0.02, min(sim.srtt, 0.4)))
        S += st.sent
        A += st.acked
        L += st.lost
        qmax_ok &= sim.queue <= params.queue_bytes + 1e-6
        neg_inflight |= sim.inflight < -1e-9
    return sim, S, A, L, qmax_ok, neg_inflight


def t_sim_invariants():
    rng = np.random.default_rng(1)
    worst, qok, bwok, neg = 0.0, True, True, False
    for i in range(40):
        p = sample_params(rng)
        rand_cwnd = lambda s, r: float(np.clip(s.cwnd * r.choice([0.8, 0.95, 1.0, 1.05, 1.25]),
                                               MIN_CWND, MAX_CWND))
        sim, S, A, L, q, n = run_policy_sim(p, i, cwnd_fn=rand_cwnd)
        worst = max(worst, abs(S - A - L - sim.inflight) / max(S, 1.0))
        qok &= q
        neg |= n
        maxbw = p.bw_mbps * 1e6 / 8 * max(1.0, p.change_factor)
        bwok &= A <= maxbw * sim.t * 1.0001
    check("sim: bytes conserved (sent = acked + lost + inflight)", worst < 1e-6, f"worst rel err {worst:.2e}")
    check("sim: queue never exceeds its limit", qok)
    check("sim: never delivers more than link capacity", bwok)
    check("sim: inflight never negative", not neg)


def t_determinism():
    p = scenario_params("S2")
    a = run_policy_sim(p, 7)[0].total_acked
    b = run_policy_sim(p, 7)[0].total_acked
    c = run_policy_sim(p, 8)[0].total_acked
    check("sim: same seed gives identical run", a == b)
    check("sim: different seed changes random-loss runs", a != c, warn_only=True)


def t_leakage():
    rng = np.random.default_rng(2)
    P = [sample_params(rng) for _ in range(20000)]
    bw = max(p.bw_mbps for p in P)
    bw_after = max(p.bw_mbps * p.change_factor for p in P)
    rtt = max(p.rtt for p in P) * 1000
    loss = max(p.loss for p in P)
    qmin = min(p.queue_bytes / _bdp(p.bw_mbps, p.rtt) for p in P)
    print(f"      training sample maxima: bw {bw:.1f}, bw after change {bw_after:.1f} Mbit/s, "
          f"rtt {rtt:.0f} ms, loss {loss:.3f}, min queue {qmin:.2f} BDP")
    check("held-out: start bandwidth never above 20 Mbit/s", bw <= 20.0 + 1e-9)
    check("held-out: bandwidth AFTER a mid-episode change never above 20 Mbit/s",
          bw_after <= 20.0 + 1e-9, f"max seen in training: {bw_after:.1f} Mbit/s")
    check("held-out: RTT never above 120 ms", rtt <= 120.0 + 1e-9)
    check("held-out: loss never above 2%", loss <= 0.02 + 1e-12)
    check("held-out: queue never below 0.5 BDP", qmin >= 0.5 - 1e-9)


def policy_returns(make_policy, seeds):
    env = CCEnv()
    R, thr, dly, los = [], [], [], []
    for s in seeds:
        env.reset(seed=s)
        pol = make_policy(env)
        tot, n = 0.0, 0
        parts = []
        for _ in range(env.max_steps):
            act = pol(env)
            _, r, _, trunc, info = env.step(act)
            tot += r
            n += 1
            parts.append((info["thr"], info["delay"], info["loss"]))
            if trunc:
                break
        R.append(tot / n)
        a = np.mean(parts, axis=0)
        thr.append(a[0]); dly.append(a[1]); los.append(a[2])
    return np.mean(R), np.mean(thr), np.mean(dly), np.mean(los), np.max(thr)


def t_reward():
    seeds = range(100, 130)

    def absolute(fn):
        def make(env):
            def act(e):
                e.sim.cwnd = float(np.clip(fn(e), MIN_CWND, MAX_CWND))
                return 2  # x1.0: the window is set directly
            return act
        return make

    rng = np.random.default_rng(5)
    pols = {
        "idle (2 pkts)": absolute(lambda e: MIN_CWND),
        "flood (1000 pkts)": absolute(lambda e: MAX_CWND),
        "fixed 40 pkts": absolute(lambda e: 40 * MSS),
        "oracle (cwnd = BDP)": absolute(lambda e: e.sim.bw * e.params.rtt),
        "random actions": lambda env: (lambda e: int(rng.integers(5))),
    }
    res = {}
    print(f"      {'policy':22s} {'reward':>8s} {'thr':>6s} {'delay':>6s} {'loss':>6s}")
    for name, mk in pols.items():
        res[name] = policy_returns(mk, seeds)
        r = res[name]
        print(f"      {name:22s} {r[0]:8.3f} {r[1]:6.2f} {r[2]:6.2f} {r[3]:6.3f}")
    best = max(res, key=lambda k: res[k][0])
    check("reward: oracle (cwnd = BDP) scores highest of all tried policies", best.startswith("oracle"),
          f"best was '{best}'")
    check("reward: sending almost nothing scores clearly below the oracle",
          res["idle (2 pkts)"][0] < res["oracle (cwnd = BDP)"][0] - 0.3)
    check("reward: flooding the link scores clearly below the oracle",
          res["flood (1000 pkts)"][0] < res["oracle (cwnd = BDP)"][0] - 0.1,
          f"flood {res['flood (1000 pkts)'][0]:.2f} vs oracle {res['oracle (cwnd = BDP)'][0]:.2f}")
    check("reward: random actions score below the oracle", res["random actions"][0] < res["oracle (cwnd = BDP)"][0])
    mx = max(r[4] for r in res.values())
    check("reward: per-episode mean throughput term never meaningfully above 1", mx <= 1.05,
          f"max {mx:.3f}", warn_only=True)


def t_speed():
    env = CCEnv()
    rng = np.random.default_rng(3)
    t0 = time.time()
    steps = 0
    for ep in range(15):
        env.reset(seed=ep)
        for _ in range(env.max_steps):
            _, _, _, tr, _ = env.step(int(rng.integers(5)))
            steps += 1
    sps = steps / (time.time() - t0)
    print(f"      env speed: {sps:.0f} steps/s  ->  200k steps ~ {200000 / sps / 60:.1f} min, "
          f"1M steps ~ {1e6 / sps / 60:.0f} min (single CPU core, before PPO overhead)")
    check("speed: env fast enough to train on CPU (1M steps under 90 min)", 1e6 / sps / 60 < 90, warn_only=True)


if __name__ == "__main__":
    for fn in (t_features, t_sim_invariants, t_determinism, t_leakage, t_reward, t_speed):
        print(f"\n== {fn.__name__[2:]}")
        fn()
    print(f"\nSUMMARY: {counts['PASS']} pass, {counts['WARN']} warn, {counts['FAIL']} fail")
    sys.exit(1 if counts["FAIL"] else 0)
