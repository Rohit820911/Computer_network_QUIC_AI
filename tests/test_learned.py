"""Offline tests for the real learned controller (no network, no root).

    python tests/test_learned.py

1. constants copied into controllers/learned.py equal the simulator's (fails on drift)
2. the numpy policy picks the same action as Stable Baselines3 on states from real sim rollouts
3. interval accounting: a synthetic packet stream is driven through the controller's callbacks;
   no byte is lost or double counted, intervals close on time, every recorded state equals what
   common/features.py computes from the recorded raw stats, and every action matches the policy
4. safety clamps: always-shrink stops at 2 packets, always-grow stops at 1000 packets
"""
import os
import sys
from types import SimpleNamespace

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from common.features import FeatureTracker, interval_length  # noqa: E402
from controllers import learned as L  # noqa: E402
from sim import env as simenv  # noqa: E402
from sim import link_sim as ls  # noqa: E402

counts = {"PASS": 0, "FAIL": 0, "SKIP": 0}


def check(name, ok, detail=""):
    counts["PASS" if ok else "FAIL"] += 1
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def t_constants():
    check("actions equal the simulator's", tuple(L.ACTIONS) == tuple(simenv.ACTIONS))
    check("MSS and window clamps equal the simulator's",
          L.MSS == ls.MSS and L.MIN_CWND_PKTS * L.MSS == ls.MIN_CWND and L.MAX_CWND_PKTS * L.MSS == ls.MAX_CWND)
    c = L.LearnedCongestionControl(max_datagram_size=1200, policy=SimpleNamespace(act=lambda s: 2))
    check("initial window equals the simulator's", c.congestion_window == ls.INIT_CWND)


def t_policy_parity():
    path = os.path.join(ROOT, "models", "ppo_v1")
    if not (os.path.exists(path + ".zip") and os.path.exists(path + ".npz")):
        counts["SKIP"] += 1
        print("[SKIP] numpy vs SB3 on sim states (models/ppo_v1.zip and .npz needed)")
        return
    from stable_baselines3 import PPO
    model = PPO.load(path, device="cpu")
    pol = L.NumpyPolicy(path + ".npz")
    rng = np.random.default_rng(0)
    same = total = 0
    for i in range(6):
        p = ls.sample_params(rng)
        sim, tr = ls.LinkSim(p, seed=i), FeatureTracker()
        for _ in range(80):
            st = sim.run_interval(max(0.02, min(sim.srtt, 0.4)))
            tr.update(acked_bytes=st.acked, sent_bytes=st.sent, lost_bytes=st.lost,
                      rtt_mean=st.rtt_mean, rtt_min=st.rtt_min, cwnd=st.cwnd)
            s = tr.state()
            a_ref = int(model.predict(s, deterministic=True)[0])
            a = pol.act(s)
            same += a == a_ref
            total += 1
            sim.cwnd = float(np.clip(sim.cwnd * L.ACTIONS[a_ref], ls.MIN_CWND, ls.MAX_CWND))
    check("numpy policy = SB3 on states from simulator rollouts", same == total, f"{same}/{total}")


def drive(policy, seconds=3.0, rtt=0.05, loss_every=200):
    """Feed a synthetic stream through the controller callbacks: 1 packet/ms, acked after `rtt`,
    every `loss_every`-th packet declared lost instead. Returns controller and true totals."""
    c = L.LearnedCongestionControl(max_datagram_size=1200, policy=policy)
    ev = []
    tot = dict(sent=0, acked=0, lost=0)
    for k in range(int(seconds * 1000)):
        t = k / 1000
        ev.append((t, 0, k))
        ev.append((t + (rtt + 0.01 if k % loss_every == loss_every - 1 else rtt),
                   2 if k % loss_every == loss_every - 1 else 1, k))
    ev.sort()
    for t, kind, k in ev:
        if t > seconds:
            break
        pkt = SimpleNamespace(sent_bytes=1200)
        if kind == 0:
            c.on_packet_sent(packet=pkt)
            tot["sent"] += 1200
        elif kind == 1:
            c.on_packet_acked(now=t, packet=pkt)
            c.on_rtt_measurement(now=t, rtt=rtt)
            tot["acked"] += 1200
        else:
            c.on_packets_lost(now=t, packets=[pkt])
            tot["lost"] += 1200
    return c, tot


def t_accounting():
    path = os.path.join(ROOT, "models", "ppo_v1.npz")
    pol = L.NumpyPolicy(path) if os.path.exists(path) else SimpleNamespace(act=lambda s: [1, 2, 3, 4, 0][int(s[-1] * 7) % 5])
    c, tot = drive(pol)
    rows = [dict(zip(c.decision_header, r)) for r in c.decision_rows]
    check("controller made decisions", len(rows) > 20, f"{len(rows)} intervals in 3 s")
    for key, col, rem in (("sent", "sent", c.i_sent), ("acked", "acked", c.i_acked), ("lost", "lost", c.i_lost)):
        s = sum(r[col] for r in rows) + rem
        check(f"no {key} bytes lost or double counted", s == tot[key], f"{s} vs {tot[key]}")
    t_prev, srtt_prev, ok_len = None, None, True
    for r in rows:
        if t_prev is not None:
            d = r["t"] - t_prev
            ok_len &= d >= interval_length(srtt_prev) - 1e-9 and d <= interval_length(srtt_prev) + 0.003
        t_prev, srtt_prev = r["t"], r["srtt"]
    check("each interval is as long as interval_length(srtt) (within callback granularity)", ok_len)
    tr, ok_state, ok_act = FeatureTracker(), True, True
    for r in rows:
        tr.update(acked_bytes=r["acked"], sent_bytes=r["sent"], lost_bytes=r["lost"],
                  rtt_mean=r["rtt_mean"] if r["rtt_mean"] != "" else None,
                  rtt_min=r["rtt_min"] if r["rtt_min"] != "" else None, cwnd=r["cwnd_before"])
        s = tr.state()
        ok_state &= bool(np.allclose(s, [r[f"s{i}"] for i in range(len(s))], atol=1e-6))
        ok_act &= pol.act(s) == r["action"]
        want = int(np.clip(r["cwnd_before"] * L.ACTIONS[r["action"]], 2 * 1200, 1000 * 1200))
        ok_act &= r["cwnd_after"] == want
    check("recorded states equal common/features.py recomputed from raw stats", ok_state)
    check("recorded actions equal the policy's and windows follow ACTIONS and clamps", ok_act)
    check("a constant 50 ms RTT shows no queueing in the rtt feature", abs(rows[5]["s13"]) < 1e-9, f"s13={rows[5]['s13']}")


def t_clamps():
    c, _ = drive(SimpleNamespace(act=lambda s: 0))
    check("always-shrink stops at the 2-packet minimum", c.congestion_window == 2 * 1200, str(c.congestion_window))
    c, _ = drive(SimpleNamespace(act=lambda s: 4), seconds=8.0)
    check("always-grow stops at the 1000-packet maximum", c.congestion_window == 1000 * 1200, str(c.congestion_window))


if __name__ == "__main__":
    for fn in (t_constants, t_policy_parity, t_accounting, t_clamps):
        print(f"\n== {fn.__name__[2:]}")
        fn()
    print(f"\nSUMMARY: {counts['PASS']} pass, {counts['FAIL']} fail, {counts['SKIP']} skipped")
    sys.exit(1 if counts["FAIL"] else 0)
