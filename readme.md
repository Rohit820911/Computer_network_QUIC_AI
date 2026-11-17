# AI-Driven Congestion Control for QUIC

Small-scale evaluation of a PPO-trained congestion controller against Reno and
Cubic inside aioquic. Follows the project guide (phases 1-8, go/no-go gates).

## Findings from the Section 11 "verify" list (aioquic 1.8.0)

- Controllers shipped: **reno**, **cubic**. No BBR.
- Interface: `aioquic/quic/congestion/base.py`, not `recovery.py`. Subclass
  `QuicCongestionControl` and call `register_congestion_control(name, factory)`.
  Select with `QuicConfiguration(congestion_control_algorithm="name")`.
- Required methods: `on_packet_acked`, `on_packet_sent`, `on_packets_expired`,
  `on_packets_lost`, `on_persistent_congestion`, `on_rtt_measurement`.
  The controller owns `congestion_window` and `bytes_in_flight`.
- Built-ins register on import of `aioquic.quic.recovery`; `K_MINIMUM_WINDOW = 2`
  packets, initial window 10 packets (these are the safety-clamp reference values).
- Version is pinned in `requirements.txt` (aioquic 1.8.0).

## Layout

    testbed/netns.sh        two-namespace link (up / down / show), queue = 1 BDP
    controllers/reno_copy.py  Phase 3 plug-in test, registered as "reno_copy", logs CSV
    experiments/transfer.py   server (sender) / client (receiver); samples cwnd, RTT, loss to CSV
    experiments/run_experiments.py   shuffled, repeated runs over scenarios S1-S4
    sim/                    Gymnasium bottleneck env (Phase 4)
    logs/  results/         raw per-run CSVs and plots

## Setup

    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt

## Testbed

    sudo ./testbed/netns.sh up 10 20 0     # 10 Mbit/s, 40 ms RTT, 0% loss
    sudo ./testbed/netns.sh show           # verify with tc -s qdisc
    sudo ./testbed/netns.sh down

Run client and server with `sudo ip netns exec ns_c ...` / `ns_s ...`.
Verify the shaping with a quick throughput test before trusting any result.

## Running experiments

Smoke test (no root, no shaping, numbers are meaningless, just checks the scripts):

    python experiments/run_experiments.py --loopback --scenarios S1 --runs 1 --size-mb 2

Real runs on the shaped link (needs root; use the venv python):

    sudo .venv/bin/python experiments/run_experiments.py \
        --controllers reno,cubic,reno_copy --scenarios S1 --runs 5 --size-mb 10

The sender is the QUIC *server*, so `--cc` is applied on the server side. Raw
per-run logs go to `logs/`, one summary row per run to `results/results.csv`.
Notes: the queue is fixed at 1 BDP of the starting rate (in S3 it stays at the
20 Mbit/s value after the drop); S5 (held-out) is defined after training.

## Status

- [x] Environment and pinned dependencies
- [x] aioquic congestion interface inspected
- [x] Reno-copy controller written and registered (needs behavior check on the link)
- [x] Transfer script and run harness written (tested on loopback only)
- [ ] Phase 1: first aioquic transfer through the namespaces, repeatable results
- [ ] Phase 2: Reno/Cubic baselines, 5+ runs per scenario
- [ ] Phase 3 gate: reno_copy matches built-in reno on the real link
- [ ] Phases 4-8
