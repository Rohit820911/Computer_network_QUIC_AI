"""Run controllers x scenarios, repeated, in shuffled order. Needs root (namespaces).

    sudo .venv/bin/python experiments/run_experiments.py \
        --controllers reno,cubic,reno_copy --scenarios S1,S2,S3,S4 --runs 5

Smoke test without root or namespaces (plain loopback, no shaping, no S3 drop):

    .venv/bin/python experiments/run_experiments.py --loopback --runs 1 --size-mb 2

Per run it saves:  logs/<scenario>_<cc>_run<k>.csv (+ .json)   server-side samples
and appends a row to results/results.csv (goodput, loss, drop/start timestamps).
Rebuilds the namespace link before every run so qdisc state is always clean.
Controllers are shuffled within each round (seeded) to spread out drift.
"""
import argparse
import csv
import json
import os
import random
import subprocess
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NETNS = os.path.join(ROOT, "testbed", "netns.sh")
TRANSFER = os.path.join(ROOT, "experiments", "transfer.py")
LOGS = os.path.join(ROOT, "logs")
RESULTS = os.path.join(ROOT, "results", "results.csv")
PORT = 4433

# rate in Mbit/s, delay = ONE-WAY ms (RTT = 2x), loss in percent.
# S3: starts at `rate`, drops to `drop_to` after `drop_at` seconds.
# The queue is set once, to 1 BDP of the starting rate, and left unchanged.
# S5 (held-out) is defined after training, using values outside the training ranges.
SCENARIOS = {
    "S1": dict(rate=10, delay=20, loss=0),
    "S2": dict(rate=10, delay=20, loss=1),
    # S3 needs a longer file: the transfer must still be running well after the drop.
    # 25 MB: ~3 s at 20 Mbit/s (~7 MB), then the remaining ~18 MB at 5 Mbit/s (~30 s).
    "S3": dict(rate=20, delay=20, loss=0, drop_to=5, drop_at=3.0, size_mb=25),
    "S4": dict(rate=10, delay=100, loss=0),
    # S5: held-out, one factor outside the training ranges (see sim/link_sim.py TRAIN)
    "S5a": dict(rate=10, delay=20, loss=4),                # 4% loss (training max 2%)
    "S5b": dict(rate=10, delay=125, loss=0),               # 250 ms RTT (training max 120)
    "S5c": dict(rate=10, delay=20, loss=0, queue_bdp=0.2), # 0.2 BDP queue (training min 0.5)
}
FIELDS = ["scenario", "cc", "run", "seed", "size_mb", "bytes", "seconds",
          "goodput_mbps", "sent_bytes", "lost_bytes", "lost_packets",
          "loss_ratio", "log_file", "t0_epoch", "drop_epoch", "timestamp"]


def sh(cmd, **kw):
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kw)


def link_up(sc):
    sh(["bash", NETNS, "up", str(sc["rate"]), str(sc["delay"]), str(sc["loss"]),
        str(int(round(sc.get("queue_bdp", 1.0) * 100)))])


def link_down():
    subprocess.run(["bash", NETNS, "down"], capture_output=True)


def drop_rate(sc, holder):
    # keep the original queue limit (1 BDP of the starting rate)
    bdp = sc["rate"] * 1000000 // 8 * 2 * sc["delay"] // 1000
    bdp = max(bdp, 3000)
    sh(["ip", "netns", "exec", "ns_s", "tc", "qdisc", "change", "dev", "veth_s",
        "parent", "1:1", "handle", "10:", "tbf", "rate", f"{sc['drop_to']}mbit",
        "burst", "32kbit", "limit", str(bdp)])
    holder["drop_epoch"] = time.time()


def one_run(args, name, sc, cc, k, seed):
    tag = f"{name}_{cc}_run{k}"
    size_mb = sc.get("size_mb", args.size_mb)
    log = os.path.join(LOGS, tag + ".csv")
    ns = lambda n: [] if args.loopback else ["ip", "netns", "exec", n]
    host = "127.0.0.1" if args.loopback else "10.0.0.2"
    if not args.loopback:
        link_up(sc)
    server = subprocess.Popen(
        ns("ns_s") + [sys.executable, TRANSFER, "server", "--cc", cc, "--log", log,
                      "--once", "--port", str(PORT), "--timeout", str(args.timeout)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        if "listening" not in server.stdout.readline():
            raise RuntimeError("server failed: " + server.stderr.read())
        holder, timer = {}, None
        if "drop_to" in sc and not args.loopback:
            timer = threading.Timer(sc["drop_at"], drop_rate, args=(sc, holder))
            timer.start()
        cl = subprocess.run(
            ns("ns_c") + [sys.executable, TRANSFER, "client", "--host", host,
                          "--port", str(PORT), "--size-mb", str(size_mb),
                          "--timeout", str(args.timeout)],
            capture_output=True, text=True)
        if timer:
            timer.cancel()
        if cl.returncode != 0:
            raise RuntimeError("client failed: " + cl.stderr[-500:])
        res = json.loads(cl.stdout.strip().splitlines()[-1])
        server.wait(timeout=30)
    finally:
        if server.poll() is None:
            server.kill()
        if not args.loopback:
            link_down()
    with open(log + ".json") as f:
        srv = json.load(f)
    if "drop_to" in sc and not args.loopback and "drop_epoch" not in holder:
        print(f"WARNING {tag}: bandwidth drop never fired (transfer ended first); "
              "this run is not a valid S3 run", flush=True)
    return dict(
        scenario=name, cc=cc, run=k, seed=seed, size_mb=size_mb,
        bytes=res["bytes"], seconds=res["seconds"], goodput_mbps=res["goodput_mbps"],
        sent_bytes=srv["sent_bytes"], lost_bytes=srv["lost_bytes"],
        lost_packets=srv["lost_packets"],
        loss_ratio=round(srv["lost_bytes"] / max(srv["sent_bytes"], 1), 5),
        log_file=os.path.relpath(log, ROOT), t0_epoch=srv["t0_epoch"],
        drop_epoch=holder.get("drop_epoch", ""), timestamp=time.strftime("%F %T"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--controllers", default="reno,cubic,reno_copy")
    p.add_argument("--scenarios", default="S1,S2,S3,S4")
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--size-mb", type=float, default=10)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--timeout", type=float, default=300)
    p.add_argument("--loopback", action="store_true")
    args = p.parse_args()

    if not args.loopback and os.geteuid() != 0:
        sys.exit("needs root for network namespaces: run with sudo (or use --loopback)")
    ccs = args.controllers.split(",")
    names = args.scenarios.split(",")
    os.makedirs(LOGS, exist_ok=True)
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    new = not os.path.exists(RESULTS)
    rng = random.Random(args.seed)
    print(f"seed={args.seed} controllers={ccs} scenarios={names} runs={args.runs} "
          f"size={args.size_mb}MB", flush=True)
    with open(RESULTS, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for name in names:
            sc = SCENARIOS[name]
            for k in range(1, args.runs + 1):
                order = ccs[:]
                rng.shuffle(order)
                for cc in order:
                    row = one_run(args, name, sc, cc, k, args.seed)
                    w.writerow(row)
                    f.flush()
                    print(f"{name} run{k} {cc:10s} {row['goodput_mbps']:8.2f} Mbit/s "
                          f"loss={row['loss_ratio']:.4f}", flush=True)


if __name__ == "__main__":
    main()
