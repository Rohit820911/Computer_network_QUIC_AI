"""Phase 3 proof that we control cwnd inside aioquic (no root, no namespaces needed).

    python tests/check_control.py

Runs short loopback transfers with fixed windows of 4, 8 and 16 packets and checks:
  1. the logged cwnd stays exactly at the configured value,
  2. bytes in flight never exceed the window (plus 2 packets of slack for probe packets),
  3. a larger window gives strictly more throughput.
Loopback cannot show that throughput is PROPORTIONAL to cwnd: Python's event loop, not the
window, limits the sender there (only about 60-80% of the window is ever in flight). That
proof belongs on the shaped lab link, where the prediction is simply cwnd / RTT:
    sudo .venv/bin/python experiments/run_experiments.py --scenarios S1 \
        --controllers fixed4,fixed8,fixed16,fixed40 --runs 2 --size-mb 3
    expected goodput at 40 ms RTT: fixed4 ~0.96, fixed8 ~1.9, fixed16 ~3.8 Mbit/s,
    fixed40 ~9.2 (capped by the 10 Mbit/s link).
"""
import csv
import json
import os
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRANSFER = os.path.join(ROOT, "experiments", "transfer.py")
PORT = 4466
MSS = 1200


def run(cc, size_mb=2):
    log = os.path.join(tempfile.mkdtemp(), f"{cc}.csv")
    srv = subprocess.Popen([sys.executable, TRANSFER, "server", "--cc", cc, "--log", log, "--once",
                            "--port", str(PORT)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        if "listening" not in srv.stdout.readline():
            raise RuntimeError(srv.stderr.read())
        cl = subprocess.run([sys.executable, TRANSFER, "client", "--port", str(PORT), "--size-mb", str(size_mb)],
                            capture_output=True, text=True, timeout=300)
        if cl.returncode:
            raise RuntimeError(cl.stderr[-500:])
        res = json.loads(cl.stdout.strip().splitlines()[-1])
        srv.wait(timeout=30)
    finally:
        if srv.poll() is None:
            srv.kill()
    with open(log) as f:
        rows = list(csv.DictReader(f))
    cwnds = {int(float(r["cwnd"])) for r in rows}
    max_bif = max(float(r["bytes_in_flight"]) for r in rows)
    return res["goodput_mbps"], cwnds, max_bif


def main():
    out = {}
    for n in (4, 8, 16):
        out[n] = run(f"fixed{n}")
        time.sleep(0.3)
    ok = True
    print(f"{'window':>8s} {'goodput Mbit/s':>15s} {'max in flight':>14s}  logged cwnd values")
    for n, (g, cw, mb) in out.items():
        held = cw == {n * MSS}
        respected = mb <= n * MSS + 2 * MSS
        print(f"{n:>6d}pk {g:15.2f} {mb:14.0f}  {sorted(cw)}  held={held} respected={respected}")
        ok &= held and respected
    ok &= out[4][0] < out[8][0] < out[16][0]
    print("PASS: window is held exactly, never exceeded, and more window gives more throughput"
          if ok else "FAIL: window not held / exceeded / throughput does not rise with window")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
