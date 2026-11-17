"""Adds the S5 held-out scenarios to the project. Run ONCE from the project root:

    python add_s5.py

Edits (each file is first copied to <file>.bak; every edit is skipped if already applied,
and the script stops with a message if an expected line is not found in your copy):
  testbed/netns.sh              5th argument: queue size as % of BDP (default 100 = old behaviour)
  experiments/run_experiments.py  scenarios S5a, S5b, S5c + passes the queue size to netns.sh
  sim/link_sim.py               the same three scenarios for the simulator (scenario_params, FILE_MB)
  experiments/analyze.py        colours for fixed4/8/16, ppo_v1, learned_ppo_v2 and S5 titles

S5 = conditions OUTSIDE the training ranges (TRAIN in sim/link_sim.py), one factor at a time:
  S5a  4% random loss        (training max 2%)
  S5b  250 ms RTT            (training max 120 ms)
  S5c  queue 0.2 BDP         (training min 0.5 BDP)
All three use 10 Mbit/s, the S1 link, so only one thing changes against S1/S2/S4.
Bandwidth above 20 Mbit/s is not included: Python's QUIC sender becomes the bottleneck there.
"""
import os
import shutil
import sys

ROOT = os.getcwd()


def edit(rel, pairs, marker):
    path = os.path.join(ROOT, rel)
    if not os.path.exists(path):
        sys.exit(f"missing {rel}: run this from the project root")
    s = open(path).read()
    if marker in s:
        print(f"skip   {rel} (already patched)")
        return
    for old, new in pairs:
        if s.count(old) != 1:
            sys.exit(f"STOP: in {rel} I expected exactly one copy of this line and found "
                     f"{s.count(old)}:\n    {old.strip()}\nYour file differs from mine; "
                     "tell me and I will adapt the patch. Nothing in this file was changed.")
        s = s.replace(old, new)
    shutil.copy(path, path + ".bak")
    open(path, "w").write(s)
    print(f"patched {rel}  (backup: {rel}.bak)")


edit("testbed/netns.sh", [
    ("LOSS=${4:-0}       # percent random loss\n",
     "LOSS=${4:-0}       # percent random loss\nQPCT=${5:-100}     # queue size as % of one BDP (100 = 1 BDP)\n"),
    ("BDP=$(( RATE * 1000000 / 8 * 2 * DELAY / 1000 ))",
     "BDP=$(( RATE * 1000000 / 8 * 2 * DELAY / 1000 * QPCT / 100 ))"),
    ("queue=${BDP}B (1 BDP)", "queue=${BDP}B (${QPCT}% of BDP)"),
], "QPCT")

edit("experiments/run_experiments.py", [
    ('    "S4": dict(rate=10, delay=100, loss=0),\n',
     '    "S4": dict(rate=10, delay=100, loss=0),\n'
     '    # S5: held-out, one factor outside the training ranges (see sim/link_sim.py TRAIN)\n'
     '    "S5a": dict(rate=10, delay=20, loss=4),                # 4% loss (training max 2%)\n'
     '    "S5b": dict(rate=10, delay=125, loss=0),               # 250 ms RTT (training max 120)\n'
     '    "S5c": dict(rate=10, delay=20, loss=0, queue_bdp=0.2), # 0.2 BDP queue (training min 0.5)\n'),
    ('sh(["bash", NETNS, "up", str(sc["rate"]), str(sc["delay"]), str(sc["loss"])])',
     'sh(["bash", NETNS, "up", str(sc["rate"]), str(sc["delay"]), str(sc["loss"]),\n'
     '        str(int(round(sc.get("queue_bdp", 1.0) * 100)))])'),
], '"S5a"')

edit("sim/link_sim.py", [
    ("    raise KeyError(name)\n",
     '    if name == "S5a":\n        return LinkParams(10, 0.040, _bdp(10, 0.040), loss=0.04)\n'
     '    if name == "S5b":\n        return LinkParams(10, 0.250, _bdp(10, 0.250))\n'
     '    if name == "S5c":\n        return LinkParams(10, 0.040, 0.2 * _bdp(10, 0.040))\n'
     "    raise KeyError(name)\n"),
    ('FILE_MB = {"S1": 10, "S2": 10, "S3": 25, "S4": 10}',
     'FILE_MB = {"S1": 10, "S2": 10, "S3": 25, "S4": 10, "S5a": 10, "S5b": 10, "S5c": 10}'),
], '"S5a"')

# analyze.py: add colours without touching your existing dictionaries
path = os.path.join(ROOT, "experiments", "analyze.py")
s = open(path).read()
if "S5_PATCH" in s:
    print("skip   experiments/analyze.py (already patched)")
else:
    lines = s.split("\n")

    def stmt_end(prefix):
        """Index of the last line of the top-level statement starting with `prefix`."""
        starts = [i for i, l in enumerate(lines) if l.startswith(prefix)]
        if len(starts) != 1:
            sys.exit(f"STOP: could not find a single '{prefix}' line in experiments/analyze.py")
        i, depth = starts[0], 0
        while True:
            depth += sum(lines[i].count(c) for c in "([{") - sum(lines[i].count(c) for c in ")]}")
            if depth <= 0:
                return i
            i += 1

    j = max(stmt_end("ORDER = ["), stmt_end("TITLES = {"), stmt_end("COLORS = {"),
            stmt_end("STYLES = {"))
    block = '''# S5_PATCH: colours for controllers added after the first version of this file
for _cc, _col, _ls in (("learned_ppo_v2", "#1baf7a", "-."), ("ppo_v1", "#1baf7a", ":"),
                       ("fixed4", "#c9c8c4", ":"), ("fixed8", "#b3b2ae", ":"),
                       ("fixed16", "#9d9c97", ":")):
    COLORS.setdefault(_cc, _col)
    STYLES.setdefault(_cc, _ls)
    if _cc not in ORDER:
        ORDER.append(_cc)
TITLES.setdefault("S5a", "S5a held-out\\n4% loss")
TITLES.setdefault("S5b", "S5b held-out\\n250 ms RTT")
TITLES.setdefault("S5c", "S5c held-out\\n0.2 BDP queue")'''
    lines[j + 1:j + 1] = block.split("\n")
    shutil.copy(path, path + ".bak")
    open(path, "w").write("\n".join(lines))
    print("patched experiments/analyze.py  (backup: experiments/analyze.py.bak)")
print("\ndone. Next: python experiments/s5_predict.py   then the lab runs shown in the message.")
