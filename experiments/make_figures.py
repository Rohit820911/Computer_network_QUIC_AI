"""Report figures from results/summary.csv (written by experiments/analyze.py).

    python experiments/make_figures.py            # writes results/figures/fig1..fig5 (.png)

Reads only the summary table, so rerun it after any new lab runs. Controller names it
expects: reno, cubic, fixed40, learned_ppo_v2 (the main model), learned_ppo_v2_s11 / _s12
(extra training seeds) and learned (= policy v1, trained without RTT noise).
Error bars are the standard deviation over runs (5 runs each).
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if "experiments" in os.path.abspath(__file__) else os.getcwd()
OUT = os.path.join(ROOT, "results", "figures")
os.makedirs(OUT, exist_ok=True)
S = pd.read_csv(os.path.join(ROOT, "results", "summary.csv"))

SC = ["S1", "S2", "S3", "S4", "S5a", "S5b", "S5c"]
LABEL = {"S1": "S1\n10 Mbit/s\n40 ms", "S2": "S2\n1% loss", "S3": "S3\nrate drop\n20→5 Mbit/s",
         "S4": "S4\n200 ms RTT", "S5a": "S5a\n4% loss\n(held-out)", "S5b": "S5b\n250 ms RTT\n(held-out)",
         "S5c": "S5c\n0.2 BDP queue\n(held-out)"}
BASE_RTT = {"S1": 40, "S2": 40, "S3": 40, "S4": 200, "S5a": 40, "S5b": 250, "S5c": 40}
CC = ["reno", "cubic", "fixed40", "learned_ppo_v2"]
NAME = {"reno": "Reno", "cubic": "Cubic", "fixed40": "Fixed window (40 pkts)",
        "learned_ppo_v2": "Learned (PPO v2)", "learned": "Learned v1 (no RTT noise in training)",
        "learned_ppo_v2_s11": "v2, seed 11", "learned_ppo_v2_s12": "v2, seed 12"}
COL = {"reno": "#2a78d6", "cubic": "#eb6834", "fixed40": "#eda100", "learned_ppo_v2": "#1baf7a",
       "learned": "#c23b3b", "learned_ppo_v2_s11": "#6fd1ad", "learned_ppo_v2_s12": "#0f7d57"}
INK, GRID = "#2b2a28", "#e4e3df"


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, color=GRID, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK, labelsize=9)


def get(sc, cc, col):
    r = S[(S.scenario == sc) & (S.cc == cc)]
    return float(r[col].iloc[0]) if len(r) else np.nan


def grouped(ax, scs, ccs, col, errcol=None, labels=True, fmt="{:.1f}"):
    w = 0.8 / len(ccs)
    x0 = np.arange(len(scs))
    for i, cc in enumerate(ccs):
        v = [get(s, cc, col) for s in scs]
        e = [get(s, cc, errcol) for s in scs] if errcol else None
        xs = x0 - 0.4 + w * (i + 0.5)
        ax.bar(xs, v, w * 0.92, color=COL[cc], label=NAME[cc], zorder=3, yerr=e,
               error_kw=dict(ecolor=INK, elinewidth=1, capsize=2))
        if labels:
            for j, (x, y) in enumerate(zip(xs, v)):
                top = y + (0 if e is None or np.isnan(e[j]) else e[j])
                ax.text(x, top, fmt.format(y), ha="center", va="bottom", fontsize=7, color=INK)
    ax.set_xticks(x0)
    ax.set_xticklabels([LABEL[s] for s in scs], fontsize=8.5, color=INK)
    style(ax)
    return w


# ---- Fig 1: goodput -----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(13, 5.2), facecolor="white")
grouped(ax, SC, CC, "goodput_mean", "goodput_std")
ax.set_ylabel("Goodput (Mbit/s), higher is better")
ax.set_title("Goodput: the learned controller beats Reno and Cubic where random loss or a tiny queue hurts them,\n"
             "and trails Reno by 3-18% on S1, S3, S4 and S5b", loc="left", fontsize=11, color=INK)
ax.set_ylim(0, 11.5)
ax.legend(frameon=False, ncol=4, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, 0.99))
fig.text(0.01, 0.005, "5 runs per bar, error bars = standard deviation. S3: 20 Mbit/s dropping to 5 Mbit/s "
         "after 3 s, so goodput there is not comparable with the other scenarios.", fontsize=8, color="#6b6a66")
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(os.path.join(OUT, "fig1_goodput.png"), dpi=170)
plt.close(fig)

# ---- Fig 2: p95 RTT (two scales) ----------------------------------------------------------
LOW, HIGH = ["S1", "S2", "S3", "S5a", "S5c"], ["S4", "S5b"]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw=dict(width_ratios=[5, 2.3]),
                             facecolor="white")
for ax, scs in ((a1, LOW), (a2, HIGH)):
    w = grouped(ax, scs, CC, "rtt_p95_mean", fmt="{:.0f}")
    for i, s in enumerate(scs):
        ax.hlines(BASE_RTT[s], i - 0.42, i + 0.42, colors=INK, linestyles="--", linewidth=1.3, zorder=4)
a1.set_ylabel("95th-percentile RTT (ms), lower is better")
a1.set_title("Delay: where Reno and Cubic build a queue (S1, S3, S4, S5b) the learned controller\nadds far less delay; elsewhere all four stay near the base RTT",
             loc="left", fontsize=11, color=INK)
a2.set_title("Long-RTT scenarios (note the different scale)", loc="left", fontsize=10, color=INK)
a1.set_ylim(0, 240)
a2.set_ylim(0, 560)
a1.legend(frameon=False, ncol=2, fontsize=9, loc="upper left")
a1.text(0.99, 0.97, "dashed line = base RTT (no queue)", transform=a1.transAxes, ha="right", va="top",
        fontsize=8.5, color=INK)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig2_delay.png"), dpi=170)
plt.close(fig)

# ---- Fig 3: v1 failure and fix ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5), facecolor="white")
ccs3 = ["learned", "learned_ppo_v2", "reno", "cubic", "fixed40"]
grouped(ax, ["S1", "S2"], ccs3, "goodput_mean", "goodput_std")
ax.axhline(9.2, color=COL["learned"], linestyle=":", linewidth=1.4, zorder=2)
ax.text(0.5, 10.1, "dotted line: v1's simulator prediction (9.2)", ha="center", fontsize=8.5, color=COL["learned"])
ax.set_ylabel("Goodput (Mbit/s)")
ax.set_ylim(0, 11.5)
ax.set_title("Sim-to-real failure and fix: v1 read lab RTT jitter as congestion;\n"
             "v2, trained with RTT noise, recovers", loc="left", fontsize=11, color=INK)
ax.legend(frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig3_v1_failure_and_fix.png"), dpi=170)
plt.close(fig)

# ---- Fig 4: seed reproducibility ----------------------------------------------------------
fig, (g, d) = plt.subplots(2, 1, figsize=(11, 7), sharex=True, facecolor="white")
x = np.arange(len(SC))
for ax, col, tolf, unit in ((g, "goodput_mean", lambda s: 10 if s == "S5b" else 5, "% vs v2"),
                            (d, "rtt_p95_mean", lambda s: 25 if s in ("S4", "S5b") else 5, "ms vs v2")):
    for i, s in enumerate(SC):
        t = tolf(s)
        ax.add_patch(plt.Rectangle((i - 0.4, -t), 0.8, 2 * t, color="#eceae5", zorder=1))
    for k, off in (("learned_ppo_v2_s11", -0.08), ("learned_ppo_v2_s12", 0.08)):
        v = []
        for s in SC:
            a, b = get(s, k, col), get(s, "learned_ppo_v2", col)
            v.append((a - b) / b * 100 if col == "goodput_mean" else a - b)
        ax.scatter(x + off, v, s=55, color=COL[k], label=NAME[k], zorder=3, edgecolor="white")
    ax.axhline(0, color=INK, linewidth=0.8, zorder=2)
    ax.set_ylabel(unit)
    style(ax)
g.set_title("Seed check: two independently trained models vs the main model (grey = pre-registered tolerance)",
            loc="left", fontsize=11, color=INK)
g.set_ylabel("Goodput difference (%)")
d.set_ylabel("p95 RTT difference (ms)")
d.set_xticks(x)
d.set_xticklabels([LABEL[s] for s in SC], fontsize=8)
g.legend(frameon=False, fontsize=9, loc="upper left")
d.annotate("Seed 12 falls outside the band at S4 and S5b:\nlower delay, slightly lower goodput",
           xy=(3.08, get("S4", "learned_ppo_v2_s12", "rtt_p95_mean") - get("S4", "learned_ppo_v2", "rtt_p95_mean")),
           xytext=(0.2, -33), fontsize=8.5, color=INK, arrowprops=dict(arrowstyle="->", color=INK))
d.set_ylim(-42, 32)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig4_seed_check.png"), dpi=170)
plt.close(fig)

# ---- Fig 5: simulator vs lab for v2 -------------------------------------------------------
# Simulator values from my run of the v2 policy with 1.6 ms RTT noise added to its observation
# (noise measured on S1 fixed40 lab logs); S5 rows were fixed BEFORE the lab runs.
SIM = {"S1": 9.14, "S2": 9.09, "S3": 5.67, "S4": 7.50, "S5a": 8.74, "S5b": 6.99, "S5c": 9.10}
fig, ax = plt.subplots(figsize=(11, 5), facecolor="white")
x = np.arange(len(SC))
real = [get(s, "learned_ppo_v2", "goodput_mean") for s in SC]
ax.bar(x - 0.2, [SIM[s] for s in SC], 0.38, color="#b9b8b3", label="Simulator (with lab RTT noise)", zorder=3)
ax.bar(x + 0.2, real, 0.38, color=COL["learned_ppo_v2"], label="Real lab link (5 runs)", zorder=3,
       yerr=[get(s, "learned_ppo_v2", "goodput_std") for s in SC], error_kw=dict(ecolor=INK, capsize=2))
for i, s in enumerate(SC):
    gap = (real[i] - SIM[s]) / SIM[s] * 100
    ax.text(i, max(SIM[s], real[i]) + 0.25, f"{gap:+.0f}%", ha="center", fontsize=9, color=INK)
ax.set_xticks(x)
ax.set_xticklabels([LABEL[s] for s in SC], fontsize=8.5)
ax.set_ylim(0, 11)
ax.set_ylabel("Goodput (Mbit/s)")
ax.set_title("Simulator vs lab for the learned controller: the lab is 1–6% lower",
             loc="left", fontsize=11, color=INK)
ax.legend(frameon=False, fontsize=9, loc="upper right")
style(ax)
fig.text(0.01, 0.005, "The noise level was measured on the same lab, so S1–S4 are a consistency check; "
         "S5 values were fixed before the lab runs.", fontsize=8, color="#6b6a66")
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(os.path.join(OUT, "fig5_sim_vs_lab.png"), dpi=170)
plt.close(fig)
print("wrote 5 figures to", OUT)
