"""Summarize results and make plots.

    python experiments/analyze.py                 # reads results/results.csv, logs/
    python experiments/analyze.py --include-copy  # also show reno_copy (the plug-in test)

Writes to results/ (or --outdir):
    summary.csv            mean/std/n per scenario x controller (a table view of every plot)
    summary.png            goodput and p95 RTT per scenario (mean +/- std across runs)
    timeseries_<S>.png     cwnd and RTT over time, run 1 of each controller
                           (S3 shows the bandwidth-drop time)

Metric definitions (state these in the report; they are my choices, not a standard):
    rtt_p95_ms / rtt_mean_ms  from the server's rtt_latest samples (10 ms grid) over the run.
    S3 only, measured from the moment of the bandwidth drop:
      drop_rtt_peak_ms   highest rtt_latest after the drop (queue build-up).
      post_drop_rtt_mean_ms  mean rtt_latest from the drop to the end (how much queue it keeps).
      drop_lost_kb       bytes declared lost in the 2 s after the drop (overshoot cost).
      settle_time_s      first time the 0.5 s sending rate has stayed within +/-20% of the
                         new link rate for 1.0 s; NaN = never settled (e.g. sawtooth).
Controller colors are fixed per controller (categorical slots 1-3 of a validated palette),
never reassigned by rank. reno_copy is neutral gray: it is a check, not a contender.
"""
import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
from run_experiments import SCENARIOS  # noqa: E402

COLORS = {"reno": "#2a78d6", "cubic": "#eb6834", "learned": "#0e7a56", "learned_ppo_v2": "#1baf7a",
          "fixed40": "#eda100",
          "reno_copy": "#8a8984"}
STYLES = {"reno": "-", "cubic": "--", "learned": "-.", "learned_ppo_v2": "-", "fixed40": (0, (5, 1, 1, 1)), "reno_copy": ":"}
ORDER = ["reno", "cubic", "learned", "learned_ppo_v2", "fixed40", "reno_copy"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
TITLES = {"S1": "S1 baseline\n10 Mbit/s, 40 ms", "S2": "S2 random loss\n1% loss",
          "S3": "S3 bandwidth drop\n20→5 Mbit/s", "S4": "S4 long delay\n200 ms RTT",
          "S5": "S5 held-out"}
# S5_PATCH: colours for controllers added after the first version of this file
for _cc, _col, _ls in (("learned_ppo_v2", "#1baf7a", "-."), ("ppo_v1", "#1baf7a", ":"),
                       ("fixed4", "#c9c8c4", ":"), ("fixed8", "#b3b2ae", ":"),
                       ("fixed16", "#9d9c97", ":"),
                       ("learned_ppo_v2_s11", "#6fd1ad", "--"), ("learned_ppo_v2_s12", "#0f7d57", "--")):
    COLORS.setdefault(_cc, _col)
    STYLES.setdefault(_cc, _ls)
    if _cc not in ORDER:
        ORDER.append(_cc)
TITLES.setdefault("S5a", "S5a held-out\n4% loss")
TITLES.setdefault("S5b", "S5b held-out\n250 ms RTT")
TITLES.setdefault("S5c", "S5c held-out\n0.2 BDP queue")


def style_ax(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)
    ax.title.set_color(INK)


def load_log(row):
    path = os.path.join(ROOT, row["log_file"])
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    # the sampler keeps running until the connection closes: drop the idle tail so
    # repeated last-values don't bias RTT statistics or draw flat lines
    active = np.flatnonzero(df["bytes_in_flight"].to_numpy() > 0)
    return df.iloc[: active[-1] + 1] if len(active) else df


def send_rate_mbps(df, window=0.5):
    t, sent = df["t"].to_numpy(), df["sent_bytes"].to_numpy()
    j = np.searchsorted(t, t - window, side="left")
    dt = np.maximum(t - t[j], 1e-6)
    return (sent - sent[j]) * 8 / dt / 1e6


def run_metrics(row):
    out = {"rtt_p95_ms": np.nan, "rtt_mean_ms": np.nan, "drop_rtt_peak_ms": np.nan,
           "post_drop_rtt_mean_ms": np.nan,
           "drop_lost_kb": np.nan, "settle_time_s": np.nan}
    df = load_log(row)
    if df is None:
        return out
    rtt = df.loc[df["rtt_latest"] > 0, "rtt_latest"] * 1000
    if len(rtt):
        out["rtt_p95_ms"] = float(np.percentile(rtt, 95))
        out["rtt_mean_ms"] = float(rtt.mean())
    sc = SCENARIOS.get(row["scenario"], {})
    if "drop_to" in sc and pd.notna(row["drop_epoch"]) and row["drop_epoch"] != "":
        td = float(row["drop_epoch"]) - float(row["t0_epoch"])
        after = df[df["t"] >= td]
        if len(after):
            out["drop_rtt_peak_ms"] = float(after["rtt_latest"].max() * 1000)
            post = after.loc[after["rtt_latest"] > 0, "rtt_latest"] * 1000
            out["post_drop_rtt_mean_ms"] = float(post.mean())
            end = after[after["t"] <= td + 2.0]
            out["drop_lost_kb"] = float((end["lost_bytes"].iloc[-1] - after["lost_bytes"].iloc[0]) / 1000)
            rate = send_rate_mbps(df)
            target = sc["drop_to"]
            ok = (rate >= 0.8 * target) & (rate <= 1.2 * target) & (df["t"].to_numpy() >= td)
            t = df["t"].to_numpy()
            start = None
            for i in range(len(t)):
                if ok[i]:
                    start = t[i] if start is None else start
                    if t[i] - start >= 1.0:
                        out["settle_time_s"] = float(start - td)
                        break
                else:
                    start = None
    return out


def summarize(res):
    extra = pd.DataFrame([run_metrics(r) for _, r in res.iterrows()], index=res.index)
    res = pd.concat([res, extra], axis=1)
    g = res.groupby(["scenario", "cc"])
    summ = g.agg(n=("run", "count"),
                 goodput_mean=("goodput_mbps", "mean"), goodput_std=("goodput_mbps", "std"),
                 rtt_p95_mean=("rtt_p95_ms", "mean"), rtt_p95_std=("rtt_p95_ms", "std"),
                 rtt_mean=("rtt_mean_ms", "mean"),
                 loss_ratio_mean=("loss_ratio", "mean"), loss_ratio_std=("loss_ratio", "std"),
                 drop_rtt_peak_ms=("drop_rtt_peak_ms", "mean"),
                 post_drop_rtt_mean_ms=("post_drop_rtt_mean_ms", "mean"),
                 drop_lost_kb=("drop_lost_kb", "mean"),
                 settle_time_s=("settle_time_s", "mean"),
                 settled_runs=("settle_time_s", lambda s: int(s.notna().sum()))).reset_index()
    return res, summ


def bar_panel(ax, summ, scenarios, ccs, mean_col, std_col, ylabel):
    w = 0.8 / max(len(ccs), 1)
    x0 = np.arange(len(scenarios))
    for i, cc in enumerate(ccs):
        m, s = [], []
        for sc in scenarios:
            r = summ[(summ.scenario == sc) & (summ.cc == cc)]
            m.append(r[mean_col].iloc[0] if len(r) else np.nan)
            s.append(r[std_col].iloc[0] if len(r) else 0)
        ax.bar(x0 + (i - (len(ccs) - 1) / 2) * w, m, w * 0.9, yerr=np.nan_to_num(s),
               color=COLORS[cc], label=cc, error_kw=dict(ecolor=INK2, elinewidth=1, capsize=2))
    ax.set_xticks(x0)
    ax.set_xticklabels([TITLES.get(s, s) for s in scenarios], fontsize=8, color=INK2)
    ax.set_ylabel(ylabel)
    style_ax(ax)


def plot_summary(summ, ccs, outdir):
    scenarios = [s for s in SCENARIOS if s in set(summ.scenario)] + \
                [s for s in sorted(set(summ.scenario)) if s not in SCENARIOS]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURFACE)
    bar_panel(axes[0], summ, scenarios, ccs, "goodput_mean", "goodput_std", "Goodput (Mbit/s)")
    axes[0].set_title("Goodput (higher is better)", fontsize=11, loc="left")
    bar_panel(axes[1], summ, scenarios, ccs, "rtt_p95_mean", "rtt_p95_std", "95th percentile RTT (ms)")
    axes[1].set_title("Delay (lower is better)", fontsize=11, loc="left")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=len(ccs), frameon=False, fontsize=10,
               labelcolor=INK)
    fig.text(0.5, 0.005, "Bars: mean across runs; whiskers: ±1 std. See summary.csv for every value.",
             ha="center", fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 0.93))
    fig.savefig(os.path.join(outdir, "summary.png"), dpi=160, facecolor=SURFACE)
    plt.close(fig)


def plot_timeseries(res, name, ccs, outdir):
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True, facecolor=SURFACE)
    drop_t = None
    for cc in ccs:
        r = res[(res.scenario == name) & (res.cc == cc) & (res.run == 1)]
        if r.empty:
            continue
        row = r.iloc[0]
        df = load_log(row)
        if df is None:
            continue
        kw = dict(color=COLORS[cc], linestyle=STYLES[cc], linewidth=1.5, label=cc)
        axes[0].plot(df["t"], df["cwnd"] / 1000, **kw)
        v = df["rtt_latest"] > 0
        axes[1].plot(df.loc[v, "t"], df.loc[v, "rtt_latest"] * 1000, **kw)
        if pd.notna(row["drop_epoch"]) and row["drop_epoch"] != "":
            drop_t = float(row["drop_epoch"]) - float(row["t0_epoch"])
    axes[0].set_ylabel("Congestion window (KB)")
    axes[1].set_ylabel("RTT (ms)")
    axes[1].set_xlabel("Time since transfer start (s)")
    for ax in axes:
        style_ax(ax)
        if drop_t is not None:
            ax.axvline(drop_t, color=INK2, linewidth=1, linestyle=(0, (2, 3)))
    if drop_t is not None:
        axes[0].text(drop_t, axes[0].get_ylim()[1], " bandwidth drop", color=INK2,
                     fontsize=8, va="top")
    axes[0].set_title(f"{TITLES.get(name, name).replace(chr(10), ', ')}: run 1", fontsize=11,
                      loc="left")
    axes[0].legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"timeseries_{name}.png"), dpi=160, facecolor=SURFACE)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default=os.path.join(ROOT, "results", "results.csv"))
    p.add_argument("--outdir", default=os.path.join(ROOT, "results"))
    p.add_argument("--include-copy", action="store_true",
                   help="show reno_copy (plug-in test) next to the real controllers")
    args = p.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    res = pd.read_csv(args.results)
    res, summ = summarize(res)
    summ.to_csv(os.path.join(args.outdir, "summary.csv"), index=False)
    ccs = [c for c in ORDER if c in set(res.cc) and (c != "reno_copy" or args.include_copy)]
    unknown = [c for c in sorted(set(res.cc)) if c not in COLORS]
    if unknown:
        print("note: no fixed color for", unknown, "- add them to COLORS; skipped in plots")
    plot_summary(summ[summ.cc.isin(ccs)], ccs, args.outdir)
    for name in sorted(set(res.scenario)):
        plot_timeseries(res, name, ccs, args.outdir)

    show = summ[summ.cc.isin(ccs)].copy()
    cols = ["scenario", "cc", "n", "goodput_mean", "goodput_std", "rtt_p95_mean",
            "loss_ratio_mean", "post_drop_rtt_mean_ms", "settle_time_s"]
    print(show[cols].round(3).to_string(index=False))
    print(f"\nwrote summary.csv, summary.png, timeseries_*.png to {args.outdir}")


if __name__ == "__main__":
    main()
