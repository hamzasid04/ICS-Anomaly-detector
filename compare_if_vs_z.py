#!/usr/bin/env python
"""
Compares trains_iforest.py scores and outputs vs z-score: align timelines, produce a small table and a side-by-side plot.

Inputs
  outputs/scores_iforest.csv  (from train_iforest.py)
  outputs/scores.csv          (from make_zscore_alert)

Outputs
  outputs/comparison.csv
  outputs/if_vs_z_timelines.png
  Console: alerts/day metrics
"""

from pathlib import Path
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def alerts_per_day(alerts_df):
    # average number of alerts per day for a quick sanity metric
    if alerts_df.empty:
        return 0.0
    d = alerts_df.copy()
    d["day"] = d.index.normalize()
    return d.groupby("day").size().mean()

def main():
    ap = argparse.ArgumentParser()
    # inputs from earlier steps
    ap.add_argument("--if_scores", default="outputs/scores_iforest.csv")
    ap.add_argument("--z_scores",  default="outputs/scores.csv")
    ap.add_argument("--if_alerts", default="outputs/alerts_iforest.csv")
    ap.add_argument("--z_alerts",  default="outputs/alerts.csv")
    # outputs we create here
    ap.add_argument("--out_csv",   default="outputs/comparison.csv")
    ap.add_argument("--out_png",   default="outputs/if_vs_z_timelines.png")
    args = ap.parse_args()

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)

    # load both score files and align them on time
    ifs = pd.read_csv(args.if_scores, parse_dates=["timestamp"]).set_index("timestamp").sort_index()
    zbs = pd.read_csv(args.z_scores,  parse_dates=["timestamp"]).set_index("timestamp").sort_index()

    # merge on time index so each row has if_score and z_score for the same minute
    merged = ifs.rename(columns={"score":"if_score"}).join(
        zbs.rename(columns={"score":"z_score"}), how="inner"
    ) 

    # load alert tables if they exist
    try:
        if_alerts = pd.read_csv(args.if_alerts, parse_dates=["timestamp"]).set_index("timestamp").sort_index()
    except FileNotFoundError:
        if_alerts = merged.iloc[0:0][[]]
    try:
        z_alerts  = pd.read_csv(args.z_alerts,  parse_dates=["timestamp"]).set_index("timestamp").sort_index()
    except FileNotFoundError:
        z_alerts  = merged.iloc[0:0][[]]

    # build a short scoreboard
    # take the top 20 times by IF score and mark whether each method alerted
    top = merged.sort_values("if_score", ascending=False).head(20)
    out = top.copy()
    out["if_is_alert"] = out.index.isin(if_alerts.index)
    out["z_is_alert"]  = out.index.isin(z_alerts.index)
    out.to_csv(args.out_csv)

    # quick sanity metric printed to console
    if_apd = alerts_per_day(if_alerts)
    z_apd  = alerts_per_day(z_alerts)
    print(f"IF alerts per day about {if_apd:.2f}  Z alerts per day about {z_apd:.2f}")
    print(f"Wrote {args.out_csv}")

    # draw two lanes
    fig = plt.figure(figsize=(12,6))

    # top lane IF timeline
    ax1 = fig.add_subplot(2,1,1)
    ax1.plot(ifs.index, ifs["score"])
    if not if_alerts.empty:
        ax1.plot(if_alerts.index, if_alerts["score"], linestyle="", marker="o")
    ax1.set_title("Isolation Forest score  dots are IF alerts")
    ax1.set_ylabel("IF score")

    # bottom lane Z timeline
    ax2 = fig.add_subplot(2,1,2, sharex=ax1)
    ax2.plot(zbs.index, zbs["score"])
    if not z_alerts.empty:
        ax2.plot(z_alerts.index, z_alerts["score"], linestyle="", marker="o")
    ax2.set_title("Z score baseline  dots are Z alerts")
    ax2.set_ylabel("max abs z")
    ax2.set_xlabel("time")

    fig.tight_layout()
    fig.savefig(args.out_png, dpi=150)
    plt.close(fig)
    print(f"Wrote {args.out_png}")

if __name__ == "__main__":
    main()
