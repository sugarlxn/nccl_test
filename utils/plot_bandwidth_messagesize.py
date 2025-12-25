#!/usr/bin/env python3
import argparse
import json
import gzip
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# Utilities
# -----------------------------
def smart_open(path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path, "r")

def bytes_to_kb(bytes_):
    return bytes_ / 1024.0

# -----------------------------
# Argument Parser
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot NCCL AllReduce bandwidth vs message size (Figure 1)"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to NCCL Inspector log file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="figure1_allreduce_bandwidth",
        help="Output figure name (without extension)",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="Default NCCL",
        help="Legend label (e.g., Default NCCL / Tuned NCCL)",
    )
    return parser.parse_args()

# -----------------------------
# Main
# -----------------------------
def main():
    args = parse_args()
    input_path = Path(args.input)

    records = []

    with smart_open(input_path) as f:
        for line in f:
            rec = json.loads(line)

            if rec["coll_perf"]["coll"] != "AllReduce":
                continue
            if rec["header"]["rank"] != 0:
                continue

            records.append(
                {
                    "msg_size_bytes": rec["coll_perf"]["coll_msg_size_bytes"],
                    "algobw_gbs": rec["coll_perf"]["coll_algobw_gbs"],
                }
            )

    df = pd.DataFrame(records)

    # Aggregate: mean bandwidth per message size
    df = (
        df.groupby("msg_size_bytes", as_index=False)
        .agg(mean_algobw_gbs=("algobw_gbs", "mean"))
        .sort_values("msg_size_bytes")
    )

    # -----------------------------
    # Plot (Publication-quality)
    # -----------------------------
    plt.figure(figsize=(7.0, 4.5), dpi=300)

    x_kb = df["msg_size_bytes"].apply(bytes_to_kb)
    y_bw = df["mean_algobw_gbs"]

    plt.plot(
        x_kb,
        y_bw,
        color="#1f77b4",          # Deep blue (Default NCCL)
        marker="o",
        markersize=6,
        linewidth=2.5,
        markeredgewidth=0.8,
        markeredgecolor="black",
        label=args.label,
    )

    plt.xscale("log", base=2)
    plt.xlabel("Message Size (KB, log scale)", fontsize=11)
    plt.ylabel("Bandwidth (GB/s)", fontsize=11)

    plt.grid(
        True,
        which="both",
        linestyle="--",
        linewidth=0.6,
        alpha=0.6,
    )

    plt.legend(
        loc="lower right",
        fontsize=10,
        frameon=False,
    )

    plt.tight_layout()

    plt.savefig(f"{args.output}.png", dpi=300)
    plt.savefig(f"{args.output}.pdf")
    plt.close()

    print(f"[OK] Figure saved as {args.output}.png / .pdf")

if __name__ == "__main__":
    main()
