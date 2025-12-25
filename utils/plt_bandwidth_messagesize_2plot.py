#!/usr/bin/env python3
import argparse
import json
import gzip
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Utilities
# -----------------------------
def smart_open(path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path, "r")

def bytes_to_kb(bytes_):
    return bytes_ / 1024.0

def load_nccl_log(path: Path):
    """Load NCCL Inspector log and aggregate mean algBW per message size"""
    records = []

    with smart_open(path) as f:
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

    df = (
        df.groupby("msg_size_bytes", as_index=False)
        .agg(mean_algobw_gbs=("algobw_gbs", "mean"))
        .sort_values("msg_size_bytes")
    )

    return df

# -----------------------------
# Argument Parser
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare NCCL AllReduce bandwidth vs message size"
    )
    parser.add_argument("--input1", type=str, required=True, help="Log file 1")
    parser.add_argument("--input2", type=str, required=True, help="Log file 2")
    parser.add_argument(
        "--label1", type=str, default="Default NCCL", help="Legend label for input1"
    )
    parser.add_argument(
        "--label2", type=str, default="Tuned NCCL", help="Legend label for input2"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="figure_allreduce_bandwidth_compare",
        help="Output figure name (without extension)",
    )
    return parser.parse_args()

# -----------------------------
# Main
# -----------------------------
def main():
    args = parse_args()

    df1 = load_nccl_log(Path(args.input1))
    df2 = load_nccl_log(Path(args.input2))

    # -----------------------------
    # Plot (Publication-quality)
    # -----------------------------
    plt.figure(figsize=(7.0, 4.5), dpi=300)

    # input1
    plt.plot(
        df1["msg_size_bytes"].apply(bytes_to_kb),
        df1["mean_algobw_gbs"],
        color="#1f77b4",      # Deep blue
        marker="o",
        markersize=5.5,
        linewidth=2.4,
        markeredgewidth=0.8,
        markeredgecolor="black",
        label=args.label1,
    )

    # input2
    plt.plot(
        df2["msg_size_bytes"].apply(bytes_to_kb),
        df2["mean_algobw_gbs"],
        color="#ff7f0e",      # Warm orange
        marker="s",
        markersize=5.5,
        linewidth=2.4,
        markeredgewidth=0.8,
        markeredgecolor="black",
        label=args.label2,
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
