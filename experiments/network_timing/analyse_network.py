#!/usr/bin/env python3
"""
experiments/network_timing/analyse_network.py
=============================================
TVLA analysis for the network timing attack experiment.

Reads network_timing.csv (timestamp_us, class, rtt_us), splits by class,
removes outliers (z > 5.0), runs Welch's t-test, and reports the result
in the same format as harness/tvla_analysis.py.

Usage:
    python3 experiments/network_timing/analyse_network.py \\
        experiments/network_timing/logs/network_timing.csv \\
        [--out experiments/network_timing/logs/network_tvla_report.txt]
"""

import sys
import os
import argparse
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

TVLA_THRESHOLD = 4.5


# ── Loader ────────────────────────────────────────────────

def load_csv(path: str):
    """
    Load network_timing.csv.
    Returns (timestamps_us, classes, rtt_us) as numpy arrays.
    """
    if not os.path.exists(path):
        print(f"[ERROR] File not found: {path}")
        sys.exit(1)

    timestamps, classes, rtts = [], [], []
    with open(path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if i == 0 or not line:
                continue  # skip header
            parts = line.split(',')
            if len(parts) != 3:
                continue
            try:
                timestamps.append(float(parts[0]))
                classes.append(int(parts[1]))
                rtts.append(float(parts[2]))
            except ValueError:
                continue

    if not timestamps:
        print(f"[ERROR] No data rows in {path}")
        sys.exit(1)

    return (np.array(timestamps),
            np.array(classes, dtype=int),
            np.array(rtts))


# ── Statistics ────────────────────────────────────────────

def remove_outliers(data: np.ndarray, z_threshold: float = 5.0):
    mean = np.mean(data)
    std  = np.std(data)
    if std == 0:
        return data, 0
    mask = np.abs((data - mean) / std) < z_threshold
    return data[mask], int(np.sum(~mask))


def welch_t_test(c0: np.ndarray, c1: np.ndarray):
    return stats.ttest_ind(c0, c1, equal_var=False)


def sliding_window_tvla(c0: np.ndarray, c1: np.ndarray,
                        window_size: int = 1000, step: int = 100):
    n = min(len(c0), len(c1))
    t_vals = []
    for start in range(0, n - window_size, step):
        end = start + window_size
        t, _ = welch_t_test(c0[start:end], c1[start:end])
        t_vals.append(abs(float(t)))
    return np.array(t_vals)


def print_stats(label: str, data: np.ndarray) -> None:
    print(f"  {label}:")
    print(f"    N:      {len(data):,}")
    print(f"    Mean:   {np.mean(data):.4f} us")
    print(f"    StdDev: {np.std(data):.4f} us")
    print(f"    Median: {np.median(data):.4f} us")
    print(f"    Min:    {np.min(data):.4f} us")
    print(f"    Max:    {np.max(data):.4f} us")


# ── Main ──────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="TVLA analysis for the network timing attack experiment"
    )
    parser.add_argument("csv",
        help="network_timing.csv (timestamp_us,class,rtt_us)")
    parser.add_argument("--out", default=None,
        help="Save report to this path")
    args = parser.parse_args()

    print()
    print("=" * 56)
    print("  mlkem-gpu-sec Network Timing Attack: TVLA Analysis")
    print("=" * 56)

    # ── Load ──────────────────────────────────────────────
    print(f"\n[Loading] {args.csv}")
    ts, classes, rtts = load_csv(args.csv)

    mask0 = classes == 0
    mask1 = classes == 1
    rtt0_raw = rtts[mask0]
    rtt1_raw = rtts[mask1]

    print(f"  Total rows:      {len(rtts):,}")
    print(f"  Class 0 (valid): {len(rtt0_raw):,} measurements")
    print(f"  Class 1 (random):{len(rtt1_raw):,} measurements")

    if len(rtt0_raw) < 30 or len(rtt1_raw) < 30:
        print("[ERROR] Fewer than 30 measurements for one class — aborting.")
        return 1

    # ── Outlier removal ───────────────────────────────────
    print("\n[Preprocessing]")
    rtt0, n_out0 = remove_outliers(rtt0_raw)
    rtt1, n_out1 = remove_outliers(rtt1_raw)
    print(f"  Class 0: removed {n_out0} outliers (z > 5.0)")
    print(f"  Class 1: removed {n_out1} outliers (z > 5.0)")

    # ── Statistics ────────────────────────────────────────
    print("\n[Statistics — Round-Trip Time (RTT)]")
    print_stats("Class 0 (valid ciphertext)", rtt0)
    print()
    print_stats("Class 1 (random ciphertext)", rtt1)

    mean_diff = float(np.mean(rtt1) - np.mean(rtt0))
    print(f"\n  Mean difference (class1 − class0): {mean_diff:.4f} us")

    # ── Welch's t-test ────────────────────────────────────
    print("\n[TVLA — Welch's t-test on RTT]")
    t_stat, p_value = welch_t_test(rtt0, rtt1)
    t_abs = abs(float(t_stat))

    print(f"  t-statistic:   {t_stat:.6f}")
    print(f"  |t-statistic|: {t_abs:.6f}")
    print(f"  p-value:       {p_value:.2e}")
    print(f"  Threshold:     {TVLA_THRESHOLD}")

    verdict = "*** LEAKAGE DETECTED ***" if t_abs >= TVLA_THRESHOLD else "No leakage detected"
    print()
    print("=" * 56)
    if t_abs >= TVLA_THRESHOLD:
        print(f"  VERDICT: {verdict}")
        print(f"  |t| = {t_abs:.4f} >= {TVLA_THRESHOLD}")
        print(f"  Round-trip time differs significantly between classes.")
        print(f"  A remote attacker can distinguish valid from random ct.")
    else:
        print(f"  VERDICT: {verdict}")
        print(f"  |t| = {t_abs:.4f} < {TVLA_THRESHOLD}")
        print(f"  No significant RTT difference between classes.")
    print("=" * 56)

    # ── Sliding window ────────────────────────────────────
    print("\n[Sliding Window TVLA]")
    t_sw = sliding_window_tvla(rtt0, rtt1)
    max_sw  = float(np.max(t_sw))   if len(t_sw) > 0 else 0.0
    frac_sw = float(np.mean(t_sw >= TVLA_THRESHOLD)) if len(t_sw) > 0 else 0.0
    print(f"  Max |t| in windows: {max_sw:.4f}")
    print(f"  Windows above threshold: {frac_sw * 100:.1f}%")

    # ── Build report ──────────────────────────────────────
    report_lines = [
        "mlkem-gpu-sec Network Timing Attack TVLA Report",
        "=" * 56,
        f"Input CSV:     {args.csv}",
        f"Total rows:    {len(rtts):,}",
        f"Class 0 (valid):  {len(rtt0_raw):,} raw  →  {len(rtt0):,} after outlier removal",
        f"Class 1 (random): {len(rtt1_raw):,} raw  →  {len(rtt1):,} after outlier removal",
        f"",
        f"[Round-Trip Time]",
        f"  Class 0 mean:  {np.mean(rtt0):.4f} us  std: {np.std(rtt0):.4f} us",
        f"  Class 1 mean:  {np.mean(rtt1):.4f} us  std: {np.std(rtt1):.4f} us",
        f"  Mean diff:     {mean_diff:.4f} us  (class1 − class0)",
        f"",
        f"[Welch's t-test]",
        f"  t-statistic:   {t_stat:.6f}",
        f"  |t-statistic|: {t_abs:.6f}",
        f"  p-value:       {p_value:.2e}",
        f"  Threshold:     {TVLA_THRESHOLD}",
        f"  Verdict:       {verdict}",
        f"",
        f"[Sliding Window]",
        f"  Max |t|:               {max_sw:.4f}",
        f"  Windows above threshold: {frac_sw * 100:.1f}%",
    ]

    # ── Save report ───────────────────────────────────────
    if args.out:
        out_path = args.out
    else:
        script_dir = os.path.dirname(os.path.abspath(args.csv))
        out_path = os.path.join(script_dir, "network_tvla_report.txt")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(report_lines) + "\n")
    print(f"\n  Report saved: {out_path}")

    return 0 if t_abs < TVLA_THRESHOLD else 1


if __name__ == "__main__":
    sys.exit(main())
