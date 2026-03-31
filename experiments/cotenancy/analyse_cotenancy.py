#!/usr/bin/env python3
"""
experiments/cotenancy/analyse_cotenancy.py
==========================================
Post-hoc analysis of the co-tenancy side-channel experiment.

Reads victim_log.csv (timestamp_us, class) and attacker_log.csv
(timestamp_us, wall_us, gpu_us), assigns each attacker probe to
the victim class that was active at that moment using nearest-
timestamp matching, then runs Welch's t-test to determine whether
the attacker can distinguish victim class from its own probe timing.

Both wall-clock and GPU-event timings are analysed independently
and reported in the same format as harness/tvla_analysis.py.

Usage:
    python3 experiments/cotenancy/analyse_cotenancy.py \\
        /logs/victim_log.csv \\
        /logs/attacker_log.csv \\
        [--victim-offset-us 2000000] \\
        [--out report.txt]

The --victim-offset-us flag specifies how many microseconds before
the attacker the victim started (default: 2 000 000 = 2 seconds,
matching the 'sleep 2' in run_cotenancy.sh).
"""

import sys
import os
import argparse
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

TVLA_THRESHOLD = 4.5


# ── Loaders ───────────────────────────────────────────────

def load_victim_log(path):
    """Load victim_log.csv -> (timestamps_us, classes) numpy arrays."""
    if not os.path.exists(path):
        print(f"[ERROR] File not found: {path}")
        sys.exit(1)
    timestamps, classes = [], []
    with open(path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if i == 0 or not line:
                continue  # skip header
            parts = line.split(',')
            if len(parts) != 2:
                continue
            try:
                timestamps.append(float(parts[0]))
                classes.append(int(parts[1]))
            except ValueError:
                continue
    if not timestamps:
        print(f"[ERROR] No data rows in {path}")
        sys.exit(1)
    return np.array(timestamps), np.array(classes)


def load_attacker_log(path):
    """Load attacker_log.csv -> dict of numpy arrays."""
    if not os.path.exists(path):
        print(f"[ERROR] File not found: {path}")
        sys.exit(1)
    ts, wall, gpu = [], [], []
    with open(path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if i == 0 or not line:
                continue  # skip header
            parts = line.split(',')
            if len(parts) != 3:
                continue
            try:
                ts.append(float(parts[0]))
                wall.append(float(parts[1]))
                gpu.append(float(parts[2]))
            except ValueError:
                continue
    if not ts:
        print(f"[ERROR] No data rows in {path}")
        sys.exit(1)
    return {
        'timestamp_us': np.array(ts),
        'wall_us':       np.array(wall),
        'gpu_us':        np.array(gpu),
    }


# ── Class matching ────────────────────────────────────────

def match_classes(atk_ts, victim_ts, victim_classes, victim_offset_us):
    """
    Assign each attacker probe to the victim class active at that time.

    atk_ts is relative to attacker start.
    victim_ts is relative to victim start.
    The victim started victim_offset_us microseconds before the attacker, so:
        victim_time = atk_ts + victim_offset_us

    Only probes whose victim_time falls within [victim_ts[0], victim_ts[-1]]
    are included — probes outside the victim's run window are discarded.

    Returns (valid_probe_mask, matched_classes).
    """
    victim_min = victim_ts[0]
    victim_max = victim_ts[-1]

    valid_mask = np.zeros(len(atk_ts), dtype=bool)
    matched    = np.zeros(len(atk_ts), dtype=int)

    for i, ts in enumerate(atk_ts):
        vt = ts + victim_offset_us
        if vt < victim_min or vt > victim_max:
            continue  # outside victim run window
        idx = int(np.argmin(np.abs(victim_ts - vt)))
        valid_mask[i] = True
        matched[i] = victim_classes[idx]

    return valid_mask, matched


# ── Statistical analysis ──────────────────────────────────

def remove_outliers(data, z_threshold=5.0):
    mean = np.mean(data)
    std  = np.std(data)
    if std == 0:
        return data, 0
    mask = np.abs((data - mean) / std) < z_threshold
    return data[mask], int(np.sum(~mask))


def welch_tvla(c0, c1):
    return stats.ttest_ind(c0, c1, equal_var=False)


def sliding_window_tvla(c0, c1, window_size=200, step=50):
    n = min(len(c0), len(c1))
    t_vals = []
    for start in range(0, n - window_size, step):
        end = start + window_size
        t, _ = welch_tvla(c0[start:end], c1[start:end])
        t_vals.append(abs(t))
    return np.array(t_vals)


def run_analysis(label, class0_raw, class1_raw):
    """
    Run full TVLA on one timing metric.
    Returns (report_lines, t_abs, p_value).
    """
    lines = []
    lines.append("")
    lines.append(f"[{label}]")

    c0, n0 = remove_outliers(class0_raw)
    c1, n1 = remove_outliers(class1_raw)
    lines.append(f"  Removed {n0} outliers from class 0, {n1} from class 1")

    lines.append(
        f"  Class 0: N={len(c0):,}  mean={np.mean(c0):.4f} us"
        f"  std={np.std(c0):.4f} us"
    )
    lines.append(
        f"  Class 1: N={len(c1):,}  mean={np.mean(c1):.4f} us"
        f"  std={np.std(c1):.4f} us"
    )
    mean_diff = np.mean(c1) - np.mean(c0)
    lines.append(f"  Mean difference: {mean_diff:.4f} us")

    t, p = welch_tvla(c0, c1)
    t_abs = abs(t)
    verdict = (
        "*** LEAKAGE DETECTED ***" if t_abs >= TVLA_THRESHOLD
        else "No leakage detected"
    )
    lines.append(f"  t-statistic: {t:.6f}")
    lines.append(f"  |t-statistic|: {t_abs:.6f}")
    lines.append(f"  p-value: {p:.2e}")
    lines.append(f"  Threshold: {TVLA_THRESHOLD}")
    lines.append("")
    lines.append("=" * 52)
    lines.append(f"  VERDICT: {verdict}")
    lines.append(f"  |t| = {t_abs:.4f} {'>' if t_abs >= TVLA_THRESHOLD else '<'} {TVLA_THRESHOLD}")
    lines.append("=" * 52)

    t_sw = sliding_window_tvla(c0, c1)
    max_sw = float(np.max(t_sw)) if len(t_sw) > 0 else 0.0
    frac   = float(np.mean(t_sw >= TVLA_THRESHOLD)) if len(t_sw) > 0 else 0.0
    lines.append(f"  Sliding window max |t|: {max_sw:.4f}")
    lines.append(f"  Windows above threshold: {frac * 100:.1f}%")

    return lines, t_abs, p


# ── Main ──────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Co-tenancy side-channel analysis'
    )
    parser.add_argument('victim_log',
        help='victim_log.csv (timestamp_us,class)')
    parser.add_argument('attacker_log',
        help='attacker_log.csv (timestamp_us,wall_us,gpu_us)')
    parser.add_argument('--victim-offset-us', type=float, default=2_000_000.0,
        help='Microseconds by which victim started before attacker '
             '(default: 2000000)')
    parser.add_argument('--out', default=None,
        help='Save report to this file path')
    args = parser.parse_args()

    print()
    print("=" * 60)
    print("  mlkem-gpu-sec Co-tenancy Side-Channel Analysis")
    print("=" * 60)

    # Load logs
    print("\n[Loading logs]")
    v_ts, v_cls = load_victim_log(args.victim_log)
    atk = load_attacker_log(args.attacker_log)

    n_class0_victim = int(np.sum(v_cls == 0))
    n_class1_victim = int(np.sum(v_cls == 1))
    print(f"  Victim log:   {len(v_ts):,} entries "
          f"(class 0: {n_class0_victim:,}  class 1: {n_class1_victim:,})")
    print(f"  Attacker log: {len(atk['timestamp_us']):,} probes")
    print(f"  Victim start offset: {args.victim_offset_us / 1e6:.1f}s "
          f"before attacker")

    # Match probes to victim classes
    print("\n[Matching attacker probes to victim class]")
    valid_mask, matched_classes = match_classes(
        atk['timestamp_us'], v_ts, v_cls, args.victim_offset_us)

    n_valid = int(np.sum(valid_mask))
    n_discarded = len(atk['timestamp_us']) - n_valid
    print(f"  Probes in overlap window: {n_valid:,} "
          f"({n_discarded:,} outside victim run — discarded)")

    class0_mask = valid_mask & (matched_classes == 0)
    class1_mask = valid_mask & (matched_classes == 1)
    n0 = int(np.sum(class0_mask))
    n1 = int(np.sum(class1_mask))
    print(f"  Probes matched to class 0: {n0:,}")
    print(f"  Probes matched to class 1: {n1:,}")

    if n0 < 30 or n1 < 30:
        print("\n[ERROR] Too few probes matched to one class "
              f"(n0={n0}, n1={n1}).")
        print("  Check --victim-offset-us or verify log files contain data.")
        sys.exit(1)

    wall0 = atk['wall_us'][class0_mask]
    wall1 = atk['wall_us'][class1_mask]
    gpu0  = atk['gpu_us'][class0_mask]
    gpu1  = atk['gpu_us'][class1_mask]

    # Run TVLA on both metrics
    report_header = [
        "mlkem-gpu-sec Co-tenancy TVLA Report",
        "=" * 60,
        f"Victim log:      {args.victim_log}",
        f"Attacker log:    {args.attacker_log}",
        f"Victim offset:   {args.victim_offset_us:.0f} us",
        f"Total probes:    {len(atk['timestamp_us']):,}",
        f"In-window probes:{n_valid:,}",
        f"Matched class 0: {n0:,}",
        f"Matched class 1: {n1:,}",
        f"TVLA threshold:  {TVLA_THRESHOLD}",
    ]

    all_report_lines = report_header[:]

    for label, c0, c1 in [
        ("Wall-clock timing (wall_us)", wall0, wall1),
        ("GPU-event timing  (gpu_us)",  gpu0,  gpu1),
    ]:
        lines, t_abs, p = run_analysis(label, c0, c1)
        for ln in lines:
            print(ln)
        all_report_lines.extend(lines)

    # Save report
    report_text = '\n'.join(all_report_lines) + '\n'
    if args.out:
        out_path = args.out
    else:
        out_path = os.path.join(
            os.path.dirname(os.path.abspath(args.attacker_log)),
            'cotenancy_tvla_report.txt'
        )
    with open(out_path, 'w') as f:
        f.write(report_text)
    print(f"\n  Report saved: {out_path}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
