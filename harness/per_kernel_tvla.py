#!/usr/bin/env python3
"""
harness/per_kernel_tvla.py
============================================================
Per-kernel TVLA analysis for serialized decapsulation traces.

Loads two per-kernel CSV files (class 0 and class 1) produced by
trace_ser_kyberXXX.out, runs Welch's t-test independently for each
kernel, and reports |t| per kernel sorted by magnitude.

This decomposes the aggregate |t| signal into per-kernel contributions
to identify exactly which kernels carry the timing leakage.

Usage:
  python3 harness/per_kernel_tvla.py \\
    experiments/traces/per_kernel/kyber512_class0_n100000_per_kernel.csv \\
    experiments/traces/per_kernel/kyber512_class1_n100000_per_kernel.csv \\
    [--plot] [--out report.txt]

Output:
  Sorted table of |t| per kernel.  Kernels above |t| >= 4.5 are flagged
  as leakage contributors.

Author: mlkem-gpu-sec team
============================================================
"""

import sys
import os
import argparse
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    PLOT_AVAILABLE = True
except ImportError:
    PLOT_AVAILABLE = False
    print("[WARN] matplotlib not available — plots skipped")

TVLA_THRESHOLD = 4.5
Z_OUTLIER = 5.0


# ── Load CSV ──────────────────────────────────────────────────────────────
def load_per_kernel_csv(filepath):
    """
    Returns dict: kernel_name → np.ndarray of elapsed_us values.
    Skips comment lines (starting with #) and the header row.
    """
    if not os.path.exists(filepath):
        print(f"[ERROR] Not found: {filepath}")
        sys.exit(1)

    data = {}  # kernel_name -> list of floats
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('trace_id'):
                continue  # header
            parts = line.split(',')
            if len(parts) != 3:
                continue
            try:
                kernel = parts[1]
                elapsed = float(parts[2])
                if kernel not in data:
                    data[kernel] = []
                data[kernel].append(elapsed)
            except ValueError:
                continue

    return {k: np.array(v) for k, v in data.items()}


# ── Outlier removal per kernel ────────────────────────────────────────────
def remove_outliers(arr, z_threshold=Z_OUTLIER):
    if len(arr) < 2:
        return arr
    mean, std = np.mean(arr), np.std(arr)
    if std == 0:
        return arr
    mask = np.abs((arr - mean) / std) < z_threshold
    n_removed = np.sum(~mask)
    return arr[mask], n_removed


# ── Welch t-test ──────────────────────────────────────────────────────────
def welch_t(a, b):
    t, p = stats.ttest_ind(a, b, equal_var=False)
    return t, p


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='Per-kernel TVLA analysis for serialized Kyber traces'
    )
    parser.add_argument('class0_csv', help='Per-kernel CSV for class 0 (valid)')
    parser.add_argument('class1_csv', help='Per-kernel CSV for class 1 (invalid)')
    parser.add_argument('--plot', action='store_true', help='Generate bar chart')
    parser.add_argument('--out', default=None, help='Save text report to file')
    args = parser.parse_args()

    print()
    print('=' * 60)
    print('  mlkem-gpu-sec: Per-Kernel TVLA Analysis')
    print('=' * 60)

    print(f"\n[Loading] {args.class0_csv}")
    c0_data = load_per_kernel_csv(args.class0_csv)
    print(f"[Loading] {args.class1_csv}")
    c1_data = load_per_kernel_csv(args.class1_csv)

    # Kernel order as they appear in trace_serialized.cu
    canonical_order = [
        'decompress_u', 'decompress_v', 'decode_s',
        'fwdntt_u', 's_times_u', 'intt_su', 'v_minus_su', 'poly_tomsg',
        'hash_coin',
        'decode_t', 'poly_frommsg', 'generate_at',
        'generate_r', 'generate_e1', 'generate_e2',
        'fwdntt_r', 'a_times_r', 't_times_r', 'intt_ar', 'intt_tr',
        'ar_plus_e1', 'tr_plus_e2_plus_m', 'compress_u', 'compress_v',
        'hash_ct', 'verify_cmov', 'kdf',
    ]

    all_kernels = sorted(set(c0_data.keys()) | set(c1_data.keys()),
                         key=lambda k: canonical_order.index(k)
                         if k in canonical_order else 999)

    print(f"\n  Kernels found: {len(all_kernels)}")
    print(f"  Traces per kernel (class 0): "
          f"{len(next(iter(c0_data.values()))):,}" if c0_data else "  (no data)")

    # ── Per-kernel analysis ───────────────────────────────────────────────
    results = []
    total_removed = 0

    print('\n[Per-kernel results]')
    print(f"  {'Kernel':<25} {'N0':>7} {'N1':>7} "
          f"{'Mean0':>9} {'Mean1':>9} {'Diff':>9} {'|t|':>8} {'p':>10} {'Verdict'}")
    print('  ' + '-' * 100)

    for kernel in all_kernels:
        if kernel not in c0_data or kernel not in c1_data:
            print(f"  {kernel:<25}  [MISSING in one class — skipped]")
            continue

        c0_raw = c0_data[kernel]
        c1_raw = c1_data[kernel]

        c0, nr0 = remove_outliers(c0_raw)
        c1, nr1 = remove_outliers(c1_raw)
        total_removed += nr0 + nr1

        if len(c0) < 2 or len(c1) < 2:
            print(f"  {kernel:<25}  [insufficient data after outlier removal]")
            continue

        t, p = welch_t(c0, c1)
        t_abs = abs(t)
        mean_diff = np.mean(c1) - np.mean(c0)
        verdict = '*** LEAK ***' if t_abs >= TVLA_THRESHOLD else 'ok'

        print(f"  {kernel:<25} {len(c0):>7,} {len(c1):>7,} "
              f"{np.mean(c0):>9.3f} {np.mean(c1):>9.3f} "
              f"{mean_diff:>+9.3f} {t_abs:>8.2f} {p:>10.2e}  {verdict}")

        results.append({
            'kernel': kernel,
            'n0': len(c0), 'n1': len(c1),
            'mean0': np.mean(c0), 'mean1': np.mean(c1),
            'mean_diff': mean_diff,
            't': t, 't_abs': t_abs, 'p': p,
        })

    if not results:
        print("\n[ERROR] No results computed — check input files.")
        sys.exit(1)

    # ── Sorted summary ─────────────────────────────────────────────────────
    sorted_results = sorted(results, key=lambda r: r['t_abs'], reverse=True)

    print()
    print('=' * 60)
    print('  SORTED BY |t| (highest leakage first)')
    print('=' * 60)
    print(f"  {'Rank':<5} {'Kernel':<25} {'|t|':>8} {'Verdict'}")
    print('  ' + '-' * 55)
    for rank, r in enumerate(sorted_results, 1):
        flag = '*** LEAK ***' if r['t_abs'] >= TVLA_THRESHOLD else ''
        print(f"  {rank:<5} {r['kernel']:<25} {r['t_abs']:>8.2f}  {flag}")

    n_leaking = sum(1 for r in results if r['t_abs'] >= TVLA_THRESHOLD)
    print()
    print(f"  Kernels with |t| >= {TVLA_THRESHOLD}: {n_leaking} / {len(results)}")
    print(f"  Total outliers removed: {total_removed:,}")
    print('=' * 60)

    # ── Plot ──────────────────────────────────────────────────────────────
    if args.plot and PLOT_AVAILABLE:
        kernels  = [r['kernel'] for r in sorted_results]
        t_values = [r['t_abs']  for r in sorted_results]
        colors   = ['tomato' if t >= TVLA_THRESHOLD else 'steelblue'
                    for t in t_values]

        fig, ax = plt.subplots(figsize=(max(10, len(kernels) * 0.55), 6))
        bars = ax.bar(range(len(kernels)), t_values, color=colors)
        ax.axhline(TVLA_THRESHOLD, color='red', linestyle='--', linewidth=1.5,
                   label=f'TVLA threshold = {TVLA_THRESHOLD}')
        ax.set_xticks(range(len(kernels)))
        ax.set_xticklabels(kernels, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('|t-statistic|')
        ax.set_title('Per-Kernel TVLA |t| — Sorted by Magnitude\n'
                     '(red = leakage, blue = clean)')
        ax.legend()
        plt.tight_layout()

        out_dir = os.path.join(os.path.dirname(os.path.abspath(args.class0_csv)),
                               '..', 'results')
        os.makedirs(out_dir, exist_ok=True)
        plot_path = os.path.join(out_dir, 'per_kernel_tvla.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n  Plot saved: {plot_path}")

    # ── Text report ───────────────────────────────────────────────────────
    report_lines = [
        'mlkem-gpu-sec Per-Kernel TVLA Report',
        '=' * 60,
        f"Class 0 CSV: {args.class0_csv}",
        f"Class 1 CSV: {args.class1_csv}",
        f"Kernels analysed: {len(results)}",
        f"Kernels leaking (|t|>={TVLA_THRESHOLD}): {n_leaking}",
        '',
        f"{'Rank':<5} {'Kernel':<25} {'|t|':>8} {'Mean0':>9} {'Mean1':>9} "
        f"{'Diff':>9} {'p':>10}",
        '-' * 80,
    ]
    for rank, r in enumerate(sorted_results, 1):
        flag = ' *** LEAK' if r['t_abs'] >= TVLA_THRESHOLD else ''
        report_lines.append(
            f"{rank:<5} {r['kernel']:<25} {r['t_abs']:>8.2f} "
            f"{r['mean0']:>9.3f} {r['mean1']:>9.3f} {r['mean_diff']:>+9.3f} "
            f"{r['p']:>10.2e}{flag}"
        )

    if args.out:
        with open(args.out, 'w') as f:
            f.write('\n'.join(report_lines) + '\n')
        print(f"  Report saved: {args.out}")
    else:
        auto = args.class0_csv.replace('class0', 'per_kernel_tvla_report').replace('.csv', '.txt')
        with open(auto, 'w') as f:
            f.write('\n'.join(report_lines) + '\n')
        print(f"  Report auto-saved: {auto}")

    print()
    return 0


if __name__ == '__main__':
    sys.exit(main())
