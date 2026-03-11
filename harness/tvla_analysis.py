#!/usr/bin/env python3
"""
harness/tvla_analysis.py
============================================================
Phase 2: TVLA (Test Vector Leakage Assessment) Analysis

Loads timing traces from collect_traces and runs Welch's
t-test to determine whether GPU Kyber decapsulation leaks
timing information.

TVLA Decision Rule:
  |t| < 4.5  →  No leakage detected (pass)
  |t| >= 4.5 →  Leakage detected (fail)

Usage:
  python3 harness/tvla_analysis.py <class0_csv> <class1_csv> [--plot] [--out report.txt]

  python3 harness/tvla_analysis.py \\
    experiments/traces/kyber512_class0_n100000.csv \\
    experiments/traces/kyber512_class1_n100000.csv \\
    --plot --out experiments/results/tvla_kyber512.txt

Author: Person 2
Project: mlkem-gpu-sec
============================================================
"""

import sys
import os
import argparse
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Optional matplotlib — gracefully skip if not available
try:
    import matplotlib
    matplotlib.use('Agg')  # non-interactive backend for RunPod
    import matplotlib.pyplot as plt
    PLOT_AVAILABLE = True
except ImportError:
    PLOT_AVAILABLE = False
    print("[WARN] matplotlib not available — plots will be skipped")


# ── TVLA threshold ────────────────────────────────────────
TVLA_THRESHOLD = 4.5


# ── Load traces from CSV ──────────────────────────────────
def load_traces(filepath):
    """Load timing traces from CSV file produced by collect_traces."""
    if not os.path.exists(filepath):
        print(f"[ERROR] File not found: {filepath}")
        sys.exit(1)

    timings = []
    metadata = {}

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('#'):
                # Parse metadata
                if ':' in line:
                    key, val = line[1:].split(':', 1)
                    metadata[key.strip()] = val.strip()
                continue
            if line == 'timing_us':
                continue
            try:
                timings.append(float(line))
            except ValueError:
                continue

    return np.array(timings), metadata


# ── Remove outliers ───────────────────────────────────────
def remove_outliers(traces, z_threshold=5.0):
    """Remove extreme outliers using z-score threshold."""
    mean = np.mean(traces)
    std  = np.std(traces)
    if std == 0:
        return traces
    z_scores = np.abs((traces - mean) / std)
    mask = z_scores < z_threshold
    n_removed = np.sum(~mask)
    if n_removed > 0:
        print(f"  Removed {n_removed} outliers (z > {z_threshold})")
    return traces[mask]


# ── Welch's t-test ────────────────────────────────────────
def welch_t_test(class0, class1):
    """
    Welch's t-test for unequal variances.
    Returns t-statistic and p-value.
    """
    t_stat, p_value = stats.ttest_ind(class0, class1, equal_var=False)
    return t_stat, p_value


# ── Sliding window t-test ─────────────────────────────────
def sliding_window_tvla(class0, class1, window_size=1000, step=100):
    """
    Compute t-statistic over sliding windows.
    Reveals whether leakage is consistent or intermittent.
    """
    n = min(len(class0), len(class1))
    t_values = []
    positions = []

    for start in range(0, n - window_size, step):
        end = start + window_size
        t, _ = welch_t_test(class0[start:end], class1[start:end])
        t_values.append(abs(t))
        positions.append(start + window_size // 2)

    return np.array(positions), np.array(t_values)


# ── Print statistics ──────────────────────────────────────
def print_stats(name, traces):
    print(f"  {name}:")
    print(f"    N:      {len(traces):,}")
    print(f"    Mean:   {np.mean(traces):.4f} us")
    print(f"    StdDev: {np.std(traces):.4f} us")
    print(f"    Median: {np.median(traces):.4f} us")
    print(f"    Min:    {np.min(traces):.4f} us")
    print(f"    Max:    {np.max(traces):.4f} us")


# ── Plot results ──────────────────────────────────────────
def plot_results(class0, class1, t_stat, positions, t_sliding,
                 variant, n_traces, output_dir):
    if not PLOT_AVAILABLE:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'TVLA Analysis — Kyber-{variant} Decapsulation\n'
                 f'n={n_traces:,} traces per class | |t| = {abs(t_stat):.2f} '
                 f'(threshold = {TVLA_THRESHOLD})',
                 fontsize=13, fontweight='bold')

    # ── Plot 1: Timing distributions ──────────────────────
    ax = axes[0, 0]
    bins = np.linspace(
        min(np.percentile(class0, 1), np.percentile(class1, 1)),
        max(np.percentile(class0, 99), np.percentile(class1, 99)),
        100
    )
    ax.hist(class0, bins=bins, alpha=0.6, color='steelblue',
            label='Class 0 (valid)', density=True)
    ax.hist(class1, bins=bins, alpha=0.6, color='tomato',
            label='Class 1 (invalid)', density=True)
    ax.set_xlabel('Decapsulation Time (μs)')
    ax.set_ylabel('Density')
    ax.set_title('Timing Distributions')
    ax.legend()

    # ── Plot 2: Sliding window t-statistic ────────────────
    ax = axes[0, 1]
    ax.plot(positions, t_sliding, color='purple', linewidth=0.8)
    ax.axhline(y=TVLA_THRESHOLD, color='red', linestyle='--',
               linewidth=1.5, label=f'Threshold = {TVLA_THRESHOLD}')
    ax.fill_between(positions, t_sliding, TVLA_THRESHOLD,
                    where=(t_sliding >= TVLA_THRESHOLD),
                    alpha=0.3, color='red', label='Leakage region')
    ax.set_xlabel('Trace Index')
    ax.set_ylabel('|t-statistic|')
    ax.set_title('Sliding Window TVLA')
    ax.legend()

    # ── Plot 3: Boxplot comparison ────────────────────────
    ax = axes[1, 0]
    bp = ax.boxplot([class0, class1],
                    labels=['Class 0\n(valid)', 'Class 1\n(invalid)'],
                    patch_artist=True,
                    showfliers=False)
    bp['boxes'][0].set_facecolor('steelblue')
    bp['boxes'][1].set_facecolor('tomato')
    for patch in bp['boxes']:
        patch.set_alpha(0.7)
    ax.set_ylabel('Decapsulation Time (μs)')
    ax.set_title('Timing Comparison (no outliers)')

    # ── Plot 4: TVLA verdict ──────────────────────────────
    ax = axes[1, 1]
    ax.axis('off')

    t_abs = abs(t_stat)
    verdict = "LEAKAGE DETECTED" if t_abs >= TVLA_THRESHOLD else "NO LEAKAGE DETECTED"
    color   = "red" if t_abs >= TVLA_THRESHOLD else "green"

    ax.text(0.5, 0.70, verdict, transform=ax.transAxes,
            fontsize=16, fontweight='bold', color=color,
            ha='center', va='center')
    ax.text(0.5, 0.55, f'|t-statistic| = {t_abs:.4f}',
            transform=ax.transAxes, fontsize=13,
            ha='center', va='center')
    ax.text(0.5, 0.42, f'Threshold = {TVLA_THRESHOLD}',
            transform=ax.transAxes, fontsize=11, color='gray',
            ha='center', va='center')
    ax.text(0.5, 0.28,
            f'Mean diff = {np.mean(class1) - np.mean(class0):.4f} μs',
            transform=ax.transAxes, fontsize=11,
            ha='center', va='center')
    ax.text(0.5, 0.15,
            f'n = {len(class0):,} + {len(class1):,} traces',
            transform=ax.transAxes, fontsize=10, color='gray',
            ha='center', va='center')

    plt.tight_layout()

    plot_path = os.path.join(output_dir, f'tvla_kyber{variant}_n{n_traces}.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved: {plot_path}")


# ── Main ──────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='TVLA analysis for GPU Kyber timing traces'
    )
    parser.add_argument('class0_csv', help='CSV file for class 0 (valid ciphertexts)')
    parser.add_argument('class1_csv', help='CSV file for class 1 (invalid ciphertexts)')
    parser.add_argument('--plot', action='store_true', help='Generate plots')
    parser.add_argument('--out', default=None, help='Save report to file')
    parser.add_argument('--no-outlier-removal', action='store_true',
                        help='Skip outlier removal')
    args = parser.parse_args()

    print()
    print("=" * 52)
    print("  mlkem-gpu-sec Phase 2: TVLA Analysis")
    print("=" * 52)

    # Load traces
    print("\n[Loading traces]")
    class0_raw, meta0 = load_traces(args.class0_csv)
    class1_raw, meta1 = load_traces(args.class1_csv)

    print(f"  Class 0: {len(class0_raw):,} traces loaded from {args.class0_csv}")
    print(f"  Class 1: {len(class1_raw):,} traces loaded from {args.class1_csv}")

    # Extract metadata
    variant  = meta0.get('variant', 'Kyber-???').replace('Kyber-', '')
    n_traces = len(class0_raw)

    # Remove outliers
    print("\n[Preprocessing]")
    if args.no_outlier_removal:
        class0 = class0_raw
        class1 = class1_raw
        print("  Outlier removal: skipped")
    else:
        class0 = remove_outliers(class0_raw)
        class1 = remove_outliers(class1_raw)

    # Print statistics
    print("\n[Statistics]")
    print_stats("Class 0 (valid ciphertexts)", class0)
    print()
    print_stats("Class 1 (invalid ciphertexts)", class1)

    mean_diff = np.mean(class1) - np.mean(class0)
    print(f"\n  Mean difference: {mean_diff:.4f} us")

    # Run TVLA
    print("\n[TVLA — Welch's t-test]")
    t_stat, p_value = welch_t_test(class0, class1)
    t_abs = abs(t_stat)

    print(f"  t-statistic: {t_stat:.6f}")
    print(f"  |t-statistic|: {t_abs:.6f}")
    print(f"  p-value: {p_value:.2e}")
    print(f"  Threshold: {TVLA_THRESHOLD}")

    # Verdict
    print()
    print("=" * 52)
    if t_abs >= TVLA_THRESHOLD:
        print(f"  VERDICT: *** LEAKAGE DETECTED ***")
        print(f"  |t| = {t_abs:.4f} >= {TVLA_THRESHOLD}")
        print(f"  Timing difference is statistically significant.")
        print(f"  Decapsulation time depends on ciphertext class.")
    else:
        print(f"  VERDICT: No leakage detected")
        print(f"  |t| = {t_abs:.4f} < {TVLA_THRESHOLD}")
        print(f"  No statistically significant timing difference.")
    print("=" * 52)

    # Sliding window analysis
    print("\n[Sliding Window TVLA]")
    positions, t_sliding = sliding_window_tvla(class0, class1)
    max_t = np.max(t_sliding) if len(t_sliding) > 0 else 0
    frac_above = np.mean(t_sliding >= TVLA_THRESHOLD) if len(t_sliding) > 0 else 0
    print(f"  Max |t| in windows: {max_t:.4f}")
    print(f"  Windows above threshold: {frac_above*100:.1f}%")

    # Generate plots
    if args.plot:
        print("\n[Generating plots]")
        output_dir = os.path.join(
            os.path.dirname(os.path.abspath(args.class0_csv)),
            '..', 'results'
        )
        os.makedirs(output_dir, exist_ok=True)
        plot_results(class0, class1, t_stat, positions, t_sliding,
                     variant, n_traces, output_dir)

    # Save report
    report_lines = [
        "mlkem-gpu-sec TVLA Report",
        "=" * 52,
        f"Variant:       Kyber-{variant}",
        f"Traces:        {len(class0):,} + {len(class1):,}",
        f"t-statistic:   {t_stat:.6f}",
        f"|t-statistic|: {t_abs:.6f}",
        f"p-value:       {p_value:.2e}",
        f"Threshold:     {TVLA_THRESHOLD}",
        f"Mean diff:     {mean_diff:.4f} us",
        f"Verdict:       {'LEAKAGE DETECTED' if t_abs >= TVLA_THRESHOLD else 'No leakage detected'}",
        f"Sliding max:   {max_t:.4f}",
        f"Windows >thr:  {frac_above*100:.1f}%",
    ]

    if args.out:
        with open(args.out, 'w') as f:
            f.write('\n'.join(report_lines) + '\n')
        print(f"\n  Report saved: {args.out}")
    else:
        # Auto-save alongside traces
        auto_out = args.class0_csv.replace('class0', 'tvla_report').replace('.csv', '.txt')
        with open(auto_out, 'w') as f:
            f.write('\n'.join(report_lines) + '\n')
        print(f"\n  Report auto-saved: {auto_out}")

    print()
    return 0 if t_abs < TVLA_THRESHOLD else 1


if __name__ == '__main__':
    sys.exit(main())