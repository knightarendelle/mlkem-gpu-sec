#!/usr/bin/env python3
"""
harness/ncu_cache_profile.py

Differential cache profiling via Nsight Compute for Kyber-1024 decapsulation.
Runs trace_ser_kyber1024.out twice — class 0 (valid ct) and class 1 (random ct) —
and compares per-kernel L1/L2/DRAM hardware metrics to test the DCC hypothesis.

Hypothesis under test:
  Valid Kyber-1024 ciphertexts contain compressed polynomial coefficients (values
  mod 3329 < 2^12), which have more zero high-bits than uniform random bytes.
  Ada Lovelace's DCC (Data Compression Cache) compresses DRAM traffic for such
  data, reducing latency.  If DCC drives the timing leakage, the leaking kernels
  (generate_at, hash_ct, hash_coin, generate_e2, generate_r, generate_e1, v_minus_su)
  should show lower DRAM traffic or higher L2 hit rates for class 0 (valid ct).

Usage:
  python3 harness/ncu_cache_profile.py [options]

Options:
  --binary   PATH  path to trace_ser_kyber1024.out
                   (default: baseline/atpqc-cuda/target/trace_ser_kyber1024.out)
  --ntraces  N     decapsulations per class profiled (default: 1000)
  --outdir   DIR   directory for NCU CSV output and text report
                   (default: experiments/results)
  --skip-run       skip ncu runs; parse existing CSVs in --outdir
  --sudo           prefix ncu command with sudo

Prerequisites:
  - ncu (Nsight Compute CLI) on PATH
  - Profiling permissions: sudo, or /proc/driver/nvidia/params with
    RmProfilingAdminOnly=0, or running as root in container
  - Build: make -C baseline/atpqc-cuda trace_ser_kyber1024
"""

import argparse
import csv
import io
import os
import subprocess
import sys
from collections import defaultdict

# ── Metrics to collect ────────────────────────────────────────────────────────
METRICS = [
    "lts__t_sectors_op_read.sum",
    "lts__t_sectors_op_write.sum",
    "lts__t_sectors_lookup_hit.sum",
    "lts__t_sectors_lookup_miss.sum",
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum",
]

METRIC_SHORT = {
    "lts__t_sectors_op_read.sum":                      "L2_rd_sec",
    "lts__t_sectors_op_write.sum":                     "L2_wr_sec",
    "lts__t_sectors_lookup_hit.sum":                   "L2_hit",
    "lts__t_sectors_lookup_miss.sum":                  "L2_miss",
    "dram__bytes_read.sum":                            "DRAM_rd_B",
    "dram__bytes_write.sum":                           "DRAM_wr_B",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum":  "L1_ld_sec",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum":  "L1_st_sec",
}

# Kernels that showed |t| >= 4.5 in per-kernel TVLA (phase 6).
LEAKING_KERNELS = {
    "generate_at", "hash_ct", "hash_coin",
    "generate_e2", "generate_r/e1", "v_minus_su",
}


# ── CUDA name → logical name mapping ─────────────────────────────────────────
def cuda_name_to_logical(cuda_name: str) -> str:
    """
    Map a (possibly demangled) CUDA kernel name to a human-readable logical name.

    Many logical kernels share the same CUDA kernel template instantiation
    (e.g. decode_s and decode_t both use polyvec_frombytes<4>).  Those are merged
    into a combined label.  For the DCC analysis this is acceptable: the user cares
    about which *kernel type* is affected, not which pipeline slot.

    Note: setup-graph kernels (keypair hash, enc re-encryption) share CUDA names
    with their dec-path counterparts.  They are ~2% of total invocations and are
    identical between class 0 and class 1, so they do not bias the comparison.
    """
    n = cuda_name.lower()

    # ── SHA-3 / SHAKE ────────────────────────────────────────────────────────
    if "sha3" in n:
        return "hash_coin" if "512" in cuda_name else "hash_ct"
    if "shake" in n:
        return "kdf"

    # ── genmatrix / gennoise ─────────────────────────────────────────────────
    if "genmatrix" in n:
        return "generate_at" if ("true" in n or ", true" in cuda_name or ",true" in cuda_name) else "generate_a(keypair)"
    if "gennoise" in n:
        # Kyber-1024: k=4, eta1=2, eta2=2.
        # gennoise<1,2> → generate_e2;  gennoise<4,2> → generate_r/e1 (also s/e in keypair).
        if "<1" in cuda_name or ", 1," in cuda_name or "<1u" in cuda_name or "<1," in cuda_name:
            return "generate_e2"
        return "generate_r/e1"

    # ── encode / decode ───────────────────────────────────────────────────────
    if "polyvec_decompress" in n:
        return "decompress_u"
    if "poly_decompress" in n:
        return "decompress_v"
    if "polyvec_frombytes" in n or ("polyvec" in n and "frombytes" in n):
        return "decode_s/t"
    if "poly_frombytes" in n:
        return "decode_s/t"
    if "polyvec_tobytes" in n or ("polyvec" in n and "tobytes" in n):
        return "encode(keypair)"
    if "poly_tomsg" in n:
        return "poly_tomsg"
    if "poly_frommsg" in n:
        return "poly_frommsg"
    if "polyvec_compress" in n:
        return "compress_u"
    if "poly_compress" in n:
        return "compress_v"

    # ── NTT ───────────────────────────────────────────────────────────────────
    if "fwdntt" in n:
        return "fwdntt_u/r"
    if "invntt" in n:
        # invntt_tomont<1> → intt_su/tr;  invntt_tomont<4> → intt_ar
        if "<1" in cuda_name or "<1u" in cuda_name:
            return "intt_su/tr"
        return "intt_ar"

    # ── arithmetic ────────────────────────────────────────────────────────────
    if "mattimesvec_tomont" in n:
        return "mtvpv(keypair)"
    if "mattimesvec" in n:
        return "a_times_r"
    if "vectimesvec" in n:
        return "s/t_times_u/r"
    if "vecadd2" in n:
        return "ar_plus_e1"
    if "polyadd3" in n:
        return "tr_plus_e2_plus_m"
    if "polysub" in n:
        return "v_minus_su"

    # ── verify / cmov ─────────────────────────────────────────────────────────
    if "verify" in n or "cmov" in n:
        return "verify_cmov"

    return cuda_name  # fallback


# ── Run ncu ───────────────────────────────────────────────────────────────────
def run_ncu(binary: str, ct_class: int, ntraces: int,
            csv_out: str, use_sudo: bool) -> None:
    """Run ncu on binary with given ct_class; write CSV output to csv_out."""
    # trace_serialized stdin: ninputs genmat_nw genvec_nw genpoly_nw fips_nw ntraces ct_class
    stdin_str = f"1 4 4 4 4 {ntraces} {ct_class}\n"

    ncu_bin = ["sudo", "ncu"] if use_sudo else ["ncu"]
    cmd = ncu_bin + [
        "--metrics", ",".join(METRICS),
        "--csv",
        "--target-processes", "all",
        "--kernel-name-base", "demangled",
        binary,
    ]

    print(f"\n[ncu class {ct_class}] {' '.join(cmd)}", flush=True)
    print(f"[ncu class {ct_class}] stdin: {stdin_str.strip()}", flush=True)
    print(f"[ncu class {ct_class}] output → {csv_out}", flush=True)
    print(f"[ncu class {ct_class}] (this will take several minutes...)", flush=True)

    with open(csv_out, "w") as fout:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=fout,
            stderr=sys.stderr,
            text=True,
        )
        proc.communicate(input=stdin_str)

    if proc.returncode != 0:
        print(f"[ERROR] ncu exited with code {proc.returncode}", file=sys.stderr)
        sys.exit(1)

    print(f"[ncu class {ct_class}] done.", flush=True)


# ── Parse ncu CSV ─────────────────────────────────────────────────────────────
def parse_ncu_csv(filepath: str) -> dict:
    """
    Parse NCU `--csv` output.
    Returns: logical_name → metric_name → list[float]

    NCU CSV format:
      "ID","Process ID","Process Name","Host Name","Kernel Name","Kernel Time",
      "Context","Stream","Section Name","Metric Name","Metric Unit","Metric Value"

    Each row is one (kernel invocation, metric) pair.
    """
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        raw = f.read()

    # NCU may prefix output with a copyright banner; skip to the CSV header.
    lines = raw.splitlines()
    header_idx = None
    for i, line in enumerate(lines):
        stripped = line.strip().strip('"')
        if stripped.startswith("ID") or '"ID"' in line:
            header_idx = i
            break

    if header_idx is None:
        print(f"[ERROR] No CSV header found in {filepath}", file=sys.stderr)
        print(f"  First 10 lines:", file=sys.stderr)
        for l in lines[:10]:
            print(f"    {l!r}", file=sys.stderr)
        sys.exit(1)

    csv_text = "\n".join(lines[header_idx:])
    reader = csv.DictReader(io.StringIO(csv_text))

    data = defaultdict(lambda: defaultdict(list))
    unknown_metrics = set()
    total_rows = 0

    for row in reader:
        total_rows += 1
        cuda_name = row.get("Kernel Name", "").strip()
        if not cuda_name:
            continue

        metric = row.get("Metric Name", "").strip()
        if metric not in METRICS:
            if metric:
                unknown_metrics.add(metric)
            continue

        val_str = row.get("Metric Value", "").strip()
        # NCU sometimes formats large numbers with commas: "1,234,567"
        # After csv.DictReader strips quotes, we just remove remaining commas.
        val_str = val_str.replace(",", "")
        try:
            val = float(val_str)
        except ValueError:
            continue

        logical = cuda_name_to_logical(cuda_name)
        data[logical][metric].append(val)

    print(f"  Parsed {total_rows} rows from {filepath}")
    print(f"  Distinct logical kernels: {len(data)}")
    if unknown_metrics:
        print(f"  [warn] unrecognised metric names (ignored): {sorted(unknown_metrics)[:5]}")

    return data


# ── Build and print comparison table ─────────────────────────────────────────
DISPLAY_ORDER = [
    # These appear in the per-kernel TVLA leaking set:
    "generate_at",
    "hash_ct",
    "hash_coin",
    "generate_e2",
    "generate_r/e1",
    "v_minus_su",
    # Clean kernels for reference:
    "decompress_u",
    "decompress_v",
    "decode_s/t",
    "fwdntt_u/r",
    "s/t_times_u/r",
    "intt_su/tr",
    "intt_ar",
    "poly_tomsg",
    "poly_frommsg",
    "a_times_r",
    "ar_plus_e1",
    "tr_plus_e2_plus_m",
    "compress_u",
    "compress_v",
    "verify_cmov",
    "kdf",
]


def format_table(data0: dict, data1: dict) -> str:
    lines = []
    sep = "=" * 118

    lines.append(sep)
    lines.append("  NCU Cache-Metric Differential: Kyber-1024  class 0 (valid ct) vs class 1 (random ct)")
    lines.append("  Kernels marked <<LEAK>> showed |t| >= 4.5 in per-kernel TVLA.")
    lines.append("  DCC hypothesis: valid ct bytes are more compressible → lower DRAM traffic for class 0.")
    lines.append(sep)

    hdr = (f"  {'Kernel':<22}  {'Metric':<12}  "
           f"{'Class0_mean':>14}  {'Class1_mean':>14}  "
           f"{'%diff (C1-C0)':>14}  {'N0':>6}  {'N1':>6}  {'Flag'}")
    lines.append(hdr)
    lines.append("  " + "-" * 114)

    # Collect all logical names seen in either class, display in preferred order.
    all_names = set(data0.keys()) | set(data1.keys())
    ordered = [n for n in DISPLAY_ORDER if n in all_names]
    ordered += sorted(all_names - set(ordered))

    for logical in ordered:
        d0 = data0.get(logical, {})
        d1 = data1.get(logical, {})
        flag = "<<LEAK>>" if logical in LEAKING_KERNELS else ""
        first = True
        for metric in METRICS:
            vals0 = d0.get(metric, [])
            vals1 = d1.get(metric, [])
            if not vals0 and not vals1:
                continue
            mean0 = sum(vals0) / len(vals0) if vals0 else float("nan")
            mean1 = sum(vals1) / len(vals1) if vals1 else float("nan")
            n0 = len(vals0)
            n1 = len(vals1)
            if mean0 != mean0 or mean1 != mean1:  # nan check
                pct_str = "   n/a"
            elif mean0 == 0:
                pct_str = "   inf"
            else:
                pct = (mean1 - mean0) / mean0 * 100.0
                pct_str = f"{pct:+.2f}%"

            k_col  = logical if first else ""
            f_col  = flag    if first else ""
            short  = METRIC_SHORT.get(metric, metric[:12])
            m0_str = f"{mean0:,.0f}" if mean0 == mean0 else "n/a"
            m1_str = f"{mean1:,.0f}" if mean1 == mean1 else "n/a"
            lines.append(
                f"  {k_col:<22}  {short:<12}  "
                f"{m0_str:>14}  {m1_str:>14}  "
                f"{pct_str:>14}  {n0:>6}  {n1:>6}  {f_col}"
            )
            first = False
        if not first:
            lines.append("")

    lines.append(sep)
    lines.append("")

    # Summary: % DRAM difference for leaking vs clean kernels.
    dram_metric = "dram__bytes_read.sum"
    leak_diffs, clean_diffs = [], []
    for logical in ordered:
        d0 = data0.get(logical, {})
        d1 = data1.get(logical, {})
        vals0 = d0.get(dram_metric, [])
        vals1 = d1.get(dram_metric, [])
        if not vals0 or not vals1:
            continue
        m0 = sum(vals0) / len(vals0)
        m1 = sum(vals1) / len(vals1)
        if m0 == 0:
            continue
        pct = (m1 - m0) / m0 * 100.0
        if logical in LEAKING_KERNELS:
            leak_diffs.append((logical, pct))
        else:
            clean_diffs.append((logical, pct))

    lines.append("  DRAM read bytes % difference summary (class1 vs class0, positive = more DRAM for random ct):")
    lines.append(f"  {'Kernel':<25}  {'DRAM_rd %diff':>14}  Group")
    lines.append("  " + "-" * 55)
    for name, pct in sorted(leak_diffs + clean_diffs, key=lambda x: -abs(x[1])):
        grp = "LEAK" if name in LEAKING_KERNELS else "clean"
        lines.append(f"  {name:<25}  {pct:>+13.2f}%  {grp}")
    lines.append(sep)

    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> int:
    parser = argparse.ArgumentParser(
        description="NCU cache-metric differential profiling for Kyber-1024 decaps",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--binary",
        default="baseline/atpqc-cuda/target/trace_ser_kyber1024.out",
        help="Path to trace_ser_kyber1024.out",
    )
    parser.add_argument(
        "--ntraces", type=int, default=1000,
        help="Decapsulations per class for NCU profiling (default: 1000)",
    )
    parser.add_argument(
        "--outdir", default="experiments/results",
        help="Output directory for NCU CSV files and text report",
    )
    parser.add_argument(
        "--skip-run", action="store_true",
        help="Skip running ncu; parse existing CSVs from --outdir",
    )
    parser.add_argument(
        "--sudo", action="store_true",
        help="Prefix ncu with sudo (needed if profiling permissions not set globally)",
    )
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    csv0 = os.path.join(args.outdir, "ncu_class0_raw.csv")
    csv1 = os.path.join(args.outdir, "ncu_class1_raw.csv")
    report_path = os.path.join(args.outdir, "dcc_ncu.txt")

    if not args.skip_run:
        if not os.path.isfile(args.binary):
            print(f"[ERROR] Binary not found: {args.binary}", file=sys.stderr)
            print("  Build with: make -C baseline/atpqc-cuda trace_ser_kyber1024", file=sys.stderr)
            return 1

        run_ncu(args.binary, 0, args.ntraces, csv0, args.sudo)
        run_ncu(args.binary, 1, args.ntraces, csv1, args.sudo)
    else:
        for p in [csv0, csv1]:
            if not os.path.isfile(p):
                print(f"[ERROR] --skip-run set but CSV not found: {p}", file=sys.stderr)
                return 1
        print(f"[skip-run] parsing existing: {csv0}, {csv1}")

    print("\n[parse] class 0 CSV...", flush=True)
    data0 = parse_ncu_csv(csv0)
    print("[parse] class 1 CSV...", flush=True)
    data1 = parse_ncu_csv(csv1)

    table = format_table(data0, data1)
    print("\n" + table)

    with open(report_path, "w") as f:
        f.write(table)
    print(f"[report] saved → {report_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
