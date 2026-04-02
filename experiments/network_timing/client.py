#!/usr/bin/env python3
"""
experiments/network_timing/client.py
=====================================
Network timing attack client.

Connects to the server on 127.0.0.1:9999, sends 200,000 ciphertexts
(100,000 class-0 valid, 100,000 class-1 random) in alternating batches
of 500, measures round-trip time for each, and saves the results.

Class 0 (valid):  the 768-byte ciphertext written by the server on startup
                  to /tmp/mlkem_valid_ct.bin.
Class 1 (random): 768 bytes generated with the same LCG used in
                  trace_main.cu (deterministic, seed-free).

Output: experiments/network_timing/logs/network_timing.csv
  Header: timestamp_us,class,rtt_us
  Rows:   <us since client start>,<0 or 1>,<round-trip time in us>

Usage:
  python3 experiments/network_timing/client.py [--host 127.0.0.1] [--port 9999]
"""

import argparse
import os
import socket
import sys
import time

# ── Constants ─────────────────────────────────────────────
CT_SIZE            = 768          # Kyber-512 ciphertext bytes
N_PER_CLASS        = 2_000_000    # measurements per class
BATCH_SIZE         = 500          # how many of each class before switching
VALID_CT_PATH      = "/tmp/mlkem_valid_ct.bin"
DEFAULT_HOST       = "127.0.0.1"
DEFAULT_PORT       = 9999
PROGRESS_INTERVAL  = 10_000      # print a progress line every N requests


def make_random_ct() -> bytes:
    """
    768 bytes from the same LCG used in trace_main.cu / victim_loop.cu:
        byte[i] = (i * 6364136223846793005 + 1442695040888963407) >> 56  (mod 2^64)
    This is deterministic and requires no seed.
    """
    MASK64 = (1 << 64) - 1
    MUL    = 6364136223846793005
    ADD    = 1442695040888963407
    return bytes(
        ((i * MUL + ADD) & MASK64) >> 56
        for i in range(CT_SIZE)
    )


def recv_exact(sock: socket.socket, n: int) -> bytes:
    """Receive exactly n bytes from sock, blocking until available."""
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Server closed connection unexpectedly")
        buf += chunk
    return buf


def run(host: str, port: int, out_path: str) -> None:
    # ── Load / verify valid ciphertext ─────────────────────
    if not os.path.exists(VALID_CT_PATH):
        print(f"[ERROR] Valid ciphertext file not found: {VALID_CT_PATH}")
        print("        Start the server first — it writes this file on startup.")
        sys.exit(1)

    valid_ct = open(VALID_CT_PATH, "rb").read()
    if len(valid_ct) != CT_SIZE:
        print(f"[ERROR] {VALID_CT_PATH} is {len(valid_ct)} bytes, expected {CT_SIZE}")
        sys.exit(1)

    random_ct = make_random_ct()
    print(f"[client] Valid CT loaded from {VALID_CT_PATH} ({len(valid_ct)} bytes)")
    print(f"[client] Random CT generated ({len(random_ct)} bytes)")
    print(f"[client] Total measurements: {N_PER_CLASS * 2:,} "
          f"({N_PER_CLASS:,} per class, batch size {BATCH_SIZE})")

    # ── Build measurement sequence ──────────────────────────
    # Alternate: BATCH_SIZE of class 0, BATCH_SIZE of class 1, repeat.
    total = N_PER_CLASS * 2
    sequence = []
    n0, n1 = 0, 0
    while n0 < N_PER_CLASS or n1 < N_PER_CLASS:
        # Class 0 batch
        take = min(BATCH_SIZE, N_PER_CLASS - n0)
        sequence.extend([0] * take)
        n0 += take
        # Class 1 batch
        take = min(BATCH_SIZE, N_PER_CLASS - n1)
        sequence.extend([1] * take)
        n1 += take

    assert len(sequence) == total, f"Sequence length mismatch: {len(sequence)} != {total}"

    # ── Connect to server ──────────────────────────────────
    print(f"[client] Connecting to {host}:{port} ...")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((host, port))
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    print(f"[client] Connected")

    # ── Measurement loop ───────────────────────────────────
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    t_start_abs = time.perf_counter_ns()

    with open(out_path, "w") as log:
        log.write("timestamp_us,class,rtt_us\n")

        for i, cls in enumerate(sequence):
            ct = valid_ct if cls == 0 else random_ct

            t0 = time.perf_counter_ns()
            sock.sendall(ct)
            recv_exact(sock, 1)          # wait for server ACK
            t1 = time.perf_counter_ns()

            timestamp_us = (t0 - t_start_abs) / 1_000.0
            rtt_us       = (t1 - t0) / 1_000.0

            log.write(f"{timestamp_us:.3f},{cls},{rtt_us:.3f}\n")

            if (i + 1) % PROGRESS_INTERVAL == 0:
                pct = (i + 1) * 100 // total
                print(f"  {i + 1:>7,} / {total:,}  ({pct}%)", flush=True)

    sock.close()

    elapsed_s = (time.perf_counter_ns() - t_start_abs) / 1e9
    print(f"[client] Done — {total:,} measurements in {elapsed_s:.1f}s")
    print(f"[client] Log: {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Network timing attack client for ML-KEM GPU decapsulation"
    )
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--out", default=None,
        help="Output CSV path (default: experiments/network_timing/logs/network_timing.csv)")
    args = parser.parse_args()

    if args.out is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        args.out = os.path.join(script_dir, "logs", "network_timing.csv")

    run(args.host, args.port, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
