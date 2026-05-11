"""Wasted-CTA throughput benchmark for the extend-attention grid fix.

Demonstrates that the compact WCA grid (vLLM-style upper-bound) delivers
higher kernel throughput than the legacy rectangular data-centric grid
(B * heads * ceil(max_ext / BLOCK_M)) under GPU-saturating heterogeneous
loads.

The benchmark toggles SGLANG_GLUON_LEGACY_GRID at import time to switch
between legacy and fixed grid modes, then measures kernel-level GPU time
for each case.  It reports:

  - Kernel wall time (us)
  - Useful tile throughput (tiles/us): sum(ceil(ext_i/BM)) * H_q / time
  - Grid tile counts for both legacy and fixed formulas
  - Speedup ratio (legacy_time / fixed_time)

Because the toggle forces the dispatch path at module-import time (the env
var is read once), the script spawns itself as a subprocess for each mode
to get a clean import with the desired setting.

Usage:
    python bench_wasted_cta_throughput.py                  # full run
    python bench_wasted_cta_throughput.py --cases uniform   # uniform only
    python bench_wasted_cta_throughput.py --cases highvar   # high-var only
    python bench_wasted_cta_throughput.py --iters 100       # fewer iters

Internal (used by the subprocess driver):
    python bench_wasted_cta_throughput.py --_mode legacy --_cases_json '...'
    python bench_wasted_cta_throughput.py --_mode fixed  --_cases_json '...'
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import textwrap
from typing import Any


# ── Case definitions ──────────────────────────────────────────────────────

def _make_cases() -> dict[str, list[dict[str, Any]]]:
    """Return all benchmark cases grouped by category."""
    uniform = [
        dict(label="uniform_B64_ext256", ext_lens=[256]*64, pfx=0, H_q=32, H_kv=8, D=128),
        dict(label="uniform_B64_ext64",  ext_lens=[64]*64,  pfx=0, H_q=32, H_kv=8, D=128),
        dict(label="uniform_B32_ext512", ext_lens=[512]*32, pfx=0, H_q=32, H_kv=8, D=128),
    ]

    highvar = [
        dict(label="highvar_B64_1x63+300",
             ext_lens=[1]*63 + [300],               pfx=0,    H_q=32, H_kv=8, D=128),
        dict(label="highvar_B32_chat_mix",
             ext_lens=[17, 33, 65, 128]*8,          pfx=4096, H_q=32, H_kv=8, D=128),
        dict(label="highvar_B64_short_tail",
             ext_lens=[17, 37, 65, 97]*16,          pfx=0,    H_q=32, H_kv=8, D=128),
        dict(label="highvar_B16_extreme",
             ext_lens=[1]*15 + [1000],              pfx=0,    H_q=32, H_kv=8, D=128),
        dict(label="highvar_B64_sharegpt",
             ext_lens=([10, 20, 40, 60, 80, 100, 150, 200,
                        250, 300, 400, 500, 600, 700, 800, 1000] * 4),
             pfx=0, H_q=32, H_kv=8, D=128),
    ]

    d64 = [
        dict(label="d64_uniform_B64_ext256",
             ext_lens=[256]*64, pfx=0, H_q=64, H_kv=8, D=64),
        dict(label="d64_highvar_B64_1x63+300",
             ext_lens=[1]*63 + [300],  pfx=0, H_q=64, H_kv=8, D=64),
        dict(label="d64_highvar_B32_chat_mix",
             ext_lens=[17, 33, 65, 128]*8, pfx=4096, H_q=64, H_kv=8, D=64),
    ]

    return {"uniform": uniform, "highvar": highvar, "d64": d64}


# ── Tile-count analytics ─────────────────────────────────────────────────

def compute_tile_stats(ext_lens: list[int], H_q: int, BLOCK_M: int) -> dict:
    B = len(ext_lens)
    max_ext = max(ext_lens)
    useful_tiles = sum(math.ceil(e / BLOCK_M) for e in ext_lens) * H_q
    legacy_tiles = B * math.ceil(max_ext / BLOCK_M) * H_q
    total_ext = sum(ext_lens)
    fixed_tiles = ((total_ext + B * (BLOCK_M - 1)) // BLOCK_M) * H_q

    legacy_waste = 1.0 - useful_tiles / max(1, legacy_tiles)
    fixed_waste = 1.0 - useful_tiles / max(1, fixed_tiles)
    return dict(
        useful=useful_tiles,
        legacy=legacy_tiles,
        fixed=fixed_tiles,
        legacy_waste_pct=legacy_waste * 100,
        fixed_waste_pct=fixed_waste * 100,
    )


# ── Subprocess worker ─────────────────────────────────────────────────────

def _run_worker(mode: str, cases: list[dict], warmup: int, iters: int) -> list[dict]:
    """Measure kernel times in a subprocess with the given grid mode."""
    cases_json = json.dumps(cases)
    env = os.environ.copy()
    if mode == "legacy":
        env["SGLANG_GLUON_LEGACY_GRID"] = "1"
    else:
        env.pop("SGLANG_GLUON_LEGACY_GRID", None)

    cmd = [
        sys.executable, __file__,
        "--_mode", mode,
        "--_cases_json", cases_json,
        "--warmup", str(warmup),
        "--iters", str(iters),
    ]
    result = subprocess.run(
        cmd, capture_output=True, text=True, env=env, timeout=600,
    )
    if result.returncode != 0:
        print(f"Worker ({mode}) failed:\n{result.stderr}", file=sys.stderr)
        raise RuntimeError(f"Worker subprocess ({mode}) exited with {result.returncode}")
    return json.loads(result.stdout)


def _worker_main(mode: str, cases_json: str, warmup: int, iters: int):
    """Internal: run inside a subprocess with SGLANG_GLUON_LEGACY_GRID set."""
    import torch

    from sglang.srt.layers.attention.gluon_ops.cdna4.extend_attention import (
        gluon_extend_attention_fwd,
    )
    from sglang.srt.layers.attention.gluon_ops.cdna4.extend_attention.extend_attention_gfx950 import (
        _FORCE_LEGACY_GRID,
        _get_wca_heuristic_config,
        _resolve_qk_split_dims,
    )

    legacy_active = _FORCE_LEGACY_GRID
    assert (mode == "legacy") == legacy_active, (
        f"Expected legacy={mode=='legacy'}, got _FORCE_LEGACY_GRID={legacy_active}"
    )

    BF16 = torch.bfloat16
    DEV = "cuda:0"
    cases = json.loads(cases_json)

    def mk_hetero(ext_lens, pfx_per_seq, H_q, H_kv, D):
        torch.manual_seed(42)
        B = len(ext_lens)
        pfx_lens = [pfx_per_seq] * B
        total_ext = sum(ext_lens)
        total_pfx = sum(pfx_lens)
        total_kv = sum(e + p for e, p in zip(ext_lens, pfx_lens))

        q = torch.randn(total_ext, H_q, D, device=DEV, dtype=BF16) / 8
        k = torch.randn(total_ext, H_kv, D, device=DEV, dtype=BF16) / 8
        v = torch.randn(total_ext, H_kv, D, device=DEV, dtype=BF16) / 8
        o = torch.empty_like(q)
        kb = torch.randn(total_kv + 16, H_kv, D, device=DEV, dtype=BF16) / 8
        vb = torch.randn(total_kv + 16, H_kv, D, device=DEV, dtype=BF16) / 8

        qo_indptr = torch.zeros(B + 1, dtype=torch.int64, device=DEV)
        kv_indptr = torch.zeros(B + 1, dtype=torch.int32, device=DEV)
        kv_indices_list = []
        kv_offset = 0
        for i in range(B):
            qo_indptr[i + 1] = qo_indptr[i] + ext_lens[i]
            kv_indptr[i + 1] = kv_indptr[i] + pfx_lens[i]
            kv_indices_list.append(
                torch.arange(kv_offset, kv_offset + pfx_lens[i], device=DEV, dtype=torch.int64)
            )
            kv_offset += pfx_lens[i] + ext_lens[i]
        kv_indices = (
            torch.cat(kv_indices_list) if total_pfx > 0
            else torch.empty(0, dtype=torch.int64, device=DEV)
        )
        mask_indptr = torch.zeros(B + 1, device=DEV, dtype=torch.int32)
        return dict(
            q=q, k=k, v=v, o=o, kb=kb, vb=vb,
            qo_indptr=qo_indptr, kv_indptr=kv_indptr,
            kv_indices=kv_indices, mask_indptr=mask_indptr,
            max_len_extend=max(ext_lens), min_len_extend=min(ext_lens),
            total_prefix_len=total_pfx, total_extend_len=total_ext,
        )

    def call_gluon(t):
        gluon_extend_attention_fwd(
            t["q"], t["k"], t["v"], t["o"],
            t["kb"], t["vb"],
            t["qo_indptr"], t["kv_indptr"], t["kv_indices"],
            None, True, t["mask_indptr"], t["max_len_extend"],
            1.0, 1.0,
            min_len_extend=t["min_len_extend"],
            total_prefix_len=t["total_prefix_len"],
            total_extend_len=t["total_extend_len"],
        )

    def bench_gpu(fn, warmup_n, iters_n):
        for _ in range(warmup_n):
            fn()
        torch.cuda.synchronize()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters_n)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters_n)]
        for i in range(iters_n):
            starts[i].record()
            fn()
            ends[i].record()
        torch.cuda.synchronize()
        times_us = sorted(s.elapsed_time(e) * 1000 for s, e in zip(starts, ends))
        n = len(times_us)
        median = times_us[n // 2]
        mean = sum(times_us) / n
        std = (sum((t - mean) ** 2 for t in times_us) / n) ** 0.5
        p5 = times_us[max(0, n * 5 // 100)]
        p95 = times_us[min(n - 1, n * 95 // 100)]
        return dict(median_us=median, mean_us=mean, std_us=std, p5_us=p5, p95_us=p95)

    results = []
    for case in cases:
        t = mk_hetero(case["ext_lens"], case["pfx"], case["H_q"], case["H_kv"], case["D"])
        timing = bench_gpu(lambda: call_gluon(t), warmup, iters)
        results.append(dict(label=case["label"], **timing))
        sys.stderr.write(f"  [{mode}] {case['label']}: {timing['median_us']:.1f} us\n")
        sys.stderr.flush()

    print(json.dumps(results))


# ── Report formatting ─────────────────────────────────────────────────────

def _get_block_m(D: int, ext_lens: list[int], H_q: int, pfx: int) -> int:
    """Estimate BLOCK_M for the data-centric path to compute tile stats.

    Uses heuristic defaults matching the kernel's config selector.
    """
    B = len(ext_lens)
    max_ext = max(ext_lens)
    if D == 64:
        if B >= 4 and (B * max_ext >= 2048 or max_ext >= 512):
            return 256
        if B >= 16 and max_ext <= 32:
            return 64
        if B >= 16:
            if max_ext >= 512:
                return 256
            return 64
        return 128
    if B >= 16 and max_ext <= 16:
        return 16
    if B >= 16 and max_ext <= 64:
        return 64
    return 64


def format_report(
    all_cases: list[dict],
    legacy_results: list[dict],
    fixed_results: list[dict],
) -> str:
    lines = []
    lines.append("=" * 110)
    lines.append("WASTED-CTA THROUGHPUT BENCHMARK: legacy grid vs fixed (WCA) grid")
    lines.append("=" * 110)
    lines.append("")

    hdr = (
        f"{'case':<30s}  {'legacy':>8s}  {'fixed':>8s}  "
        f"{'speedup':>7s}  {'legacy_wst%':>11s}  {'fixed_wst%':>10s}  "
        f"{'useful':>8s}  {'legacy':>8s}  {'fixed':>8s}"
    )
    lines.append(hdr)
    sub = (
        f"{'':30s}  {'(us)':>8s}  {'(us)':>8s}  "
        f"{'':>7s}  {'':>11s}  {'':>10s}  "
        f"{'tiles':>8s}  {'tiles':>8s}  {'tiles':>8s}"
    )
    lines.append(sub)
    lines.append("-" * len(hdr))

    for case, lr, fr in zip(all_cases, legacy_results, fixed_results):
        BM = _get_block_m(case["D"], case["ext_lens"], case["H_q"], case["pfx"])
        stats = compute_tile_stats(case["ext_lens"], case["H_q"], BM)
        l_us = lr["median_us"]
        f_us = fr["median_us"]
        speedup = l_us / f_us if f_us > 0 else float("inf")
        lines.append(
            f"{case['label']:<30s}  {l_us:>7.1f}   {f_us:>7.1f}   "
            f"{speedup:>6.2f}x  {stats['legacy_waste_pct']:>10.1f}%  "
            f"{stats['fixed_waste_pct']:>9.1f}%  "
            f"{stats['useful']:>8d}  {stats['legacy']:>8d}  {stats['fixed']:>8d}"
        )

    lines.append("")
    lines.append("--- Statistics (mean +/- std, p5, p95) ---")
    lines.append("")
    hdr2 = (
        f"{'case':<30s}  {'legacy mean':>11s}  {'legacy std':>10s}  "
        f"{'fixed mean':>10s}  {'fixed std':>9s}  "
        f"{'leg p5':>7s}  {'leg p95':>7s}  {'fix p5':>7s}  {'fix p95':>7s}"
    )
    lines.append(hdr2)
    lines.append("-" * len(hdr2))
    for case, lr, fr in zip(all_cases, legacy_results, fixed_results):
        lines.append(
            f"{case['label']:<30s}  "
            f"{lr['mean_us']:>10.1f}   {lr['std_us']:>9.1f}   "
            f"{fr['mean_us']:>9.1f}   {fr['std_us']:>8.1f}   "
            f"{lr['p5_us']:>6.1f}   {lr['p95_us']:>6.1f}   "
            f"{fr['p5_us']:>6.1f}   {fr['p95_us']:>6.1f}"
        )

    lines.append("")
    lines.append("EXPECTED:")
    lines.append("  - Uniform cases: speedup ~1.0x (both grids launch similar work)")
    lines.append("  - High-variance cases: speedup >> 1.0x (legacy wastes CTA slots)")
    lines.append("")
    return "\n".join(lines)


# ── Main driver ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Wasted-CTA throughput benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Spawns two subprocesses (legacy grid, fixed grid) and compares
            kernel-level GPU throughput for heterogeneous extend-attention
            batches.  High-variance cases should show significant speedup
            with the fixed grid; uniform cases should show ~1.0x.
        """),
    )
    parser.add_argument(
        "--cases",
        nargs="*",
        choices=["uniform", "highvar", "d64", "all"],
        default=["all"],
        help="Case categories to run (default: all)",
    )
    parser.add_argument("--warmup", type=int, default=50, help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=200, help="Measurement iterations")
    parser.add_argument("--output", type=str, default=None, help="Write report to file")

    # Internal subprocess args (not for user consumption)
    parser.add_argument("--_mode", type=str, default=None)
    parser.add_argument("--_cases_json", type=str, default=None)

    args = parser.parse_args()

    if args._mode is not None:
        _worker_main(args._mode, args._cases_json, args.warmup, args.iters)
        return

    all_case_groups = _make_cases()
    selected = set(args.cases)
    if "all" in selected:
        selected = set(all_case_groups.keys())

    cases = []
    for group_name in ["uniform", "highvar", "d64"]:
        if group_name in selected and group_name in all_case_groups:
            cases.extend(all_case_groups[group_name])

    if not cases:
        print("No cases selected.", file=sys.stderr)
        sys.exit(1)

    print(f"Running {len(cases)} cases, {args.warmup} warmup + {args.iters} iters each")
    print(f"Modes: legacy (data-centric rectangular grid), fixed (compact WCA grid)")
    print()

    print("==> Running LEGACY grid mode...")
    legacy_results = _run_worker("legacy", cases, args.warmup, args.iters)

    print("==> Running FIXED grid mode...")
    fixed_results = _run_worker("fixed", cases, args.warmup, args.iters)

    report = format_report(cases, legacy_results, fixed_results)
    print()
    print(report)

    if args.output:
        with open(args.output, "w") as f:
            f.write(report)
        print(f"\nReport written to {args.output}")


if __name__ == "__main__":
    main()
