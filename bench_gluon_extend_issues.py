"""Extended extend-attention benchmark targeting two known issues.

Issue 1 — max_len_extend wasted CTAs
--------------------------------------
The current grid uses B * ceil(max_ext / BLOCK_M) CTAs.  When extend lengths
are heterogeneous (high variance within a batch), most of those CTAs exit
immediately because there is no work for them.  This sweep measures:

  wasted_tile_ratio = 1 - sum_i(ceil(ext_i / BLOCK_M)) / (B * ceil(max_ext / BLOCK_M))

and times the Gluon auto-dispatch and Triton kernels across batches with
varying levels of extend-length variance, to quantify the performance cost.

Issue 2 — WCA vs split-K dispatch heuristic
---------------------------------------------
For ragged-prefix and ragged-extend shapes the kernel has three candidate
paths: data-centric (DC), split-K, and WCA.  This sweep forces each path
by calling the private _launch_* helpers directly, then compares their
wall times across a (B, prefix_len, extend_len) grid, so we can validate
(or correct) the auto-dispatch heuristic.

Usage:
    python bench_gluon_extend_issues.py            # both sweeps
    python bench_gluon_extend_issues.py --issue 1  # wasted-CTA sweep only
    python bench_gluon_extend_issues.py --issue 2  # WCA/splitk sweep only
"""

from __future__ import annotations

import argparse
import math

import torch

from sglang.srt.layers.attention.gluon_ops.cdna4.extend_attention import (
    gluon_extend_attention_fwd,
)
from sglang.srt.layers.attention.gluon_ops.cdna4.extend_attention.extend_attention_gfx950 import (
    _get_wca_heuristic_config,
    _launch_data_centric_grid,
    _launch_splitk,
    _launch_wca,
    _resolve_qk_split_dims,
)
from sglang.srt.layers.attention.triton_ops.extend_attention import (
    extend_attention_fwd as triton_fwd,
)

BF16 = torch.bfloat16
DEV = "cuda:0"

# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------


def bench_gpu(fn, warmup: int = 50, iters: int = 300) -> float:
    """Return median kernel time in microseconds."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    times_us = sorted(s.elapsed_time(e) * 1000 for s, e in zip(starts, ends))
    return times_us[iters // 2]


# ---------------------------------------------------------------------------
# Tensor builders
# ---------------------------------------------------------------------------


def mk_uniform(B: int, S: int, H_q: int, H_kv: int, D: int):
    """Uniform batch: every sequence has the same extend length S, no prefix."""
    torch.manual_seed(0)
    total = B * S
    q = torch.randn(total, H_q, D, device=DEV, dtype=BF16) / 8
    k = torch.randn(total, H_kv, D, device=DEV, dtype=BF16) / 8
    v = torch.randn(total, H_kv, D, device=DEV, dtype=BF16) / 8
    o = torch.empty_like(q)
    psz = total + 16
    kb = torch.zeros(psz, H_kv, D, device=DEV, dtype=BF16)
    kb[:total] = k
    vb = torch.zeros(psz, H_kv, D, device=DEV, dtype=BF16)
    vb[:total] = v
    qo_indptr = torch.arange(0, total + 1, S, device=DEV, dtype=torch.int64)
    kv_indptr = torch.arange(0, total + 1, S, device=DEV, dtype=torch.int32)
    kv_indices = torch.arange(0, total, device=DEV, dtype=torch.int64)
    mask_indptr = torch.zeros(B + 1, device=DEV, dtype=torch.int32)
    return dict(
        q=q, k=k, v=v, o=o,
        kb=kb, vb=vb,
        qo_indptr=qo_indptr,
        kv_indptr=kv_indptr,
        kv_indices=kv_indices,
        mask_indptr=mask_indptr,
        max_len_extend=S,
        min_len_extend=S,
        total_prefix_len=total,
        total_extend_len=total,
    )


def mk_hetero(
    ext_lens: list[int],
    pfx_lens: list[int],
    H_q: int,
    H_kv: int,
    D: int,
):
    """Heterogeneous batch with per-sequence extend and prefix lengths."""
    torch.manual_seed(0)
    B = len(ext_lens)
    assert len(pfx_lens) == B

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
    kv_indices_list: list[torch.Tensor] = []

    kv_offset = 0
    for i in range(B):
        qo_indptr[i + 1] = qo_indptr[i] + ext_lens[i]
        kv_indptr[i + 1] = kv_indptr[i] + pfx_lens[i]
        kv_indices_list.append(
            torch.arange(kv_offset, kv_offset + pfx_lens[i], device=DEV, dtype=torch.int64)
        )
        kv_offset += pfx_lens[i] + ext_lens[i]

    kv_indices = torch.cat(kv_indices_list) if total_pfx > 0 else torch.empty(0, dtype=torch.int64, device=DEV)
    mask_indptr = torch.zeros(B + 1, device=DEV, dtype=torch.int32)

    return dict(
        q=q, k=k, v=v, o=o,
        kb=kb, vb=vb,
        qo_indptr=qo_indptr,
        kv_indptr=kv_indptr,
        kv_indices=kv_indices,
        mask_indptr=mask_indptr,
        max_len_extend=max(ext_lens),
        min_len_extend=min(ext_lens),
        total_prefix_len=total_pfx,
        total_extend_len=total_ext,
        max_prefix_len=max(pfx_lens) if pfx_lens and max(pfx_lens) > 0 else None,
    )


# ---------------------------------------------------------------------------
# Kernel callers
# ---------------------------------------------------------------------------


def call_gluon_auto(t):
    gluon_extend_attention_fwd(
        t["q"], t["k"], t["v"], t["o"],
        t["kb"], t["vb"],
        t["qo_indptr"], t["kv_indptr"], t["kv_indices"],
        None, True, t["mask_indptr"], t["max_len_extend"],
        1.0, 1.0,
        min_len_extend=t["min_len_extend"],
        total_prefix_len=t["total_prefix_len"],
        total_extend_len=t["total_extend_len"],
        max_prefix_len=t.get("max_prefix_len"),
    )


def call_triton(t):
    triton_fwd(
        t["q"], t["k"], t["v"], t["o"],
        t["kb"], t["vb"],
        t["qo_indptr"], t["kv_indptr"],
        t["kv_indices"].to(torch.int64),
        None, True, t["mask_indptr"], t["max_len_extend"],
        1.0, 1.0,
    )


def call_wca(t):
    _launch_wca(
        t["q"], t["k"], t["v"], t["o"],
        t["kb"], t["vb"],
        t["qo_indptr"], t["kv_indptr"], t["kv_indices"],
        None, True, t["mask_indptr"], t["max_len_extend"],
        1.0, 1.0,
        None,   # sm_scale
        0.0,    # logit_cap
        -1,     # sliding_window_size
        None,   # sinks
        None,   # window_kv_offsets
        -1,     # xai_temperature_len
        False,  # kv_is_fp8
        t["total_prefix_len"],
        t["min_len_extend"],
    )


def call_splitk(t, D: int):
    _launch_splitk(
        t["q"], t["k"], t["v"], t["o"],
        t["kb"], t["vb"],
        t["qo_indptr"], t["kv_indptr"], t["kv_indices"],
        None,   # custom_mask
        t["mask_indptr"],
        None,   # window_kv_offsets
        None,   # sm_scale
        1.0, 1.0,
        0.0,    # logit_cap
        D,
        True,   # is_causal
        t["max_len_extend"], t["min_len_extend"],
        None,   # sinks
        -1,     # xai_temperature_len
        -1,     # sliding_window_size
        BLOCK_M=64,
        BLOCK_N=64,
        num_warps=4,
        NUM_STAGES=2,
        total_prefix_len=t["total_prefix_len"],
        kv_is_fp8=False,
    )


def call_data_centric(t, D: int, B: int, H_q: int, BLOCK_M=128, BLOCK_N=64, NUM_STAGES=2, num_warps=8):
    BLOCK_DMODEL = _resolve_qk_split_dims(D)
    _launch_data_centric_grid(
        t["q"], t["k"], t["v"], t["o"],
        t["kb"], t["vb"],
        t["qo_indptr"], t["kv_indptr"], t["kv_indices"],
        True,   # is_causal
        t["max_len_extend"],
        1.0, 1.0,
        None,   # sm_scale
        0.0,    # logit_cap
        -1,     # sliding_window_size
        None,   # sinks
        -1,     # xai_temperature_len
        D, H_q, B,
        False,  # kv_is_fp8
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=BLOCK_DMODEL,
        NUM_STAGES=NUM_STAGES,
        num_warps=num_warps,
        total_prefix_len_hint=t["total_prefix_len"],
        total_extend_len_hint=t["total_extend_len"],
        min_len_extend_hint=t["min_len_extend"],
    )


# ---------------------------------------------------------------------------
# Issue 1 — wasted CTA sweep
# ---------------------------------------------------------------------------


def wasted_tile_ratio(ext_lens: list[int], BLOCK_M: int = 16) -> float:
    max_ext = max(ext_lens)
    useful = sum(math.ceil(e / BLOCK_M) for e in ext_lens)
    total = len(ext_lens) * math.ceil(max_ext / BLOCK_M)
    return 1.0 - useful / max(1, total)


def wca_wasted_tile_ratio(
    ext_lens: list[int], BLOCK_M: int, head_num: int
) -> tuple[float, int, int]:
    """Return (waste_fraction, exact_tiles, wca_tiles) using the WCA tile formula.

    WCA launches ``(sum(ext_i) + B*(BLOCK_M-1)) // BLOCK_M * H`` CTAs, which is
    an upper bound on the exact tile count ``sum_i ceil(ext_i/BLOCK_M) * H``.
    Any overestimated slots exit early inside the kernel via ``_schedule_wca``.
    """
    B = len(ext_lens)
    total_ext = sum(ext_lens)
    exact_tiles = sum(math.ceil(e / BLOCK_M) for e in ext_lens) * head_num
    wca_tiles = ((total_ext + B * (BLOCK_M - 1)) // BLOCK_M) * head_num
    waste = 1.0 - exact_tiles / max(1, wca_tiles)
    return waste, exact_tiles, wca_tiles


# Each entry: (label, B, ext_lens list, prefix_len_per_seq, H_q, H_kv, D)
# ext_lens drives the waste; all seqs share the same prefix length.
ISSUE1_CASES: list[tuple] = [
    # ---- D=128 cases -------------------------------------------------------
    # Uniform — zero waste baseline
    ("D128 uniform B=8 ext=256",    [256] * 8,            0,   32, 8, 128),
    ("D128 uniform B=16 ext=512",   [512] * 16,            0,  32, 8, 128),
    # Moderate heterogeneity
    ("D128 hetero B=8 ext=[1..256]",
        [1, 32, 64, 96, 128, 160, 192, 256],          0,  32, 8, 128),
    # High heterogeneity (one long, rest short) — worst case for max_len grid
    ("D128 high-var B=8 ext=[1..300]",
        [1, 1, 1, 1, 1, 1, 1, 300],                   0,  32, 8, 128),
    ("D128 high-var B=16 ext=[1..300]",
        [1] * 15 + [300],                              0,  32, 8, 128),
    # ShareGPT-like distribution (rough sketch)
    ("D128 sharegpt-like B=16",
        [10, 20, 40, 60, 80, 100, 150, 200,
         250, 300, 400, 500, 600, 700, 800, 1000],     0,  32, 8, 128),
    # With prefix (long prefix + short extend — spec-decode style)
    ("D128 prefix B=8 pfx=4096 ext=[1..64]",
        [1, 4, 8, 16, 32, 48, 56, 64],             4096,  32, 8, 128),
    # ---- D=64 cases (Issue 1 focus: no pipelined WCA before this fix) ------
    # Uniform — zero waste baseline for D=64
    ("D64 uniform B=8 ext=256",     [256] * 8,            0,   64, 8, 64),
    # Moderate heterogeneity D=64
    ("D64 hetero B=8 ext=[1..256]",
        [1, 32, 64, 96, 128, 160, 192, 256],          0,  64, 8, 64),
    # High heterogeneity D=64 — these are the fix targets
    ("D64 high-var B=8 ext=[1..300]",
        [1, 1, 1, 1, 1, 1, 1, 300],                   0,  64, 8, 64),
    ("D64 high-var B=16 ext=[1..300]",
        [1] * 15 + [300],                              0,  64, 8, 64),
    # ShareGPT-like D=64
    ("D64 sharegpt-like B=16",
        [10, 20, 40, 60, 80, 100, 150, 200,
         250, 300, 400, 500, 600, 700, 800, 1000],     0,  64, 8, 64),
]


def run_issue1():
    print("\n" + "=" * 80)
    print("ISSUE 1: max_len_extend wasted CTAs")
    print("=" * 80)
    DC_BLOCK_M = 16  # proxy used in the DC waste formula (actual kernel BM varies)
    hdr = (
        f"{'case':<38s}  {'DC waste%':>9s}  {'WCA waste%':>10s}  "
        f"{'Triton':>7s}  {'Gluon':>7s}  {'gain%':>6s}"
    )
    print(hdr)
    print("-" * len(hdr))
    for label, ext_lens, pfx_per_seq, H_q, H_kv, D in ISSUE1_CASES:
        B = len(ext_lens)
        pfx_lens = [pfx_per_seq] * B
        t = mk_hetero(ext_lens, pfx_lens, H_q, H_kv, D)
        dc_waste = wasted_tile_ratio(ext_lens, BLOCK_M=DC_BLOCK_M) * 100

        wca_cfg = _get_wca_heuristic_config(
            D,
            _resolve_qk_split_dims(D),
            B,
            max(ext_lens),
            min(ext_lens),
            pfx_per_seq * B,
            sum(ext_lens),
            False,
            1,
            None,
        )
        wca_waste_frac, _, _ = wca_wasted_tile_ratio(ext_lens, wca_cfg.block_m, H_q)
        wca_waste = wca_waste_frac * 100

        t_triton = bench_gpu(lambda: call_triton(t))
        t_gluon = bench_gpu(lambda: call_gluon_auto(t))
        gain = (t_triton - t_gluon) / t_triton * 100 if t_triton > 0 else 0
        flag = "" if abs(gain) < 1 else ("WIN" if gain > 0 else "REGR")
        print(
            f"{label:<38s}  {dc_waste:>8.1f}%  {wca_waste:>9.1f}%  "
            f"{t_triton:>5.1f}us  {t_gluon:>5.1f}us  {gain:>+5.1f}%  {flag}"
        )


# ---------------------------------------------------------------------------
# Issue 2 — WCA vs split-K vs data-centric heuristic sweep
# ---------------------------------------------------------------------------

# Each entry: (label, B, prefix_per_seq, extend_per_seq, H_q, H_kv, D)
ISSUE2_CASES: list[tuple] = [
    # Short extend, long prefix — WCA should win (reclaims prefix-partition work)
    ("B=8  pfx=8192 ext=1",    8,  8192,   1, 32, 8, 128),
    ("B=8  pfx=8192 ext=32",   8,  8192,  32, 32, 8, 128),
    ("B=8  pfx=8192 ext=64",   8,  8192,  64, 32, 8, 128),
    ("B=8  pfx=8192 ext=128",  8,  8192, 128, 32, 8, 128),
    ("B=16 pfx=4096 ext=32",   16, 4096,  32, 32, 8, 128),
    ("B=16 pfx=4096 ext=64",   16, 4096,  64, 32, 8, 128),
    ("B=16 pfx=4096 ext=128",  16, 4096, 128, 32, 8, 128),
    # Long extend, short prefix — data-centric / split-K should win
    ("B=4  pfx=256  ext=512",   4,  256, 512, 32, 8, 128),
    ("B=4  pfx=256  ext=1024",  4,  256, 1024, 32, 8, 128),
    ("B=8  pfx=512  ext=512",   8,  512, 512, 32, 8, 128),
    ("B=8  pfx=512  ext=1024",  8,  512, 1024, 32, 8, 128),
    # Medium prefix, medium extend — boundary region
    ("B=8  pfx=2048 ext=256",   8, 2048, 256, 32, 8, 128),
    ("B=8  pfx=2048 ext=512",   8, 2048, 512, 32, 8, 128),
    ("B=16 pfx=2048 ext=256",  16, 2048, 256, 32, 8, 128),
    ("B=16 pfx=2048 ext=512",  16, 2048, 512, 32, 8, 128),
    # Ragged-extend high-variance (issue 1 overlap) with medium prefix
    ("B=16 pfx=2048 ext=ragged",
        None, 2048, None, 32, 8, 128),  # handled specially below
    # --- Ragged PREFIX cases (the actual target of the PR) ---
    # Ragged prefix, uniform short extend (spec-decode pattern)
    ("B=8  pfx=ragged ext=7",
        8, None, 7, 32, 8, 128),  # handled specially below
    # Ragged prefix, moderate extend (chat continuation pattern)
    ("B=8  pfx=ragged ext=64",
        8, None, 64, 32, 8, 128),
    # Ragged prefix + ragged extend (realistic mixed batch)
    ("B=8  pfx=ragged ext=ragged",
        8, None, None, 32, 8, 128),
    # Ragged prefix, large batch
    ("B=16 pfx=ragged ext=7",
        16, None, 7, 32, 8, 128),
    # Ragged prefix D=64 (spec-decode pattern)
    ("B=8  pfx=ragged ext=7 D64",
        8, None, 7, 64, 8, 64),
    # Ragged prefix D=64, moderate extend
    ("B=8  pfx=ragged ext=64 D64",
        8, None, 64, 64, 8, 64),
]


def run_issue2():
    print("\n" + "=" * 80)
    print("ISSUE 2: WCA vs split-K vs data-centric heuristic")
    print("=" * 80)
    print("  'best' = fastest forced path; 'auto routed' = which path auto chose")
    print("  'penalty' = how much slower auto is vs best (blank if <2%)")
    print()
    hdr = (
        f"{'case':<32s}  {'auto':>7s}  {'DC':>7s}  "
        f"{'splitK':>7s}  {'WCA':>7s}  {'best':>6s}  {'auto routed':>12s}  {'penalty':>7s}"
    )
    print(hdr)
    print("-" * len(hdr))

    ragged_ext = [10, 20, 40, 80, 100, 150, 200, 250,
                  300, 400, 450, 500, 550, 600, 650, 700]
    ragged_pfx_8 = [100, 512, 1024, 2048, 4096, 6000, 8000, 8192]
    ragged_pfx_16 = [100, 256, 512, 768, 1024, 1536, 2048, 3072,
                     4096, 5000, 6000, 6500, 7000, 7500, 8000, 8192]
    ragged_ext_mixed_8 = [7, 32, 64, 128, 7, 256, 32, 7]

    for row in ISSUE2_CASES:
        label, B, pfx, ext, H_q, H_kv, D = row
        if pfx is None and ext is None:
            # Ragged prefix + ragged extend
            pfx_lens = ragged_pfx_8[:B]
            ext_lens = ragged_ext_mixed_8[:B]
        elif pfx is None:
            # Ragged prefix, uniform extend
            pfx_lens = ragged_pfx_8[:B] if B <= 8 else ragged_pfx_16[:B]
            ext_lens = [ext] * B
        elif ext is None:
            # Uniform prefix, ragged extend
            ext_lens = ragged_ext[:B]
            pfx_lens = [pfx] * len(ext_lens)
        else:
            ext_lens = [ext] * B
            pfx_lens = [pfx] * B
        B_actual = len(ext_lens)

        t = mk_hetero(ext_lens, pfx_lens, H_q, H_kv, D)

        t_auto = bench_gpu(lambda: call_gluon_auto(t))
        t_dc = bench_gpu(lambda: call_data_centric(t, D, B_actual, H_q))

        try:
            t_sk = bench_gpu(lambda: call_splitk(t, D))
        except (AssertionError, RuntimeError):
            t_sk = float("inf")
        try:
            t_wca = bench_gpu(lambda: call_wca(t))
        except (AssertionError, RuntimeError):
            t_wca = float("inf")

        forced = []
        forced.append(("DC", t_dc))
        if t_sk != float("inf"):
            forced.append(("splitK", t_sk))
        if t_wca != float("inf"):
            forced.append(("WCA", t_wca))

        best_name, best_time = min(forced, key=lambda x: x[1])
        auto_routed = min(forced, key=lambda x: abs(x[1] - t_auto))[0]
        penalty_pct = (t_auto - best_time) / best_time * 100 if best_time > 0 else 0
        penalty_str = f"{penalty_pct:>+5.1f}%" if penalty_pct >= 2.0 else ""

        sk_str = f"{t_sk:>5.1f}us" if t_sk != float("inf") else "    --  "
        wca_str = f"{t_wca:>5.1f}us" if t_wca != float("inf") else "    --  "
        print(
            f"{label:<32s}  {t_auto:>5.1f}us  {t_dc:>5.1f}us  "
            f"{sk_str}  {wca_str}  {best_name:>6s}  {auto_routed:>12s}  {penalty_str:>7s}"
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--issue",
        type=int,
        choices=[1, 2],
        default=None,
        help="Run only issue 1 or issue 2 sweep (default: both)",
    )
    args = parser.parse_args()

    if args.issue in (None, 1):
        run_issue1()
    if args.issue in (None, 2):
        run_issue2()
