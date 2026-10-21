"""Cache pressure probe for vLLM prefix-caching.

Purpose: Verify KV-cache eviction behaviour when total prefixes exceed GPU
KV capacity (108,912 tokens), and show that affinity scheduling (same prefix,
consecutive calls) maintains high hit rates even under pressure.

Experiment groups
-----------------
  pressure  : N=16, L≈8000  (~128k > 108,912, cannot all fit)
  control   : N=8,  L≈8000  (~64k  < 108,912, all fit)
  affinity  : pressure prefixes, each sent twice consecutively

All requests are serialised (no concurrency) so that Prometheus delta metrics
are unambiguous.

Usage:
    ./.venv/bin/python scripts/probe_cache_pressure.py 2>&1 | tee /tmp/probe_cache_pressure.out
"""

from __future__ import annotations

import asyncio
import json
import sys
import os
import textwrap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentos.llm.base import LLMRequest
from agentos.llm.vllm_backend import VLLMBackend

# ---------------------------------------------------------------------------
# Token-count calibration
# Each "paragraph" of 64 chars ≈ ~16 tokens on Qwen tokeniser (ratio ≈ 4).
# We'll measure exact ratio from the first calibration call and adjust.
# ---------------------------------------------------------------------------

CHARS_PER_PARA = 256   # characters per paragraph chunk
TARGET_TOKENS  = 8000  # desired prefix length in tokens


def build_prefix(index: int, n_paras: int) -> str:
    """Build a unique prefix string made of `n_paras` paragraphs."""
    lines = []
    for p in range(n_paras):
        # First word differs across prefixes to defeat any shared leading blocks.
        lines.append(
            f"[PREFIX_{index:04d}_PARA_{p:04d}] "
            + textwrap.fill(
                f"Document segment {index}-{p}: " + (
                    f"This is synthetic text for cache pressure experiments. "
                    f"Each paragraph has a unique identifier ensuring vLLM prefix "
                    f"cache treats prefixes {index} and {index+1} as completely "
                    f"distinct token sequences from the very first block. "
                    f"The following filler ensures sufficient length: "
                    + ("lorem ipsum dolor sit amet consectetur adipiscing elit " * 4)
                ),
                width=200
            )
        )
    return "\n".join(lines)


async def calibrate(backend: VLLMBackend) -> float:
    """Send one short call, measure chars→tokens ratio, return chars_per_token."""
    sample_paras = 4
    prefix = build_prefix(0, sample_paras)
    req = LLMRequest(
        agent_id="calibrate",
        messages=[
            {"role": "system", "content": prefix},
            {"role": "user", "content": "Say 'OK'."},
        ],
        max_tokens=4,
    )
    resp = await backend.generate(req)
    m = resp.metrics
    chars = len(prefix)
    toks  = m.prompt_tokens
    # prompt_tokens includes the user "Say OK" turn too, subtract ~6 tokens safety
    prefix_toks = max(toks - 8, 1)
    ratio = chars / prefix_toks  # chars per token
    print(f"[CALIB] chars={chars}, prompt_tokens={toks}, est_prefix_tokens={prefix_toks}, "
          f"chars_per_token={ratio:.2f}")
    return ratio


def compute_n_paras(target_tokens: int, chars_per_token: float) -> int:
    """How many paragraphs reach target_tokens?"""
    chars_per_para = CHARS_PER_PARA  # rough; will be refined after calibration
    # Build one paragraph to measure exact length
    sample = build_prefix(0, 1)
    chars_per_para_actual = len(sample)
    tokens_per_para = chars_per_para_actual / chars_per_token
    n = max(1, round(target_tokens / tokens_per_para))
    return n


async def call_and_log(
    backend: VLLMBackend,
    group: str,
    prefix_idx: int,
    prefix: str,
    user_msg: str,
) -> dict:
    req = LLMRequest(
        agent_id=f"{group}-{prefix_idx}",
        messages=[
            {"role": "system", "content": prefix},
            {"role": "user", "content": user_msg},
        ],
        max_tokens=16,
    )
    resp = await backend.generate(req)
    m = resp.metrics
    row = {
        "group":      group,
        "prefix_idx": prefix_idx,
        "prompt_tok": m.prompt_tokens,
        "hit":        m.cache_hit_tokens,
        "miss":       m.cache_miss_tokens,
        "ratio":      m.cache_hit_ratio,
        "latency_s":  round(m.latency_secs, 3),
    }
    print(
        f"[{group:16s}] prefix={prefix_idx:2d} "
        f"prompt={m.prompt_tokens:5d} hit={m.cache_hit_tokens:5d} "
        f"miss={m.cache_miss_tokens:5d} ratio={m.cache_hit_ratio:5.1%} "
        f"latency={m.latency_secs:.3f}s"
    )
    return row


def avg(vals):
    return sum(vals) / len(vals) if vals else 0.0


async def main() -> None:
    backend = VLLMBackend()

    # ---- Calibration ----
    print("=" * 70)
    print("CALIBRATION")
    print("=" * 70)
    chars_per_token = await calibrate(backend)

    n_paras = compute_n_paras(TARGET_TOKENS, chars_per_token)

    # Check actual token estimate
    sample_prefix = build_prefix(0, n_paras)
    est_tokens = len(sample_prefix) / chars_per_token
    print(f"[CALIB] n_paras={n_paras}, est_prefix_tokens≈{est_tokens:.0f}")

    # If estimate is too far off, tune n_paras
    if est_tokens < TARGET_TOKENS * 0.7:
        n_paras = round(n_paras * TARGET_TOKENS / est_tokens)
        print(f"[CALIB] Adjusted n_paras={n_paras} (was too short)")
    elif est_tokens > TARGET_TOKENS * 1.3:
        n_paras = round(n_paras * TARGET_TOKENS / est_tokens)
        print(f"[CALIB] Adjusted n_paras={n_paras} (was too long)")

    # Recalculate final estimate
    sample_prefix = build_prefix(0, n_paras)
    est_tokens_final = len(sample_prefix) / chars_per_token
    print(f"[CALIB] Final n_paras={n_paras}, final est_tokens≈{est_tokens_final:.0f}")

    # Build all prefixes
    N_PRESSURE  = 16
    N_CONTROL   = 8
    TOTAL_CAPACITY = 108912

    prefixes_pressure = [build_prefix(i,       n_paras) for i in range(N_PRESSURE)]
    prefixes_control  = [build_prefix(100 + i, n_paras) for i in range(N_CONTROL)]

    total_pressure = N_PRESSURE * est_tokens_final
    total_control  = N_CONTROL  * est_tokens_final
    print(f"\n[DESIGN] Pressure: N={N_PRESSURE}, est_total={total_pressure:.0f} vs capacity={TOTAL_CAPACITY}")
    print(f"[DESIGN] Control:  N={N_CONTROL},  est_total={total_control:.0f} vs capacity={TOTAL_CAPACITY}")

    if total_pressure <= TOTAL_CAPACITY:
        print("[WARN] Pressure group may not exceed capacity — eviction might not trigger.")
    if total_control >= TOTAL_CAPACITY:
        print("[WARN] Control group may exceed capacity — use fewer prefixes.")

    results: dict[str, list[dict]] = {
        "pressure_r1": [],
        "pressure_r2": [],
        "control_r1":  [],
        "control_r2":  [],
        "affinity":    [],
    }

    user_questions = [
        "What is the main topic of this document?",
        "Summarize this document in one sentence.",
        "What key identifiers appear in this text?",
        "List the main concepts described.",
        "What is the purpose of this document?",
        "Describe the content briefly.",
        "What numerical identifiers are used?",
        "Give a one-word category for this document.",
        "What format is this document in?",
        "What is unique about this text?",
        "How many paragraphs does this document have?",
        "What language pattern does this document use?",
        "Identify any repeated phrases.",
        "What does 'lorem ipsum' indicate here?",
        "Is this document structured or unstructured?",
        "What is the index of this prefix?",
    ]

    # ===================================================================
    # PRESSURE GROUP — Round 1 (cold fill)
    # ===================================================================
    print("\n" + "=" * 70)
    print("PRESSURE GROUP — Round 1 (cold, filling cache)")
    print("=" * 70)
    for i in range(N_PRESSURE):
        row = await call_and_log(backend, "pressure_r1", i,
                                  prefixes_pressure[i], user_questions[i % len(user_questions)])
        results["pressure_r1"].append(row)

    # ===================================================================
    # PRESSURE GROUP — Round 2 (should trigger evictions if total > cap)
    # ===================================================================
    print("\n" + "=" * 70)
    print("PRESSURE GROUP — Round 2 (replay, expect eviction-driven misses)")
    print("=" * 70)
    for i in range(N_PRESSURE):
        q = user_questions[(i + 1) % len(user_questions)]  # different question
        row = await call_and_log(backend, "pressure_r2", i,
                                  prefixes_pressure[i], q)
        results["pressure_r2"].append(row)

    # ===================================================================
    # CONTROL GROUP — Round 1 (cold fill, should fit in cache)
    # ===================================================================
    print("\n" + "=" * 70)
    print("CONTROL GROUP — Round 1 (cold, all prefixes should fit)")
    print("=" * 70)
    for i in range(N_CONTROL):
        row = await call_and_log(backend, "control_r1", i,
                                  prefixes_control[i], user_questions[i % len(user_questions)])
        results["control_r1"].append(row)

    # ===================================================================
    # CONTROL GROUP — Round 2 (expect near-100% hits)
    # ===================================================================
    print("\n" + "=" * 70)
    print("CONTROL GROUP — Round 2 (replay, expect high cache hit ratio)")
    print("=" * 70)
    for i in range(N_CONTROL):
        q = user_questions[(i + 1) % len(user_questions)]
        row = await call_and_log(backend, "control_r2", i,
                                  prefixes_control[i], q)
        results["control_r2"].append(row)

    # ===================================================================
    # AFFINITY GROUP — pressure prefixes, consecutive pairs
    # ===================================================================
    print("\n" + "=" * 70)
    print("AFFINITY GROUP — each pressure prefix sent twice consecutively")
    print("=" * 70)
    for i in range(N_PRESSURE):
        # First call (may miss — previous rounds may have evicted this)
        row1 = await call_and_log(backend, "affinity_1st", i,
                                   prefixes_pressure[i], user_questions[i % len(user_questions)])
        # Second call — must be a hit (same prefix, back-to-back)
        row2 = await call_and_log(backend, "affinity_2nd", i,
                                   prefixes_pressure[i],
                                   user_questions[(i + 2) % len(user_questions)])
        row2["pair_first_hit"] = row1["ratio"]
        results["affinity"].append(row2)

    # ===================================================================
    # SUMMARY
    # ===================================================================
    def group_summary(rows: list[dict]) -> dict:
        n = len(rows)
        if n == 0:
            return {}
        actual_l = avg([r["prompt_tok"] for r in rows])
        avg_ratio = avg([r["ratio"] for r in rows])
        avg_lat   = avg([r["latency_s"] for r in rows])
        return {
            "n_calls":       n,
            "avg_prompt_tok": round(actual_l, 1),
            "avg_hit_ratio":  round(avg_ratio, 4),
            "avg_latency_s":  round(avg_lat, 3),
        }

    summary = {
        "config": {
            "N_pressure": N_PRESSURE,
            "N_control":  N_CONTROL,
            "n_paras":    n_paras,
            "est_L_tokens": round(est_tokens_final),
            "chars_per_token": round(chars_per_token, 2),
            "GPU_KV_capacity": TOTAL_CAPACITY,
        },
        "pressure_round1": group_summary(results["pressure_r1"]),
        "pressure_round2": group_summary(results["pressure_r2"]),
        "control_round1":  group_summary(results["control_r1"]),
        "control_round2":  group_summary(results["control_r2"]),
        "affinity_2nd_call": group_summary(results["affinity"]),
    }

    print("\n" + "=" * 70)
    print("SUMMARY JSON")
    print("=" * 70)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
