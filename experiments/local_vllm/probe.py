"""探测本地 vLLM：前缀缓存命中粒度、首 Token 时间与解码吞吐。

用法：python probe.py --base-url http://127.0.0.1:18095 --model qwen3.5-9b --output out.json
只读取服务端返回的 usage 与 /metrics，不改服务状态（除了缓存本身）。
"""
from __future__ import annotations

import argparse
import json
import random
import re
import time
import uuid

import httpx


def metrics(client, base):
    text = client.get(base + "/metrics").text
    out = {}
    for name in ("prefix_cache_queries_total", "prefix_cache_hits_total", "prompt_tokens_total",
                 "generation_tokens_total"):
        m = re.search(rf"^vllm:{name}\{{[^}}]*\}} ([0-9.e+]+)$", text, re.M)
        out[name] = float(m.group(1)) if m else None
    return out


def filler(seed, words):
    rng = random.Random(seed)
    vocab = ("scheduler context cache prefix runtime agent token memory kernel task graph "
             "worker message event replay policy budget segment shared private snapshot").split()
    return " ".join(rng.choice(vocab) + str(rng.randint(0, 999)) for _ in range(words))


def chat(client, base, model, content, max_tokens, stream=False):
    payload = {"model": model, "messages": [{"role": "user", "content": content}],
               "max_tokens": max_tokens, "temperature": 0.0, "seed": 0}
    # Qwen3.5 思考模式会吞掉输出预算；探测只看缓存与时延，关闭思考。
    payload["chat_template_kwargs"] = {"enable_thinking": False}
    start = time.monotonic()
    if not stream:
        r = client.post(base + "/v1/chat/completions", json=payload)
        r.raise_for_status()
        body = r.json()
        return body, time.monotonic() - start, None
    payload["stream"] = True
    payload["stream_options"] = {"include_usage": True}
    first, usage, finish = None, None, None
    with client.stream("POST", base + "/v1/chat/completions", json=payload) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            if chunk.get("usage"):
                usage = chunk["usage"]
            for ch in chunk.get("choices", []):
                if first is None and (ch.get("delta") or {}).get("content"):
                    first = time.monotonic() - start
                finish = ch.get("finish_reason") or finish
    return {"usage": usage, "choices": [{"finish_reason": finish}]}, time.monotonic() - start, first


def cached(body):
    usage = body.get("usage") or {}
    return (usage.get("prompt_tokens_details") or {}).get("cached_tokens"), usage.get("prompt_tokens")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    base = args.base_url.rstrip("/")
    result = {"model": args.model, "base_url": base, "granularity": [], "ttft": [], "decode": None}
    with httpx.Client(timeout=600) as client:
        m0 = metrics(client, base)
        # 1. 命中粒度：同一前缀，后缀不同；记录第二次请求的 cached_tokens。
        for words in (200, 600, 1200, 2400, 4800):
            tag = uuid.uuid4().hex
            prefix = f"[{tag}] " + filler(tag, words)
            a, _, _ = chat(client, base, args.model, prefix + "\nQuestion A: reply OK.", 4)
            b, _, _ = chat(client, base, args.model, prefix + "\nQuestion B: reply OK.", 4)
            ca, pa = cached(a)
            cb, pb = cached(b)
            result["granularity"].append({"prefix_words": words, "first_prompt": pa, "first_cached": ca,
                                          "second_prompt": pb, "second_cached": cb})
        # 2. 冷启动与命中时的首 Token 时间（长前缀，各 3 次）。
        for rep in range(3):
            tag = uuid.uuid4().hex
            prefix = f"[{tag}] " + filler(tag, 4800)
            cold, total_c, ttft_c = chat(client, base, args.model, prefix + "\nSay hi.", 8, stream=True)
            warm, total_w, ttft_w = chat(client, base, args.model, prefix + "\nSay bye.", 8, stream=True)
            result["ttft"].append({"rep": rep, "cold_ttft": ttft_c, "warm_ttft": ttft_w,
                                   "cold_cached": cached(cold)[0], "warm_cached": cached(warm)[0],
                                   "prompt_tokens": cached(cold)[1]})
        # 3. 单流解码吞吐：要求长输出，记录 finish_reason 以确认没有被截断以外的提前结束。
        body, total, ttft = chat(client, base, args.model,
                                 "Write a detailed 600-word essay about operating system schedulers.", 1024, stream=True)
        out_tokens = (body.get("usage") or {}).get("completion_tokens")
        result["decode"] = {"completion_tokens": out_tokens, "seconds": total, "ttft": ttft,
                            "tokens_per_second": out_tokens / (total - ttft) if out_tokens and ttft else None,
                            "finish_reason": body["choices"][0]["finish_reason"]}
        m1 = metrics(client, base)
        result["metrics_delta"] = {k: (m1[k] - m0[k]) if m1[k] is not None and m0[k] is not None else None for k in m0}
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
