#!/usr/bin/env python3
"""Task 14: 从 Multi-SWE-bench 抽取非 Python 多语言任务，构造成与任务 10 完全同构的
honest_bench 工作负载文件（tasks_multi_swe_*.json）。

与 prepare_verified_tasks.py（任务 10）一致的 shared_context 构造思路：
  shared_context = <issue>{官方 resolved_issues 文本}</issue>
                   <code>{fix_patch 触碰的 gold 源文件在 base commit 的完整内容}</code>
即"问题 + 金标源文件"，和 SWE-bench oracle 的 text 字段同构。gold 文件通过 GitHub raw
在 base sha 上取（等价于 oracle 的金标文件），语言无关。

抽样标准（写入产物，与结果无关）：
  - 每个仓库/语言按数据集原顺序遍历，取前若干个"能干净取到 gold 文件、且 shared_context
    落在 token 窗口 [3000,12000] 内"的实例（窗口与任务 10 一致，保证共享上下文规模可比、
    成本适中）；每仓库上限 PER_REPO，语言尽量均衡。
  - 不按难度/结果挑选。
"""
from __future__ import annotations

import glob
import json
import re
import sys
import urllib.request
from pathlib import Path

MIN_TOK, MAX_TOK = 3000, 12000
PER_REPO = 3
MAX_SCAN = 25          # 每仓库最多扫描的实例数（数据集原顺序），够找到 PER_REPO 个即可
MAX_FILE_CHARS = 80000  # 单个 gold 文件超此长度直接跳过（避免单头文件类超大文件拖垮抽样）
OUT = Path("runs/swe_bench/tasks_multi_swe_15.json")

# 语言 -> 数据集 jsonl（已 sparse 下载到 _hf 缓存）。每语言一个代表性仓库。
# cpp 用 fmtlib/fmt（正常多文件结构；nlohmann/json 是单头库、gold 文件~900KB 不适用）。
REPO_FILES = {
    "go": "cli__cli_dataset.jsonl",
    "java": "google__gson_dataset.jsonl",
    "rust": "clap-rs__clap_dataset.jsonl",
    "cpp": "fmtlib__fmt_dataset.jsonl",
    "c": "jqlang__jq_dataset.jsonl",
    "javascript": "axios__axios_dataset.jsonl",
}
HF_ROOT = "runs/external_data/multi_swe_bench/_hf"


def fetch_raw(org: str, repo: str, sha: str, path: str) -> str | None:
    url = f"https://raw.githubusercontent.com/{org}/{repo}/{sha}/{path}"
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            return r.read().decode("utf-8", "replace")
    except Exception:
        return None


def build_context(inst: dict) -> tuple[str, int] | None:
    base = inst["base"] if isinstance(inst["base"], dict) else json.loads(inst["base"])
    sha = base["sha"]
    org, repo = inst["org"], inst["repo"]
    ri = inst["resolved_issues"]
    if isinstance(ri, str):
        ri = json.loads(ri)
    issue = "\n\n".join(
        ((x.get("title") or "") + "\n" + (x.get("body") or "")).strip() for x in ri
    ).strip()
    if not issue:
        issue = (inst.get("title") or "") + "\n" + (inst.get("body") or "")
    files = re.findall(r"diff --git a/(\S+) b/(\S+)", inst.get("fix_patch", ""))
    blobs: list[str] = []
    for a_path, _ in files:
        content = fetch_raw(org, repo, sha, a_path)
        if content is None:  # 新增文件在 base 不存在 / 取不到，跳过该文件
            continue
        if len(content) > MAX_FILE_CHARS:  # 超大单文件（如单头库）跳过，避免规模失衡
            continue
        blobs.append(f"[start of {a_path}]\n{content}\n[end of {a_path}]")
    if not blobs:  # 没有任何 gold 文件可用 → 放弃该任务
        return None
    code = "\n\n".join(blobs)
    shared = f"<issue>\n{issue}\n</issue>\n<code>\n{code}\n</code>"
    return shared, len(shared) // 4


def main() -> None:
    tasks: list[dict] = []
    alloc: dict[str, int] = {}
    for lang, fname in REPO_FILES.items():
        matches = glob.glob(f"{HF_ROOT}/**/{fname}", recursive=True)
        if not matches:
            print(f"[skip] {lang}: {fname} not downloaded", file=sys.stderr)
            continue
        kept = 0
        scanned = 0
        with open(matches[0], encoding="utf-8") as fh:  # 流式读，避免载入整个 167MB 文件
            for ln in fh:
                if kept >= PER_REPO or scanned >= MAX_SCAN:
                    break
                scanned += 1
                inst = json.loads(ln)
                if not inst.get("fix_patch"):
                    continue
                built = build_context(inst)
                if built is None:
                    continue
                shared, est = built
                if not (MIN_TOK <= est <= MAX_TOK):
                    continue
                base = inst["base"] if isinstance(inst["base"], dict) else json.loads(inst["base"])
                tasks.append({
                    "instance_id": inst["instance_id"],
                    "repo": f"{inst['org']}/{inst['repo']}",
                    "base_commit": base["sha"],
                    "language": lang,
                    "problem_statement": (inst.get("title") or "")[:500],
                    "shared_context": shared,
                    "est_context_tokens": est,
                    "difficulty": "n/a",
                })
                kept += 1
                print(f"    [{lang}] {inst['instance_id']:32s} ~{est:>6} tok", flush=True)
        alloc[lang] = kept
        print(f"[{lang}] kept {kept}/{PER_REPO} after scanning {scanned}", flush=True)

    tasks.sort(key=lambda t: (t["language"], t["instance_id"]))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "source": "bytedance-research/Multi-SWE-bench (per-repo dataset jsonl); "
                  "gold files fetched from GitHub raw at base sha",
        "note": "Real multi-language coding tasks used as a workload; NOT scored for correctness. "
                "Same honest_bench pipeline as task 10, only the task source changes.",
        "sampling_standard": (f"non-Python only; token window [{MIN_TOK},{MAX_TOK}] est tokens "
                              "(same as Verified batch); first {PER_REPO}/repo in dataset order that "
                              "fetch cleanly and land in window; results-independent."),
        "n_tasks": len(tasks),
        "token_window": [MIN_TOK, MAX_TOK],
        "language_allocation": alloc,
        "tasks": tasks,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[saved] {OUT}  ({len(tasks)} tasks)  langs={alloc}")


if __name__ == "__main__":
    main()
