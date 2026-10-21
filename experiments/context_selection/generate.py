"""Freeze AST-scored questions from local stdlib source without executing it."""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from itertools import groupby
import hashlib
import io
import json
import math
from pathlib import Path
import random
import re
import sys
import sysconfig
import tokenize
from typing import Any

SCHEMA_VERSION = 1
DEFAULT_SEED = 20261019
MIN_CHARS = 4000
MAX_CHARS = 12000
_EXCLUDED = {"test", "tests", "site-packages", "dist-packages", "__pycache__", "idle_test", "lib2to3"}


def _module_name(path: Path, root: Path) -> str:
    parts = list(path.relative_to(root).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _start(node: ast.AST) -> int:
    return min([node.lineno] + [item.lineno for item in getattr(node, "decorator_list", [])])


def _function_fact(node: ast.FunctionDef | ast.AsyncFunctionDef, module: str, prefix: str = "") -> dict[str, Any]:
    positional = [*node.args.posonlyargs, *node.args.args]
    excluded = {"self", "cls"} if prefix else set()
    count = sum(arg.arg not in excluded for arg in [*positional, *node.args.kwonlyargs])
    count += int(node.args.vararg is not None) + int(node.args.kwarg is not None)
    defaults: list[tuple[str, str]] = []
    for arg, default in zip(positional[len(positional) - len(node.args.defaults):], node.args.defaults):
        if arg.arg not in excluded:
            defaults.append((arg.arg, ast.unparse(default)))
    for arg, default in zip(node.args.kwonlyargs, node.args.kw_defaults):
        if default is not None and arg.arg not in excluded:
            defaults.append((arg.arg, ast.unparse(default)))
    return {"symbol": f"{module}.{prefix}{node.name}", "kind": "method" if prefix else "function",
            "parameter_count": count, "first_default": list(defaults[0]) if defaults else None}


def _facts(node: ast.AST, module: str, prefix: str = "") -> list[dict[str, Any]]:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return [_function_fact(node, module, prefix)]
    if isinstance(node, ast.ClassDef):
        result = []
        for child in node.body:
            result.extend(_facts(child, module, f"{prefix}{node.name}."))
        return result
    return []


def _blocks(body: list[ast.stmt], lines: list[str], module: str, prefix: str = "") -> list[dict[str, Any]]:
    """Keep definitions whole; oversized classes split at member boundaries.

    Oversized individual functions and undersized groups are omitted. Adjacent
    complete definitions may share a document, retaining intervening source.
    """
    result: list[dict[str, Any]] = []
    for node in body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        start, end = _start(node), node.end_lineno
        char_count = len("".join(lines[start - 1:end]))
        if char_count > MAX_CHARS:
            if isinstance(node, ast.ClassDef):
                result.extend(_blocks(node.body, lines, module, f"{prefix}{node.name}."))
            continue
        facts = _facts(node, module, prefix)
        if facts:
            result.append({"start": start, "end": end, "facts": facts, "scope": prefix})
    return result


def _group_scope(blocks: list[dict[str, Any]], lines: list[str]) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for block in blocks:
        if current is not None:
            combined_chars = len("".join(lines[current["start"] - 1:block["end"]]))
            if combined_chars > MAX_CHARS:
                if len("".join(lines[current["start"] - 1:current["end"]])) >= MIN_CHARS:
                    groups.append(current)
                current = None
        if current is None:
            current = {"start": block["start"], "end": block["end"], "facts": list(block["facts"]), "scope": block["scope"]}
        else:
            current["end"] = block["end"]
            current["facts"].extend(block["facts"])
        if len("".join(lines[current["start"] - 1:current["end"]])) >= MIN_CHARS:
            groups.append(current)
            current = None
    if current is not None:
        if groups and len("".join(lines[groups[-1]["start"] - 1:current["end"]])) <= MAX_CHARS:
            groups[-1]["end"] = current["end"]
            groups[-1]["facts"].extend(current["facts"])
        elif len("".join(lines[current["start"] - 1:current["end"]])) >= MIN_CHARS:
            groups.append(current)
    return groups


def _group_blocks(blocks: list[dict[str, Any]], lines: list[str]) -> list[dict[str, Any]]:
    # Preserve syntax after dedenting; never join a class member with a module definition.
    result = []
    for _, scoped_blocks in groupby(blocks, key=lambda block: block["scope"]):
        result.extend(_group_scope(list(scoped_blocks), lines))
    return result


def collect_documents(source_root: Path | str) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    root = Path(source_root).resolve()
    documents: dict[str, dict[str, Any]] = {}
    facts: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*.py")):
        if any(part in _EXCLUDED for part in path.relative_to(root).parts):
            continue
        module = _module_name(path, root)
        if not module:
            continue
        try:
            raw = path.read_bytes()
            with tokenize.open(path) as handle:
                source = handle.read()
            tree = ast.parse(source, filename=str(path))
        except (OSError, SyntaxError, UnicodeError):
            continue
        lines = io.StringIO(source).readlines()  # AST line numbers count LF, not form-feed.
        source_sha = hashlib.sha256(raw).hexdigest()
        for group in _group_blocks(_blocks(tree.body, lines, module), lines):
            text = "".join(lines[group["start"] - 1:group["end"]])
            digest = hashlib.sha256(f"{path.relative_to(root)}:{source_sha}:{group['start']}:{group['end']}".encode()).hexdigest()[:16]
            doc_id = f"doc_{digest}"
            symbols = [fact["symbol"] for fact in group["facts"]]
            documents[doc_id] = {
                "id": doc_id, "name": f"{module}:{group['start']}-{group['end']}",
                "description": "Python 标准库定义：" + ", ".join(symbols), "text": text,
                "estimated_tokens": math.ceil(len(text) / 4), "char_count": len(text),
                "token_estimation": "ceil(unicode_characters / 4)",
                "source_path": path.relative_to(root).as_posix(), "source_sha256": source_sha,
                "start_line": group["start"], "end_line": group["end"], "module": module,
                "definition_scope": group["scope"],
                "symbols": symbols, "kinds": sorted({fact["kind"] for fact in group["facts"]}),
            }
            facts.extend({**fact, "doc_id": doc_id, "module": module} for fact in group["facts"])
    return documents, facts


def _similarity(document: dict[str, Any], targets: list[dict[str, Any]]) -> tuple[int, int, int]:
    modules = {target["module"] for target in targets}
    kinds = {target["kind"] for target in targets}
    words = set(re.findall(r"[a-z]+", " ".join(t["symbol"].lower() for t in targets)))
    doc_words = set(re.findall(r"[a-z]+", " ".join(document["symbols"]).lower()))
    return (int(document["module"] in modules), int(bool(kinds.intersection(document["kinds"]))), len(words.intersection(doc_words)))


def build_manifest(source_root: Path | str | None = None, seed: int = DEFAULT_SEED,
                   n_questions: int = 120, probe_size: int = 20, candidates: int = 8) -> dict[str, Any]:
    if not 0 <= probe_size <= n_questions or n_questions < 1 or candidates < 3:
        raise ValueError("需要 n_questions>=1、0<=probe_size<=n_questions、candidates>=3")
    root = Path(source_root or sysconfig.get_path("stdlib")).resolve()
    documents, facts = collect_documents(root)
    symbol_counts = Counter(fact["symbol"] for fact in facts)
    facts = [fact for fact in facts if symbol_counts[fact["symbol"]] == 1]
    if len(documents) < candidates:
        raise ValueError(f"只有 {len(documents)} 个合格文档，候选数量要求 {candidates}")
    if not facts:
        raise ValueError("没有可用函数签名")
    rng = random.Random(seed)
    questions: list[dict[str, Any]] = []
    used: set[tuple[str, ...]] = set()
    selected_documents: set[str] = set()
    for index in range(n_questions):
        n_targets = 2 + index % 2
        for _ in range(20000):
            targets = rng.sample(facts, n_targets)
            identity = tuple(sorted(target["symbol"] for target in targets))
            ranked = sorted(targets, key=lambda item: item["parameter_count"], reverse=True)
            winner = ranked[0]
            default = winner["first_default"]
            if (len({target["doc_id"] for target in targets}) == n_targets
                    and identity not in used and winner["parameter_count"] > ranked[1]["parameter_count"]
                    and default is not None and len(default[1]) <= 240):
                break
        else:
            raise ValueError("无法生成足够的不同问题；请提供更大的源码语料")
        used.add(identity)
        required = [target["doc_id"] for target in targets]
        distractors = [doc for doc in documents.values() if doc["id"] not in required]
        rng.shuffle(distractors)
        distractors.sort(key=lambda doc: _similarity(doc, targets), reverse=True)
        refs = required + [doc["id"] for doc in distractors[:candidates - n_targets]]
        rng.shuffle(refs)
        selected_documents.update(refs)
        symbols = "、".join(f"`{target['symbol']}`" for target in targets)
        prompt = (
            f"比较 Python 标准库中 {symbols} 的定义：哪个函数或方法的形参数量最多？"
            "再给出它按签名从左到右第一个具有默认值的形参名及默认值表达式。"
            "形参数量包括仅限位置形参、普通形参、仅限关键字形参，以及 *args 和 **kwargs（各算一个）；"
            "方法中的 self 和 cls 不计入。保证最多者唯一且至少有一个默认值。"
            "只输出严格 JSON：{\"winner\":\"完整限定符号名\",\"parameter\":\"形参名\",\"default\":\"Python 表达式\"}。"
        )
        questions.append({
            "id": f"q{index + 1:03d}", "split": "probe" if index < probe_size else "formal",
            "family": f"compare_{n_targets}_signatures", "prompt": prompt,
            "candidate_refs": refs, "required_refs": required,
            "targets": [{"symbol": target["symbol"], "doc_id": target["doc_id"]} for target in targets],
            "answer": {"winner": winner["symbol"], "parameter": default[0], "default": default[1]},
        })
    return {
        "schema_version": SCHEMA_VERSION, "seed": seed,
        "generator": {"version": 1, "python_version": sys.version.split()[0], "ast": "stdlib source; never imported/executed"},
        "source_root": str(root),
        "token_estimation": {"method": "ceil(unicode_characters / 4)", "target_min_tokens": 1000, "target_max_tokens": 3000},
        "protocol": {
            "questions": n_questions, "probe_questions": probe_size, "formal_questions": n_questions - probe_size,
            "candidates_per_question": candidates, "required_documents": [2, 3],
            "parameter_count": "posonly + positional + kwonly + vararg + kwarg; omit self/cls for methods",
            "distractors": "rank by same module, same definition kind, shared name words; seeded tie shuffle",
            "answer_fields": ["winner", "parameter", "default"],
            "target_filter": "exclude repeated qualified symbols; unique maximum with default expression <=240 characters",
        },
        "documents": {ref: documents[ref] for ref in sorted(selected_documents)}, "questions": questions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--questions", type=int, default=120)
    parser.add_argument("--probe-size", type=int, default=20)
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("manifest.json"))
    parser.add_argument("--overwrite", action="store_true", help="explicitly replace an existing manifest")
    args = parser.parse_args()
    manifest = build_manifest(args.source_root, args.seed, args.questions, args.probe_size, args.candidates)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with args.output.open("w" if args.overwrite else "x", encoding="utf-8") as handle:
            handle.write(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    except FileExistsError:
        parser.error(f"manifest already exists: {args.output}; use a new path or explicit --overwrite")
    print(json.dumps({"output": str(args.output), "questions": len(manifest["questions"]), "documents": len(manifest["documents"]), "seed": args.seed}, ensure_ascii=False))


if __name__ == "__main__":
    main()
