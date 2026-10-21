"""Only oracle reads gold labels; both BM25 budgets are outcome-independent."""
from fractions import Fraction
from experiments.context_selection.policies import (SelectionContextManager,
    retrieval_budget as declared_budget, bm25 as _bm25)
from .common import digest


def bm25(question, documents, query, k):
    indexed = {r: {**documents[r], "name": documents[r]["name"] + "\n" + documents[r]["description"]}
               for r in question["candidate_refs"]}
    return _bm25(question, indexed, query, k)


def retrieval_budget(manifest, bundle, split, *, repeats=1, append_summary=None):
    primary, info = declared_budget(manifest, bundle, split)
    if append_summary is None:
        return {(q, t, r): k for (q, t), k in primary.items() for r in range(repeats)}, info
    source = append_summary
    if (source.get("arm") != "declared_append" or source.get("split") != split
            or source.get("manifest_sha256") != digest(manifest) or source.get("plans_sha256") != digest(bundle)
            or source["config"]["repeats"] != repeats):
        raise ValueError("secondary BM25 needs matching declared_append summary/split/repeats")
    for key in ("backend", "model", "base_url", "seed", "max_tokens", "temperature", "enable_thinking"):
        if source["config"].get(key) != bundle["config"].get(key):
            raise ValueError("secondary BM25 source configuration mismatch")
    expected = {(q, t, r) for q, t in primary for r in range(repeats)}
    entries = {}
    expected_rows = {(q["id"], r) for q in manifest["questions"] if split == "all" or q["split"] == split
                     for r in range(repeats)}
    seen_rows = set()
    for row in source["questions"]:
        row_key = (row["question_id"], row["replicate"])
        if row_key in seen_rows or row_key not in expected_rows:
            raise ValueError("invalid append question/replicate grid")
        seen_rows.add(row_key)
        # Shared planner failures have no executable readers and consume no
        # retrieval budget. They still remain failures in all arm summaries.
        if bundle["questions"][row["question_id"]]["status"] != "completed":
            if row["model_calls"] or row.get("final_documents"):
                raise ValueError("failed plan unexpectedly executed readers")
            continue
        # 探测后登记：执行无效（截断或 reader 失败）的题仍按其 reader 实际最终文档数计入预算，
        # 这些题在各组都单独计为无效，不影响预算与结果无关的性质。
        for tid, refs in row["final_documents"].items():
            key = (row["question_id"], tid, row["replicate"])
            if key in entries or key not in expected or len(refs) != len(set(refs)):
                raise ValueError("invalid append observation grid")
            q = next(q for q in manifest["questions"] if q["id"] == key[0])
            if not set(refs) <= set(q["candidate_refs"]):
                raise ValueError("append final refs outside candidates")
            entries[key] = len(refs)
    if seen_rows != expected_rows or set(entries) != expected or not entries:
        raise ValueError("missing append final reader document counts")
    total = sum(entries.values())
    base, remainder = divmod(total, len(entries))
    order = sorted(entries, key=lambda row: digest([bundle["config"]["seed"], *row]))
    return {key: base + int(i < remainder) for i, key in enumerate(order)}, {
        "readers": len(entries), "final_references": total, "mean_k": total / len(entries),
        "mean_k_fraction": str(Fraction(total, len(entries))), "floor_k": base,
        "ceil_k_readers": remainder, "source_sha256": digest(source),
        "rule": "hash_order_floor_ceil_exact_mean_over_reader_replicates"}


def selected_documents(arm, question, documents, task=None, *, k=None):
    if task is None:  # Every answerer only receives result references.
        return []
    if arm == "all":
        refs = question["candidate_refs"]
    elif arm == "oracle":
        refs = question["required_refs"]
    elif arm in ("declared", "declared_append"):
        refs = task["context_refs"]
    elif arm in ("bm25", "bm25_append"):
        if k is None:
            raise ValueError("BM25 requires a frozen budget")
        refs = bm25(question, documents, question["prompt"] + "\n" + task["goal"], k)
    else:
        raise ValueError(f"unknown arm: {arm}")
    return sorted(set(refs))
