"""Selectors consume frozen plans, never gold except for the oracle arm."""
from __future__ import annotations
import math
import re
from collections import Counter
from dataclasses import replace
from agentos.context.manager import ContextManager
from .common import digest

def terms(text):
    return re.findall(r"[a-z0-9]+", text.lower().replace("_", " "))

def bm25(question, documents, goal, k):
    refs = list(question["candidate_refs"])
    corpus = [Counter(terms(documents[r]["name"] + "\n" + documents[r]["text"])) for r in refs]
    lengths = [sum(row.values()) for row in corpus]
    average = sum(lengths) / len(lengths) or 1
    query = set(terms(goal))
    df = {t: sum(t in row for row in corpus) for t in query}
    def score(i):
        row, length = corpus[i], lengths[i]
        return sum(math.log(1 + (len(corpus) - df[t] + .5) / (df[t] + .5))
                   * row[t] * 2.5 / (row[t] + 1.5 * (.25 + .75 * length / average))
                   for t in query if row[t])
    return sorted(refs, key=lambda ref: (-score(refs.index(ref)), ref))[:k]

def retrieval_budget(manifest, bundle, split):
    """Distribute floor/ceil k deterministically to match declared mean exactly."""
    entries = []
    for q in manifest["questions"]:
        if split != "all" and q["split"] != split:
            continue
        frozen = bundle["questions"].get(q["id"], {})
        if frozen.get("status") != "completed":
            continue
        for task in frozen["plan"]["readers"]:
            entries.append((q["id"], task["id"], len(task["context_refs"])))
    if not entries:
        return {}, {"readers": 0, "declared_references": 0, "mean_k": None}
    total = sum(row[2] for row in entries)
    base, remainder = divmod(total, len(entries))
    ordered = sorted(entries, key=lambda row: digest([bundle["config"]["seed"], row[0], row[1]]))
    budget = {(qid, tid): base + int(index < remainder)
              for index, (qid, tid, _) in enumerate(ordered)}
    return budget, {"readers": len(entries), "declared_references": total,
                    "mean_k": total / len(entries), "floor_k": base,
                    "ceil_k_readers": remainder, "rule": "hash_order_floor_ceil_exact_mean"}

def selected_documents(arm, question, documents, task=None, *, k=None):
    if arm == "all":
        selected = question["candidate_refs"]
    elif arm == "oracle":
        selected = question["required_refs"]
    elif task is None:
        selected = []
    elif arm == "declared":
        selected = task["context_refs"]
    elif arm == "bm25":
        if k is None:
            raise ValueError("BM25 requires a frozen retrieval budget")
        selected = bm25(question, documents, task["goal"], k)
    else:
        raise ValueError(f"unknown arm: {arm}")
    return sorted(set(selected))

class SelectionContextManager(ContextManager):
    """Same serialization/order in every arm; no cache feedback or compression."""
    def prepare_context(self, agent_id, user_messages, *, context_ref_names=None):
        if context_ref_names is None:
            raise ValueError("experiment tasks must declare explicit refs including control")
        prepared = super().prepare_context(agent_id, user_messages,
                                            context_ref_names=context_ref_names)
        return replace(prepared, selection_details={
            "selection": "explicit", "selected_refs": list(prepared.shared_refs),
            "prefix_order": self._prefix_order, "lossy_compression": False,
            "cache_feedback": False})
