"""Deterministic stdlib AST questions with neutral candidate sampling."""
from __future__ import annotations
import argparse
import ast
from collections import Counter
import json
from pathlib import Path
import random
import re
import sys
import tokenize
from experiments.context_selection.generate import (
    collect_documents, _function_fact, _module_name, _EXCLUDED, DEFAULT_SEED)
from .common import APPEND_LIMITS, MANIFEST_VERSION, write_new


def first_sentence(node):
    doc = ast.get_docstring(node) or ""
    return re.split(r"(?<=[.!?])\s+", " ".join(doc.split()), maxsplit=1)[0]


def document_functions(document):
    """Keep original indentation (including zero-indented multiline strings)."""
    scope = document["definition_scope"]
    text = document["text"]
    if scope:
        indent = len(text) - len(text.lstrip(" "))
        tree = ast.parse("class Frozen:\n" + " " * indent + "pass\n" + text)
        body = tree.body[0].body[1:]
    else:
        body = ast.parse(text).body
    def visit(body, prefix):
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                yield _function_fact(node, document["module"], prefix), node
            elif isinstance(node, ast.ClassDef):
                yield from visit(node.body, prefix + node.name + ".")
    return list(visit(body, scope))


def direct_names(node):
    """Syntactic Name calls in this function body, excluding nested scopes.

    Local bindings/parameters exclude shadowed global names. No runtime code is
    imported or evaluated; dynamic dispatch/aliases are outside this question.
    """
    names, bound = set(), {a.arg for a in ast.walk(node.args) if isinstance(a, ast.arg)}
    class Calls(ast.NodeVisitor):
        def visit_Call(self, item):
            if isinstance(item.func, ast.Name):
                names.add(item.func.id)
            self.generic_visit(item)
        def visit_Name(self, item):
            if isinstance(item.ctx, (ast.Store, ast.Del)):
                bound.add(item.id)
        def visit_FunctionDef(self, item):
            bound.add(item.name)
        visit_AsyncFunctionDef = visit_FunctionDef
        def visit_ClassDef(self, item):
            bound.add(item.name)
        def visit_Lambda(self, item):
            pass
        def visit_Import(self, item):
            bound.update(a.asname or a.name.split(".")[0] for a in item.names)
        def visit_ImportFrom(self, item):
            bound.update(a.asname or a.name for a in item.names)
        def visit_ExceptHandler(self, item):
            if item.name:
                bound.add(item.name)
            self.generic_visit(item)
    visitor = Calls()
    for stmt in node.body:
        visitor.visit(stmt)
    return names - bound


def symbol_leak(prompt, refs, documents):
    words = set(re.findall(r"\w+", prompt.casefold()))
    return any(s.rsplit(".", 1)[-1].casefold() in words
               for ref in refs for s in documents[ref]["symbols"])


def neutral_candidates(documents, required, rng, count=8):
    """Uniform without replacement within module, package, then corpus tiers."""
    chosen = list(dict.fromkeys(required))
    modules = {documents[r]["module"] for r in required}
    packages = {m.split(".")[0] for m in modules}
    for tier in range(3):
        pool = [ref for ref, doc in sorted(documents.items()) if ref not in chosen and (
            (doc["module"] in modules) if tier == 0 else
            (doc["module"].split(".")[0] in packages) if tier == 1 else True)]
        chosen.extend(rng.sample(pool, min(count - len(chosen), len(pool))))
        if len(chosen) == count:
            break
    if len(chosen) != count:
        raise ValueError("insufficient candidate documents")
    rng.shuffle(chosen)
    return chosen


def derive_answer(question, documents):
    """Re-derive from frozen text only (no answer/fact metadata, no source files)."""
    facts = {f["symbol"]: (f, n, ref) for ref in question["required_refs"]
             for f, n in document_functions(documents[ref])}
    if question["family"] == "hop":
        # Caller is explicitly present in the question, never read callee labels.
        caller = re.search(r"`([^`]+)`", question["prompt"]).group(1)
        _, node, source_ref = facts[caller]
        calls = direct_names(node)
        found = [(f, n, r) for f, n, r in facts.values() if f["kind"] == "function"
                 and r != source_ref and f["symbol"].rsplit(".", 1)[0] == caller.rsplit(".", 1)[0]
                 and n.name in calls]
        if len(found) != 1:
            raise ValueError("hop must have one distinct cross-document callee")
        fact = found[0][0]
        default = fact["first_default"]
        return {"callee": fact["symbol"], "parameter_count": fact["parameter_count"],
                "parameter": default[0] if default else None, "default": default[1] if default else None}
    sentences = question["descriptions"]
    targets = [f for f, n, _ in facts.values() if first_sentence(n) in sentences]
    if len(targets) != 2:
        raise ValueError("describe must identify exactly two functions")
    targets.sort(key=lambda f: f["parameter_count"], reverse=True)
    winner = targets[0]
    if winner["parameter_count"] == targets[1]["parameter_count"] or not winner["first_default"]:
        raise ValueError("describe requires unique maximum with a default")
    return {"winner": winner["symbol"], "parameter": winner["first_default"][0],
            "default": winner["first_default"][1]}


def build_manifest(source_root=Path("/usr/lib/python3.11"), seed=DEFAULT_SEED,
                   n_questions=220, probe_size=20, candidates=8):
    if n_questions < 2 or n_questions % 2 or probe_size % 2 or not 0 <= probe_size <= n_questions or candidates != 8:
        raise ValueError("even sizes required, exactly eight candidates")
    root = Path(source_root).resolve()
    documents, raw_facts = collect_documents(root)
    symbol_counts = Counter(f["symbol"] for f in raw_facts)
    # Count docstrings and top-level definitions across ALL source, including
    # functions omitted by document-length filters.
    sentences, globals_by_module = Counter(), {}
    for path in sorted(root.rglob("*.py")):
        if any(p in _EXCLUDED for p in path.relative_to(root).parts):
            continue
        try:
            with tokenize.open(path) as stream:
                tree = ast.parse(stream.read())
        except (OSError, SyntaxError, UnicodeError):
            continue
        sentences.update(first_sentence(n) for n in ast.walk(tree)
                         if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)))
        globals_by_module[_module_name(path, root)] = Counter(
            n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)))
    facts = {}
    for ref, doc in documents.items():
        for fact, node in document_functions(doc):
            if symbol_counts[fact["symbol"]] == 1:
                facts[fact["symbol"]] = {**fact, "node": node, "doc_id": ref, "module": doc["module"]}
    hops, descriptions = [], []
    for fact in facts.values():
        node, module = fact["node"], fact["module"]
        if fact["kind"] == "function":
            globals_ = globals_by_module[module]
            # Count every directly named module-level function; require exactly
            # one other function, which must have a unique frozen definition.
            calls = (direct_names(node) & globals_.keys()) - {node.name}
            if len(calls) == 1:
                name = next(iter(calls))
                target = facts.get(module + "." + name)
                if target and globals_[name] == 1 and globals_[node.name] == 1 and target["doc_id"] != fact["doc_id"]:
                    if not target["first_default"] or len(target["first_default"][1]) <= 240:
                        hops.append((fact, target))
        sentence = first_sentence(node)
        if (sentences[sentence] == 1 and len(re.findall(r"\b[A-Za-z]+\b", sentence)) >= 5
                and len(sentence) <= 500):
            descriptions.append((fact, sentence))
    rng = random.Random(seed)
    # Reject callee leaves matching the named caller/module or output schema.
    hops = [(caller, target) for caller, target in hops
            if target["node"].name.casefold() not in (
                caller["symbol"].casefold() + " python callee parameter_count parameter default null args kwargs")]
    rng.shuffle(hops)
    if len(hops) < n_questions // 2:
        raise ValueError(f"only {len(hops)} eligible hops")
    questions, selected, used_pairs = [], set(), set()
    for index in range(n_questions):
        if index % 2 == 0:
            caller, target = hops[index // 2]
            prompt = (f"检查 Python 标准库函数 `{caller['symbol']}` 的函数体：它以裸名称直接调用的另一个同模块模块级函数是哪一个？"
                      "保证除自身外恰好一个，且定义位于另一文档段；不计嵌套函数体、装饰器和签名中的调用，不追踪别名或动态调用。"
                      "请给出被调函数完整限定名、形参数量、从左到右第一个带默认值的形参及默认值表达式；没有默认值时后两项填 null。"
                      '形参数量含位置、仅限关键字、*args、**kwargs。输出 {"callee":"限定名","parameter_count":0,"parameter":null,"default":null}。')
            if target["node"].name.casefold() in prompt.casefold():
                raise ValueError("callee name leaked into hop prompt")
            required = [caller["doc_id"], target["doc_id"]]
            refs = neutral_candidates(documents, required, rng, candidates)
            extra = {"targets": [{"symbol": f["symbol"], "doc_id": f["doc_id"]} for f in (caller, target)]}
            family = "hop"
        else:
            for _ in range(100000):
                pair = rng.sample(descriptions, 2)
                fs = [p[0] for p in pair]
                identity = tuple(sorted(f["symbol"] for f in fs))
                ranked = sorted(fs, key=lambda f: f["parameter_count"], reverse=True)
                default = ranked[0]["first_default"]
                if (identity in used_pairs or fs[0]["doc_id"] == fs[1]["doc_id"]
                        or ranked[0]["parameter_count"] == ranked[1]["parameter_count"]
                        or not default or len(default[1]) > 240):
                    continue
                prompt = ("下面是 Python 标准库两个函数或方法的 docstring 首句（分别唯一）：\n"
                          f"描述 A：{pair[0][1]}\n描述 B：{pair[1][1]}\n"
                          "比较对应定义，输出形参数量唯一最多者的完整限定名、从左到右第一个带默认值的形参及默认值表达式。"
                          "计数包含位置、仅限关键字、*args、**kwargs，方法排除 self/cls；保证最多者有默认值。"
                          '输出 {"winner":"限定名","parameter":"形参名","default":"Python 表达式"}。')
                required = [f["doc_id"] for f in fs]
                refs = neutral_candidates(documents, required, rng, candidates)
                if not symbol_leak(prompt, refs, documents):
                    break
            else:
                raise ValueError("insufficient describe pairs without symbol leaks")
            used_pairs.add(identity)
            extra = {"descriptions": [p[1] for p in pair],
                     "targets": [{"symbol": f["symbol"], "doc_id": f["doc_id"]} for f in fs]}
            family = "describe"
        q = {"id": f"q{index+1:03d}", "split": "probe" if index < probe_size else "formal",
             "family": family, "prompt": prompt, "required_refs": required, "candidate_refs": refs, **extra}
        q["answer"] = derive_answer(q, documents)
        questions.append(q)
        selected.update(refs)
    return {"schema_version": 2, "seed": seed, "source_root": str(root),
            "generator": {"version": 2, "python_version": sys.version.split()[0], "eligible_hops": len(hops),
                          "eligible_descriptions": len(descriptions)},
            "protocol": {"version": MANIFEST_VERSION, "questions": n_questions, "probe_questions": probe_size,
                         "formal_questions": n_questions-probe_size, "candidates_per_question": candidates,
                         "families": "half hop, half describe in each split; 2 required documents each",
                         "distractors": "uniform without replacement: target modules, then top-level packages, then corpus; seeded shuffle; reject describe pairs if any candidate leaf symbol occurs as a case-insensitive word in prompt",
                         "hop": "one distinct bare-name module-level callee excluding self and nested scopes; reject local shadowing; allow no default (null)",
                         "describe": "first sentence unique across all parsed stdlib function docstrings, >=5 alphabetic words; no candidate leaf symbol as a case-insensitive word; unique maximum with default <=240 chars",
                         "append": APPEND_LIMITS, "reader_max_calls": 3,
                         "answerer_source_documents": False, "fences": "strip exactly one enclosing Markdown JSON fence for all roles"},
            "documents": {r: documents[r] for r in sorted(selected)}, "questions": questions}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path("/usr/lib/python3.11"))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--questions", type=int, default=220)
    parser.add_argument("--probe-size", type=int, default=20)
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("manifest.json"))
    args = parser.parse_args()
    manifest = build_manifest(args.source_root, args.seed, args.questions, args.probe_size, args.candidates)
    write_new(args.output, manifest)
    print({"questions": len(manifest["questions"]), "documents": len(manifest["documents"]), **manifest["generator"]})

if __name__ == "__main__":
    main()
