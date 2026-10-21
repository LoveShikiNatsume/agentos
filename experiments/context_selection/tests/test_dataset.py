from __future__ import annotations

import ast
import json
import textwrap
from pathlib import Path

import pytest

from experiments.context_selection.generate import build_manifest, collect_documents
from experiments.context_selection.scoring import normalize_answer, score_answer


@pytest.fixture()
def source_root(tmp_path: Path) -> Path:
    for module in range(12):
        definitions = []
        for index in range(5):
            arguments = [f"arg{i}" for i in range(index + 1)]
            arguments[-1] += f"={index}"
            definitions.append(f"def fn{index}({', '.join(arguments)}):\n    '''{'stdlib fixture ' * 320}'''\n    return arg0\n")
        (tmp_path / f"module{module}.py").write_text("\n".join(definitions), encoding="utf-8")
    return tmp_path


def test_generation_is_deterministic_and_seeded(source_root: Path) -> None:
    first = build_manifest(source_root, seed=7, n_questions=12, probe_size=4)
    assert first == build_manifest(source_root, seed=7, n_questions=12, probe_size=4)
    assert first != build_manifest(source_root, seed=8, n_questions=12, probe_size=4)
    assert len(first["questions"]) == 12
    assert [q["split"] for q in first["questions"]].count("probe") == 4
    assert [len(q["required_refs"]) for q in first["questions"]] == [2, 3] * 6
    for question in first["questions"]:
        assert len(set(question["candidate_refs"])) == 8
        assert set(question["required_refs"]) <= set(question["candidate_refs"])
        assert len({target["doc_id"] for target in question["targets"]}) == len(question["required_refs"])
        assert score_answer(question["answer"], question["answer"])
        winner = max(question["targets"], key=lambda item: int(item["symbol"].split("fn")[-1]))
        index = int(winner["symbol"].split("fn")[-1])
        assert question["answer"] == {"winner": winner["symbol"], "parameter": f"arg{index}", "default": str(index)}
        assert all(set(target) == {"symbol", "doc_id"} for target in question["targets"])
    for document in first["documents"].values():
        assert 1000 <= document["estimated_tokens"] <= 3000
        assert document["estimated_tokens"] == (document["char_count"] + 3) // 4
        assert len(document["source_sha256"]) == 64
        source = (source_root / document["source_path"]).read_text().splitlines(keepends=True)
        assert document["text"] == "".join(source[document["start_line"] - 1:document["end_line"]])
        ast.parse(document["text"])


def test_signature_semantics_and_large_class_splitting(tmp_path: Path) -> None:
    filler = "source text " * 380
    source = "class Example:\n"
    source += f"    def first(self, a, /, b=1, *args, c=2, **kwargs):\n        '''{filler}'''\n        pass\n"
    source += f"    def second(cls, x=3):\n        '''{filler}'''\n        pass\n"
    source += f"    def third(self, x, y=4):\n        '''{filler}'''\n        pass\n"
    (tmp_path / "sample.py").write_text(source)
    documents, facts = collect_documents(tmp_path)
    assert len(documents) == 3
    first = next(fact for fact in facts if fact["symbol"] == "sample.Example.first")
    assert first["parameter_count"] == 5
    assert first["first_default"] == ["b", "1"]
    assert all(doc["kinds"] == ["method"] for doc in documents.values())
    for doc in documents.values():
        ast.parse(textwrap.dedent(doc["text"]))


@pytest.mark.parametrize("default,normalized", [
    (' "hello" ', "'hello'"), ("((1 + 2))", "1 + 2"),
    ("{ 'a' : [1,2] }", "{'a': [1, 2]}"), ("0x10", "16"),
])
def test_scoring_normalizes_expression(default: str, normalized: str) -> None:
    expected = {"winner": "module.func", "parameter": "option", "default": normalized}
    raw = json.dumps({"default": default, "parameter": " option ", "winner": " module.func "})
    assert normalize_answer(raw) == expected
    assert score_answer(raw, expected)


@pytest.mark.parametrize("raw", [
    'prefix {"winner":"module.func","parameter":"x","default":"1"}',
    '```json\n{"winner":"module.func","parameter":"x","default":"1"}\n```',
    '{"winner":"module.func","parameter":"x","default":1}',
    '{"winner":"module.func","parameter":"x","default":"1","extra":"x"}',
    '{"winner":"module.func","winner":"other.func","parameter":"x","default":"1"}',
    '{"winner":"func","parameter":"x","default":"1"}',
    '{"winner":"module.func","parameter":"x","default":"x = 1"}',
    '[]', 'null',
])
def test_scoring_rejects_malformed_answers(raw: str) -> None:
    expected = {"winner": "module.func", "parameter": "x", "default": "1"}
    assert not score_answer(raw, expected)
    with pytest.raises(ValueError):
        normalize_answer(raw)


def test_scoring_preserves_identifiers_and_expression_structure() -> None:
    expected = {"winner": "module.func", "parameter": "x", "default": "1"}
    assert not score_answer({**expected, "winner": "module.Func"}, expected)
    assert not score_answer({**expected, "parameter": "other"}, expected)
    assert not score_answer({**expected, "default": "1.0"}, expected)
    assert not score_answer({**expected, "default": "2"}, expected)


def test_frozen_manifest_contract() -> None:
    manifest = json.loads((Path(__file__).parents[1] / "manifest.json").read_text())
    assert len(manifest["questions"]) == 120
    assert sum(q["split"] == "probe" for q in manifest["questions"]) == 20
    assert sum(q["split"] == "formal" for q in manifest["questions"]) == 100
    for document in manifest["documents"].values():
        assert 1000 <= document["estimated_tokens"] <= 3000
        source = ("class _Scope:\n" if document["definition_scope"] else "") + document["text"]
        ast.parse(source)
    for question in manifest["questions"]:
        assert len(question["candidate_refs"]) == 8
        assert len(question["required_refs"]) in (2, 3)
        assert set(question["required_refs"]) <= set(question["candidate_refs"])
        assert score_answer(question["answer"], question["answer"])
        assert all(ref in manifest["documents"] for ref in question["candidate_refs"])


def test_formfeed_does_not_shift_ast_line_ranges(tmp_path: Path) -> None:
    source = "# before\n\f\ndef example(value=3):\n    '''" + "real text " * 500 + "'''\n    return value\n"
    (tmp_path / "sample.py").write_text(source)
    documents, facts = collect_documents(tmp_path)
    assert len(documents) == 1
    document = next(iter(documents.values()))
    assert document["text"].startswith("def example")
    assert document["text"].endswith("    return value\n")
    assert facts[0]["first_default"] == ["value", "3"]
    ast.parse(document["text"])


def test_manifest_answers_follow_frozen_document_signatures() -> None:
    manifest = json.loads((Path(__file__).parents[1] / "manifest.json").read_text())
    signatures = {}
    for document in manifest["documents"].values():
        scope = document["definition_scope"]
        tree = ast.parse(("class _Scope:\n" if scope else "") + document["text"])
        body = tree.body[0].body if scope else tree.body
        def visit(nodes, prefix):
            for node in nodes:
                if isinstance(node, ast.ClassDef):
                    visit(node.body, prefix + node.name + ".")
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    args = node.args
                    positional = [*args.posonlyargs, *args.args]
                    omit = {"self", "cls"} if prefix else set()
                    count = len([arg for arg in [*positional, *args.kwonlyargs] if arg.arg not in omit])
                    count += bool(args.vararg) + bool(args.kwarg)
                    defaults = [(arg.arg, ast.unparse(default)) for arg, default in
                                zip(positional[len(positional)-len(args.defaults):], args.defaults) if arg.arg not in omit]
                    defaults += [(arg.arg, ast.unparse(default)) for arg, default in
                                 zip(args.kwonlyargs, args.kw_defaults) if default is not None and arg.arg not in omit]
                    signatures[(document["id"], document["module"] + "." + prefix + node.name)] = (count, defaults)
        visit(body, scope)
    for question in manifest["questions"]:
        ranked = sorted(question["targets"], key=lambda target: signatures[(target["doc_id"], target["symbol"])][0], reverse=True)
        winner = ranked[0]
        winner_count, defaults = signatures[(winner["doc_id"], winner["symbol"])]
        assert winner_count > signatures[(ranked[1]["doc_id"], ranked[1]["symbol"])][0]
        assert question["answer"] == {"winner": winner["symbol"], "parameter": defaults[0][0], "default": defaults[0][1]}
