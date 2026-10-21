from agentos.context.dependency import (
    ContextDependencyGraph,
    DependencyQuery,
)


def test_dependency_graph_matches_refs_by_file_symbol_and_task():
    graph = ContextDependencyGraph()
    graph.add_ref(
        "repo-runtime",
        read_files=["agentos/runtime/engine.py"],
        symbols=["Runtime.run"],
        consumer_task_ids=["implement-fix"],
        role="repo_summary",
    )
    graph.add_ref(
        "test-plan",
        read_files=["tests/test_runtime.py"],
        symbols=["test_runtime_recovery"],
        producer_task_id="plan-tests",
        weight=1.5,
    )

    matches = graph.match(
        DependencyQuery.build(
            task_id="implement-fix",
            read_files=["agentos/runtime/engine.py"],
            symbols=["Runtime.run"],
        )
    )

    assert matches[0].ref_name == "repo-runtime"
    assert set(matches[0].reasons) == {"task", "read_file", "symbol"}
    assert "test-plan" not in [match.ref_name for match in matches]


def test_dependency_graph_ranks_weighted_matches():
    graph = ContextDependencyGraph()
    graph.add_ref("low", read_files=["a.py"], weight=1.0)
    graph.add_ref("high", read_files=["a.py"], weight=2.0)

    refs = graph.relevant_refs(
        DependencyQuery.build(read_files=["a.py"]),
    )

    assert refs == ["high", "low"]


def test_dependency_graph_exports_indexes():
    graph = ContextDependencyGraph()
    graph.add_ref(
        "patch-context",
        read_files=["a.py"],
        write_files=["b.py"],
        symbols=["Foo"],
        producer_task_id="patch",
    )

    data = graph.to_dict()

    assert data["files"]["a.py"] == ["patch-context"]
    assert data["files"]["b.py"] == ["patch-context"]
    assert data["symbols"]["Foo"] == ["patch-context"]
    assert data["tasks"]["patch"] == ["patch-context"]
