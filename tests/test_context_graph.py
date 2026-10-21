from agentos.context.graph import ContextLayout, PrefixGraph
from agentos.context.prefix import ContextPriority, PrefixSegment


def test_context_layout_is_merkle_stable_for_same_segments():
    segments = [
        PrefixSegment("repo", "repo summary", ContextPriority.REPO_SUMMARY, 1),
        PrefixSegment("system", "system prompt", ContextPriority.SYSTEM_INSTRUCTION, 0),
    ]

    left = ContextLayout.from_prefix_segments(segments)
    right = ContextLayout.from_prefix_segments(list(reversed(segments)))

    assert left.layout_id == right.layout_id
    assert left.segment_names == ("system", "repo")
    assert len(left.cumulative_hashes) == 2


def test_context_layout_divergence_finds_shared_prefix_segments():
    left = ContextLayout.from_prefix_segments([
        PrefixSegment("system", "system", ContextPriority.SYSTEM_INSTRUCTION, 0),
        PrefixSegment("repo", "repo", ContextPriority.REPO_SUMMARY, 1),
        PrefixSegment("api", "api design", ContextPriority.SHARED_CONTEXT, 2),
    ])
    right = ContextLayout.from_prefix_segments([
        PrefixSegment("system", "system", ContextPriority.SYSTEM_INSTRUCTION, 0),
        PrefixSegment("repo", "repo", ContextPriority.REPO_SUMMARY, 1),
        PrefixSegment("tests", "test plan", ContextPriority.SHARED_CONTEXT, 2),
    ])

    divergence = left.divergence(right)

    assert divergence.reason == "diverged"
    assert divergence.shared_segment_names == ("system", "repo")
    assert divergence.first_divergence_index == 2
    assert divergence.left_only_segment_names == ("api",)
    assert divergence.right_only_segment_names == ("tests",)


def test_prefix_graph_exports_pairwise_shared_prefix_matrix():
    graph = PrefixGraph()
    graph.add_layout(
        "planner",
        ContextLayout.from_prefix_segments([
            PrefixSegment("system", "system", ContextPriority.SYSTEM_INSTRUCTION, 0),
            PrefixSegment("repo", "repo", ContextPriority.REPO_SUMMARY, 1),
        ]),
    )
    graph.add_layout(
        "coder",
        ContextLayout.from_prefix_segments([
            PrefixSegment("system", "system", ContextPriority.SYSTEM_INSTRUCTION, 0),
            PrefixSegment("repo", "repo", ContextPriority.REPO_SUMMARY, 1),
            PrefixSegment("plan", "plan", ContextPriority.SHARED_CONTEXT, 2),
        ]),
    )

    data = graph.to_dict()

    assert "planner" in data["layouts"]
    assert data["shared_prefix_matrix"]["coder->planner"]["reason"] == "prefix_extension"
