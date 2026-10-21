from agentos.context.prefix import ContextPriority, PrefixCanonicalizer, PrefixSegment


class TestContextPriority:
    def test_ordering(self):
        assert ContextPriority.SYSTEM_INSTRUCTION < ContextPriority.TASK_BACKGROUND
        assert ContextPriority.TASK_BACKGROUND < ContextPriority.REPO_SUMMARY
        assert ContextPriority.REPO_SUMMARY <= ContextPriority.SHARED_CONTEXT
        assert ContextPriority.SHARED_CONTEXT < ContextPriority.AGENT_ROLE
        assert ContextPriority.DEFAULT == ContextPriority.SHARED_CONTEXT


class TestPrefixCanonicalizer:
    def test_canonicalize_sorts_by_priority_then_name(self):
        segments = [
            PrefixSegment(name="repo", content="repo info", priority=20),
            PrefixSegment(name="sys", content="system prompt", priority=0),
            PrefixSegment(name="bg", content="task background", priority=10),
        ]
        result = PrefixCanonicalizer.canonicalize(segments)
        assert [s.name for s in result] == ["sys", "bg", "repo"]

    def test_canonicalize_tiebreaks_by_name(self):
        segments = [
            PrefixSegment(name="zebra", content="z", priority=30),
            PrefixSegment(name="alpha", content="a", priority=30),
            PrefixSegment(name="mid", content="m", priority=30),
        ]
        result = PrefixCanonicalizer.canonicalize(segments)
        assert [s.name for s in result] == ["alpha", "mid", "zebra"]

    def test_canonicalize_uses_ordinal_before_name(self):
        segments = [
            PrefixSegment(name="alpha", content="a", priority=30, ordinal=2),
            PrefixSegment(name="zebra", content="z", priority=30, ordinal=1),
        ]
        result = PrefixCanonicalizer.canonicalize(segments)
        assert [s.name for s in result] == ["zebra", "alpha"]

    def test_system_instruction_comes_first(self):
        segments = [
            PrefixSegment(name="bg", content="bg", priority=ContextPriority.TASK_BACKGROUND),
            PrefixSegment(name="sys", content="sys", priority=ContextPriority.SYSTEM_INSTRUCTION),
            PrefixSegment(name="repo", content="repo", priority=ContextPriority.REPO_SUMMARY),
        ]
        result = PrefixCanonicalizer.canonicalize(segments)
        assert result[0].name == "sys"

    def test_prefix_id_deterministic_regardless_of_input_order(self):
        seg_a = PrefixSegment(name="a", content="alpha", priority=10)
        seg_b = PrefixSegment(name="b", content="beta", priority=20)

        id1 = PrefixCanonicalizer.compute_prefix_id([seg_a, seg_b])
        id2 = PrefixCanonicalizer.compute_prefix_id([seg_b, seg_a])
        assert id1 == id2

    def test_prefix_id_changes_with_different_content(self):
        seg1 = [PrefixSegment(name="a", content="hello")]
        seg2 = [PrefixSegment(name="a", content="world")]
        assert PrefixCanonicalizer.compute_prefix_id(seg1) != PrefixCanonicalizer.compute_prefix_id(seg2)

    def test_build_prefix_content(self):
        segments = [
            PrefixSegment(name="b", content="second", priority=20),
            PrefixSegment(name="a", content="first", priority=10),
        ]
        result = PrefixCanonicalizer.build_prefix_content(segments)
        assert result == "first\n\nsecond"

    def test_empty_segments(self):
        assert PrefixCanonicalizer.canonicalize([]) == []
        assert PrefixCanonicalizer.build_prefix_content([]) == ""

    def test_compute_overlap_identical(self):
        left = [
            PrefixSegment(name="bg", content="same background", priority=10),
            PrefixSegment(name="repo", content="same repo", priority=20),
        ]
        overlap = PrefixCanonicalizer.compute_overlap(left, list(reversed(left)))

        assert overlap.reason == "identical"
        assert overlap.shorter_overlap_ratio == 1.0
        assert overlap.shared_segment_names == ("bg", "repo")

    def test_compute_overlap_detects_prefix_extension(self):
        left = [
            PrefixSegment(name="bg", content="shared background", priority=10),
        ]
        right = [
            PrefixSegment(name="bg", content="shared background", priority=10),
            PrefixSegment(name="draft", content="new downstream notes", priority=30),
        ]
        overlap = PrefixCanonicalizer.compute_overlap(left, right)

        assert overlap.reason == "prefix_extension"
        assert overlap.left_overlap_ratio == 1.0
        assert overlap.right_overlap_ratio < 1.0
        assert overlap.shared_segment_names == ("bg",)

    def test_compute_overlap_detects_divergence(self):
        left = [PrefixSegment(name="bg", content="abc shared then left")]
        right = [PrefixSegment(name="bg", content="abc shared then right")]

        overlap = PrefixCanonicalizer.compute_overlap(left, right)

        assert overlap.reason == "diverged"
        assert overlap.shared_prefix_tokens > 0
        assert overlap.first_difference_index is not None
