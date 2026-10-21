"""Tests for the demo-video director mode (replay feeder + storyboard)."""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import pytest

from agentos.dashboard.broadcast import EventBroadcaster
from agentos.events.types import EventCategory, RuntimeEvent


def test_subscriber_count_tracks_subscribe_unsubscribe():
    b = EventBroadcaster()
    assert b.subscriber_count() == 0

    async def check():
        q1 = b.subscribe()
        q2 = b.subscribe()
        assert b.subscriber_count() == 2
        b.unsubscribe(q1)
        assert b.subscriber_count() == 1
        b.unsubscribe(q2)

    asyncio.run(check())
    assert b.subscriber_count() == 0


from agentos.dashboard.replay import (
    Scene,
    Segment,
    load_storyboard,
    slice_segment,
)


def _write_storyboard(tmp_path: Path, scenes: list[dict], total: float = 30) -> Path:
    p = tmp_path / "storyboard.json"
    p.write_text(
        json.dumps({"total_duration": total, "scenes": scenes}, ensure_ascii=False),
        encoding="utf-8",
    )
    return p


def test_load_storyboard_parses_scenes_and_segments(tmp_path):
    p = _write_storyboard(tmp_path, [
        {"scene": "intro", "start": 0, "duration": 10,
         "chapter_card": {"title": "T", "hold": 3}},
        {"scene": "act1", "start": 10, "duration": 20, "camera": "#zone-process",
         "caption": "cap",
         "replay": {"file": "a.jsonl",
                    "segments": [{"to_marker": "task.dynamic_submitted", "weight": 2}]}},
    ])
    scenes, total = load_storyboard(p)
    assert total == 30
    assert [s.scene for s in scenes] == ["intro", "act1"]
    assert scenes[0].replay_file is None
    assert scenes[1].replay_file == "a.jsonl"
    assert scenes[1].segments[0].to_marker == "task.dynamic_submitted"
    assert scenes[1].segments[0].weight == 2


def test_load_storyboard_rejects_gap_between_scenes(tmp_path):
    p = _write_storyboard(tmp_path, [
        {"scene": "a", "start": 0, "duration": 10},
        {"scene": "b", "start": 15, "duration": 15},
    ])
    with pytest.raises(ValueError, match="starts at"):
        load_storyboard(p)


def test_load_storyboard_rejects_wrong_total(tmp_path):
    p = _write_storyboard(tmp_path, [
        {"scene": "a", "start": 0, "duration": 10},
    ], total=30)
    with pytest.raises(ValueError, match="total"):
        load_storyboard(p)


def _mk_events(types: list[str]) -> list[RuntimeEvent]:
    return [
        RuntimeEvent(event_type=t, category=EventCategory.SYSTEM,
                     data={}, timestamp=100.0 + i)
        for i, t in enumerate(types)
    ]


def test_slice_segment_by_markers():
    events = _mk_events(["a", "b", "task.dynamic_submitted", "c", "d"])
    head = slice_segment(events, Segment(to_marker="task.dynamic_submitted"))
    tail = slice_segment(events, Segment(from_marker="task.dynamic_submitted"))
    assert [e.event_type for e in head] == ["a", "b"]
    assert [e.event_type for e in tail] == ["task.dynamic_submitted", "c", "d"]


def test_slice_segment_by_frac():
    events = _mk_events(["a", "b", "c", "d"])
    first_half = slice_segment(events, Segment(to_frac=0.5))
    second_half = slice_segment(events, Segment(from_frac=0.5))
    assert [e.event_type for e in first_half] == ["a", "b"]
    assert [e.event_type for e in second_half] == ["c", "d"]


def test_slice_segment_missing_marker_raises():
    events = _mk_events(["a", "b"])
    with pytest.raises(ValueError, match="marker"):
        slice_segment(events, Segment(from_marker="never.happens"))


from agentos.dashboard.replay import build_schedule


def _two_scene_storyboard(tmp_path: Path) -> Path:
    return _write_storyboard(tmp_path, [
        {"scene": "intro", "start": 0, "duration": 10,
         "chapter_card": {"title": "T", "hold": 8}},
        {"scene": "act1", "start": 10, "duration": 20, "camera": "#zone-process",
         "chapter_card": {"title": "A1", "hold": 2},
         "replay": {"file": "a.jsonl", "segments": [{}]}},
    ], total=30)


def test_build_schedule_scales_event_spacing(tmp_path):
    scenes, total = load_storyboard(_two_scene_storyboard(tmp_path))
    # 原始事件跨度 8s（100→108），要塞进 20-2(hold)=18s 的槽
    events = [
        RuntimeEvent(event_type=f"e{i}", category=EventCategory.SYSTEM,
                     timestamp=100.0 + 4 * i)
        for i in range(3)
    ]
    sched = build_schedule(scenes, total, lambda name: list(events))
    data_offsets = [off for off, e in sched if e.event_type.startswith("e")]
    assert len(data_offsets) == 3
    # 第一条落在 hold 之后
    assert data_offsets[0] == pytest.approx(12.0, abs=0.01)
    # 等比缩放：原始等间距 → 回放也等间距
    gap1 = data_offsets[1] - data_offsets[0]
    gap2 = data_offsets[2] - data_offsets[1]
    assert gap1 == pytest.approx(gap2, rel=0.01)
    # 收尾不超出本幕槽位（scene act1 结束于 30s）
    assert data_offsets[-1] < 30.0


def test_build_schedule_emits_director_cues_in_order(tmp_path):
    scenes, total = load_storyboard(_two_scene_storyboard(tmp_path))
    sched = build_schedule(scenes, total, lambda name: _mk_events(["x"]))
    types = [e.event_type for _, e in sched]
    assert types[0] == "director.show_start"
    assert types[-1] == "director.show_end"
    assert types.count("director.scene_start") == 2
    assert types.count("director.scene_end") == 2
    # scene_start 携带前端需要的全部元数据
    starts = [e for _, e in sched if e.event_type == "director.scene_start"]
    meta = starts[1].data
    assert meta["scene"] == "act1"
    assert meta["camera"] == "#zone-process"
    assert meta["show_total"] == 30
    assert meta["index"] == 1 and meta["scene_count"] == 2
    # 全部 director 事件 category 合法（能被 RuntimeEvent.from_dict round-trip）
    for _, e in sched:
        if e.event_type.startswith("director."):
            RuntimeEvent.from_dict(e.to_dict())


def test_build_schedule_offsets_monotonic(tmp_path):
    scenes, total = load_storyboard(_two_scene_storyboard(tmp_path))
    sched = build_schedule(scenes, total, lambda name: _mk_events(["x", "y", "z"]))
    offsets = [off for off, _ in sched]
    assert offsets == sorted(offsets)


def test_load_storyboard_rejects_hold_gte_duration(tmp_path):
    p = _write_storyboard(tmp_path, [
        {"scene": "bad", "start": 0, "duration": 10,
         "chapter_card": {"title": "T", "hold": 10}},
    ], total=10)
    with pytest.raises(ValueError, match="hold="):
        load_storyboard(p)


from agentos.dashboard.replay import run_replay


def test_run_replay_publishes_all_events_with_rewritten_timestamps(tmp_path):
    # 微型 storyboard：0.2s 的秀，事件全部落进去
    sb = _write_storyboard(tmp_path, [
        {"scene": "s1", "start": 0, "duration": 0.2, "camera": "#x",
         "replay": {"file": "mini.jsonl", "segments": [{}]}},
    ], total=0.2)
    from agentos.events.log import EventLog
    log = EventLog(tmp_path / "mini.jsonl")
    for e in _mk_events(["a", "b"]):
        log.append(e)

    b = EventBroadcaster()

    async def scenario():
        # 先挂一个订阅者，run_replay 才会开演
        b.subscribe()
        t0 = time.time()
        await run_replay(b, sb, tmp_path, lead_in=0.0)
        return t0

    t0 = asyncio.run(scenario())
    types = [e.event_type for e in b.snapshot()]
    assert types[0] == "director.show_start"
    assert "a" in types and "b" in types
    assert types[-1] == "director.show_end"
    # 时间戳被改写为回放墙钟（而非原始录制时间 100.x）
    data_events = [e for e in b.snapshot() if e.event_type == "a"]
    assert data_events[0].timestamp >= t0


def test_run_replay_times_out_without_subscriber(tmp_path):
    sb = _write_storyboard(tmp_path, [
        {"scene": "s1", "start": 0, "duration": 0.1},
    ], total=0.1)
    b = EventBroadcaster()
    with pytest.raises(TimeoutError):
        asyncio.run(run_replay(b, sb, tmp_path, lead_in=0.0,
                               subscriber_timeout=0.3))


def test_server_serves_director_static_files():
    from agentos.dashboard.server import DashboardApp

    app = DashboardApp(EventBroadcaster())

    async def fetch(path):
        return await app.handle_request("GET", path)

    status, headers, body = asyncio.run(fetch("/director.js"))
    assert status == 200
    assert "javascript" in headers["Content-Type"]
    status, headers, body = asyncio.run(fetch("/director.css"))
    assert status == 200
    assert "css" in headers["Content-Type"]


REPO_ROOT = Path(__file__).resolve().parent.parent
DEMO_DIR = REPO_ROOT / "demo" / "replay"
REFINED_STORYBOARD = DEMO_DIR / "storyboard_qhd_refined.json"


def test_real_storyboard_is_valid():
    scenes, total = load_storyboard(DEMO_DIR / "storyboard.json")
    assert total == 285
    assert scenes[0].scene == "act1_acb"
    assert scenes[-1].scene == "outro"


def test_qhd_refined_storyboard_is_valid_and_uses_confirmed_copy():
    scenes, total = load_storyboard(REFINED_STORYBOARD)

    assert total == 233
    assert [(s.scene, s.start, s.duration) for s in scenes] == [
        ("intro", 0.0, 7.0),
        ("act1_acb", 7.0, 35.0),
        ("act2_dag", 42.0, 50.0),
        ("act3_context", 92.0, 30.0),
        ("act3_cache", 122.0, 34.0),
        ("act4_fault", 156.0, 55.0),
        ("results", 211.0, 18.0),
        ("outro", 229.0, 4.0),
    ]

    by_name = {scene.scene: scene for scene in scenes}

    assert "Linux cgroup v2 资源域" in by_name["act1_acb"].beats[0]["banner"]
    assert "实时依赖状态重算" in by_name["act2_dag"].beats[0]["banner"]
    assert "76.7% 提升至 93.7%" in by_name["act3_cache"].beats[0]["banner"]
    assert "原生 240/720 提升至 Runtime 720/720" in by_name["act4_fault"].beats[0]["banner"]

    card = by_name["results"].data_card
    assert card["kicker"] == "跨框架主实验 · 三场景泛化验证"
    assert card["overview"]["token"]["value"] == "−44.5%"
    assert card["overview"]["cost"]["value"] == "−52.8%"
    assert card["overview"]["cache"]["from"] == "76.7%"
    assert card["overview"]["cache"]["to"] == "93.7%"
    assert [item["source"] for item in card["evidence"]] == ["N10", "N11", "N6"]
    assert card["recovery"]["runtime"] == "720/720"
    assert card["recovery"]["native"] == "原生 240/720"
    assert "footnote" not in card

    outro = by_name["outro"].chapter_card
    assert outro["hold"] == 3.5
    assert outro["closing"] == "统一调度 · 共享上下文 · 局部恢复"
    assert "tags" not in outro


@pytest.mark.skipif(not (DEMO_DIR / "act12_pipeline.jsonl").exists(),
                    reason="demo master logs not present")
def test_real_storyboard_markers_resolve_against_master_logs():
    from agentos.events.log import EventLog

    scenes, total = load_storyboard(DEMO_DIR / "storyboard.json")

    def load(name):
        return list(EventLog(DEMO_DIR / name).read())

    # 不 raise 即为通过：所有 marker / frac 都能在母带里切出非空段
    sched = build_schedule(scenes, total, load)
    data = [e for _, e in sched if not e.event_type.startswith("director.")]
    assert len(data) > 100


def test_server_strips_query_string_on_static_routes():
    """Regression: /?director=1 must resolve to /, not 404 (Task 12 uncovered this)."""
    from agentos.dashboard.server import DashboardApp

    app = DashboardApp(EventBroadcaster())

    async def fetch(path):
        return await app.handle_request("GET", path)

    status, headers, body = asyncio.run(fetch("/?director=1"))
    assert status == 200
    assert "text/html" in headers["Content-Type"]

    status, headers, body = asyncio.run(fetch("/director.js?v=abc"))
    assert status == 200
    assert "javascript" in headers["Content-Type"]


# ---------------------------------------------------------------------------
# Phase A additions: scene beats + slow-motion time warp
# ---------------------------------------------------------------------------


def _mk_events_with_ts(pairs: list[tuple[str, float]]) -> list[RuntimeEvent]:
    return [
        RuntimeEvent(event_type=t, category=EventCategory.SYSTEM,
                     data={}, timestamp=ts)
        for t, ts in pairs
    ]


def _scene_beat_storyboard(tmp_path: Path, beats: list[dict]) -> Path:
    return _write_storyboard(tmp_path, [
        {"scene": "s1", "start": 0, "duration": 20,
         "chapter_card": {"title": "T", "hold": 2},
         "replay": {"file": "a.jsonl", "segments": [{}]},
         "beats": beats},
    ], total=20)


def test_time_beat_scheduled_at_scene_offset(tmp_path):
    """A time-triggered beat lands at scene.start + trigger.at with full payload."""
    sb = _scene_beat_storyboard(tmp_path, [
        {"trigger": {"at": 5.0},
         "banner": "hi", "hold": 3.0, "pulse": ["#zone-dag"]}
    ])
    scenes, total = load_storyboard(sb)
    sched = build_schedule(scenes, total,
                           lambda name: _mk_events(["x", "y", "z"]))
    beats = [(off, e) for off, e in sched if e.event_type == "director.beat"]
    assert len(beats) == 1
    off, e = beats[0]
    assert off == pytest.approx(5.0)
    assert e.data["banner"] == "hi"
    assert e.data["hold"] == 3.0
    assert e.data["pulse"] == ["#zone-dag"]
    assert e.data["scene"] == "s1"


def test_event_beat_scheduled_at_marker_offset(tmp_path):
    """An event-triggered beat lands at the scaled offset of the marker event."""
    sb = _scene_beat_storyboard(tmp_path, [
        {"trigger": {"on_event": "y"}, "banner": "boom", "hold": 2.5}
    ])
    scenes, total = load_storyboard(sb)
    sched = build_schedule(scenes, total,
                           lambda name: _mk_events(["x", "y", "z"]))
    beats = [(off, e) for off, e in sched if e.event_type == "director.beat"]
    assert len(beats) == 1
    beat_off = beats[0][0]
    y_off = next(off for off, e in sched if e.event_type == "y")
    assert beat_off == pytest.approx(y_off)


def test_event_beat_missing_marker_raises(tmp_path):
    """Beat trigger references an event type absent from the replay → ValueError."""
    sb = _scene_beat_storyboard(tmp_path, [
        {"trigger": {"on_event": "never"}, "banner": "x", "hold": 1.0}
    ])
    scenes, total = load_storyboard(sb)
    with pytest.raises(ValueError, match=r"s1.*never"):
        build_schedule(scenes, total, lambda name: _mk_events(["a", "b"]))


def test_event_beat_second_occurrence(tmp_path):
    """occurrence=2 lands on the second matching event."""
    sb = _scene_beat_storyboard(tmp_path, [
        {"trigger": {"on_event": "x", "occurrence": 2},
         "banner": "second", "hold": 1.0}
    ])
    scenes, total = load_storyboard(sb)
    events = _mk_events(["x", "y", "x", "z", "x"])
    sched = build_schedule(scenes, total, lambda name: events)
    x_offsets = [off for off, e in sched if e.event_type == "x"]
    beat_off = next(off for off, e in sched if e.event_type == "director.beat")
    assert beat_off == pytest.approx(x_offsets[1])


@pytest.mark.parametrize("bad_beat, match", [
    ({"trigger": {}, "banner": "b", "hold": 1.0}, "exactly one"),
    ({"trigger": {"at": 5, "on_event": "y"}, "banner": "b", "hold": 1.0}, "exactly one"),
    ({"trigger": {"at": -1}, "banner": "b", "hold": 1.0}, "at="),
    ({"trigger": {"at": 100}, "banner": "b", "hold": 1.0}, "at="),
    ({"trigger": {"on_event": ""}, "banner": "b", "hold": 1.0}, "non-empty"),
    ({"trigger": {"on_event": "y", "occurrence": 0}, "banner": "b", "hold": 1.0}, "occurrence"),
    ({"trigger": {"at": 5.0}, "hold": 1.0}, "banner"),
    ({"trigger": {"at": 5.0}, "banner": "", "hold": 1.0}, "banner"),
    ({"trigger": {"at": 5.0}, "banner": "b"}, "hold"),
    ({"trigger": {"at": 5.0}, "banner": "b", "hold": 0}, "hold"),
    ({"trigger": {"at": 5.0}, "banner": "b", "hold": 30}, "hold"),
    ({"trigger": {"at": 5.0}, "banner": "b", "hold": 1.0, "pulse": "not-a-list"}, "pulse"),
    ({"trigger": {"at": 5.0}, "banner": "b", "hold": 1.0, "pulse": [""]}, "pulse"),
])
def test_beat_validation_rejects_bad_shapes(tmp_path, bad_beat, match):
    sb = _scene_beat_storyboard(tmp_path, [bad_beat])
    with pytest.raises(ValueError, match=match):
        load_storyboard(sb)


def test_slow_motion_stretches_marker_window(tmp_path):
    """Events near the slow-motion marker are spaced wider than events at the edges."""
    p = _write_storyboard(tmp_path, [
        {"scene": "s1", "start": 0, "duration": 20,
         "chapter_card": {"title": "T", "hold": 0.1},
         "replay": {"file": "a.jsonl", "segments": [{}]},
         "slow_motion": {"on_event": "e5", "speed": 0.4,
                         "window_before": 1.0, "window_after": 1.0}},
    ], total=20)
    scenes, total = load_storyboard(p)
    # 10 raw events at ts=0..9 unit spacing, marker is "e5" at t=5
    events = _mk_events_with_ts([(f"e{i}", float(i)) for i in range(10)])
    sched = build_schedule(scenes, total, lambda name: events)
    offsets = {e.event_type: off for off, e in sched
               if e.event_type.startswith("e")}
    # All events land inside the scene
    for off in offsets.values():
        assert 0.0 <= off <= 20.0
    # Edge gap (e0↔e1) should be smaller than around-marker gap (e4↔e5)
    edge = offsets["e1"] - offsets["e0"]
    marker_gap = offsets["e5"] - offsets["e4"]
    assert marker_gap > edge * 1.5


def test_slow_motion_missing_marker_raises(tmp_path):
    p = _write_storyboard(tmp_path, [
        {"scene": "s1", "start": 0, "duration": 20,
         "chapter_card": {"title": "T", "hold": 0.1},
         "replay": {"file": "a.jsonl", "segments": [{}]},
         "slow_motion": {"on_event": "does_not_exist", "speed": 0.5,
                         "window_before": 1.0, "window_after": 1.0}},
    ], total=20)
    scenes, total = load_storyboard(p)
    with pytest.raises(ValueError, match=r"does_not_exist"):
        build_schedule(scenes, total, lambda name: _mk_events(["a", "b", "c"]))


@pytest.mark.parametrize("sm, match", [
    ({"on_event": "", "speed": 0.5, "window_before": 1, "window_after": 1}, "on_event"),
    ({"speed": 0.5, "window_before": 1, "window_after": 1}, "on_event"),
    ({"on_event": "y", "speed": 0, "window_before": 1, "window_after": 1}, "speed"),
    ({"on_event": "y", "speed": -0.5, "window_before": 1, "window_after": 1}, "speed"),
    ({"on_event": "y", "speed": 1.5, "window_before": 1, "window_after": 1}, "speed"),
    ({"on_event": "y", "speed": 0.5, "window_before": -1, "window_after": 1}, "window_before"),
    ({"on_event": "y", "speed": 0.5, "window_before": 1, "window_after": -1}, "window_after"),
])
def test_slow_motion_validation_rejects_bad_shapes(tmp_path, sm, match):
    p = _write_storyboard(tmp_path, [
        {"scene": "s1", "start": 0, "duration": 20,
         "chapter_card": {"title": "T", "hold": 0.1},
         "replay": {"file": "a.jsonl", "segments": [{}]},
         "slow_motion": sm},
    ], total=20)
    with pytest.raises(ValueError, match=match):
        load_storyboard(p)


# ---------------------------------------------------------------------------
# Task 1 additions: scene-level stinger / label / fade_out fields
# ---------------------------------------------------------------------------


def _sb(tmp_path: Path, scenes: list[dict], total: float) -> Path:
    p = tmp_path / "sb_stinger.json"
    p.write_text(json.dumps({"total_duration": total, "scenes": scenes}),
                 encoding="utf-8")
    return p


def test_stinger_label_fade_roundtrip_to_scene_start_meta(tmp_path):
    from agentos.dashboard.replay import build_schedule
    scenes_raw = [
        {"scene": "intro", "start": 0, "duration": 6, "camera": None,
         "label": "",
         "stinger": {"title": "AgentOS Runtime", "subtitle": "副题",
                     "hold": 3.2, "variant": "title"}},
        {"scene": "outro", "start": 6, "duration": 4, "camera": None,
         "label": "", "fade_out": 0.5,
         "stinger": {"title": "演示结束", "variant": "outro"}},
    ]
    scenes, total = load_storyboard(_sb(tmp_path, scenes_raw, 10))
    assert scenes[0].stinger["hold"] == 3.2
    assert scenes[0].label == ""
    assert scenes[1].fade_out == 0.5

    sched = build_schedule(scenes, total, load_events=lambda n: [])
    starts = [e for _, e in sched if e.event_type == "director.scene_start"]
    assert starts[0].data["stinger"]["variant"] == "title"
    assert starts[0].data["label"] == ""
    assert starts[1].data["fade_out"] == 0.5


def test_stinger_title_required(tmp_path):
    bad = [{"scene": "a", "start": 0, "duration": 5,
            "stinger": {"subtitle": "无标题"}}]
    with pytest.raises(ValueError, match="title"):
        load_storyboard(_sb(tmp_path, bad, 5))


def test_stinger_hold_must_fit_duration(tmp_path):
    bad = [{"scene": "a", "start": 0, "duration": 3,
            "stinger": {"title": "T", "hold": 3.5}}]
    with pytest.raises(ValueError, match="hold"):
        load_storyboard(_sb(tmp_path, bad, 3))


def test_stinger_variant_whitelist(tmp_path):
    bad = [{"scene": "a", "start": 0, "duration": 5,
            "stinger": {"title": "T", "variant": "banner"}}]
    with pytest.raises(ValueError, match="variant"):
        load_storyboard(_sb(tmp_path, bad, 5))


def test_fade_out_must_fit_duration(tmp_path):
    bad = [{"scene": "a", "start": 0, "duration": 4, "fade_out": 4.0}]
    with pytest.raises(ValueError, match="fade_out"):
        load_storyboard(_sb(tmp_path, bad, 4))


def test_legacy_chapter_card_still_loads(tmp_path):
    ok = [{"scene": "a", "start": 0, "duration": 5,
           "chapter_card": {"title": "旧卡", "hold": 2.5}}]
    scenes, _ = load_storyboard(_sb(tmp_path, ok, 5))
    assert scenes[0].chapter_card["title"] == "旧卡"
    assert scenes[0].stinger is None


def test_storyboard_final_matches_spec():
    path = Path(__file__).parent.parent / "demo/replay/storyboard_final.json"
    scenes, total = load_storyboard(path)
    assert total == 229
    assert [s.scene for s in scenes] == [
        "intro", "act1_acb", "act2_dag", "act3_context",
        "act3_cache", "act4_fault", "results", "outro",
    ]
    assert all(s.chapter_card is None for s in scenes)
    intro, act1 = scenes[0], scenes[1]
    assert intro.camera is None and intro.label == "" and intro.caption is None
    assert intro.segments[0].to_frac == 0.08
    assert act1.segments[0].from_frac == 0.08
    assert act1.segments[0].to_marker == "task.dynamic_submitted"
    assert scenes[4].stinger is None            # act3_cache 不打横幅
    assert scenes[4].label == "03 / 04 · KV Cache 亲和"
    assert scenes[7].fade_out == 0.5
    qhd = json.loads(
        (path.parent / "storyboard_qhd_refined.json").read_text(encoding="utf-8"))
    qhd_results = next(s for s in qhd["scenes"] if s["scene"] == "results")
    assert scenes[6].data_card == qhd_results["data_card"]   # 数字逐字一致
    assert scenes[6].beats == qhd_results["beats"]
