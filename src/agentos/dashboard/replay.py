"""Deterministic replay feeder for the demo-video director mode.

Reads a storyboard.json, slices recorded JSONL event logs into scenes,
rescales timestamps to fit each scene's time slot, and publishes the
result — interleaved with synthetic ``director.*`` cue events — through
the existing EventBroadcaster/SSE channel.  The frontend director module
(static/director.js) reacts to the cues; data rendering is untouched.
"""
from __future__ import annotations

import asyncio
import dataclasses
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from agentos.dashboard.broadcast import EventBroadcaster
from agentos.events.log import EventLog
from agentos.events.types import EventCategory, RuntimeEvent

# Events inside a segment are compressed to this share of the slot so the
# tail event lands slightly before the scene ends.
_SLOT_FILL = 0.96

# Sort priority for events sharing the same absolute offset.  Lower fires first.
_DIRECTOR_PRIORITY = {
    "director.show_start": 0,
    "director.scene_start": 1,
    "director.beat": 2,
    "director.show_end": 100,
}
_DATA_PRIORITY = 3


def _priority(event_type: str) -> int:
    return _DIRECTOR_PRIORITY.get(event_type, _DATA_PRIORITY)


@dataclass
class Segment:
    from_marker: str | None = None
    to_marker: str | None = None
    from_frac: float = 0.0
    to_frac: float = 1.0
    weight: float = 1.0


@dataclass
class Scene:
    scene: str
    start: float
    duration: float
    page: str = "local"
    camera: str | None = None
    spotlight_extra: list[str] = field(default_factory=list)
    caption: str | None = None
    chapter_card: dict | None = None
    data_card: dict | None = None
    replay_file: str | None = None
    segments: list[Segment] = field(default_factory=list)
    beats: list[dict] = field(default_factory=list)
    slow_motion: dict | None = None
    stinger: dict | None = None
    label: str | None = None
    fade_out: float | None = None


def load_storyboard(path: str | Path) -> tuple[list[Scene], float]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    total = float(raw["total_duration"])
    scenes: list[Scene] = []
    for s in raw["scenes"]:
        rep = s.get("replay") or {}
        segments = [Segment(**seg) for seg in rep.get("segments", [])]
        if rep and not segments:
            segments = [Segment()]
        scenes.append(Scene(
            scene=s["scene"],
            start=float(s["start"]),
            duration=float(s["duration"]),
            page=s.get("page", "local"),
            camera=s.get("camera"),
            spotlight_extra=list(s.get("spotlight_extra", [])),
            caption=s.get("caption"),
            chapter_card=s.get("chapter_card"),
            data_card=s.get("data_card"),
            replay_file=rep.get("file"),
            segments=segments,
            beats=list(s.get("beats", [])),
            slow_motion=s.get("slow_motion"),
            stinger=s.get("stinger"),
            label=s.get("label"),
            fade_out=s.get("fade_out"),
        ))
    _validate(scenes, total)
    return scenes, total


def _validate(scenes: list[Scene], total: float) -> None:
    if not scenes:
        raise ValueError("storyboard has no scenes")
    cursor = 0.0
    for sc in scenes:
        if sc.duration <= 0:
            raise ValueError(f"scene {sc.scene} has non-positive duration")
        hold = float((sc.chapter_card or {}).get("hold", 0.0))
        if hold >= sc.duration:
            raise ValueError(
                f"scene {sc.scene} has hold={hold}s >= duration={sc.duration}s"
            )
        if abs(sc.start - cursor) > 1e-6:
            raise ValueError(
                f"scene {sc.scene} starts at {sc.start}, expected {cursor} "
                "(scenes must tile the show without gaps)"
            )
        _validate_beats(sc)
        _validate_slow_motion(sc)
        _validate_stinger(sc)
        _validate_fade_label(sc)
        cursor += sc.duration
    if abs(cursor - total) > 1e-6:
        raise ValueError(f"scenes end at {cursor}, total_duration is {total}")


def _validate_beats(sc: Scene) -> None:
    if not sc.beats:
        return
    if not isinstance(sc.beats, list):
        raise ValueError(f"scene {sc.scene}: beats must be a list")
    for i, beat in enumerate(sc.beats):
        loc = f"scene {sc.scene} beats[{i}]"
        if not isinstance(beat, dict):
            raise ValueError(f"{loc}: beat must be a dict")
        trig = beat.get("trigger")
        if not isinstance(trig, dict):
            raise ValueError(f"{loc}: missing trigger dict")
        has_at = "at" in trig
        has_ev = "on_event" in trig
        if has_at == has_ev:
            raise ValueError(
                f"{loc}: trigger must have exactly one of 'at' or 'on_event'"
            )
        if has_at:
            at = trig["at"]
            if not isinstance(at, (int, float)) or at < 0 or at > sc.duration:
                raise ValueError(
                    f"{loc}: trigger.at={at!r} must be in [0, {sc.duration}]"
                )
        else:
            ev = trig["on_event"]
            if not isinstance(ev, str) or not ev:
                raise ValueError(f"{loc}: trigger.on_event must be non-empty string")
            occ = trig.get("occurrence", 1)
            if not isinstance(occ, int) or occ < 1:
                raise ValueError(f"{loc}: trigger.occurrence must be int >= 1")
        banner = beat.get("banner")
        if not isinstance(banner, str) or not banner:
            raise ValueError(f"{loc}: banner must be non-empty string")
        hold = beat.get("hold")
        if not isinstance(hold, (int, float)) or hold <= 0 or hold >= 30:
            raise ValueError(f"{loc}: hold must be a number in (0, 30)")
        pulse = beat.get("pulse")
        if pulse is not None:
            if not isinstance(pulse, list) or not all(
                isinstance(s, str) and s for s in pulse
            ):
                raise ValueError(f"{loc}: pulse must be a list of non-empty strings")


def _validate_slow_motion(sc: Scene) -> None:
    if sc.slow_motion is None:
        return
    sm = sc.slow_motion
    loc = f"scene {sc.scene} slow_motion"
    if not isinstance(sm, dict):
        raise ValueError(f"{loc}: must be a dict")
    ev = sm.get("on_event")
    if not isinstance(ev, str) or not ev:
        raise ValueError(f"{loc}: on_event must be non-empty string")
    speed = sm.get("speed")
    if not isinstance(speed, (int, float)) or not (0 < speed <= 1):
        raise ValueError(f"{loc}: speed must be a number in (0, 1]")
    for k in ("window_before", "window_after"):
        v = sm.get(k, 0.0)
        if not isinstance(v, (int, float)) or v < 0:
            raise ValueError(f"{loc}: {k} must be a non-negative number")


def _validate_stinger(sc: Scene) -> None:
    if sc.stinger is None:
        return
    st = sc.stinger
    loc = f"scene {sc.scene} stinger"
    if not isinstance(st, dict):
        raise ValueError(f"{loc}: must be a dict")
    title = st.get("title")
    if not isinstance(title, str) or not title:
        raise ValueError(f"{loc}: title must be a non-empty string")
    subtitle = st.get("subtitle")
    if subtitle is not None and not isinstance(subtitle, str):
        raise ValueError(f"{loc}: subtitle must be a string")
    hold = st.get("hold", 2.2)
    if not isinstance(hold, (int, float)) or not (0 < hold < 10):
        raise ValueError(f"{loc}: hold must be a number in (0, 10)")
    if hold >= sc.duration:
        raise ValueError(
            f"{loc}: hold={hold}s >= duration={sc.duration}s"
        )
    variant = st.get("variant", "chapter")
    if variant not in ("chapter", "title", "outro"):
        raise ValueError(f"{loc}: variant must be chapter|title|outro")


def _validate_fade_label(sc: Scene) -> None:
    if sc.label is not None and not isinstance(sc.label, str):
        raise ValueError(f"scene {sc.scene}: label must be a string")
    if sc.fade_out is not None:
        fo = sc.fade_out
        if not isinstance(fo, (int, float)) or not (0 < fo < sc.duration):
            raise ValueError(
                f"scene {sc.scene}: fade_out must be in (0, duration)"
            )


def slice_segment(events: list[RuntimeEvent], seg: Segment) -> list[RuntimeEvent]:
    lo, hi = 0, len(events)
    if seg.from_marker is not None:
        lo = _marker_index(events, seg.from_marker)
    elif seg.from_frac:
        lo = int(len(events) * seg.from_frac)
    if seg.to_marker is not None:
        hi = _marker_index(events, seg.to_marker)
    elif seg.to_frac < 1.0:
        hi = int(len(events) * seg.to_frac)
    return events[lo:hi]


def _marker_index(events: list[RuntimeEvent], marker: str) -> int:
    for i, e in enumerate(events):
        if e.event_type == marker:
            return i
    raise ValueError(f"marker event_type {marker!r} not found in replay file")


def _director_event(event_type: str, data: dict) -> RuntimeEvent:
    return RuntimeEvent(
        event_type=event_type, category=EventCategory.SYSTEM, data=data
    )


def _scale_into_slot(
    events: list[RuntimeEvent], slot_start: float, slot_len: float
) -> list[tuple[float, RuntimeEvent]]:
    """Map events' original timestamps proportionally into [slot_start, slot_start+slot_len)."""
    if not events:
        return []
    t0 = events[0].timestamp
    span = max(events[-1].timestamp - t0, 1e-9)
    usable = slot_len * _SLOT_FILL
    return [
        (slot_start + (e.timestamp - t0) / span * usable, e) for e in events
    ]


def _apply_slow_motion(
    events: list[RuntimeEvent], sc: Scene
) -> list[RuntimeEvent]:
    """Return a shallow-cloned event list whose raw timestamps are warped
    so the window around the slow_motion marker is stretched by 1/speed.

    Events outside the window keep their original spacing (with a uniform
    shift for those after the window).  Feeding this list into the existing
    proportional scaler makes the window occupy proportionally more of the
    scene's runtime, i.e. slow motion.
    """
    if sc.slow_motion is None or not events:
        return events
    sm = sc.slow_motion
    marker = sm["on_event"]
    speed = float(sm["speed"])
    wb = float(sm.get("window_before", 0.0))
    wa = float(sm.get("window_after", 0.0))
    t_m: float | None = None
    for e in events:
        if e.event_type == marker:
            t_m = e.timestamp
            break
    if t_m is None:
        raise ValueError(
            f"scene {sc.scene}: slow_motion.on_event {marker!r} not found in replay"
        )
    win_lo = t_m - wb
    win_hi = t_m + wa
    stretch = (wb + wa) * (1.0 / speed - 1.0)  # extra raw seconds inserted
    out: list[RuntimeEvent] = []
    for e in events:
        clone = dataclasses.replace(e)
        t = e.timestamp
        if t <= win_lo:
            clone.timestamp = t
        elif t <= win_hi:
            clone.timestamp = win_lo + (t - win_lo) / speed
        else:
            clone.timestamp = t + stretch
        out.append(clone)
    return out


def _resolve_beat_offsets(
    sc: Scene,
    scene_events: list[tuple[float, RuntimeEvent]],
) -> list[tuple[float, RuntimeEvent]]:
    """For each beat on the scene, return (absolute_offset, director.beat event)."""
    if not sc.beats:
        return []
    beat_events: list[tuple[float, RuntimeEvent]] = []
    for beat in sc.beats:
        trig = beat["trigger"]
        if "at" in trig:
            abs_offset = sc.start + float(trig["at"])
        else:
            marker = trig["on_event"]
            occurrence = int(trig.get("occurrence", 1))
            hit = 0
            abs_offset = None
            for off, e in scene_events:
                if e.event_type == marker:
                    hit += 1
                    if hit == occurrence:
                        abs_offset = off
                        break
            if abs_offset is None:
                raise ValueError(
                    f"scene {sc.scene}: beat trigger on_event={marker!r} "
                    f"occurrence={occurrence} not found in scaled events"
                )
        payload = {
            "scene": sc.scene,
            "banner": beat["banner"],
            "hold": float(beat["hold"]),
            "pulse": list(beat.get("pulse") or []),
        }
        beat_events.append(
            (abs_offset, _director_event("director.beat", payload))
        )
    return beat_events


def build_schedule(
    scenes: list[Scene],
    total: float,
    load_events: Callable[[str], list[RuntimeEvent]],
) -> list[tuple[float, RuntimeEvent]]:
    """Build the full (offset_seconds, event) schedule for one show.

    Pure function: no clocks, no IO beyond the injected load_events.
    """
    sched: list[tuple[float, RuntimeEvent]] = [
        (0.0, _director_event("director.show_start",
                              {"total": total, "scene_count": len(scenes)}))
    ]
    for idx, sc in enumerate(scenes):
        meta = {
            "index": idx, "scene_count": len(scenes), "scene": sc.scene,
            "start": sc.start, "duration": sc.duration, "show_total": total,
            "page": sc.page, "camera": sc.camera,
            "spotlight_extra": sc.spotlight_extra, "caption": sc.caption,
            "chapter_card": sc.chapter_card, "data_card": sc.data_card,
            "stinger": sc.stinger, "label": sc.label, "fade_out": sc.fade_out,
        }
        sched.append((sc.start, _director_event("director.scene_start", meta)))
        scene_events: list[tuple[float, RuntimeEvent]] = []
        if sc.replay_file:
            events = load_events(sc.replay_file)
            events = _apply_slow_motion(events, sc)
            hold = float((sc.chapter_card or {}).get("hold", 0.0))
            cursor = sc.start + hold
            slot_total = max(sc.duration - hold, 0.1)
            weight_sum = sum(seg.weight for seg in sc.segments) or 1.0
            for seg in sc.segments:
                seg_slot = slot_total * (seg.weight / weight_sum)
                scene_events.extend(
                    _scale_into_slot(slice_segment(events, seg), cursor, seg_slot)
                )
                cursor += seg_slot
        sched.extend(scene_events)
        sched.extend(_resolve_beat_offsets(sc, scene_events))
        sched.append((sc.start + sc.duration - 1e-3,
                      _director_event("director.scene_end",
                                      {"scene": sc.scene, "index": idx})))
    sched.append((total, _director_event("director.show_end", {"total": total})))
    sched.sort(key=lambda pair: (pair[0], _priority(pair[1].event_type)))
    return sched


async def _wait_for_subscriber(
    broadcaster: EventBroadcaster, timeout: float
) -> None:
    deadline = time.monotonic() + timeout
    while broadcaster.subscriber_count() == 0:
        if time.monotonic() > deadline:
            raise TimeoutError(
                "no dashboard client connected before replay start"
            )
        await asyncio.sleep(0.2)


async def run_replay(
    broadcaster: EventBroadcaster,
    storyboard_path: str | Path,
    base_dir: str | Path,
    *,
    lead_in: float = 1.0,
    subscriber_timeout: float = 600.0,
) -> None:
    """Wait for a client, then play the storyboard schedule in real time.

    Timestamps are rewritten to the replay wall clock so the dashboard's
    time displays look live instead of showing the recording date.
    """
    scenes, total = load_storyboard(storyboard_path)
    base = Path(base_dir)

    def load_events(name: str) -> list[RuntimeEvent]:
        events = list(EventLog(base / name).read())
        if not events:
            raise ValueError(f"replay file {base / name} is empty or missing")
        return events

    schedule = build_schedule(scenes, total, load_events)
    await _wait_for_subscriber(broadcaster, subscriber_timeout)
    await asyncio.sleep(lead_in)

    loop = asyncio.get_running_loop()
    wall0 = time.time()
    mono0 = loop.time()
    for offset, event in schedule:
        delay = offset - (loop.time() - mono0)
        if delay > 0:
            await asyncio.sleep(delay)
        event.timestamp = wall0 + offset
        broadcaster.publish(event)
