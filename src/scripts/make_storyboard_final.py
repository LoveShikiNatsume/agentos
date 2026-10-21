#!/usr/bin/env python3
"""Generate runs/demo_video/storyboard_final.json from the qhd_refined
storyboard.  Captions / beats / slow_motion / replay slices / data_card are
copied programmatically so banner texts and result numbers stay verbatim
(spec 2026-07-23, sec 4).  Run once; commit the output."""
from __future__ import annotations

import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[2] / "demo" / "replay"
qhd = {s["scene"]: s
       for s in json.loads((BASE / "storyboard_qhd_refined.json")
                           .read_text(encoding="utf-8"))["scenes"]}


def carry(name: str, *keys: str) -> dict:
    """Copy selected keys verbatim from the qhd_refined scene."""
    src = qhd[name]
    return {k: src[k] for k in keys if k in src and src[k] is not None}


scenes = []

scenes.append({
    "scene": "intro", "start": 0, "duration": 6,
    "page": "local", "camera": None, "caption": None, "label": "",
    "stinger": {"title": "AgentOS Runtime",
                "subtitle": "面向多 Agent 协作的操作系统级执行时",
                "hold": 3.2, "variant": "title"},
    "replay": {"file": "act12_pipeline.jsonl",
               "segments": [{"to_frac": 0.08, "weight": 1.0}]},
})

act1 = carry("act1_acb", "camera", "caption", "beats")
act1.update({
    "scene": "act1_acb", "start": 6, "duration": 32, "page": "local",
    "label": "01 / 04 · Agent 即进程 · ACB",
    "stinger": {"title": "第一幕 · Agent 即进程（ACB）",
                "subtitle": "统一生命周期、执行状态与资源归属"},
    "replay": {"file": "act12_pipeline.jsonl",
               "segments": [{"from_frac": 0.08,
                             "to_marker": "task.dynamic_submitted",
                             "weight": 1.0}]},
})
scenes.append(act1)

act2 = carry("act2_dag", "camera", "caption", "beats", "slow_motion", "replay")
act2.update({
    "scene": "act2_dag", "start": 38, "duration": 50, "page": "local",
    "label": "02 / 04 · TaskDAG 依赖调度",
    "stinger": {"title": "第二幕 · TaskDAG 依赖调度",
                "subtitle": "同层并发 · 运行时动态插入任务"},
})
scenes.append(act2)

a3 = carry("act3_context", "camera", "caption", "beats", "replay")
a3.update({
    "scene": "act3_context", "start": 88, "duration": 30, "page": "local",
    "label": "03 / 04 · Context 即内存",
    "stinger": {"title": "第三幕 · Context 即内存（ContextRef）",
                "subtitle": "共享段复用 · 私有段隔离"},
})
scenes.append(a3)

a3c = carry("act3_cache", "camera", "caption", "beats", "replay")
a3c.update({
    "scene": "act3_cache", "start": 118, "duration": 34, "page": "local",
    "label": "03 / 04 · KV Cache 亲和",
})
scenes.append(a3c)

a4 = carry("act4_fault", "camera", "caption", "beats", "replay",
           "spotlight_extra")
a4.update({
    "scene": "act4_fault", "start": 152, "duration": 55, "page": "local",
    "label": "04 / 04 · 故障注入与局部恢复",
    "stinger": {"title": "第四幕 · 故障注入与局部恢复",
                "subtitle": "fault.injected → snapshot → context_restored"},
})
scenes.append(a4)

res = carry("results", "camera", "beats", "data_card")
res.update({
    "scene": "results", "start": 207, "duration": 18, "page": "local",
    "caption": None, "label": "关键结果 · 可溯源",
})
scenes.append(res)

scenes.append({
    "scene": "outro", "start": 225, "duration": 4,
    "page": "local", "camera": None, "caption": None, "label": "",
    "fade_out": 0.5,
    "stinger": {"title": "演示结束 · 数据可复现",
                "subtitle": "runs/demo_video/ · 答辩现场可回放",
                "hold": 3.0, "variant": "outro"},
})

out = {"total_duration": 229, "scenes": scenes}
target = BASE / "storyboard_final.json"
target.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n",
                  encoding="utf-8")
print(f"wrote {target}")
