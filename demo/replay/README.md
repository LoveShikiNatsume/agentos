# demo_video 目录说明

## 母带（回放数据源）
- `act12_pipeline.jsonl` / `act3_cache.jsonl` / `act4_fault.jsonl`
- 当前版本：mock 后端排练母带（事件含 mode=mock）。
- **正式录制前必须重新生成真实母带并冻结**：
  `DEEPSEEK_API_KEY=sk-xxx .venv/bin/python scripts/demo_video.py --backend deepseek --auto --no-browser --exit-after-acts`
  生成后在本 README 记录日期与 git commit，不再改动。

## 导演模式
- 分镜脚本：`storyboard.json`（时长/镜头/字幕/结果数字的唯一真相源；
  结果数字必须逐一对照 `国赛准备/数据口径登记表.md`）
- **2K 精修分镜**：`storyboard_qhd_refined.json`（专业总结 banner、证据卡片化结果页、结尾 3–5s）
- 排练：`.venv/bin/python scripts/replay_dashboard.py` 然后浏览器打开
  `http://127.0.0.1:9420/?director=1`
- 一键录制（1080p）：`.venv/bin/python scripts/record_demo.py`
  产物在 `recordings/demo_raw.webm`（有 ffmpeg 则同时出 mp4）
- **2K 精修录制（2560×1440）**：
  ```bash
  .venv/bin/python scripts/record_demo.py \
    --width 2560 --height 1440 \
    --storyboard runs/demo_video/storyboard_qhd_refined.json \
    --filename demo_2k.webm
  ```
  产物：`recordings/demo_2k.webm`

## 定稿视觉（2026-07-23 spec）
- 分镜：`storyboard_final.json`（原位聚焦运镜 / 细横幅章节 / 数据先行开场）
- **高清录制（推荐，CDP 逐帧 PNG 无损抓图 + ffmpeg CRF12 合成）**：
  ```bash
  .venv/bin/python scripts/record_demo_frames.py \
    --storyboard runs/demo_video/storyboard_final.json \
    --filename demo_final.mp4 \
    --crf 12
  ```
  交付物：`recordings/demo_final.mp4`（2560×1440 / 30fps；源为 PNG 帧，文字与运镜清晰）
  - 同管线产物：`demo_final_hq.mp4`（CRF12）；旧低码率备份：`demo_final_lowbitrate_backup.mp4`
- 旧管线（Playwright 内置录像，源码率 ~0.7Mbps 不可调，运镜段发软）：
  ```bash
  .venv/bin/python scripts/record_demo.py \
    --width 2560 --height 1440 \
    --storyboard runs/demo_video/storyboard_final.json \
    --filename demo_final.webm
  ```
- 正式定版前仍需按上文「母带」一节重新生成真实 API 母带后重录

## 母带冻结记录
- （录好真实母带后在此登记：日期 / commit / 生成命令）
