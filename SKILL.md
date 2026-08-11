---
name: bgm-montage
description: Learn a structured style profile and audio-linked editing grammar from read-only reference videos, analyze BGM structure, search and reuse Pixabay footage, plan a music-event timeline, render a low-repetition montage, and enforce automatic media QA. Use when Codex must create a stock-footage montage from a BGM path, theme, duration, aspect ratio, and output directory.
---

# BGM Montage v1.2

把只读参考视频、BGM 与 Pixabay 素材转成可追踪、可复现的自动混剪。推荐使用 CPython 3.11 和 Skill 自有 `.venv`；系统必须能在 `PATH` 中找到 `ffmpeg` 与 `ffprobe`。

## 统一入口

真实 API Key 只从当前项目根目录 `.env` 或进程环境变量 `PIXABAY_API_KEY` 读取。不要把 Key 写入源码、命令输出、日志、报告或 ZIP。

从项目根目录运行：

```powershell
& ".\.agents\skills\bgm-montage\.venv\Scripts\python.exe" `
  ".\.agents\skills\bgm-montage\scripts\bgm_montage.py" `
  --bgm ".\music\track.wav" `
  --theme "quiet coastal solitude" `
  --duration 30 `
  --ratio 9:16 `
  --output-dir ".\renders"
```

每次运行写入 `<output-dir>/<project-slug>/<run-id>/`。未传 `--run-id` 时自动生成 UTC 时间戳加随机后缀；已有目录不会被静默覆盖。失败运行可用完全相同的输入、原 `--run-id` 和 `--resume-run` 继续。

## 执行约束

1. 参考视频目录始终只读：不得移动、重命名、修改或覆盖其中任何文件，也不得把缓存或输出写到该目录内。
2. 优先运行统一入口；仅在排错或复用中间产物时使用单阶段入口。所有入口共用 `<project>/.bgm-montage-cache/{references,bgm,pixabay}`。
3. 参考语义分析默认尝试本地/预训练 CLIP。CLIP 不可用时，统一入口默认失败；只有明确接受结构化降级时才传 `--allow-semantic-fallback`。降级结果必须保留 unavailable/degraded 状态，不得宣称已完成真实语义识别。
4. Pixabay 在下载前先建立音乐事件驱动的镜头槽位，并要求元数据候选池、重点槽位覆盖、独立来源、场景多样性、可用时长和低人脸库存达标。搜索扩展后仍不足时明确失败，不得用高度重复素材强行成片。
5. 已下载 Pixabay 素材可通过统一素材索引跨主题、跨项目复用；同一成片仍受来源复用次数、累计画面占比、重复间隔和源区间重叠限制。
6. 只在最终成片通过完整解码、音视频流、时长、分辨率、帧率、音量、黑帧、冻结、静音、尾部碎片镜头、重复率、裁剪、卡点和高潮响应检查后提交使用历史。确定性 QA 失败时会重新分配素材并重渲染，默认最多返工 2 次；全部失败则不交付成片。

## 核心产物

- `audiomap.json`：唯一可信的 BGM 分析；包含节拍、onset、能量、密度、静音、hard stop、drop、surge、climax、乐句、段落及 `beat_cut` / `phrase_flow` 判断。
- `timeline.json`：下载前的音乐事件镜头槽位计划。
- `asset_manifest.json`：候选、筛选、下载、复用、来源和实际使用区间。
- `edit_decisions.json`：逐镜头源区间、速度、裁剪、转场和选片理由。
- `render_report.json`：最终媒体探测、完整解码和结构化 QA 结果。

为兼容 v1.1，运行目录仍同时生成 `bgm_profile.json`、`edit_plan.json`、`sources.json` 和 `validation.json` 别名文件，并保留原有主要命令行参数。

## 已实现边界

参考学习会实际影响音乐边界吸附、镜头时长、段落密度、景别与运动目标、相邻镜头差异、少量转场和结尾结构。渲染仅支持硬切、淡入淡出（含淡黑）和短叠化，并可对单个源片段做安全范围内的恒速变速。

未实现或不宣称支持：复杂遮罩与 wipe、match cut、速度坡度、字幕内容或样式复刻、OCR、动态字幕/图形、人物身份识别、可靠的时序动作识别、逐帧主体跟踪，以及对参考视频复杂特效的自动复刻。主体裁剪来自采样显著区域/正脸几何；不安全时改用模糊背景填充或淘汰素材，不代表人工级构图保证。

安装、单阶段命令、缓存、配置、兼容参数和打包说明见 [references/usage.md](references/usage.md)。
