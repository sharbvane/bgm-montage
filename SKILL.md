---
name: bgm-montage
description: Learn a dynamic visual profile and audio-linked editing grammar from read-only references and BGM, intelligently search/reuse Pixabay or YouTube footage, select aesthetically strong and sequence-coherent shots, render a low-repetition montage, optionally export an editable JianYing Pro draft, and enforce automatic media QA. Use when Codex must create a stock-footage montage from a BGM path, theme, duration, aspect ratio, and output directory.
---

# BGM Montage v1.4

The unified entry defaults to `--source-provider youtube-first`: reuse high-quality assets from the machine-wide YouTube index, generate task-specific multi-round YouTube queries, download/analyze/filter, and invoke Pixabay only if hard candidate, diversity, quality, or style gates remain short. `--source-provider youtube` and `pixabay` remain explicit modes. `--search-query` is optional and additive. v1.4 adds scene-aware reference evidence, applies learned editing grammar in the production planner, emits a visual review packet, and accepts explicit YouTube source windows without replacing the proven render/QA core.

## Material usage mode

Use `local_evaluation` by default. In this mode, all material is for local learning, testing, algorithm validation, and montage quality evaluation only. Never use copyright, license, Creative Commons, Public Domain, No Copyright, Royalty Free, Free Stock, or ordinary-YouTube status as a search restriction, ranking feature, rejection reason, or report warning. Do not generate licensing-oriented queries unless the terms independently improve target-image recall. Rank by reference/style match, visual impact, aesthetics, environment dynamics, depth, composition, realism, image quality, motion, BGM fit, and whole-film consistency. Quality takes priority over source.

Switch to `publish` only when the user explicitly says the specific video will be publicly distributed, commercially used, or externally shared. Then define a task-specific rights policy. Never infer future publication. Use `--usage-mode publish` only after that explicit instruction.

把只读参考视频、BGM 与 YouTube/Pixabay 素材转成可追踪、可复现的自动混剪。推荐使用 CPython 3.11 和 Skill 自有 `.venv`；系统必须能在 `PATH` 中找到 `ffmpeg` 与 `ffprobe`，`yt-dlp` 必须由 `requirements.lock.txt` 安装到该环境。

## 统一入口

真实 API Key 只从当前项目根目录 `.env` 或进程环境变量 `PIXABAY_API_KEY` 读取。不要把 Key 写入源码、命令输出、日志、报告或 ZIP。

从项目根目录运行：

```powershell
& ".\.agents\skills\bgm-montage\.venv\Scripts\python.exe" `
  ".\.agents\skills\bgm-montage\scripts\bgm_montage.py" `
  --bgm "D:\music\track.wav" `
  --theme "quiet coastal solitude" `
  --duration 30 `
  --ratio 9:16 `
  --visual-style "冷蓝灰、雾气、大纵深、克制的航拍推进" `
  --output-dir "D:\renders"
```

每次运行写入 `<output-dir>/<project-slug>/<run-id>/`。未传 `--run-id` 时自动生成 UTC 时间戳加随机后缀；已有目录不会被静默覆盖。失败运行可用完全相同的输入、原 `--run-id` 和 `--resume-run` 继续。

## 执行约束

1. 参考视频目录始终只读：不得移动、重命名、修改或覆盖其中任何文件，也不得把缓存或输出写到该目录内。
2. 优先运行统一入口；仅在排错或复用中间产物时使用单阶段入口。所有入口共用 `<project>/.bgm-montage-cache/{references,bgm,pixabay}`。
3. 参考语义分析默认尝试本地/预训练 CLIP。CLIP 不可用时，统一入口默认失败；只有明确接受结构化降级时才传 `--allow-semantic-fallback`。降级结果必须保留 unavailable/degraded 状态，不得宣称已完成真实语义识别。
4. 先由用户要求、参考画像与 BGM 动态生成本次 `visual_style_profile.json`。检索词必须组合主题、情绪、环境/地点、光线天气、摄影方式和镜头运动；不得用固定地点或固定风格白名单代替任务推理。
5. “主题相关”只允许进入候选池。完整素材还必须通过纵深、构图、视觉冲击、光线氛围、色彩、摄影机运动、电影感、普通旅游记录风险、清晰度和可用区间检查；不足时扩搜或明确失败。
6. 已下载素材通过统一索引跨主题、跨项目复用。分析缓存带 schema、引擎版本和文件哈希；同一成片继续限制 canonical source、近似内容、累计占比、重复间隔和源区间重叠。
7. 选片在既有 BGM 槽位内增加世界观、色彩、时间天气、镜头语言和视觉连续性评分；运动方向、景别、颜色、亮度、纹理和构图关系优先由镜头本身完成，默认仍使用干净硬切。
8. 只在最终成片通过完整解码、音视频流、时长、分辨率、帧率、音量、黑帧、冻结、静音、尾部碎片、重复率、裁剪、卡点、高潮响应和序列一致性检查后提交使用历史。全部返工失败则不交付成片。

## 核心产物

- `audiomap.json`：唯一可信的 BGM 分析；包含节拍、onset、能量、密度、静音、hard stop、drop、surge、climax、乐句、段落及 `beat_cut` / `phrase_flow` 判断。
- `timeline.json`：下载前的音乐事件镜头槽位计划。
- `visual_style_profile.json`：本次任务动态形成的世界观、色彩、天气光线、摄影与质量画像。
- `asset_manifest.json`：检索轮次、审美筛选、缓存、去重、下载、复用、来源和实际使用区间。
- `edit_decisions.json`：v1.3 统一时间线真值；保留原始路径、源/目标区间、速度、裁剪/变换、BGM 独立轨和 v1.2 兼容字段。
- `render_report.json`：最终媒体探测、完整解码、音乐结构和全片视觉一致性 QA。
- `visual_review.json` / `visual_review.md`：开头、结尾、音乐事件和计划切点的可审查帧证据。
- `jianying_draft_report.json`（可选）：每镜独立、原素材引用和 BGM 独立轨的剪映草稿结构验证与未映射差异。

为兼容旧项目，运行目录仍生成 `bgm_profile.json`、`edit_plan.json`、`sources.json` 和 `validation.json` 别名；v1.3 `edit_decisions` 保留 v1.2 字段并提供显式迁移。

## 已实现边界

参考学习会在生产下载前计划中影响音乐边界吸附、镜头时长、段落密度、景别与运动目标、少量转场和结尾结构。v1.4 的视觉匹配仍是可审计的采样信号与元数据启发式，不等同于人工审美保证；轻度调色只做归一化，不能用来挽救本来不兼容的素材。

未实现或不宣称支持：复杂遮罩/wipe、语义级轮廓变形 match cut、速度坡度、字幕/OCR/动态图形、人物身份、可靠时序动作、逐帧主体跟踪及复杂特效复刻。剪映适配只映射原生可表达的切点、恒速和基础变换；裁剪或 FFmpeg 调色无法可靠映射时必须写入差异报告。

安装、单阶段命令、缓存、配置、兼容参数和打包说明见 [references/usage.md](references/usage.md)。
