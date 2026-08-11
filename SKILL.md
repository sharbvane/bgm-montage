---
name: bgm-montage
description: Learn shot-level visual semantics and audio-linked editing grammar from read-only reference videos, analyze a BGM, search and rank Pixabay footage, enforce diversity and low-face-use policies, and render a traceable music-driven montage. Use when Codex must create or refresh a stock-footage montage from a BGM path, theme, duration, aspect ratio, and output directory.
---

# BGM Montage v1.1

运行可复现、可追溯的“参考风格学习 + BGM 驱动 Pixabay 自动混剪”，并始终把参考视频目录视为只读。

## 运行流程

1. 使用 Python 3.11、Skill 本地 `.venv`，并确认 `ffmpeg`、`ffprobe` 在 `PATH`。
2. 真实 Pixabay Key 只放项目根目录 `.env` 的 `PIXABAY_API_KEY`；不得输出 Key 或带 Key 的 URL。
3. 默认通过统一入口运行。首次语义分析会获取预训练 CLIP 模型；若模型不可用，统一入口默认失败。只有用户明确接受结构统计降级时才传 `--allow-semantic-fallback`。
4. 检查运行目录中的 `style_profile.json`、`editing_grammar.json`、`bgm_profile.json`、`edit_plan.json`、`sources.json`、`validation.json` 与 `run_report.json`。
5. 仅在素材充足度和完整成片验证都通过时交付视频；素材不足必须保留搜索/淘汰记录并明确失败。

从项目根目录执行：

```powershell
& ".\.agents\skills\bgm-montage\.venv\Scripts\python.exe" `
  ".\.agents\skills\bgm-montage\scripts\bgm_montage.py" `
  --bgm ".\music\track.wav" `
  --theme "quiet coastal solitude" `
  --duration 30 `
  --ratio 9:16 `
  --output-dir ".\renders"
```

每次运行写入 `<output-dir>/<project-slug>/<run-id>/`。`run_id` 默认是 UTC 时间戳加随机后缀；指定的 `--run-id` 已存在时直接失败，不静默覆盖。

## 已实现的决策链

- 参考视频：OpenCV 统计加预训练 CLIP 零样本语义，按检测镜头输出主体、场景、表观动作、景别、构图、运动、情绪、搜索关键词和显著主体区域。
- 参考音频：把检测切点与强重音、强弱拍、乐句、段落、停顿、能量及结尾时序对齐，生成缓存的 `editing_grammar.json`。
- 搜索与筛选：风格画像、语义关键词、BGM 阶段意图和低人脸策略共同影响英文查询及评分；只下载最终候选的高清版本。
- 时间线：语法中的事件权重、分能量镜头时长、景别/运动相邻矩阵和结尾时长会影响实际选片与切点；`edit_plan.json` 逐镜头记录影响字段。
- 安全裁剪：显著区域与正脸几何决定主体感知裁剪；主体保留不足或比例转换过激时改用模糊背景填充。
- 硬门槛：独立素材数、场景数、低人脸素材数、单素材最多复用次数、单素材画面占比和正脸画面占比均受约束。

## 真实性边界

当前学习范围是采样镜头、硬切时序和可解释的统计/零样本估计。没有实现或宣称复刻复杂转场、match cut、遮罩、speed ramp、字幕样式、OCR、动态图形或人物身份识别。CLIP 的动作和情绪是单帧外观估计，不是时序动作识别；文字区域检测不是 OCR；显著区域裁剪不是人物跟踪。模型不可用时只允许显式降级，输出会标注 degraded。

## 不可破坏约束

- 不移动、不重命名、不修改、不覆盖参考目录中的任何文件。
- 不把 `.env`、`.venv`、模型、缓存、下载素材或测试成片放入发布 ZIP。
- 保留 Pixabay 素材 ID、作者、页面链接、搜索词、本地路径、复用方式和实际使用区间。
- 先复用跨项目素材库中的同一 Pixabay 素材；可硬链接时建立主题内硬链接，否则引用已有文件，不重复下载。

安装、单阶段命令、缓存目录、门槛、降级方式、测试和打包说明见 [references/usage.md](references/usage.md)。
