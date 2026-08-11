# bgm-montage v1.1 使用说明

## 目录

- [1. 环境与安装](#1-环境与安装)
- [2. 统一入口](#2-统一入口)
- [3. 单阶段入口与统一缓存契约](#3-单阶段入口与统一缓存契约)
- [4. 缓存与跨项目素材复用](#4-缓存与跨项目素材复用)
- [5. 素材充足度与低人脸策略](#5-素材充足度与低人脸策略)
- [6. 风格和语法如何进入成片](#6-风格和语法如何进入成片)
- [7. 已实现与未实现范围](#7-已实现与未实现范围)
- [8. 自动化测试与打包](#8-自动化测试与打包)

## 1. 环境与安装

推荐 CPython 3.11。`requirements.lock.txt` 是可复现安装入口；当前锁文件由 CPython 3.11.9 / Windows 11 x64 环境生成并验证。Linux、Docker 的 ZIP 解压结构兼容，但本版本不宣称已在这些系统完成现场运行验证。

系统还必须提供 `ffmpeg` 和 `ffprobe`，且统一入口渲染时二者需位于 `PATH`。第一次启用真实视觉语义时需要联网下载默认模型 `openai/clip-vit-base-patch32`；之后可使用本机模型缓存。

标准项目内安装位置是：

```text
<project>/.agents/skills/bgm-montage
```

在 PowerShell 中安装：

```powershell
$ProjectRoot = ".\造球计划"
$SkillRoot = Join-Path $ProjectRoot ".agents\skills\bgm-montage"
Set-Location -LiteralPath $ProjectRoot

py -3.11 -m venv (Join-Path $SkillRoot ".venv")
& (Join-Path $SkillRoot ".venv\Scripts\python.exe") -m pip install --upgrade pip
& (Join-Path $SkillRoot ".venv\Scripts\python.exe") -m pip install -r (Join-Path $SkillRoot "requirements.lock.txt")
& (Join-Path $SkillRoot ".venv\Scripts\python.exe") -m pip check
ffmpeg -version
ffprobe -version
```

将示例配置复制到项目根目录并只填写本机 Key：

```powershell
Copy-Item -LiteralPath (Join-Path $SkillRoot ".env.example") -Destination (Join-Path $ProjectRoot ".env")
```

```dotenv
PIXABAY_API_KEY=your_pixabay_api_key_here
```

Skill 代码旁的 `.env` 不会作为凭证来源。若 Skill 不在当前项目内，先为当前 PowerShell 进程设置项目根目录；不要把此路径依赖写进发布包：

```powershell
$env:BGM_MONTAGE_PROJECT_ROOT = ".\my-montage-project"
```

## 2. 统一入口

统一入口是生产运行方式，也是在素材充足度检查中使用目标时长的路径：

```powershell
$ProjectRoot = ".\造球计划"
$SkillRoot = Join-Path $ProjectRoot ".agents\skills\bgm-montage"
$Python = Join-Path $SkillRoot ".venv\Scripts\python.exe"
$env:BGM_MONTAGE_PROJECT_ROOT = $ProjectRoot

& $Python (Join-Path $SkillRoot "scripts\bgm_montage.py") `
  --bgm (Join-Path $ProjectRoot "音频素材\test.wav") `
  --theme "industrial future city" `
  --duration 12 `
  --ratio 9:16 `
  --output-dir (Join-Path $ProjectRoot "成片") `
  --reference-dir (Join-Path $ProjectRoot "参考视频") `
  --material-dir (Join-Path $ProjectRoot "视频素材") `
  --cache-dir (Join-Path $ProjectRoot ".bgm-montage-cache")
```

必填参数为 `--bgm`、`--theme`、`--duration`、`--ratio` 和 `--output-dir`。可选参数：

- `--project-name`：输出项目目录和文件名使用的短名称。
- `--run-id`：指定本次运行目录名；目录已存在时拒绝覆盖。未指定时自动生成 UTC 时间戳和随机后缀。
- `--assets`：覆盖按时长推导的候选入选数。
- `--min-width`、`--min-height`：Pixabay 高清候选最低分辨率，默认 1280×720。
- `--allow-semantic-fallback`：仅在用户明确接受时使用。CLIP 不可用时继续 OpenCV 结构统计，并在画像中标注降级；默认不降级。

支持 `9:16`、`16:9`、`1:1`、`4:5`、`5:4`、`3:4`、`4:3`，也支持 320–4096 像素范围内的偶数 `WIDTHxHEIGHT`。

输出位于：

```text
<output-dir>/<project-slug>/<run-id>/
```

其中包含：

- `<project-slug>_montage.mp4`：H.264/AAC、30 fps、BGM 正式音轨的成片。
- `style_profile.json`：参考视频视觉、节奏、字幕样式区域估计、镜头语义和主体区域画像。
- `editing_grammar.json`：参考音频事件与切点、镜头时长、景别/运动相邻关系和结尾时序语法。
- `bgm_profile.json`：时长、BPM、节拍、重音、乐句、段落、能量、音色、人声概率、停顿和循环估计。
- `edit_plan.json`：每个输出/源区间、速度、卡点原因、景别/运动目标、语法影响、转场和裁剪方案。
- `sources.json`：素材 ID、作者、页面、搜索词、本地路径、质量评分、复用方式和实际使用区间。
- `validation.json`：音视频流、完整解码、时长、分辨率、黑屏、冻结、静音、重复率、人脸预算和主体裁剪检查。
- `run_report.json`：各阶段状态和产物路径，不含凭证。
- `validation_frames/`：10%、50%、90% 三个代表帧，用于人工复核。

## 3. 单阶段入口与统一缓存契约

统一入口的 `--cache-dir` 接受项目缓存根目录。各单阶段入口接受自己的精确阶段目录；按下列约定传参即可与统一入口命中同一份缓存：

```text
<cache-root>/references
<cache-root>/bgm
<cache-root>/pixabay
```

### 3.1 参考视频视觉分析

```powershell
& $Python (Join-Path $SkillRoot "scripts\analyze_references.py") `
  --reference-dir (Join-Path $ProjectRoot "参考视频") `
  --cache-dir (Join-Path $ProjectRoot ".bgm-montage-cache\references") `
  --output (Join-Path $ProjectRoot ".bgm-montage-cache\references\style_profile.json") `
  --semantic-required
```

默认语义模型缓存位于项目的 `<cache-root>/models`（默认 `.bgm-montage-cache/models`）。单阶段运行可用 `--semantic-cache-dir` 指定位置。可在启动 Python 前设置 `BGM_MONTAGE_SEMANTIC_MODEL` 更换兼容的 CLIP 模型，或设置 `BGM_MONTAGE_SEMANTIC_OFFLINE=1` 只读本机缓存。`--no-semantics` 是明确的结构统计降级，不会伪造语义结果。

缓存按文件大小、纳秒 mtime 和采样/完整 SHA-256 指纹校验；仅新增或修改视频重新分析。缓存和输出路径若位于参考目录内会直接拒绝。

### 3.2 参考音频剪辑语法

```powershell
& $Python (Join-Path $SkillRoot "scripts\analyze_editing_grammar.py") `
  --reference-dir (Join-Path $ProjectRoot "参考视频") `
  --style-profile (Join-Path $ProjectRoot ".bgm-montage-cache\references\style_profile.json") `
  --cache-dir (Join-Path $ProjectRoot ".bgm-montage-cache\references") `
  --output (Join-Path $ProjectRoot ".bgm-montage-cache\references\editing_grammar.json") `
  --summary-only
```

该阶段从参考视频只读解码音频，缓存键包含源指纹、视觉切点摘要和分析器版本。没有音轨的视频会标记 `no_audio`；整个语料都无可用音轨时输出 `visual_only`，不会虚构音频语法。

### 3.3 本次 BGM 分析

```powershell
& $Python (Join-Path $SkillRoot "scripts\analyze_bgm.py") `
  --bgm ".\music\track.wav" `
  --target-duration 30 `
  --cache-dir (Join-Path $ProjectRoot ".bgm-montage-cache\bgm") `
  --output (Join-Path $ProjectRoot ".bgm-montage-cache\bgm\track_profile.json") `
  --summary-only
```

### 3.4 Pixabay 搜索、筛选与下载

```powershell
& $Python (Join-Path $SkillRoot "scripts\pixabay_pipeline.py") `
  --theme "industrial future city" `
  --style-profile (Join-Path $ProjectRoot ".bgm-montage-cache\references\style_profile.json") `
  --audio-profile (Join-Path $ProjectRoot ".bgm-montage-cache\bgm\track_profile.json") `
  --material-root (Join-Path $ProjectRoot "视频素材") `
  --cache-dir (Join-Path $ProjectRoot ".bgm-montage-cache\pixabay") `
  --desired-count 8 `
  --aspect-ratio 9:16 `
  --min-resolution 1280x720 `
  --result-json (Join-Path $ProjectRoot ".bgm-montage-cache\pixabay\stage_result.json")
```

使用 `--dry-run` 时只调用搜索与候选排序，不下载完整视频，也不执行依赖完整视频和目标时长的最终充足度检查。正式生产请使用统一入口。搜索固定执行三轮有界扩展：初始风格/音乐词、同义词和场景词、相关视觉概念；全部查询、缓存命中、筛选和拒绝原因写入主题目录的 `sources.json`。

### 3.5 来源区间与成片验证

```powershell
& $Python (Join-Path $SkillRoot "scripts\pixabay_pipeline.py") update-usage `
  --manifest ".\materials\theme\sources.json" `
  --edit-plan ".\renders\project\run-id\edit_plan.json"

& $Python (Join-Path $SkillRoot "scripts\validate_output.py") `
  ".\renders\project\run-id\project_montage.mp4" `
  --expected-duration 30 `
  --ratio 9:16 `
  --edit-plan ".\renders\project\run-id\edit_plan.json" `
  --report ".\renders\project\run-id\validation.json" `
  --frames-dir ".\renders\project\run-id\validation_frames"
```

时间线构建和渲染当前由统一入口调用 `montage.py` 的可复用函数；没有单独声明一个尚不存在的渲染 CLI。

## 4. 缓存与跨项目素材复用

项目缓存保持单层 Pixabay 命名空间：

```text
<project>/.bgm-montage-cache/
  references/
  bgm/
  pixabay/
    search/
    thumbnails/
    video_fingerprints.json
    material_index.json
```

v1.0 的 `pixabay/pixabay/search` 和 `pixabay/pixabay/thumbnails` 会复制迁移到单层目录；旧目录保留，便于回退。Pixabay JSON 搜索缓存 TTL 为 24 小时。

最终入选素材保存到 `<material-root>/<theme-slug>/`，文件名采用“序号 + 简洁中文内容描述 + Pixabay ID”；中文描述由已通过筛选的标签和本次主题生成。主题目录中的 `sources.json` 是该目录的搜索、淘汰、复用和来源账本。

机器级素材目录默认是：

- Windows：`%LOCALAPPDATA%\bgm-montage\material-library\material_index.json`
- Linux/macOS：`${XDG_CACHE_HOME:-~/.cache}/bgm-montage/material-library/material_index.json`

可在启动进程前设置 `BGM_MONTAGE_LIBRARY_ROOT` 覆盖。若另一主题或项目已下载相同 Pixabay ID，程序优先在当前主题目录创建硬链接；跨卷或文件系统不允许硬链接时直接引用已下载文件。它不会因为全局去重而拒绝素材，也不会再次下载。来源清单会记录 `reuse_mode` 和原始库项。

## 5. 素材充足度与低人脸策略

正式管线在三轮扩展和完整视频 QA 后执行硬门槛；仍不足时抛出 `InsufficientMaterialError`，保留 `sources.json`，不进入强行重复渲染。默认策略包括：

- 独立素材至少 `max(4, min(desired_count, ceil(duration / 3.5)))`。
- 单素材最多使用 2 次，成片占比不高于 30%。
- 6 秒及以上成片至少 3 个场景类别；更短成片至少 2 个。
- 低人脸风险素材至少覆盖所需独立素材数的 65%。
- 清晰正脸风险达到阈值的镜头总占比不高于 15%。
- 下载阶段的候选理论画面覆盖至少达到目标时长的 95%。

搜索优先扩展环境、建筑、自然、交通、航拍、广角、细节和无人场景。标签、缩略图与完整视频采样中的正脸几何会让自拍、采访、摆拍、人物特写和大面积人物明显降权；人物主题会减弱搜索评分惩罚，但时间线仍执行默认人脸画面预算。该检测不是人物身份识别，也不保证识别侧脸或遮挡人脸。

## 6. 风格和语法如何进入成片

`style_profile.json` 影响 Pixabay 查询、色调/运动/景别评分、参考镜头时长和统一调色。`editing_grammar.json` 在时间线中实际影响：

- 对新 BGM 的重音、节拍、乐句、段落和停顿候选切点赋权；
- 按低/中/高能量混合参考秒数与新 BGM 拍长来分配镜头时长；
- 依据参考景别与运动相邻矩阵选择下一镜头，同时对循环段落改变景别和运动目标；
- 使用参考结尾镜头时长倍率调整结尾节奏；
- 在每个镜头的 `grammar_influence` 中记录实际采用的字段。

当前参考分析只学习检测到的硬切相邻关系。渲染器能执行硬切、计划中明确给出的简单淡黑转场、统一 EQ 调色、比例适配、恒定片段变速和结尾淡出；它不会把未检测到的复杂效果伪装成已学习结果。

显著区域与正脸几何生成 `subject_profile`。目标比例接近源比例时直接适配；安全保留率达到 85% 且主体位置稳定时进行主体中心裁剪；否则使用模糊背景加完整前景的填充方式。验证报告会检查计划中的主体保留率，但这不是逐帧人物跟踪或审美保证，代表帧仍应人工复核。

## 7. 已实现与未实现范围

已实现：采样级镜头检测、结构统计、预训练 CLIP 零样本语义、显著区域/正脸几何、参考音频与切点对齐、硬切语法、BGM 结构分析、Pixabay 官方视频 API、三轮查询扩展、缩略图/完整视频 QA、指纹去重、跨项目复用、硬充足度门槛、主体感知裁剪/模糊填充、FFmpeg H.264/AAC 渲染、来源追踪和完整解码验证。

未实现：复杂转场识别或复刻、match cut、遮罩、speed ramp、字幕内容或样式复刻、OCR、动态图形、人物身份识别、可靠时序动作识别、逐帧主体跟踪。降级策略是使用硬切、恒速片段、统一调色和模糊填充；语义模型不可用时只有显式允许才能退回 OpenCV 结构画像。

## 8. 自动化测试与打包

运行项目测试：

```powershell
& $Python -m pytest -q (Join-Path $SkillRoot "tests")
```

发布包通过严格 allowlist 创建，成员名统一使用 `/`，根前缀为 `.agents/skills/bgm-montage/`。真实 `.env`、Key、`.venv`、模型、缓存、素材和测试成片不会进入 ZIP。默认拒绝覆盖已有 ZIP；只有明确传 `--force` 才会替换。

```powershell
& $Python (Join-Path $SkillRoot "scripts\package_skill.py") `
  --version 1.1 `
  --output ".\skills\造球计划\bgm-montage-v1.1.zip"
```

PowerShell 解压到项目根目录：

```powershell
Expand-Archive -LiteralPath ".\downloads\bgm-montage-v1.1.zip" -DestinationPath ".\my-montage-project"
```

Linux 或 Docker 构建环境可使用 `unzip bgm-montage-v1.1.zip -d /workspace/project`，或 `python -m zipfile -e bgm-montage-v1.1.zip /workspace/project`。这说明 ZIP 布局和分隔符兼容，不代表本版本已完成 Linux/Docker 现场依赖与渲染验证。
