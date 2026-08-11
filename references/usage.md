# bgm-montage v1.2 使用说明

## 1. 运行环境

推荐并验证的 Python 主版本是 **CPython 3.11**。依赖锁定文件面向 CPython 3.11.9 / Windows 11 x64；系统还必须提供可执行的 `ffmpeg` 和 `ffprobe`。

Skill 的标准项目结构是：

```text
<project>/.agents/skills/bgm-montage/
  SKILL.md
  agents/openai.yaml
  scripts/
  references/
  tests/
  requirements.lock.txt
  .env.example
```

Windows PowerShell 安装：

```powershell
$ProjectRoot = ".\造球计划"
$SkillRoot = Join-Path $ProjectRoot ".agents\skills\bgm-montage"
$Venv = Join-Path $SkillRoot ".venv"
$Python = Join-Path $Venv "Scripts\python.exe"

Set-Location -LiteralPath $ProjectRoot
py -3.11 -m venv $Venv
& $Python -m pip install --upgrade pip
& $Python -m pip install -r (Join-Path $SkillRoot "requirements.lock.txt")
& $Python -m pip check
ffmpeg -version
ffprobe -version
```

若 Skill 安装在 `.\bgm-montage`，使用该目录中的 `.venv`，并在运行前显式设置项目根目录：

```powershell
$env:BGM_MONTAGE_PROJECT_ROOT = ".\造球计划"
```

项目内标准布局的源码与全局 Codex 安装可以共存；缓存、参考视频、素材和输出都归当前项目，不放进 Skill 发布包。

从发布 ZIP 安装到全局 Codex Skill 时，先解压到临时目录，再同步 ZIP 内部的标准路径；不要把 ZIP 直接解压到全局 `skills` 根目录，否则会形成错误的 `.agents/skills` 嵌套。以下流程保留已有 `.venv`，并先备份旧源码：

```powershell
$Package = ".\skills\造球计划\bgm-montage-v1.2.zip"
$GlobalSkill = Join-Path $env:USERPROFILE ".codex\skills\bgm-montage"
$Stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$Stage = Join-Path ([IO.Path]::GetTempPath()) "bgm-montage-v1.2-$Stamp"
$Backup = Join-Path $env:USERPROFILE ".codex\skills\.backups\bgm-montage-$Stamp"

New-Item -ItemType Directory -Path $Stage, $Backup -Force | Out-Null
if (Test-Path -LiteralPath $GlobalSkill) {
  robocopy $GlobalSkill $Backup /E /XD .venv .pytest_cache __pycache__ /XF .env *.pyc *.pyo
  if ($LASTEXITCODE -ge 8) { throw "bgm-montage backup failed: $LASTEXITCODE" }
}
Expand-Archive -LiteralPath $Package -DestinationPath $Stage
$StagedSkill = Join-Path $Stage ".agents\skills\bgm-montage"
if (-not (Test-Path -LiteralPath (Join-Path $StagedSkill "SKILL.md"))) {
  throw "The ZIP does not contain the standard Skill path"
}
New-Item -ItemType Directory -Path $GlobalSkill -Force | Out-Null
robocopy $StagedSkill $GlobalSkill /E /XD .venv .pytest_cache __pycache__ /XF .env *.pyc *.pyo
if ($LASTEXITCODE -ge 8) { throw "bgm-montage install failed: $LASTEXITCODE" }
```

## 2. API Key 与路径

复制示例配置到项目根目录，而不是 Skill 目录：

```powershell
Copy-Item -LiteralPath (Join-Path $SkillRoot ".env.example") `
  -Destination (Join-Path $ProjectRoot ".env")
```

`.env` 内容：

```dotenv
PIXABAY_API_KEY=your_pixabay_api_key_here
```

也可在启动进程前直接设置 `PIXABAY_API_KEY`。程序不会把 Key 或带 Key 的请求 URL 写入源码、日志、JSON 报告或 ZIP。真实 `.env` 不应提交或打包。

可选路径环境变量：

- `BGM_MONTAGE_PROJECT_ROOT`：Skill 位于项目外时指定当前项目根目录。
- `BGM_MONTAGE_LIBRARY_ROOT`：覆盖机器级 Pixabay 素材索引目录。
- `BGM_MONTAGE_SEMANTIC_MODEL`：替换兼容的 CLIP 模型 ID。
- `BGM_MONTAGE_SEMANTIC_OFFLINE=1`：只使用本机已有模型缓存，不联网获取模型。

所有路径由 `pathlib` 处理，支持中文、空格和 Windows 盘符。参考视频目录始终只读；程序会拒绝把输出、素材主题目录或缓存写入参考目录。

## 3. 统一入口

统一入口是完整生产路径，也是唯一会顺序执行参考学习、BGM 分析、下载前时间线、Pixabay 搜索、素材充足度检查、选片、渲染、QA 和使用历史提交的入口。

```powershell
$ProjectRoot = ".\造球计划"
$SkillRoot = Join-Path $ProjectRoot ".agents\skills\bgm-montage"
$Python = Join-Path $SkillRoot ".venv\Scripts\python.exe"
$env:BGM_MONTAGE_PROJECT_ROOT = $ProjectRoot

& $Python (Join-Path $SkillRoot "scripts\bgm_montage.py") `
  --bgm (Join-Path $ProjectRoot "音频素材\可用\示例.mp3") `
  --theme "serene mountain and coastal journey" `
  --duration 30 `
  --ratio 9:16 `
  --output-dir (Join-Path $ProjectRoot "成片") `
  --reference-dir (Join-Path $ProjectRoot "参考视频") `
  --material-dir (Join-Path $ProjectRoot "视频素材") `
  --cache-dir (Join-Path $ProjectRoot ".bgm-montage-cache")
```

必填参数：

- `--bgm`：输入音频或带音频的视频文件。
- `--theme`：主题/风格描述；可使用中文或英文，程序会生成英文视觉查询。
- `--duration`：期望秒数。实际成片不超过可用 BGM 时长，不会用静帧补齐超过源音频的部分。
- `--ratio`：`9:16`、`16:9`、`1:1`、`4:5`、`5:4`、`3:4`、`4:3`，或 320～4096 像素范围内的偶数 `WIDTHxHEIGHT`。
- `--output-dir`：输出根目录。

常用可选参数：

- `--project-name`：输出项目短名；默认由主题生成。
- `--run-id`：指定本次运行目录名；目录已存在时拒绝覆盖。
- `--resume-run`：继续一个失败运行，必须同时给出原 `--run-id`，且输入和关键配置必须与 `run_state.json` 一致。
- `--assets`：覆盖按时长/镜头槽位推导的最终入选素材数量。
- `--candidate-pool-multiplier`：每个计划镜头槽位所需的元数据候选倍数，默认 `6`。
- `--max-search-pages`：每轮扩展查询的 Pixabay 最大页数，默认 `3`。
- `--min-width` / `--min-height`：候选素材最小分辨率，默认 `1280` / `720`。
- `--max-reuse-per-asset`（兼容别名 `--max-source-reuse`）：同一 canonical source 在本片中的最大使用次数，默认 `1`。
- `--max-asset-screen-share`（兼容别名 `--max-source-share`）：单一来源累计画面占比，默认 `0.30`。
- `--min-repeat-gap-shots` / `--min-repeat-gap-seconds`：允许有限复用时的最小镜头/时间间隔，默认 `3` 镜头和 `6` 秒。
- `--max-rework-attempts`：第一次渲染 QA 失败后的最大自动重新选片/重渲染次数，默认 `2`。
- `--allow-semantic-fallback`：CLIP 不可用时明确允许参考分析使用 structural fallback。未传时统一入口要求 CLIP 可用。

### 不覆盖与断点续跑

默认输出结构：

```text
<output-dir>/<project-slug>/<run-id>/
```

未给 `--run-id` 时会生成 UTC 时间戳和随机后缀。运行目录存在且未传 `--resume-run` 时直接失败；不会静默覆盖历史成片。`--resume-run` 会校验 BGM 指纹、主题、时长、比例、素材/缓存路径和主要约束，然后复用已成功写出的阶段产物。已经成功且成片仍存在的运行会直接返回既有报告。

## 4. v1.2 产物契约

成功运行至少包含：

- `<project-slug>_montage.mp4`：H.264/AAC、30 fps 的正式成片。
- `audiomap.json`：全流程唯一可信的 BGM 结构分析。
- `timeline.json`：正式下载前生成的镜头槽位计划。
- `asset_manifest.json`：候选、搜索轮次、拒绝原因、来源、缓存/下载状态、质量信号、可用区间和实际使用区间。
- `edit_decisions.json`：逐镜头素材、源区间、速度、音乐锚点、景别/运动目标、裁剪、转场、内容策略和选片理由。
- `render_report.json`：最终成片的结构化媒体 QA。
- `style_profile.json`：参考视频的视觉/节奏/语义风格画像。
- `editing_grammar.json`：参考音频事件与参考切点的结构化关系。
- `run_state.json`：断点续跑输入摘要与指纹。
- `run_report.json`：各阶段状态、产物路径和最终通过/失败状态。
- `attempts/attempt_XX/`：每次选片、渲染、QA 的计划、报告和检查帧。

v1.1 兼容别名仍会生成：

| v1.2 主产物 | v1.1 兼容别名 |
|---|---|
| `audiomap.json` | `bgm_profile.json` |
| `asset_manifest.json` | `sources.json` |
| `edit_decisions.json` | `edit_plan.json` |
| `render_report.json` | `validation.json` |

旧命令的 `--bgm`、`--theme`、`--duration`、`--ratio`、`--output-dir`、`--reference-dir`、`--material-dir`、`--cache-dir`、`--assets`、`--min-width`、`--min-height` 继续兼容。

## 5. 音乐分析与时间线

`audiomap.json` 由同一个分析器生成并按 BGM 内容指纹与分析配置缓存，包含：

- 总时长、估计 BPM、beat/downbeat、onset 与 accent；
- 能量曲线、密度、静音/低能量区间；
- hard stop、drop、surge、climax 候选；
- 乐句、段落边界、重复段落和 `intro` / `build` / `drop` / `break` / `climax` / `outro` 角色；
- 各段落推荐镜头时长、切换强度与节奏模式；
- `analysis_digest`，用于验证相同输入与配置的稳定结果。

节拍置信度、间隔稳定性、节拍覆盖和脉冲证据通过门槛时使用 `beat_cut`；舒缓、氛围、rubato 或节拍不稳定的音乐使用 `phrase_flow`。`phrase_flow` 优先吸附乐句、段落和能量事件，不会伪装成固定每 2/3/4 秒切镜头。

`timeline.json` 在素材下载前生成。每个槽位记录起止时间、段落角色、节奏模式、情绪、能量、推荐内容、景别、运动、重点镜头、音乐锚点和推荐转场。drop/climax 倾向更短镜头、宏大/高运动内容；intro/break/outro 倾向更长镜头。参考 `editing_grammar.json` 的可靠字段会参与边界权重、镜头时长、景别/运动衔接和结尾结构。

## 6. 素材搜索、复用和充足度

统一项目缓存：

```text
<project>/.bgm-montage-cache/
  references/
  bgm/
  pixabay/
    search/
    thumbnails/
    video_fingerprints.json
    material_index.json
  models/
```

单阶段入口既可接收项目缓存根，也可接收准确的 Pixabay 阶段目录；路径会规范化为单层 `pixabay`，不会再次追加成 `pixabay/pixabay`。旧的嵌套缓存只做非破坏性复制迁移，便于回退。

机器级素材索引默认位置：

- Windows：`%LOCALAPPDATA%\bgm-montage\material-library\material_index.json`
- Linux/macOS：`${XDG_CACHE_HOME:-~/.cache}/bgm-montage/material-library/material_index.json`

已经下载且哈希/来源匹配的 Pixabay 文件可跨主题和项目复用。可以创建硬链接时，在新主题目录创建硬链接；跨卷或文件系统不支持时直接引用已有文件。历史使用只参与轻度评分，不会把素材永久拉黑，也不会重复下载相同素材。

并行项目对同一 Pixabay ID 使用跨进程事务锁：取得锁后重新读取项目和机器级索引，先完成者下载并登记，等待方随后硬链接或共享引用。锁带 PID、owner token 和心跳，异常终止可回收；损坏索引或权限错误会中止而不是按空索引覆盖。同主题运行分别复制本次不可变 manifest 快照，因此不会被另一运行随后改写的共享 `sources.json` 污染。

同一成片默认每个 canonical source 只使用一次。即使文件名、主题目录或 Pixabay ID 不同，只要最终指纹/复用来源相同，仍按同一来源计算次数、累计占比、重复间隔和源区间重叠。

正式下载前先检查候选池：默认至少需要 `镜头槽位数 × 6` 个元数据候选，并验证重点槽位候选覆盖。搜索会依次使用主题、参考风格、BGM 段落意图、同义词、主体/场景/情绪/动作/环境和更宽但仍相关的视觉概念，同时增加查询页数并重新检索本地缓存。仍不足时抛出明确的素材不足错误并保留查询、淘汰和缺口记录，不进入重复渲染。

最终入选还检查独立来源、场景/主体/景别/运动多样性、连续可用时长、分辨率、清晰度、曝光、稳定性、文字/水印风险、比例、色调、人脸风险和片内占比。环境、建筑、自然、交通、无人场景、背影和俯拍默认优先；清晰正脸、人物特写、自拍、采访、摆拍和人物占画面较大的素材明显降权并受成片人脸占比约束。

下载后会分析素材内部区间，优先动作已开始且仍在进行、稳定、清晰、非黑帧/片头片尾、视觉变化足够的连续窗口；不会默认从源视频 0 秒开始，也不会用最后一帧冻结补时长。

## 7. 参考视频学习与 CLIP 降级

参考视频按文件大小、纳秒级修改时间和 SHA-256 指纹缓存，只重新分析新增或修改的文件。视觉结构分析包括镜头时长分布、分阶段节奏、运动强度/方向、景别、构图、颜色、亮度、转场提示、高潮密度、关键镜头位置和素材重复估计。

CLIP 是采样帧零样本语义增强，用于估计主体、场景、外观动作、情绪、构图、人类取景和搜索词。它不用于人物身份识别，也不是可靠的时序动作识别。显著区域和正脸几何用于主体感知裁剪；主体无法安全保留时使用模糊背景填充或淘汰素材。

统一入口默认要求 CLIP 成功加载；如果模型下载受限、依赖不可用或本机只允许离线运行，可以显式使用：

```powershell
--allow-semantic-fallback
```

此时继续使用 OpenCV 的颜色、清晰度、曝光、运动、切点、景别、构图、显著区域和人脸几何等 structural fallback。结果会明确标记模型不可用/降级，不会伪造语义分类。

## 8. 转场、变速和裁剪边界

渲染以硬切为主，并限制软转场占比。当前仅支持：

- 硬切；
- 淡入/淡出和淡黑；
- 短叠化；
- 单个源片段在安全范围内的恒速变速；
- 主体感知裁剪、完整画面 fit 或模糊背景填充；
- 统一亮度/饱和度/对比度和 BGM 淡入淡出。

未实现：复杂 wipe/遮罩、match cut、速度坡度、光流补帧、字幕内容/样式复刻、OCR、动态字幕/图形、人物身份识别、可靠时序动作识别、逐帧主体跟踪，以及参考视频复杂特效的自动检测与复刻。文档中的 `fade_like` / `dissolve_like` 仅是保守的参考风格提示，不代表可复刻任意特效。

## 9. 自动 QA 与返工

正式成片必须通过 `ffprobe` 和 FFmpeg 检查：

- 文件存在、非空、视频流/音频流存在；
- 完整解码；
- 总时长和音视频流时长；
- 分辨率和帧率；
- 平均音量和峰值；
- 长黑帧、冻结和异常静音；
- 计划尾镜过短、编码后尾镜缩水和末尾少于 8 帧/0.25 秒的突变闪画；
- 素材路径、源区间重叠、重复次数、画面占比和重复间隔；
- 相邻镜头多样性、明显人脸占比、主体裁剪安全；
- `beat_cut` / `phrase_flow` 对应的音乐事件对齐；
- climax/drop 相对舒缓段的镜头密度和视觉运动响应。

检查帧覆盖开头、结尾、段落边界、drop、surge、climax、hard stop 和稳定种子的随机时间点；渲染先把全部镜头量化到统一帧网格，避免逐段帧率取整累积后截短尾镜。确定性 QA 失败时，统一入口使用新的尝试种子重新分配素材、规划源区间并重渲染，默认最多返工 2 次。全部尝试失败时 `run_report.json` 保留诊断，运行返回失败，不把该成片宣称为成功，也不提交 Pixabay 使用历史。

QA 能发现确定性的媒体/时间线问题，但不能替代人工审美复核；CLIP 与显著区域裁剪都属于采样估计。

## 10. 单阶段入口

生产运行优先使用统一入口。以下命令用于调试、缓存预热或从已有中间产物继续。

### 10.1 参考视频视觉分析

要求 CLIP 可用：

```powershell
& $Python (Join-Path $SkillRoot "scripts\analyze_references.py") `
  --reference-dir (Join-Path $ProjectRoot "参考视频") `
  --cache-dir (Join-Path $ProjectRoot ".bgm-montage-cache\references") `
  --output (Join-Path $ProjectRoot ".bgm-montage-cache\references\style_profile.json") `
  --semantic-required
```

显式 structural fallback：把 `--semantic-required` 改为 `--no-semantics`。

### 10.2 参考音频剪辑语法

```powershell
& $Python (Join-Path $SkillRoot "scripts\analyze_editing_grammar.py") `
  --reference-dir (Join-Path $ProjectRoot "参考视频") `
  --style-profile (Join-Path $ProjectRoot ".bgm-montage-cache\references\style_profile.json") `
  --cache-dir (Join-Path $ProjectRoot ".bgm-montage-cache\references") `
  --output (Join-Path $ProjectRoot ".bgm-montage-cache\references\editing_grammar.json") `
  --summary-only
```

无音轨参考视频会标为 `no_audio`；整个语料都无可用音轨时输出 `visual_only`，不会虚构音频语法。

### 10.3 BGM 统一分析

```powershell
& $Python (Join-Path $SkillRoot "scripts\analyze_bgm.py") `
  --bgm ".\music\track.wav" `
  --target-duration 30 `
  --cache-dir (Join-Path $ProjectRoot ".bgm-montage-cache\bgm") `
  --output (Join-Path $ProjectRoot ".bgm-montage-cache\bgm\audiomap.json") `
  --summary-only
```

位置参数 `track.wav` 与 `--bgm track.wav` 都兼容。

### 10.4 下载前时间线

```powershell
& $Python (Join-Path $SkillRoot "scripts\timeline_planner.py") `
  --audiomap (Join-Path $ProjectRoot ".bgm-montage-cache\bgm\audiomap.json") `
  --style-profile (Join-Path $ProjectRoot ".bgm-montage-cache\references\style_profile.json") `
  --editing-grammar (Join-Path $ProjectRoot ".bgm-montage-cache\references\editing_grammar.json") `
  --duration 30 `
  --output (Join-Path $ProjectRoot ".bgm-montage-cache\bgm\timeline.json") `
  --summary-only
```

### 10.5 Pixabay 搜索/下载

```powershell
& $Python (Join-Path $SkillRoot "scripts\pixabay_pipeline.py") `
  --theme "serene mountain and coastal journey" `
  --style-profile (Join-Path $ProjectRoot ".bgm-montage-cache\references\style_profile.json") `
  --audio-profile (Join-Path $ProjectRoot ".bgm-montage-cache\bgm\audiomap.json") `
  --timeline-plan (Join-Path $ProjectRoot ".bgm-montage-cache\bgm\timeline.json") `
  --material-root (Join-Path $ProjectRoot "视频素材") `
  --cache-dir (Join-Path $ProjectRoot ".bgm-montage-cache\pixabay") `
  --desired-count 12 `
  --candidate-pool-multiplier 6 `
  --max-search-pages 3 `
  --aspect-ratio 9:16 `
  --min-resolution 1280x720 `
  --result-json (Join-Path $ProjectRoot ".bgm-montage-cache\pixabay\stage_result.json")
```

`--dry-run` 只做 API/缓存检索和候选排序，不下载完整视频，也不能替代统一入口对真实连续时长、完整视频质量和最终成片的检查。

### 10.6 使用区间与独立 QA

```powershell
& $Python (Join-Path $SkillRoot "scripts\pixabay_pipeline.py") update-usage `
  --manifest ".\renders\project\run-id\asset_manifest.json" `
  --edit-plan ".\renders\project\run-id\edit_decisions.json"

& $Python (Join-Path $SkillRoot "scripts\validate_output.py") `
  ".\renders\project\run-id\project_montage.mp4" `
  --expected-duration 30 `
  --ratio 9:16 `
  --expected-fps 30 `
  --edit-plan ".\renders\project\run-id\edit_decisions.json" `
  --audiomap ".\renders\project\run-id\audiomap.json" `
  --report ".\renders\project\run-id\render_report.json" `
  --frames-dir ".\renders\project\run-id\validation_frames"
```

## 11. 测试与打包

运行自动化测试：

```powershell
& $Python -m pytest -q (Join-Path $SkillRoot "tests")
```

构建 v1.2 ZIP：

```powershell
& $Python (Join-Path $SkillRoot "scripts\package_skill.py") `
  --version 1.2 `
  --output ".\skills\造球计划\bgm-montage-v1.2.zip"
```

ZIP 成员使用标准 `/` 分隔符，并以 `.agents/skills/bgm-montage/` 为根前缀。真实 `.env`、API Key、`.venv`、模型、缓存、下载素材、测试输出和成片不进入发布包。目标 ZIP 已存在时默认拒绝覆盖；只有明确传入 `--force` 才替换。

Windows 解压：

```powershell
Expand-Archive -LiteralPath ".\downloads\bgm-montage-v1.2.zip" `
  -DestinationPath ".\my-montage-project"
```

Linux/Docker 可使用 `unzip bgm-montage-v1.2.zip -d /workspace/project` 或 `python -m zipfile -e ...`。这只表示 ZIP 路径兼容；本版本不宣称已完成 Linux/Docker 的依赖、模型和 FFmpeg 现场运行验证。

## 12. 能力状态

已实现：确定性 `audiomap`、`beat_cut`/`phrase_flow`、下载前时间线、参考视觉与音频语法缓存、Pixabay 查询扩展、跨项目素材复用、候选/素材充足度门槛、片内来源约束、低人脸策略、素材内部最佳区间、相邻多样性、主体感知裁剪/填充、硬切/淡入淡出/短叠化、FFmpeg 渲染、完整解码 QA、音乐对齐/高潮响应检查、自动重新选片渲染、断点续跑和 v1.1 兼容别名。

实验性增强：采样帧 CLIP 零样本主体/场景/外观动作/情绪分类；采样显著区域与正脸几何裁剪；参考视频淡化/叠化提示。这些能力会在报告中保留模型和降级状态，不能当作逐帧真值。

尚未实现：复杂特效复刻、复杂遮罩/wipe、match cut、速度坡度、OCR、字幕样式复制、动态字幕/图形、人物身份识别、可靠时序动作识别和逐帧主体跟踪。
