# bgm-montage v1.4 使用说明

## YouTube-first acquisition

The unified entry defaults to `--source-provider youtube-first`. It first reuses machine-wide indexed YouTube assets, then generates task-specific multi-round queries from the visual profile, downloads and analyzes candidates, and invokes Pixabay only when hard candidate-pool, diversity, quality, or style-fit gates remain short. Explicit `youtube` and `pixabay` modes remain available. v1.4 adds scene-aware reference sampling, production-applied editing grammar, reviewable visual evidence, and explicit YouTube source windows. `yt-dlp` is installed by the formal lock file; a JavaScript runtime is still required for YouTube extraction.

```powershell
& $Python (Join-Path $SkillRoot "scripts\bgm_montage.py") `
  --source-provider youtube-first `
  --search-query "optional high-priority visual direction" `
  --youtube-results-per-query 8 `
  --youtube-max-download-candidates 30 `
  --exclude-youtube-id "REJECTED_ID" `
  --bgm "D:\music\track.mp3" `
  --theme "task-specific cinematic landscape" `
  --ratio 16:9 `
  --output-dir "D:\renders"
```

Provider-compatible manifests record the generated query plan, YouTube ID, title, channel, page URL, download section, local path, quality analysis, global index reuse, fallback reason, merge score, and usable source windows. `--search-query` is optional and additive; it never disables automatic query generation. After contact-sheet review, pass a curated manifest with `--asset-manifest PATH`; this skips acquisition only and leaves the stable editing core active.

For a reviewed long-form YouTube candidate, repeat `--youtube-source-window VIDEO_ID=START-END` to request absolute source seconds. The explicit interval overrides the automatic 38-second long-video window, is validated before acquisition, enters the resume fingerprint, and is written to the manifest. It only affects that ID if the candidate is actually discovered and downloaded.

`--usage-mode local_evaluation` is the default. It applies zero authorization/copyright weight, does not restrict ordinary YouTube material, does not generate license-oriented search terms, and suppresses repeated rights reminders in reports. Use `--usage-mode publish` only after the user explicitly states that the current video will be publicly distributed, commercially used, or externally shared; publication policy is then task-specific rather than inferred.

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
  requirements-jianying.lock.txt
  .env.example
```

Windows PowerShell 安装：

```powershell
$ProjectRoot = "E:\资料\造球计划"
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

若 Skill 安装在 `C:\Users\<用户名>\.codex\skills\bgm-montage`，使用该目录中的 `.venv`，并在运行前显式设置项目根目录：

```powershell
$env:BGM_MONTAGE_PROJECT_ROOT = "E:\资料\造球计划"
```

项目内标准布局的源码与全局 Codex 安装可以共存；缓存、参考视频、素材和输出都归当前项目，不放进 Skill 发布包。

从发布 ZIP 安装到全局 Codex Skill 时，先解压到临时目录，再同步 ZIP 内部的标准路径；不要把 ZIP 直接解压到全局 `skills` 根目录，否则会形成错误的 `.agents/skills` 嵌套。以下流程保留已有 `.venv`，并先备份旧源码：

```powershell
$Package = "E:\资料\skills\造球计划\bgm-montage-v1.4.zip"
$GlobalSkill = Join-Path $env:USERPROFILE ".codex\skills\bgm-montage"
$Stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$Stage = Join-Path ([IO.Path]::GetTempPath()) "bgm-montage-v1.4-$Stamp"
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
$ProjectRoot = "E:\资料\造球计划"
$SkillRoot = Join-Path $ProjectRoot ".agents\skills\bgm-montage"
$Python = Join-Path $SkillRoot ".venv\Scripts\python.exe"
$env:BGM_MONTAGE_PROJECT_ROOT = $ProjectRoot

& $Python (Join-Path $SkillRoot "scripts\bgm_montage.py") `
  --bgm (Join-Path $ProjectRoot "音频素材\可用\示例.mp3") `
  --theme "serene mountain and coastal journey" `
  --duration 30 `
  --ratio 9:16 `
  --visual-style "清透自然、晨雾、平缓推进、近景细节与宽景交替" `
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
- `--source-provider`：默认 `youtube-first`；也可显式选择 `youtube` 或 `pixabay`。
- `--search-query`：显式优先查询，可重复；它只提高优先级，不关闭动态查询扩展。
- `--exclude-youtube-id ID`：审片后排除指定 YouTube ID，可重复。
- `--youtube-source-window VIDEO_ID=START-END`：指定某个 YouTube 候选的绝对源时间窗，可重复；仅在该 ID 实际下载时生效。
- `--asset-manifest PATH`：使用已下载且已审片的素材清单，跳过素材获取但继续时间线、渲染与 QA。
- `--wide-aerial-only`：只保留航拍、广角或 FPV/跟随视角候选，并排除抽象、AI/CGI、微距特写、人物、动物和旗帜类元数据命中。
- `--visual-style`：本次任务的自由文本视觉方向；省略或传 `auto` 时由主题、参考画像和 BGM 自动形成。旧参数名 `--visual-cohesion-profile` 仍作为兼容别名，但不再接受固定枚举，也不会触发地点白名单。
- `--exclude-pixabay-id ID`：联系表实际画面复核后，按 Pixabay ID 硬性排除偏题素材；可重复传入。
- `--min-width` / `--min-height`：候选素材最小分辨率，默认 `1280` / `720`。
- `--max-reuse-per-asset`（兼容别名 `--max-source-reuse`）：同一 canonical source 在本片中的最大使用次数，默认 `1`。
- `--max-asset-screen-share`（兼容别名 `--max-source-share`）：单一来源累计画面占比，默认 `0.30`。
- `--min-repeat-gap-shots` / `--min-repeat-gap-seconds`：允许有限复用时的最小镜头/时间间隔，默认 `3` 镜头和 `6` 秒。
- `--max-rework-attempts`：第一次渲染 QA 失败后的最大自动重新选片/重渲染次数，默认 `2`。
- `--allow-semantic-fallback`：CLIP 不可用时明确允许参考分析使用 structural fallback。未传时统一入口要求 CLIP 可用。
- `--jianying-draft`：最终 MP4 通过 QA 后，额外从同一份 `edit_decisions.json` 生成剪映专业版可编辑草稿。
- `--jianying-draft-name` / `--jianying-draft-root` / `--jianying-python`：分别覆盖草稿名、剪映项目根和含 pyJianYingDraft 的独立 Python。未指定 Python 时会检查项目 `.venv-pyjianyingdraft`、Skill `.venv-jianying` 和当前解释器。

### 不覆盖与断点续跑

默认输出结构：

```text
<output-dir>/<project-slug>/<run-id>/
```

未给 `--run-id` 时会生成 UTC 时间戳和随机后缀。运行目录存在且未传 `--resume-run` 时直接失败；不会静默覆盖历史成片。`--resume-run` 会校验 BGM 指纹、主题、时长、比例、素材/缓存路径和主要约束，然后复用已成功写出的阶段产物。已经成功且成片仍存在的运行会直接返回既有报告。

## 4. v1.4 产物契约

成功运行至少包含：

- `<project-slug>_montage.mp4`：H.264/AAC、30 fps 的正式成片。
- `audiomap.json`：全流程唯一可信的 BGM 结构分析。
- `timeline.json`：正式下载前生成的镜头槽位计划。
- `visual_style_profile.json`：本次任务的动态世界观、色彩、时间天气、摄影、运动与审美门槛。
- `asset_manifest.json`：候选、搜索意图/轮次、拒绝原因、来源、缓存/下载状态、审美/语义/镜头元数据、哈希、可用区间和实际使用区间。
- `edit_decisions.json`：schema `1.3` 的统一时间线真值；逐镜保留原始路径、源/目标区间、恒速、裁剪/基础变换、视觉匹配评分，并含 BGM 独立轨。原 v1.2 字段仍保留。
- `render_report.json`：最终成片的结构化媒体、节奏和序列视觉一致性 QA。
- `visual_review.json` / `visual_review.md`：开头、结尾、BGM 事件和计划切点的时间戳、原因与检查帧；用于人工或代理审片，不替代硬 QA。
- `jianying_draft_report.json`（可选）：草稿路径、独立视频片段/BGM 轨验证、原素材引用和无法原生映射的差异。
- `style_profile.json`：参考视频的视觉/节奏/语义风格画像。
- `editing_grammar.json`：参考音频事件与参考切点的结构化关系。
- `run_state.json`：断点续跑输入摘要与指纹。
- `run_report.json`：各阶段状态、产物路径和最终通过/失败状态。
- `attempts/attempt_XX/`：每次选片、渲染、QA 的计划、报告和检查帧。

v1.1 兼容别名仍会生成：

| v1.4 主产物 | 兼容别名 |
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

正式下载前先检查候选池：默认至少需要 `镜头槽位数 × 6` 个元数据候选，并验证重点槽位候选覆盖。v1.3 先从本次画像提取主题、情绪、环境/地貌、用户明确地点、天气、光线、摄影方式与镜头运动，再按 precision、adjacent 和 quality-recall 多轮组合查询。地点完全来自用户/参考资料，不存在内置目的地白名单；泛词只在末轮补召回。仍不足时保留查询意图、淘汰和缺口记录并失败。

最终入选还检查独立来源、近似内容、连续可用时长、分辨率、清晰度、曝光、稳定性、文字/水印风险、纵深、构图、视觉冲击、光线氛围、色彩、摄影机运动价值、电影感和普通旅游记录风险。主题相关只代表候选资格；审美分低于动态门槛时自动尝试后续素材，不通过平淡镜头补满。人像主题会动态放宽人物风险，非人像主题仍限制显著正脸占比。

完整分析结果写入统一素材索引，包含语义/场景、世界/地貌族、景别、运动类型与方向、HSV 色彩、时间/天气、审美分、哈希和 Pixabay 来源。缓存由分析 schema、引擎版本和文件 SHA-256 共同校验；v1.2 旧字段保留，但第一次被 v1.3 选中时会升级分析。

下载后会分析素材内部区间，优先动作已开始且仍在进行、稳定、清晰、非黑帧/片头片尾、视觉变化足够的连续窗口；不会默认从源视频 0 秒开始，也不会用最后一帧冻结补时长。

## 7. 参考视频学习与 CLIP 降级

参考视频按文件大小、纳秒级修改时间和 SHA-256 指纹缓存，只重新分析新增或修改的文件。v1.4 先在全时域运行有界 FFmpeg 场景候选检测，再与均匀时间采样合并、去重并按上限保留覆盖；视觉结构分析包括镜头时长分布、分阶段节奏、运动强度/方向、景别、构图、颜色、亮度、转场提示、高潮密度、关键镜头位置和素材重复估计。场景检测失败时明确记录并回退既有均匀采样。

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

镜头顺序会在音乐槽位不变的前提下，对推进/拉远/俯冲/横移方向、景别、颜色亮度、世界族、纹理和构图连续性评分并做有限替换。它实现的是“镜头关系驱动的匹配剪辑”，不是额外堆叠特效。

未实现：复杂 wipe/遮罩、语义级轮廓变形 match cut、速度坡度、光流补帧、字幕内容/样式复刻、OCR、动态字幕/图形、人物身份识别、可靠时序动作识别、逐帧主体跟踪，以及参考视频复杂特效的自动检测与复刻。

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
- 全片世界观、色彩、时间天气、镜头语言和相邻视觉匹配分；严重断裂或平均一致性不足会使该选片尝试失败。

检查帧覆盖开头、结尾、段落边界、drop、surge、climax、hard stop、计划切点前后和稳定种子的随机时间点，并同时生成 `visual_review.json` / `visual_review.md`；渲染先把全部镜头量化到统一帧网格，避免逐段帧率取整累积后截短尾镜。确定性 QA 失败时，统一入口使用新的尝试种子重新分配素材、规划源区间并重渲染，默认最多返工 2 次。全部尝试失败时 `run_report.json` 保留诊断，运行返回失败，不把该成片宣称为成功，也不提交素材使用历史。

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
  --bgm "D:\music\track.wav" `
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
  --manifest "D:\renders\project\run-id\asset_manifest.json" `
  --edit-plan "D:\renders\project\run-id\edit_decisions.json"

& $Python (Join-Path $SkillRoot "scripts\validate_output.py") `
  "D:\renders\project\run-id\project_montage.mp4" `
  --expected-duration 30 `
  --ratio 9:16 `
  --expected-fps 30 `
  --edit-plan "D:\renders\project\run-id\edit_decisions.json" `
  --audiomap "D:\renders\project\run-id\audiomap.json" `
  --report "D:\renders\project\run-id\render_report.json" `
  --frames-dir "D:\renders\project\run-id\validation_frames"
```

### 10.7 剪映草稿适配

剪映依赖保持在独立环境，避免污染核心视频/机器学习依赖：

```powershell
$JianYingVenv = Join-Path $ProjectRoot ".venv-pyjianyingdraft"
py -3.11 -m venv $JianYingVenv
& (Join-Path $JianYingVenv "Scripts\python.exe") -m pip install `
  -r (Join-Path $SkillRoot "requirements-jianying.lock.txt")
```

单独从已有时间线创建草稿：

```powershell
& (Join-Path $JianYingVenv "Scripts\python.exe") `
  (Join-Path $SkillRoot "scripts\jianying_export.py") `
  "D:\renders\project\run-id\edit_decisions.json" `
  --draft-name "project_run-id_可编辑" `
  --report "D:\renders\project\run-id\jianying_draft_report.json"
```

适配器默认查找 `%LOCALAPPDATA%\JianyingPro\User Data\Projects\com.lveditor.draft` 并要求其中存在 `root_meta_info.json`。写入前备份根索引；已存在同名草稿时拒绝覆盖。每个视频镜头和 BGM 都是独立原始素材引用，不复制素材。切点、source range、target range、恒速、基础 scale/position/rotation/opacity 可原生映射；非完整裁剪矩形、FFmpeg 调色和无原生等价项写入 `unmapped_or_approximate`，不得静默伪装成功。

## 11. 测试与打包

运行自动化测试：

```powershell
& $Python -m pytest -q (Join-Path $SkillRoot "tests")
```

构建 v1.4 ZIP：

```powershell
& $Python (Join-Path $SkillRoot "scripts\package_skill.py") `
  --version 1.4 `
  --output "E:\资料\skills\造球计划\bgm-montage-v1.4.zip"
```

ZIP 成员使用标准 `/` 分隔符，并以 `.agents/skills/bgm-montage/` 为根前缀。真实 `.env`、API Key、`.venv`、模型、缓存、下载素材、测试输出和成片不进入发布包。目标 ZIP 已存在时默认拒绝覆盖；只有明确传入 `--force` 才替换。

Windows 解压：

```powershell
Expand-Archive -LiteralPath "D:\downloads\bgm-montage-v1.4.zip" `
  -DestinationPath "D:\my-montage-project"
```

Linux/Docker 可使用 `unzip bgm-montage-v1.4.zip -d /workspace/project` 或 `python -m zipfile -e ...`。这只表示 ZIP 路径兼容；本版本不宣称已完成 Linux/Docker 的依赖、模型、FFmpeg 或剪映现场运行验证。

## 12. 能力状态

已实现：v1.2 的确定性 `audiomap`、`beat_cut`/`phrase_flow`、下载前时间线、参考视觉/音频语法、跨项目素材复用、来源约束、内部最佳区间、裁剪、FFmpeg 渲染和完整 QA；v1.3 的动态多轴检索、审美评分/缓存、世界观与色彩画像、序列一致性、镜头视觉匹配、schema 迁移、BGM 独立轨和可选剪映可编辑草稿；以及 v1.4 的场景感知参考证据、生产时间线 grammar 接通、审片证据包和 YouTube 指定源时间窗。

实验性增强：采样帧 CLIP 零样本主体/场景/外观动作/情绪分类；采样显著区域与正脸几何裁剪；参考视频淡化/叠化提示。这些能力会在报告中保留模型和降级状态，不能当作逐帧真值。

尚未实现：复杂特效复刻、复杂遮罩/wipe、语义级轮廓变形 match cut、速度坡度、OCR、字幕样式复制、动态字幕/图形、人物身份识别、可靠时序动作识别、逐帧主体跟踪，以及 FFmpeg 调色到剪映的无损原生映射。
