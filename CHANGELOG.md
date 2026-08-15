# Changelog

## v1.4 — 2026-08-15

- Replaced reference-only uniform sampling with a bounded full-duration FFmpeg scene pass merged with temporal coverage. Structural analysis remains capped at 96 decoded frames, semantic inference at 18, and failures fall back to the exact v1.3.3 uniform schedule. Analyzer cache version `1.4.0` forces old sampled evidence to refresh.
- Connected `editing_grammar.json` to the production pre-download planner: reliability-weighted event preferences/offsets and energy duration, visual scale/motion matrices, transition distribution, and ending timing now alter real slots and record per-slot influence plus an applied-fields digest. Within v1.4, empty grammar produces the same plan as omitted grammar.
- Added `visual_review.json` and `visual_review.md`, reusing final-QA extraction for pinned opening/ending, BGM events, deterministic coverage, and up to five planned-cut before/after pairs. Successful-attempt frames and paths are promoted to the run root after the final MP4 move.
- Added repeatable `--youtube-source-window VIDEO_ID=START-END` support across unified, YouTube-first, and single-stage entry points, including input validation, resume fingerprints, manifest traceability, section-aware cache identity/reuse, and explicit-over-automatic window precedence.
- Kept the v1.3 edit schema, established FFmpeg renderer, QA gates, CLIP semantics, optional JianYing adapter, and dependency set; no Whisper, new renderer, or new runtime dependency was added.

## v1.3.3 — 2026-08-11

- Default provider strategy is now true `youtube-first`: machine-wide local reuse, dynamic multi-round YouTube discovery, sampled-frame filtering, automatic Pixabay fallback, provider-neutral merge/deduplication, and a hard combined sufficiency gate.
- Removed global storm vocabulary from YouTube query expansion and metadata scoring. Query plans now come from the same task-specific subject/environment/mood/weather/light/color/scale/depth/camera model used by the existing visual intelligence layer; explicit queries are optional high-priority additions.
- Added the machine-wide YouTube `asset_index.json`, per-task rescoring, no-network cache completion, formal `yt-dlp` dependency locking, and v1.3.3 regression coverage while preserving the v1.3 editing and render core.

## v1.3.2 — 2026-08-11

- Added an auditable `local_evaluation` default and explicit `publish` mode. Local evaluation applies zero copyright/license ranking weight, no ordinary-YouTube penalty, no authorization filter, no licensing-query generation, and no recurring rights warning; source choice remains quality-first.
- Unified YouTube, Pixabay, supplied-manifest, run-state, and run-report policy fields through `material_usage_policy.py`; legacy attribution reminders are removed from local-evaluation manifests.
- Added project-wide `AGENTS.md` guidance and regression coverage proving the default mode, zero rights weight, report normalization, explicit publish switch, and license-word-neutral YouTube metadata scoring.

## v1.3.1 — 2026-08-10

- Added a minimal YouTube-first `yt-dlp` acquisition adapter, provider-compatible manifests, repeated query/exclusion controls, and `--asset-manifest` handoff for human-reviewed pools. The v1.3 reference, BGM, timeline, editing, rendering, and QA core remains unchanged.
- Corrected climax QA routing for short tracks whose later detected drop/climax events fall inside a broadly labeled `outro`; actual event windows are used when section-role shots cover less than half of those windows. Existing pass thresholds were not lowered.

## v1.3 — 2026-08-10

- 保留 v1.2 已验证的参考学习、BGM BPM/重拍/乐句/段落/能量/停顿分析、下载前音乐槽位、恒速安全区间、FFmpeg 全局帧网格渲染、自动返工与完整媒体 QA；没有重写节奏核心。
- 新增通用 `visual_style_profile.json`：从用户要求、参考画像和 BGM 动态提取主题、情绪、环境/地貌、用户明确地点、时间天气、光线、色彩、摄影方式和镜头运动。移除 2026-08-10 临时加入的北境/冰岛/暗暖测试白名单与专用评分/调色分支。
- Pixabay 查询升级为带意图的多轴多轮组合：precision、adjacent-world 和 quality-recall；地点只来自任务输入，不再让 nature/landscape 等泛词承担主要高质量召回。
- 扩充通用中文主题归一并增加跨领域真实回归；料理/食材、暖亮、微距和横移等输入可形成 `food_culinary` 查询，不依赖自然风光词。扩展轮会跳过已执行查询，避免用缓存重复请求冒充扩搜。
- 素材完整分析新增纵深、构图、视觉冲击、光线、天气氛围、固有色彩质量、运动价值、电影感和普通旅游记录风险；主题相关只进入候选池，审美/电影感不足会继续尝试后续素材或明确失败。
- 全局素材元数据索引升级为 schema 4 / asset manifest schema 2，缓存语义、场景、世界/地貌族、景别、运动、色彩、时间天气、审美质量、SHA-256/感知哈希与 Pixabay 来源；分析缓存以 schema 2、引擎 `1.3.0` 和文件哈希失效。
- 时间线选片新增动态任务匹配及相邻视觉连续性：运动方向/类型、景别、颜色亮度、世界族、纹理和构图；在不改变 BGM 切点的前提下做有限镜头替换，默认仍为干净硬切。
- 新增序列级一致性报告和 QA：全片世界观、颜色、时间天气、镜头语言、相邻匹配均值和严重断裂位置。多样性只在与本次世界观兼容的候选中生效。
- 固定风格滤镜替换为任务色彩画像驱动的轻度、有界逐镜归一化；不兼容素材必须在上游淘汰，不用重滤镜补救。
- `edit_decisions.json` 升级为 schema `1.3` / timeline schema 3，继续保留 v1.2 字段，同时增加标准 source/timeline 字段、project/timebase、基础 transform 和 BGM 独立轨；`edit_schema.py` 可迁移旧计划。
- 新增可选 `jianying_export.py`：直接消费统一 edit decisions，引用原始素材，每镜独立、BGM 独立轨，映射源/目标区间、恒速和基础变换；写前备份剪映根索引、拒绝覆盖、结构化验证，并显式记录裁剪/调色等未映射差异。
- 剪映依赖独立锁定在 `requirements-jianying.lock.txt`；本机验证组合为 CPython 3.11.9、pyJianYingDraft 0.3.0 commit `c3318066...` 与剪映专业版 11.1.0.14287。
- 新增 v1.3 动态检索、审美聚合、缓存失效、世界观/视觉衔接、schema 迁移和剪映结构验证测试；打包白名单同步纳入新脚本与可选依赖锁。

### 升级与回退

- v1.2 运行产物可由 `edit_schema.py` 非破坏迁移；旧字段仍可读取。素材库不复制，旧视觉分析在第一次被 v1.3 选中时升级。
- 本项目升级前基线备份为 `test-output/backups/bgm-montage-v1.2-baseline-20260810T-current.zip`。回退时先保留 v1.3 输出/缓存，再用该 ZIP 恢复 `.agents/skills/bgm-montage` 源码；v1.3 新缓存字段可留存，v1.2 读取器会忽略未知字段。

## v1.2 — 2026-08-08

- 将 BGM 分析统一为结构化 `audiomap.json`：增加确定性的 beat/downbeat、onset、能量、密度、静音/低能量区间、hard stop、drop、surge、climax、乐句、段落角色、重复结构和分析摘要。
- 增加保守的节奏模式判定：节拍置信度、间隔稳定性、覆盖和脉冲证据达标时使用 `beat_cut`；舒缓、氛围或不稳定节拍使用 `phrase_flow`。
- 增加下载前 `timeline.json`，让镜头边界优先吸附真实音乐事件，并逐槽位记录段落、情绪、能量、画面内容、景别、运动、重点事件和转场建议。
- 将参考 `style_profile.json` 与 `editing_grammar.json` 接入实际时间线和选片：参与镜头时长、切点权重、段落密度、景别/运动变化、相邻镜头差异、简单转场和结尾结构。
- 统一 v1.2 主产物为 `audiomap.json`、`timeline.json`、`asset_manifest.json`、`edit_decisions.json` 和 `render_report.json`；同时保留 v1.1 的 `bgm_profile.json`、`sources.json`、`edit_plan.json` 和 `validation.json` 兼容别名。
- 强化 Pixabay 候选池门槛：默认要求每个计划镜头至少 6 个元数据候选，并检查重点槽位覆盖；扩展查询和缓存检索后仍不足时明确失败。
- 素材清单增加统一 canonical source 身份、文件哈希、可用连续区间、质量/语义标签、下载/复用状态、历史使用和实际输出/源区间；已下载素材可跨主题、跨项目复用而不重复下载。
- 增加 Pixabay ID 级跨进程事务锁、PID/token/心跳与死进程回收；锁内二次读取统一索引，保证并行项目对同一素材只下载一次。共享主题清单使用事务锁和每次运行的不可变快照，坏 JSON 或权限错误会 fail-closed。
- 时间线默认同一 canonical source 只使用一次；有限复用仍受最大次数、累计画面占比、镜头/时间间隔和源区间重叠限制。
- 增加素材内部最佳区间选择，避开片头片尾、黑帧、低变化、强抖动和动作结束后的停顿；不再用最后一帧冻结补齐镜头。
- 增加逐槽位综合评分与相邻多样性检查，综合主题/槽位语义、段落情绪、运动、景别、画质、比例、可用时长、历史复用、人脸风险以及前后镜头差异。
- 参考视频画像增加镜头时长分布、分阶段节奏、运动方向、转场提示、关键镜头、高潮剪辑密度和重复/相邻关系；v1.1 指纹缓存可非破坏性迁移并派生 v1.2 字段。
- 渲染仅使用硬切、淡入淡出（含淡黑）和短叠化，并对软转场设置预算；保留主体感知裁剪、模糊背景填充、恒速片段变速和统一调色。
- 将逐镜头时长统一量化到全局输出帧网格，合并不足 0.5 秒的尾部碎片槽位，并为源解码保留非冻结帧余量，避免 concat 逐段取整与全局截断把最后镜头压成单帧；QA 增加严格流时长、计划尾镜和编码后尾部突变检查。
- 扩展 `render_report.json`：完整解码、音视频流时长、分辨率、帧率、音量、黑帧、冻结、静音、来源路径、源区间、重复率、人脸预算、裁剪安全、相邻多样性、音乐切点对齐和高潮视觉响应。
- QA 失败时按新尝试种子重新分配素材、选择源区间并重渲染；只有最终报告通过后才写入素材使用历史。默认最多自动返工 2 次，全部失败时不交付成片。
- 增加 `run_state.json` 断点续跑校验；继续保持 UTC/随机 `run_id` 和默认不覆盖历史输出。
- CLI 新增候选池、搜索页数、来源复用/占比、重复间隔、返工次数和显式语义降级配置；v1.1 主要参数继续兼容。

### 明确边界

v1.2 不识别或复刻复杂遮罩/wipe、match cut、速度坡度、字幕内容或样式、OCR、动态字幕/图形、人物身份、可靠时序动作或逐帧主体跟踪。CLIP 仅是采样帧零样本语义增强；不可用时必须显式允许 structural fallback，报告会保留降级状态。参考视频的淡化/叠化标签是保守提示，不代表可复刻任意复杂特效。Linux/Docker 仅验证 ZIP 路径布局兼容，不宣称完成现场运行验证。

## v1.1 — 2026-08-02

- 将标准 Skill 布局迁移到 `.agents/skills/bgm-montage`，修正项目根目录安装和调用示例。
- 统一 Pixabay 缓存为 `.bgm-montage-cache/pixabay/{search,thumbnails}`，并非破坏性迁移旧的重复嵌套缓存。
- 增加机器级素材目录，以及跨主题/跨项目的硬链接或共享引用复用。
- 增加有界三轮查询扩展，以及独立素材、场景、理论时长、复用次数、单素材画面占比、低人脸库存和正脸画面占比门槛。
- 在 OpenCV 颜色、曝光、切点、运动、构图和文字区域统计之外，增加预训练 CLIP 零样本语义分析。
- 增加显著区域与正脸几何，用于主体感知裁剪；不安全时使用模糊背景填充。
- 增加参考视频音频分析和 `editing_grammar.json`，学习切点与重音、强弱拍、乐句、段落、停顿和能量的关系。
- 将参考语法接入边界权重、能量相关镜头时长、景别/运动变化、循环段变化和结尾时长。
- 增加 `run_id` 输出目录，默认拒绝覆盖既有运行。
- 扩充来源清单和验证报告，增加复用方式、实际使用区间、重复、人脸预算和裁剪检查。
- 增加 CPython 3.11 依赖锁定、自动化测试和使用标准 `/` 路径的安全 ZIP 打包器。
