# 贡献指南

感谢你愿意改进 BGM Montage。请先阅读 [README.md](README.md)、[使用说明](references/usage.md) 和 [LICENSE](LICENSE)，确认你的贡献符合当前的非商业源码可用许可范围。

## 提交前

1. 从 `main` 创建清晰、单一目的的分支。
2. 不提交 `.env`、API Key、Cookie、用户媒体、缓存、生成视频、`test-output` 或虚拟环境。
3. 对行为改动补充最小且相关的回归测试；对文档改动确认命令、路径和版本真实可用。
4. 在 Python 3.11 环境中运行：

   ```powershell
   .\.venv\Scripts\python -m pytest
   ```

5. 在 Pull Request 中说明问题、改动范围、验证命令和仍未覆盖的边界。

## Issue 建议

请提供版本、操作系统、Python/FFmpeg 版本、最小复现步骤、预期结果和实际结果。日志请先移除本地绝对路径中的私人信息、Cookie、令牌、API Key 与媒体链接。

## 设计边界

请优先复用已有 BGM 分析、时间线、素材清单、渲染和 QA 工件。核心输出契约围绕 `audiomap.json`、`timeline.json`、`asset_manifest.json`、`edit_decisions.json` 和 `render_report.json`；任何改变这些工件语义的修改，都应同时更新迁移/兼容策略、测试和文档。
