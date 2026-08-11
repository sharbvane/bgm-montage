---
name: bgm-montage
description: Learn a reusable visual and editing profile from reference videos, analyze a BGM's real musical structure, search and quality-rank Pixabay videos, and render a beat-aware montage with cached searches, deduplication, and source attribution. Use when Codex must create or refresh an automatic music-driven stock-footage montage from a BGM path, theme, duration, aspect ratio, and output directory.
---

# BGM Montage

Create a traceable montage while keeping the reference directory read-only. Run the deterministic scripts instead of inventing API calls or edit commands.

## Workflow

1. Verify `ffmpeg` and `ffprobe`, then use the skill-local `.venv`.
2. Keep the real Pixabay key only in the project-root `.env` as `PIXABAY_API_KEY`.
3. Run `scripts/bgm_montage.py` with the BGM, theme, duration, ratio, and output directory.
4. Reuse `.bgm-montage-cache`; reference fingerprints ensure that only new or changed videos are reanalyzed.
5. Inspect `style_profile.json`, `bgm_profile.json`, `edit_plan.json`, `sources.json`, and `validation.json` beside the output.
6. Treat visual topic labels, vocal likelihood, shot scale, watermark/text, and camera motion as signal-derived estimates. Do not describe them as model-certified facts.
7. Preserve Pixabay attribution and the exact used intervals in `sources.json`.

## Commands

Use the unified entry point from the skill directory (or use the project-root
form documented in `references/usage.md`):

```powershell
& .\.venv\Scripts\python.exe .\scripts\bgm_montage.py `
  --bgm ".\music\track.mp3" --theme "ocean freedom" `
  --duration 30 --ratio 9:16 --output-dir ".\output"
```

Use individual stages only for diagnostics:

```powershell
& .\.venv\Scripts\python.exe .\scripts\analyze_references.py --help
& .\.venv\Scripts\python.exe .\scripts\analyze_bgm.py --help
& .\.venv\Scripts\python.exe .\scripts\pixabay_pipeline.py --help
& .\.venv\Scripts\python.exe .\scripts\validate_output.py --help
```

Read [references/usage.md](references/usage.md) for installation, flags, output files, and cache behavior.

## Guardrails

- Never write into, move, rename, or transcode files inside the reference directory.
- Never print the API key or persist authenticated request URLs.
- Cache Pixabay search responses for at least 24 hours and avoid systematic mass downloading.
- Download only the final ranked candidates; if post-download QA rejects one, record the reason before selecting a replacement.
- Keep `.env`, `.venv`, caches, downloaded footage, and rendered tests out of distributable ZIP files.
