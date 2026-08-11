# Usage

## Install once

From the `bgm-montage` directory:

```powershell
python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
& .\.venv\Scripts\python.exe -m pip check
```

Require `ffmpeg` and `ffprobe` on `PATH`. Put the real credential in the parent project's `.env`:

```dotenv
PIXABAY_API_KEY=replace_me
```

## Unified run

Run from the project root (the directory that contains `.env`, `参考视频`, and
`视频素材`). The checked-in project keeps the skill in `bgm-montage`:

```powershell
& .\bgm-montage\.venv\Scripts\python.exe .\bgm-montage\scripts\bgm_montage.py `
  --bgm ".\music\track.wav" `
  --theme "quiet coastal solitude" `
  --duration 30 `
  --ratio 9:16 `
  --output-dir ".\renders"
```

When the ZIP is installed outside the project, set the project root for the
current PowerShell process before invoking the same script:

```powershell
$env:BGM_MONTAGE_PROJECT_ROOT = ".\my-montage-project"
```

Only that discovered project root's `.env` is loaded; a `.env` beside the
skill code is intentionally ignored.

Required user inputs are `--bgm`, `--theme`, `--duration`, `--ratio`, and `--output-dir`. Defaults point to the parent project's `参考视频` and `视频素材` directories. Use `--reference-dir` or `--material-dir` to override them.

## Outputs

- `<theme>_montage.mp4`: H.264/AAC edit with the BGM as the formal audio track.
- `style_profile.json`: aggregate reference style plus per-run cache counters.
- `bgm_profile.json`: beats, accents, phrases, sections, energy, timbre, pauses, and vocal-likelihood estimates.
- `edit_plan.json`: every output interval, source interval, speed, energy, and cut rationale.
- `sources.json`: Pixabay ID, author, page URL, query, local file, QA scores, and actual used intervals.
- `validation.json`: ffprobe, full-decode, black/freeze/silence, duration, resolution, and stream checks.
- `run_report.json`: stage status and artifact paths without secrets.

## Cache and privacy

The parent project stores runtime cache in `.bgm-montage-cache`. Reference entries are keyed by a content-aware file fingerprint. Pixabay JSON responses expire after 24 hours. Authenticated request URLs are never written. The global material fingerprint library prevents repeat downloads across themes.

The reference directory is opened read-only. Runtime outputs, caches, downloaded footage, `.env`, and `.venv` are excluded from the release ZIP.
