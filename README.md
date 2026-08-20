# bgm-montage

`bgm-montage` is a Python skill for building traceable, music-driven video montages from a BGM track, read-only reference videos, and local or provider-sourced footage. The current published release is `v1.4.4`.

## v1.4.4 highlights

- Local-library metadata truth and provenance, with missing-feature scoring normalization and bounded lazy cache migration.
- Shared asset-capacity preflight for reuse, source intervals, repeat gaps, screen share, face budget, and usable duration.
- A shared planner/validator music-event contract.
- Visual QA coverage, world/time-weather/camera consistency, and adjacent-shot diversity checks.
- `colorbalance` `pl=0` correction with the evidence boundary retained.
- Duration stage instrumentation and a Golden Fixture Identity Contract.
- Truthful climax QA when calm/reference evidence is missing: the result is `insufficient_evidence`, not a fake zero-intensity comparison or fake pass.
- A clean-extract runtime-package smoke test kept as a permanent package regression.

## Core capabilities

- Analyses BPM, beats/downbeats, onsets, energy, phrases, sections, pauses, drops, surges, and climaxes.
- Learns sampled visual profiles and audio-linked editing grammar from reference videos.
- Builds a music-event-aware timeline and aligns cuts to beats or phrase flow.
- Supports YouTube-first, Pixabay, and fully offline Local Library workflows.
- Builds a persistent six-frame visual index for large local libraries, then deeply analyses only a bounded Top-K candidate set; unchanged files are reused and usage history follows content identity.
- Uses FFmpeg rendering and automatic media QA, plus required Agent Visual Review evidence where enabled.
- Optionally exports edit decisions to JianYing Pro with independent source clips and a separate BGM track.

The `v1.4.4` boundary is deliberate: Duration remains `not reproduced / instrumented / known historical risk`; Color is `pl=0 fixed / evidence boundary retained`; further Reference Grammar `beat_cut` rhythm optimization is deferred and is not part of this release.

## Install

Check out the release tag. Python 3.11 is recommended; `ffmpeg` and `ffprobe` must be on `PATH`.

```powershell
git clone https://github.com/sharbvane/bgm-montage.git
Set-Location bgm-montage
git checkout v1.4.4
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

For Pixabay, set `PIXABAY_API_KEY` in the local `.env` or process environment. Do not commit the real `.env` or embed credentials. YouTube support uses the pinned `yt-dlp` dependency and does not require embedded credentials.

## Basic use

Read the matching [SKILL.md](SKILL.md) and [usage reference](references/usage.md), then inspect the available commands:

```powershell
.\.venv\Scripts\python .\scripts\bgm_montage.py --version
.\.venv\Scripts\python .\scripts\bgm_montage.py --help
.\.venv\Scripts\python .\scripts\pixabay_pipeline.py --help
```

Provide a BGM, task/theme, output directory, and, when available, a read-only reference or local-library directory. The workflow analyses inputs, plans the timeline, selects or reuses footage, renders, and validates the output.

## History and release artifacts

Published source snapshots include `v1`, `v1.1`, `v1.2`, `v1.3`, `v1.3.2`, `v1.3.3`, `v1.4`, `v1.4.1`, `v1.4.3`, and `v1.4.4`. The v1.4.4 release asset is attached to the GitHub Release; media, caches, logs, credentials, virtual environments, and test-output are excluded from the repository.

## License

Source-available for local learning, research, testing, and technical exchange. Commercial use, commercial distribution, or integration into a commercial product or service requires written permission. See [LICENSE](LICENSE); contact thiscui@foxmail.com.
