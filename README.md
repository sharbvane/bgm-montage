# bgm-montage

`bgm-montage` is a Python skill for building traceable, music-driven video montages from a BGM track, read-only reference videos, and locally available or provider-sourced footage. The latest published code is v1.4.3.

## Features

- Analyses BPM, beats/downbeats, onsets, energy, phrases, sections, pauses, drops, surges, and climaxes.
- Learns sampled visual profiles and audio-linked editing grammar from reference videos.
- Builds a music-event-aware timeline and aligns cuts to beats or phrase flow.
- Searches, ranks, deduplicates, and reuses footage through YouTube-first, Pixabay, or fully offline Local Library mode.
- Builds a persistent six-frame visual index for large local libraries, then deeply analyses only a bounded Top-K candidate set; unchanged files are reused and usage history follows content identity.
- Uses FFmpeg rendering and automatic QA, including decode, stream, duration, frame rate, black/freeze/silence, repetition, and music-response checks where implemented.
- Produces visual evidence for a required Agent Visual Review; `fail` findings feed the existing bounded re-edit loop.
- v1.3+ optionally exports edit decisions to JianYing Pro with cuts, constant speed, basic transforms, independent clips, and a separate BGM track.

Each tag is an actual published source snapshot, including `v1`, `v1.1`, `v1.2`, `v1.3`, `v1.3.2`, `v1.3.3`, `v1.4`, `v1.4.1`, and `v1.4.3`.

## Install

Use the Python version and dependencies documented by the checked-out tag. The later releases recommend Python 3.11 and require `ffmpeg` and `ffprobe` on `PATH`.

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

For Pixabay, set `PIXABAY_API_KEY` in `.env` or the process environment. Do not commit the real `.env`. YouTube support in v1.3.1+ uses the version-pinned `yt-dlp` dependency and does not require embedded credentials.

## Basic use

Options differ by version. Read the matching `SKILL.md`, then inspect the included commands:

```powershell
.\.venv\Scripts\python .\scripts\bgm_montage.py --help
.\.venv\Scripts\python .\scripts\pixabay_pipeline.py --help
```

Provide a project root, BGM, task/theme, output directory, and (when available) a read-only reference directory. The workflow analyses inputs, plans the timeline, selects or reuses footage, renders, and validates the output.

## License

Source-available for local learning, research, testing, and technical exchange. Commercial use, commercial distribution, or integration into a commercial product or service requires written permission. See [LICENSE](LICENSE); contact thiscui@foxmail.com.
