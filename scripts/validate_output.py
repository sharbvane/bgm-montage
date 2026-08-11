#!/usr/bin/env python3
"""Validate rendered media with ffprobe and full FFmpeg decode."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from montage import parse_ratio


def _run(command: list[str], timeout: int = 600) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )


def probe_media(path: str | Path, ffprobe: str = "ffprobe") -> dict[str, Any]:
    process = _run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(Path(path).resolve()),
        ],
        timeout=60,
    )
    if process.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {process.stderr.strip()}")
    return json.loads(process.stdout)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_detector(stderr: str, pattern: str, fields: tuple[str, ...]) -> list[dict[str, float]]:
    records: list[dict[str, float]] = []
    for line in stderr.splitlines():
        if pattern not in line:
            continue
        record: dict[str, float] = {}
        for field in fields:
            match = re.search(rf"{re.escape(field)}:\s*([0-9.]+)", line)
            if match:
                record[field] = float(match.group(1))
        if record:
            records.append(record)
    return records


def _stream_duration(stream: dict[str, Any]) -> float:
    """Return a stream duration without falling back to container duration.

    Keeping stream and container durations separate catches files whose audio
    continues after the video stream has already ended.
    """

    try:
        value = float(stream.get("duration") or 0.0)
        if value > 0:
            return value
    except (TypeError, ValueError):
        pass
    try:
        duration_ts = float(stream.get("duration_ts") or 0.0)
        numerator, denominator = str(stream.get("time_base") or "0/1").split("/", 1)
        time_base = float(numerator) / float(denominator)
        value = duration_ts * time_base
        return value if value > 0 else 0.0
    except (TypeError, ValueError, ZeroDivisionError):
        return 0.0


def _detector_durations(
    records: list[dict[str, float]],
    start_field: str,
    end_field: str,
    duration_field: str,
    media_duration: float,
) -> list[float]:
    """Normalize detector events, including intervals left open at EOF."""

    durations: list[float] = []
    pending_start: float | None = None
    for record in records:
        if start_field in record:
            pending_start = max(0.0, float(record[start_field]))
        if duration_field in record:
            durations.append(max(0.0, float(record[duration_field])))
            pending_start = None
        elif end_field in record and pending_start is not None:
            durations.append(max(0.0, float(record[end_field]) - pending_start))
            pending_start = None
    if pending_start is not None and media_duration > pending_start:
        durations.append(media_duration - pending_start)
    return durations


def validate_output(
    media_path: str | Path,
    expected_duration: float | None = None,
    expected_ratio: str | None = None,
    report_path: str | Path | None = None,
    frames_dir: str | Path | None = None,
    ffmpeg: str = "ffmpeg",
    ffprobe: str = "ffprobe",
) -> dict[str, Any]:
    path = Path(media_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if not shutil.which(ffmpeg) or not shutil.which(ffprobe):
        raise RuntimeError("ffmpeg and ffprobe must be available on PATH")
    probe = probe_media(path, ffprobe)
    streams = probe.get("streams", [])
    video_streams = [stream for stream in streams if stream.get("codec_type") == "video"]
    audio_streams = [stream for stream in streams if stream.get("codec_type") == "audio"]
    video = video_streams[0] if video_streams else {}
    duration = float(probe.get("format", {}).get("duration") or video.get("duration") or 0.0)
    video_duration = _stream_duration(video)
    audio_duration = _stream_duration(audio_streams[0]) if audio_streams else 0.0
    width, height = int(video.get("width") or 0), int(video.get("height") or 0)

    null_sink = "NUL" if __import__("os").name == "nt" else "/dev/null"
    decode = _run([ffmpeg, "-v", "error", "-i", str(path), "-map", "0:v:0", "-map", "0:a?", "-f", "null", null_sink])
    detectors = _run(
        [
            ffmpeg,
            "-hide_banner",
            "-nostats",
            "-i",
            str(path),
            "-vf",
            "blackdetect=d=0.25:pix_th=0.10,freezedetect=n=-55dB:d=1.5",
            "-af",
            "silencedetect=n=-50dB:d=1.0",
            "-f",
            "null",
            null_sink,
        ]
    )
    black = _parse_detector(detectors.stderr, "black_", ("black_start", "black_end", "black_duration"))
    freeze = _parse_detector(detectors.stderr, "freeze_", ("freeze_start", "freeze_end", "freeze_duration"))
    silence = _parse_detector(detectors.stderr, "silence_", ("silence_start", "silence_end", "silence_duration"))

    black_durations = _detector_durations(black, "black_start", "black_end", "black_duration", duration)
    freeze_durations = _detector_durations(freeze, "freeze_start", "freeze_end", "freeze_duration", duration)
    silence_durations = _detector_durations(silence, "silence_start", "silence_end", "silence_duration", duration)
    duration_tolerance = max(0.35, (expected_duration or duration) * 0.02)
    target_duration = expected_duration if expected_duration is not None else duration
    stream_duration_ok = target_duration > 0 and video_duration > 0 and abs(video_duration - target_duration) <= duration_tolerance
    audio_duration_ok = target_duration > 0 and audio_duration > 0 and abs(audio_duration - target_duration) <= duration_tolerance
    long_black_limit = max(0.75, duration * 0.20)
    total_black_limit = max(1.0, duration * 0.25)
    long_freeze_limit = max(2.5, duration * 0.45)
    long_silence_limit = max(1.5, duration * 0.80)

    checks = {
        "file_exists": path.is_file(),
        "nonempty": path.stat().st_size > 1024,
        "has_video": bool(video_streams),
        "has_audio": bool(audio_streams),
        "full_decode": decode.returncode == 0,
        "detectors_ran": detectors.returncode == 0,
        "duration": expected_duration is None or abs(duration - expected_duration) <= max(0.35, expected_duration * 0.02),
        "video_stream_duration": stream_duration_ok,
        "audio_stream_duration": audio_duration_ok,
        "no_long_black": max(black_durations, default=0.0) < long_black_limit
        and sum(black_durations) < total_black_limit,
        "no_long_freeze": max(freeze_durations, default=0.0) < long_freeze_limit,
        "audio_not_entirely_silent": max(silence_durations, default=0.0) < long_silence_limit,
        "resolution": True,
    }
    expected_dimensions: dict[str, int] | None = None
    if expected_ratio:
        spec = parse_ratio(expected_ratio)
        expected_dimensions = {"width": spec.width, "height": spec.height}
        checks["resolution"] = width == spec.width and height == spec.height

    representative_frames: list[str] = []
    if frames_dir:
        frames = Path(frames_dir).expanduser().resolve()
        frames.mkdir(parents=True, exist_ok=True)
        for index, fraction in enumerate((0.1, 0.5, 0.9), start=1):
            timestamp = max(0.0, duration * fraction)
            frame = frames / f"frame_{index}_{timestamp:.2f}s.jpg"
            extraction = _run(
                [ffmpeg, "-hide_banner", "-loglevel", "error", "-ss", f"{timestamp:.4f}", "-i", str(path), "-frames:v", "1", "-q:v", "2", "-y", str(frame)],
                timeout=60,
            )
            if extraction.returncode == 0 and frame.is_file():
                representative_frames.append(str(frame))
        checks["representative_frames"] = len(representative_frames) == 3

    report = {
        "schema_version": 1,
        "path": str(path),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
        "duration_seconds": round(duration, 4),
        "video": {
            "codec": video.get("codec_name"),
            "width": width,
            "height": height,
            "pixel_format": video.get("pix_fmt"),
            "frame_rate": video.get("avg_frame_rate"),
            "duration_seconds": round(video_duration, 4),
        },
        "audio": {
            "codec": audio_streams[0].get("codec_name") if audio_streams else None,
            "sample_rate": audio_streams[0].get("sample_rate") if audio_streams else None,
            "channels": audio_streams[0].get("channels") if audio_streams else None,
            "duration_seconds": round(audio_duration, 4),
        },
        "expected_dimensions": expected_dimensions,
        "checks": checks,
        "passed": all(checks.values()),
        "decode_errors": decode.stderr.splitlines()[-20:],
        "detectors": {
            "black": black,
            "freeze": freeze,
            "silence": silence,
            "normalized_durations": {
                "black": [round(value, 6) for value in black_durations],
                "freeze": [round(value, 6) for value in freeze_durations],
                "silence": [round(value, 6) for value in silence_durations],
            },
        },
        "representative_frames": representative_frames,
    }
    if report_path:
        output = Path(report_path).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("media")
    parser.add_argument("--expected-duration", type=float)
    parser.add_argument("--ratio")
    parser.add_argument("--report")
    parser.add_argument("--frames-dir")
    args = parser.parse_args()
    result = validate_output(args.media, args.expected_duration, args.ratio, args.report, args.frames_dir)
    print(json.dumps({"passed": result["passed"], "path": result["path"], "checks": result["checks"]}, ensure_ascii=False))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
