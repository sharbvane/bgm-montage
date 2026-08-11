#!/usr/bin/env python3
"""Validate rendered media with ffprobe and full FFmpeg decode."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from montage import adjacent_diversity_issues, canonical_source_key, parse_ratio


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


def _load_payload(value: str | Path | dict[str, Any] | None) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    if value is None:
        return None
    try:
        payload = json.loads(Path(value).expanduser().resolve().read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else None
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def _fraction(value: Any) -> float:
    try:
        text = str(value or "0/1")
        if "/" in text:
            numerator, denominator = text.split("/", 1)
            return float(numerator) / max(float(denominator), 1e-9)
        return float(text)
    except (TypeError, ValueError, ZeroDivisionError):
        return 0.0


def _event_time(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, dict):
        for key in ("time", "time_seconds", "boundary", "start", "start_seconds"):
            if key in value:
                return _event_time(value[key])
    return None


def _collect_event_times(audiomap: dict[str, Any] | None, duration: float) -> dict[str, list[float]]:
    result: dict[str, list[float]] = {
        "beats": [], "downbeats": [], "onsets": [], "accents": [],
        "hard_stops": [], "drops": [], "surges": [], "climaxes": [],
        "phrases": [], "sections": [],
    }
    if not audiomap:
        return result
    events = audiomap.get("events", {}) if isinstance(audiomap.get("events"), dict) else {}
    aliases = {
        "beats": (events.get("beats"), audiomap.get("beats")),
        "downbeats": (events.get("downbeats"), audiomap.get("downbeats")),
        "onsets": (events.get("onsets"), audiomap.get("onsets")),
        "accents": (events.get("accents"), audiomap.get("accents")),
        "hard_stops": (events.get("hard_stops"), audiomap.get("hard_stops")),
        "drops": (events.get("drops"), audiomap.get("drops")),
        "surges": (events.get("surges"), audiomap.get("surges")),
        "climaxes": (events.get("climaxes"), audiomap.get("climaxes")),
        "phrases": (events.get("phrase_boundaries"), audiomap.get("phrase_boundaries"), audiomap.get("phrases")),
        "sections": (events.get("section_boundaries"), audiomap.get("section_boundaries")),
    }
    sections = audiomap.get("sections")
    if isinstance(sections, list):
        aliases["sections"] = (*aliases["sections"], sections)
    for group, candidates in aliases.items():
        times: list[float] = []
        for candidate in candidates:
            values = candidate if isinstance(candidate, list) else []
            for value in values:
                if group in {"phrases", "sections"} and isinstance(value, dict):
                    for key in ("start", "end", "time", "boundary"):
                        if value.get(key) is not None:
                            parsed = _event_time(value[key])
                            if parsed is not None:
                                times.append(parsed)
                else:
                    parsed = _event_time(value)
                    if parsed is not None:
                        times.append(parsed)
        result[group] = sorted({round(value, 5) for value in times if 0.0 <= value <= duration})
    return result


def _volume_metrics(stderr: str) -> dict[str, float | None]:
    result: dict[str, float | None] = {"mean_volume_db": None, "max_volume_db": None}
    for key, label in (("mean_volume_db", "mean_volume"), ("max_volume_db", "max_volume")):
        match = re.search(rf"{label}:\s*(-?(?:inf|[0-9.]+))\s*dB", stderr, flags=re.IGNORECASE)
        if match:
            token = match.group(1).lower()
            result[key] = float("-inf") if token == "-inf" else float(token)
    return result


def _scene_change_times(stderr: str) -> list[float]:
    """Parse FFmpeg showinfo timestamps emitted by a scene-select filter."""

    values: list[float] = []
    for match in re.finditer(r"\bpts_time:\s*([0-9]+(?:\.[0-9]+)?)", stderr):
        try:
            values.append(float(match.group(1)))
        except ValueError:
            continue
    return sorted({round(value, 6) for value in values})


def _source_interval_overlap_ratio(shots: list[dict[str, Any]]) -> float:
    by_source: dict[str, list[tuple[float, float]]] = {}
    maximum = 0.0
    for shot in shots:
        identity = str(shot.get("canonical_source_key") or canonical_source_key(shot))
        start = float(shot.get("source_start") or 0.0)
        end = float(shot.get("source_end") or start)
        for left, right in by_source.setdefault(identity, []):
            overlap = max(0.0, min(end, right) - max(start, left))
            maximum = max(maximum, overlap / max(1e-6, min(end - start, right - left)))
        by_source[identity].append((start, end))
    return maximum


def _cut_alignment(
    shots: list[dict[str, Any]],
    audiomap: dict[str, Any] | None,
    duration: float,
) -> dict[str, Any] | None:
    if not audiomap or len(shots) < 2:
        return None
    groups = _collect_event_times(audiomap, duration)
    rhythm = audiomap.get("rhythm_mode", {})
    mode = str(rhythm.get("mode") if isinstance(rhythm, dict) else rhythm or "phrase_flow")
    if mode == "beat_cut":
        allowed = sorted(set(groups["beats"] + groups["downbeats"] + groups["accents"] + groups["drops"] + groups["hard_stops"]))
        beat_period = 0.0
        tempo = audiomap.get("tempo", {}) if isinstance(audiomap.get("tempo"), dict) else {}
        beat_period = float(tempo.get("beat_period_seconds") or 0.0)
        tolerance = max(0.10, min(0.28, beat_period * 0.32 if beat_period else 0.20))
    else:
        allowed = sorted(set(groups["phrases"] + groups["sections"] + groups["drops"] + groups["surges"] + groups["hard_stops"]))
        tolerance = 0.55
    if not allowed:
        return {"mode": mode, "available": False, "passed": False, "reason": "no alignment events"}
    offsets = []
    for shot in shots[:-1]:
        boundary = float(shot.get("output_end") or 0.0)
        nearest = min(allowed, key=lambda value: abs(value - boundary))
        offsets.append({"boundary": round(boundary, 5), "nearest_event": nearest, "error_seconds": round(abs(boundary - nearest), 5)})
    errors = [item["error_seconds"] for item in offsets]
    aligned_share = sum(error <= tolerance for error in errors) / max(1, len(errors))
    # A small number of energy-grid cuts between strong anchors is acceptable;
    # requiring every cut to hit a beat would recreate mechanical beat cutting.
    required_share = 0.70 if mode == "beat_cut" else 0.60
    return {
        "mode": mode,
        "available": True,
        "tolerance_seconds": tolerance,
        "required_aligned_share": required_share,
        "aligned_share": round(aligned_share, 4),
        "mean_error_seconds": round(sum(errors) / max(1, len(errors)), 5),
        "max_error_seconds": round(max(errors, default=0.0), 5),
        "boundaries": offsets,
        "passed": aligned_share >= required_share,
    }


def _climax_metrics(
    shots: list[dict[str, Any]],
    audiomap: dict[str, Any] | None,
    duration: float,
) -> dict[str, Any] | None:
    if not audiomap or not shots:
        return None
    events = _collect_event_times(audiomap, duration)
    windows: list[tuple[float, float]] = []
    for item in audiomap.get("climaxes", []) if isinstance(audiomap.get("climaxes"), list) else []:
        if isinstance(item, dict) and item.get("start") is not None and item.get("end") is not None:
            windows.append((float(item["start"]), float(item["end"])))
    for time_value in events["climaxes"] + events["drops"]:
        windows.append((max(0.0, time_value - 1.2), min(duration, time_value + 1.8)))
    if not windows:
        return None

    def overlaps(shot: dict[str, Any], ranges: list[tuple[float, float]]) -> bool:
        start, end = float(shot.get("output_start") or 0.0), float(shot.get("output_end") or 0.0)
        return any(max(start, left) < min(end, right) for left, right in ranges)

    role_climax = [
        shot
        for shot in shots
        if str(shot.get("audio_section_role") or shot.get("section_role") or "").lower()
        in {"drop", "climax"}
    ]
    role_calm = [
        shot
        for shot in shots
        if str(shot.get("audio_section_role") or shot.get("section_role") or "").lower()
        in {"intro", "break", "outro"}
    ]
    if role_climax and role_calm:
        climax_shots = role_climax
        calm_shots = role_calm
        comparison_method = "audiomap_section_roles"
    else:
        climax_shots = [shot for shot in shots if overlaps(shot, windows)]
        calm_shots = [shot for shot in shots if not overlaps(shot, windows)]
        comparison_method = "drop_and_climax_event_windows"

    def density(items: list[dict[str, Any]]) -> float:
        total = sum(float(item.get("output_duration") or 0.0) for item in items)
        return len(items) / max(total, 1e-6)

    def intensity(items: list[dict[str, Any]]) -> float:
        values = []
        for item in items:
            motion = float(item.get("source_motion") or 0.0)
            emphasis = 0.14 if item.get("is_emphasis") else 0.0
            scale = 0.10 if str(item.get("source_shot_scale")) == "wide" else 0.04
            values.append(min(1.0, motion + emphasis + scale))
        return sum(values) / max(1, len(values))

    climax_density, calm_density = density(climax_shots), density(calm_shots)
    climax_intensity, calm_intensity = intensity(climax_shots), intensity(calm_shots)
    return {
        "windows": [[round(left, 4), round(right, 4)] for left, right in windows],
        "comparison_method": comparison_method,
        "climax_shot_count": len(climax_shots),
        "climax_cut_density": round(climax_density, 4),
        "calm_cut_density": round(calm_density, 4),
        "climax_visual_intensity": round(climax_intensity, 4),
        "calm_visual_intensity": round(calm_intensity, 4),
        "passed": bool(climax_shots)
        and (
            climax_density > calm_density
            if comparison_method == "audiomap_section_roles"
            else climax_density + 0.05 >= calm_density * 0.90
        )
        and climax_intensity + 0.08 >= calm_intensity,
    }


def validate_output(
    media_path: str | Path,
    expected_duration: float | None = None,
    expected_ratio: str | None = None,
    report_path: str | Path | None = None,
    frames_dir: str | Path | None = None,
    edit_plan: str | Path | dict[str, Any] | None = None,
    ffmpeg: str = "ffmpeg",
    ffprobe: str = "ffprobe",
    audiomap: str | Path | dict[str, Any] | None = None,
    expected_fps: float | None = None,
) -> dict[str, Any]:
    path = Path(media_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if (not shutil.which(ffmpeg) and not Path(ffmpeg).is_file()) or (
        not shutil.which(ffprobe) and not Path(ffprobe).is_file()
    ):
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
    frame_rate = _fraction(video.get("avg_frame_rate") or video.get("r_frame_rate"))

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
    volume_process = _run(
        [
            ffmpeg,
            "-hide_banner",
            "-nostats",
            "-i",
            str(path),
            "-map",
            "0:a:0?",
            "-af",
            "volumedetect",
            "-f",
            "null",
            null_sink,
        ]
    )
    scene_process = _run(
        [
            ffmpeg,
            "-hide_banner",
            "-nostats",
            "-i",
            str(path),
            "-vf",
            "select='gt(scene,0.10)',showinfo",
            "-an",
            "-f",
            "null",
            null_sink,
        ]
    )
    volume = _volume_metrics(volume_process.stderr)
    scene_change_times = _scene_change_times(scene_process.stderr)
    black = _parse_detector(detectors.stderr, "black_", ("black_start", "black_end", "black_duration"))
    freeze = _parse_detector(detectors.stderr, "freeze_", ("freeze_start", "freeze_end", "freeze_duration"))
    silence = _parse_detector(detectors.stderr, "silence_", ("silence_start", "silence_end", "silence_duration"))

    black_durations = _detector_durations(black, "black_start", "black_end", "black_duration", duration)
    freeze_durations = _detector_durations(freeze, "freeze_start", "freeze_end", "freeze_duration", duration)
    silence_durations = _detector_durations(silence, "silence_start", "silence_end", "silence_duration", duration)
    # A montage that is short by several frames can hide a truncated terminal
    # shot even though the container is otherwise decodable.  Allow normal
    # codec timestamp rounding, but not a visibly missing tail.
    tolerance_frame_rate = frame_rate if frame_rate > 0 else float(expected_fps or 30.0)
    duration_tolerance = max(0.06, 2.0 / max(tolerance_frame_rate, 1.0))
    target_duration = expected_duration if expected_duration is not None else duration
    stream_duration_ok = target_duration > 0 and video_duration > 0 and abs(video_duration - target_duration) <= duration_tolerance
    audio_duration_ok = target_duration > 0 and audio_duration > 0 and abs(audio_duration - target_duration) <= duration_tolerance
    long_black_limit = max(0.55, duration * 0.12)
    total_black_limit = max(0.80, duration * 0.16)
    long_freeze_limit = max(1.60, duration * 0.18)
    total_freeze_limit = max(2.00, duration * 0.22)
    long_silence_limit = max(1.25, duration * 0.45)
    total_silence_limit = max(1.50, duration * 0.55)
    mean_volume = volume["mean_volume_db"]
    max_volume = volume["max_volume_db"]
    fps_target = float(expected_fps) if expected_fps is not None else None
    terminal_scene_seconds = (
        max(0.0, duration - scene_change_times[-1]) if scene_change_times else duration
    )
    terminal_scene_minimum = max(0.25, 8.0 / max(frame_rate, 24.0))

    checks = {
        "file_exists": path.is_file(),
        "nonempty": path.stat().st_size > 1024,
        "has_video": bool(video_streams),
        "has_audio": bool(audio_streams),
        "full_decode": decode.returncode == 0,
        "detectors_ran": detectors.returncode == 0,
        "scene_detector_ran": scene_process.returncode == 0,
        "duration": expected_duration is None
        or abs(duration - expected_duration) <= duration_tolerance,
        "video_stream_duration": stream_duration_ok,
        "audio_stream_duration": audio_duration_ok,
        "no_long_black": max(black_durations, default=0.0) < long_black_limit
        and sum(black_durations) < total_black_limit,
        "no_long_freeze": max(freeze_durations, default=0.0) < long_freeze_limit
        and sum(freeze_durations) < total_freeze_limit,
        "audio_not_entirely_silent": max(silence_durations, default=0.0) < long_silence_limit
        and sum(silence_durations) < total_silence_limit,
        "frame_rate": frame_rate > 0
        and (abs(frame_rate - fps_target) <= 0.55 if fps_target is not None else 12.0 <= frame_rate <= 120.0),
        "audio_level": bool(audio_streams)
        and mean_volume is not None
        and math.isfinite(float(mean_volume))
        and -42.0 <= float(mean_volume) <= -4.0,
        "audio_peak_safe": bool(audio_streams)
        and max_volume is not None
        and math.isfinite(float(max_volume))
        and -30.0 <= float(max_volume) <= 0.5,
        "no_terminal_microshot": terminal_scene_seconds + 1e-6 >= terminal_scene_minimum,
        "resolution": True,
    }
    expected_dimensions: dict[str, int] | None = None
    if expected_ratio:
        spec = parse_ratio(expected_ratio)
        expected_dimensions = {"width": spec.width, "height": spec.height}
        checks["resolution"] = width == spec.width and height == spec.height

    plan_payload = _load_payload(edit_plan)
    audiomap_payload = _load_payload(audiomap)
    if edit_plan is not None:
        checks["edit_plan_readable"] = plan_payload is not None
    if audiomap is not None:
        checks["audiomap_readable"] = audiomap_payload is not None
    plan_metrics: dict[str, Any] | None = None
    alignment_metrics: dict[str, Any] | None = None
    climax_metrics: dict[str, Any] | None = None
    if plan_payload is not None:
        shots = [shot for shot in plan_payload.get("shots", []) if isinstance(shot, dict)]
        policy = plan_payload.get("content_policy", {}) if isinstance(plan_payload.get("content_policy"), dict) else {}
        counts: dict[str, int] = {}
        screen_time: dict[str, float] = {}
        occurrences: dict[str, list[tuple[int, float, float]]] = {}
        prominent_face_seconds = 0.0
        unsafe_crop = 0
        missing_sources: list[str] = []
        for shot in shots:
            identity = str(shot.get("canonical_source_key") or canonical_source_key(shot))
            shot_duration = float(shot.get("output_duration") or 0.0)
            counts[identity] = counts.get(identity, 0) + 1
            screen_time[identity] = screen_time.get(identity, 0.0) + shot_duration
            occurrences.setdefault(identity, []).append(
                (
                    int(shot.get("index", len(occurrences.get(identity, [])))),
                    float(shot.get("output_start") or 0.0),
                    float(shot.get("output_end") or 0.0),
                )
            )
            local = Path(str(shot.get("local_path") or "")).expanduser()
            if not local.is_file():
                missing_sources.append(str(local))
            if float(shot.get("face_content_risk") or 0.0) >= float(policy.get("prominent_face_threshold", 0.65)):
                prominent_face_seconds += shot_duration
            crop = shot.get("crop_plan", {}) if isinstance(shot.get("crop_plan"), dict) else {}
            mode = str(crop.get("mode", ""))
            retention = float(crop.get("retention", 0.0) or 0.0)
            if mode not in {"subject_crop", "blur_fill", "fit"} or (mode == "subject_crop" and retention < 0.85):
                unsafe_crop += 1
        unique = len(counts)
        max_reuse = max(counts.values(), default=0)
        max_share = max(screen_time.values(), default=0.0) / max(duration, 1e-6)
        face_share = prominent_face_seconds / max(duration, 1e-6)
        max_overlap_ratio = _source_interval_overlap_ratio(shots)
        repeat_gap_failures: list[dict[str, Any]] = []
        min_gap_shots = int(policy.get("min_repeat_gap_shots", 0))
        min_gap_seconds = float(policy.get("min_repeat_gap_seconds", 0.0))
        for identity, values in occurrences.items():
            values.sort()
            for previous, current in zip(values, values[1:]):
                if current[0] - previous[0] < min_gap_shots or current[1] - previous[2] < min_gap_seconds:
                    repeat_gap_failures.append(
                        {
                            "canonical_source_key": identity,
                            "previous_index": previous[0],
                            "current_index": current[0],
                            "gap_seconds": round(current[1] - previous[2], 5),
                        }
                    )
        diversity_issues = [
            {
                "left_index": index - 1,
                "right_index": index,
                "issues": adjacent_diversity_issues(shots[index - 1], shots[index]),
            }
            for index in range(1, len(shots))
            if adjacent_diversity_issues(shots[index - 1], shots[index])
        ]
        severe_limit = int(policy.get("max_adjacent_similarity_dimensions", 3))
        checks["material_repetition"] = (
            unique >= int(policy.get("min_unique_assets", 1))
            and max_reuse <= int(policy.get("max_reuse_per_asset", max_reuse or 1))
            and max_share <= float(policy.get("max_asset_screen_share", 1.0)) + 0.006
        )
        checks["source_paths_exist"] = not missing_sources
        checks["source_intervals_nonoverlap"] = max_overlap_ratio <= float(
            policy.get("max_source_interval_overlap", 0.02)
        ) + 0.006
        checks["repeat_gap"] = not repeat_gap_failures
        checks["adjacent_diversity"] = not any(
            "same_source" in item["issues"] or len(item["issues"]) > severe_limit + 2
            for item in diversity_issues
        )
        checks["prominent_face_budget"] = face_share <= float(
            policy.get("max_prominent_face_screen_share", 1.0)
        ) + 0.006
        checks["subject_crop_safe"] = bool(shots) and unsafe_crop == 0
        planned_durations = [float(shot.get("output_duration") or 0.0) for shot in shots]
        checks["no_planned_microshots"] = bool(planned_durations) and min(planned_durations) >= 0.24
        planned_terminal_minimum = 0.40
        checks["no_terminal_planned_microshot"] = (
            bool(planned_durations) and planned_durations[-1] >= planned_terminal_minimum
        )
        alignment_metrics = _cut_alignment(shots, audiomap_payload, duration)
        if alignment_metrics is not None:
            checks["music_cut_alignment"] = bool(alignment_metrics.get("passed"))
        climax_metrics = _climax_metrics(shots, audiomap_payload, duration)
        if climax_metrics is not None:
            checks["climax_visual_response"] = bool(climax_metrics.get("passed"))
        plan_metrics = {
            "shot_count": len(shots),
            "unique_asset_count": unique,
            "unique_canonical_source_count": unique,
            "repeat_shot_ratio": round(1.0 - unique / max(1, len(shots)), 4),
            "max_reuse_count": max_reuse,
            "max_asset_screen_share": round(max_share, 4),
            "prominent_face_screen_share": round(face_share, 4),
            "unsafe_crop_count": unsafe_crop,
            "maximum_source_interval_overlap_ratio": round(max_overlap_ratio, 5),
            "repeat_gap_failures": repeat_gap_failures,
            "missing_source_paths": missing_sources,
            "adjacent_diversity_issues": diversity_issues,
            "minimum_planned_shot_seconds": round(min(planned_durations, default=0.0), 4),
            "terminal_planned_shot_seconds": round(planned_durations[-1], 4)
            if planned_durations
            else 0.0,
            "terminal_planned_minimum_seconds": planned_terminal_minimum,
        }

    representative_frames: list[str] = []
    event_frames: list[dict[str, Any]] = []
    media_sha256 = _sha256(path)
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
        if audiomap_payload is not None:
            events = _collect_event_times(audiomap_payload, duration)
            requested: dict[float, set[str]] = {
                0.0: {"opening"},
                max(0.0, duration - max(0.04, 1.0 / max(frame_rate, 24.0))): {"ending"},
            }
            safe_last_frame = max(0.0, duration - max(0.04, 1.0 / max(frame_rate, 24.0)))
            for group in ("sections", "drops", "surges", "climaxes", "hard_stops"):
                for timestamp in events[group]:
                    requested.setdefault(round(min(safe_last_frame, max(0.0, timestamp)), 4), set()).add(group)
            randomizer = random.Random(int(media_sha256[:16], 16))
            for _ in range(2):
                requested.setdefault(round(randomizer.uniform(0.05 * duration, 0.95 * duration), 4), set()).add("random")
            ordered_times = sorted(requested)
            if len(ordered_times) > 24:
                mandatory = [
                    value for value in ordered_times
                    if requested[value] - {"sections", "random"}
                ]
                remaining = [value for value in ordered_times if value not in mandatory]
                ordered_times = sorted((mandatory + remaining[: max(0, 24 - len(mandatory))])[:24])
            for index, timestamp in enumerate(ordered_times, start=1):
                event_name = "_".join(sorted(requested[timestamp]))
                frame = frames / f"event_{index:02d}_{event_name}_{timestamp:.2f}s.jpg"
                extraction = _run(
                    [
                        ffmpeg, "-hide_banner", "-loglevel", "error", "-ss", f"{timestamp:.4f}",
                        "-i", str(path), "-frames:v", "1", "-q:v", "2", "-y", str(frame),
                    ],
                    timeout=60,
                )
                if extraction.returncode == 0 and frame.is_file():
                    event_frames.append(
                        {"time_seconds": timestamp, "event_types": sorted(requested[timestamp]), "path": str(frame)}
                    )
            checks["event_frames"] = len(event_frames) == len(ordered_times)

    report = {
        "schema_version": "1.2",
        "artifact_type": "render_report",
        "path": str(path),
        "sha256": media_sha256,
        "size_bytes": path.stat().st_size,
        "duration_seconds": round(duration, 4),
        "expected_duration_seconds": round(target_duration, 4),
        "duration_tolerance_seconds": round(duration_tolerance, 6),
        "video": {
            "codec": video.get("codec_name"),
            "width": width,
            "height": height,
            "pixel_format": video.get("pix_fmt"),
            "frame_rate": video.get("avg_frame_rate"),
            "frame_rate_fps": round(frame_rate, 5),
            "duration_seconds": round(video_duration, 4),
        },
        "audio": {
            "codec": audio_streams[0].get("codec_name") if audio_streams else None,
            "sample_rate": audio_streams[0].get("sample_rate") if audio_streams else None,
            "channels": audio_streams[0].get("channels") if audio_streams else None,
            "duration_seconds": round(audio_duration, 4),
            "mean_volume_db": mean_volume,
            "max_volume_db": max_volume,
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
            "scene_changes": scene_change_times,
            "terminal_scene_seconds": round(terminal_scene_seconds, 6),
            "terminal_scene_minimum_seconds": round(terminal_scene_minimum, 6),
        },
        "representative_frames": representative_frames,
        "event_frames": event_frames,
        "edit_plan_metrics": plan_metrics,
        "music_cut_alignment": alignment_metrics,
        "climax_visual_response": climax_metrics,
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
    parser.add_argument("--edit-plan")
    parser.add_argument("--audiomap")
    parser.add_argument("--expected-fps", type=float)
    args = parser.parse_args()
    result = validate_output(
        args.media,
        args.expected_duration,
        args.ratio,
        args.report,
        args.frames_dir,
        args.edit_plan,
        audiomap=args.audiomap,
        expected_fps=args.expected_fps,
    )
    print(json.dumps({"passed": result["passed"], "path": result["path"], "checks": result["checks"]}, ensure_ascii=False))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
