#!/usr/bin/env python3
"""Build and render a signal-driven montage timeline with FFmpeg."""

from __future__ import annotations

import json
import math
import random
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


class MontageError(RuntimeError):
    """Raised when a montage plan cannot be built or rendered."""


@dataclass(frozen=True)
class OutputSpec:
    width: int
    height: int
    ratio: str


def parse_ratio(value: str) -> OutputSpec:
    value = value.strip().lower().replace("x", ":")
    named = {
        "9:16": (1080, 1920),
        "16:9": (1920, 1080),
        "1:1": (1080, 1080),
        "4:5": (1080, 1350),
        "5:4": (1350, 1080),
        "3:4": (1080, 1440),
        "4:3": (1440, 1080),
    }
    if value in named:
        width, height = named[value]
        return OutputSpec(width, height, value)
    match = re.fullmatch(r"(\d{3,4}):(\d{3,4})", value)
    if not match:
        raise MontageError(f"Unsupported ratio '{value}'. Use 9:16, 16:9, 1:1, 4:5, or WIDTHxHEIGHT.")
    width, height = int(match.group(1)), int(match.group(2))
    width -= width % 2
    height -= height % 2
    if width < 320 or height < 320 or width > 4096 or height > 4096:
        raise MontageError("Custom output dimensions must be between 320 and 4096 pixels.")
    return OutputSpec(width, height, f"{width}:{height}")


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def _nested_values(data: Any, wanted: set[str]) -> Iterable[Any]:
    if isinstance(data, dict):
        for key, value in data.items():
            if key.lower() in wanted:
                yield value
            yield from _nested_values(value, wanted)
    elif isinstance(data, list):
        for value in data:
            yield from _nested_values(value, wanted)


def _numeric_times(value: Any, duration: float) -> list[float]:
    output: list[float] = []
    if isinstance(value, (int, float)):
        number = _as_float(value, -1.0)
        if 0.0 <= number <= duration:
            output.append(number)
    elif isinstance(value, dict):
        for key in ("time", "time_seconds", "start", "start_seconds", "boundary", "timestamp"):
            if key in value:
                output.extend(_numeric_times(value[key], duration))
                break
    elif isinstance(value, list):
        for item in value:
            output.extend(_numeric_times(item, duration))
    return output


def extract_musical_times(audio_profile: dict[str, Any], duration: float) -> dict[str, list[float]]:
    groups = {
        "beats": {"beats", "beat_times", "beat_times_seconds"},
        "accents": {"accents", "accent_times", "accent_times_seconds", "emphasis_nodes", "onsets"},
        "phrases": {"phrases", "phrase_boundaries", "phrase_boundaries_seconds"},
        "sections": {"sections", "segments", "section_boundaries", "section_boundaries_seconds"},
        "pauses": {"pauses", "silences", "pause_intervals"},
    }
    result: dict[str, list[float]] = {}
    for name, keys in groups.items():
        times: list[float] = []
        for value in _nested_values(audio_profile, keys):
            if name == "pauses" and isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        times.extend(_numeric_times(item.get("start", item.get("start_seconds")), duration))
                    elif isinstance(item, (list, tuple)) and item:
                        times.extend(_numeric_times(item[0], duration))
            else:
                times.extend(_numeric_times(value, duration))
        result[name] = sorted({round(x, 4) for x in times if 0.02 < x < duration - 0.02})
    return result


def _energy_points(audio_profile: dict[str, Any], duration: float) -> list[tuple[float, float]]:
    candidates = list(
        _nested_values(
            audio_profile,
            {"energy_curve", "normalized_energy", "energy_envelope", "rms_curve", "energy_timeline"},
        )
    )
    points: list[tuple[float, float]] = []
    for candidate in candidates:
        if not isinstance(candidate, list):
            continue
        for index, item in enumerate(candidate):
            if isinstance(item, dict):
                time_value = item.get("time", item.get("time_seconds", item.get("t", index)))
                energy_value = item.get(
                    "value",
                    item.get("energy", item.get("level", item.get("normalized", item.get("local_level", item.get("rms", 0.5))))),
                )
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                time_value, energy_value = item[0], item[1]
            elif isinstance(item, (int, float)):
                time_value = duration * index / max(1, len(candidate) - 1)
                energy_value = item
            else:
                continue
            time_float = _as_float(time_value, -1.0)
            if 0.0 <= time_float <= duration:
                points.append((time_float, _as_float(energy_value, 0.5)))
        if points:
            break
    if not points:
        return [(0.0, 0.5), (duration, 0.5)]
    values = [value for _, value in points]
    low, high = min(values), max(values)
    if high - low > 1e-9:
        points = [(time, max(0.0, min(1.0, (value - low) / (high - low)))) for time, value in points]
    else:
        points = [(time, max(0.0, min(1.0, value))) for time, value in points]
    return sorted(points)


def _energy_at(points: list[tuple[float, float]], time_value: float) -> float:
    if time_value <= points[0][0]:
        return points[0][1]
    for (left_t, left_v), (right_t, right_v) in zip(points, points[1:]):
        if left_t <= time_value <= right_t:
            span = max(1e-9, right_t - left_t)
            alpha = (time_value - left_t) / span
            return left_v + alpha * (right_v - left_v)
    return points[-1][1]


def _audio_sections(audio_profile: dict[str, Any], duration: float) -> list[dict[str, Any]]:
    for value in _nested_values(audio_profile, {"sections"}):
        if not isinstance(value, list):
            continue
        sections: list[dict[str, Any]] = []
        for index, item in enumerate(value):
            if not isinstance(item, dict):
                continue
            start = _as_float(item.get("start", item.get("start_seconds")), -1.0)
            end = _as_float(item.get("end", item.get("end_seconds")), -1.0)
            if 0.0 <= start < end and start < duration:
                sections.append({**item, "index": int(item.get("index", index)), "start": start, "end": min(duration, end)})
        if sections:
            return sorted(sections, key=lambda section: section["start"])
    return []


def _extract_assets(media_result: Any) -> list[dict[str, Any]]:
    if isinstance(media_result, list):
        raw_assets = media_result
    elif isinstance(media_result, dict):
        raw_assets = []
        for key in ("selected", "selected_assets", "assets", "downloads", "items", "videos"):
            value = media_result.get(key)
            if isinstance(value, list) and value:
                raw_assets = value
                break
        if not raw_assets:
            for value in media_result.values():
                if isinstance(value, list) and value and all(isinstance(item, dict) for item in value):
                    if any(any(k in item for k in ("local_path", "path", "file")) for item in value):
                        raw_assets = value
                        break
    else:
        raw_assets = []

    assets: list[dict[str, Any]] = []
    for item in raw_assets:
        if not isinstance(item, dict):
            continue
        path_value = item.get("local_path", item.get("path", item.get("file", item.get("download_path"))))
        if not path_value:
            continue
        path = Path(str(path_value)).expanduser().resolve()
        if not path.is_file():
            continue
        quality = item.get("quality", {}) if isinstance(item.get("quality"), dict) else {}
        duration = _as_float(
            item.get("duration", item.get("duration_seconds", quality.get("duration", quality.get("duration_seconds")))),
            0.0,
        )
        assets.append(
            {
                **item,
                "local_path": str(path),
                "duration": max(0.0, duration),
                "asset_id": str(item.get("pixabay_id", item.get("id", item.get("asset_id", path.stem)))),
                "score": _as_float(
                    item.get(
                        "score",
                        item.get(
                            "final_score",
                            quality.get(
                                "overall_score",
                                item.get("diversity_adjusted_score", item.get("pre_score", quality.get("score", 0.5))),
                            ),
                        ),
                    ),
                    0.5,
                ),
                "motion": _as_float(
                    item.get("motion", item.get("motion_score", quality.get("motion", quality.get("motion_score", 0.5)))),
                    0.5,
                ),
                "shot_scale": str(item.get("shot_scale", quality.get("shot_scale", "unknown"))),
            }
        )
    if not assets:
        raise MontageError("Pixabay stage did not return any usable local video files.")
    return assets


def _style_average_shot(style_profile: dict[str, Any]) -> float:
    for value in _nested_values(
        style_profile,
        {"average_shot_duration", "average_shot_duration_seconds", "avg_shot_duration", "median_shot_duration"},
    ):
        number = _as_float(value, 0.0)
        if 0.35 <= number <= 8.0:
            return number
    return 2.0


def _choose_boundary(
    current: float,
    ideal: float,
    duration: float,
    musical_times: dict[str, list[float]],
    energy: float,
) -> tuple[float, str]:
    min_length = 0.48 if energy > 0.72 else 0.72
    max_length = 1.8 if energy > 0.75 else (2.8 if energy > 0.4 else 4.5)
    low = current + min_length
    high = min(duration, current + max_length)
    target = min(high, max(low, ideal))
    if energy < 0.25:
        priority = ["pauses", "phrases", "sections", "beats", "accents"]
    elif energy > 0.58:
        priority = ["accents", "beats", "phrases", "sections", "pauses"]
    else:
        priority = ["phrases", "sections", "beats", "pauses", "accents"]
    best: tuple[float, str, float] | None = None
    tolerance = 0.42 if energy > 0.58 else 0.72
    for group in priority:
        for point in musical_times[group]:
            if low <= point <= high:
                distance = abs(point - target)
                group_bias = priority.index(group) * 0.035
                candidate = (point, group, distance + group_bias)
                if best is None or candidate[2] < best[2]:
                    best = candidate
    if best and abs(best[0] - target) <= tolerance:
        return best[0], best[1]
    return target, "energy_grid"


def build_timeline(
    audio_profile: dict[str, Any],
    media_result: Any,
    duration: float,
    style_profile: dict[str, Any] | None = None,
    seed: str = "bgm-montage",
) -> dict[str, Any]:
    """Create a deterministic, beat-aware edit decision list."""
    if duration <= 0:
        raise MontageError("Target duration must be positive.")
    style_profile = style_profile or {}
    assets = _extract_assets(media_result)
    musical_times = extract_musical_times(audio_profile, duration)
    energy_points = _energy_points(audio_profile, duration)
    audio_sections = _audio_sections(audio_profile, duration)
    reference_shot = _style_average_shot(style_profile)
    rng = random.Random(seed)
    usage_count = {asset["asset_id"]: 0 for asset in assets}
    recent_ids: list[str] = []
    timeline: list[dict[str, Any]] = []
    current = 0.0
    role_cycle = ["wide", "medium", "detail"]

    while current < duration - 0.03:
        energy = _energy_at(energy_points, current + 0.05)
        musical_target = 0.72 + (1.0 - energy) * 2.65
        ideal_length = 0.65 * musical_target + 0.35 * max(0.6, min(3.5, reference_shot))
        proposed = current + ideal_length
        end, cut_reason = _choose_boundary(current, proposed, duration, musical_times, energy)
        end = min(duration, max(current + 0.36, end))
        if duration - end < 0.28:
            end = duration
        output_duration = end - current
        active_section = next(
            (section for section in audio_sections if section["start"] <= current < section["end"]),
            None,
        )
        repetition = active_section.get("repetition", {}) if active_section else {}
        repeat_group = repetition.get("group") if isinstance(repetition, dict) else None
        repeat_shift = int(active_section.get("index", 0)) % len(role_cycle) if repeat_group else 0
        desired_role = role_cycle[(len(timeline) + repeat_shift) % len(role_cycle)]
        motion_variation = ((repeat_shift - 1) * 0.12) if repeat_group else 0.0
        desired_motion = max(0.05, min(0.95, 0.25 + 0.7 * energy + motion_variation))

        def asset_rank(asset: dict[str, Any]) -> tuple[float, float]:
            motion_match = 1.0 - abs(asset["motion"] - desired_motion)
            repetition_penalty = 0.55 * usage_count[asset["asset_id"]]
            recent_penalty = 1.5 if asset["asset_id"] in recent_ids[-2:] else 0.0
            role_bonus = 0.22 if desired_role in asset["shot_scale"].lower() else 0.0
            jitter = rng.random() * 0.05
            return (asset["score"] + 0.5 * motion_match + role_bonus - repetition_penalty - recent_penalty + jitter, asset["score"])

        asset = max(assets, key=asset_rank)
        clip_duration = asset["duration"] or max(4.0, output_duration * 2.0)
        speed = 0.90 + 0.28 * energy
        if asset["motion"] > 0.75 and energy < 0.35:
            speed = 0.82
        elif asset["motion"] < 0.25 and energy > 0.72:
            speed = 1.24
        speed = max(0.72, min(1.32, speed))
        source_duration = output_duration * speed
        if source_duration > clip_duration * 0.96:
            # Preserve the requested output duration even for unusually short
            # clips.  A lower speed is preferable to silently ending the video
            # stream before the BGM/container duration.
            speed = max(0.05, clip_duration * 0.96 / max(0.01, output_duration))
            source_duration = min(clip_duration * 0.96, output_duration * speed)
        available_start = max(0.0, clip_duration - source_duration - 0.05)
        use_index = usage_count[asset["asset_id"]]
        source_start = 0.0 if available_start <= 0 else (available_start * ((use_index * 0.37 + len(timeline) * 0.19) % 1.0))
        source_end = min(clip_duration, source_start + source_duration)
        timeline.append(
            {
                "index": len(timeline),
                "output_start": round(current, 4),
                "output_end": round(end, 4),
                "output_duration": round(output_duration, 4),
                "local_path": asset["local_path"],
                "asset_id": asset["asset_id"],
                "pixabay_id": asset.get("pixabay_id", asset.get("id")),
                "page_url": asset.get("page_url", asset.get("pageURL")),
                "search_query": asset.get("search_query", asset.get("query")),
                "source_start": round(source_start, 4),
                "source_end": round(source_end, 4),
                "speed": round(speed, 4),
                "energy": round(energy, 4),
                "cut_reason": cut_reason,
                "desired_shot_role": desired_role,
                "source_motion": round(asset["motion"], 4),
                "audio_section_index": active_section.get("index") if active_section else None,
                "audio_section_role": active_section.get("role") if active_section else None,
                "audio_repetition_group": repeat_group,
                "repeat_pass_variation": {
                    "shot_role_shift": repeat_shift,
                    "target_motion": round(desired_motion, 4),
                }
                if repeat_group
                else None,
            }
        )
        usage_count[asset["asset_id"]] += 1
        recent_ids.append(asset["asset_id"])
        current = end

    return {
        "schema_version": 1,
        "duration_seconds": round(duration, 4),
        "reference_average_shot_seconds": round(reference_shot, 4),
        "musical_alignment_points": musical_times,
        "shots": timeline,
        "asset_usage_counts": {key: value for key, value in usage_count.items() if value},
    }


def _find_style_scalar(profile: dict[str, Any], keys: set[str], default: float) -> float:
    for value in _nested_values(profile, keys):
        if isinstance(value, (int, float)):
            return _as_float(value, default)
    return default


def _eq_values(style_profile: dict[str, Any]) -> tuple[float, float, float]:
    brightness_raw = _find_style_scalar(style_profile, {"brightness", "mean_brightness", "luma_mean"}, 0.5)
    saturation_raw = _find_style_scalar(style_profile, {"saturation", "mean_saturation"}, 0.5)
    contrast_raw = _find_style_scalar(style_profile, {"contrast", "mean_contrast"}, 0.5)
    if brightness_raw > 1.5:
        brightness_raw /= 255.0
    if saturation_raw > 1.5:
        saturation_raw /= 255.0
    if contrast_raw > 2.0:
        contrast_raw /= 64.0
    brightness = max(-0.06, min(0.06, (brightness_raw - 0.5) * 0.09))
    saturation = max(0.82, min(1.28, 0.88 + saturation_raw * 0.42))
    contrast = max(0.92, min(1.18, 0.96 + contrast_raw * 0.14))
    return brightness, saturation, contrast


def render_timeline(
    plan: dict[str, Any],
    bgm_path: str | Path,
    output_path: str | Path,
    ratio: str,
    style_profile: dict[str, Any] | None = None,
    ffmpeg: str = "ffmpeg",
    fps: int = 30,
) -> Path:
    """Render the timeline to H.264/AAC and return the resolved output path."""
    if not shutil.which(ffmpeg) and not Path(ffmpeg).is_file():
        raise MontageError("ffmpeg is not available on PATH.")
    bgm = Path(bgm_path).expanduser().resolve()
    if not bgm.is_file():
        raise MontageError(f"BGM file does not exist: {bgm}")
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    shots = plan.get("shots") or []
    if not shots:
        raise MontageError("Edit plan contains no shots.")
    spec = parse_ratio(ratio)
    style_profile = style_profile or {}
    brightness, saturation, contrast = _eq_values(style_profile)
    duration = _as_float(plan.get("duration_seconds"), 0.0)
    if duration <= 0:
        duration = max(_as_float(shot.get("output_end"), 0.0) for shot in shots)

    command: list[str] = [ffmpeg, "-hide_banner", "-y", "-i", str(bgm)]
    for shot in shots:
        source_start = _as_float(shot.get("source_start"), 0.0)
        source_end = _as_float(shot.get("source_end"), source_start + 1.0)
        source_duration = max(0.05, source_end - source_start)
        command.extend(["-ss", f"{source_start:.5f}", "-t", f"{source_duration:.5f}", "-i", str(shot["local_path"])])

    filters: list[str] = []
    labels: list[str] = []
    for index, shot in enumerate(shots, start=1):
        speed = max(0.05, _as_float(shot.get("speed"), 1.0))
        output_duration = max(0.05, _as_float(shot.get("output_duration"), 1.0))
        label = f"v{index}"
        chain = (
            f"[{index}:v]setpts=(PTS-STARTPTS)/{speed:.6f},"
            f"tpad=stop_mode=clone:stop_duration={output_duration:.6f},"
            f"trim=duration={output_duration:.6f},setpts=PTS-STARTPTS,"
            f"scale={spec.width}:{spec.height}:force_original_aspect_ratio=increase:flags=lanczos,"
            f"crop={spec.width}:{spec.height},fps={fps},setsar=1,"
            f"eq=brightness={brightness:.5f}:saturation={saturation:.5f}:contrast={contrast:.5f},"
            f"format=yuv420p[{label}]"
        )
        filters.append(chain)
        labels.append(f"[{label}]")
    filters.append(f"{''.join(labels)}concat=n={len(labels)}:v=1:a=0[vout]")
    fade_out_start = max(0.0, duration - min(1.2, duration * 0.08))
    filters.append(
        f"[0:a]atrim=0:{duration:.6f},asetpts=PTS-STARTPTS,"
        f"afade=t=in:st=0:d={min(0.35, duration * 0.04):.4f},"
        f"afade=t=out:st={fade_out_start:.4f}:d={max(0.05, duration - fade_out_start):.4f},"
        "loudnorm=I=-14:TP=-1.5:LRA=11[aout]"
    )
    command.extend(
        [
            "-filter_complex",
            ";".join(filters),
            "-map",
            "[vout]",
            "-map",
            "[aout]",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-r",
            str(fps),
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-t",
            f"{duration:.6f}",
            "-movflags",
            "+faststart",
            str(output),
        ]
    )
    process = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if process.returncode != 0:
        tail = "\n".join(process.stderr.splitlines()[-30:])
        raise MontageError(f"FFmpeg render failed:\n{tail}")
    if not output.is_file() or output.stat().st_size < 1024:
        raise MontageError("FFmpeg returned success but the output file is missing or empty.")
    return output


def write_plan(plan: dict[str, Any], path: str | Path) -> Path:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    return output
