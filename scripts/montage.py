#!/usr/bin/env python3
"""Build and render a signal-driven montage timeline with FFmpeg."""

from __future__ import annotations

import json
import hashlib
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


class InsufficientMaterialError(MontageError):
    """Raised when diversity/reuse/screen-share constraints cannot be met."""


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


def _infer_scene_category(tags: str) -> str:
    text = tags.lower()
    rules = (
        ("nature", ("nature", "forest", "mountain", "landscape", "tree", "wildlife")),
        ("water_coast", ("ocean", "sea", "coast", "beach", "water", "wave")),
        ("architecture", ("architecture", "building", "city", "urban", "interior")),
        ("transport", ("road", "traffic", "car", "vehicle", "train", "airplane")),
        ("industrial", ("factory", "machine", "workshop", "industrial", "production")),
        ("technology", ("technology", "computer", "digital", "electronics")),
        ("food", ("food", "cooking", "kitchen", "coffee")),
        ("people", ("people", "person", "woman", "man", "portrait", "interview")),
        ("abstract", ("abstract", "background", "graphic")),
    )
    for category, terms in rules:
        if any(term in text for term in terms):
            return category
    return "general"


def _face_risk_from_asset(item: dict[str, Any], quality: dict[str, Any]) -> float:
    for source in (item, quality, item.get("thumbnail_signals", {})):
        if not isinstance(source, dict):
            continue
        for key in ("face_content_risk", "face_risk", "prominent_face_risk"):
            if key in source:
                return max(0.0, min(1.0, _as_float(source[key], 0.0)))
    tags = str(item.get("tags", "")).lower()
    if any(term in tags for term in ("selfie", "interview", "portrait", "face", "close up person")):
        return 0.9
    if any(term in tags for term in ("woman", "man", "people", "person", "model")):
        return 0.45
    return 0.05


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
                "scene_category": str(
                    item.get(
                        "scene_category",
                        quality.get("scene_category", _infer_scene_category(str(item.get("tags", "")))),
                    )
                ),
                "face_risk": _face_risk_from_asset(item, quality),
                "subject_profile": item.get("subject_profile", quality.get("subject_profile", {})),
                "width": int(
                    _as_float(
                        item.get("width", item.get("resolution", {}).get("width", quality.get("width", 0)))
                        if isinstance(item.get("resolution", {}), dict)
                        else item.get("width", quality.get("width", 0)),
                        0.0,
                    )
                ),
                "height": int(
                    _as_float(
                        item.get("height", item.get("resolution", {}).get("height", quality.get("height", 0)))
                        if isinstance(item.get("resolution", {}), dict)
                        else item.get("height", quality.get("height", 0)),
                        0.0,
                    )
                ),
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


def default_content_policy(duration: float) -> dict[str, Any]:
    """Return the hard anti-repetition and face-exposure policy."""

    return {
        "min_unique_assets": max(4, min(12, math.ceil(duration / 3.5))),
        "max_reuse_per_asset": 2,
        "max_asset_screen_share": 0.30,
        "min_scene_categories": 3 if duration >= 6.0 else 2,
        "max_prominent_face_screen_share": 0.15,
        "prominent_face_threshold": 0.65,
    }


def evaluate_material_sufficiency(
    assets: list[dict[str, Any]],
    duration: float,
    policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    configured = {**default_content_policy(duration), **(policy or {})}
    unique_ids = {asset["asset_id"] for asset in assets}
    scenes = {asset.get("scene_category", "general") for asset in assets}
    non_face = [
        asset for asset in assets
        if _as_float(asset.get("face_risk"), 0.0) < _as_float(configured["prominent_face_threshold"], 0.65)
    ]
    failures: list[str] = []
    if len(unique_ids) < int(configured["min_unique_assets"]):
        failures.append(
            f"independent assets {len(unique_ids)} < {int(configured['min_unique_assets'])}"
        )
    if len(scenes) < int(configured["min_scene_categories"]):
        failures.append(
            f"scene categories {len(scenes)} < {int(configured['min_scene_categories'])}"
        )
    required_non_face = max(1, math.ceil(int(configured["min_unique_assets"]) * 0.65))
    if len(non_face) < required_non_face:
        failures.append(
            f"low-face-risk assets {len(non_face)} < {required_non_face}"
        )
    return {
        "passed": not failures,
        "failures": failures,
        "policy": configured,
        "available_unique_assets": len(unique_ids),
        "available_scene_categories": sorted(scenes),
        "available_low_face_risk_assets": len(non_face),
    }


def _mapping_at(data: Any, names: tuple[str, ...]) -> dict[str, Any]:
    if not isinstance(data, dict):
        return {}
    for name in names:
        value = data.get(name)
        if isinstance(value, dict):
            return value
    for value in data.values():
        found = _mapping_at(value, names)
        if found:
            return found
    return {}


def _grammar_event_weights(grammar: dict[str, Any]) -> dict[str, float]:
    raw = _mapping_at(grammar, ("event_weights", "cut_event_weights", "primary_distribution"))
    aliases = {
        "strong_beat": "accents", "accent": "accents", "accents": "accents",
        "downbeat": "accents", "weak_beat": "beats", "beat": "beats", "beats": "beats",
        "phrase": "phrases", "phrase_boundary": "phrases", "phrases": "phrases",
        "section": "sections", "section_boundary": "sections", "sections": "sections",
        "pause": "pauses", "pause_edge": "pauses", "pauses": "pauses",
        "offgrid": "energy_grid",
    }
    weights = {"accents": 1.0, "beats": 0.7, "phrases": 0.9, "sections": 1.0, "pauses": 0.65, "energy_grid": 0.2}
    if raw:
        collected: dict[str, list[float]] = {}
        for key, value in raw.items():
            canonical = aliases.get(str(key).lower())
            if canonical and isinstance(value, (int, float)):
                collected.setdefault(canonical, []).append(max(0.0, float(value)))
        if collected:
            maximum = max(max(values) for values in collected.values()) or 1.0
            for key, values in collected.items():
                weights[key] = max(values) / maximum
    return weights


def _grammar_event_offsets(grammar: dict[str, Any]) -> dict[str, float]:
    raw = _mapping_at(grammar, ("boundary_offsets_seconds", "event_offsets_seconds"))
    aliases = {
        "strong_accent": "accents", "accent": "accents", "downbeat": "accents",
        "weak_beat": "beats", "beat": "beats", "phrase_boundary": "phrases",
        "section_boundary": "sections", "pause_edge": "pauses",
    }
    grouped: dict[str, list[float]] = {}
    for key, value in raw.items():
        canonical = aliases.get(str(key).lower())
        if canonical and isinstance(value, (int, float)):
            grouped.setdefault(canonical, []).append(max(-0.35, min(0.35, float(value))))
    return {
        key: sum(values) / len(values)
        for key, values in grouped.items()
        if values
    }


def _grammar_duration(
    grammar: dict[str, Any], energy: float, fallback: float, beat_seconds: float
) -> float:
    raw = _mapping_at(grammar, ("energy_shot_duration_seconds", "shot_duration_by_energy", "energy_duration"))
    band = "high" if energy >= 0.66 else ("low" if energy <= 0.33 else "mid")
    value = raw.get(band, raw.get(f"{band}_energy")) if raw else None
    if isinstance(value, dict):
        beats = _as_float(value.get("median_beats", value.get("beats_median")), 0.0)
        if beats > 0 and beat_seconds > 0:
            return max(0.35, min(8.0, beats * beat_seconds))
        value = value.get("median", value.get("median_seconds", value.get("p50")))
    parsed = _as_float(value, 0.0)
    return parsed if 0.35 <= parsed <= 8.0 else fallback


def _transition_choice(matrix: dict[str, Any], previous: str, fallback_cycle: list[str], index: int) -> str:
    row = matrix.get(previous, {}) if isinstance(matrix, dict) else {}
    if not row and previous in {"wide", "medium", "detail"} and isinstance(matrix, dict):
        row = next(
            (
                value
                for key, value in matrix.items()
                if _normalize_scale(str(key)) == previous and isinstance(value, dict)
            ),
            {},
        )
    if isinstance(row, dict) and row:
        valid = [(str(key), _as_float(value, 0.0)) for key, value in row.items()]
        valid = [item for item in valid if item[1] > 0]
        if valid:
            return max(valid, key=lambda item: (item[1], item[0]))[0]
    return fallback_cycle[index % len(fallback_cycle)]


def _normalize_scale(value: str) -> str:
    text = value.lower()
    if "wide" in text or "establish" in text:
        return "wide"
    if "close" in text or "detail" in text or "macro" in text:
        return "detail"
    return "medium"


def plan_subject_crop(asset: dict[str, Any], spec: OutputSpec) -> dict[str, Any]:
    """Plan a saliency/subject-aware crop, or use blurred contain fallback."""

    width = max(1, int(asset.get("width") or 0))
    height = max(1, int(asset.get("height") or 0))
    if width <= 1 or height <= 1:
        return {"mode": "blur_fill", "reason": "source dimensions unavailable", "retention": 1.0}
    source_ratio = width / height
    target_ratio = spec.width / spec.height
    profile = asset.get("subject_profile", {}) if isinstance(asset.get("subject_profile"), dict) else {}
    center = profile.get("center", {}) if isinstance(profile.get("center"), dict) else {}
    center_x = max(0.0, min(1.0, _as_float(center.get("x"), 0.5)))
    center_y = max(0.0, min(1.0, _as_float(center.get("y"), 0.5)))
    bbox = profile.get("bbox", [0.15, 0.15, 0.85, 0.85])
    if not isinstance(bbox, list) or len(bbox) != 4:
        bbox = [0.15, 0.15, 0.85, 0.85]
    bbox = [max(0.0, min(1.0, _as_float(value, 0.5))) for value in bbox]
    confidence = _as_float(profile.get("confidence"), 0.0)
    if abs(source_ratio - target_ratio) / max(source_ratio, target_ratio) < 0.035:
        return {"mode": "fit", "crop_rect_norm": [0.0, 0.0, 1.0, 1.0], "retention": 1.0, "confidence": confidence}
    if source_ratio > target_ratio:
        crop_width = target_ratio / source_ratio
        left = max(0.0, min(1.0 - crop_width, center_x - crop_width / 2.0))
        rect = [left, 0.0, left + crop_width, 1.0]
    else:
        crop_height = source_ratio / target_ratio
        top = max(0.0, min(1.0 - crop_height, center_y - crop_height / 2.0))
        rect = [0.0, top, 1.0, top + crop_height]
    bbox_area = max(1e-6, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
    overlap = max(0.0, min(rect[2], bbox[2]) - max(rect[0], bbox[0])) * max(
        0.0, min(rect[3], bbox[3]) - max(rect[1], bbox[1])
    )
    retention = max(0.0, min(1.0, overlap / bbox_area))
    spread = profile.get("center_spread", {}) if isinstance(profile.get("center_spread"), dict) else {}
    unstable = max(_as_float(spread.get("x"), 0.0), _as_float(spread.get("y"), 0.0)) > 0.16
    if confidence < 0.12 and min(source_ratio / target_ratio, target_ratio / source_ratio) < 0.70:
        return {"mode": "blur_fill", "reason": "low subject confidence on severe ratio conversion", "retention": 1.0, "confidence": confidence}
    if retention < 0.85 or unstable:
        return {"mode": "blur_fill", "reason": "unsafe subject retention", "estimated_crop_retention": round(retention, 4), "retention": 1.0, "confidence": confidence}
    return {
        "mode": "subject_crop",
        "crop_rect_norm": [round(value, 6) for value in rect],
        "retention": round(retention, 4),
        "confidence": round(confidence, 4),
    }


def _choose_boundary(
    current: float,
    ideal: float,
    duration: float,
    musical_times: dict[str, list[float]],
    energy: float,
    event_weights: dict[str, float] | None = None,
    event_offsets: dict[str, float] | None = None,
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
    weights = event_weights or {}
    offsets = event_offsets or {}
    for group in priority:
        for point in musical_times[group]:
            learned_point = point + _as_float(offsets.get(group), 0.0)
            if low <= learned_point <= high:
                distance = abs(learned_point - target)
                learned_weight = max(0.0, min(1.0, _as_float(weights.get(group), 0.5)))
                group_bias = priority.index(group) * 0.025 + (1.0 - learned_weight) * 0.28
                candidate = (learned_point, group, distance + group_bias)
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
    editing_grammar: dict[str, Any] | None = None,
    content_policy: dict[str, Any] | None = None,
    ratio: str = "16:9",
) -> dict[str, Any]:
    """Create a deterministic timeline constrained by learned editing grammar."""
    if duration <= 0:
        raise MontageError("Target duration must be positive.")
    style_profile = style_profile or {}
    editing_grammar = editing_grammar or {}
    assets = _extract_assets(media_result)
    sufficiency = evaluate_material_sufficiency(assets, duration, content_policy)
    if not sufficiency["passed"]:
        raise InsufficientMaterialError(
            "Material sufficiency gate failed: " + "; ".join(sufficiency["failures"])
        )
    policy = sufficiency["policy"]
    spec = parse_ratio(ratio)
    musical_times = extract_musical_times(audio_profile, duration)
    energy_points = _energy_points(audio_profile, duration)
    audio_sections = _audio_sections(audio_profile, duration)
    reference_shot = _style_average_shot(style_profile)
    beat_gaps = [
        right - left
        for left, right in zip(musical_times["beats"], musical_times["beats"][1:])
        if 0.18 <= right - left <= 2.0
    ]
    beat_seconds = sorted(beat_gaps)[len(beat_gaps) // 2] if beat_gaps else 0.5
    event_weights = _grammar_event_weights(editing_grammar)
    event_offsets = _grammar_event_offsets(editing_grammar)
    scale_matrix = _mapping_at(editing_grammar, ("scale_transition_matrix", "shot_scale_transition_matrix"))
    motion_matrix = _mapping_at(editing_grammar, ("motion_transition_matrix", "camera_motion_transition_matrix"))
    scene_matrix = _mapping_at(editing_grammar, ("scene_transition_matrix",))
    grammar_policy = editing_grammar.get("montage_policy", {}) if isinstance(editing_grammar, dict) else {}
    ending = (
        grammar_policy.get("ending", {})
        if isinstance(grammar_policy, dict) and isinstance(grammar_policy.get("ending"), dict)
        else _mapping_at(editing_grammar, ("ending", "ending_structure"))
    )
    transition_distribution = _mapping_at(
        editing_grammar, ("transition_distribution", "supported_transition_distribution")
    )
    rng = random.Random(seed)
    usage_count = {asset["asset_id"]: 0 for asset in assets}
    screen_time = {asset["asset_id"]: 0.0 for asset in assets}
    prominent_face_time = 0.0
    recent_ids: list[str] = []
    recent_scenes: list[str] = []
    timeline: list[dict[str, Any]] = []
    current = 0.0
    role_cycle = ["wide", "medium", "detail"]
    max_share_seconds = float(policy["max_asset_screen_share"]) * duration
    max_face_seconds = float(policy["max_prominent_face_screen_share"]) * duration
    prominent_threshold = float(policy["prominent_face_threshold"])
    previous_scale = "wide"
    previous_motion_label = "static_like"
    learned_final_multiplier = max(
        0.6,
        min(
            2.0,
            _as_float(
                ending.get(
                    "final_shot_multiplier",
                    ending.get("last_shot_multiplier", ending.get("last_shot_duration_multiplier")),
                ),
                1.0,
            ),
        ),
    )

    while current < duration - 0.03:
        energy = _energy_at(energy_points, current + 0.05)
        musical_target = 0.72 + (1.0 - energy) * 2.65
        learned_length = _grammar_duration(editing_grammar, energy, reference_shot, beat_seconds)
        ideal_length = 0.48 * musical_target + 0.52 * max(0.45, min(4.5, learned_length))
        if duration - current <= ideal_length * 1.8:
            ideal_length *= learned_final_multiplier
        proposed = current + ideal_length
        end, cut_reason = _choose_boundary(
            current, proposed, duration, musical_times, energy, event_weights, event_offsets
        )
        end = min(duration, max(current + 0.36, end))
        if end - current > max_share_seconds:
            end = current + max_share_seconds
            cut_reason = "content_share_guard"
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
        desired_role = _normalize_scale(
            _transition_choice(
                scale_matrix,
                previous_scale,
                role_cycle,
                len(timeline) + repeat_shift,
            )
        )
        motion_variation = ((repeat_shift - 1) * 0.12) if repeat_group else 0.0
        desired_motion = max(0.05, min(0.95, 0.25 + 0.7 * energy + motion_variation))
        desired_motion_label = _transition_choice(
            motion_matrix,
            previous_motion_label,
            ["static_like", "pan_or_tilt_like", "push_or_pull_like"],
            len(timeline) + repeat_shift,
        )
        scene_row = scene_matrix.get(recent_scenes[-1], {}) if recent_scenes and isinstance(scene_matrix, dict) else {}
        desired_scene = (
            max(scene_row.items(), key=lambda item: _as_float(item[1], 0.0))[0]
            if isinstance(scene_row, dict) and scene_row
            else None
        )

        def asset_rank(asset: dict[str, Any]) -> tuple[float, float]:
            motion_match = 1.0 - abs(asset["motion"] - desired_motion)
            repetition_penalty = 0.85 * usage_count[asset["asset_id"]]
            recent_penalty = 1.5 if asset["asset_id"] in recent_ids[-2:] else 0.0
            role_bonus = 0.34 if desired_role == _normalize_scale(asset["shot_scale"]) else 0.0
            scene_bonus = 0.24 if asset.get("scene_category") not in recent_scenes[-2:] else -0.30
            if isinstance(scene_row, dict):
                scene_bonus += 0.34 * _as_float(scene_row.get(str(asset.get("scene_category"))), 0.0)
            if repeat_group and recent_scenes and asset.get("scene_category") == recent_scenes[-1]:
                scene_bonus -= 0.45
            face_penalty = 1.25 * _as_float(asset.get("face_risk"), 0.0)
            jitter = rng.random() * 0.05
            return (
                asset["score"] + 0.5 * motion_match + role_bonus + scene_bonus
                - repetition_penalty - recent_penalty - face_penalty + jitter,
                asset["score"],
            )

        eligible = []
        for candidate in assets:
            identity = candidate["asset_id"]
            if usage_count[identity] >= int(policy["max_reuse_per_asset"]):
                continue
            if screen_time[identity] + output_duration > max_share_seconds + 0.02:
                continue
            if (
                _as_float(candidate.get("face_risk"), 0.0) >= prominent_threshold
                and prominent_face_time + output_duration > max_face_seconds + 0.02
            ):
                continue
            eligible.append(candidate)
        if len({key for key, count in usage_count.items() if count > 0}) < int(policy["min_unique_assets"]):
            unused = [candidate for candidate in eligible if usage_count[candidate["asset_id"]] == 0]
            if unused:
                eligible = unused
        if len(set(recent_scenes)) < int(policy["min_scene_categories"]):
            new_scene = [candidate for candidate in eligible if candidate.get("scene_category") not in set(recent_scenes)]
            if new_scene:
                eligible = new_scene
        if not eligible:
            raise InsufficientMaterialError(
                "Material constraints became infeasible while planning: reuse, screen-share, or face budget exhausted."
            )
        asset = max(eligible, key=asset_rank)
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
        next_section_boundary = any(
            abs(end - _as_float(section.get("end"), -99.0)) <= 0.10
            for section in audio_sections
        )
        fade_share = max(
            (_as_float(value, 0.0) for key, value in transition_distribution.items() if "fade" in str(key).lower()),
            default=0.0,
        )
        transition_out = "fade_through_black" if next_section_boundary and fade_share >= 0.08 else "hard_cut"
        crop_plan = plan_subject_crop(asset, spec)
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
                "source_motion_label": desired_motion_label,
                "source_shot_scale": _normalize_scale(asset["shot_scale"]),
                "scene_category": asset.get("scene_category"),
                "face_content_risk": round(_as_float(asset.get("face_risk"), 0.0), 4),
                "crop_plan": crop_plan,
                "transition_in": "fade_through_black"
                if timeline and timeline[-1].get("transition_out") == "fade_through_black"
                else "hard_cut",
                "transition_out": transition_out,
                "audio_section_index": active_section.get("index") if active_section else None,
                "audio_section_role": active_section.get("role") if active_section else None,
                "audio_repetition_group": repeat_group,
                "repeat_pass_variation": {
                    "shot_role_shift": repeat_shift,
                    "target_motion": round(desired_motion, 4),
                }
                if repeat_group
                else None,
                "grammar_influence": {
                    "learned_shot_target_seconds": round(learned_length, 4),
                    "event_weight": round(event_weights.get(cut_reason, 0.0), 4),
                    "learned_boundary_offset_seconds": round(event_offsets.get(cut_reason, 0.0), 4),
                    "scale_transition": f"{previous_scale}->{desired_role}",
                    "motion_transition": f"{previous_motion_label}->{desired_motion_label}",
                    "learned_scene_target": desired_scene,
                    "section_boundary_transition": transition_out,
                },
            }
        )
        usage_count[asset["asset_id"]] += 1
        screen_time[asset["asset_id"]] += output_duration
        if _as_float(asset.get("face_risk"), 0.0) >= prominent_threshold:
            prominent_face_time += output_duration
        recent_ids.append(asset["asset_id"])
        recent_scenes.append(str(asset.get("scene_category", "general")))
        previous_scale = _normalize_scale(asset["shot_scale"])
        previous_motion_label = desired_motion_label
        current = end

    if timeline:
        final_fade = _as_float(
            ending.get("fade_out_seconds", ending.get("final_fade_seconds")), 0.0
        )
        timeline[-1]["ending_structure"] = {
            "learned_final_shot_multiplier": round(learned_final_multiplier, 4),
            "visual_fade_out_seconds": round(max(0.0, min(timeline[-1]["output_duration"] * 0.8, final_fade)), 4),
            "hold_last_frame": bool(ending.get("hold_last_frame", False)),
        }

    used_assets = {shot["asset_id"] for shot in timeline}
    used_scenes = {str(shot.get("scene_category", "general")) for shot in timeline}
    final_failures: list[str] = []
    if len(used_assets) < int(policy["min_unique_assets"]):
        final_failures.append(f"used unique assets {len(used_assets)} < {policy['min_unique_assets']}")
    if len(used_scenes) < int(policy["min_scene_categories"]):
        final_failures.append(f"used scene categories {len(used_scenes)} < {policy['min_scene_categories']}")
    actual_max_share = max(screen_time.values(), default=0.0) / duration
    actual_face_share = prominent_face_time / duration
    if actual_max_share > float(policy["max_asset_screen_share"]) + 0.005:
        final_failures.append("single-asset screen share exceeded policy")
    if actual_face_share > float(policy["max_prominent_face_screen_share"]) + 0.005:
        final_failures.append("prominent-face screen share exceeded policy")
    if final_failures:
        raise InsufficientMaterialError("Timeline sufficiency gate failed: " + "; ".join(final_failures))

    grammar_digest = hashlib.sha256(
        json.dumps(editing_grammar, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16] if editing_grammar else None

    return {
        "schema_version": 2,
        "duration_seconds": round(duration, 4),
        "reference_average_shot_seconds": round(reference_shot, 4),
        "editing_grammar_digest": grammar_digest,
        "editing_grammar_applied": bool(editing_grammar),
        "content_policy": policy,
        "sufficiency": {
            **sufficiency,
            "used_unique_assets": len(used_assets),
            "used_scene_categories": sorted(used_scenes),
            "max_asset_screen_share_actual": round(actual_max_share, 4),
            "prominent_face_screen_share_actual": round(actual_face_share, 4),
        },
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
    overwrite: bool = False,
) -> Path:
    """Render the timeline to H.264/AAC and return the resolved output path."""
    if not shutil.which(ffmpeg) and not Path(ffmpeg).is_file():
        raise MontageError("ffmpeg is not available on PATH.")
    bgm = Path(bgm_path).expanduser().resolve()
    if not bgm.is_file():
        raise MontageError(f"BGM file does not exist: {bgm}")
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not overwrite:
        raise MontageError(f"Output already exists; choose a new run_id or pass overwrite=True: {output}")
    shots = plan.get("shots") or []
    if not shots:
        raise MontageError("Edit plan contains no shots.")
    spec = parse_ratio(ratio)
    style_profile = style_profile or {}
    brightness, saturation, contrast = _eq_values(style_profile)
    duration = _as_float(plan.get("duration_seconds"), 0.0)
    if duration <= 0:
        duration = max(_as_float(shot.get("output_end"), 0.0) for shot in shots)

    command: list[str] = [ffmpeg, "-hide_banner", "-y" if overwrite else "-n", "-i", str(bgm)]
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
        pre_label = f"pre{index}"
        filters.append(
            f"[{index}:v]setpts=(PTS-STARTPTS)/{speed:.6f},"
            f"tpad=stop_mode=clone:stop_duration={output_duration:.6f},"
            f"trim=duration={output_duration:.6f},setpts=PTS-STARTPTS[{pre_label}]"
        )
        crop_plan = shot.get("crop_plan", {}) if isinstance(shot.get("crop_plan"), dict) else {}
        crop_mode = str(crop_plan.get("mode", "blur_fill"))
        composed_label = f"composed{index}"
        if crop_mode == "subject_crop":
            rect = crop_plan.get("crop_rect_norm", [0.0, 0.0, 1.0, 1.0])
            if not isinstance(rect, list) or len(rect) != 4:
                rect = [0.0, 0.0, 1.0, 1.0]
            x0, y0, x1, y1 = [_as_float(value, 0.0) for value in rect]
            filters.append(
                f"[{pre_label}]crop=w='iw*{max(0.01, x1 - x0):.8f}':"
                f"h='ih*{max(0.01, y1 - y0):.8f}':x='iw*{max(0.0, x0):.8f}':"
                f"y='ih*{max(0.0, y0):.8f}',"
                f"scale={spec.width}:{spec.height}:flags=lanczos[{composed_label}]"
            )
        elif crop_mode == "fit":
            filters.append(
                f"[{pre_label}]scale={spec.width}:{spec.height}:flags=lanczos[{composed_label}]"
            )
        else:
            bg_label = f"bg{index}"
            fg_label = f"fg{index}"
            filters.append(f"[{pre_label}]split=2[{bg_label}src][{fg_label}src]")
            filters.append(
                f"[{bg_label}src]scale={spec.width}:{spec.height}:force_original_aspect_ratio=increase:flags=lanczos,"
                f"crop={spec.width}:{spec.height},boxblur=24:8[{bg_label}]"
            )
            filters.append(
                f"[{fg_label}src]scale={spec.width}:{spec.height}:force_original_aspect_ratio=decrease:flags=lanczos[{fg_label}]"
            )
            filters.append(
                f"[{bg_label}][{fg_label}]overlay=(W-w)/2:(H-h)/2:shortest=1[{composed_label}]"
            )
        fade_filters: list[str] = []
        transition_duration = min(0.16, output_duration * 0.18)
        if shot.get("transition_in") == "fade_through_black":
            fade_filters.append(f"fade=t=in:st=0:d={transition_duration:.5f}")
        if shot.get("transition_out") == "fade_through_black":
            fade_filters.append(
                f"fade=t=out:st={max(0.0, output_duration - transition_duration):.5f}:d={transition_duration:.5f}"
            )
        ending_structure = shot.get("ending_structure", {}) if isinstance(shot.get("ending_structure"), dict) else {}
        final_fade = min(output_duration * 0.8, _as_float(ending_structure.get("visual_fade_out_seconds"), 0.0))
        if final_fade > 0.02:
            fade_filters.append(
                f"fade=t=out:st={max(0.0, output_duration - final_fade):.5f}:d={final_fade:.5f}"
            )
        suffix = ",".join(
            [
                f"fps={fps}",
                "setsar=1",
                f"eq=brightness={brightness:.5f}:saturation={saturation:.5f}:contrast={contrast:.5f}",
                *fade_filters,
                "format=yuv420p",
            ]
        )
        filters.append(f"[{composed_label}]{suffix}[{label}]")
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
