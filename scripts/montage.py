#!/usr/bin/env python3
"""Build and render a signal-driven montage timeline with FFmpeg."""

from __future__ import annotations

import json
import hashlib
import math
import os
import random
import re
import shutil
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from visual_intelligence import (
    asset_profile_fit,
    asset_visual_features,
    build_light_grade,
    build_visual_style_profile,
    evaluate_sequence_consistency,
    transition_match,
)


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
        ("architecture", ("architecture", "building", "city", "urban", "interior")),
        ("transport", ("road", "traffic", "car", "vehicle", "train", "airplane")),
        ("industrial", ("factory", "machine", "workshop", "industrial", "production")),
        ("technology", ("technology", "computer", "digital", "electronics")),
        ("food", ("food", "cooking", "kitchen", "coffee")),
        ("people", ("people", "person", "woman", "man", "portrait", "interview")),
        ("abstract", ("abstract", "background", "graphic")),
        ("polar_ice", ("polar", "arctic", "antarctic", "glacier", "iceberg", "ice cave")),
        ("sky_space", ("starry", "stars", "galaxy", "milky way", "aurora", "night sky", "cloudscape")),
        ("water_coast", ("ocean", "sea", "coast", "beach", "water", "wave")),
        ("mountain_canyon", ("mountain", "canyon", "cliff", "valley", "alpine", "peak", "summit", "gorge")),
        ("forest_wilderness", ("forest", "woods", "woodland", "jungle", "tree")),
        ("nature", ("nature", "landscape", "wildlife", "wilderness", "outdoors")),
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


def _first_mapping_value(data: Any, keys: tuple[str, ...]) -> Any:
    """Return the first non-empty value from a shallow mapping.

    Pixabay records evolved across v1.0-v1.2.  Keeping the lookup shallow and
    explicit avoids accidentally treating an unrelated nested hash as the
    source identity.
    """

    if not isinstance(data, dict):
        return None
    for key in keys:
        value = data.get(key)
        if value not in (None, "", [], {}):
            return value
    return None


def _fingerprint_token(value: Any) -> str | None:
    if value in (None, "", [], {}):
        return None
    if isinstance(value, dict):
        preferred = _first_mapping_value(
            value,
            (
                "sha256",
                "file_sha256",
                "content_sha256",
                "hash",
                "perceptual_hash",
                "phash",
            ),
        )
        if preferred is not None:
            return str(preferred).strip().lower()
        stable = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(stable.encode("utf-8")).hexdigest()
    return str(value).strip().lower() or None


def canonical_source_key(asset: dict[str, Any]) -> str:
    """Return a cross-project source identity, independent of theme aliases.

    A perceptually reused Pixabay item may have a new Pixabay ID while pointing
    at an existing file.  File hashes/fingerprints and explicit reuse origins
    therefore take precedence over the API ID and local display filename.
    """

    direct = _first_mapping_value(
        asset,
        ("file_hash", "file_sha256", "sha256", "content_sha256", "fingerprint"),
    )
    token = _fingerprint_token(direct)
    if token:
        return f"fingerprint:{token}"
    origin = asset.get("reuse_origin")
    if isinstance(origin, dict):
        origin_value = _first_mapping_value(
            origin,
            (
                "canonical_source_key",
                "file_hash",
                "file_sha256",
                "sha256",
                "fingerprint",
                "duplicate_of",
                "pixabay_id",
                "local_path",
            ),
        )
        origin_token = _fingerprint_token(origin_value)
        if origin_token:
            return f"reuse:{origin_token}"
    elif origin not in (None, ""):
        return f"reuse:{_fingerprint_token(origin)}"
    pixabay_id = _first_mapping_value(asset, ("pixabay_id", "id", "asset_id"))
    if pixabay_id not in (None, ""):
        return f"pixabay:{pixabay_id}"
    path = Path(str(asset.get("local_path") or asset.get("path") or "")).expanduser()
    try:
        normalized = os.path.normcase(str(path.resolve()))
    except OSError:
        normalized = os.path.normcase(str(path))
    return f"path:{normalized}"


def _label_value(data: dict[str, Any], keys: tuple[str, ...], default: str = "unknown") -> str:
    for source in (data, data.get("quality", {}), data.get("semantic", {}), data.get("thumbnail_signals", {})):
        if not isinstance(source, dict):
            continue
        value = _first_mapping_value(source, keys)
        if isinstance(value, dict):
            value = _first_mapping_value(value, ("label", "name", "value", "primary"))
        if isinstance(value, list) and value:
            value = value[0]
        if value not in (None, ""):
            return str(value).strip().lower()
    return default


def _score_value(data: dict[str, Any], keys: tuple[str, ...], default: float) -> float:
    for source in (data, data.get("quality", {}), data.get("thumbnail_signals", {})):
        if not isinstance(source, dict):
            continue
        value = _first_mapping_value(source, keys)
        if value is not None:
            return max(0.0, min(1.0, _as_float(value, default)))
    return max(0.0, min(1.0, default))


def _terms(value: Any) -> set[str]:
    text_parts: list[str] = []
    if isinstance(value, str):
        text_parts.append(value)
    elif isinstance(value, dict):
        for item in value.values():
            if isinstance(item, (str, list, tuple, set, dict)):
                text_parts.extend(_terms(item))
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            text_parts.extend(_terms(item))
    if text_parts and all(isinstance(item, str) and " " not in item for item in text_parts):
        # Recursive calls return normalized tokens.  Avoid reparsing them as a
        # single concatenated phrase.
        return {str(item) for item in text_parts if item}
    text = " ".join(str(item) for item in text_parts).lower()
    stop = {
        "the", "a", "an", "and", "or", "of", "in", "on", "with", "for",
        "cinematic", "video", "footage", "scene", "shot", "visual",
    }
    return {
        token
        for token in re.findall(r"[a-z0-9][a-z0-9_-]{1,}|[\u4e00-\u9fff]{1,8}", text)
        if token not in stop
    }


def _semantic_overlap(left: Any, right: Any) -> float:
    left_terms, right_terms = _terms(left), _terms(right)
    if not left_terms or not right_terms:
        return 0.0
    return len(left_terms & right_terms) / max(1, min(len(left_terms), len(right_terms)))


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
        visual_analysis = (
            quality.get("visual_analysis", {})
            if isinstance(quality.get("visual_analysis"), dict)
            else {}
        )
        mean_hsv = quality.get("mean_hsv", {}) if isinstance(quality.get("mean_hsv"), dict) else {}
        motion_signals = (
            quality.get("motion_signals", {})
            if isinstance(quality.get("motion_signals"), dict)
            else {}
        )
        duration = _as_float(
            item.get("duration", item.get("duration_seconds", quality.get("duration", quality.get("duration_seconds")))),
            0.0,
        )
        normalized_asset = {
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
                "subject_label": _label_value(item, ("subject", "subject_label", "primary_subject"), "unknown"),
                "composition": _label_value(item, ("composition", "composition_label"), "unknown"),
                "color_tendency": _label_value(
                    item, ("color_tendency", "color_label", "dominant_color", "tone"), "unknown"
                ),
                "mean_hsv": {
                    "hue_degrees": _as_float(mean_hsv.get("hue_degrees"), 0.0),
                    "saturation": _as_float(mean_hsv.get("saturation"), 0.0),
                    "value": _as_float(mean_hsv.get("value"), 0.0),
                },
                "motion_signals": dict(motion_signals),
                "motion_label": _label_value(
                    item, ("motion_label", "camera_motion", "camera_motion_label"), str(visual_analysis.get("motion_type") or "unknown")
                ),
                "motion_direction": _label_value(
                    item, ("motion_direction", "camera_motion_direction", "direction"), "unknown"
                ),
                "stability_score": _score_value(
                    item, ("stability_score", "stability", "stable_score"), 0.65
                ),
                "quality_score": _score_value(
                    item, ("overall_score", "quality_score", "visual_quality_score", "score"), 0.65
                ),
                "visual_analysis": dict(visual_analysis),
                "history_usage_count": int(
                    max(
                        0.0,
                        _as_float(
                            item.get(
                                "history_usage_count",
                                item.get("historical_usage_count", len(item.get("usage_intervals", []) or [])),
                            ),
                            0.0,
                        ),
                    )
                ),
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
        if normalized_asset["motion_direction"] == "unknown":
            normalized_asset["motion_direction"] = str(
                quality.get("motion_direction", motion_signals.get("motion_direction", "unknown"))
            ).strip().lower()
        normalized_asset["canonical_source_key"] = canonical_source_key(normalized_asset)
        tags = str(normalized_asset.get("tags", "")).lower()
        normalized_asset["is_aerial"] = bool(
            normalized_asset.get("is_aerial")
            or any(term in tags for term in ("aerial", "drone", "bird's eye", "top down"))
        )
        normalized_asset["is_static_like"] = bool(
            normalized_asset.get("is_static_like")
            or normalized_asset["motion"] < 0.22
            or normalized_asset["motion_label"] in {"static", "static_like", "locked", "tripod"}
        )
        assets.append(normalized_asset)
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
        # v1.2 defaults to one use per canonical source.  Explicitly raising
        # this value is the bounded material-shortage fallback.
        "max_reuse_per_asset": 1,
        "max_asset_screen_share": 0.30,
        "min_scene_categories": 3 if duration >= 6.0 else 2,
        "max_prominent_face_screen_share": 0.15,
        "prominent_face_threshold": 0.65,
        "min_repeat_gap_shots": 3,
        "min_repeat_gap_seconds": 6.0,
        "max_source_interval_overlap": 0.02,
        "max_adjacent_similarity_dimensions": 3,
        "max_soft_transition_share": 0.28,
    }


def evaluate_material_sufficiency(
    assets: list[dict[str, Any]],
    duration: float,
    policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    configured = {**default_content_policy(duration), **(policy or {})}
    unique_ids = {asset.get("canonical_source_key") or canonical_source_key(asset) for asset in assets}
    scenes = {
        str(asset.get("scene_category") or "general")
        for asset in assets
        if str(asset.get("scene_category") or "general") not in {"", "unknown"}
    }
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
        "available_canonical_sources": len(unique_ids),
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
        "strong_beat": "accents", "strong_accent": "accents", "accent": "accents", "accents": "accents",
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


def _normalize_transition(value: Any) -> str:
    if isinstance(value, dict):
        value = _first_mapping_value(value, ("type", "name", "transition", "recommended"))
    text = str(value or "hard_cut").strip().lower().replace("-", "_").replace(" ", "_")
    if text in {"crossfade", "cross_fade", "dissolve", "blend"}:
        return "dissolve"
    if text in {"fade", "soft_fade"}:
        return "fade"
    if text in {"fadeblack", "fade_black", "fade_through_black", "black_fade"}:
        return "fade_through_black"
    return "hard_cut"


def _slot_motion_target(value: Any, energy: float) -> tuple[float, str]:
    if isinstance(value, dict):
        numeric = _as_float(_first_mapping_value(value, ("score", "intensity", "value")), -1.0)
        label = str(_first_mapping_value(value, ("label", "type", "direction")) or "unknown").lower()
        if numeric >= 0:
            return max(0.0, min(1.0, numeric)), label
        value = label
    if isinstance(value, (int, float)):
        return max(0.0, min(1.0, float(value))), "intensity"
    text = str(value or "").lower()
    if any(term in text for term in ("high", "strong", "fast", "dynamic", "aerial")):
        return 0.84, text or "high"
    if any(term in text for term in ("low", "slow", "calm", "static", "locked")):
        return 0.20, text or "low"
    if text:
        return max(0.25, min(0.80, 0.28 + 0.62 * energy)), text
    return max(0.20, min(0.90, 0.25 + 0.70 * energy)), "energy_driven"


def _coerce_timeline_slots(
    timeline_plan: dict[str, Any] | None,
    slots: list[dict[str, Any]] | None,
    duration: float,
) -> list[dict[str, Any]] | None:
    raw: Any = slots
    if raw is None and isinstance(timeline_plan, dict):
        raw = timeline_plan.get("slots") or timeline_plan.get("timeline_slots")
    if raw is None:
        return None
    if not isinstance(raw, list) or not raw:
        raise MontageError("timeline_plan/slots must contain at least one slot.")
    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise MontageError(f"Timeline slot {index} is not an object.")
        start = _as_float(item.get("start", item.get("output_start")), -1.0)
        end = _as_float(item.get("end", item.get("output_end")), -1.0)
        if end <= start and start >= 0:
            end = start + _as_float(item.get("duration", item.get("output_duration")), 0.0)
        if start < 0 or end <= start:
            raise MontageError(f"Timeline slot {index} has an invalid start/end interval.")
        normalized.append(
            {
                **item,
                "index": index,
                "start": round(start, 6),
                "end": round(end, 6),
                "duration": round(end - start, 6),
                "section_index": item.get("section_index", item.get("audio_section_index")),
                "section_role": item.get("section_role", item.get("audio_section_role")),
                "rhythm_mode": item.get("rhythm_mode", (timeline_plan or {}).get("rhythm_mode")),
                "mood": item.get("mood", item.get("emotion", "balanced")),
                "recommended_content": item.get(
                    "recommended_content", item.get("visual_intent", item.get("content", []))
                ),
                "recommended_shot_scale": item.get(
                    "recommended_shot_scale", item.get("recommended_scale", item.get("shot_scale", "medium"))
                ),
                "recommended_motion": item.get(
                    "recommended_motion", item.get("motion", item.get("visual_motion", "medium"))
                ),
                "is_emphasis": bool(item.get("is_emphasis", item.get("highlight", False))),
                "anchor_event": item.get("anchor_event", item.get("event", {})),
                "transition": _normalize_transition(item.get("transition", item.get("transition_out"))),
            }
        )
    normalized.sort(key=lambda item: (item["start"], item["end"], item["index"]))
    previous_end = 0.0
    for index, item in enumerate(normalized):
        if abs(item["start"] - previous_end) <= 0.08:
            item["start"] = round(previous_end, 6)
            item["duration"] = round(item["end"] - item["start"], 6)
        elif item["start"] > previous_end:
            raise MontageError(f"Timeline slots contain a gap before slot {index}.")
        else:
            raise MontageError(f"Timeline slots overlap before slot {index}.")
        if item["duration"] < 0.24:
            raise MontageError(f"Timeline slot {index} is shorter than the safe 0.24 second minimum.")
        previous_end = item["end"]
    if abs(previous_end - duration) <= 0.08:
        normalized[-1]["end"] = round(duration, 6)
        normalized[-1]["duration"] = round(duration - normalized[-1]["start"], 6)
    elif abs(previous_end - duration) > 0.08:
        raise MontageError(
            f"Timeline slots end at {previous_end:.4f}s but target duration is {duration:.4f}s."
        )
    return normalized


def _slot_energy(slot: dict[str, Any], energy_points: list[tuple[float, float]]) -> float:
    value = slot.get("energy")
    if isinstance(value, dict):
        value = _first_mapping_value(value, ("value", "mean", "level", "normalized"))
    parsed = _as_float(value, -1.0)
    if parsed >= 0:
        return max(0.0, min(1.0, parsed))
    label = str(slot.get("energy_level", "")).lower()
    if label in {"high", "peak", "climax"}:
        return 0.85
    if label in {"low", "quiet", "calm"}:
        return 0.20
    midpoint = (float(slot["start"]) + float(slot["end"])) / 2.0
    return _energy_at(energy_points, midpoint)


def _usable_segments(asset: dict[str, Any]) -> list[dict[str, float]]:
    duration = _as_float(asset.get("duration"), 0.0)
    raw = asset.get("usable_segments")
    if not isinstance(raw, list):
        quality = asset.get("quality", {}) if isinstance(asset.get("quality"), dict) else {}
        raw = quality.get("usable_segments")
    segments: list[dict[str, float]] = []
    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            start = _as_float(item.get("start", item.get("source_start")), -1.0)
            end = _as_float(item.get("end", item.get("source_end")), -1.0)
            if duration > 0:
                start, end = max(0.0, start), min(duration, end)
            if start >= 0 and end - start >= 0.28:
                segments.append(
                    {
                        "start": start,
                        "end": end,
                        "score": max(
                            0.0,
                            min(
                                1.0,
                                _as_float(
                                    item.get(
                                        "score",
                                        item.get("quality_score", item.get("activity_score", 0.70)),
                                    ),
                                    0.70,
                                ),
                            ),
                        ),
                        "preferred_start": _as_float(
                            item.get("preferred_start", item.get("action_start")), -1.0
                        ),
                    }
                )
    if not segments and duration > 0.45:
        # Exclude preparation/title and frozen-tail zones even when the asset
        # analyzer has not emitted explicit usable windows.
        margin = min(0.75, max(0.12, duration * 0.06))
        start, end = margin, duration - margin
        if end - start >= 0.28:
            segments.append({"start": start, "end": end, "score": 0.58, "preferred_start": -1.0})
    return segments


def _interval_overlap(left: tuple[float, float], right: tuple[float, float]) -> float:
    return max(0.0, min(left[1], right[1]) - max(left[0], right[0]))


def _best_source_window(
    asset: dict[str, Any],
    output_duration: float,
    energy: float,
    used_intervals: list[tuple[float, float]],
    policy: dict[str, Any],
    seed: str,
    slot_index: int,
) -> dict[str, float] | None:
    desired_speed = 0.92 + 0.20 * energy
    if asset["motion"] > 0.72 and energy < 0.35:
        desired_speed = 0.88
    elif asset["motion"] < 0.24 and energy > 0.70:
        desired_speed = 1.16
    speeds = []
    for value in (desired_speed, 1.0, 0.92, 1.08, 0.84, 1.18):
        value = max(0.82, min(1.20, value))
        if all(abs(value - existing) > 0.005 for existing in speeds):
            speeds.append(value)
    maximum_overlap = max(0.0, min(0.25, _as_float(policy.get("max_source_interval_overlap"), 0.02)))
    options: list[dict[str, float]] = []
    for segment_index, segment in enumerate(_usable_segments(asset)):
        segment_start, segment_end = segment["start"], segment["end"]
        for speed in speeds:
            needed = output_duration * speed
            if needed > segment_end - segment_start + 1e-6:
                continue
            preferred = segment.get("preferred_start", -1.0)
            span = max(0.0, segment_end - segment_start - needed)
            starts = [
                preferred,
                segment_start + min(0.18, span * 0.20),
                segment_start + span * 0.35,
                segment_start + span * 0.62,
                segment_start + span * 0.82,
            ]
            for candidate_start in starts:
                if candidate_start < segment_start:
                    continue
                candidate_start = min(segment_end - needed, max(segment_start, candidate_start))
                candidate_end = candidate_start + needed
                overlap = sum(
                    _interval_overlap((candidate_start, candidate_end), interval)
                    for interval in used_intervals
                )
                if overlap / max(needed, 1e-6) > maximum_overlap + 1e-6:
                    continue
                edge_clearance = min(candidate_start - segment_start, segment_end - candidate_end)
                stable_jitter = int(
                    hashlib.sha256(
                        f"{seed}|{slot_index}|{asset['canonical_source_key']}|{segment_index}|{candidate_start:.4f}|{speed:.4f}".encode(
                            "utf-8"
                        )
                    ).hexdigest()[:8],
                    16,
                ) / 0xFFFFFFFF
                option_score = (
                    0.58 * segment["score"]
                    + 0.22 * min(1.0, edge_clearance / 0.45)
                    + 0.12 * (1.0 - abs(speed - desired_speed) / 0.38)
                    + 0.08 * stable_jitter
                )
                options.append(
                    {
                        "source_start": candidate_start,
                        "source_end": candidate_end,
                        "speed": speed,
                        "segment_score": segment["score"],
                        "window_score": option_score,
                        "overlap_seconds": overlap,
                    }
                )
    if not options:
        return None
    return max(
        options,
        key=lambda item: (item["window_score"], item["source_start"], item["speed"]),
    )


def _meaningful_label(value: Any) -> str | None:
    text = str(value or "").strip().lower()
    return None if text in {"", "unknown", "unresolved", "general", "none", "n/a"} else text


def adjacent_diversity_issues(previous: dict[str, Any], current: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    if previous.get("canonical_source_key") == current.get("canonical_source_key"):
        issues.append("same_source")
    comparisons = (
        ("subject_label", "same_subject"),
        ("scene_category", "same_scene"),
        ("source_shot_scale", "same_shot_scale"),
        ("composition", "same_composition"),
        ("color_tendency", "same_color_tendency"),
        ("motion_direction", "same_motion_direction"),
    )
    for key, issue in comparisons:
        left, right = _meaningful_label(previous.get(key)), _meaningful_label(current.get(key))
        if left is not None and left == right:
            issues.append(issue)
    if previous.get("is_static_like") and current.get("is_static_like"):
        issues.append("consecutive_static")
    if (
        previous.get("is_aerial")
        and current.get("is_aerial")
        and _meaningful_label(previous.get("motion_direction"))
        == _meaningful_label(current.get("motion_direction"))
    ):
        issues.append("same_direction_aerial")
    return issues


def _asset_hsv(asset: dict[str, Any]) -> tuple[float, float, float] | None:
    hsv = asset.get("mean_hsv") if isinstance(asset.get("mean_hsv"), dict) else {}
    saturation = _as_float(hsv.get("saturation"), -1.0)
    value = _as_float(hsv.get("value"), -1.0)
    if saturation < 0.0 or value <= 0.0:
        return None
    return _as_float(hsv.get("hue_degrees"), 0.0) % 360.0, saturation, value


def _adjacent_cohesion(previous: dict[str, Any] | None, asset: dict[str, Any]) -> tuple[float, float]:
    if previous is None:
        return 0.5, 0.5
    left, right = _asset_hsv(previous), _asset_hsv(asset)
    if left is None or right is None:
        color_match = 0.45
    else:
        lh, ls, lv = left
        rh, rs, rv = right
        hue_distance = min(abs(lh - rh), 360.0 - abs(lh - rh)) / 180.0
        hue_weight = min(0.30, max(0.04, min(ls, rs) * 0.5))
        color_match = max(
            0.0,
            1.0 - hue_weight * hue_distance - 0.38 * abs(ls - rs) - 0.48 * abs(lv - rv),
        )
    left_direction = _meaningful_label(previous.get("motion_direction"))
    right_direction = _meaningful_label(asset.get("motion_direction"))
    if left_direction is None or right_direction is None:
        motion_match = 0.45
    elif left_direction == right_direction:
        motion_match = 1.0
    elif "static" in {left_direction, right_direction}:
        motion_match = 0.20
    else:
        motion_match = 0.55
    return color_match, motion_match


def timeline_diversity_issues(shots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"left_index": index - 1, "right_index": index, "issues": adjacent_diversity_issues(shots[index - 1], shots[index])}
        for index in range(1, len(shots))
        if adjacent_diversity_issues(shots[index - 1], shots[index])
    ]


def _candidate_score(
    asset: dict[str, Any],
    slot: dict[str, Any],
    previous: dict[str, Any] | None,
    desired_scale: str,
    desired_motion: float,
    desired_motion_label: str,
    spec: OutputSpec,
    theme: str,
    seed: str,
    visual_profile: dict[str, Any] | None = None,
) -> tuple[float, dict[str, float]]:
    visual_profile = visual_profile or {}
    asset_text = [
        asset.get("tags"), asset.get("scene_category"), asset.get("subject_label"),
        asset.get("search_query"), asset.get("search_queries"), asset.get("semantic"),
    ]
    slot_text = [slot.get("recommended_content"), slot.get("mood"), slot.get("section_role")]
    semantic = _semantic_overlap(asset_text, slot_text)
    theme_match = _semantic_overlap(asset_text, theme)
    source_motion = _as_float(asset.get("motion"), 0.5)
    section_role = str(slot.get("section_role") or slot.get("audio_section_role") or "").lower()
    # Reserve the genuinely high-motion inventory for drop/climax sections.
    # A local accent in an intro/outro may still request some movement, but it
    # must not consume every strongest clip before the structural peak arrives.
    if section_role in {"drop", "climax"}:
        structural_motion_target = max(desired_motion, 0.82)
    elif section_role in {"intro", "break", "outro"}:
        structural_motion_target = min(desired_motion, 0.52 if slot.get("is_emphasis") else 0.34)
    else:
        structural_motion_target = desired_motion
    motion_match = max(0.0, 1.0 - abs(source_motion - structural_motion_target))
    motion_label_match = _semantic_overlap(asset.get("motion_label"), desired_motion_label)
    scale_match = 1.0 if _normalize_scale(str(asset.get("shot_scale"))) == desired_scale else 0.15
    mood_match = max(
        _semantic_overlap(asset_text, slot.get("mood")),
        0.65 if str(slot.get("mood", "")).lower() in {"balanced", "neutral", ""} else 0.20,
    )
    source_ratio = _as_float(asset.get("width"), 0.0) / max(1.0, _as_float(asset.get("height"), 0.0))
    target_ratio = spec.width / spec.height
    ratio_fit = max(0.0, 1.0 - abs(math.log(max(source_ratio, 1e-4) / target_ratio)) / 1.6) if source_ratio > 0 else 0.35
    resolution = min(
        1.0,
        max(0.0, _as_float(asset.get("width"), 0.0) / spec.width),
        max(0.0, _as_float(asset.get("height"), 0.0) / spec.height),
    )
    quality = _score_value(asset, ("quality_score", "overall_score"), 0.65)
    stability = _score_value(asset, ("stability_score", "stability"), 0.65)
    history = min(1.0, _as_float(asset.get("history_usage_count"), 0.0) / 10.0)
    face_penalty = _as_float(asset.get("face_risk"), 0.0)
    difference_penalty = 0.0
    visual_fit = asset_profile_fit(asset, visual_profile)
    visual_transition = transition_match(previous, asset, visual_profile)
    adjacent_color_match = visual_transition["color"]
    adjacent_motion_match = visual_transition["motion"]
    if previous is not None:
        preview = {
            "canonical_source_key": asset["canonical_source_key"],
            "subject_label": asset.get("subject_label"),
            "scene_category": asset.get("scene_category"),
            "source_shot_scale": _normalize_scale(str(asset.get("shot_scale"))),
            "composition": asset.get("composition"),
            "color_tendency": asset.get("color_tendency"),
            "motion_direction": asset.get("motion_direction"),
            "is_static_like": asset.get("is_static_like"),
            "is_aerial": asset.get("is_aerial"),
        }
        issues = adjacent_diversity_issues(previous, preview)
        difference_penalty = sum(
            1.0 if issue == "same_source" else (0.07 if issue in {"consecutive_static", "same_direction_aerial"} else 0.025)
            for issue in issues
        )
    digest = hashlib.sha256(
        f"{seed}|{slot['index']}|{asset['canonical_source_key']}".encode("utf-8")
    ).digest()
    jitter = int.from_bytes(digest[:4], "big") / 0xFFFFFFFF
    components = {
        "semantic": semantic,
        "theme": theme_match,
        "mood": mood_match,
        "motion": motion_match,
        "structural_motion_target": structural_motion_target,
        "motion_label": motion_label_match,
        "shot_scale": scale_match,
        "stability": stability,
        "quality": quality,
        "ratio_fit": ratio_fit,
        "resolution": resolution,
        "history_penalty": history,
        "face_penalty": face_penalty,
        "adjacent_similarity_penalty": difference_penalty,
        "adjacent_color_match": adjacent_color_match,
        "adjacent_motion_match": adjacent_motion_match,
        "visual_profile_fit": visual_fit["total"],
        "world_fit": visual_fit["world"],
        "color_profile_fit": visual_fit["color"],
        "time_weather_fit": visual_fit["time_weather"],
        "camera_language_fit": visual_fit["camera_language"],
        "aesthetic_quality": visual_fit["aesthetic"],
        "cinematic_quality": visual_fit["cinematic"],
        "visual_transition_match": visual_transition["total"],
        "transition_scale_match": visual_transition["scale"],
        "transition_world_match": visual_transition["world"],
        "transition_texture_match": visual_transition["texture"],
        "transition_composition_match": visual_transition["composition"],
        "seed_jitter": jitter,
    }
    score = (
        0.18 * semantic
        + 0.10 * theme_match
        + 0.08 * mood_match
        + 0.14 * motion_match
        + 0.05 * motion_label_match
        + 0.11 * scale_match
        + 0.08 * stability
        + 0.09 * quality
        + 0.06 * ratio_fit
        + 0.04 * resolution
        + 0.07 * max(0.0, min(1.0, _as_float(asset.get("score"), 0.5)))
        + 0.16 * visual_fit["total"]
        + 0.15 * visual_transition["total"]
        + 0.08 * visual_fit["aesthetic"]
        + 0.04 * visual_fit["cinematic"]
        + 0.055 * jitter
        - 0.10 * history
        - 0.16 * face_penalty
        - difference_penalty
    )
    return score, {key: round(value, 5) for key, value in components.items()}


def _shot_from_choice(
    slot: dict[str, Any],
    asset: dict[str, Any],
    window: dict[str, float],
    score: float,
    components: dict[str, float],
    spec: OutputSpec,
    desired_scale: str,
    desired_motion: float,
    desired_motion_label: str,
    candidate_log: list[dict[str, Any]],
    grammar_influence: dict[str, Any],
) -> dict[str, Any]:
    energy = _as_float(slot.get("_energy"), 0.5)
    anchor = slot.get("anchor_event", {}) if isinstance(slot.get("anchor_event"), dict) else {}
    return {
        "index": int(slot["index"]),
        "slot_index": int(slot["index"]),
        "output_start": round(_as_float(slot["start"]), 4),
        "output_end": round(_as_float(slot["end"]), 4),
        "output_duration": round(_as_float(slot["duration"]), 4),
        "local_path": asset["local_path"],
        "asset_id": asset["asset_id"],
        "canonical_source_key": asset["canonical_source_key"],
        "pixabay_id": asset.get("pixabay_id", asset.get("id")),
        "page_url": asset.get("page_url", asset.get("pageURL")),
        "search_query": asset.get("search_query", asset.get("query")),
        "search_queries": asset.get("search_queries", []),
        "tags": asset.get("tags", ""),
        "source_start": round(window["source_start"], 4),
        "source_end": round(window["source_end"], 4),
        "source_segment_score": round(window["segment_score"], 4),
        "source_window_score": round(window["window_score"], 4),
        "speed": round(window["speed"], 4),
        "energy": round(energy, 4),
        "energy_level": slot.get("energy_level"),
        "rhythm_mode": slot.get("rhythm_mode"),
        "mood": slot.get("mood"),
        "cut_intensity": slot.get("cut_intensity"),
        "is_emphasis": bool(slot.get("is_emphasis")),
        "anchor_event": anchor,
        "cut_reason": anchor.get("type", slot.get("cut_reason", "timeline_slot")),
        "recommended_content": slot.get("recommended_content"),
        "desired_shot_role": desired_scale,
        "desired_motion": round(desired_motion, 4),
        "desired_motion_label": desired_motion_label,
        "source_motion": round(_as_float(asset.get("motion"), 0.5), 4),
        "source_motion_label": asset.get("motion_label", "unknown"),
        "motion_direction": asset.get("motion_direction", "unknown"),
        "source_shot_scale": _normalize_scale(str(asset.get("shot_scale"))),
        "scene_category": asset.get("scene_category"),
        "subject_label": asset.get("subject_label"),
        "composition": asset.get("composition"),
        "color_tendency": asset.get("color_tendency"),
        "mean_hsv": asset.get("mean_hsv", {}),
        "motion_signals": asset.get("motion_signals", {}),
        "quality": asset.get("quality", {}),
        "visual_analysis": asset.get("visual_analysis", {}),
        "visual_features": asset_visual_features(asset),
        "is_static_like": bool(asset.get("is_static_like")),
        "is_aerial": bool(asset.get("is_aerial")),
        "face_content_risk": round(_as_float(asset.get("face_risk"), 0.0), 4),
        "crop_plan": plan_subject_crop(asset, spec),
        "transform": {
            "crop": plan_subject_crop(asset, spec),
            "scale": {"x": 1.0, "y": 1.0},
            "position": {"x": 0.0, "y": 0.0},
            "rotation_degrees": 0.0,
            "opacity": 1.0,
        },
        "transition_in": "hard_cut",
        "transition_out": _normalize_transition(slot.get("transition")),
        "audio_section_index": slot.get("section_index"),
        "audio_section_role": slot.get("section_role"),
        "selection_score": round(score, 6),
        "selection_score_components": components,
        "candidate_scores": candidate_log,
        "grammar_influence": grammar_influence,
        "timeline_start": round(_as_float(slot["start"]), 4),
        "timeline_end": round(_as_float(slot["end"]), 4),
        "duration": round(_as_float(slot["duration"]), 4),
        "source_path": asset["local_path"],
    }


def build_timeline(
    audio_profile: dict[str, Any],
    media_result: Any,
    duration: float,
    style_profile: dict[str, Any] | None = None,
    seed: str = "bgm-montage",
    editing_grammar: dict[str, Any] | None = None,
    content_policy: dict[str, Any] | None = None,
    ratio: str = "16:9",
    timeline_plan: dict[str, Any] | None = None,
    slots: list[dict[str, Any]] | None = None,
    theme: str = "",
) -> dict[str, Any]:
    """Assign real assets and safe source intervals to music-derived slots.

    ``timeline_plan``/``slots`` are optional v1.2-compatible inputs.  When absent, the
    v1.1 signal-driven boundary planner remains available for compatibility.
    The same seed is reproducible; changing it changes near-tie ordering so a
    bounded QA rework attempt can produce a genuinely different edit.
    """

    if duration <= 0:
        raise MontageError("Target duration must be positive.")
    style_profile = style_profile or {}
    editing_grammar = editing_grammar or {}
    embedded_visual = style_profile.get("visual_style_profile")
    visual_profile = (
        dict(embedded_visual)
        if isinstance(embedded_visual, dict) and str(embedded_visual.get("schema_version")) == "1.3"
        else build_visual_style_profile(theme, style_profile, audio_profile, "")
    )
    style_profile = {**style_profile, "visual_style_profile": visual_profile}
    spec = parse_ratio(ratio)
    assets = _extract_assets(media_result)
    musical_times = extract_musical_times(audio_profile, duration)
    energy_points = _energy_points(audio_profile, duration)
    audio_sections = _audio_sections(audio_profile, duration)
    reference_shot = _style_average_shot(style_profile)
    event_weights = _grammar_event_weights(editing_grammar)
    event_offsets = _grammar_event_offsets(editing_grammar)
    beat_gaps = [
        right - left
        for left, right in zip(musical_times["beats"], musical_times["beats"][1:])
        if 0.18 <= right - left <= 2.0
    ]
    beat_seconds = sorted(beat_gaps)[len(beat_gaps) // 2] if beat_gaps else 0.5
    grammar_policy = editing_grammar.get("montage_policy", {}) if isinstance(editing_grammar, dict) else {}
    ending = (
        grammar_policy.get("ending", {})
        if isinstance(grammar_policy, dict) and isinstance(grammar_policy.get("ending"), dict)
        else _mapping_at(editing_grammar, ("ending", "ending_structure"))
    )
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

    slot_records = _coerce_timeline_slots(timeline_plan, slots, duration)
    policy_input = dict(content_policy or {})
    policy = {**default_content_policy(duration), **policy_input}
    policy["visual_style_profile_digest"] = visual_profile.get("profile_digest")
    policy["visual_style_profile_schema"] = visual_profile.get("schema_version")
    if slot_records is None:
        scale_matrix = _mapping_at(editing_grammar, ("scale_transition_matrix", "shot_scale_transition_matrix"))
        motion_matrix = _mapping_at(editing_grammar, ("motion_transition_matrix", "camera_motion_transition_matrix"))
        transition_distribution = _mapping_at(
            editing_grammar, ("transition_distribution", "supported_transition_distribution")
        )
        fade_share = max(
            (
                _as_float(value, 0.0)
                for key, value in transition_distribution.items()
                if "fade" in str(key).lower() or "dissolve" in str(key).lower()
            ),
            default=0.0,
        )
        slot_records = []
        current = 0.0
        previous_scale = "wide"
        previous_motion = "static_like"
        max_slot_seconds = max(0.36, float(policy["max_asset_screen_share"]) * duration)
        while current < duration - 0.03:
            energy = _energy_at(energy_points, current + 0.05)
            learned_length = _grammar_duration(editing_grammar, energy, reference_shot, beat_seconds)
            musical_target = 0.72 + (1.0 - energy) * 2.65
            ideal_length = 0.48 * musical_target + 0.52 * max(0.45, min(4.5, learned_length))
            if duration - current <= ideal_length * 1.8:
                ideal_length *= learned_final_multiplier
            end, cut_reason = _choose_boundary(
                current,
                current + ideal_length,
                duration,
                musical_times,
                energy,
                event_weights,
                event_offsets,
            )
            end = min(duration, max(current + 0.36, end))
            if end - current > max_slot_seconds:
                end = min(duration, current + max_slot_seconds)
                cut_reason = "content_share_guard"
            if duration - end < 0.28:
                end = duration
            active_section = next(
                (section for section in audio_sections if section["start"] <= current < section["end"]),
                None,
            )
            desired_scale = _normalize_scale(
                _transition_choice(
                    scale_matrix,
                    previous_scale,
                    ["wide", "medium", "detail"],
                    len(slot_records),
                )
            )
            desired_motion_label = _transition_choice(
                motion_matrix,
                previous_motion,
                ["static_like", "pan_or_tilt_like", "push_or_pull_like"],
                len(slot_records),
            )
            at_boundary = any(abs(end - _as_float(section.get("end"), -99.0)) <= 0.10 for section in audio_sections)
            slot_records.append(
                {
                    "index": len(slot_records),
                    "start": round(current, 6),
                    "end": round(end, 6),
                    "duration": round(end - current, 6),
                    "section_index": active_section.get("index") if active_section else None,
                    "section_role": active_section.get("role") if active_section else None,
                    "rhythm_mode": "beat_cut" if energy >= 0.58 else "phrase_flow",
                    "mood": active_section.get("estimated_mood", "balanced") if active_section else "balanced",
                    "energy": energy,
                    "recommended_content": [active_section.get("role")] if active_section else [],
                    "recommended_shot_scale": desired_scale,
                    "recommended_motion": {"intensity": 0.25 + 0.70 * energy, "label": desired_motion_label},
                    "is_emphasis": cut_reason in {"accents", "sections"},
                    "anchor_event": {"type": cut_reason, "time": round(end, 4), "strength": event_weights.get(cut_reason, 0.0)},
                    "transition": "fade_through_black" if at_boundary and fade_share >= 0.08 else "hard_cut",
                    "_learned_length": learned_length,
                    "_previous_scale": previous_scale,
                    "_previous_motion": previous_motion,
                }
            )
            previous_scale, previous_motion = desired_scale, desired_motion_label
            current = end
    if not policy_input.get("min_unique_assets"):
        policy["min_unique_assets"] = min(int(policy["min_unique_assets"]), len(slot_records))
    sufficiency = evaluate_material_sufficiency(assets, duration, policy)
    policy = sufficiency["policy"]
    if not sufficiency["passed"]:
        raise InsufficientMaterialError(
            "Material sufficiency gate failed: " + "; ".join(sufficiency["failures"])
        )

    # Limit soft transitions even if an upstream plan over-requests them.
    max_soft = max(0, math.floor(len(slot_records) * float(policy.get("max_soft_transition_share", 0.28))))
    soft_seen = 0
    for slot in slot_records[:-1]:
        if _normalize_transition(slot.get("transition")) != "hard_cut":
            soft_seen += 1
            if soft_seen > max_soft and not slot.get("is_emphasis"):
                slot["transition"] = "hard_cut"
                slot["transition_policy_adjustment"] = "soft_transition_budget"
    if slot_records:
        slot_records[-1]["transition"] = "hard_cut"

    usage_count: dict[str, int] = {asset["canonical_source_key"]: 0 for asset in assets}
    screen_time: dict[str, float] = {asset["canonical_source_key"]: 0.0 for asset in assets}
    output_occurrences: dict[str, list[tuple[int, float, float]]] = {asset["canonical_source_key"]: [] for asset in assets}
    source_intervals: dict[str, list[tuple[float, float]]] = {asset["canonical_source_key"]: [] for asset in assets}
    prominent_face_time = 0.0
    max_share_seconds = float(policy["max_asset_screen_share"]) * duration
    max_face_seconds = float(policy["max_prominent_face_screen_share"]) * duration
    prominent_threshold = float(policy["prominent_face_threshold"])
    timeline: list[dict[str, Any]] = []

    for slot in slot_records:
        output_duration = _as_float(slot.get("duration"), 0.0)
        energy = _slot_energy(slot, energy_points)
        slot["_energy"] = energy
        desired_scale = _normalize_scale(str(slot.get("recommended_shot_scale", "medium")))
        desired_motion, desired_motion_label = _slot_motion_target(slot.get("recommended_motion"), energy)
        previous = timeline[-1] if timeline else None
        options: list[dict[str, Any]] = []
        for asset in assets:
            identity = asset["canonical_source_key"]
            if usage_count[identity] >= int(policy["max_reuse_per_asset"]):
                continue
            if screen_time[identity] + output_duration > max_share_seconds + 0.02:
                continue
            occurrences = output_occurrences[identity]
            if occurrences:
                last_index, _, last_end = occurrences[-1]
                if int(slot["index"]) - last_index < int(policy.get("min_repeat_gap_shots", 0)):
                    continue
                if _as_float(slot["start"]) - last_end < _as_float(policy.get("min_repeat_gap_seconds"), 0.0):
                    continue
            if (
                _as_float(asset.get("face_risk"), 0.0) >= prominent_threshold
                and prominent_face_time + output_duration > max_face_seconds + 0.02
            ):
                continue
            window = _best_source_window(
                asset,
                output_duration,
                energy,
                source_intervals[identity],
                policy,
                seed,
                int(slot["index"]),
            )
            if window is None:
                continue
            score, components = _candidate_score(
                asset,
                slot,
                previous,
                desired_scale,
                desired_motion,
                desired_motion_label,
                spec,
                theme or str((timeline_plan or {}).get("theme", "")),
                seed,
                visual_profile,
            )
            score += 0.10 * window["window_score"]
            options.append({"asset": asset, "window": window, "score": score, "components": components})
        used_sources = {key for key, count in usage_count.items() if count > 0}
        if len(used_sources) < int(policy["min_unique_assets"]):
            unused = [option for option in options if usage_count[option["asset"]["canonical_source_key"]] == 0]
            if unused:
                options = unused
        used_scenes = {str(shot.get("scene_category") or "general") for shot in timeline}
        if len(used_scenes) < int(policy["min_scene_categories"]):
            novel_scene = [
                option
                for option in options
                if str(option["asset"].get("scene_category") or "general") not in used_scenes
                and option["components"].get("world_fit", 0.0) >= 0.50
                and option["components"].get("visual_transition_match", 0.0) >= 0.40
            ]
            if novel_scene:
                options = novel_scene
        if not options:
            raise InsufficientMaterialError(
                f"Material constraints became infeasible at slot {slot['index']} "
                "(canonical reuse/share/gap, face budget, or non-overlapping usable segment exhausted)."
            )
        options.sort(
            key=lambda option: (option["score"], option["asset"]["canonical_source_key"]),
            reverse=True,
        )
        selected = options[0]
        candidate_log = [
            {
                "canonical_source_key": option["asset"]["canonical_source_key"],
                "asset_id": option["asset"]["asset_id"],
                "score": round(option["score"], 6),
                "score_components": option["components"],
                "source_start": round(option["window"]["source_start"], 4),
                "source_end": round(option["window"]["source_end"], 4),
            }
            for option in options[:8]
        ]
        grammar_influence = {
            "learned_shot_target_seconds": round(_as_float(slot.get("_learned_length"), output_duration), 4),
            "event_weight": round(event_weights.get(str((slot.get("anchor_event") or {}).get("type")), 0.0), 4),
            "learned_boundary_offset_seconds": round(event_offsets.get(str((slot.get("anchor_event") or {}).get("type")), 0.0), 4),
            "scale_transition": f"{slot.get('_previous_scale', 'slot')}->{desired_scale}",
            "motion_transition": f"{slot.get('_previous_motion', 'slot')}->{desired_motion_label}",
            "section_boundary_transition": _normalize_transition(slot.get("transition")),
        }
        shot = _shot_from_choice(
            slot,
            selected["asset"],
            selected["window"],
            selected["score"],
            selected["components"],
            spec,
            desired_scale,
            desired_motion,
            desired_motion_label,
            candidate_log,
            grammar_influence,
        )
        shot["_alternatives"] = options[1:8]
        timeline.append(shot)
        identity = selected["asset"]["canonical_source_key"]
        usage_count[identity] += 1
        screen_time[identity] += output_duration
        output_occurrences[identity].append((int(slot["index"]), float(slot["start"]), float(slot["end"])))
        source_intervals[identity].append((selected["window"]["source_start"], selected["window"]["source_end"]))
        if _as_float(selected["asset"].get("face_risk"), 0.0) >= prominent_threshold:
            prominent_face_time += output_duration

    before_repair = timeline_diversity_issues(timeline)
    repairs: list[dict[str, Any]] = []
    severe_limit = int(policy.get("max_adjacent_similarity_dimensions", 3))
    # Greedy scoring already penalizes similarity.  This second pass replaces a
    # remaining obviously repetitive shot with an unused ranked alternative.
    repair_passes = 2
    for _ in range(repair_passes):
        changed = False
        used_now = {shot["canonical_source_key"] for shot in timeline}
        for index in range(1, len(timeline)):
            current_issues = adjacent_diversity_issues(timeline[index - 1], timeline[index])
            if "same_source" not in current_issues and len(current_issues) <= severe_limit:
                continue
            next_shot = timeline[index + 1] if index + 1 < len(timeline) else None
            old_local_count = len(current_issues) + (
                len(adjacent_diversity_issues(timeline[index], next_shot)) if next_shot else 0
            )
            for option in timeline[index].get("_alternatives", []):
                asset = option["asset"]
                if asset["canonical_source_key"] in used_now:
                    continue
                slot = slot_records[index]
                replacement = _shot_from_choice(
                    slot,
                    asset,
                    option["window"],
                    option["score"],
                    option["components"],
                    spec,
                    timeline[index]["desired_shot_role"],
                    timeline[index]["desired_motion"],
                    timeline[index]["desired_motion_label"],
                    timeline[index]["candidate_scores"],
                    timeline[index]["grammar_influence"],
                )
                replacement["transition_in"] = timeline[index].get("transition_in", "hard_cut")
                replacement["transition_out"] = timeline[index].get("transition_out", "hard_cut")
                new_local_count = len(adjacent_diversity_issues(timeline[index - 1], replacement)) + (
                    len(adjacent_diversity_issues(replacement, next_shot)) if next_shot else 0
                )
                if new_local_count < old_local_count:
                    previous_key = timeline[index]["canonical_source_key"]
                    timeline[index] = replacement
                    used_now.discard(previous_key)
                    used_now.add(replacement["canonical_source_key"])
                    repairs.append(
                        {
                            "slot_index": index,
                            "replaced_source": previous_key,
                            "replacement_source": replacement["canonical_source_key"],
                            "issues_before": current_issues,
                            "local_issue_count_before": old_local_count,
                            "local_issue_count_after": new_local_count,
                        }
                    )
                    changed = True
                    break
        if not changed:
            break

    visual_repairs: list[dict[str, Any]] = []
    used_now = {shot["canonical_source_key"] for shot in timeline}
    for index in range(1, len(timeline)):
        previous = timeline[index - 1]
        current = timeline[index]
        following = timeline[index + 1] if index + 1 < len(timeline) else None
        current_left = transition_match(previous, current, visual_profile)["total"]
        current_right = transition_match(current, following, visual_profile)["total"] if following else 0.55
        current_pair_score = (current_left + current_right) / 2.0
        if current_pair_score >= 0.48:
            continue
        best_replacement: dict[str, Any] | None = None
        best_pair_score = current_pair_score
        for option in current.get("_alternatives", []):
            asset = option["asset"]
            if asset["canonical_source_key"] in used_now:
                continue
            slot = slot_records[index]
            replacement = _shot_from_choice(
                slot,
                asset,
                option["window"],
                option["score"],
                option["components"],
                spec,
                current["desired_shot_role"],
                current["desired_motion"],
                current["desired_motion_label"],
                current["candidate_scores"],
                current["grammar_influence"],
            )
            replacement["transition_in"] = current.get("transition_in", "hard_cut")
            replacement["transition_out"] = current.get("transition_out", "hard_cut")
            left_score = transition_match(previous, replacement, visual_profile)["total"]
            right_score = transition_match(replacement, following, visual_profile)["total"] if following else 0.55
            pair_score = (left_score + right_score) / 2.0
            if pair_score > best_pair_score + 0.045:
                best_pair_score = pair_score
                best_replacement = replacement
        if best_replacement is not None:
            old_key = current["canonical_source_key"]
            timeline[index] = best_replacement
            used_now.discard(old_key)
            used_now.add(best_replacement["canonical_source_key"])
            visual_repairs.append(
                {
                    "slot_index": index,
                    "replaced_source": old_key,
                    "replacement_source": best_replacement["canonical_source_key"],
                    "pair_score_before": round(current_pair_score, 5),
                    "pair_score_after": round(best_pair_score, 5),
                    "reason": "visual continuity repair",
                }
            )

    for index, shot in enumerate(timeline):
        shot.pop("_alternatives", None)
        shot["index"] = index
        shot["transition_in"] = (
            timeline[index - 1].get("transition_out", "hard_cut") if index else "hard_cut"
        )
    if timeline:
        final_fade = _as_float(ending.get("fade_out_seconds", ending.get("final_fade_seconds")), 0.0)
        timeline[-1]["ending_structure"] = {
            "learned_final_shot_multiplier": round(learned_final_multiplier, 4),
            "visual_fade_out_seconds": round(
                max(0.0, min(timeline[-1]["output_duration"] * 0.8, final_fade)), 4
            ),
            # Kept for schema compatibility; v1.2 never fabricates missing
            # duration with a cloned final frame.
            "hold_last_frame": bool(ending.get("hold_last_frame", False)),
            "hold_last_frame_rendered": False,
            "requested_hold_last_frame": bool(ending.get("hold_last_frame", False)),
        }

    usage_count = {}
    screen_time = {}
    prominent_face_time = 0.0
    used_scenes: set[str] = set()
    for shot in timeline:
        identity = str(shot["canonical_source_key"])
        usage_count[identity] = usage_count.get(identity, 0) + 1
        screen_time[identity] = screen_time.get(identity, 0.0) + _as_float(shot["output_duration"])
        used_scenes.add(str(shot.get("scene_category") or "general"))
        if _as_float(shot.get("face_content_risk"), 0.0) >= prominent_threshold:
            prominent_face_time += _as_float(shot["output_duration"])
    used_assets = set(usage_count)
    actual_max_share = max(screen_time.values(), default=0.0) / duration
    actual_face_share = prominent_face_time / duration
    after_repair = timeline_diversity_issues(timeline)
    sequence_consistency = evaluate_sequence_consistency(timeline, visual_profile)
    final_failures: list[str] = []
    if len(used_assets) < int(policy["min_unique_assets"]):
        final_failures.append(f"used unique canonical sources {len(used_assets)} < {policy['min_unique_assets']}")
    if len(used_scenes) < int(policy["min_scene_categories"]):
        final_failures.append(f"used scene categories {len(used_scenes)} < {policy['min_scene_categories']}")
    if max(usage_count.values(), default=0) > int(policy["max_reuse_per_asset"]):
        final_failures.append("canonical source reuse exceeded policy")
    if actual_max_share > float(policy["max_asset_screen_share"]) + 0.005:
        final_failures.append("single-source screen share exceeded policy")
    if actual_face_share > float(policy["max_prominent_face_screen_share"]) + 0.005:
        final_failures.append("prominent-face screen share exceeded policy")
    if any("same_source" in item["issues"] for item in after_repair):
        final_failures.append("adjacent shots reuse the same canonical source")
    if any(len(item["issues"]) > severe_limit + 2 for item in after_repair):
        final_failures.append("adjacent visual diversity remained severely insufficient after repair")
    if not sequence_consistency["passed"]:
        final_failures.extend(
            f"visual sequence consistency: {failure}"
            for failure in sequence_consistency.get("failures", [])
        )
    if final_failures:
        raise InsufficientMaterialError("Timeline sufficiency gate failed: " + "; ".join(final_failures))

    grammar_digest = (
        hashlib.sha256(
            json.dumps(editing_grammar, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()[:16]
        if editing_grammar
        else None
    )
    slot_digest = hashlib.sha256(
        json.dumps(
            [{key: value for key, value in slot.items() if not str(key).startswith("_")} for slot in slot_records],
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()[:16]
    return {
        "schema_version": "1.3",
        "timeline_schema_version": 3,
        "compatible_readers": ["1.2", "1.3"],
        "legacy_schema_version": 2,
        "artifact_type": "edit_decisions",
        "duration_seconds": round(duration, 4),
        "seed": seed,
        "reference_average_shot_seconds": round(reference_shot, 4),
        "editing_grammar_digest": grammar_digest,
        "editing_grammar_applied": bool(editing_grammar),
        "timeline_plan_digest": slot_digest,
        "timeline_plan_applied": timeline_plan is not None or slots is not None,
        "content_policy": policy,
        "visual_style_profile": visual_profile,
        "visual_sequence_consistency": sequence_consistency,
        "sufficiency": {
            **sufficiency,
            "used_unique_assets": len(used_assets),
            "used_unique_canonical_sources": len(used_assets),
            "used_scene_categories": sorted(used_scenes),
            "max_asset_screen_share_actual": round(actual_max_share, 4),
            "prominent_face_screen_share_actual": round(actual_face_share, 4),
        },
        "musical_alignment_points": musical_times,
        "slots": [
            {key: value for key, value in slot.items() if not str(key).startswith("_")}
            for slot in slot_records
        ],
        "shots": timeline,
        "asset_usage_counts": usage_count,
        "diversity": {
            "issues_before_repair": before_repair,
            "repairs": repairs,
            "visual_continuity_repairs": visual_repairs,
            "issues_after_repair": after_repair,
            "passed": not any("same_source" in item["issues"] for item in after_repair),
        },
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
    """Render H.264/AAC atomically with real hard/fade/dissolve transitions."""
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
    visual_profile = (
        plan.get("visual_style_profile")
        if isinstance(plan.get("visual_style_profile"), dict)
        else style_profile.get("visual_style_profile", {})
    )
    duration = _as_float(plan.get("duration_seconds"), 0.0)
    if duration <= 0:
        duration = max(_as_float(shot.get("output_end"), 0.0) for shot in shots)
    expected_start = 0.0
    for index, shot in enumerate(shots):
        source = Path(str(shot.get("local_path") or "")).expanduser().resolve()
        if not source.is_file():
            raise MontageError(f"Edit plan source is missing at shot {index}: {source}")
        start = _as_float(shot.get("output_start"), expected_start)
        output_duration = _as_float(shot.get("output_duration"), 0.0)
        end = _as_float(shot.get("output_end"), start + output_duration)
        if abs(start - expected_start) > 0.08 or output_duration < 0.05 or end <= start:
            raise MontageError(f"Edit plan has a gap/overlap or invalid duration at shot {index}.")
        expected_start = end
        speed = _as_float(shot.get("speed"), 1.0)
        source_duration = _as_float(shot.get("source_end"), 0.0) - _as_float(shot.get("source_start"), 0.0)
        if not 0.75 <= speed <= 1.35:
            raise MontageError(f"Unsafe speed {speed:.4f} at shot {index}; source replacement is required.")
        if source_duration + 0.025 < output_duration * speed:
            raise MontageError(
                f"Shot {index} source interval is too short; refusing to freeze-pad the missing duration."
            )
    if abs(expected_start - duration) > max(0.08, duration * 0.005):
        raise MontageError(
            f"Edit plan covers {expected_start:.4f}s but declares {duration:.4f}s."
        )

    transition_types: list[str] = []
    transition_durations: list[float] = []
    for index, shot in enumerate(shots[:-1]):
        kind = _normalize_transition(shot.get("transition_out"))
        transition_types.append(kind)
        if kind == "hard_cut":
            transition_durations.append(0.0)
            continue
        left = _as_float(shot.get("output_duration"), 1.0)
        right = _as_float(shots[index + 1].get("output_duration"), 1.0)
        requested = _as_float(shot.get("transition_duration_seconds"), 0.0)
        default = 0.22 if kind == "dissolve" else 0.16
        transition_durations.append(
            max(0.06, min(requested or default, left * 0.22, right * 0.22, 0.30))
        )
    # Quantize the complete edit on one global frame grid before rendering.
    # Independently rounding every segment and then applying a global `-t`
    # can accumulate enough PTS error to collapse the final planned shot to a
    # single frame.  Cumulative boundaries guarantee that the shot frame
    # counts sum to the exact output frame count.
    total_frame_count = max(1, int(math.ceil(duration * fps - 1e-9)))
    base_frame_counts: list[int] = []
    previous_frame = 0
    for index, shot in enumerate(shots):
        if index == len(shots) - 1:
            end_frame = total_frame_count
        else:
            end_frame = int(round(_as_float(shot.get("output_end"), 0.0) * fps))
            end_frame = max(previous_frame + 1, min(total_frame_count - 1, end_frame))
        base_frame_counts.append(end_frame - previous_frame)
        previous_frame = end_frame
    if previous_frame != total_frame_count or any(value <= 0 for value in base_frame_counts):
        raise MontageError("Edit plan cannot be represented on the requested output frame grid.")

    transition_frame_counts: list[int] = []
    for value in transition_durations:
        transition_frame_counts.append(0 if value <= 0 else max(2, int(round(value * fps))))
    transition_durations = [value / fps for value in transition_frame_counts]
    render_frame_counts = list(base_frame_counts)
    for index, transition_frames in enumerate(transition_frame_counts):
        if transition_frames <= 0:
            continue
        left_extension = transition_frames // 2
        right_extension = transition_frames - left_extension
        render_frame_counts[index] += left_extension
        render_frame_counts[index + 1] += right_extension
    render_durations = [value / fps for value in render_frame_counts]

    part_suffix = output.suffix if output.suffix else ".mp4"
    part = output.with_name(f".{output.stem}.{uuid.uuid4().hex}.part{part_suffix}")
    command: list[str] = [ffmpeg, "-hide_banner", "-y", "-i", str(bgm)]
    for shot in shots:
        source_start = _as_float(shot.get("source_start"), 0.0)
        source_end = _as_float(shot.get("source_end"), source_start + 1.0)
        source_duration = max(0.05, source_end - source_start)
        # Demuxer time bases and keyframe seeking can otherwise return one or
        # two frames fewer than the analyzed interval.  Read a small tail
        # reserve, while the exact planned frame count remains enforced by the
        # filter graph below; this avoids both cumulative tail truncation and
        # freeze-frame padding.
        speed = _as_float(shot.get("speed"), 1.0)
        decode_reserve = max(0.25, (4.0 * speed) / fps)
        command.extend(
            [
                "-ss",
                f"{source_start:.5f}",
                "-t",
                f"{source_duration + decode_reserve:.5f}",
                "-i",
                str(shot["local_path"]),
            ]
        )

    filters: list[str] = []
    labels: list[str] = []
    for index, shot in enumerate(shots, start=1):
        speed = _as_float(shot.get("speed"), 1.0)
        output_duration = max(0.05, _as_float(shot.get("output_duration"), 1.0))
        render_duration = render_durations[index - 1]
        stretch = render_duration / output_duration
        label = f"v{index}"
        pre_label = f"pre{index}"
        filters.append(
            f"[{index}:v]setpts=((PTS-STARTPTS)/{speed:.6f})*{stretch:.8f},"
            f"trim=duration={render_duration:.6f},setpts=PTS-STARTPTS[{pre_label}]"
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
                f"[{pre_label}]scale={spec.width}:{spec.height}:force_original_aspect_ratio=decrease:flags=lanczos,"
                f"pad={spec.width}:{spec.height}:(ow-iw)/2:(oh-ih)/2:color=black[{composed_label}]"
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
        ending_structure = shot.get("ending_structure", {}) if isinstance(shot.get("ending_structure"), dict) else {}
        final_fade = min(render_duration * 0.8, _as_float(ending_structure.get("visual_fade_out_seconds"), 0.0))
        if final_fade > 0.02:
            fade_filters.append(
                f"fade=t=out:st={max(0.0, render_duration - final_fade):.5f}:d={final_fade:.5f}"
            )
        exact_frame_count = render_frame_counts[index - 1]
        shot_brightness, shot_saturation, shot_contrast = brightness, saturation, contrast
        grade_filters: list[str] = []
        if isinstance(visual_profile, dict) and visual_profile:
            grade = build_light_grade(shot, visual_profile, brightness, saturation, contrast)
            shot["color_grade"] = grade
            shot_brightness = _as_float(grade.get("brightness"), brightness)
            shot_saturation = _as_float(grade.get("saturation"), saturation)
            shot_contrast = _as_float(grade.get("contrast"), contrast)
            balance = grade["colorbalance"]
            if _as_float(grade.get("strength"), 0.0) > 0.01:
                grade_filters.append(
                    "colorbalance="
                    f"rs={balance['rs']:.4f}:gs={balance['gs']:.4f}:bs={balance['bs']:.4f}:"
                    f"rm={balance['rm']:.4f}:gm={balance['gm']:.4f}:bm={balance['bm']:.4f}:"
                    f"rh={balance['rh']:.4f}:gh={balance['gh']:.4f}:bh={balance['bh']:.4f}:pl=1"
                )
        suffix = ",".join(
            [
                    f"fps={fps}",
                    f"trim=end_frame={exact_frame_count}",
                    "settb=AVTB",
                    f"setpts=N/({fps}*TB)",
                    "setsar=1",
                f"eq=brightness={shot_brightness:.5f}:saturation={shot_saturation:.5f}:contrast={shot_contrast:.5f}",
                *grade_filters,
                *fade_filters,
                "format=yuv420p",
            ]
        )
        filters.append(f"[{composed_label}]{suffix}[{label}]")
        labels.append(f"[{label}]")
    if len(labels) == 1:
        filters.append(f"{labels[0]}null[vout]")
    elif not any(transition_durations):
        filters.append(f"{''.join(labels)}concat=n={len(labels)}:v=1:a=0[vout]")
    else:
        chain_label = "v1"
        chain_frame_count = render_frame_counts[0]
        for boundary_index in range(len(shots) - 1):
            next_label = f"v{boundary_index + 2}"
            output_label = "vout" if boundary_index == len(shots) - 2 else f"chain{boundary_index + 1}"
            transition_duration = transition_durations[boundary_index]
            if transition_duration <= 0.0:
                filters.append(
                    f"[{chain_label}][{next_label}]concat=n=2:v=1:a=0[{output_label}]"
                )
                chain_frame_count += render_frame_counts[boundary_index + 1]
            else:
                kind = transition_types[boundary_index]
                ffmpeg_transition = (
                    "dissolve" if kind == "dissolve" else "fadeblack" if kind == "fade_through_black" else "fade"
                )
                transition_frames = transition_frame_counts[boundary_index]
                offset = max(0.0, (chain_frame_count - transition_frames) / fps)
                filters.append(
                    f"[{chain_label}][{next_label}]xfade=transition={ffmpeg_transition}:"
                    f"duration={transition_duration:.6f}:offset={offset:.6f}[{output_label}]"
                )
                chain_frame_count += render_frame_counts[boundary_index + 1] - transition_frames
            chain_label = output_label
        if chain_frame_count != total_frame_count:
            raise MontageError(
                "Transition frame accounting does not match the declared output duration."
            )
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
            "-frames:v",
            str(total_frame_count),
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-t",
            f"{duration:.6f}",
            "-movflags",
            "+faststart",
            str(part),
        ]
    )
    try:
        process = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if process.returncode != 0:
            tail = "\n".join(process.stderr.splitlines()[-30:])
            raise MontageError(f"FFmpeg render failed:\n{tail}")
        if not part.is_file() or part.stat().st_size < 1024:
            raise MontageError("FFmpeg returned success but the atomic part file is missing or empty.")
        if output.exists() and not overwrite:
            raise MontageError(f"Output appeared during rendering and was not overwritten: {output}")
        os.replace(part, output)
    except Exception:
        part.unlink(missing_ok=True)
        raise
    return output


def write_plan(plan: dict[str, Any], path: str | Path) -> Path:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    return output
