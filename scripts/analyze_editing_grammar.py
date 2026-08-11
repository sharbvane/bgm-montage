#!/usr/bin/env python3
"""Learn an audio-linked editing grammar from reference videos.

This stage consumes the cut times already measured by ``analyze_references``
and analyzes each reference video's *own* audio with :func:`analyze_bgm`.  It
then measures how cuts relate to accents, strong/weak beats, phrases, sections,
pauses and local energy.  Reference files are opened read-only; all component
cache and output files are written below the supplied reference cache root.

Public API:
    analyze_editing_grammar(reference_dir, reference_profile, cache_dir,
                            output_path=None)

``reference_profile`` may be the mapping returned by ``analyze_references`` or
the path to its ``style_profile.json``.  ``cache_dir`` is the exact reference
stage cache directory (normally ``<project>/.bgm-montage-cache/references``).
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import statistics
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

from analyze_bgm import ANALYZER_VERSION as AUDIO_ANALYZER_VERSION
from analyze_bgm import analyze_bgm


SCHEMA_VERSION = "1.1"
ANALYZER_VERSION = "1.1.0"
CACHE_SCHEMA_VERSION = 1
FINGERPRINT_BLOCK_BYTES = 256 * 1024

EVENT_TYPES = (
    "section_boundary",
    "phrase_boundary",
    "strong_accent",
    "downbeat",
    "pause_edge",
    "accent",
    "weak_beat",
)

EVENT_PRIORITY = {
    "section_boundary": 1.00,
    "phrase_boundary": 0.95,
    "strong_accent": 0.92,
    "downbeat": 0.88,
    "pause_edge": 0.76,
    "accent": 0.68,
    "weak_beat": 0.60,
}


class EditingGrammarError(RuntimeError):
    """Raised when the grammar stage cannot safely analyze its inputs."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return number if math.isfinite(number) else default


def _round(value: Any, digits: int = 4) -> float:
    return round(_float(value), digits)


def _clamp(value: Any, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, _float(value)))


def _mean(values: Iterable[Any], default: float = 0.0) -> float:
    numbers = [_float(value) for value in values]
    return statistics.fmean(numbers) if numbers else default


def _quantile(values: Iterable[Any], q: float, default: float = 0.0) -> float:
    numbers = sorted(_float(value) for value in values)
    if not numbers:
        return default
    if len(numbers) == 1:
        return numbers[0]
    position = _clamp(q) * (len(numbers) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return numbers[lower]
    fraction = position - lower
    return numbers[lower] * (1.0 - fraction) + numbers[upper] * fraction


def _stats(values: Iterable[Any]) -> dict[str, Any]:
    numbers = [_float(value) for value in values]
    if not numbers:
        return {
            "count": 0,
            "minimum": None,
            "p25": None,
            "median": None,
            "p75": None,
            "maximum": None,
            "mean": None,
        }
    return {
        "count": len(numbers),
        "minimum": _round(min(numbers), 4),
        "p25": _round(_quantile(numbers, 0.25), 4),
        "median": _round(_quantile(numbers, 0.50), 4),
        "p75": _round(_quantile(numbers, 0.75), 4),
        "maximum": _round(max(numbers), 4),
        "mean": _round(_mean(numbers), 4),
    }


def _distribution(labels: Iterable[str]) -> dict[str, float]:
    counts = Counter(str(label) for label in labels if label)
    total = sum(counts.values())
    if not total:
        return {}
    return {
        label: _round(count / total, 5)
        for label, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    }


def _atomic_json_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            json.dump(payload, temporary, ensure_ascii=False, indent=2, allow_nan=False)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, path)
    finally:
        if temporary_name and os.path.exists(temporary_name):
            try:
                os.unlink(temporary_name)
            except OSError:
                pass


def _read_json(path: Path, default: Any) -> Any:
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            return json.load(handle)
    except (OSError, UnicodeError, ValueError, TypeError):
        return default


def _load_reference_profile(value: str | os.PathLike[str] | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return copy.deepcopy(dict(value))
    path = Path(value).expanduser().resolve()
    payload = _read_json(path, None)
    if not isinstance(payload, Mapping):
        raise EditingGrammarError(f"Reference style profile is missing or invalid JSON: {path}")
    return dict(payload)


def _fingerprint(path: Path) -> dict[str, Any]:
    """Return a content-aware fingerprint without modifying the source."""

    stat = path.stat()
    size = int(stat.st_size)
    block = FINGERPRINT_BLOCK_BYTES
    digest = hashlib.sha256()
    if size <= block * 3:
        mode = "full_sha256"
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        sampled_bytes = size
    else:
        mode = "first_middle_last_sha256"
        offsets = [0, max(0, size // 2 - block // 2), max(0, size - block)]
        with path.open("rb") as handle:
            for offset in offsets:
                handle.seek(offset)
                digest.update(handle.read(block))
        sampled_bytes = min(size, block * len(offsets))
    return {
        "size": size,
        "mtime_ns": int(stat.st_mtime_ns),
        "content_sha256": digest.hexdigest(),
        "hash_mode": mode,
        "sample_bytes": sampled_bytes,
    }


def _cut_digest(duration: float, cut_times: Sequence[float], has_audio: bool) -> str:
    payload = {
        "duration": round(duration, 5),
        "cut_times": [round(value, 5) for value in cut_times],
        "has_audio": bool(has_audio),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _video_entries(profile: Mapping[str, Any]) -> list[dict[str, Any]]:
    videos = profile.get("videos")
    if isinstance(videos, list):
        return [dict(item) for item in videos if isinstance(item, Mapping)]
    for key in ("reference_profile", "analysis", "profile"):
        nested = profile.get(key)
        if isinstance(nested, Mapping) and isinstance(nested.get("videos"), list):
            return [dict(item) for item in nested["videos"] if isinstance(item, Mapping)]
    return []


def _safe_source_path(reference_root: Path, relative_path: Any) -> Path:
    text = str(relative_path or "").strip().replace("\\", "/")
    if not text:
        raise EditingGrammarError("Reference profile contains a video without relative_path")
    pure = PurePosixPath(text)
    if pure.is_absolute() or ".." in pure.parts:
        raise EditingGrammarError(f"Unsafe reference relative path: {text}")
    source = (reference_root / Path(*pure.parts)).resolve()
    if not source.is_relative_to(reference_root):
        raise EditingGrammarError(f"Reference path escapes the read-only root: {text}")
    return source


def _cut_times(video: Mapping[str, Any], duration: float) -> list[float]:
    rhythm = video.get("editing_rhythm")
    raw = rhythm.get("cut_times_seconds") if isinstance(rhythm, Mapping) else []
    values: list[float] = []
    if isinstance(raw, (list, tuple)):
        for value in raw:
            number = _float(value, -1.0)
            if 0.12 <= number <= duration - 0.12:
                values.append(number)
    ordered: list[float] = []
    for value in sorted(set(round(item, 4) for item in values)):
        if not ordered or value - ordered[-1] >= 0.12:
            ordered.append(value)
    return ordered


def _time_from_item(value: Any, keys: Sequence[str] = ("time", "start")) -> float | None:
    if isinstance(value, (int, float)):
        result = _float(value, -1.0)
        return result if result >= 0 else None
    if isinstance(value, Mapping):
        for key in keys:
            if value.get(key) is not None:
                result = _float(value.get(key), -1.0)
                return result if result >= 0 else None
    return None


def _beat_period(audio_profile: Mapping[str, Any]) -> float:
    global_profile = audio_profile.get("global")
    if isinstance(global_profile, Mapping):
        period = _float(global_profile.get("beat_period_seconds"), 0.0)
        if 0.18 <= period <= 2.0:
            return period
        bpm = _float(global_profile.get("tempo_bpm_estimate", global_profile.get("bpm")), 0.0)
        if 30.0 <= bpm <= 260.0:
            return 60.0 / bpm
    beat_times = [
        time_value
        for item in audio_profile.get("beats", []) or []
        if (time_value := _time_from_item(item)) is not None
    ]
    intervals = [right - left for left, right in zip(beat_times, beat_times[1:]) if right > left]
    return _quantile(intervals, 0.5, 0.5)


def _append_event(
    events: dict[str, list[dict[str, Any]]],
    event_type: str,
    time_value: Any,
    duration: float,
    strength: Any = 1.0,
) -> None:
    time_number = _float(time_value, -1.0)
    if event_type not in events or not 0.0 <= time_number <= duration:
        return
    rounded = round(time_number, 4)
    if any(abs(_float(item.get("time")) - rounded) < 1e-4 for item in events[event_type]):
        return
    events[event_type].append({"time": rounded, "strength": _round(_clamp(strength), 4)})


def _audio_events(audio_profile: Mapping[str, Any], duration: float) -> tuple[dict[str, list[dict[str, Any]]], float]:
    events: dict[str, list[dict[str, Any]]] = {event_type: [] for event_type in EVENT_TYPES}
    period = _beat_period(audio_profile)

    for beat in audio_profile.get("beats", []) or []:
        if not isinstance(beat, Mapping):
            continue
        event_type = "downbeat" if bool(beat.get("downbeat_estimate")) else "weak_beat"
        _append_event(events, event_type, beat.get("time"), duration, beat.get("strength", 0.5))

    for accent in audio_profile.get("accents", []) or []:
        if not isinstance(accent, Mapping):
            continue
        strength = _float(accent.get("strength"), 0.5)
        event_type = "strong_accent" if accent.get("level") == "strong" or strength >= 0.78 else "accent"
        _append_event(events, event_type, accent.get("time"), duration, strength)

    for phrase in audio_profile.get("phrases", []) or []:
        if not isinstance(phrase, Mapping):
            continue
        for key in ("start", "end"):
            value = _float(phrase.get(key), -1.0)
            if 0.05 < value < duration - 0.05:
                _append_event(events, "phrase_boundary", value, duration)

    for boundary in audio_profile.get("section_boundaries", []) or []:
        value = _time_from_item(boundary)
        if value is not None and 0.05 < value < duration - 0.05:
            strength = boundary.get("confidence", boundary.get("novelty", 1.0)) if isinstance(boundary, Mapping) else 1.0
            _append_event(events, "section_boundary", value, duration, strength)
    for section in audio_profile.get("sections", []) or []:
        if not isinstance(section, Mapping):
            continue
        value = _float(section.get("start"), -1.0)
        if 0.05 < value < duration - 0.05:
            _append_event(events, "section_boundary", value, duration)

    pauses = audio_profile.get("pause_intervals")
    if not isinstance(pauses, list):
        pause_root = audio_profile.get("pauses")
        pauses = pause_root.get("intervals", []) if isinstance(pause_root, Mapping) else []
    for pause in pauses or []:
        if not isinstance(pause, Mapping):
            continue
        for key in ("start", "end"):
            value = _float(pause.get(key), -1.0)
            if 0.05 < value < duration - 0.05:
                _append_event(events, "pause_edge", value, duration)

    for values in events.values():
        values.sort(key=lambda item: item["time"])
    return events, period


def _event_tolerances(beat_period: float) -> dict[str, float]:
    return {
        "section_boundary": min(0.50, max(0.25, beat_period * 0.70)),
        "phrase_boundary": min(0.42, max(0.22, beat_period * 0.55)),
        "strong_accent": min(0.20, max(0.10, beat_period * 0.30)),
        "downbeat": min(0.18, max(0.09, beat_period * 0.25)),
        "pause_edge": min(0.28, max(0.14, beat_period * 0.38)),
        "accent": min(0.17, max(0.08, beat_period * 0.24)),
        "weak_beat": min(0.14, max(0.07, beat_period * 0.20)),
    }


def _energy_points(audio_profile: Mapping[str, Any], duration: float) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    raw = audio_profile.get("energy_curve")
    if isinstance(raw, list):
        for index, item in enumerate(raw):
            if isinstance(item, Mapping):
                time_value = _float(item.get("time"), -1.0)
                level = _float(item.get("level", item.get("energy")), 0.5)
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                time_value, level = _float(item[0], -1.0), _float(item[1], 0.5)
            elif isinstance(item, (int, float)):
                time_value = duration * index / max(1, len(raw) - 1)
                level = _float(item, 0.5)
            else:
                continue
            if 0.0 <= time_value <= duration:
                points.append((time_value, _clamp(level)))
    if not points:
        return [(0.0, 0.5), (duration, 0.5)]
    return sorted(points)


def _energy_at(points: Sequence[tuple[float, float]], time_value: float) -> float:
    if time_value <= points[0][0]:
        return points[0][1]
    for (left_t, left_v), (right_t, right_v) in zip(points, points[1:]):
        if left_t <= time_value <= right_t:
            alpha = (time_value - left_t) / max(1e-9, right_t - left_t)
            return left_v + alpha * (right_v - left_v)
    return points[-1][1]


def _energy_band(value: float) -> str:
    if value < 0.34:
        return "low"
    if value >= 0.67:
        return "high"
    return "medium"


def _section_role(audio_profile: Mapping[str, Any], time_value: float) -> str:
    for section in audio_profile.get("sections", []) or []:
        if not isinstance(section, Mapping):
            continue
        start = _float(section.get("start"), -1.0)
        end = _float(section.get("end"), -1.0)
        if 0.0 <= start <= time_value < end:
            return str(section.get("role") or "unassigned")
    return "unassigned"


def _align_cut(
    cut_time: float,
    events: Mapping[str, Sequence[Mapping[str, Any]]],
    tolerances: Mapping[str, float],
) -> tuple[list[dict[str, Any]], str]:
    alignments: list[dict[str, Any]] = []
    for event_type in EVENT_TYPES:
        candidates = events.get(event_type, [])
        if not candidates:
            continue
        nearest = min(candidates, key=lambda item: abs(_float(item.get("time")) - cut_time))
        event_time = _float(nearest.get("time"))
        delta = cut_time - event_time
        tolerance = _float(tolerances.get(event_type), 0.15)
        if abs(delta) <= tolerance:
            normalized = abs(delta) / max(tolerance, 1e-9)
            importance = EVENT_PRIORITY[event_type]
            strength = _clamp(nearest.get("strength", 1.0))
            match_score = importance * (1.0 - normalized) * (0.75 + 0.25 * strength)
            alignments.append(
                {
                    "type": event_type,
                    "event_time": _round(event_time, 4),
                    "delta_seconds": _round(delta, 4),
                    "tolerance_seconds": _round(tolerance, 4),
                    "event_strength": _round(strength, 4),
                    "match_score": _round(match_score, 5),
                }
            )
    alignments.sort(key=lambda item: (-_float(item["match_score"]), abs(_float(item["delta_seconds"])), item["type"]))
    return alignments, str(alignments[0]["type"]) if alignments else "off_grid"


def _analyze_video_audio_grammar(
    relative_path: str,
    duration: float,
    cut_times: Sequence[float],
    audio_profile: Mapping[str, Any],
) -> dict[str, Any]:
    events, beat_period = _audio_events(audio_profile, duration)
    tolerances = _event_tolerances(beat_period)
    energy_points = _energy_points(audio_profile, duration)
    cut_records: list[dict[str, Any]] = []
    for cut_time in cut_times:
        alignments, primary = _align_cut(cut_time, events, tolerances)
        before = _energy_at(energy_points, max(0.0, cut_time - 0.25))
        at_cut = _energy_at(energy_points, cut_time)
        after = _energy_at(energy_points, min(duration, cut_time + 0.25))
        delta = after - before
        trend = "rising" if delta > 0.08 else "falling" if delta < -0.08 else "steady"
        cut_records.append(
            {
                "time": _round(cut_time, 4),
                "primary_alignment": primary,
                "alignments": alignments,
                "energy": _round(at_cut, 4),
                "energy_band": _energy_band(at_cut),
                "energy_delta_around_cut": _round(delta, 4),
                "energy_trend": trend,
                "section_role": _section_role(audio_profile, cut_time),
            }
        )

    boundaries = [0.0, *cut_times, duration]
    shot_records: list[dict[str, Any]] = []
    for start, end in zip(boundaries, boundaries[1:]):
        shot_duration = end - start
        if shot_duration < 0.12:
            continue
        midpoint = start + shot_duration / 2.0
        energy = _energy_at(energy_points, midpoint)
        shot_records.append(
            {
                "start": _round(start, 4),
                "end": _round(end, 4),
                "duration_seconds": _round(shot_duration, 4),
                "duration_beats": _round(shot_duration / max(0.18, beat_period), 4),
                "energy": _round(energy, 4),
                "energy_band": _energy_band(energy),
                "section_role": _section_role(audio_profile, midpoint),
            }
        )

    last_shot = shot_records[-1] if shot_records else None
    typical_duration = _quantile((shot["duration_seconds"] for shot in shot_records), 0.5, duration)
    end_alignments, end_primary = _align_cut(duration, events, tolerances)
    global_profile = audio_profile.get("global") if isinstance(audio_profile.get("global"), Mapping) else {}
    return {
        "relative_path": relative_path,
        "status": "ok",
        "duration_seconds": _round(duration, 4),
        "audio": {
            "tempo_bpm_estimate": _round(global_profile.get("tempo_bpm_estimate", global_profile.get("bpm")), 3),
            "tempo_confidence": _round(global_profile.get("tempo_confidence"), 4),
            "meter_estimate": global_profile.get("meter_estimate"),
            "beat_period_seconds": _round(beat_period, 5),
        },
        "event_tolerances_seconds": {key: _round(value, 4) for key, value in tolerances.items()},
        "event_counts": {key: len(events[key]) for key in EVENT_TYPES},
        "cut_count": len(cut_records),
        "cuts": cut_records,
        "shots": shot_records,
        "ending": {
            "last_shot_duration_seconds": _round(last_shot["duration_seconds"], 4) if last_shot else None,
            "last_shot_to_median_ratio": _round(last_shot["duration_seconds"] / max(0.12, typical_duration), 4) if last_shot else None,
            "end_primary_alignment": end_primary,
            "end_alignments": end_alignments,
        },
    }


def _aggregate(per_video: Sequence[Mapping[str, Any]], profile_video_count: int) -> dict[str, Any]:
    successful = [item for item in per_video if item.get("status") == "ok"]
    all_cuts = [cut for video in successful for cut in video.get("cuts", []) if isinstance(cut, Mapping)]
    all_shots = [shot for video in successful for shot in video.get("shots", []) if isinstance(shot, Mapping)]

    multi_counts: Counter[str] = Counter()
    offsets: defaultdict[str, list[float]] = defaultdict(list)
    for cut in all_cuts:
        seen: set[str] = set()
        for alignment in cut.get("alignments", []) or []:
            if not isinstance(alignment, Mapping):
                continue
            event_type = str(alignment.get("type") or "")
            if event_type and event_type not in seen:
                multi_counts[event_type] += 1
                seen.add(event_type)
            if event_type:
                offsets[event_type].append(_float(alignment.get("delta_seconds")))
    total_cuts = len(all_cuts)
    multi_rates = {
        event_type: _round(multi_counts[event_type] / total_cuts, 5) if total_cuts else 0.0
        for event_type in EVENT_TYPES
    }
    primary_distribution = _distribution(str(cut.get("primary_alignment") or "off_grid") for cut in all_cuts)

    duration_by_energy: dict[str, Any] = {}
    for band in ("low", "medium", "high"):
        shots = [shot for shot in all_shots if shot.get("energy_band") == band]
        duration_by_energy[band] = {
            "seconds": _stats(shot.get("duration_seconds") for shot in shots),
            "beats": _stats(shot.get("duration_beats") for shot in shots),
        }

    raw_weights: dict[str, float] = {}
    for event_type in EVENT_TYPES:
        primary_share = _float(primary_distribution.get(event_type))
        raw_weights[event_type] = 0.12 + multi_rates[event_type] + primary_share
    maximum_weight = max(raw_weights.values(), default=1.0)
    boundary_weights = {
        event_type: _round(value / max(maximum_weight, 1e-9), 4)
        for event_type, value in raw_weights.items()
    }

    ending_ratios = [
        _float(video.get("ending", {}).get("last_shot_to_median_ratio"))
        for video in successful
        if isinstance(video.get("ending"), Mapping)
        and video.get("ending", {}).get("last_shot_to_median_ratio") is not None
    ]
    end_distribution = _distribution(
        str(video.get("ending", {}).get("end_primary_alignment") or "off_grid")
        for video in successful
        if isinstance(video.get("ending"), Mapping)
    )
    tempo_confidences = [
        _float(video.get("audio", {}).get("tempo_confidence"))
        for video in successful
        if isinstance(video.get("audio"), Mapping)
    ]
    audio_coverage = len(successful) / max(1, profile_video_count)
    reliability_score = _clamp(
        0.42 * audio_coverage
        + 0.30 * _mean(tempo_confidences)
        + 0.28 * min(1.0, total_cuts / 40.0)
    )
    reliability_label = "high" if reliability_score >= 0.72 else "medium" if reliability_score >= 0.45 else "low"

    dominant_end_event = next(iter(end_distribution), "off_grid")
    policy_durations = {
        band: {
            "median_seconds": value["seconds"]["median"],
            "median_beats": value["beats"]["median"],
            "p25_seconds": value["seconds"]["p25"],
            "p75_seconds": value["seconds"]["p75"],
        }
        for band, value in duration_by_energy.items()
    }
    # ``mid`` is a compatibility alias for planners that predate the canonical
    # ``medium`` spelling.  Both carry identical data and remain machine-readable.
    policy_durations["mid"] = dict(policy_durations["medium"])
    return {
        "cut_alignment": {
            "total_reference_cuts": total_cuts,
            "primary_distribution": primary_distribution,
            "multi_alignment_rates": multi_rates,
            "event_offset_seconds": {event_type: _stats(values) for event_type, values in offsets.items()},
            "energy_band_at_cut_distribution": _distribution(str(cut.get("energy_band")) for cut in all_cuts),
            "energy_trend_at_cut_distribution": _distribution(str(cut.get("energy_trend")) for cut in all_cuts),
            "section_role_at_cut_distribution": _distribution(str(cut.get("section_role")) for cut in all_cuts),
        },
        "shot_duration_model": {
            "global_seconds": _stats(shot.get("duration_seconds") for shot in all_shots),
            "global_beats": _stats(shot.get("duration_beats") for shot in all_shots),
            "by_energy_band": duration_by_energy,
            "transfer_rule": "Blend reference seconds with duration-in-beats scaled to the new BGM tempo; respect confidence and hard min/max limits.",
        },
        "ending_structure": {
            "last_shot_to_median_ratio": _stats(ending_ratios),
            "end_event_distribution": end_distribution,
            "preferred_end_event": dominant_end_event,
            "note": "This stage learns timing only; it does not infer or claim visual outro effects.",
        },
        "montage_policy": {
            "boundary_event_weights": boundary_weights,
            "event_weights": boundary_weights,
            "boundary_offsets_seconds": {
                event_type: _round(_quantile(values, 0.5), 4)
                for event_type, values in offsets.items()
                if values
            },
            "shot_duration_by_energy": policy_durations,
            "ending": {
                "last_shot_duration_multiplier": _round(_quantile(ending_ratios, 0.5, 1.0), 4),
                "preferred_end_event": dominant_end_event,
            },
            "application_contract": [
                "Use boundary_event_weights and learned offsets when scoring new-BGM cut candidates.",
                "Use energy-conditional seconds/beats to allocate shot duration, blended by reliability.",
                "Use the ending multiplier and preferred event only when the corresponding event exists in the new BGM.",
                "Record which grammar fields affected every planned shot; do not claim application from file generation alone.",
            ],
        },
        "reliability": {
            "label": reliability_label,
            "score": _round(reliability_score, 4),
            "audio_video_coverage": _round(audio_coverage, 4),
            "mean_tempo_confidence": _round(_mean(tempo_confidences), 4),
            "reference_cut_count": total_cuts,
        },
    }


def _visual_transition_grammar(videos: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Learn scale/motion/scene adjacency from semantic shot records."""

    scale_counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
    motion_counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
    scene_counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
    scene_same = 0
    transitions = 0

    def normalized_rows(rows: Mapping[str, Counter[str]]) -> dict[str, dict[str, float]]:
        output: dict[str, dict[str, float]] = {}
        for source, counts in rows.items():
            total = sum(counts.values())
            if total:
                output[source] = {
                    target: _round(count / total, 5)
                    for target, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
                }
        return output

    for video in videos:
        shots = video.get("shots", []) if isinstance(video, Mapping) else []
        if not isinstance(shots, list):
            continue
        for left, right in zip(shots, shots[1:]):
            if not isinstance(left, Mapping) or not isinstance(right, Mapping):
                continue
            left_scale = str(left.get("shot_scale") or "unresolved")
            right_scale = str(right.get("shot_scale") or "unresolved")
            left_motion = str(left.get("camera_motion") or "unresolved")
            right_motion = str(right.get("camera_motion") or "unresolved")
            left_scene = str(left.get("scene_category") or left.get("scene") or "unresolved")
            right_scene = str(right.get("scene_category") or right.get("scene") or "unresolved")
            scale_counts[left_scale][right_scale] += 1
            motion_counts[left_motion][right_motion] += 1
            scene_counts[left_scene][right_scene] += 1
            transitions += 1
            scene_same += int(left_scene == right_scene)
    return {
        "observed_transition_count": transitions,
        "scale_transition_matrix": normalized_rows(scale_counts),
        "motion_transition_matrix": normalized_rows(motion_counts),
        "scene_transition_matrix": normalized_rows(scene_counts),
        "scene_continuity_rate": _round(scene_same / transitions, 5) if transitions else None,
        "supported_transition_distribution": {"hard_cut": 1.0} if transitions else {},
        "transition_scope": "Only sampled hard-cut adjacency is learned; visual effect classification is not implemented.",
    }


def analyze_editing_grammar(
    reference_dir: str | os.PathLike[str],
    reference_profile: str | os.PathLike[str] | Mapping[str, Any],
    cache_dir: str | os.PathLike[str],
    output_path: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Analyze and cache reference audio/cut relationships.

    Unchanged component results are reused when the source fingerprint, visual
    cut digest, grammar version and BGM-analyzer version all match.  The
    function never writes into ``reference_dir``.
    """

    reference_root = Path(reference_dir).expanduser().resolve(strict=True)
    if not reference_root.is_dir():
        raise NotADirectoryError(reference_root)
    cache_root = Path(cache_dir).expanduser().resolve(strict=False)
    destination = (
        Path(output_path).expanduser().resolve(strict=False)
        if output_path is not None
        else cache_root / "editing_grammar.json"
    )
    if cache_root == reference_root or cache_root.is_relative_to(reference_root):
        raise EditingGrammarError("cache_dir must be outside the read-only reference directory")
    if destination == reference_root or destination.is_relative_to(reference_root):
        raise EditingGrammarError("output_path must be outside the read-only reference directory")

    profile = _load_reference_profile(reference_profile)
    videos = _video_entries(profile)
    if not videos:
        raise EditingGrammarError("Reference profile contains no per-video summaries; run analyze_references first")

    cache_root.mkdir(parents=True, exist_ok=True)
    audio_cache = cache_root / "reference_audio"
    cache_path = cache_root / "editing_grammar_cache.json"
    old_cache = _read_json(cache_path, {})
    cache_compatible = (
        isinstance(old_cache, Mapping)
        and old_cache.get("cache_schema_version") == CACHE_SCHEMA_VERSION
        and old_cache.get("analyzer_version") == ANALYZER_VERSION
        and old_cache.get("audio_analyzer_version") == AUDIO_ANALYZER_VERSION
        and old_cache.get("reference_directory") == str(reference_root)
    )
    old_entries = old_cache.get("entries", {}) if cache_compatible else {}
    if not isinstance(old_entries, Mapping):
        old_entries = {}

    report: dict[str, Any] = {
        "profile_videos": len(videos),
        "analyzed": 0,
        "reused": 0,
        "no_audio": 0,
        "missing": 0,
        "failed": 0,
        "removed_from_cache": 0,
        "errors": [],
    }
    entries: dict[str, Any] = {}
    per_video: list[dict[str, Any]] = []

    for video in videos:
        relative_path = str(video.get("relative_path") or "")
        try:
            source = _safe_source_path(reference_root, relative_path)
        except EditingGrammarError as exc:
            report["failed"] += 1
            report["errors"].append({"relative_path": relative_path, "stage": "path", "error": str(exc)})
            continue
        if not source.is_file():
            report["missing"] += 1
            report["errors"].append({"relative_path": relative_path, "stage": "source", "error": "reference file is missing"})
            continue

        metadata = video.get("metadata") if isinstance(video.get("metadata"), Mapping) else {}
        duration = _float(metadata.get("duration_seconds"), 0.0)
        cuts = _cut_times(video, duration)
        has_audio = bool(metadata.get("has_audio"))
        try:
            fingerprint = _fingerprint(source)
        except OSError as exc:
            report["failed"] += 1
            report["errors"].append({"relative_path": relative_path, "stage": "fingerprint", "error": f"{type(exc).__name__}: {exc}"})
            continue
        digest = _cut_digest(duration, cuts, has_audio)
        old_entry = old_entries.get(relative_path)
        if (
            isinstance(old_entry, Mapping)
            and old_entry.get("fingerprint") == fingerprint
            and old_entry.get("cut_digest") == digest
            and isinstance(old_entry.get("result"), Mapping)
        ):
            result = copy.deepcopy(dict(old_entry["result"]))
            entries[relative_path] = copy.deepcopy(dict(old_entry))
            per_video.append(result)
            report["reused"] += 1
            if result.get("status") == "no_audio":
                report["no_audio"] += 1
            continue

        report["analyzed"] += 1
        if not has_audio:
            result = {
                "relative_path": relative_path,
                "status": "no_audio",
                "duration_seconds": _round(duration, 4),
                "cut_count": len(cuts),
                "reason": "ffprobe metadata reports no audio stream",
            }
            report["no_audio"] += 1
        else:
            try:
                audio_profile = analyze_bgm(
                    source,
                    audio_cache,
                    output_path=None,
                    target_duration=duration if duration > 0 else None,
                )
                analyzed_duration = _float(
                    audio_profile.get("input", {}).get("analysis_window", {}).get("duration"),
                    duration,
                )
                effective_duration = min(value for value in (duration, analyzed_duration) if value > 0)
                effective_cuts = [cut for cut in cuts if 0.12 <= cut <= effective_duration - 0.12]
                result = _analyze_video_audio_grammar(
                    relative_path,
                    effective_duration,
                    effective_cuts,
                    audio_profile,
                )
                result["audio_cache_hit"] = bool(audio_profile.get("cache", {}).get("hit"))
            except Exception as exc:
                report["failed"] += 1
                message = f"{type(exc).__name__}: {exc}"
                report["errors"].append({"relative_path": relative_path, "stage": "audio_analysis", "error": message})
                result = {
                    "relative_path": relative_path,
                    "status": "audio_analysis_failed",
                    "duration_seconds": _round(duration, 4),
                    "cut_count": len(cuts),
                    "error": message,
                }

        entry = {
            "fingerprint": fingerprint,
            "cut_digest": digest,
            "analyzed_at": _utc_now(),
            "result": result,
        }
        entries[relative_path] = entry
        per_video.append(result)

    report["removed_from_cache"] = len(set(old_entries) - set(entries))
    cache_payload = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "analyzer_version": ANALYZER_VERSION,
        "audio_analyzer_version": AUDIO_ANALYZER_VERSION,
        "reference_directory": str(reference_root),
        "updated_at": _utc_now(),
        "entries": entries,
    }
    _atomic_json_write(cache_path, cache_payload)

    learned = _aggregate(per_video, len(videos))
    visual_grammar = _visual_transition_grammar(videos)
    learned["visual_transition_grammar"] = visual_grammar
    montage_policy = learned.get("montage_policy")
    if isinstance(montage_policy, dict):
        montage_policy["scale_transition_matrix"] = visual_grammar["scale_transition_matrix"]
        montage_policy["motion_transition_matrix"] = visual_grammar["motion_transition_matrix"]
        montage_policy["scene_transition_matrix"] = visual_grammar["scene_transition_matrix"]
        montage_policy["transition_distribution"] = visual_grammar["supported_transition_distribution"]
    successful_count = sum(item.get("status") == "ok" for item in per_video)
    no_audio_count = sum(item.get("status") == "no_audio" for item in per_video)
    status = "ok" if successful_count == len(videos) else "visual_only" if successful_count == 0 and no_audio_count else "partial"
    grammar: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "status": status,
        "analyzer": {
            "name": "bgm-montage reference editing-grammar analyzer",
            "version": ANALYZER_VERSION,
            "audio_analyzer_version": AUDIO_ANALYZER_VERSION,
            "basis": "reference visual cut times aligned to each reference video's decoded audio waveform",
            "disclosure": "Beat, accent, phrase, section and alignment fields are signal-derived estimates, not ground-truth editorial annotations.",
        },
        "reference_directory": str(reference_root),
        "source_policy": "read_only",
        "cache": {
            "path": str(cache_path),
            "audio_cache_directory": str(audio_cache),
            "invalidation": "source fingerprint + visual cut digest + grammar/analyze_bgm versions",
        },
        "run_report": report,
        "corpus": {
            "profile_video_count": len(videos),
            "audio_analyzed_video_count": successful_count,
            "no_audio_video_count": no_audio_count,
            "failed_video_count": sum(item.get("status") == "audio_analysis_failed" for item in per_video),
        },
        **learned,
        "per_video": per_video,
        "implemented_scope": [
            "hard-cut timing relationships to measured audio events",
            "shot-duration distributions in seconds and estimated beats",
            "cut relationships to energy band, energy change and estimated section role",
            "timing-only ending relationships",
            "sampled shot-scale, camera-motion and semantic scene adjacency matrices",
            "hard-cut transition prevalence only",
        ],
        "limitations": [
            "Visual cut times inherit the sampling precision and false-positive/false-negative limits of analyze_references.",
            "Downbeats, weak beats, phrases and sections are estimates from analyze_bgm and should be weighted by reliability.",
            "This module does not detect or claim crossfades, wipes, masked transitions, match cuts, speed ramps, typography, motion graphics or other visual effects.",
            "Generating editing_grammar.json alone does not prove that a montage used it; the timeline stage must record applied fields and be tested counterfactually.",
        ],
    }
    _atomic_json_write(destination, grammar)
    return grammar


def _default_paths() -> tuple[Path, Path, Path]:
    try:
        from runtime_paths import RuntimePaths

        paths = RuntimePaths.build()
        return paths.project_root / "参考视频", paths.reference_cache / "style_profile.json", paths.reference_cache
    except Exception:
        skill_root = Path(__file__).resolve().parent.parent
        project_root = Path(os.environ.get("BGM_MONTAGE_PROJECT_ROOT", skill_root.parents[2])).expanduser().resolve()
        cache = project_root / ".bgm-montage-cache" / "references"
        return project_root / "参考视频", cache / "style_profile.json", cache


def build_parser() -> argparse.ArgumentParser:
    default_reference, default_profile, default_cache = _default_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", default=str(default_reference), help="Read-only reference-video directory")
    parser.add_argument("--style-profile", default=str(default_profile), help="analyze_references style_profile.json")
    parser.add_argument("--cache-dir", default=str(default_cache), help="Exact reference-stage cache directory")
    parser.add_argument("--output", help="editing_grammar.json path (default: CACHE_DIR/editing_grammar.json)")
    parser.add_argument("--summary-only", action="store_true", help="Print only status, counters and artifact path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        grammar = analyze_editing_grammar(
            reference_dir=args.reference_dir,
            reference_profile=args.style_profile,
            cache_dir=args.cache_dir,
            output_path=args.output,
        )
    except Exception as exc:
        print(f"analyze_editing_grammar: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    output = Path(args.output).expanduser().resolve() if args.output else Path(args.cache_dir).expanduser().resolve() / "editing_grammar.json"
    summary = {
        "status": grammar["status"],
        "editing_grammar": str(output),
        "corpus": grammar["corpus"],
        "run_report": grammar["run_report"],
        "reliability": grammar["reliability"],
    }
    print(json.dumps(summary if args.summary_only else grammar, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
