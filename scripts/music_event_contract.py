"""One normalized music-event contract shared by planning and QA."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping


SCHEMA_VERSION = "music-event-contract.1"

_TYPE_ALIASES = {
    "beats": "beat",
    "downbeats": "downbeat",
    "onsets": "onset",
    "accents": "accent",
    "hard_stops": "hard_stop",
    "drops": "drop",
    "surges": "surge",
    "climaxes": "climax",
    "phrases": "phrase_boundary",
    "phrase_boundaries": "phrase_boundary",
    "sections": "section_boundary",
    "section_boundaries": "section_boundary",
}
_EVENT_GROUPS = {
    "beat": "beats", "downbeat": "downbeats", "onset": "onsets", "accent": "accents",
    "hard_stop": "hard_stops", "drop": "drops", "surge": "surges", "climax": "climaxes",
    "phrase_boundary": "phrases", "section_boundary": "sections",
}
_EVENT_NAMES = tuple(_TYPE_ALIASES)
_MODE_ALLOWED_TYPES = {
    # These are the planner's complete ordered candidate sets.  The validator
    # uses the same set, including lower-priority fallback events.
    "beat_cut": (
        "drop", "climax", "hard_stop", "surge", "section_boundary", "accent",
        "downbeat", "beat", "phrase_boundary", "onset",
    ),
    "phrase_flow": (
        "drop", "climax", "hard_stop", "surge", "section_boundary", "phrase_boundary",
        "accent", "onset",
    ),
}


def _number(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return number if math.isfinite(number) else default


def _clamp(value: Any, default: float = 0.55) -> float:
    return max(0.0, min(1.0, _number(value, default)))


def _canonical_type(value: Any, fallback: str) -> str:
    text = str(value or fallback).strip().lower().replace("-", "_").replace(" ", "_")
    return _TYPE_ALIASES.get(text, text)


def _time_value(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        number = _number(value, -1.0)
        return number if number >= 0 else None
    if isinstance(value, Mapping):
        for key in ("time", "time_seconds", "boundary", "start", "start_seconds"):
            if value.get(key) is not None:
                return _time_value(value[key])
    return None


def _event(value: Any, event_type: str, source: str) -> dict[str, Any] | None:
    time_value = _time_value(value)
    if time_value is None:
        return None
    record = value if isinstance(value, Mapping) else {}
    canonical = _canonical_type(record.get("type"), event_type)
    return {
        "type": canonical,
        "group": _EVENT_GROUPS.get(canonical, canonical),
        "time": time_value,
        "strength": _clamp(record.get("strength", record.get("confidence", 0.55))),
        "confidence": _clamp(record.get("confidence", 0.55)),
        "source": source,
    }


def _values(audiomap: Mapping[str, Any], name: str) -> tuple[Any, str]:
    events = audiomap.get("events") if isinstance(audiomap.get("events"), Mapping) else {}
    aliases = {
        "beats": ("beats", "beat_times", "beat_times_seconds"),
        "downbeats": ("downbeats", "downbeat_times", "downbeat_times_seconds"),
        "onsets": ("onsets", "onset_times", "onset_times_seconds"),
        "accents": ("accents", "accent_times", "accent_times_seconds", "emphasis_nodes"),
        "hard_stops": ("hard_stops", "hard_stop_times", "hard_stop_times_seconds"),
        "drops": ("drops", "drop_times", "drop_times_seconds"),
        "surges": ("surges", "surge_times", "surge_times_seconds"),
        "climaxes": ("climaxes", "climax_times", "climax_times_seconds"),
        "phrases": ("phrase_boundaries", "phrase_boundaries_seconds", "phrases"),
        "phrase_boundaries": ("phrase_boundaries", "phrase_boundaries_seconds", "phrases"),
        "sections": ("section_boundaries", "section_boundaries_seconds", "sections"),
        "section_boundaries": ("section_boundaries", "section_boundaries_seconds", "sections"),
    }
    for candidate in aliases.get(name, (name,)):
        value = events.get(candidate)
        if isinstance(value, list):
            return value, f"events.{candidate}"
        value = audiomap.get(candidate)
        if isinstance(value, list):
            return value, f"top_level.{candidate}"
    return None, f"top_level.{name}"


def normalize_music_event_contract(audiomap: Mapping[str, Any] | None, duration: float) -> dict[str, Any]:
    """Normalize legacy aliases and expose mode-specific allowed events."""

    source = audiomap if isinstance(audiomap, Mapping) else {}
    total = max(0.0, _number(duration, 0.0))
    rhythm = source.get("rhythm_mode")
    mode = str(rhythm.get("mode") if isinstance(rhythm, Mapping) else rhythm or "phrase_flow").strip().lower()
    if mode not in _MODE_ALLOWED_TYPES:
        mode = "phrase_flow"
    events: list[dict[str, Any]] = []
    for name in _EVENT_NAMES:
        values, path = _values(source, name)
        if not isinstance(values, list):
            continue
        fallback_type = _TYPE_ALIASES[name]
        for value in values:
            item = _event(value, fallback_type, path)
            if item and 0.0 < item["time"] < total:
                events.append(item)

    # Legacy v1.1 phrase/section records store start/end intervals.
    for collection, event_type in (("phrases", "phrase_boundary"), ("sections", "section_boundary")):
        values = source.get(collection)
        if not isinstance(values, list):
            continue
        for index, record in enumerate(values):
            if not isinstance(record, Mapping):
                continue
            for key in ("start", "end"):
                time_value = _number(record.get(key), -1.0)
                if 0.0 < time_value < total:
                    events.append({
                        "type": event_type,
                        "group": _EVENT_GROUPS[event_type],
                        "time": time_value,
                        "strength": 0.72,
                        "confidence": 0.65,
                        "source": f"top_level.{collection}[{index}].{key}",
                    })

    deduplicated: list[dict[str, Any]] = []
    for item in sorted(events, key=lambda value: (value["time"], value["type"], value["source"])):
        duplicate = next(
            (
                existing for existing in deduplicated
                if existing["type"] == item["type"] and abs(existing["time"] - item["time"]) <= 0.004
            ),
            None,
        )
        if duplicate is None or item["strength"] > duplicate["strength"]:
            if duplicate is not None:
                deduplicated.remove(duplicate)
            deduplicated.append(item)

    groups = {
        group: sorted({round(item["time"], 5) for item in deduplicated if item["group"] == group})
        for group in ("beats", "downbeats", "onsets", "accents", "hard_stops", "drops", "surges", "climaxes", "phrases", "sections")
    }
    allowed_types = list(_MODE_ALLOWED_TYPES[mode])
    allowed_times = sorted({
        round(item["time"], 5)
        for item in deduplicated
        if item["type"] in allowed_types
    })
    tempo = source.get("tempo") if isinstance(source.get("tempo"), Mapping) else {}
    beat_period = _number(tempo.get("beat_period_seconds"), 0.0)
    tolerance = max(0.10, min(0.28, beat_period * 0.32 if beat_period else 0.20)) if mode == "beat_cut" else 0.55
    required_share = 0.70 if mode == "beat_cut" else 0.60
    digest_payload = {
        "schema_version": SCHEMA_VERSION,
        "mode": mode,
        "duration_seconds": round(total, 5),
        "events": deduplicated,
        "allowed_event_types": allowed_types,
    }
    digest = hashlib.sha256(
        json.dumps(digest_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": SCHEMA_VERSION,
        "mode": mode,
        "duration_seconds": round(total, 5),
        "events": deduplicated,
        "groups": groups,
        "allowed_event_types": allowed_types,
        "allowed_times": allowed_times,
        "tolerance_seconds": round(tolerance, 5),
        "required_aligned_share": required_share,
        "available": bool(allowed_times),
        "event_counts": {key: len(value) for key, value in groups.items()},
        "contract_digest": digest,
    }
