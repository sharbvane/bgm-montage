#!/usr/bin/env python3
"""Versioned edit-decision schema shared by rendering and editor adapters.

The v1.3 schema is a backward-compatible envelope around the established v1.2
shot records.  Existing field aliases remain present so older project tooling
can continue to read new plans while adapters use the normalized fields.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any, Mapping


EDIT_SCHEMA_VERSION = "1.3"
TIMELINE_SCHEMA_VERSION = 3


def _number(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def _first(mapping: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value is not None:
            return value
    return default


def _normalize_transform(shot: Mapping[str, Any]) -> dict[str, Any]:
    existing = shot.get("transform")
    transform = copy.deepcopy(existing) if isinstance(existing, Mapping) else {}
    crop = _first(shot, "crop_plan", "crop", default=transform.get("crop"))
    if isinstance(crop, Mapping):
        transform["crop"] = copy.deepcopy(dict(crop))
    transform.setdefault("scale", {"x": 1.0, "y": 1.0})
    transform.setdefault("position", {"x": 0.0, "y": 0.0})
    transform.setdefault("rotation_degrees", 0.0)
    transform.setdefault("opacity", 1.0)
    return transform


def _normalize_shot(shot: Mapping[str, Any], index: int) -> dict[str, Any]:
    item = copy.deepcopy(dict(shot))
    timeline_start = _number(_first(item, "timeline_start", "output_start", "start"))
    duration = _number(_first(item, "duration", "output_duration"))
    timeline_end = _number(
        _first(item, "timeline_end", "output_end"),
        timeline_start + duration,
    )
    if duration <= 0.0:
        duration = max(0.0, timeline_end - timeline_start)
    if timeline_end <= timeline_start and duration > 0.0:
        timeline_end = timeline_start + duration

    source_start = _number(_first(item, "source_start", "trim_start"))
    source_end = _number(_first(item, "source_end", "trim_end"))
    speed = max(0.000001, _number(item.get("speed"), 1.0))
    if source_end <= source_start and duration > 0.0:
        source_end = source_start + duration * speed

    source_path = str(_first(item, "source_path", "local_path", "path", default=""))
    item.update(
        {
            "shot_index": int(_first(item, "shot_index", "index", default=index)),
            "source_path": source_path,
            "source_start": round(source_start, 6),
            "source_end": round(source_end, 6),
            "timeline_start": round(timeline_start, 6),
            "timeline_end": round(timeline_end, 6),
            "duration": round(duration, 6),
            "speed": round(speed, 8),
            "transform": _normalize_transform(item),
        }
    )
    # v1.2 compatibility aliases.  Do not remove without a schema migration.
    item["local_path"] = source_path
    item["output_start"] = item["timeline_start"]
    item["output_end"] = item["timeline_end"]
    item["output_duration"] = item["duration"]
    return item


def normalize_edit_decisions(
    payload: Mapping[str, Any],
    *,
    bgm_path: str | Path | None = None,
    ratio: str | None = None,
    width: int | None = None,
    height: int | None = None,
    fps: float | None = None,
) -> dict[str, Any]:
    """Return a v1.3 edit plan while preserving compatible v1.2 fields."""
    result = copy.deepcopy(dict(payload))
    old_schema = str(result.get("schema_version") or result.get("version") or "legacy")
    source_shots = result.get("shots")
    if not isinstance(source_shots, list):
        source_shots = result.get("timeline") if isinstance(result.get("timeline"), list) else []
    shots = [_normalize_shot(shot, index) for index, shot in enumerate(source_shots) if isinstance(shot, Mapping)]
    shots.sort(key=lambda shot: (float(shot["timeline_start"]), int(shot["shot_index"])))

    project = copy.deepcopy(result.get("project")) if isinstance(result.get("project"), Mapping) else {}
    if ratio:
        project["ratio"] = ratio
    if width:
        project["width"] = int(width)
    if height:
        project["height"] = int(height)
    if fps:
        project["fps"] = float(fps)
    project.setdefault("timeline_duration", round(max((s["timeline_end"] for s in shots), default=0.0), 6))
    project.setdefault("timebase", "seconds")

    audio_tracks = copy.deepcopy(result.get("audio_tracks")) if isinstance(result.get("audio_tracks"), list) else []
    if bgm_path:
        wanted = str(Path(bgm_path).resolve())
        bgm = next((track for track in audio_tracks if isinstance(track, Mapping) and track.get("role") == "bgm"), None)
        if bgm is None:
            bgm = {
                "track_index": 0,
                "role": "bgm",
                "source_path": wanted,
                "source_start": 0.0,
                "timeline_start": 0.0,
                "timeline_end": project["timeline_duration"],
                "duration": project["timeline_duration"],
                "volume": 1.0,
            }
            audio_tracks.append(bgm)
        else:
            bgm["source_path"] = wanted
            bgm.setdefault("timeline_start", 0.0)
            bgm.setdefault("timeline_end", project["timeline_duration"])
            bgm.setdefault("duration", project["timeline_duration"])
            bgm.setdefault("source_start", 0.0)
            bgm.setdefault("volume", 1.0)

    result.update(
        {
            "schema_version": EDIT_SCHEMA_VERSION,
            "timeline_schema_version": TIMELINE_SCHEMA_VERSION,
            "compatible_readers": ["1.2", "1.3"],
            "project": project,
            "shots": shots,
            "audio_tracks": audio_tracks,
        }
    )
    if old_schema != EDIT_SCHEMA_VERSION:
        result["migrated_from_schema"] = old_schema
    return result


def validate_edit_decisions(payload: Mapping[str, Any], *, require_sources: bool = False) -> dict[str, Any]:
    normalized = normalize_edit_decisions(payload)
    errors: list[str] = []
    warnings: list[str] = []
    previous_end = 0.0
    source_paths: list[str] = []
    for index, shot in enumerate(normalized["shots"]):
        if shot["duration"] <= 0.0:
            errors.append(f"shot {index}: non-positive duration")
        if abs(shot["timeline_start"] - previous_end) > 0.08:
            warnings.append(
                f"shot {index}: timeline gap/overlap {shot['timeline_start'] - previous_end:+.3f}s"
            )
        if shot["source_end"] <= shot["source_start"]:
            errors.append(f"shot {index}: invalid source range")
        source_path = str(shot.get("source_path") or "")
        source_paths.append(source_path)
        if not source_path:
            errors.append(f"shot {index}: missing source path")
        elif require_sources and not Path(source_path).is_file():
            errors.append(f"shot {index}: source does not exist: {source_path}")
        previous_end = shot["timeline_end"]

    return {
        "passed": not errors,
        "schema_version": normalized["schema_version"],
        "shot_count": len(normalized["shots"]),
        "unique_source_count": len(set(source_paths)),
        "timeline_duration": normalized["project"]["timeline_duration"],
        "errors": errors,
        "warnings": warnings,
    }


def load_edit_decisions(path: str | Path, **kwargs: Any) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return normalize_edit_decisions(payload, **kwargs)


def write_edit_decisions(path: str | Path, payload: Mapping[str, Any], **kwargs: Any) -> dict[str, Any]:
    normalized = normalize_edit_decisions(payload, **kwargs)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(normalized, ensure_ascii=False, indent=2), encoding="utf-8")
    return normalized


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate or validate bgm-montage edit decisions")
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bgm", type=Path)
    parser.add_argument("--require-sources", action="store_true")
    args = parser.parse_args()
    normalized = load_edit_decisions(args.input, bgm_path=args.bgm)
    if args.output:
        write_edit_decisions(args.output, normalized)
    print(json.dumps(validate_edit_decisions(normalized, require_sources=args.require_sources), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
