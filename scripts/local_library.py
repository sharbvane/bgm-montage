#!/usr/bin/env python3
"""Persistent incremental indexing and two-stage selection for local video libraries."""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import requests
import numpy as np

from pixabay_pipeline import (
    _atomic_write_json,
    _exclusive_catalog_lock,
    _exclusive_lock_file,
    _ffprobe,
    _image_signals,
    _infer_shot_scale,
    _infer_shot_type,
    _motion_and_stability,
    _ratio_label,
    _sample_video_frames,
    _scene_category,
    _score_candidates,
    _semantic_tags,
    _video_quality,
    evaluate_selected_sufficiency,
)
from visual_intelligence import (
    FEATURE_METADATA_SCHEMA_VERSION,
    asset_visual_features,
    analysis_cache_valid,
    build_feature_detail,
    english_terms,
    metadata_feature_details,
)


SCHEMA_VERSION = "1.4.3-local.2"
LIGHTWEIGHT_SCHEMA_VERSION = "1.4.3-light.1"
LIGHTWEIGHT_FRAME_COUNT = 6
INDEX_LOCK_TIMEOUT_SECONDS = 3600.0
VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".mkv", ".webm", ".avi", ".mts", ".m2ts"}


class LocalLibraryError(RuntimeError):
    """Raised when a local library cannot be indexed or selected safely."""


class InsufficientMaterialError(LocalLibraryError):
    """Raised when the local library cannot satisfy the requested montage."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _library_id(root: Path) -> str:
    normalized = os.path.normcase(str(root.resolve()))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:20]


def _index_path(root: Path, cache_root: Path) -> Path:
    return cache_root / "local-library" / "libraries" / _library_id(root) / "library_index.json"


def _load_index(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"schema_version": SCHEMA_VERSION, "entries": []}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise LocalLibraryError(f"Local library index is unreadable: {path}: {exc}") from None
    if not isinstance(payload, Mapping):
        raise LocalLibraryError(f"Local library index must be a JSON object: {path}")
    return dict(payload)


def _video_files(root: Path) -> list[Path]:
    return sorted(
        (path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES),
        key=lambda path: path.relative_to(root).as_posix().casefold(),
    )


def _signature(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {"size_bytes": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def _content_fingerprint(path: Path) -> dict[str, Any]:
    """Hash bounded samples so moves and replacements have a stable identity."""

    size = path.stat().st_size
    chunk_size = 256 * 1024
    offsets = sorted({0, max(0, size // 2 - chunk_size // 2), max(0, size - chunk_size)})
    digest = hashlib.sha256(f"sampled-sha256-v1:{size}".encode("ascii"))
    sampled = 0
    with path.open("rb") as handle:
        for offset in offsets:
            handle.seek(offset)
            block = handle.read(chunk_size)
            digest.update(offset.to_bytes(8, "big"))
            digest.update(block)
            sampled += len(block)
    return {
        "algorithm": "sampled-sha256-v1",
        "digest": digest.hexdigest(),
        "size_bytes": int(size),
        "sampled_bytes": sampled,
    }


def _content_digest(entry: Mapping[str, Any] | None) -> str:
    if not isinstance(entry, Mapping):
        return ""
    fingerprint = entry.get("content_fingerprint")
    return str(fingerprint.get("digest") or "") if isinstance(fingerprint, Mapping) else ""


def _lightweight_valid(entry: Mapping[str, Any] | None) -> bool:
    profile = entry.get("lightweight_visual") if isinstance(entry, Mapping) else None
    return bool(
        isinstance(profile, Mapping)
        and profile.get("schema_version") == LIGHTWEIGHT_SCHEMA_VERSION
        and int(profile.get("sampled_frame_count") or 0) >= 1
        and _content_digest(entry)
    )


def _median(values: Sequence[Any], default: float = 0.5) -> float:
    numbers = [float(value) for value in values if isinstance(value, (int, float))]
    return float(np.median(numbers)) if numbers else default


def _visual_shot_scale(signals: Sequence[Mapping[str, Any]], depth: float) -> str:
    areas: list[float] = []
    for signal in signals:
        subject = signal.get("subject_profile") if isinstance(signal.get("subject_profile"), Mapping) else {}
        bbox = subject.get("bbox") if isinstance(subject, Mapping) else None
        if isinstance(bbox, Sequence) and len(bbox) == 4:
            try:
                areas.append(max(0.0, float(bbox[2]) - float(bbox[0])) * max(0.0, float(bbox[3]) - float(bbox[1])))
            except (TypeError, ValueError):
                pass
    area = _median(areas, 0.0)
    if area >= 0.38:
        return "close_up"
    if area >= 0.14:
        return "medium"
    return "wide" if depth >= 0.52 or not areas else "medium_wide"


def _lightweight_visual(path: Path, tags: str) -> dict[str, Any]:
    frames, fps, decoded_duration = _sample_video_frames(path, max_samples=LIGHTWEIGHT_FRAME_COUNT)
    if not frames:
        raise LocalLibraryError("video could not be decoded for lightweight visual indexing")
    neutral_color = {"hue_degrees": 180.0, "saturation": 0.5, "value": 0.5}
    signals = [_image_signals(frame, neutral_color) for frame in frames]
    motion, stability, motion_raw = _motion_and_stability(frames)
    aesthetic_keys = (
        "spatial_depth", "composition_quality", "visual_impact", "lighting_quality",
        "atmosphere_quality", "color_quality", "ordinary_travelogue_risk",
    )
    aesthetics = {
        key: _median([
            signal.get("aesthetic_metrics", {}).get(key)
            for signal in signals
            if isinstance(signal.get("aesthetic_metrics"), Mapping)
        ], 0.45)
        for key in aesthetic_keys
    }
    hsv = {
        "hue_degrees": _median([signal.get("mean_hsv", {}).get("hue_degrees") for signal in signals], 180.0),
        "saturation": _median([signal.get("mean_hsv", {}).get("saturation") for signal in signals], 0.5),
        "value": _median([signal.get("mean_hsv", {}).get("value") for signal in signals], 0.5),
    }
    face_risk = _median([signal.get("face_content_risk") for signal in signals], 0.0)
    depth = aesthetics["spatial_depth"]
    shot_scale = _visual_shot_scale(signals, depth)
    subject_class = "people" if face_risk >= 0.28 else "environment" if depth >= 0.48 else "detail"
    scene = _scene_category(tags, None)
    motion_label = "dynamic" if motion >= 0.62 else "gentle" if motion >= 0.25 else "static"
    sharpness = _median([signal.get("sharpness_score") for signal in signals])
    exposure = _median([signal.get("exposure_score") for signal in signals])
    visual_analysis = {
        "aesthetic_score": round(_median([aesthetics["composition_quality"], aesthetics["visual_impact"], aesthetics["lighting_quality"]]), 4),
        "cinematic_score": round(_median([depth, aesthetics["composition_quality"], aesthetics["atmosphere_quality"]]), 4),
        "spatial_depth_score": round(depth, 4),
        "composition_quality_score": round(aesthetics["composition_quality"], 4),
        "visual_impact_score": round(aesthetics["visual_impact"], 4),
        "lighting_quality_score": round(aesthetics["lighting_quality"], 4),
        "atmosphere_quality_score": round(aesthetics["atmosphere_quality"], 4),
        "intrinsic_color_quality_score": round(aesthetics["color_quality"], 4),
        "ordinary_travelogue_risk": round(aesthetics["ordinary_travelogue_risk"], 4),
    }
    quality = {
        "overall_score": round(0.32 * sharpness + 0.26 * exposure + 0.18 * stability + 0.24 * visual_analysis["aesthetic_score"], 4),
        "sharpness_score": round(sharpness, 4),
        "exposure_score": round(exposure, 4),
        "stability_score": round(stability, 4),
        "text_watermark_risk": round(_median([signal.get("text_watermark_risk") for signal in signals], 0.0), 4),
        "face_content_risk": round(face_risk, 4),
        "motion_score": round(motion, 4),
        "motion_direction": motion_raw.get("motion_direction", "unknown"),
        "scene_category": scene,
        "mean_hsv": {key: round(value, 4) for key, value in hsv.items()},
        "visual_analysis": visual_analysis,
        "subject_profile": {"class": subject_class},
    }
    return {
        "schema_version": LIGHTWEIGHT_SCHEMA_VERSION,
        "engine": "existing-cv-signals",
        "sample_target": LIGHTWEIGHT_FRAME_COUNT,
        "sampled_frame_count": len(frames),
        "fps": round(fps, 4),
        "decoded_duration_seconds": round(decoded_duration, 4),
        "scene_category": scene,
        "subject_class": subject_class,
        "shot_scale": shot_scale,
        "motion_score": round(motion, 4),
        "motion_label": motion_label,
        "motion_direction": motion_raw.get("motion_direction", "unknown"),
        "mean_hsv": quality["mean_hsv"],
        "quality": quality,
        "perceptual_hashes": [str(signal["perceptual_hash"]) for signal in signals if signal.get("perceptual_hash")],
    }


def _tags(root: Path, path: Path) -> str:
    relative = path.relative_to(root)
    parts = [root.name, *relative.parts[:-1], relative.stem]
    return " ".join(str(part).replace("_", " ").replace("-", " ") for part in parts)


def _filename_semantics(root: Path, path: Path) -> dict[str, Any]:
    """Extract cheap path semantics without claiming visual recognition."""

    relative = path.relative_to(root)
    raw_parts = [*relative.parts[:-1], relative.stem]
    raw_text = " ".join(str(part).replace("_", " ").replace("-", " ") for part in raw_parts)
    terms = [
        term
        for term in english_terms(raw_text)
        if not re.fullmatch(r"(?:mmexport|clip|video|mov|img)?\d+", term.replace(" ", ""))
        and not re.fullmatch(r"[a-f0-9]{8,}", term.replace(" ", ""))
    ]
    input_digest = hashlib.sha256(
        json.dumps(raw_parts, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "raw": raw_text,
        "terms": list(dict.fromkeys(terms))[:32],
        "available": bool(terms),
        "source": "filename",
        "confidence": 0.62 if terms else 0.0,
        "input_digest": input_digest,
        "schema_version": FEATURE_METADATA_SCHEMA_VERSION,
    }


def _merge_semantic_tags(*groups: Sequence[Any]) -> list[str]:
    values: list[str] = []
    for group in groups:
        for value in group:
            text = str(value).strip().lower()
            if text and text not in values:
                values.append(text)
    return values[:48]


def _metadata_features(
    *,
    tags: str,
    filename_semantics: Mapping[str, Any],
    semantic_tags: Sequence[Any],
    shot_scale: str,
    shot_scale_detail: Mapping[str, Any],
    quality: Mapping[str, Any] | None = None,
    lightweight_visual: Mapping[str, Any] | None = None,
    motion_direction: str = "unknown",
    mean_hsv: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    probe: dict[str, Any] = {
        "tags": tags,
        "filename_semantics": dict(filename_semantics),
        "semantic_tags": list(semantic_tags),
        "shot_scale": shot_scale,
        "shot_scale_detail": dict(shot_scale_detail),
        "quality": dict(quality or {}),
        "lightweight_visual": dict(lightweight_visual or {}),
        "motion_direction": motion_direction,
        "mean_hsv": dict(mean_hsv or {}),
    }
    features = asset_visual_features(probe)
    details = metadata_feature_details(probe, features)
    details["shot_scale"] = dict(shot_scale_detail)
    return details


def _metadata_refresh_needed(entry: Mapping[str, Any], filename: Mapping[str, Any]) -> bool:
    if str(entry.get("metadata_schema_version") or "") != FEATURE_METADATA_SCHEMA_VERSION:
        return True
    existing = entry.get("filename_semantics") if isinstance(entry.get("filename_semantics"), Mapping) else {}
    if str(existing.get("input_digest") or "") != str(filename.get("input_digest") or ""):
        return True
    details = entry.get("metadata_features")
    if not isinstance(details, Mapping):
        return True
    return not all(isinstance(details.get(key), Mapping) for key in ("shot_scale", "world", "time_weather", "camera_language", "motion"))


def _refresh_metadata_entry(
    root: Path,
    path: Path,
    entry: Mapping[str, Any],
    filename: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Refresh path-derived metadata without re-decoding the video."""

    current = dict(entry)
    tags = _tags(root, path)
    filename = dict(filename or _filename_semantics(root, path))
    lightweight = current.get("lightweight_visual") if isinstance(current.get("lightweight_visual"), Mapping) else {}
    quality = current.get("quality") if isinstance(current.get("quality"), Mapping) else {}
    visual_features = quality.get("visual_features") if isinstance(quality.get("visual_features"), Mapping) else {}
    visual_scale = str(lightweight.get("shot_scale") or "").strip().lower()
    legacy_scale = str(current.get("shot_scale") or visual_features.get("shot_scale") or "medium").strip().lower()
    shot_scale = visual_scale or legacy_scale
    shot_scale_detail = build_feature_detail(
        shot_scale,
        available=bool(visual_scale),
        source="lightweight_visual" if visual_scale else "legacy_field",
        confidence=0.72 if visual_scale else 0.0,
        evidence=["lightweight_visual"] if visual_scale else ["legacy_flat_field"],
    )
    scene = current.get("scene_category") or quality.get("scene_category") or _scene_category(tags, None)
    shot_type = "close_up" if shot_scale == "close_up" else "wide" if "wide" in shot_scale else _infer_shot_type(tags)
    semantic_tags = _merge_semantic_tags(
        _semantic_tags(tags, scene, shot_type, shot_scale, (lightweight.get("subject_class"), lightweight.get("motion_label"))),
        filename.get("terms") or [],
    )
    metadata = _metadata_features(
        tags=tags,
        filename_semantics=filename,
        semantic_tags=semantic_tags,
        shot_scale=shot_scale,
        shot_scale_detail=shot_scale_detail,
        quality=quality,
        lightweight_visual=lightweight,
        motion_direction=str(current.get("motion_direction") or quality.get("motion_direction") or "unknown"),
        mean_hsv=current.get("mean_hsv") if isinstance(current.get("mean_hsv"), Mapping) else quality.get("mean_hsv", {}),
    )
    updated_quality = dict(quality)
    updated_visual = dict(visual_features)
    if updated_visual:
        updated_visual["shot_scale"] = shot_scale
        updated_visual["shot_scale_detail"] = shot_scale_detail
        updated_visual["feature_details"] = metadata
        updated_quality["visual_features"] = updated_visual
    current.update(
        {
            "tags": tags,
            "filename_semantics": filename,
            "metadata_features": metadata,
            "metadata_schema_version": FEATURE_METADATA_SCHEMA_VERSION,
            "semantic_tags": semantic_tags,
            "scene_category": scene,
            "shot_type": shot_type,
            "shot_scale": shot_scale,
            "shot_scale_detail": shot_scale_detail,
            "quality": updated_quality,
            "metadata_refresh": {"status": "refreshed", "reason": "path_or_schema_change"},
        }
    )
    return current


def _asset_id(identity: str) -> str:
    return "local-" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]


def _deep_analysis_valid(entry: Mapping[str, Any]) -> bool:
    quality = entry.get("quality") if isinstance(entry.get("quality"), Mapping) else {}
    fingerprint = entry.get("fingerprint") if isinstance(entry.get("fingerprint"), Mapping) else {}
    file_hash = str(entry.get("file_hash") or fingerprint.get("sha256") or "")
    return entry.get("analysis_status") == "indexed" and analysis_cache_valid(quality, file_hash)


def _inventory_entry(
    root: Path,
    path: Path,
    previous: Mapping[str, Any] | None,
    content_fingerprint: Mapping[str, Any],
) -> dict[str, Any]:
    relative = path.relative_to(root).as_posix()
    tags = _tags(root, path)
    filename_semantics = _filename_semantics(root, path)
    content_digest = str(content_fingerprint.get("digest") or "")
    asset_id = _asset_id(content_digest or relative.casefold())
    base = {
        "id": asset_id,
        "asset_id": asset_id,
        "provider": "local-library",
        "relative_path": relative,
        "local_path": str(path.resolve()),
        "path": str(path.resolve()),
        "signature": _signature(path),
        "content_fingerprint": dict(content_fingerprint),
        "tags": tags,
        "usage_history": list((previous or {}).get("usage_history") or []),
        "historical_usage_count": int((previous or {}).get("historical_usage_count") or 0),
    }
    try:
        probe = _ffprobe(path)
        video = next(
            stream for stream in probe.get("streams", [])
            if isinstance(stream, Mapping) and stream.get("codec_type") == "video"
        )
        width = int(video.get("width") or 0)
        height = int(video.get("height") or 0)
        duration = float(video.get("duration") or probe.get("format", {}).get("duration") or 0.0)
        if width <= 0 or height <= 0 or duration <= 0:
            raise ValueError("missing video dimensions or duration")
        lightweight = _lightweight_visual(path, tags)
    except (OSError, RuntimeError, StopIteration, TypeError, ValueError) as exc:
        return {**base, "available": False, "analysis_status": "failed", "failure_reason": str(exc)}
    light_quality = dict(lightweight.get("quality") or {})
    scene = lightweight.get("scene_category") or _scene_category(tags, None)
    visual_scale = str(lightweight.get("shot_scale") or "").strip().lower()
    shot_scale = visual_scale or _infer_shot_scale(tags)
    shot_scale_detail = build_feature_detail(
        shot_scale,
        available=bool(visual_scale),
        source="lightweight_visual" if visual_scale else "tag_fallback",
        confidence=0.72 if visual_scale else 0.12,
        evidence=["lightweight_visual"] if visual_scale else ["filename_or_tags"],
    )
    shot_type = "close_up" if shot_scale == "close_up" else "wide" if "wide" in shot_scale else _infer_shot_type(tags)
    semantic_tags = _merge_semantic_tags(
        _semantic_tags(
            tags,
            scene,
            shot_type,
            shot_scale,
            (lightweight.get("subject_class"), lightweight.get("motion_label")),
        ),
        filename_semantics.get("terms") or [],
    )
    metadata = _metadata_features(
        tags=tags,
        filename_semantics=filename_semantics,
        semantic_tags=semantic_tags,
        shot_scale=shot_scale,
        shot_scale_detail=shot_scale_detail,
        quality=light_quality,
        lightweight_visual=lightweight,
        motion_direction=str(lightweight.get("motion_direction") or "unknown"),
        mean_hsv=lightweight.get("mean_hsv") if isinstance(lightweight.get("mean_hsv"), Mapping) else {},
    )
    light_quality["metadata_features"] = metadata
    return {
        **base,
        "available": True,
        "analysis_status": "inventory",
        "failure_reason": None,
        "canonical_source_id": f"local-content:{content_digest or asset_id}",
        "duration": duration,
        "duration_seconds": duration,
        "width": width,
        "height": height,
        "resolution": {"width": width, "height": height},
        "ratio": _ratio_label(width, height),
        "media": {
            "width": width,
            "height": height,
            "duration_seconds": duration,
            "codec": str(video.get("codec_name") or ""),
            "size_bytes": int(path.stat().st_size),
            "ratio": _ratio_label(width, height),
        },
        "quality": light_quality,
        "fingerprint": {"perceptual_hashes": list(lightweight.get("perceptual_hashes") or [])},
        "lightweight_visual": lightweight,
        "filename_semantics": filename_semantics,
        "metadata_features": metadata,
        "metadata_schema_version": FEATURE_METADATA_SCHEMA_VERSION,
        "semantic_tags": semantic_tags,
        "scene_category": scene,
        "shot_type": shot_type,
        "shot_scale": shot_scale,
        "shot_scale_detail": shot_scale_detail,
        "motion_score": float(lightweight.get("motion_score") or 0.0),
        "motion_direction": lightweight.get("motion_direction") or "unknown",
        "mean_hsv": dict(lightweight.get("mean_hsv") or {}),
        "usable_segments": [],
    }


def _analyze_entry(root: Path, path: Path, previous: Mapping[str, Any] | None) -> dict[str, Any]:
    relative = path.relative_to(root).as_posix()
    tags = _tags(root, path)
    filename_semantics = _filename_semantics(root, path)
    signature = _signature(path)
    content_fingerprint = dict((previous or {}).get("content_fingerprint") or _content_fingerprint(path))
    content_digest = str(content_fingerprint.get("digest") or "")
    asset_id = _asset_id(content_digest or relative.casefold())
    base = {
        "id": asset_id,
        "asset_id": asset_id,
        "provider": "local-library",
        "relative_path": relative,
        "local_path": str(path.resolve()),
        "path": str(path.resolve()),
        "signature": signature,
        "content_fingerprint": content_fingerprint,
        "tags": tags,
        "usage_history": list((previous or {}).get("usage_history") or []),
        "historical_usage_count": int((previous or {}).get("historical_usage_count") or 0),
        "lightweight_visual": dict((previous or {}).get("lightweight_visual") or {}),
    }
    try:
        # Intrinsic indexing is task-independent; current BGM/theme matching happens later.
        quality, media = _video_quality(path, {}, (1, 1), tags=tags)
    except Exception as exc:
        return {
            **base,
            "available": False,
            "analysis_status": "failed",
            "failure_reason": str(exc),
        }
    fingerprint = dict(media.get("fingerprint") or {})
    quality = dict(quality)
    width = int(media.get("width") or 0)
    height = int(media.get("height") or 0)
    duration = float(media.get("duration_seconds") or 0.0)
    canonical = f"local-content:{content_digest or fingerprint.get('sha256') or asset_id}"
    lightweight = base.get("lightweight_visual") if isinstance(base.get("lightweight_visual"), Mapping) else {}
    visual_scale = str(lightweight.get("shot_scale") or "").strip().lower()
    legacy_scale = str(
        (quality.get("visual_features") or {}).get("shot_scale")
        if isinstance(quality.get("visual_features"), Mapping)
        else ""
    ).strip().lower()
    shot_scale = visual_scale or legacy_scale or _infer_shot_scale(tags)
    shot_scale_detail = build_feature_detail(
        shot_scale,
        available=bool(visual_scale),
        source="lightweight_visual" if visual_scale else "tag_fallback",
        confidence=0.72 if visual_scale else 0.12,
        evidence=["lightweight_visual"] if visual_scale else ["legacy_visual_or_tags"],
    )
    scene = quality.get("scene_category") or _scene_category(tags, None)
    shot_type = "close_up" if shot_scale == "close_up" else "wide" if "wide" in shot_scale else _infer_shot_type(tags)
    semantic_tags = _merge_semantic_tags(
        _semantic_tags(tags, scene, shot_type, shot_scale),
        filename_semantics.get("terms") or [],
    )
    visual_features = dict(quality.get("visual_features") or {}) if isinstance(quality.get("visual_features"), Mapping) else {}
    visual_features["shot_scale"] = shot_scale
    visual_features["shot_scale_detail"] = shot_scale_detail
    metadata = _metadata_features(
        tags=tags,
        filename_semantics=filename_semantics,
        semantic_tags=semantic_tags,
        shot_scale=shot_scale,
        shot_scale_detail=shot_scale_detail,
        quality={**quality, "visual_features": visual_features},
        lightweight_visual=lightweight,
        motion_direction=str(quality.get("motion_direction") or "unknown"),
        mean_hsv=quality.get("mean_hsv") if isinstance(quality.get("mean_hsv"), Mapping) else {},
    )
    visual_features["feature_details"] = metadata
    quality["visual_features"] = visual_features
    quality["metadata_features"] = metadata
    return {
        **base,
        "available": bool(quality.get("passed")),
        "analysis_status": "indexed",
        "failure_reason": None if quality.get("passed") else "; ".join(quality.get("rejection_reasons") or []),
        "canonical_source_id": canonical,
        "file_hash": str(fingerprint.get("sha256") or ""),
        "duration": duration,
        "duration_seconds": duration,
        "width": width,
        "height": height,
        "resolution": {"width": width, "height": height},
        "ratio": _ratio_label(width, height),
        "quality": quality,
        "media": media,
        "fingerprint": fingerprint,
        "filename_semantics": filename_semantics,
        "metadata_features": metadata,
        "metadata_schema_version": FEATURE_METADATA_SCHEMA_VERSION,
        "semantic_tags": semantic_tags,
        "scene_category": scene,
        "shot_type": shot_type,
        "shot_scale": shot_scale,
        "shot_scale_detail": shot_scale_detail,
        "motion_score": float(quality.get("motion_score") or 0.5),
        "motion_direction": quality.get("motion_direction") or "unknown",
        "usable_segments": list(quality.get("usable_segments") or []),
    }


def sync_local_library(
    library_root: str | os.PathLike[str],
    cache_root: str | os.PathLike[str],
    *,
    metadata_refresh_limit: int = 0,
) -> dict[str, Any]:
    """Transactionally sync the index without decoding the whole library.

    Existing entries receive bounded path-metadata migration only when the
    caller explicitly supplies a positive budget. New/changed files still get
    their normal inventory profile.
    """

    root = Path(library_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Local library directory not found: {root}")
    metadata_refresh_limit = max(0, int(metadata_refresh_limit))
    index_path = _index_path(root, Path(cache_root).expanduser().resolve())
    # ponytail: one catalog lock serializes cold sync; shard only after measured contention.
    with _exclusive_catalog_lock(index_path, timeout_seconds=INDEX_LOCK_TIMEOUT_SECONDS):
        previous = _load_index(index_path)
        previous_by_relative = {
            str(entry.get("relative_path") or "").casefold(): dict(entry)
            for entry in previous.get("entries", [])
            if isinstance(entry, Mapping) and entry.get("relative_path")
        }
        previous_by_content = {
            _content_digest(entry): dict(entry)
            for entry in previous.get("entries", [])
            if isinstance(entry, Mapping) and _content_digest(entry)
        }
        files = _video_files(root)
        current_keys = {path.relative_to(root).as_posix().casefold() for path in files}
        entries: list[dict[str, Any]] = []
        added = changed = moved = reused = migrated = failed = 0
        added_paths: list[str] = []
        changed_paths: list[str] = []
        moved_paths: list[str] = []
        moved_source_keys: set[str] = set()
        light_analyzed_paths: list[str] = []
        metadata_refreshed_paths: list[str] = []
        metadata_deferred_paths: list[str] = []
        metadata_refresh_budget_used = 0
        for path in files:
            relative = path.relative_to(root).as_posix()
            old = previous_by_relative.get(relative.casefold())
            signature = _signature(path)
            filename_semantics = _filename_semantics(root, path)
            unchanged = bool(old and old.get("signature") == signature and _lightweight_valid(old))
            if unchanged:
                entry = dict(old)
                entry.update({"local_path": str(path.resolve()), "path": str(path.resolve())})
                if _metadata_refresh_needed(entry, filename_semantics):
                    if metadata_refresh_budget_used < metadata_refresh_limit:
                        entry = _refresh_metadata_entry(root, path, entry, filename_semantics)
                        metadata_refreshed_paths.append(relative)
                        metadata_refresh_budget_used += 1
                    else:
                        metadata_deferred_paths.append(relative)
                if entry.get("analysis_status") == "indexed" and not _deep_analysis_valid(entry):
                    entry["analysis_status"] = "inventory"
                entries.append(entry)
                previous_by_content.setdefault(_content_digest(entry), entry)
                reused += 1
                continue

            content_fingerprint = _content_fingerprint(path)
            content_digest = str(content_fingerprint["digest"])
            content_previous = previous_by_content.get(content_digest)
            same_old_content = bool(old and _content_digest(old) == content_digest)
            legacy_migration = bool(old and old.get("signature") == signature and not _content_digest(old))
            carried = content_previous or (old if same_old_content or legacy_migration else None)
            is_move = bool(
                old is None
                and content_previous
                and str(content_previous.get("relative_path") or "").casefold() not in current_keys
            )
            if content_previous and _lightweight_valid(content_previous):
                entry = dict(content_previous)
                entry.update(
                    {
                        "relative_path": relative,
                        "local_path": str(path.resolve()),
                        "path": str(path.resolve()),
                        "signature": signature,
                        "tags": _tags(root, path),
                    }
                )
                if _metadata_refresh_needed(entry, filename_semantics):
                    entry = _refresh_metadata_entry(root, path, entry, filename_semantics)
                    metadata_refreshed_paths.append(relative)
                moved += int(is_move)
                if is_move:
                    moved_paths.append(relative)
                    moved_source_keys.add(str(content_previous.get("relative_path") or "").casefold())
            else:
                entry = _inventory_entry(root, path, carried, content_fingerprint)
                light_analyzed_paths.append(relative)
                migrated += int(legacy_migration)
            entries.append(entry)
            previous_by_content[content_digest] = entry
            if old is None and not is_move:
                added += 1
                added_paths.append(relative)
            elif old is not None and not legacy_migration and not same_old_content:
                changed += 1
                changed_paths.append(relative)
            failed += entry.get("analysis_status") == "failed"

        deleted_paths = sorted(set(previous_by_relative) - current_keys - moved_source_keys)
        deleted = len(deleted_paths)
        profiled = sum(_lightweight_valid(entry) for entry in entries)
        report = {
            "scanned_files": len(files),
            "added": added,
            "changed": changed,
            "moved": moved,
            "migrated": migrated,
            "deleted": deleted,
            "reused": reused,
            "light_profiled_entries": profiled,
            "light_analyzed": len(light_analyzed_paths),
            "light_cache_hits": profiled - len(light_analyzed_paths),
            "light_cache_hit_rate": round((profiled - len(light_analyzed_paths)) / max(1, profiled), 6),
            "metadata_refreshed": len(metadata_refreshed_paths),
            "metadata_refreshed_paths": metadata_refreshed_paths,
            "metadata_refresh_limit": metadata_refresh_limit,
            "metadata_refresh_deferred": len(metadata_deferred_paths),
            "metadata_refresh_deferred_paths": metadata_deferred_paths,
            "deep_analyzed": 0,
            "failed": failed,
            "added_paths": added_paths,
            "changed_paths": changed_paths,
            "moved_paths": moved_paths,
            "light_analyzed_paths": light_analyzed_paths,
            "deleted_paths": deleted_paths,
        }
        payload = {
            "schema_version": SCHEMA_VERSION,
            "library_root": str(root),
            "library_id": _library_id(root),
            "selection_signature": previous.get("selection_signature"),
            "updated_at": _utc_now(),
            "sync": report,
            "entries": entries,
        }
        _atomic_write_json(index_path, payload)
    return {**payload, "index_path": str(index_path)}


def _candidate(entry: Mapping[str, Any]) -> dict[str, Any]:
    media = entry.get("media") if isinstance(entry.get("media"), Mapping) else {}
    quality = entry.get("quality") if isinstance(entry.get("quality"), Mapping) else {}
    width = int(entry.get("width") or media.get("width") or 0)
    height = int(entry.get("height") or media.get("height") or 0)
    return {
        "id": entry.get("id") or entry.get("asset_id"),
        "asset_id": entry.get("asset_id") or entry.get("id"),
        "provider": "local-library",
        "tags": entry.get("tags") or "",
        "duration": float(entry.get("duration_seconds") or 0.0),
        "user": "local-library",
        "views": 0,
        "downloads": 0,
        "likes": 0,
        "comments": 0,
        "variant": {"name": "local", "url": "", "width": width, "height": height, "size": int(entry.get("signature", {}).get("size_bytes") or 0)},
        "matched_queries": ["local library index"],
        "search_rounds": [0],
        "raw": {"id": entry.get("id") or entry.get("asset_id")},
        "local_reuse_entry": entry,
        "canonical_source_id": entry.get("canonical_source_id"),
        "semantic_tags": list(entry.get("semantic_tags") or []),
        "filename_semantics": dict(entry.get("filename_semantics") or {})
        if isinstance(entry.get("filename_semantics"), Mapping)
        else {},
        "metadata_features": dict(entry.get("metadata_features") or {})
        if isinstance(entry.get("metadata_features"), Mapping)
        else {},
        "scene_category": entry.get("scene_category"),
        "shot_type": entry.get("shot_type"),
        "shot_scale": entry.get("shot_scale"),
        "shot_scale_detail": dict(entry.get("shot_scale_detail") or {})
        if isinstance(entry.get("shot_scale_detail"), Mapping)
        else {},
        "motion_score_estimate": float(entry.get("motion_score") or 0.5),
        "face_content_risk": float(quality.get("face_content_risk") or 0.0),
        "historical_usage_count": int(entry.get("historical_usage_count") or 0),
    }


def _selected_record(candidate: Mapping[str, Any]) -> dict[str, Any]:
    entry = dict(candidate.get("local_reuse_entry") or {})
    quality = dict(entry.get("quality") or {})
    media = dict(entry.get("media") or {})
    return {
        **entry,
        "id": entry.get("id") or entry.get("asset_id"),
        "asset_id": entry.get("asset_id") or entry.get("id"),
        "provider": "local-library",
        "score": float(candidate.get("diversity_adjusted_score") or candidate.get("pre_score") or quality.get("overall_score") or 0.5),
        "pre_score": float(candidate.get("pre_score") or 0.0),
        "diversity_adjusted_score": float(candidate.get("diversity_adjusted_score") or candidate.get("pre_score") or 0.0),
        "score_components": dict(candidate.get("score_components") or {}),
        "duration": float(entry.get("duration_seconds") or media.get("duration_seconds") or 0.0),
        "duration_seconds": float(entry.get("duration_seconds") or media.get("duration_seconds") or 0.0),
        "quality": quality,
        "media": media,
        "available": True,
        "download_status": "local_index",
        "search_queries": ["local library index"],
    }


def _indexed_entry(index: Mapping[str, Any], target: Mapping[str, Any]) -> dict[str, Any] | None:
    digest = _content_digest(target)
    relative = str(target.get("relative_path") or "").casefold()
    entries = [entry for entry in index.get("entries", []) if isinstance(entry, Mapping)]
    for entry in entries:
        if relative and str(entry.get("relative_path") or "").casefold() == relative:
            return dict(entry)
    for entry in entries:
        if digest and _content_digest(entry) == digest:
            return dict(entry)
    return None


def refresh_local_library_metadata(
    library_root: str | os.PathLike[str],
    index_path: str | os.PathLike[str],
    relative_paths: Sequence[str],
    *,
    limit: int = 64,
) -> dict[str, Any]:
    """Refresh cheap metadata for a bounded, already-ranked candidate set."""

    root = Path(library_root).expanduser().resolve()
    index_file = Path(index_path).expanduser().resolve()
    requested = list(dict.fromkeys(str(path) for path in relative_paths))[: max(0, int(limit))]
    if not requested:
        return {**_load_index(index_file), "metadata_refreshed_paths": []}
    with _exclusive_catalog_lock(index_file, timeout_seconds=INDEX_LOCK_TIMEOUT_SECONDS):
        latest = _load_index(index_file)
        by_relative = {
            str(entry.get("relative_path") or "").casefold(): dict(entry)
            for entry in latest.get("entries", [])
            if isinstance(entry, Mapping) and entry.get("relative_path")
        }
        refreshed: list[str] = []
        for relative in requested:
            key = relative.casefold()
            entry = by_relative.get(key)
            path = root / Path(relative)
            if not entry or not path.is_file():
                continue
            filename_semantics = _filename_semantics(root, path)
            if _metadata_refresh_needed(entry, filename_semantics):
                by_relative[key] = _refresh_metadata_entry(root, path, entry, filename_semantics)
                refreshed.append(relative)
        if refreshed:
            latest["entries"] = [
                by_relative.get(str(entry.get("relative_path") or "").casefold(), entry)
                for entry in latest.get("entries", [])
            ]
            sync = dict(latest.get("sync") or {})
            prior_paths = list(sync.get("metadata_refreshed_paths") or [])
            sync.update(
                {
                    "metadata_refreshed": int(sync.get("metadata_refreshed") or 0) + len(refreshed),
                    "metadata_refreshed_paths": list(dict.fromkeys([*prior_paths, *refreshed])),
                    "metadata_migration_mode": "bounded_ranked_candidates",
                }
            )
            latest.update({"schema_version": SCHEMA_VERSION, "updated_at": _utc_now(), "sync": sync})
            _atomic_write_json(index_file, latest)
        return {**latest, "metadata_refreshed_paths": refreshed, "index_path": str(index_file)}


def _commit_deep_result(index_path: Path, analyzed: Mapping[str, Any]) -> dict[str, Any]:
    with _exclusive_catalog_lock(index_path, timeout_seconds=INDEX_LOCK_TIMEOUT_SECONDS):
        latest = _load_index(index_path)
        current = _indexed_entry(latest, analyzed)
        if current is None or current.get("signature") != analyzed.get("signature") or _content_digest(current) != _content_digest(analyzed):
            return current or dict(analyzed)
        merged = dict(analyzed)
        merged["local_path"] = current.get("local_path")
        merged["path"] = current.get("path")
        merged["relative_path"] = current.get("relative_path")
        merged["usage_history"] = list(current.get("usage_history") or [])
        merged["historical_usage_count"] = int(current.get("historical_usage_count") or 0)
        entries = [
            merged if _content_digest(entry) == _content_digest(current) and str(entry.get("relative_path") or "").casefold() == str(current.get("relative_path") or "").casefold() else entry
            for entry in latest.get("entries", [])
        ]
        latest.update({"schema_version": SCHEMA_VERSION, "updated_at": _utc_now(), "entries": entries})
        _atomic_write_json(index_path, latest)
        return merged


def _deep_entry_transaction(root: Path, index_path: Path, entry: Mapping[str, Any]) -> tuple[dict[str, Any], bool]:
    digest = _content_digest(entry) or hashlib.sha256(str(entry.get("relative_path") or "").encode("utf-8")).hexdigest()
    lock_path = index_path.parent / ".entry-locks" / f"{digest}.lock"
    with _exclusive_lock_file(
        lock_path,
        purpose=f"local library deep analysis {digest[:12]}",
        timeout_seconds=INDEX_LOCK_TIMEOUT_SECONDS,
    ):
        with _exclusive_catalog_lock(index_path, timeout_seconds=INDEX_LOCK_TIMEOUT_SECONDS):
            current = _indexed_entry(_load_index(index_path), entry) or dict(entry)
        if _deep_analysis_valid(current):
            return current, True
        analyzed = _analyze_entry(root, Path(str(current["local_path"])), current)
        return _commit_deep_result(index_path, analyzed), False


def _commit_selection_signature(index_path: Path, signature: str) -> dict[str, Any]:
    with _exclusive_catalog_lock(index_path, timeout_seconds=INDEX_LOCK_TIMEOUT_SECONDS):
        latest = _load_index(index_path)
        latest.update({"schema_version": SCHEMA_VERSION, "selection_signature": signature, "updated_at": _utc_now()})
        _atomic_write_json(index_path, latest)
        return latest


def run_local_library_pipeline(
    theme: str,
    style_profile: Mapping[str, Any],
    audio_profile: Mapping[str, Any],
    library_root: str | os.PathLike[str],
    cache_root: str | os.PathLike[str],
    desired_count: int,
    ratio: str,
    *,
    min_resolution: tuple[int, int] = (1280, 720),
    target_duration: float | None = None,
    timeline_plan: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Sync, coarse-rank the index, then expose only a bounded fine candidate set."""

    # Production selection never performs an unbounded cache migration during
    # the catalog sync. Only the already-ranked fine candidate set may be
    # refreshed below.
    sync = sync_local_library(library_root, cache_root, metadata_refresh_limit=0)
    root = Path(library_root).expanduser().resolve()
    index_path = Path(str(sync["index_path"]))
    min_long, min_short = sorted((int(min_resolution[0]), int(min_resolution[1])), reverse=True)
    entries = []
    for entry in sync.get("entries", []):
        if not isinstance(entry, Mapping) or not entry.get("available"):
            continue
        local_path = Path(str(entry.get("local_path") or ""))
        long_side, short_side = sorted((int(entry.get("width") or 0), int(entry.get("height") or 0)), reverse=True)
        if local_path.is_file() and long_side >= min_long and short_side >= min_short:
            entries.append(entry)
    if not entries:
        raise InsufficientMaterialError("Local library contains no indexed videos that pass current quality and resolution gates")
    ratio_value = {"16:9": 16 / 9, "9:16": 9 / 16, "1:1": 1.0, "4:5": 4 / 5}.get(ratio, 16 / 9)
    candidates = [_candidate(entry) for entry in entries]
    with requests.Session() as session:
        ranked = _score_candidates(
            session,
            candidates,
            theme,
            style_profile,
            audio_profile,
            Path(cache_root).expanduser().resolve() / "local-library",
            ratio_value,
            min_resolution,
            desired_count,
        )
    slots = timeline_plan.get("slots", []) if isinstance(timeline_plan, Mapping) else list(timeline_plan or [])
    fine_limit = min(len(ranked), min(64, max(16, desired_count * 3, len(slots) * 2)))
    metadata_candidates = []
    for candidate in ranked[:fine_limit]:
        entry = candidate.get("local_reuse_entry") if isinstance(candidate, Mapping) else None
        relative = str(entry.get("relative_path") or "") if isinstance(entry, Mapping) else ""
        path = root / Path(relative) if relative else None
        if isinstance(entry, Mapping) and path and path.is_file() and _metadata_refresh_needed(
            entry, _filename_semantics(root, path)
        ):
            metadata_candidates.append(relative)
    if metadata_candidates:
        refreshed_index = refresh_local_library_metadata(
            root,
            index_path,
            metadata_candidates,
            limit=fine_limit,
        )
        refreshed_by_relative = {
            str(entry.get("relative_path") or "").casefold(): dict(entry)
            for entry in refreshed_index.get("entries", [])
            if isinstance(entry, Mapping) and entry.get("relative_path")
        }
        for index, candidate in enumerate(ranked):
            entry = candidate.get("local_reuse_entry") if isinstance(candidate, Mapping) else None
            relative = str(entry.get("relative_path") or "") if isinstance(entry, Mapping) else ""
            updated = refreshed_by_relative.get(relative.casefold())
            if updated:
                ranked[index] = _candidate(updated)
        sync["sync"].update(
            {
                "metadata_migration_mode": "bounded_ranked_candidates",
                "metadata_candidate_count": len(metadata_candidates),
                "metadata_refreshed_paths": list(refreshed_index.get("metadata_refreshed_paths") or []),
            }
        )
        sync["entries"] = refreshed_index.get("entries", [])
    selection_signature = hashlib.sha256(
        json.dumps(
            {
                "theme": theme,
                "audio": audio_profile.get("analysis_digest"),
                "style": style_profile.get("analysis_digest") or style_profile.get("profile_digest"),
                "ratio": ratio,
                "desired_count": desired_count,
                "slot_count": len(slots),
            },
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    reusable_ranked = [candidate for candidate in ranked if _deep_analysis_valid(candidate["local_reuse_entry"])]
    migrated_signature = bool(
        not sync.get("selection_signature")
        and not sync["sync"].get("added")
        and not sync["sync"].get("changed")
        and len(reusable_ranked) >= fine_limit
    )
    same_selection = sync.get("selection_signature") == selection_signature or migrated_signature
    unchanged_library = not any(sync["sync"].get(key) for key in ("added", "changed", "moved", "migrated"))
    fine_candidates = (
        reusable_ranked[:fine_limit]
        if same_selection and unchanged_library and len(reusable_ranked) >= desired_count
        else ranked[:fine_limit]
    )

    deep_analyzed = deep_reused = 0
    refreshed: list[dict[str, Any]] = []
    for candidate in fine_candidates:
        entry = candidate["local_reuse_entry"]
        analyzed, reused_after_lock = _deep_entry_transaction(root, index_path, entry)
        deep_reused += int(reused_after_lock)
        deep_analyzed += int(not reused_after_lock)
        candidate["local_reuse_entry"] = analyzed
        refreshed.append(candidate)

    sync["sync"]["deep_analyzed"] = deep_analyzed
    sync["sync"]["deep_reused"] = deep_reused
    sync["selection_signature"] = selection_signature
    latest_index = _commit_selection_signature(index_path, selection_signature)
    sync["entries"] = latest_index.get("entries", [])

    with requests.Session() as session:
        fine_candidates = _score_candidates(
            session,
            refreshed,
            theme,
            style_profile,
            audio_profile,
            Path(cache_root).expanduser().resolve() / "local-library",
            ratio_value,
            min_resolution,
            desired_count,
        )
    selected = [
        _selected_record(candidate) for candidate in fine_candidates
        if candidate["local_reuse_entry"].get("available") and _deep_analysis_valid(candidate["local_reuse_entry"])
    ][:desired_count]
    sufficiency = evaluate_selected_sufficiency(selected, desired_count, target_duration)
    # A user-curated local library may intentionally contain one world/scene;
    # keep scene variety advisory while retaining independence, face and coverage gates.
    sufficiency["failures"] = [
        failure for failure in sufficiency.get("failures", [])
        if not str(failure).startswith("scene categories ")
    ]
    sufficiency["passed"] = not sufficiency["failures"]
    sufficiency["scene_diversity_gate"] = "advisory_for_local_library"
    if not sufficiency.get("passed"):
        raise InsufficientMaterialError("Local library selection failed: " + "; ".join(sufficiency.get("failures") or []))

    run_manifest = index_path.parent / "run_manifests" / f"sources-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}-{secrets.token_hex(4)}.json"
    selection = {
        "coarse_candidate_count": len(ranked),
        "fine_candidate_count": len(fine_candidates),
        "selected_count": len(selected),
        "deep_analysis_during_selection": deep_analyzed,
        "deep_analysis_reused": deep_reused,
        "selection_signature_reused": same_selection,
        "selection_signature_migrated": migrated_signature,
        "bounded_to_reusable_cache": bool(same_selection and unchanged_library and len(reusable_ranked) >= desired_count),
        "fine_analysis_source": "persistent_index",
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "asset_manifest_schema_version": 3,
        "manifest_type": "asset_manifest",
        "provider": "local-library",
        "generated_at": _utc_now(),
        "library_root": str(Path(library_root).expanduser().resolve()),
        "library_index": str(index_path),
        "material_libraries": {"local_library": str(index_path)},
        "sync": sync.get("sync", {}),
        "selection": selection,
        "candidate_count": len(ranked),
        "candidate_pool_gate": sufficiency,
        "sufficiency": sufficiency,
        "search_rounds": [],
        "rejections": [],
        "sources": selected,
        "assets": selected,
        "selected": selected,
        "selected_assets": selected,
        "selected_count": len(selected),
    }
    _atomic_write_json(run_manifest, manifest)
    return {**manifest, "sources_manifest": str(run_manifest), "asset_manifest": str(run_manifest)}
