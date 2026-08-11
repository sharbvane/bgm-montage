#!/usr/bin/env python3
"""Task-driven YouTube acquisition for bgm-montage v1.3.3.

This module only implements material discovery, machine-wide reuse, download,
sampled-frame QA, and provider-compatible manifests.  The stable reference,
audio, timeline, edit, render, and output-QA stages remain outside this file.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import secrets
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from material_usage_policy import USAGE_MODES, apply_usage_policy, normalize_usage_mode
from pixabay_pipeline import _infer_shot_scale, _video_quality, evaluate_selected_sufficiency
from runtime_paths import RuntimePaths
from visual_intelligence import (
    asset_profile_fit,
    build_visual_style_profile,
    metadata_profile_fit,
    plan_visual_search_queries,
)


SCHEMA_VERSION = "1.3.3-youtube.2"
ASSET_MANIFEST_SCHEMA_VERSION = 3
INDEX_SCHEMA_VERSION = 1


class YouTubePipelineError(RuntimeError):
    pass


class InsufficientMaterialError(YouTubePipelineError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_slug(value: str, fallback: str = "youtube") -> str:
    value = re.sub(r"[^\w\u3400-\u9fff-]+", "_", str(value).strip(), flags=re.UNICODE)
    return (value.strip(" ._-")[:56] or fallback).rstrip(" ._")


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(4)}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)
    return path


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _run(command: Sequence[str], timeout: int = 180) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command), capture_output=True, text=True, encoding="utf-8",
        errors="replace", timeout=timeout, check=False,
    )


def _yt_dlp_executable() -> str:
    configured = os.environ.get("BGM_MONTAGE_YT_DLP", "").strip()
    candidates = [
        Path(configured).expanduser() if configured else None,
        Path(sys.executable).resolve().with_name("yt-dlp.exe" if os.name == "nt" else "yt-dlp"),
    ]
    for candidate in candidates:
        if candidate and candidate.is_file():
            return str(candidate)
    discovered = shutil.which("yt-dlp")
    if discovered:
        return discovered
    raise YouTubePipelineError(
        "yt-dlp is not installed in the active bgm-montage environment; install requirements.lock.txt"
    )


def build_youtube_query_plan(
    theme: str,
    style_profile: Mapping[str, Any] | None,
    audio_profile: Mapping[str, Any] | None,
    visual_cohesion_profile: str = "auto",
    priority_queries: Sequence[str] = (),
    *,
    max_rounds: int = 3,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Build provider-neutral, task-specific multi-round queries.

    Explicit ``--search-query`` values are placed first but never replace the
    shared visual-profile planner.  No theme vocabulary is hardcoded here.
    """

    explicit = "" if str(visual_cohesion_profile).lower() in {"", "auto", "none"} else str(visual_cohesion_profile)
    profile = build_visual_style_profile(theme, style_profile, audio_profile, explicit)
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for query in priority_queries:
        cleaned = " ".join(str(query).split()).strip()
        if cleaned and cleaned.casefold() not in seen:
            seen.add(cleaned.casefold())
            records.append({"query": cleaned, "intent": "user_priority", "expansion_level": -1, "priority": True})
    subject_terms = list((profile.get("terms") or {}).get("subject") or [])
    task_anchor = " ".join([*subject_terms[:8], "cinematic", "footage"]).strip()
    if task_anchor and task_anchor.casefold() not in seen:
        seen.add(task_anchor.casefold())
        records.append({"query": task_anchor[:100], "intent": "task_subject_anchor", "expansion_level": 0, "priority": False})
    for level in range(max(1, int(max_rounds))):
        for record in plan_visual_search_queries(profile, level):
            query = " ".join(str(record.get("query") or "").split()).strip()
            if not query or query.casefold() in seen:
                continue
            seen.add(query.casefold())
            records.append({**record, "query": query, "priority": False})
    return profile, records


def _search_query(query: str, count: int, yt_dlp: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    command = [
        yt_dlp, "--ignore-config", "--js-runtimes", "node", "--flat-playlist",
        "--dump-single-json", "--playlist-end", str(count), f"ytsearch{count}:{query}",
    ]
    process = _run(command, timeout=150)
    if process.returncode != 0:
        return [], {"query": query, "status": "failed", "error": (process.stderr or process.stdout or "yt-dlp search failed")[-1200:], "result_count": 0}
    try:
        payload = json.loads(process.stdout)
    except json.JSONDecodeError as exc:
        return [], {"query": query, "status": "failed", "error": str(exc), "result_count": 0}
    rows: list[dict[str, Any]] = []
    for rank, item in enumerate(payload.get("entries", []), start=1):
        if not isinstance(item, Mapping):
            continue
        video_id = str(item.get("id") or "").strip()
        if not re.fullmatch(r"[A-Za-z0-9_-]{6,20}", video_id):
            continue
        duration = item.get("duration")
        rows.append({
            "id": video_id, "youtube_id": video_id,
            "title": str(item.get("title") or ""),
            "description": str(item.get("description") or ""),
            "channel": str(item.get("channel") or item.get("uploader") or ""),
            "duration": float(duration) if isinstance(duration, (int, float)) else None,
            "page_url": f"https://www.youtube.com/watch?v={video_id}",
            "query": query, "query_rank": rank,
            "view_count": int(item.get("view_count") or 0),
            "upload_date": str(item.get("upload_date") or ""),
            "timestamp": item.get("timestamp"),
        })
    return rows, {"query": query, "status": "ok", "result_count": len(rows)}


_QUALITY_TERMS = ("4k", "uhd", "cinematic", "footage", "real time", "raw video", "wide view")
_REJECT_TERMS = (
    "news", "forecast", "interview", "podcast", "explained", "reaction", "gameplay",
    "gaming", "cgi", "animation", "ai generated", "livestream", "live coverage",
    "top 10", "slideshow", "tutorial", "review", "vlog",
)


def _metadata_score(candidate: Mapping[str, Any], visual_profile: Mapping[str, Any] | None = None) -> float:
    """Score relevance to the current task; rights vocabulary is deliberately neutral."""

    text = f"{candidate.get('title', '')} {candidate.get('description', '')} {candidate.get('channel', '')}".casefold()
    profile = visual_profile or {"terms": {}, "world_model": {}}
    fit = metadata_profile_fit(text, profile)
    score = 1.8 * float(fit.get("relevance") or 0.0) + 1.25 * float(fit.get("world_fit") or 0.0)
    score += 0.35 * min(2, sum(term in text for term in _QUALITY_TERMS))
    score -= 4.0 * sum(term in text for term in _REJECT_TERMS)
    score -= 0.65 if "time lapse" in text or "timelapse" in text or "hyperlapse" in text else 0.0
    if not fit.get("allowed", True):
        score -= 3.0
    duration = candidate.get("duration")
    if isinstance(duration, (int, float)):
        if 6.0 <= float(duration) <= 90.0:
            score += 1.0
        elif 90.0 < float(duration) <= 600.0:
            score += 0.35
        elif float(duration) < 4.0 or float(duration) > 1800.0:
            score -= 1.2
    rank = max(1, int(candidate.get("query_rank") or 1))
    score += max(0.0, 0.8 - 0.08 * (rank - 1))
    score += min(0.45, math.log10(max(1, int(candidate.get("view_count") or 0))) * 0.06)
    return round(score, 5)


def _merge_candidates(rows: Iterable[Mapping[str, Any]], visual_profile: Mapping[str, Any]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for row in rows:
        video_id = str(row.get("youtube_id") or row.get("id") or "")
        if not video_id:
            continue
        query = str(row.get("query") or "")
        if video_id not in merged:
            merged[video_id] = {**row, "matched_queries": [query] if query else []}
        else:
            item = merged[video_id]
            if query and query not in item["matched_queries"]:
                item["matched_queries"].append(query)
            item["query_rank"] = min(int(item.get("query_rank") or 999), int(row.get("query_rank") or 999))
    for item in merged.values():
        item["metadata_score"] = round(
            _metadata_score(item, visual_profile) + min(1.2, 0.22 * max(0, len(item["matched_queries"]) - 1)), 5
        )
    return sorted(merged.values(), key=lambda item: (-float(item["metadata_score"]), int(item.get("query_rank") or 999), str(item["youtube_id"])))


def _segment_window(duration: float | None) -> tuple[float, float] | None:
    if not isinstance(duration, (int, float)) or duration <= 85.0:
        return None
    clip_duration = 38.0
    start = min(max(2.0, duration * 0.18), max(0.0, duration - clip_duration - 2.0))
    return round(start, 3), round(min(duration, start + clip_duration), 3)


def _download_candidate(candidate: Mapping[str, Any], destination: Path, yt_dlp: str) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_stem = destination.parent / f".youtube_{candidate['youtube_id']}_{os.getpid()}"
    for old in destination.parent.glob(f"{temp_stem.name}.*"):
        old.unlink(missing_ok=True)
    command = [
        yt_dlp, "--ignore-config", "--js-runtimes", "node", "--no-playlist", "--no-part",
        "--force-overwrites", "--merge-output-format", "mp4", "--remux-video", "mp4",
        "-f", "bv*[height<=1080]+ba/b[height<=1080]", "-o", f"{temp_stem}.%(ext)s",
    ]
    window = _segment_window(candidate.get("duration"))
    if window:
        command.extend(["--download-sections", f"*{window[0]}-{window[1]}", "--force-keyframes-at-cuts"])
    command.append(str(candidate["page_url"]))
    process = _run(command, timeout=480)
    files = [path for path in destination.parent.glob(f"{temp_stem.name}.*") if path.is_file()]
    media = max(files, key=lambda path: path.stat().st_size) if files else None
    if process.returncode != 0 or media is None or media.stat().st_size < 64 * 1024:
        for path in files:
            path.unlink(missing_ok=True)
        raise YouTubePipelineError((process.stderr or process.stdout or "yt-dlp download failed")[-1600:])
    destination.unlink(missing_ok=True)
    os.replace(media, destination)
    for path in files:
        path.unlink(missing_ok=True)
    return {"section": {"start": window[0], "end": window[1]} if window else None}


def _asset_score(record: Mapping[str, Any], visual_profile: Mapping[str, Any]) -> tuple[float, dict[str, Any]]:
    quality = record.get("quality") if isinstance(record.get("quality"), Mapping) else {}
    visual = quality.get("visual_analysis") if isinstance(quality.get("visual_analysis"), Mapping) else {}
    fit = asset_profile_fit(record, visual_profile)
    score = (
        0.34 * float(quality.get("overall_score") or 0.0)
        + 0.18 * float(visual.get("aesthetic_score") or 0.0)
        + 0.16 * float(visual.get("cinematic_score") or 0.0)
        + 0.12 * float(visual.get("spatial_depth_score") or 0.0)
        + 0.08 * float(visual.get("visual_impact_score") or 0.0)
        + 0.12 * float(fit.get("total") or 0.0)
    )
    return round(score, 5), fit


def _asset_record(candidate: Mapping[str, Any], path: Path, quality: Mapping[str, Any], media: Mapping[str, Any], download_info: Mapping[str, Any], visual_profile: Mapping[str, Any], usage_mode: str) -> dict[str, Any]:
    title = str(candidate.get("title") or "")
    queries = [str(value) for value in candidate.get("matched_queries", [])]
    tags = " ".join([title, str(candidate.get("description") or ""), *queries])
    fingerprint = dict(media.get("fingerprint") or {})
    video_id = str(candidate["youtube_id"])
    record: dict[str, Any] = {
        "id": video_id, "asset_id": video_id, "youtube_id": video_id, "video_id": video_id, "provider": "youtube",
        "platform": "youtube", "author": str(candidate.get("channel") or ""),
        "channel": str(candidate.get("channel") or ""), "title": title,
        "description": str(candidate.get("description") or ""),
        "page_url": str(candidate["page_url"]), "url": str(candidate["page_url"]),
        "download_url": str(candidate["page_url"]), "tags": tags,
        "semantic_tags": queries, "search_query": queries[0] if queries else "", "search_queries": queries,
        "local_path": str(path.resolve()), "path": str(path.resolve()), "duration": float(media.get("duration_seconds") or 0.0),
        "duration_seconds": float(media.get("duration_seconds") or 0.0),
        "width": int(media.get("width") or 0), "height": int(media.get("height") or 0),
        "fps": float(media.get("fps") or 0.0), "ratio": media.get("ratio"),
        "dimensions": {"width": int(media.get("width") or 0), "height": int(media.get("height") or 0)},
        "shot_scale": _infer_shot_scale(tags), "scene_category": str(quality.get("scene_category") or "general"),
        "motion_score": float(quality.get("motion_score") or 0.0),
        "motion_label": str(quality.get("motion_type") or "unknown"),
        "motion_direction": str(quality.get("motion_direction") or "unknown"),
        "face_content_risk": float(quality.get("face_content_risk") or 0.0),
        "subject_profile": quality.get("subject_profile", {}), "quality": dict(quality),
        "visual_profile": quality.get("visual_features", {}), "fingerprint": fingerprint,
        "file_hash": str(fingerprint.get("sha256") or ""), "hash": str(fingerprint.get("sha256") or ""), "canonical_source_id": f"youtube:{video_id}",
        "available": True, "download_status": "downloaded_youtube", "download_section": download_info.get("section"),
        "metadata_score": float(candidate.get("metadata_score") or 0.0),
        "upload_date": str(candidate.get("upload_date") or ""), "upload_timestamp": candidate.get("timestamp"),
        "upload_metadata": {"date": str(candidate.get("upload_date") or ""), "timestamp": candidate.get("timestamp")},
        "usage_intervals": [], "actual_usage_intervals": [], "historical_usage_count": 0,
        "usage_history": [], "usage_mode": usage_mode,
        "attribution": {"platform": "YouTube", "video_id": video_id, "title": title, "channel": str(candidate.get("channel") or ""), "url": str(candidate["page_url"]), "traceability_only": True},
    }
    score, fit = _asset_score(record, visual_profile)
    record["visual_profile_fit"] = fit
    record["score"] = score
    record["last_evaluated_profile"] = visual_profile.get("profile_digest")
    return record


def _youtube_library(cache_dir: str | os.PathLike[str]) -> tuple[Path, Path]:
    runtime = RuntimePaths.build(cache_root=cache_dir)
    root = runtime.library_root / "youtube"
    return root, root / "asset_index.json"


def _load_library(index_path: Path) -> list[dict[str, Any]]:
    payload = _read_json(index_path, {})
    rows = payload.get("assets", []) if isinstance(payload, Mapping) else []
    result: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        path = Path(str(row.get("local_path") or ""))
        if row.get("youtube_id") and path.is_file() and path.stat().st_size >= 64 * 1024:
            normalized = dict(row)
            normalized.setdefault("platform", "youtube")
            normalized.setdefault("video_id", str(row.get("youtube_id")))
            normalized.setdefault("url", str(row.get("page_url") or ""))
            normalized.setdefault("path", str(path.resolve()))
            normalized.setdefault("dimensions", {"width": int(row.get("width") or 0), "height": int(row.get("height") or 0)})
            normalized.setdefault("upload_metadata", {"date": str(row.get("upload_date") or ""), "timestamp": row.get("upload_timestamp")})
            normalized.setdefault("hash", str(row.get("file_hash") or (row.get("fingerprint") or {}).get("sha256") or ""))
            result.append(normalized)
    return result


def _persist_library(index_path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    merged: dict[str, dict[str, Any]] = {}
    for row in [*_load_library(index_path), *rows]:
        key = str(row.get("youtube_id") or row.get("file_hash") or "")
        if key:
            merged[key] = dict(row)
    _write_json(index_path, {"schema_version": INDEX_SCHEMA_VERSION, "updated_at": _utc_now(), "assets": list(merged.values())})


def _reuse_record(record: Mapping[str, Any], destination: Path, visual_profile: Mapping[str, Any], usage_mode: str) -> dict[str, Any] | None:
    source = Path(str(record.get("local_path") or ""))
    if not source.is_file() or source.stat().st_size < 64 * 1024:
        return None
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)
    reused = dict(record)
    reused["local_path"] = str(destination.resolve())
    reused["path"] = str(destination.resolve())
    reused["download_status"] = "reused_global_youtube"
    reused["reuse_mode"] = "global_index"
    reused["usage_mode"] = usage_mode
    score, fit = _asset_score(reused, visual_profile)
    reused["score"], reused["visual_profile_fit"] = score, fit
    reused["last_evaluated_profile"] = visual_profile.get("profile_digest")
    return reused


def _candidate_pool_gate(available: int, desired_count: int, timeline_plan: Mapping[str, Any] | None, multiplier: int) -> dict[str, Any]:
    slots = len((timeline_plan or {}).get("slots", []))
    required = max(desired_count, slots * max(1, int(multiplier)))
    failures = [] if available >= required else [f"metadata candidates {available} < {required} for {slots} planned slots"]
    return {
        "evaluated": True, "passed": not failures, "planned_slot_count": slots,
        "candidate_pool_multiplier": int(multiplier), "required_candidate_count": required,
        "available_candidate_count": int(available), "candidate_to_target_ratio": round(available / max(1, desired_count), 4),
        "failures": failures,
    }


def _selection_sufficiency(selected: Sequence[Mapping[str, Any]], desired_count: int, target_duration: float | None, gate: Mapping[str, Any], visual_profile: Mapping[str, Any]) -> dict[str, Any]:
    base = evaluate_selected_sufficiency(selected, desired_count, target_duration)
    failures = list(base.get("failures", []))
    if len(selected) < desired_count:
        failures.insert(0, f"selected assets {len(selected)} < requested {desired_count}")
    failures.extend(str(value) for value in gate.get("failures", []))
    qualities = [float((item.get("quality") or {}).get("overall_score") or 0.0) for item in selected]
    fits = [float((item.get("visual_profile_fit") or {}).get("total") or 0.0) for item in selected]
    average_quality = sum(qualities) / len(qualities) if qualities else 0.0
    average_fit = sum(fits) / len(fits) if fits else 0.0
    quality_floor = float((visual_profile.get("quality") or {}).get("aesthetic_floor") or 0.42) * 0.72
    if selected and average_quality < quality_floor:
        failures.append(f"average quality {average_quality:.4f} < {quality_floor:.4f}")
    if selected and average_fit < 0.30:
        failures.append(f"average style fit {average_fit:.4f} < 0.3000")
    channel_counts = Counter(str(item.get("channel") or item.get("author") or "unknown") for item in selected)
    max_share = max(channel_counts.values(), default=0) / max(1, len(selected))
    if len(selected) >= 4 and max_share > 0.60:
        failures.append(f"maximum channel share {max_share:.4f} > 0.6000")
    return {
        **base, "passed": not failures, "failures": list(dict.fromkeys(failures)),
        "usable_count": len(selected), "average_quality": round(average_quality, 4),
        "average_style_fit": round(average_fit, 4), "maximum_channel_share": round(max_share, 4),
        "candidate_pool_passed": bool(gate.get("passed")),
    }


def run_youtube_pipeline(
    theme: str, style_profile: Mapping[str, Any] | None, audio_profile: Mapping[str, Any] | None,
    material_root: str | os.PathLike[str], cache_dir: str | os.PathLike[str], desired_count: int,
    aspect_ratio: str, *, min_resolution: tuple[int, int] = (1280, 720), target_duration: float | None = None,
    timeline_plan: Mapping[str, Any] | None = None, candidate_pool_multiplier: int = 6,
    priority_queries: Sequence[str] = (), visual_cohesion_profile: str = "auto",
    excluded_youtube_ids: Sequence[str] = (), results_per_query: int = 8,
    max_download_candidates: int = 36, max_search_rounds: int = 3,
    usage_mode: str = "local_evaluation", allow_insufficient: bool = False,
) -> dict[str, Any]:
    if int(desired_count) < 1:
        raise ValueError("desired_count must be positive")
    usage_mode = normalize_usage_mode(usage_mode)
    visual_profile, query_plan = build_youtube_query_plan(
        theme, style_profile, audio_profile, visual_cohesion_profile, priority_queries, max_rounds=max_search_rounds
    )
    cache_root = Path(cache_dir).expanduser().resolve()
    material_dir = Path(material_root).expanduser().resolve() / _safe_slug(theme, "youtube")
    project_videos = cache_root / "videos"
    run_manifests = cache_root / "run_manifests" / _safe_slug(theme, "youtube")
    library_root, index_path = _youtube_library(cache_dir)
    global_videos = library_root / "videos"
    for directory in (material_dir, project_videos, run_manifests, global_videos):
        directory.mkdir(parents=True, exist_ok=True)

    excluded = {str(value) for value in excluded_youtube_ids}
    library = [row for row in _load_library(index_path) if str(row.get("youtube_id")) not in excluded]
    if library:
        _persist_library(index_path, library)
    local_candidates: list[dict[str, Any]] = []
    for row in library:
        destination = material_dir / f"youtube_{row['youtube_id']}.mp4"
        reused = _reuse_record(row, destination, visual_profile, usage_mode)
        if reused:
            local_candidates.append(reused)
    local_candidates.sort(key=lambda item: (-float(item.get("score") or 0.0), str(item.get("youtube_id"))))
    quality_floor = float((visual_profile.get("quality") or {}).get("aesthetic_floor") or 0.42) * 0.72
    eligible_local = [
        item for item in local_candidates
        if bool((item.get("quality") or {}).get("passed", True))
        and float((item.get("quality") or {}).get("overall_score") or 0.0) >= quality_floor
        and float((item.get("visual_profile_fit") or {}).get("total") or 0.0) >= 0.30
    ]
    gate = _candidate_pool_gate(len(eligible_local), desired_count, timeline_plan, candidate_pool_multiplier)
    local_selected = eligible_local[:desired_count]
    local_sufficiency = _selection_sufficiency(local_selected, desired_count, target_duration, gate, visual_profile)

    search_rows: list[dict[str, Any]] = []
    search_rounds: list[dict[str, Any]] = []
    ranked: list[dict[str, Any]] = []
    yt_dlp: str | None = None
    if not local_sufficiency["passed"]:
        yt_dlp = _yt_dlp_executable()
        for query_record in query_plan:
            rows, report = _search_query(str(query_record["query"]), max(1, int(results_per_query)), yt_dlp)
            report.update({"intent": query_record.get("intent"), "expansion_level": query_record.get("expansion_level"), "priority": query_record.get("priority", False)})
            search_rows.extend(rows)
            search_rounds.append(report)
            ranked = _merge_candidates(search_rows, visual_profile)
            eligible_ranked = [
                item for item in ranked
                if str(item.get("youtube_id")) not in excluded
                and float(item.get("metadata_score") or 0.0) >= 0.2
                and not any(term in f"{item.get('title', '')} {item.get('description', '')} {item.get('channel', '')}".casefold() for term in _REJECT_TERMS)
            ]
            gate = _candidate_pool_gate(len(eligible_local) + len(eligible_ranked), desired_count, timeline_plan, candidate_pool_multiplier)
            if gate["passed"]:
                break

    selected_by_id = {str(item.get("youtube_id")): item for item in local_selected}
    candidate_log: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    attempted_downloads = 0
    new_library_rows: list[dict[str, Any]] = []
    channel_counts = Counter(str(item.get("channel") or "unknown").casefold() for item in selected_by_id.values())
    eligible_ranked = [
        item for item in ranked
        if str(item.get("youtube_id")) not in excluded
        and float(item.get("metadata_score") or 0.0) >= 0.2
        and not any(term in f"{item.get('title', '')} {item.get('description', '')} {item.get('channel', '')}".casefold() for term in _REJECT_TERMS)
    ]
    for candidate in eligible_ranked:
        video_id = str(candidate["youtube_id"])
        public = {key: candidate.get(key) for key in ("youtube_id", "title", "channel", "duration", "page_url", "matched_queries", "query_rank", "metadata_score")}
        if video_id in excluded or video_id in selected_by_id:
            continue
        text = f"{candidate.get('title', '')} {candidate.get('description', '')} {candidate.get('channel', '')}".casefold()
        reasons = [f"metadata avoid term: {term}" for term in _REJECT_TERMS if term in text]
        if reasons or float(candidate.get("metadata_score") or 0.0) < 0.2:
            public.update({"decision": "rejected", "stage": "metadata", "reasons": reasons or ["low task-specific metadata relevance"]})
            candidate_log.append(public); rejections.append(public); continue
        channel = str(candidate.get("channel") or "unknown").casefold()
        if channel_counts[channel] >= 3:
            public.update({"decision": "rejected", "stage": "source_diversity", "reasons": ["channel candidate cap reached"]})
            candidate_log.append(public); rejections.append(public); continue
        if attempted_downloads >= int(max_download_candidates):
            break
        assert yt_dlp is not None
        cached = global_videos / f"youtube_{video_id}.mp4"
        destination = material_dir / f"youtube_{video_id}.mp4"
        try:
            attempted_downloads += 1
            download_info = _download_candidate(candidate, cached, yt_dlp)
            if not destination.exists():
                try:
                    os.link(cached, destination)
                except OSError:
                    shutil.copy2(cached, destination)
            tags = " ".join([str(candidate.get("title") or ""), str(candidate.get("description") or ""), *candidate.get("matched_queries", [])])
            quality, media = _video_quality(destination, {**(style_profile or {}), "visual_style_profile": visual_profile}, min_resolution, tags=tags, human_focused=False)
            if not quality.get("passed"):
                reasons = list(quality.get("rejection_reasons") or ["post-download quality gate failed"])
                public.update({"decision": "rejected", "stage": "post_download_qa", "reasons": reasons, "quality": quality})
                candidate_log.append(public); rejections.append(public); continue
            record = _asset_record(candidate, destination, quality, media, download_info, visual_profile, usage_mode)
            library_record = {**record, "local_path": str(cached.resolve()), "path": str(cached.resolve())}
            new_library_rows.append(library_record)
            selected_by_id[video_id] = record
            channel_counts[channel] += 1
            public.update({"decision": "selected", "stage": "post_download_qa", "local_path": str(destination), "quality": quality, "score": record["score"]})
            candidate_log.append(public)
            if len(selected_by_id) >= desired_count:
                break
        except (OSError, subprocess.SubprocessError, YouTubePipelineError) as exc:
            public.update({"decision": "rejected", "stage": "download_or_decode", "reasons": [str(exc)[-1400:]]})
            candidate_log.append(public); rejections.append(public)

    if new_library_rows:
        _persist_library(index_path, new_library_rows)
    selected = sorted(selected_by_id.values(), key=lambda item: (-float(item.get("score") or 0.0), str(item.get("youtube_id"))))[:desired_count]
    gate = _candidate_pool_gate(len(eligible_local) + len(eligible_ranked), desired_count, timeline_plan, candidate_pool_multiplier)
    sufficiency = _selection_sufficiency(selected, desired_count, target_duration, gate, visual_profile)
    status = "ok" if sufficiency["passed"] else "insufficient_material"
    manifest = apply_usage_policy({
        "schema_version": SCHEMA_VERSION, "asset_manifest_schema_version": ASSET_MANIFEST_SCHEMA_VERSION,
        "manifest_type": "asset_manifest", "provider": "youtube", "generated_at": _utc_now(),
        "status": status, "theme": theme, "theme_directory": str(material_dir),
        "requested": {"desired_count": desired_count, "aspect_ratio": aspect_ratio, "min_resolution": {"width": min_resolution[0], "height": min_resolution[1]}, "candidate_pool_multiplier": candidate_pool_multiplier, "results_per_query": results_per_query, "max_download_candidates": max_download_candidates, "priority_queries": list(priority_queries), "excluded_youtube_ids": sorted(excluded)},
        "query_plan": query_plan, "timeline_plan": {"provided": bool(timeline_plan), "slot_count": len((timeline_plan or {}).get("slots", []))},
        "candidate_pool_gate": gate, "visual_style_profile": visual_profile,
        "cache_layout": {"youtube_root": str(cache_root), "global_library": str(library_root), "asset_index": str(index_path), "videos": str(global_videos)},
        "search_rounds": search_rounds, "candidate_count": len(eligible_local) + len(eligible_ranked),
        "network_candidate_count": len(eligible_ranked), "local_indexed_candidate_count": len(library),
        "local_reusable_candidate_count": len(eligible_local),
        "download_candidate_count": attempted_downloads, "candidate_log": candidate_log, "rejections": rejections,
        "sources": selected, "assets": selected, "selected_count": len(selected), "sufficiency": sufficiency,
        "reuse_summary": dict(Counter(str(item.get("reuse_mode") or "new_download") for item in selected)),
        "heuristic_notice": "Task relevance and sampled-frame visual quality are estimates; final montage QA remains authoritative.",
    }, usage_mode)
    snapshot = run_manifests / f"sources-{int(datetime.now().timestamp() * 1000)}-{secrets.token_hex(8)}.json"
    _write_json(snapshot, manifest)
    result = {**manifest, "sources_manifest": str(snapshot), "asset_manifest": str(snapshot), "asset_manifest_path": str(snapshot), "selected": selected, "selected_assets": selected, "priority_queries": list(priority_queries), "target_duration": target_duration, "global_asset_index": str(index_path)}
    if not sufficiency["passed"] and not allow_insufficient:
        raise InsufficientMaterialError("YouTube material sufficiency gate failed: " + "; ".join(sufficiency["failures"]) + f". Manifest: {snapshot}")
    return result


def _load_object(path: str) -> dict[str, Any]:
    payload = _read_json(Path(path).expanduser().resolve(), {})
    if not isinstance(payload, dict):
        raise YouTubePipelineError(f"Expected JSON object: {path}")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--theme", required=True); parser.add_argument("--style-profile", required=True)
    parser.add_argument("--audio-profile", required=True); parser.add_argument("--timeline-plan", required=True)
    parser.add_argument("--material-root", required=True); parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--desired-count", type=int, required=True); parser.add_argument("--aspect-ratio", default="16:9")
    parser.add_argument("--min-resolution", default="1280x720"); parser.add_argument("--target-duration", type=float)
    parser.add_argument("--candidate-pool-multiplier", type=int, default=6)
    parser.add_argument("--search-query", action="append", dest="queries", default=[], help="Optional high-priority query; dynamic queries are always generated.")
    parser.add_argument("--visual-style", default="auto"); parser.add_argument("--exclude-youtube-id", action="append", default=[])
    parser.add_argument("--results-per-query", type=int, default=8); parser.add_argument("--max-download-candidates", type=int, default=36)
    parser.add_argument("--max-search-rounds", type=int, default=3); parser.add_argument("--usage-mode", choices=USAGE_MODES, default="local_evaluation")
    parser.add_argument("--allow-insufficient", action="store_true"); parser.add_argument("--result-json")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    match = re.fullmatch(r"(\d+)x(\d+)", args.min_resolution.lower())
    if not match:
        raise SystemExit("--min-resolution must be WIDTHxHEIGHT")
    result = run_youtube_pipeline(
        args.theme, _load_object(args.style_profile), _load_object(args.audio_profile), args.material_root,
        args.cache_dir, args.desired_count, args.aspect_ratio,
        min_resolution=(int(match.group(1)), int(match.group(2))), target_duration=args.target_duration,
        timeline_plan=_load_object(args.timeline_plan), candidate_pool_multiplier=args.candidate_pool_multiplier,
        priority_queries=args.queries, visual_cohesion_profile=args.visual_style,
        excluded_youtube_ids=args.exclude_youtube_id, results_per_query=args.results_per_query,
        max_download_candidates=args.max_download_candidates, max_search_rounds=args.max_search_rounds,
        usage_mode=args.usage_mode, allow_insufficient=args.allow_insufficient,
    )
    if args.result_json:
        _write_json(Path(args.result_json).expanduser().resolve(), result)
    print(json.dumps({"status": result["status"], "provider": "youtube", "candidate_count": result["candidate_count"], "download_candidate_count": result["download_candidate_count"], "selected_count": result["selected_count"], "sources_manifest": result["sources_manifest"]}, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
