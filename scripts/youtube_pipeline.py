#!/usr/bin/env python3
"""YouTube-first acquisition adapter for bgm-montage v1.3.

The adapter deliberately stops at the existing asset-manifest boundary.  It
uses yt-dlp for search/download, then reuses the v1.3 sampled-frame quality and
visual-feature analyzer so the timing, edit, render, and QA core remain
unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
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

from pixabay_pipeline import _infer_shot_scale, _video_quality
from material_usage_policy import USAGE_MODES, apply_usage_policy, normalize_usage_mode
from visual_intelligence import asset_profile_fit, build_visual_style_profile


SCHEMA_VERSION = "1.3-youtube.1"
ASSET_MANIFEST_SCHEMA_VERSION = 2


class YouTubePipelineError(RuntimeError):
    pass


class InsufficientMaterialError(YouTubePipelineError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_slug(value: str, fallback: str = "youtube") -> str:
    value = re.sub(r"[^\w\u3400-\u9fff-]+", "_", str(value).strip(), flags=re.UNICODE)
    value = value.strip(" ._-")
    return (value[:56] or fallback).rstrip(" ._")


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
        list(command),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
    )


def _yt_dlp_executable() -> str:
    value = shutil.which("yt-dlp")
    if not value:
        raise YouTubePipelineError("yt-dlp is not available on PATH")
    return value


def _search_query(query: str, count: int, yt_dlp: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    command = [
        yt_dlp,
        "--ignore-config",
        "--js-runtimes",
        "node",
        "--flat-playlist",
        "--dump-single-json",
        "--playlist-end",
        str(count),
        f"ytsearch{count}:{query}",
    ]
    process = _run(command, timeout=150)
    if process.returncode != 0:
        return [], {
            "query": query,
            "status": "failed",
            "error": (process.stderr or process.stdout or "yt-dlp search failed")[-1200:],
            "result_count": 0,
        }
    try:
        payload = json.loads(process.stdout)
    except json.JSONDecodeError as exc:
        return [], {"query": query, "status": "failed", "error": str(exc), "result_count": 0}
    entries = [item for item in payload.get("entries", []) if isinstance(item, Mapping)]
    public = []
    for rank, item in enumerate(entries, start=1):
        video_id = str(item.get("id") or "").strip()
        if not re.fullmatch(r"[A-Za-z0-9_-]{6,20}", video_id):
            continue
        duration = item.get("duration")
        public.append(
            {
                "id": video_id,
                "youtube_id": video_id,
                "title": str(item.get("title") or ""),
                "channel": str(item.get("channel") or item.get("uploader") or ""),
                "duration": float(duration) if isinstance(duration, (int, float)) else None,
                "url": str(item.get("url") or f"https://www.youtube.com/watch?v={video_id}"),
                "page_url": f"https://www.youtube.com/watch?v={video_id}",
                "query": query,
                "query_rank": rank,
                "view_count": int(item.get("view_count") or 0),
            }
        )
    return public, {"query": query, "status": "ok", "result_count": len(public)}


_CORE_TERMS = (
    "shelf cloud", "supercell", "storm front", "wall cloud", "gust front", "derecho",
    "rain wall", "severe storm", "massive storm", "approaching storm", "ominous storm",
    "extreme wind", "violent storm", "storm rolling", "storm approaching",
)
_ENVIRONMENT_TERMS = (
    "field", "farm", "farmland", "countryside", "rural", "road", "highway", "prairie",
    "grassland", "forest", "trees", "city", "skyline", "mountain", "valley", "coast",
    "ocean", "waves", "landscape", "horizon",
)
_IMPACT_TERMS = (
    "massive", "giant", "extreme", "violent", "ominous", "severe", "apocalyptic",
    "monster", "dramatic", "powerful", "dark", "turbulent", "bending", "rolling",
)
_QUALITY_TERMS = ("4k", "uhd", "cinematic", "footage", "real time", "stock video")
_REJECT_TERMS = (
    "news", "forecast", "interview", "podcast", "explained", "documentary", "trailer",
    "game", "gaming", "cgi", "animation", "ai generated", "reaction", "livestream",
    "live coverage", "compilation", "top 10", "shorts compilation",
)


def _metadata_score(candidate: Mapping[str, Any]) -> float:
    text = f"{candidate.get('title', '')} {candidate.get('channel', '')}".casefold()
    score = 0.0
    score += 2.4 * sum(term in text for term in _CORE_TERMS)
    score += 0.75 * min(3, sum(term in text for term in _ENVIRONMENT_TERMS))
    score += 0.55 * min(3, sum(term in text for term in _IMPACT_TERMS))
    score += 0.35 * min(2, sum(term in text for term in _QUALITY_TERMS))
    score -= 4.0 * sum(term in text for term in _REJECT_TERMS)
    score -= 0.6 if "time lapse" in text or "timelapse" in text or "hyperlapse" in text else 0.0
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


def _merge_candidates(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for row in rows:
        video_id = str(row.get("youtube_id") or row.get("id") or "")
        if not video_id:
            continue
        if video_id not in merged:
            merged[video_id] = {**row, "matched_queries": [str(row.get("query") or "")]}
        else:
            item = merged[video_id]
            query = str(row.get("query") or "")
            if query and query not in item["matched_queries"]:
                item["matched_queries"].append(query)
            item["query_rank"] = min(int(item.get("query_rank") or 999), int(row.get("query_rank") or 999))
    for item in merged.values():
        item["metadata_score"] = _metadata_score(item) + min(1.2, 0.22 * (len(item["matched_queries"]) - 1))
    return sorted(
        merged.values(),
        key=lambda item: (-float(item["metadata_score"]), int(item.get("query_rank") or 999), str(item["youtube_id"])),
    )


def _segment_window(duration: float | None) -> tuple[float, float] | None:
    if not isinstance(duration, (int, float)) or duration <= 0 or duration <= 85.0:
        return None
    clip_duration = 38.0 if duration >= 50.0 else max(12.0, duration - 2.0)
    start = min(max(2.0, duration * 0.18), max(0.0, duration - clip_duration - 2.0))
    return round(start, 3), round(min(duration, start + clip_duration), 3)


def _download_candidate(candidate: Mapping[str, Any], destination: Path, yt_dlp: str) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_stem = destination.parent / f".youtube_{candidate['youtube_id']}_{os.getpid()}"
    for old in destination.parent.glob(f"{temp_stem.name}.*"):
        old.unlink(missing_ok=True)
    command = [
        yt_dlp,
        "--ignore-config",
        "--js-runtimes",
        "node",
        "--no-playlist",
        "--no-part",
        "--force-overwrites",
        "--merge-output-format",
        "mp4",
        "--remux-video",
        "mp4",
        "-f",
        "bv*[height<=1080]+ba/b[height<=1080]",
        "-o",
        f"{temp_stem}.%(ext)s",
    ]
    window = _segment_window(candidate.get("duration"))
    if window is not None:
        command.extend(["--download-sections", f"*{window[0]}-{window[1]}", "--force-keyframes-at-cuts"])
    command.append(str(candidate["page_url"]))
    process = _run(command, timeout=480)
    files = [path for path in destination.parent.glob(f"{temp_stem.name}.*") if path.is_file()]
    media = max(files, key=lambda path: path.stat().st_size) if files else None
    if process.returncode != 0 or media is None or media.stat().st_size < 64 * 1024:
        for path in files:
            path.unlink(missing_ok=True)
        raise YouTubePipelineError((process.stderr or process.stdout or "yt-dlp download failed")[-1600:])
    if destination.exists():
        destination.unlink()
    os.replace(media, destination)
    for path in files:
        if path.exists():
            path.unlink(missing_ok=True)
    return {"section": {"start": window[0], "end": window[1]} if window else None}


def _asset_record(
    candidate: Mapping[str, Any],
    path: Path,
    quality: Mapping[str, Any],
    media: Mapping[str, Any],
    download_info: Mapping[str, Any],
    visual_profile: Mapping[str, Any],
    usage_mode: str,
) -> dict[str, Any]:
    title = str(candidate.get("title") or "")
    tags = " ".join([title, *[str(value) for value in candidate.get("matched_queries", [])]])
    fingerprint = dict(media.get("fingerprint") or {})
    record = {
        "id": str(candidate["youtube_id"]),
        "asset_id": str(candidate["youtube_id"]),
        "youtube_id": str(candidate["youtube_id"]),
        "provider": "youtube",
        "author": str(candidate.get("channel") or ""),
        "channel": str(candidate.get("channel") or ""),
        "title": title,
        "page_url": str(candidate["page_url"]),
        "download_url": str(candidate["page_url"]),
        "tags": tags,
        "search_query": str((candidate.get("matched_queries") or [""])[0]),
        "search_queries": list(candidate.get("matched_queries") or []),
        "local_path": str(path.resolve()),
        "duration": float(media.get("duration_seconds") or 0.0),
        "duration_seconds": float(media.get("duration_seconds") or 0.0),
        "width": int(media.get("width") or 0),
        "height": int(media.get("height") or 0),
        "fps": float(media.get("fps") or 0.0),
        "ratio": media.get("ratio"),
        "shot_scale": _infer_shot_scale(tags),
        "scene_category": str(quality.get("scene_category") or "general"),
        "motion_score": float(quality.get("motion_score") or 0.0),
        "motion_label": str(quality.get("motion_type") or "unknown"),
        "motion_direction": str(quality.get("motion_direction") or "unknown"),
        "face_content_risk": float(quality.get("face_content_risk") or 0.0),
        "subject_profile": quality.get("subject_profile", {}),
        "quality": dict(quality),
        "fingerprint": fingerprint,
        "file_hash": str(fingerprint.get("sha256") or ""),
        "canonical_source_id": f"youtube:{candidate['youtube_id']}",
        "available": True,
        "download_status": "downloaded_youtube",
        "download_section": download_info.get("section"),
        "metadata_score": float(candidate.get("metadata_score") or 0.0),
        "usage_intervals": [],
        "actual_usage_intervals": [],
        "historical_usage_count": 0,
        "usage_history": [],
        "usage_mode": usage_mode,
        "attribution": {
            "platform": "YouTube",
            "video_id": str(candidate["youtube_id"]),
            "title": title,
            "channel": str(candidate.get("channel") or ""),
            "url": str(candidate["page_url"]),
            "traceability_only": True,
        },
    }
    fit = asset_profile_fit(record, visual_profile)
    record["visual_profile_fit"] = fit
    record["score"] = round(
        0.38 * float(quality.get("overall_score") or 0.0)
        + 0.24 * float((quality.get("visual_analysis") or {}).get("aesthetic_score") or 0.0)
        + 0.18 * float((quality.get("visual_analysis") or {}).get("cinematic_score") or 0.0)
        + 0.12 * float(fit.get("total") or 0.0)
        + 0.08 * min(1.0, max(0.0, float(candidate.get("metadata_score") or 0.0) / 8.0)),
        5,
    )
    return record


def run_youtube_pipeline(
    theme: str,
    style_profile: Mapping[str, Any] | None,
    audio_profile: Mapping[str, Any] | None,
    material_root: str | os.PathLike[str],
    cache_dir: str | os.PathLike[str],
    desired_count: int,
    aspect_ratio: str,
    *,
    min_resolution: tuple[int, int] = (1280, 720),
    target_duration: float | None = None,
    timeline_plan: Mapping[str, Any] | None = None,
    candidate_pool_multiplier: int = 6,
    priority_queries: Sequence[str] = (),
    visual_cohesion_profile: str = "auto",
    excluded_youtube_ids: Sequence[str] = (),
    results_per_query: int = 8,
    max_download_candidates: int = 36,
    usage_mode: str = "local_evaluation",
) -> dict[str, Any]:
    if desired_count < 1:
        raise ValueError("desired_count must be positive")
    usage_mode = normalize_usage_mode(usage_mode)
    queries = [str(value).strip() for value in priority_queries if str(value).strip()]
    if not queries:
        raise YouTubePipelineError("YouTube-first mode requires at least one --search-query")
    yt_dlp = _yt_dlp_executable()
    cache_root = Path(cache_dir).expanduser().resolve()
    material_dir = Path(material_root).expanduser().resolve() / _safe_slug(theme, "youtube_storm")
    videos_dir = cache_root / "videos"
    run_manifests = cache_root / "run_manifests" / _safe_slug(theme, "youtube")
    videos_dir.mkdir(parents=True, exist_ok=True)
    material_dir.mkdir(parents=True, exist_ok=True)
    explicit = "" if str(visual_cohesion_profile).lower() in {"", "auto", "none"} else str(visual_cohesion_profile)
    visual_profile = build_visual_style_profile(theme, style_profile, audio_profile, explicit)

    search_rows: list[dict[str, Any]] = []
    search_rounds: list[dict[str, Any]] = []
    for query in queries:
        rows, report = _search_query(query, max(1, int(results_per_query)), yt_dlp)
        search_rows.extend(rows)
        search_rounds.append(report)
    ranked = _merge_candidates(search_rows)
    excluded = {str(value) for value in excluded_youtube_ids}
    candidate_pool_required = max(desired_count, len((timeline_plan or {}).get("slots", [])) * max(1, candidate_pool_multiplier))
    candidate_pool_gate = {
        "passed": len(ranked) >= candidate_pool_required,
        "required_candidate_count": candidate_pool_required,
        "available_candidate_count": len(ranked),
        "failures": [] if len(ranked) >= candidate_pool_required else [f"metadata candidates {len(ranked)} < {candidate_pool_required}"],
    }

    selected: list[dict[str, Any]] = []
    candidate_log: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    channel_counts: Counter[str] = Counter()
    attempted_downloads = 0
    for candidate in ranked:
        public = {key: candidate.get(key) for key in ("youtube_id", "title", "channel", "duration", "page_url", "matched_queries", "query_rank", "metadata_score")}
        if candidate["youtube_id"] in excluded:
            public.update({"decision": "rejected", "stage": "explicit_exclusion", "reasons": ["explicitly excluded after visual review"]})
            candidate_log.append(public)
            rejections.append(public)
            continue
        text = f"{candidate.get('title', '')} {candidate.get('channel', '')}".casefold()
        metadata_reasons = [f"metadata avoid term: {term}" for term in _REJECT_TERMS if term in text]
        if metadata_reasons or float(candidate.get("metadata_score") or 0.0) < 0.2:
            public.update({"decision": "rejected", "stage": "metadata", "reasons": metadata_reasons or ["low metadata relevance score"]})
            candidate_log.append(public)
            rejections.append(public)
            continue
        channel = str(candidate.get("channel") or "unknown").casefold()
        if channel_counts[channel] >= 3:
            public.update({"decision": "rejected", "stage": "source_diversity", "reasons": ["channel candidate cap reached"]})
            candidate_log.append(public)
            rejections.append(public)
            continue
        if attempted_downloads >= max_download_candidates:
            public.update({"decision": "not_downloaded", "stage": "download_budget", "reasons": ["download candidate budget reached"]})
            candidate_log.append(public)
            continue
        video_id = str(candidate["youtube_id"])
        cached = videos_dir / f"youtube_{video_id}.mp4"
        destination = material_dir / f"youtube_{video_id}.mp4"
        try:
            download_info: dict[str, Any] = {"section": None}
            if not cached.is_file() or cached.stat().st_size < 64 * 1024:
                attempted_downloads += 1
                download_info = _download_candidate(candidate, cached, yt_dlp)
            else:
                attempted_downloads += 1
            if destination.exists():
                if destination.stat().st_size != cached.stat().st_size:
                    destination.unlink()
                else:
                    pass
            if not destination.exists():
                try:
                    os.link(cached, destination)
                except OSError:
                    shutil.copy2(cached, destination)
            tags = " ".join([str(candidate.get("title") or ""), *candidate.get("matched_queries", [])])
            quality, media = _video_quality(destination, {**(style_profile or {}), "visual_style_profile": visual_profile}, min_resolution, tags=tags, human_focused=False)
            if not quality.get("passed"):
                reasons = list(quality.get("rejection_reasons") or ["post-download quality gate failed"])
                public.update({"decision": "rejected", "stage": "post_download_qa", "reasons": reasons, "quality": quality, "local_path": str(destination)})
                candidate_log.append(public)
                rejections.append(public)
                continue
            record = _asset_record(candidate, destination, quality, media, download_info, visual_profile, usage_mode)
            selected.append(record)
            channel_counts[channel] += 1
            public.update({"decision": "selected", "stage": "post_download_qa", "local_path": str(destination), "quality": quality, "score": record["score"]})
            candidate_log.append(public)
            if len(selected) >= desired_count:
                break
        except (OSError, subprocess.SubprocessError, YouTubePipelineError) as exc:
            public.update({"decision": "rejected", "stage": "download_or_decode", "reasons": [str(exc)[-1400:]]})
            candidate_log.append(public)
            rejections.append(public)

    selected.sort(key=lambda item: (-float(item.get("score") or 0.0), str(item.get("youtube_id") or "")))
    sufficiency_failures: list[str] = []
    if len(selected) < desired_count:
        sufficiency_failures.append(f"selected assets {len(selected)} < requested {desired_count}")
    if len({item.get("canonical_source_id") for item in selected}) < desired_count:
        sufficiency_failures.append("selected sources are not unique")
    status = "ok" if not sufficiency_failures else "insufficient_material"
    manifest = apply_usage_policy({
        "schema_version": SCHEMA_VERSION,
        "asset_manifest_schema_version": ASSET_MANIFEST_SCHEMA_VERSION,
        "manifest_type": "asset_manifest",
        "provider": "youtube",
        "generated_at": _utc_now(),
        "status": status,
        "theme": theme,
        "theme_directory": str(material_dir),
        "requested": {
            "desired_count": desired_count,
            "aspect_ratio": aspect_ratio,
            "min_resolution": {"width": min_resolution[0], "height": min_resolution[1]},
            "candidate_pool_multiplier": candidate_pool_multiplier,
            "results_per_query": results_per_query,
            "max_download_candidates": max_download_candidates,
            "priority_queries": queries,
            "excluded_youtube_ids": sorted(excluded),
        },
        "timeline_plan": {"provided": bool(timeline_plan), "slot_count": len((timeline_plan or {}).get("slots", []))},
        "candidate_pool_gate": candidate_pool_gate,
        "visual_style_profile": visual_profile,
        "cache_layout": {"youtube_root": str(cache_root), "videos": str(videos_dir)},
        "search_rounds": search_rounds,
        "candidate_count": len(ranked),
        "download_candidate_count": attempted_downloads,
        "candidate_log": candidate_log,
        "rejections": rejections,
        "sources": selected,
        "assets": selected,
        "selected_count": len(selected),
        "sufficiency": {"passed": not sufficiency_failures, "failures": sufficiency_failures},
        "reuse_summary": dict(Counter("cached_or_downloaded" for _ in selected)),
        "heuristic_notice": "Search relevance and sampled-frame visual quality are estimates; final contact-sheet review remains required.",
    }, usage_mode)
    snapshot = run_manifests / f"sources-{int(datetime.now().timestamp() * 1000)}-{secrets.token_hex(8)}.json"
    _write_json(snapshot, manifest)
    if sufficiency_failures:
        raise InsufficientMaterialError(
            "YouTube material sufficiency gate failed: " + "; ".join(sufficiency_failures) + f". Manifest: {snapshot}"
        )
    return {
        **manifest,
        "sources_manifest": str(snapshot),
        "asset_manifest": str(snapshot),
        "asset_manifest_path": str(snapshot),
        "selected": selected,
        "selected_assets": selected,
        "priority_queries": queries,
        "target_duration": target_duration,
    }


def _load_object(path: str) -> dict[str, Any]:
    payload = _read_json(Path(path).expanduser().resolve(), {})
    if not isinstance(payload, dict):
        raise YouTubePipelineError(f"Expected JSON object: {path}")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--theme", required=True)
    parser.add_argument("--style-profile", required=True)
    parser.add_argument("--audio-profile", required=True)
    parser.add_argument("--timeline-plan", required=True)
    parser.add_argument("--material-root", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--desired-count", type=int, required=True)
    parser.add_argument("--aspect-ratio", default="16:9")
    parser.add_argument("--min-resolution", default="1280x720")
    parser.add_argument("--target-duration", type=float)
    parser.add_argument("--candidate-pool-multiplier", type=int, default=6)
    parser.add_argument("--search-query", action="append", dest="queries", default=[])
    parser.add_argument("--visual-style", default="auto")
    parser.add_argument("--exclude-youtube-id", action="append", default=[])
    parser.add_argument("--results-per-query", type=int, default=8)
    parser.add_argument("--max-download-candidates", type=int, default=36)
    parser.add_argument("--usage-mode", choices=USAGE_MODES, default="local_evaluation")
    parser.add_argument("--result-json")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    match = re.fullmatch(r"(\d+)x(\d+)", args.min_resolution.lower())
    if not match:
        raise SystemExit("--min-resolution must be WIDTHxHEIGHT")
    result = run_youtube_pipeline(
        args.theme,
        _load_object(args.style_profile),
        _load_object(args.audio_profile),
        args.material_root,
        args.cache_dir,
        args.desired_count,
        args.aspect_ratio,
        min_resolution=(int(match.group(1)), int(match.group(2))),
        target_duration=args.target_duration,
        timeline_plan=_load_object(args.timeline_plan),
        candidate_pool_multiplier=args.candidate_pool_multiplier,
        priority_queries=args.queries,
        visual_cohesion_profile=args.visual_style,
        excluded_youtube_ids=args.exclude_youtube_id,
        results_per_query=args.results_per_query,
        max_download_candidates=args.max_download_candidates,
        usage_mode=args.usage_mode,
    )
    if args.result_json:
        _write_json(Path(args.result_json).expanduser().resolve(), result)
    print(json.dumps({
        "status": result["status"],
        "provider": "youtube",
        "candidate_count": result["candidate_count"],
        "download_candidate_count": result["download_candidate_count"],
        "selected_count": result["selected_count"],
        "sources_manifest": result["sources_manifest"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
