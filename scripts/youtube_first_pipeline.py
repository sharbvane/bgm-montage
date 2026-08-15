#!/usr/bin/env python3
"""YouTube-first orchestration with bounded Pixabay fallback for v1.4."""

from __future__ import annotations

import json
import math
import os
import secrets
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from material_usage_policy import apply_usage_policy, normalize_usage_mode
from pixabay_pipeline import InsufficientMaterialError as PixabayInsufficientMaterialError
from pixabay_pipeline import PixabayPipelineError, evaluate_selected_sufficiency, run_pixabay_pipeline
from visual_intelligence import asset_profile_fit, build_visual_style_profile
from youtube_pipeline import run_youtube_pipeline


SCHEMA_VERSION = "1.4-youtube-first.1"


class YouTubeFirstPipelineError(RuntimeError):
    pass


class InsufficientMaterialError(YouTubeFirstPipelineError):
    pass


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(4)}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)
    return path


def unified_asset_score(asset: Mapping[str, Any], visual_profile: Mapping[str, Any]) -> tuple[float, dict[str, Any]]:
    """Provider-neutral visual score used after YouTube and Pixabay merge."""

    quality = asset.get("quality") if isinstance(asset.get("quality"), Mapping) else {}
    visual = quality.get("visual_analysis") if isinstance(quality.get("visual_analysis"), Mapping) else {}
    fit = asset_profile_fit(asset, visual_profile)
    score = (
        0.30 * float(quality.get("overall_score") or 0.0)
        + 0.18 * float(visual.get("aesthetic_score") or 0.0)
        + 0.16 * float(visual.get("cinematic_score") or 0.0)
        + 0.12 * float(visual.get("spatial_depth_score") or 0.0)
        + 0.10 * float(visual.get("visual_impact_score") or 0.0)
        + 0.06 * float(quality.get("motion_score") or asset.get("motion_score") or 0.0)
        + 0.08 * float(fit.get("total") or 0.0)
    )
    return round(score, 5), fit


def _identity(asset: Mapping[str, Any]) -> str:
    fingerprint = asset.get("fingerprint") if isinstance(asset.get("fingerprint"), Mapping) else {}
    return str(
        fingerprint.get("sha256") or asset.get("file_hash") or asset.get("canonical_source_id")
        or f"{asset.get('provider')}:{asset.get('asset_id') or asset.get('id') or asset.get('local_path')}"
    )


def merge_and_rank_assets(
    youtube_assets: Sequence[Mapping[str, Any]],
    pixabay_assets: Sequence[Mapping[str, Any]],
    desired_count: int,
    visual_profile: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    deduped: dict[str, dict[str, Any]] = {}
    duplicate_count = 0
    for raw in [*youtube_assets, *pixabay_assets]:
        item = dict(raw)
        score, fit = unified_asset_score(item, visual_profile)
        item["score"] = score
        item["unified_score"] = score
        item["visual_profile_fit"] = fit
        key = _identity(item)
        if key in deduped:
            duplicate_count += 1
            if score > float(deduped[key].get("unified_score") or 0.0):
                deduped[key] = item
        else:
            deduped[key] = item
    pool = sorted(deduped.values(), key=lambda item: (-float(item.get("unified_score") or 0.0), _identity(item)))
    # A small scene-diversity bonus is provider-neutral and prevents one visual
    # setup from monopolizing the final pool when scores are nearly tied.
    selected: list[dict[str, Any]] = []
    scene_counts: Counter[str] = Counter()
    remaining = list(pool)
    while remaining and len(selected) < int(desired_count):
        best_index = max(
            range(len(remaining)),
            key=lambda index: float(remaining[index].get("unified_score") or 0.0)
            + (0.055 if scene_counts[str(remaining[index].get("scene_category") or "general")] == 0 else 0.0),
        )
        chosen = remaining.pop(best_index)
        selected.append(chosen)
        scene_counts[str(chosen.get("scene_category") or "general")] += 1
    return selected, {
        "input_count": len(youtube_assets) + len(pixabay_assets),
        "deduplicated_count": len(pool), "duplicate_count": duplicate_count,
        "selected_count": len(selected), "scene_distribution": dict(scene_counts),
        "policy": "provider-neutral visual score plus bounded scene-diversity bonus",
    }


def evaluate_combined_sufficiency(
    selected: Sequence[Mapping[str, Any]], desired_count: int, target_duration: float | None,
    available_candidates: int, required_candidates: int, visual_profile: Mapping[str, Any],
) -> dict[str, Any]:
    base = evaluate_selected_sufficiency(selected, desired_count, target_duration)
    failures = list(base.get("failures", []))
    if len(selected) < desired_count:
        failures.insert(0, f"selected assets {len(selected)} < requested {desired_count}")
    if available_candidates < required_candidates:
        failures.append(f"combined metadata candidates {available_candidates} < {required_candidates}")
    qualities = [float((item.get("quality") or {}).get("overall_score") or 0.0) for item in selected]
    fits = [float((item.get("visual_profile_fit") or {}).get("total") or 0.0) for item in selected]
    average_quality = sum(qualities) / len(qualities) if qualities else 0.0
    average_fit = sum(fits) / len(fits) if fits else 0.0
    quality_floor = float((visual_profile.get("quality") or {}).get("aesthetic_floor") or 0.42) * 0.72
    if selected and average_quality < quality_floor:
        failures.append(f"average quality {average_quality:.4f} < {quality_floor:.4f}")
    if selected and average_fit < 0.30:
        failures.append(f"average style fit {average_fit:.4f} < 0.3000")
    identities = [_identity(item) for item in selected]
    max_reuse = max(Counter(identities).values(), default=0)
    max_share = max_reuse / max(1, len(selected))
    if max_reuse > 1:
        failures.append("final pool repeats a canonical source")
    return {
        **base, "passed": not failures, "failures": list(dict.fromkeys(failures)),
        "usable_count": len(selected), "available_candidate_count": available_candidates,
        "required_candidate_count": required_candidates,
        "candidate_to_target_ratio": round(available_candidates / max(1, desired_count), 4),
        "average_quality": round(average_quality, 4), "average_style_fit": round(average_fit, 4),
        "maximum_canonical_share": round(max_share, 4), "maximum_reuse_count": max_reuse,
    }


def run_youtube_first_pipeline(
    theme: str, style_profile: Mapping[str, Any] | None, audio_profile: Mapping[str, Any] | None,
    material_root: str | os.PathLike[str], cache_dir: str | os.PathLike[str], desired_count: int,
    aspect_ratio: str, *, min_resolution: tuple[int, int] = (1280, 720), target_duration: float | None = None,
    timeline_plan: Mapping[str, Any] | None = None, candidate_pool_multiplier: int = 6,
    max_search_pages: int = 3, priority_queries: Sequence[str] = (),
    visual_cohesion_profile: str = "auto", excluded_youtube_ids: Sequence[str] = (),
    excluded_pixabay_ids: Sequence[str] = (), results_per_query: int = 8,
    max_download_candidates: int = 36, max_search_rounds: int = 3,
    wide_aerial_only: bool = False, usage_mode: str = "local_evaluation",
    source_windows: Mapping[str, Any] | Sequence[str] | None = None,
) -> dict[str, Any]:
    usage_mode = normalize_usage_mode(usage_mode)
    explicit = "" if str(visual_cohesion_profile).lower() in {"", "auto", "none"} else str(visual_cohesion_profile)
    visual_profile = build_visual_style_profile(theme, style_profile, audio_profile, explicit)
    youtube = run_youtube_pipeline(
        theme, style_profile, audio_profile, material_root, Path(cache_dir) / "youtube", desired_count,
        aspect_ratio, min_resolution=min_resolution, target_duration=target_duration,
        timeline_plan=timeline_plan, candidate_pool_multiplier=candidate_pool_multiplier,
        priority_queries=priority_queries, visual_cohesion_profile=visual_cohesion_profile,
        excluded_youtube_ids=excluded_youtube_ids, results_per_query=results_per_query,
        max_download_candidates=max_download_candidates, max_search_rounds=max_search_rounds,
        usage_mode=usage_mode, allow_insufficient=True,
        source_windows=source_windows,
    )
    youtube_ok = bool((youtube.get("candidate_pool_gate") or {}).get("passed")) and bool((youtube.get("sufficiency") or {}).get("passed"))
    pixabay: dict[str, Any] | None = None
    fallback = {"triggered": not youtube_ok, "reason": [] if youtube_ok else list((youtube.get("sufficiency") or {}).get("failures", [])), "status": "not_needed" if youtube_ok else "pending"}
    if not youtube_ok:
        try:
            pixabay = run_pixabay_pipeline(
                theme, style_profile, audio_profile, material_root, Path(cache_dir) / "pixabay",
                desired_count, aspect_ratio, min_resolution=min_resolution, dry_run=False,
                target_duration=target_duration, timeline_plan=timeline_plan,
                candidate_pool_multiplier=candidate_pool_multiplier, max_search_pages=max_search_pages,
                priority_queries=priority_queries, wide_aerial_only=wide_aerial_only,
                visual_cohesion_profile=visual_cohesion_profile,
                excluded_pixabay_ids=excluded_pixabay_ids, usage_mode=usage_mode,
            )
            fallback["status"] = "completed"
        except (PixabayInsufficientMaterialError, PixabayPipelineError) as exc:
            fallback["status"] = "failed"
            fallback["error"] = str(exc)[-1800:]

    youtube_assets = list(youtube.get("selected") or [])
    pixabay_assets = list((pixabay or {}).get("selected") or [])
    selected, merge_report = merge_and_rank_assets(youtube_assets, pixabay_assets, desired_count, visual_profile)
    slot_count = len((timeline_plan or {}).get("slots", []))
    required_candidates = max(desired_count, slot_count * max(1, int(candidate_pool_multiplier)))
    available_candidates = int(youtube.get("candidate_count") or 0) + int((pixabay or {}).get("candidate_count") or 0)
    sufficiency = evaluate_combined_sufficiency(
        selected, desired_count, target_duration, available_candidates, required_candidates, visual_profile
    )
    status = "ok" if sufficiency["passed"] else "insufficient_material"
    cache_root = Path(cache_dir).expanduser().resolve()
    snapshot = cache_root / "youtube-first" / "run_manifests" / f"sources-{int(datetime.now().timestamp() * 1000)}-{secrets.token_hex(8)}.json"
    manifest = apply_usage_policy({
        "schema_version": SCHEMA_VERSION, "asset_manifest_schema_version": 3,
        "manifest_type": "asset_manifest", "provider": "youtube-first", "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status, "theme": theme, "requested": {"desired_count": desired_count, "aspect_ratio": aspect_ratio, "candidate_pool_multiplier": candidate_pool_multiplier, "priority_queries": list(priority_queries), "source_windows": source_windows or {}},
        "strategy": ["local reusable assets", "YouTube dynamic search", "Pixabay fallback when needed", "provider-neutral merge and hard sufficiency gate"],
        "youtube": {"status": youtube.get("status"), "manifest": youtube.get("sources_manifest"), "candidate_count": youtube.get("candidate_count"), "selected_count": youtube.get("selected_count"), "candidate_pool_gate": youtube.get("candidate_pool_gate"), "search_rounds": youtube.get("search_rounds", []), "query_plan": youtube.get("query_plan", [])},
        "pixabay_fallback": {**fallback, "manifest": (pixabay or {}).get("sources_manifest"), "candidate_count": (pixabay or {}).get("candidate_count", 0), "selected_count": (pixabay or {}).get("selected_count", 0), "search_rounds": (pixabay or {}).get("search_rounds", [])},
        "visual_style_profile": visual_profile, "candidate_count": available_candidates,
        "candidate_pool_gate": {"passed": available_candidates >= required_candidates, "available_candidate_count": available_candidates, "required_candidate_count": required_candidates, "candidate_to_target_ratio": round(available_candidates / max(1, desired_count), 4), "failures": [] if available_candidates >= required_candidates else [f"combined metadata candidates {available_candidates} < {required_candidates}"]},
        "merge": merge_report, "sources": selected, "assets": selected, "selected_count": len(selected),
        "sufficiency": sufficiency, "source_distribution": dict(Counter(str(item.get("provider") or "unknown") for item in selected)),
    }, usage_mode)
    _write_json(snapshot, manifest)
    result = {**manifest, "sources_manifest": str(snapshot), "asset_manifest": str(snapshot), "asset_manifest_path": str(snapshot), "selected": selected, "selected_assets": selected, "youtube_result": youtube, "pixabay_result": pixabay}
    if not sufficiency["passed"]:
        raise InsufficientMaterialError("YouTube-first combined material gate failed: " + "; ".join(sufficiency["failures"]) + f". Manifest: {snapshot}")
    return result
