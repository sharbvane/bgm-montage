#!/usr/bin/env python3
"""Unified entry point for reference-style, BGM-driven montage creation."""

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
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from analyze_bgm import analyze_bgm
from analyze_references import analyze_references
from analyze_editing_grammar import analyze_editing_grammar
from edit_schema import normalize_edit_decisions, write_edit_decisions
from montage import InsufficientMaterialError as TimelineInsufficientMaterialError
from montage import MontageError, build_timeline, parse_ratio, render_timeline, write_plan
from pixabay_pipeline import InsufficientMaterialError as PixabayInsufficientMaterialError
from pixabay_pipeline import material_theme_directory, run_pixabay_pipeline, update_usage_intervals
from runtime_paths import RuntimePaths, discover_project_root
from timeline_planner import plan_timeline_slots
from validate_output import validate_output
from visual_intelligence import build_visual_style_profile
from youtube_pipeline import InsufficientMaterialError as YouTubeInsufficientMaterialError
from youtube_pipeline import run_youtube_pipeline
from material_usage_policy import USAGE_MODES, apply_usage_policy, material_usage_policy, normalize_usage_mode


SCRIPT_DIR = Path(__file__).resolve().parent
SKILL_DIR = SCRIPT_DIR.parent


def _discover_project_root() -> Path:
    return discover_project_root()


PROJECT_ROOT = _discover_project_root()


def _safe_slug(text: str, fallback: str = "montage") -> str:
    slug = re.sub(r"[^\w\u4e00-\u9fff-]+", "_", text.strip(), flags=re.UNICODE).strip("_.-")
    return slug[:48] or fallback


def _assert_writable_tree_separate(name: str, writable: Path, reference_dir: Path) -> None:
    """Reject an actual write target that is in the read-only reference tree."""

    if writable == reference_dir or writable.is_relative_to(reference_dir):
        raise ValueError(
            f"{name} must not be the reference directory or one of its children: {writable}"
        )


def _write_json(path: Path, data: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(4)}.tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)
    return path


def _file_fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": digest.hexdigest()}


def _json_copy(source: Path, destination: Path) -> Path:
    payload = json.loads(source.read_text(encoding="utf-8"))
    return _write_json(destination, payload)


def _first_number(data: Any, keys: set[str]) -> float | None:
    if isinstance(data, dict):
        for key, value in data.items():
            if key.lower() in keys and isinstance(value, (int, float)):
                return float(value)
        for value in data.values():
            found = _first_number(value, keys)
            if found is not None:
                return found
    elif isinstance(data, list):
        for value in data:
            found = _first_number(value, keys)
            if found is not None:
                return found
    return None


def _strip_secret(text: str) -> str:
    secret = os.getenv("PIXABAY_API_KEY", "")
    if secret:
        text = text.replace(secret, "[REDACTED]")
    text = re.sub(r"([?&]key=)[^&\s]+", r"\1[REDACTED]", text, flags=re.IGNORECASE)
    return text


def _style_payload(result: dict[str, Any]) -> dict[str, Any]:
    for key in ("style_profile", "profile", "aggregate"):
        value = result.get(key)
        if isinstance(value, dict):
            return value
    return result


def _audio_payload(result: dict[str, Any]) -> dict[str, Any]:
    for key in ("audio_profile", "bgm_profile", "profile", "analysis"):
        value = result.get(key)
        if isinstance(value, dict):
            return value
    return result


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _media_result_from_manifest(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    selected = payload.get("selected") or payload.get("sources") or payload.get("assets") or []
    return {
        **payload,
        "selected": selected,
        "selected_assets": selected,
        "sources_manifest": str(path),
    }


def _arg(args: argparse.Namespace, name: str, default: Any) -> Any:
    value = getattr(args, name, default)
    return default if value is None else value


def _merge_reference_audio_learning(
    reference_result: dict[str, Any],
    style_profile: dict[str, Any],
    editing_grammar: dict[str, Any],
) -> dict[str, Any]:
    """Expose the learned audio/cut relation in the style payload consumers use."""

    merged = dict(style_profile)
    merged["reference_audio_editing_grammar"] = {
        "status": editing_grammar.get("status"),
        "reliability": editing_grammar.get("reliability", {}),
        "cut_alignment": editing_grammar.get("cut_alignment", {}),
        "shot_duration_model": editing_grammar.get("shot_duration_model", {}),
        "ending_structure": editing_grammar.get("ending_structure", {}),
        "montage_policy": editing_grammar.get("montage_policy", {}),
        "application": (
            "Consumed by pre-download slot planning, boundary weighting, shot-duration allocation, "
            "scale/motion progression, transition choice, and ending structure."
        ),
    }
    reference_result["style_profile"] = merged
    reference_result["style"] = merged
    return merged


def _invocation_payload(
    *,
    bgm_path: Path,
    reference_dir: Path,
    material_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "bgm": {"path": str(bgm_path), "fingerprint": _file_fingerprint(bgm_path)},
        "reference_dir": str(reference_dir),
        "material_dir": str(material_dir),
        "theme": args.theme,
        "duration": float(args.duration),
        "ratio": args.ratio,
        "assets": _arg(args, "assets", None),
        "min_resolution": [int(args.min_width), int(args.min_height)],
        "candidate_pool_multiplier": int(_arg(args, "candidate_pool_multiplier", 6)),
        "max_search_pages": int(_arg(args, "max_search_pages", 3)),
        "priority_queries": list(_arg(args, "priority_queries", []) or []),
        "wide_aerial_only": bool(_arg(args, "wide_aerial_only", False)),
        "source_provider": str(_arg(args, "source_provider", "pixabay")),
        "usage_mode": str(_arg(args, "usage_mode", "local_evaluation")),
        "asset_manifest": str(_arg(args, "asset_manifest", "") or ""),
        "youtube_results_per_query": int(_arg(args, "youtube_results_per_query", 8)),
        "youtube_max_download_candidates": int(_arg(args, "youtube_max_download_candidates", 36)),
        "excluded_youtube_ids": list(_arg(args, "excluded_youtube_ids", []) or []),
        "visual_style": str(_arg(args, "visual_style", "auto")),
        "excluded_pixabay_ids": list(_arg(args, "excluded_pixabay_ids", []) or []),
        "max_reuse_per_asset": int(_arg(args, "max_reuse_per_asset", 1)),
        "max_asset_screen_share": float(_arg(args, "max_asset_screen_share", 0.30)),
        "min_repeat_gap_shots": int(_arg(args, "min_repeat_gap_shots", 3)),
        "min_repeat_gap_seconds": float(_arg(args, "min_repeat_gap_seconds", 6.0)),
        "jianying_draft": bool(_arg(args, "jianying_draft", False)),
        "jianying_draft_name": _arg(args, "jianying_draft_name", None),
        "jianying_draft_root": str(_arg(args, "jianying_draft_root", "") or ""),
    }


def _invocation_digest(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _material_sources_path(media_result: dict[str, Any]) -> Path | None:
    for key in ("sources_manifest", "sources_path", "manifest_path"):
        value = media_result.get(key)
        if value:
            path = Path(str(value)).expanduser().resolve()
            if path.is_file():
                return path
    return None


def _copy_or_create_sources(media_result: dict[str, Any], destination: Path) -> Path:
    source = _material_sources_path(media_result)
    if source and source != destination:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        return destination
    if source:
        return source
    selected = media_result.get("selected") or media_result.get("selected_assets") or []
    return _write_json(
        destination,
        {
            "schema_version": 1,
            "sources": selected,
            "search_rounds": media_result.get("search_rounds", []),
            "rejections": media_result.get("rejections", []),
        },
    )


def _find_jianying_python(explicit: str | None = None) -> Path:
    candidates = [
        Path(explicit).expanduser() if explicit else None,
        PROJECT_ROOT / ".venv-pyjianyingdraft" / "Scripts" / "python.exe",
        SKILL_DIR / ".venv-jianying" / "Scripts" / "python.exe",
        Path(sys.executable),
    ]
    checked: list[str] = []
    for candidate in candidates:
        if candidate is None:
            continue
        resolved = candidate.resolve()
        if not resolved.is_file() or str(resolved) in checked:
            continue
        checked.append(str(resolved))
        probe = subprocess.run(
            [str(resolved), "-c", "import pyJianYingDraft"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=20,
            check=False,
        )
        if probe.returncode == 0:
            return resolved
    raise RuntimeError(
        "Editable JianYing draft was requested, but no compatible Python with pyJianYingDraft was found. "
        "Pass --jianying-python; the tested dependency is documented in requirements-jianying.lock.txt."
    )


def _export_jianying_draft(
    args: argparse.Namespace,
    edit_decisions_path: Path,
    run_dir: Path,
    slug: str,
    run_id: str,
) -> dict[str, Any]:
    python = _find_jianying_python(_arg(args, "jianying_python", None))
    draft_name = str(_arg(args, "jianying_draft_name", None) or f"{slug}_{run_id}_可编辑")
    report_path = run_dir / "jianying_draft_report.json"
    command = [
        str(python),
        str(SCRIPT_DIR / "jianying_export.py"),
        str(edit_decisions_path),
        "--draft-name",
        draft_name,
        "--report",
        str(report_path),
    ]
    if _arg(args, "jianying_draft_root", None):
        command.extend(["--draft-root", str(_arg(args, "jianying_draft_root", None))])
    process = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=180,
        check=False,
    )
    if process.returncode != 0 or not report_path.is_file():
        reason = process.stderr.strip() or process.stdout.strip() or f"exit code {process.returncode}"
        raise RuntimeError(f"JianYing draft export failed: {_strip_secret(reason)[-1800:]}")
    report = _read_json(report_path)
    if not report.get("passed"):
        raise RuntimeError(f"JianYing draft structural validation failed: {report_path}")
    return report


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run the v1.3 pipeline, checkpointing every reusable stage."""

    load_dotenv(PROJECT_ROOT / ".env", override=False)
    source_provider = str(_arg(args, "source_provider", "pixabay")).lower()
    usage_mode = normalize_usage_mode(_arg(args, "usage_mode", "local_evaluation"))
    if source_provider == "pixabay" and not os.getenv("PIXABAY_API_KEY"):
        raise RuntimeError(f"PIXABAY_API_KEY is missing. Put it in {PROJECT_ROOT / '.env'}.")

    bgm_path = Path(args.bgm).expanduser().resolve()
    reference_dir = Path(args.reference_dir).expanduser().resolve()
    material_dir = Path(args.material_dir).expanduser().resolve()
    output_root = Path(args.output_dir).expanduser().resolve()
    if not bgm_path.is_file():
        raise FileNotFoundError(f"BGM not found: {bgm_path}")
    if not reference_dir.is_dir():
        raise FileNotFoundError(f"Reference directory not found: {reference_dir}")
    parse_ratio(args.ratio)
    if args.duration <= 0:
        raise ValueError("--duration must be greater than zero")
    if int(_arg(args, "candidate_pool_multiplier", 6)) < 1:
        raise ValueError("--candidate-pool-multiplier must be at least 1")
    if int(_arg(args, "max_search_pages", 3)) < 1:
        raise ValueError("--max-search-pages must be at least 1")
    if int(_arg(args, "max_reuse_per_asset", 1)) < 1:
        raise ValueError("--max-reuse-per-asset must be at least 1")
    if not 0 < float(_arg(args, "max_asset_screen_share", 0.30)) <= 1:
        raise ValueError("--max-asset-screen-share must be in (0, 1]")

    resume = bool(_arg(args, "resume_run", False))
    if resume and not getattr(args, "run_id", None):
        raise ValueError("--resume-run requires an explicit --run-id")
    slug = _safe_slug(args.project_name or args.theme)
    run_id = _safe_slug(
        getattr(args, "run_id", None)
        or f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{secrets.token_hex(2)}",
        "run",
    )
    run_dir = output_root / slug / run_id
    _assert_writable_tree_separate("output run directory", run_dir, reference_dir)
    _assert_writable_tree_separate(
        "material theme directory", material_theme_directory(material_dir, args.theme), reference_dir
    )
    cache_root = Path(args.cache_dir).expanduser().resolve()
    for cache_name in ("references", "bgm", source_provider):
        _assert_writable_tree_separate(f"{cache_name} cache directory", cache_root / cache_name, reference_dir)

    invocation = _invocation_payload(
        bgm_path=bgm_path,
        reference_dir=reference_dir,
        material_dir=material_dir,
        args=args,
    )
    invocation_digest = _invocation_digest(invocation)
    state_path = run_dir / "run_state.json"
    if run_dir.exists() and not resume:
        raise FileExistsError(
            f"Run directory already exists; choose another --run-id. Existing output was not overwritten: {run_dir}"
        )
    if run_dir.exists() and resume:
        if not state_path.is_file():
            raise FileNotFoundError(f"Cannot safely resume without run_state.json: {state_path}")
        existing_state = _read_json(state_path)
        if existing_state.get("invocation_digest") != invocation_digest:
            raise ValueError("Resume invocation differs from the original run; use a new --run-id")
        previous_report = run_dir / "run_report.json"
        if previous_report.is_file():
            prior = _read_json(previous_report)
            prior_video = Path(str(prior.get("artifacts", {}).get("video", "")))
            if prior.get("passed") is True and prior_video.is_file():
                return prior
    else:
        run_dir.mkdir(parents=True, exist_ok=False)
    cache_root.mkdir(parents=True, exist_ok=True)
    _write_json(
        state_path,
        {
            "schema_version": "1.3",
            "run_id": run_id,
            "invocation_digest": invocation_digest,
            "invocation": invocation,
            "resumed": resume,
        },
    )

    run_report_path = run_dir / "run_report.json"
    report: dict[str, Any] = {
        "schema_version": "1.3",
        "skill_version": "1.3",
        "run_id": run_id,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "resumed": resume,
        "theme": args.theme,
        "requested_duration_seconds": args.duration,
        "ratio": args.ratio,
        "inputs": {
            "bgm": str(bgm_path),
            "reference_dir": str(reference_dir),
            "material_dir": str(material_dir),
        },
        "source_provider": source_provider,
        "usage_mode": usage_mode,
        "material_usage_policy": material_usage_policy(usage_mode),
        "api_key_configured": bool(os.getenv("PIXABAY_API_KEY")) if source_provider == "pixabay" else None,
        "stages": {},
        "artifacts": {"run_state": str(state_path), "run_report": str(run_report_path)},
        "passed": False,
    }

    def checkpoint() -> None:
        _write_json(run_report_path, report)

    checkpoint()
    try:
        style_path = run_dir / "style_profile.json"
        print(f"[1/7] Analyzing reference style and shot semantics ({reference_dir})", flush=True)
        if resume and style_path.is_file():
            reference_result = _read_json(style_path)
            reference_resumed = True
        else:
            reference_result = analyze_references(
                reference_dir,
                cache_root / "references",
                style_path,
                semantic_required=not getattr(args, "allow_semantic_fallback", False),
            )
            reference_resumed = False
        style_profile = _style_payload(reference_result)
        reference_run_report = (
            reference_result.get("run_report", {})
            if isinstance(reference_result.get("run_report"), dict)
            else {}
        )
        report["stages"]["references"] = {
            "status": "resumed" if reference_resumed else "ok",
            "analyzed": reference_run_report.get("analyzed"),
            "reused": reference_run_report.get("reused"),
            "failed": reference_run_report.get("failed"),
        }
        report["artifacts"]["style_profile"] = str(style_path)
        checkpoint()

        editing_grammar_path = run_dir / "editing_grammar.json"
        print("[2/7] Learning reference audio-to-cut editing grammar", flush=True)
        if resume and editing_grammar_path.is_file():
            editing_grammar = _read_json(editing_grammar_path)
            grammar_resumed = True
        else:
            editing_grammar = analyze_editing_grammar(
                reference_dir,
                reference_result,
                cache_root / "references",
                editing_grammar_path,
            )
            grammar_resumed = False
        style_profile = _merge_reference_audio_learning(reference_result, style_profile, editing_grammar)
        _write_json(style_path, reference_result)
        report["stages"]["editing_grammar"] = {
            "status": "resumed" if grammar_resumed else editing_grammar.get("status"),
            "reliability": editing_grammar.get("reliability"),
            "analyzed": editing_grammar.get("run_report", {}).get("analyzed"),
            "reused": editing_grammar.get("run_report", {}).get("reused"),
        }
        report["artifacts"]["editing_grammar"] = str(editing_grammar_path)
        checkpoint()

        audiomap_path = run_dir / "audiomap.json"
        bgm_profile_path = run_dir / "bgm_profile.json"
        print(f"[3/7] Analyzing deterministic BGM structure ({bgm_path.name})", flush=True)
        if resume and audiomap_path.is_file():
            bgm_result = _read_json(audiomap_path)
            bgm_resumed = True
        else:
            bgm_result = analyze_bgm(bgm_path, cache_root / "bgm", audiomap_path, args.duration)
            bgm_resumed = False
        audio_profile = _audio_payload(bgm_result)
        _write_json(bgm_profile_path, bgm_result)
        bgm_duration = _first_number(
            audio_profile, {"duration", "duration_seconds", "analyzed_duration_seconds"}
        )
        target_duration = min(args.duration, bgm_duration) if bgm_duration and bgm_duration > 0 else args.duration
        target_duration = round(float(target_duration), 4)
        if target_duration < 0.5:
            raise RuntimeError("BGM analysis returned less than 0.5 seconds of usable audio")
        report["actual_duration_seconds"] = target_duration
        report["stages"]["bgm"] = {
            "status": "resumed" if bgm_resumed else "ok",
            "duration_seconds": bgm_duration,
            "target_seconds": target_duration,
            "rhythm_mode": (audio_profile.get("rhythm_mode") or {}).get("mode"),
            "analysis_digest": audio_profile.get("analysis_digest"),
        }
        report["artifacts"].update(
            {"audiomap": str(audiomap_path), "bgm_profile_compat": str(bgm_profile_path)}
        )
        checkpoint()

        visual_style_path = run_dir / "visual_style_profile.json"
        visual_request = str(_arg(args, "visual_style", "auto") or "auto").strip()
        visual_style_profile = build_visual_style_profile(
            args.theme,
            style_profile,
            audio_profile,
            "" if visual_request.lower() in {"", "auto", "none"} else visual_request,
        )
        style_profile = {**style_profile, "visual_style_profile": visual_style_profile}
        _write_json(visual_style_path, visual_style_profile)
        report["stages"]["visual_style"] = {
            "status": "ok",
            "profile_digest": visual_style_profile.get("profile_digest"),
            "profile_confidence": visual_style_profile.get("sequence", {}).get("profile_confidence"),
            "world_families": visual_style_profile.get("world_model", {}).get("preferred_families", []),
            "color_profile": visual_style_profile.get("color_profile", {}),
        }
        report["artifacts"]["visual_style_profile"] = str(visual_style_path)
        checkpoint()

        timeline_path = run_dir / "timeline.json"
        print("[4/7] Planning music-event shot slots before material acquisition", flush=True)
        if resume and timeline_path.is_file():
            timeline_plan = _read_json(timeline_path)
            timeline_resumed = True
        else:
            timeline_plan = plan_timeline_slots(
                audio_profile,
                target_duration,
                style_profile,
                editing_grammar,
            )
            _write_json(timeline_path, timeline_plan)
            timeline_resumed = False
        slot_count = len(timeline_plan.get("slots", []))
        if slot_count <= 0:
            raise RuntimeError("Pre-download timeline planner returned no shot slots")
        report["stages"]["timeline_planning"] = {
            "status": "resumed" if timeline_resumed else "ok",
            "slots": slot_count,
            "rhythm_mode": timeline_plan.get("rhythm_mode"),
            "event_snap_ratio": timeline_plan.get("metrics", {}).get("event_snap_ratio"),
        }
        report["artifacts"]["timeline"] = str(timeline_path)
        checkpoint()

        desired_assets = int(
            _arg(
                args,
                "assets",
                max(slot_count, max(4, min(30, math.ceil(target_duration / 1.7) + 3))),
            )
        )
        asset_manifest_path = run_dir / "asset_manifest.json"
        sources_path = run_dir / "sources.json"
        print(
            f"[5/7] Searching a >= {_arg(args, 'candidate_pool_multiplier', 6)}x pool and selecting {desired_assets} {source_provider} assets",
            flush=True,
        )
        material_sources_path: Path | None = None
        if resume and asset_manifest_path.is_file():
            media_result = _media_result_from_manifest(asset_manifest_path)
            material_candidate = material_theme_directory(material_dir, args.theme) / "sources.json"
            material_sources_path = material_candidate if material_candidate.is_file() else None
            pixabay_resumed = True
        elif _arg(args, "asset_manifest", None):
            provided_manifest = Path(str(_arg(args, "asset_manifest", None))).expanduser().resolve()
            if not provided_manifest.is_file():
                raise FileNotFoundError(f"Provided asset manifest not found: {provided_manifest}")
            media_result = _media_result_from_manifest(provided_manifest)
            material_sources_path = provided_manifest
            _copy_or_create_sources(media_result, asset_manifest_path)
            pixabay_resumed = True
        else:
            if source_provider == "youtube":
                media_result = run_youtube_pipeline(
                    args.theme,
                    style_profile,
                    audio_profile,
                    material_dir,
                    cache_root / "youtube",
                    desired_assets,
                    args.ratio,
                    min_resolution=(args.min_width, args.min_height),
                    target_duration=target_duration,
                    timeline_plan=timeline_plan,
                    candidate_pool_multiplier=int(_arg(args, "candidate_pool_multiplier", 6)),
                    priority_queries=list(_arg(args, "priority_queries", []) or []),
                    visual_cohesion_profile=visual_request,
                    excluded_youtube_ids=list(_arg(args, "excluded_youtube_ids", []) or []),
                    results_per_query=int(_arg(args, "youtube_results_per_query", 8)),
                    max_download_candidates=int(_arg(args, "youtube_max_download_candidates", 36)),
                    usage_mode=usage_mode,
                )
            else:
                media_result = run_pixabay_pipeline(
                    args.theme,
                    style_profile,
                    audio_profile,
                    material_dir,
                    cache_root / "pixabay",
                    desired_assets,
                    args.ratio,
                    min_resolution=(args.min_width, args.min_height),
                    dry_run=False,
                    target_duration=target_duration,
                    timeline_plan=timeline_plan,
                    candidate_pool_multiplier=int(_arg(args, "candidate_pool_multiplier", 6)),
                    max_search_pages=int(_arg(args, "max_search_pages", 3)),
                    priority_queries=list(_arg(args, "priority_queries", []) or []),
                    wide_aerial_only=bool(_arg(args, "wide_aerial_only", False)),
                    visual_cohesion_profile=visual_request,
                    excluded_pixabay_ids=list(_arg(args, "excluded_pixabay_ids", []) or []),
                    usage_mode=usage_mode,
                )
            material_sources_path = _material_sources_path(media_result)
            _copy_or_create_sources(media_result, asset_manifest_path)
            pixabay_resumed = False
        if not asset_manifest_path.is_file():
            _copy_or_create_sources(media_result, asset_manifest_path)
        _write_json(asset_manifest_path, apply_usage_policy(_read_json(asset_manifest_path), usage_mode))
        _json_copy(asset_manifest_path, sources_path)
        selected = media_result.get("selected") or media_result.get("selected_assets") or []
        if not selected:
            raise RuntimeError(f"{source_provider} search completed without any selected local videos")
        report["stages"][source_provider] = {
            "status": "provided_manifest" if _arg(args, "asset_manifest", None) else ("resumed" if pixabay_resumed else "ok"),
            "selected": len(selected),
            "candidate_count": media_result.get("candidate_count"),
            "candidate_pool_gate": media_result.get("candidate_pool_gate"),
            "search_rounds": len(media_result.get("search_rounds", [])),
            "rejections": len(media_result.get("rejections", [])),
        }
        report["artifacts"].update(
            {"asset_manifest": str(asset_manifest_path), "sources_compat": str(sources_path)}
        )
        checkpoint()

        content_policy = {
            "max_reuse_per_asset": int(_arg(args, "max_reuse_per_asset", 1)),
            "max_asset_screen_share": float(_arg(args, "max_asset_screen_share", 0.30)),
            "min_repeat_gap_shots": int(_arg(args, "min_repeat_gap_shots", 3)),
            "min_repeat_gap_seconds": float(_arg(args, "min_repeat_gap_seconds", 6.0)),
            "visual_style_profile_digest": visual_style_profile.get("profile_digest"),
        }
        max_rework_attempts = max(0, int(_arg(args, "max_rework_attempts", 2)))
        final_output = run_dir / f"{slug}_montage.mp4"
        edit_decisions_path = run_dir / "edit_decisions.json"
        edit_plan_path = run_dir / "edit_plan.json"
        render_report_path = run_dir / "render_report.json"
        validation_path = run_dir / "validation.json"
        attempts: list[dict[str, Any]] = []
        successful_plan: dict[str, Any] | None = None
        successful_validation: dict[str, Any] | None = None
        print(
            f"[6/7] Assigning assets, rendering, and auto-reworking up to {max_rework_attempts} time(s)",
            flush=True,
        )
        for attempt_index in range(max_rework_attempts + 1):
            attempt_number = attempt_index + 1
            attempt_dir = run_dir / "attempts" / f"attempt_{attempt_number:02d}"
            attempt_dir.mkdir(parents=True, exist_ok=True)
            attempt_plan_path = attempt_dir / "edit_decisions.json"
            attempt_video = attempt_dir / f"{slug}_attempt_{attempt_number:02d}.mp4"
            attempt_report_path = attempt_dir / "render_report.json"
            seed = f"{args.theme}|{bgm_path.name}|{run_id}|attempt={attempt_number}"
            try:
                plan = build_timeline(
                    audio_profile,
                    media_result,
                    target_duration,
                    style_profile,
                    seed=seed,
                    editing_grammar=editing_grammar,
                    content_policy=content_policy,
                    ratio=args.ratio,
                    timeline_plan=timeline_plan,
                    theme=args.theme,
                )
                plan["run_id"] = run_id
                plan["attempt"] = attempt_number
                output_spec = parse_ratio(args.ratio)
                plan = normalize_edit_decisions(
                    plan,
                    bgm_path=bgm_path,
                    ratio=args.ratio,
                    width=output_spec.width,
                    height=output_spec.height,
                    fps=30.0,
                )
                write_edit_decisions(attempt_plan_path, plan)
                if not (resume and attempt_video.is_file()):
                    render_timeline(plan, bgm_path, attempt_video, args.ratio, style_profile)
                validation = validate_output(
                    attempt_video,
                    expected_duration=target_duration,
                    expected_ratio=args.ratio,
                    report_path=attempt_report_path,
                    frames_dir=attempt_dir / "validation_frames",
                    edit_plan=plan,
                    audiomap=audio_profile,
                    expected_fps=30.0,
                )
                attempt_record = {
                    "attempt": attempt_number,
                    "status": "passed" if validation.get("passed") else "failed_qa",
                    "seed": seed,
                    "plan": str(attempt_plan_path),
                    "video": str(attempt_video),
                    "report": str(attempt_report_path),
                    "failed_checks": [
                        key for key, passed in validation.get("checks", {}).items() if not passed
                    ],
                }
                attempts.append(attempt_record)
                report["stages"]["render_attempts"] = attempts
                checkpoint()
                if validation.get("passed"):
                    successful_plan = plan
                    successful_validation = validation
                    if final_output.exists():
                        raise FileExistsError(f"Final output already exists and was not overwritten: {final_output}")
                    os.replace(attempt_video, final_output)
                    break
            except (TimelineInsufficientMaterialError, MontageError) as exc:
                attempts.append(
                    {
                        "attempt": attempt_number,
                        "status": "failed_edit_or_render",
                        "seed": seed,
                        "error": _strip_secret(str(exc)),
                        "plan": str(attempt_plan_path),
                    }
                )
                report["stages"]["render_attempts"] = attempts
                checkpoint()

        if successful_plan is None or successful_validation is None:
            failures = [item.get("failed_checks") or item.get("error") for item in attempts]
            raise RuntimeError(
                "All render/QA attempts failed; no montage was delivered. "
                f"Attempt diagnostics: {failures}"
            )

        write_edit_decisions(
            edit_decisions_path,
            successful_plan,
            bgm_path=bgm_path,
            ratio=args.ratio,
            width=parse_ratio(args.ratio).width,
            height=parse_ratio(args.ratio).height,
            fps=30.0,
        )
        successful_plan = _read_json(edit_decisions_path)
        _json_copy(edit_decisions_path, edit_plan_path)
        successful_validation["rework"] = {
            "attempts": attempts,
            "successful_attempt": successful_plan.get("attempt"),
            "automatic_rework_count": max(0, int(successful_plan.get("attempt", 1)) - 1),
        }
        _write_json(render_report_path, successful_validation)
        _write_json(validation_path, successful_validation)
        # Historical usage is committed only after the final video passes QA.
        for manifest in {
            path for path in (material_sources_path, asset_manifest_path) if path is not None and path.is_file()
        }:
            # Usage history is part of the success contract.  The stage helper
            # performs a locked, fail-closed read/modify/write; falling back to
            # an unlocked legacy rewrite here could clobber another concurrent
            # run, so a real persistence error must fail the run explicitly.
            update_usage_intervals(manifest, successful_plan)
        if asset_manifest_path.is_file():
            _json_copy(asset_manifest_path, sources_path)

        report["stages"]["render"] = {
            "status": "ok",
            "shots": len(successful_plan.get("shots", [])),
            "successful_attempt": successful_plan.get("attempt"),
        }
        report["stages"]["validation"] = {
            "status": "ok",
            "checks": successful_validation.get("checks", {}),
        }
        report["stages"]["visual_consistency"] = {
            "status": "ok" if successful_plan.get("visual_sequence_consistency", {}).get("passed") else "not_evaluated",
            **successful_plan.get("visual_sequence_consistency", {}),
        }
        report["artifacts"].update(
            {
                "edit_decisions": str(edit_decisions_path),
                "edit_plan_compat": str(edit_plan_path),
                "render_report": str(render_report_path),
                "validation_compat": str(validation_path),
                "video": str(final_output),
            }
        )
        if bool(_arg(args, "jianying_draft", False)):
            print("[7/8] Building and structurally validating editable JianYing draft", flush=True)
            jianying_report = _export_jianying_draft(
                args,
                edit_decisions_path,
                run_dir,
                slug,
                run_id,
            )
            report["stages"]["jianying_draft"] = {
                "status": "ok",
                "draft_path": jianying_report.get("draft_path"),
                "validation": jianying_report.get("validation", {}),
                "unmapped_or_approximate_count": len(jianying_report.get("unmapped_or_approximate", [])),
            }
            report["artifacts"].update(
                {
                    "jianying_draft": jianying_report.get("draft_path"),
                    "jianying_draft_report": jianying_report.get("report_path", str(run_dir / "jianying_draft_report.json")),
                }
            )
        else:
            report["stages"]["jianying_draft"] = {"status": "not_requested"}
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        report["passed"] = True
        checkpoint()
        print("[8/8] Final full-decode QA passed; usage history committed", flush=True)
        return report
    except Exception as exc:
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        report["passed"] = False
        report["failure"] = {"type": type(exc).__name__, "message": _strip_secret(str(exc))}
        if isinstance(exc, (PixabayInsufficientMaterialError, YouTubeInsufficientMaterialError, TimelineInsufficientMaterialError)):
            report["failure"]["category"] = "insufficient_material"
        checkpoint()
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bgm", required=True, help="Input BGM/audio file")
    parser.add_argument("--theme", required=True, help="Theme used to generate English visual search queries")
    parser.add_argument("--duration", required=True, type=float, help="Requested output duration in seconds")
    parser.add_argument("--ratio", required=True, help="9:16, 16:9, 1:1, 4:5, or WIDTHxHEIGHT")
    parser.add_argument("--output-dir", required=True, help="Directory that receives the run subdirectory")
    parser.add_argument("--project-name", help="Optional safe output subdirectory/file name")
    parser.add_argument(
        "--run-id",
        help="Optional unique run directory name; defaults to UTC timestamp plus random suffix and never overwrites.",
    )
    parser.add_argument(
        "--resume-run",
        action="store_true",
        help="Resume a failed run with the exact same inputs; requires --run-id and validates run_state.json.",
    )
    parser.add_argument("--reference-dir", default=str(PROJECT_ROOT / "参考视频"))
    parser.add_argument("--material-dir", default=str(PROJECT_ROOT / "视频素材"))
    parser.add_argument("--cache-dir", default=str(PROJECT_ROOT / ".bgm-montage-cache"))
    parser.add_argument(
        "--source-provider",
        choices=("pixabay", "youtube"),
        default="pixabay",
        help="Material acquisition provider. YouTube mode uses yt-dlp and does not require PIXABAY_API_KEY.",
    )
    parser.add_argument(
        "--usage-mode",
        choices=USAGE_MODES,
        default="local_evaluation",
        help="Default local evaluation applies no copyright/license restriction or ranking weight; use publish only after explicit distribution intent.",
    )
    parser.add_argument(
        "--asset-manifest",
        help="Use an already downloaded and visually reviewed asset manifest; skips provider search/download only.",
    )
    parser.add_argument("--assets", type=int, help="Override final asset count (default derives from duration)")
    parser.add_argument("--min-width", type=int, default=1280)
    parser.add_argument("--min-height", type=int, default=720)
    parser.add_argument(
        "--candidate-pool-multiplier",
        type=int,
        default=6,
        help="Minimum metadata candidates per planned shot slot before downloads (default: 6).",
    )
    parser.add_argument(
        "--max-search-pages",
        type=int,
        default=3,
        help="Maximum Pixabay pages per expanded query round (default: 3).",
    )
    parser.add_argument(
        "--search-query",
        dest="priority_queries",
        action="append",
        default=[],
        help="Exact priority provider query; repeat for multiple visual search directions.",
    )
    parser.add_argument("--youtube-results-per-query", type=int, default=8)
    parser.add_argument("--youtube-max-download-candidates", type=int, default=36)
    parser.add_argument(
        "--exclude-youtube-id",
        dest="excluded_youtube_ids",
        action="append",
        default=[],
        help="Exclude a YouTube video ID after contact-sheet review; repeat as needed.",
    )
    parser.add_argument(
        "--wide-aerial-only",
        action="store_true",
        help="Exclude abstract/close-up metadata hits and strongly prefer aerial, FPV, and wide footage.",
    )
    parser.add_argument(
        "--visual-style",
        "--visual-cohesion-profile",
        dest="visual_style",
        default="auto",
        help="Free-form visual direction; auto derives it from theme, references, and BGM.",
    )
    parser.add_argument(
        "--exclude-pixabay-id",
        dest="excluded_pixabay_ids",
        action="append",
        default=[],
        help="Exclude a visually rejected Pixabay asset ID; repeat after contact-sheet review.",
    )
    parser.add_argument(
        "--max-reuse-per-asset",
        "--max-source-reuse",
        dest="max_reuse_per_asset",
        type=int,
        default=1,
        help="Maximum uses of one canonical source in this montage (default: 1).",
    )
    parser.add_argument(
        "--max-asset-screen-share",
        "--max-source-share",
        dest="max_asset_screen_share",
        type=float,
        default=0.30,
        help="Maximum cumulative output share for one canonical source (default: 0.30).",
    )
    parser.add_argument("--min-repeat-gap-shots", type=int, default=3)
    parser.add_argument("--min-repeat-gap-seconds", type=float, default=6.0)
    parser.add_argument(
        "--max-rework-attempts",
        type=int,
        default=2,
        help="Automatic re-edit attempts after the first failed render QA (default: 2).",
    )
    parser.add_argument(
        "--allow-semantic-fallback",
        action="store_true",
        help="Allow explicit structural-only reference analysis if the pretrained CLIP model is unavailable.",
    )
    parser.add_argument(
        "--jianying-draft",
        action="store_true",
        help="Also create an editable JianYing Pro draft from edit_decisions after final QA.",
    )
    parser.add_argument("--jianying-draft-name", help="Optional unique JianYing project name.")
    parser.add_argument("--jianying-draft-root", help="Explicit JianYing project root containing root_meta_info.json.")
    parser.add_argument("--jianying-python", help="Python executable containing the optional pyJianYingDraft dependency.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        report = run(args)
        print(json.dumps({"passed": report["passed"], "artifacts": report["artifacts"]}, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        message = _strip_secret(str(exc))
        print(f"ERROR: {message}", file=sys.stderr)
        if os.getenv("BGM_MONTAGE_DEBUG") == "1":
            print(_strip_secret(traceback.format_exc()), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
