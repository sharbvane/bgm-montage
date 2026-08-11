#!/usr/bin/env python3
"""Unified entry point for reference-style, BGM-driven Pixabay montage creation."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import secrets
import shutil
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from analyze_bgm import analyze_bgm
from analyze_references import analyze_references
from analyze_editing_grammar import analyze_editing_grammar
from montage import InsufficientMaterialError as TimelineInsufficientMaterialError
from montage import MontageError, build_timeline, parse_ratio, render_timeline, write_plan
from pixabay_pipeline import InsufficientMaterialError as PixabayInsufficientMaterialError
from pixabay_pipeline import material_theme_directory, run_pixabay_pipeline, update_usage_intervals
from runtime_paths import RuntimePaths, discover_project_root
from validate_output import validate_output


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
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


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


def _apply_usage_to_manifest(manifest: Path, plan: dict[str, Any]) -> None:
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    if isinstance(payload, list):
        sources = payload
    elif isinstance(payload, dict):
        sources = payload.get("sources", payload.get("selected", payload.get("assets", [])))
    else:
        return
    if not isinstance(sources, list):
        return
    for source in sources:
        if not isinstance(source, dict):
            continue
        source_id = str(source.get("pixabay_id", source.get("id", source.get("asset_id", ""))))
        source_path = str(source.get("local_path", source.get("path", "")))
        intervals = []
        for shot in plan.get("shots", []):
            shot_id = str(shot.get("pixabay_id", shot.get("asset_id", "")))
            if (source_id and source_id == shot_id) or (source_path and Path(source_path) == Path(str(shot.get("local_path", "")))):
                intervals.append(
                    {
                        "output_start": shot.get("output_start"),
                        "output_end": shot.get("output_end"),
                        "source_start": shot.get("source_start"),
                        "source_end": shot.get("source_end"),
                        "speed": shot.get("speed"),
                    }
                )
        source["usage_intervals"] = intervals
        source["actual_usage_intervals"] = intervals
    manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    if not os.getenv("PIXABAY_API_KEY"):
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

    slug = _safe_slug(args.project_name or args.theme)
    run_id = _safe_slug(
        getattr(args, "run_id", None)
        or f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{secrets.token_hex(2)}",
        "run",
    )
    run_dir = output_root / slug / run_id
    if run_dir.exists():
        raise FileExistsError(
            f"Run directory already exists; choose another --run-id. Existing output was not overwritten: {run_dir}"
        )
    _assert_writable_tree_separate("output run directory", run_dir, reference_dir)
    _assert_writable_tree_separate(
        "material theme directory", material_theme_directory(material_dir, args.theme), reference_dir
    )
    cache_root = Path(args.cache_dir).expanduser().resolve()
    for cache_name in ("references", "bgm", "pixabay"):
        _assert_writable_tree_separate(f"{cache_name} cache directory", cache_root / cache_name, reference_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "schema_version": 2,
        "skill_version": "1.1",
        "run_id": run_id,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "theme": args.theme,
        "requested_duration_seconds": args.duration,
        "ratio": args.ratio,
        "inputs": {"bgm": str(bgm_path), "reference_dir": str(reference_dir), "material_dir": str(material_dir)},
        "api_key_configured": True,
        "stages": {},
        "artifacts": {},
    }

    style_path = run_dir / "style_profile.json"
    print(f"[1/6] Analyzing reference style and shot semantics ({reference_dir})", flush=True)
    reference_result = analyze_references(
        reference_dir,
        cache_root / "references",
        style_path,
        semantic_required=not getattr(args, "allow_semantic_fallback", False),
    )
    style_profile = _style_payload(reference_result)
    if not style_path.is_file():
        _write_json(style_path, reference_result)
    reference_run_report = reference_result.get("run_report", {}) if isinstance(reference_result.get("run_report"), dict) else {}
    report["stages"]["references"] = {
        "status": "ok",
        "analyzed": reference_result.get("analyzed_count", reference_result.get("analyzed", reference_run_report.get("analyzed"))),
        "reused": reference_result.get("reused_count", reference_result.get("reused", reference_run_report.get("reused"))),
        "failed": reference_run_report.get("failed"),
    }
    report["artifacts"]["style_profile"] = str(style_path)

    editing_grammar_path = run_dir / "editing_grammar.json"
    print("[2/6] Learning reference audio-to-cut editing grammar", flush=True)
    editing_grammar = analyze_editing_grammar(
        reference_dir,
        reference_result,
        cache_root / "references",
        editing_grammar_path,
    )
    report["stages"]["editing_grammar"] = {
        "status": editing_grammar.get("status"),
        "reliability": editing_grammar.get("reliability"),
        "analyzed": editing_grammar.get("run_report", {}).get("analyzed"),
        "reused": editing_grammar.get("run_report", {}).get("reused"),
    }
    report["artifacts"]["editing_grammar"] = str(editing_grammar_path)

    bgm_profile_path = run_dir / "bgm_profile.json"
    print(f"[3/6] Analyzing BGM structure ({bgm_path.name})", flush=True)
    bgm_result = analyze_bgm(bgm_path, cache_root / "bgm", bgm_profile_path, args.duration)
    audio_profile = _audio_payload(bgm_result)
    if not bgm_profile_path.is_file():
        _write_json(bgm_profile_path, bgm_result)
    bgm_duration = _first_number(audio_profile, {"duration", "duration_seconds", "analyzed_duration_seconds"})
    target_duration = min(args.duration, bgm_duration) if bgm_duration and bgm_duration > 0 else args.duration
    target_duration = round(float(target_duration), 4)
    if target_duration < 0.5:
        raise RuntimeError("BGM analysis returned less than 0.5 seconds of usable audio")
    report["actual_duration_seconds"] = target_duration
    report["stages"]["bgm"] = {"status": "ok", "duration_seconds": bgm_duration, "target_seconds": target_duration}
    report["artifacts"]["bgm_profile"] = str(bgm_profile_path)

    desired_assets = args.assets or max(4, min(30, math.ceil(target_duration / 1.7) + 3))
    print(f"[4/6] Searching, ranking, and downloading {desired_assets} Pixabay candidates", flush=True)
    try:
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
        )
    except PixabayInsufficientMaterialError as exc:
        report["stages"]["pixabay"] = {"status": "insufficient_material", "error": _strip_secret(str(exc))}
        manifest = material_theme_directory(material_dir, args.theme) / "sources.json"
        if manifest.is_file():
            report["artifacts"]["sources"] = str(manifest)
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        report["passed"] = False
        failed_report = run_dir / "run_report.json"
        report["artifacts"]["run_report"] = str(failed_report)
        _write_json(failed_report, report)
        raise
    selected = media_result.get("selected") or media_result.get("selected_assets") or []
    if not selected:
        raise RuntimeError("Pixabay search completed without any selected local videos")
    report["stages"]["pixabay"] = {
        "status": "ok",
        "selected": len(selected),
        "search_rounds": len(media_result.get("search_rounds", [])),
        "rejections": len(media_result.get("rejections", [])),
    }

    material_sources_path = _material_sources_path(media_result)
    sources_path = _copy_or_create_sources(media_result, run_dir / "sources.json")
    edit_plan_path = run_dir / "edit_plan.json"
    print("[5/6] Building grammar-driven timeline and rendering", flush=True)
    try:
        plan = build_timeline(
            audio_profile,
            media_result,
            target_duration,
            style_profile,
            seed=f"{args.theme}|{bgm_path.name}",
            editing_grammar=editing_grammar,
            ratio=args.ratio,
        )
    except TimelineInsufficientMaterialError as exc:
        report["stages"]["timeline"] = {"status": "insufficient_material", "error": _strip_secret(str(exc))}
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        report["passed"] = False
        failed_report = run_dir / "run_report.json"
        report["artifacts"]["run_report"] = str(failed_report)
        _write_json(failed_report, report)
        raise
    write_plan(plan, edit_plan_path)
    output_path = run_dir / f"{slug}_montage.mp4"
    render_timeline(plan, bgm_path, output_path, args.ratio, style_profile)
    for manifest in {path for path in (material_sources_path, sources_path) if path is not None}:
        try:
            update_usage_intervals(manifest, plan)
        except Exception:
            _apply_usage_to_manifest(manifest, plan)
    report["stages"]["render"] = {"status": "ok", "shots": len(plan.get("shots", []))}
    report["artifacts"].update({"edit_plan": str(edit_plan_path), "sources": str(sources_path), "video": str(output_path)})

    print("[6/6] Running full-decode and media quality validation", flush=True)
    validation_path = run_dir / "validation.json"
    validation = validate_output(
        output_path,
        expected_duration=target_duration,
        expected_ratio=args.ratio,
        report_path=validation_path,
        frames_dir=run_dir / "validation_frames",
        edit_plan=plan,
    )
    report["stages"]["validation"] = {"status": "ok" if validation["passed"] else "failed", "checks": validation["checks"]}
    report["artifacts"]["validation"] = str(validation_path)
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    report["passed"] = bool(validation["passed"])
    run_report_path = run_dir / "run_report.json"
    report["artifacts"]["run_report"] = str(run_report_path)
    _write_json(run_report_path, report)
    if not validation["passed"]:
        raise RuntimeError(f"Render completed but validation failed; inspect {validation_path}")
    return report


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
    parser.add_argument("--reference-dir", default=str(PROJECT_ROOT / "参考视频"))
    parser.add_argument("--material-dir", default=str(PROJECT_ROOT / "视频素材"))
    parser.add_argument("--cache-dir", default=str(PROJECT_ROOT / ".bgm-montage-cache"))
    parser.add_argument("--assets", type=int, help="Override final asset count (default derives from duration)")
    parser.add_argument("--min-width", type=int, default=1280)
    parser.add_argument("--min-height", type=int, default=720)
    parser.add_argument(
        "--allow-semantic-fallback",
        action="store_true",
        help="Allow explicit structural-only reference analysis if the pretrained CLIP model is unavailable.",
    )
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
