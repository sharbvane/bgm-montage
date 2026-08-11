#!/usr/bin/env python3
"""Optional editable JianYing Pro draft adapter for bgm-montage v1.3.

The adapter consumes the same versioned edit_decisions used by rendering.  It
never reconstructs cuts from the final MP4 and never copies source media.
pyJianYingDraft is loaded lazily so the core montage environment remains free
of an optional editor-specific dependency.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Mapping

from edit_schema import load_edit_decisions, validate_edit_decisions


ADAPTER_VERSION = "1.3.0"
TESTED_PYJIANYINGDRAFT_COMMIT = "c3318066d964744e2bfc66f75c71745fe8cea52a"
TESTED_JIANYING_MAJOR = 11
US = 1_000_000


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _to_us(value: float) -> int:
    return round(float(value) * US)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_draft_name(value: str) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "＊", value).strip().rstrip(".")
    if not cleaned:
        raise ValueError("JianYing draft name is empty after sanitization")
    return cleaned[:120]


def detect_draft_root(explicit: str | Path | None = None) -> Path:
    if explicit:
        candidates = [Path(explicit).expanduser()]
    elif os.getenv("JY_PROJECTS_ROOT"):
        candidates = [Path(os.environ["JY_PROJECTS_ROOT"]).expanduser()]
    elif os.getenv("LOCALAPPDATA"):
        candidates = [
            Path(os.environ["LOCALAPPDATA"])
            / "JianyingPro"
            / "User Data"
            / "Projects"
            / "com.lveditor.draft"
        ]
    else:
        candidates = []
    valid = [path.resolve() for path in candidates if path.is_dir() and (path / "root_meta_info.json").is_file()]
    if len(valid) != 1:
        raise RuntimeError(
            "Could not determine one JianYing draft root containing root_meta_info.json; "
            "pass --draft-root or set JY_PROJECTS_ROOT"
        )
    return valid[0]


def detect_jianying_version() -> dict[str, Any]:
    result: dict[str, Any] = {"display_version": None, "executable": None, "tested_major_match": False}
    if os.name != "nt":
        return result
    try:
        import winreg

        roots = (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE)
        paths = (
            r"Software\Microsoft\Windows\CurrentVersion\Uninstall\JianyingPro",
            r"Software\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall\JianyingPro",
        )
        for root in roots:
            for key_path in paths:
                try:
                    with winreg.OpenKey(root, key_path) as key:
                        result["display_version"] = winreg.QueryValueEx(key, "DisplayVersion")[0]
                        try:
                            display_icon = Path(winreg.QueryValueEx(key, "DisplayIcon")[0].strip('"'))
                            # JianYing's uninstall record often exposes uninst.exe as
                            # DisplayIcon.  Resolve the real editor binary from the
                            # versioned install directory instead of reporting that
                            # misleading path as the tested executable.
                            candidates = []
                            if display_icon.name.lower() == "jianyingpro.exe":
                                candidates.append(display_icon)
                            if result["display_version"]:
                                candidates.append(
                                    display_icon.parent
                                    / str(result["display_version"])
                                    / "JianyingPro.exe"
                                )
                            candidates.append(display_icon.parent / "JianyingPro.exe")
                            result["executable"] = next(
                                (str(candidate.resolve()) for candidate in candidates if candidate.is_file()),
                                None,
                            )
                        except OSError:
                            pass
                        break
                except OSError:
                    continue
            if result["display_version"]:
                break
    except (ImportError, OSError):
        pass
    version = str(result.get("display_version") or "")
    result["tested_major_match"] = version.split(".", 1)[0] == str(TESTED_JIANYING_MAJOR)
    return result


def _bgm_from_plan(plan: Mapping[str, Any], explicit: str | Path | None = None) -> Path:
    if explicit:
        path = Path(explicit).expanduser().resolve()
    else:
        tracks = plan.get("audio_tracks") if isinstance(plan.get("audio_tracks"), list) else []
        bgm_track = next(
            (
                track
                for track in tracks
                if isinstance(track, Mapping) and str(track.get("role") or "").lower() == "bgm"
            ),
            None,
        )
        if not bgm_track:
            raise RuntimeError("edit_decisions has no BGM audio track; pass --bgm for legacy plans")
        path = Path(str(bgm_track.get("source_path") or "")).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _project_settings(plan: Mapping[str, Any]) -> tuple[int, int, int, int]:
    project = plan.get("project") if isinstance(plan.get("project"), Mapping) else {}
    width = int(project.get("width") or 1920)
    height = int(project.get("height") or 1080)
    fps = round(float(project.get("fps") or 30))
    duration = float(project.get("timeline_duration") or max((s["timeline_end"] for s in plan["shots"]), default=0.0))
    if width <= 0 or height <= 0 or fps <= 0 or duration <= 0:
        raise ValueError("Invalid project dimensions, fps, or duration")
    return width, height, fps, _to_us(duration)


def _preflight(plan: Mapping[str, Any], bgm_path: Path) -> dict[str, Any]:
    validation = validate_edit_decisions(plan, require_sources=True)
    if not validation["passed"]:
        raise RuntimeError("Invalid edit decisions: " + "; ".join(validation["errors"]))
    _project_settings(plan)
    return {
        "edit_decisions": validation,
        "bgm_sha256": _sha256(bgm_path),
        "bgm_size": bgm_path.stat().st_size,
    }


def _backup_before_write(draft_root: Path, report_dir: Path) -> Path:
    stamp = time.strftime("%Y%m%dT%H%M%S")
    backup = report_dir / "jianying_backup_before_create" / stamp
    backup.mkdir(parents=True, exist_ok=False)
    shutil.copy2(draft_root / "root_meta_info.json", backup / "root_meta_info.json")
    _write_json(
        backup / "draft_root_inventory.json",
        {
            "draft_root": str(draft_root),
            "captured_at_local": stamp,
            "existing_draft_folders": sorted(path.name for path in draft_root.iterdir() if path.is_dir()),
        },
    )
    return backup


def _clip_settings(draft: Any, shot: Mapping[str, Any]) -> tuple[Any, list[dict[str, Any]]]:
    """Map only transformations with a stable native representation."""
    differences: list[dict[str, Any]] = []
    transform = shot.get("transform") if isinstance(shot.get("transform"), Mapping) else {}
    scale = transform.get("scale") if isinstance(transform.get("scale"), Mapping) else {}
    position = transform.get("position") if isinstance(transform.get("position"), Mapping) else {}
    kwargs = {
        "alpha": float(transform.get("opacity", 1.0)),
        "rotation": float(transform.get("rotation_degrees", 0.0)),
        "scale_x": float(scale.get("x", 1.0)),
        "scale_y": float(scale.get("y", 1.0)),
        "transform_x": float(position.get("x", 0.0)),
        "transform_y": float(position.get("y", 0.0)),
    }
    crop = transform.get("crop") if isinstance(transform.get("crop"), Mapping) else shot.get("crop_plan")
    if isinstance(crop, Mapping):
        rect = crop.get("crop_rect_norm")
        full = (
            isinstance(rect, list)
            and len(rect) == 4
            and all(abs(float(a) - b) <= 0.0001 for a, b in zip(rect, [0.0, 0.0, 1.0, 1.0]))
        )
        if crop.get("mode") not in {None, "fit", "full_frame"} or (rect and not full):
            differences.append(
                {
                    "shot_index": shot.get("shot_index"),
                    "field": "crop",
                    "status": "not_mapped",
                    "reason": "pyJianYingDraft 0.3.0 has no lossless source crop rectangle API",
                    "render_value": crop,
                }
            )
    return draft.ClipSettings(**kwargs), differences


def _write_compatibility_metadata(
    draft_path: Path,
    draft_root: Path,
    draft_name: str,
    duration_us: int,
) -> None:
    content_path = draft_path / "draft_content.json"
    content = _load_json(content_path)
    now_us = int(time.time() * US)
    content.update({"name": draft_name, "create_time": now_us, "update_time": now_us})
    _write_json(content_path, content)
    shutil.copy2(content_path, draft_path / "draft_info.json")

    meta_path = draft_path / "draft_meta_info.json"
    meta = _load_json(meta_path)
    meta.update(
        {
            "draft_id": str(uuid.uuid4()).upper(),
            "draft_name": draft_name,
            "draft_fold_path": draft_path.as_posix(),
            "draft_root_path": draft_root.as_posix(),
            "tm_duration": duration_us,
        }
    )
    _write_json(meta_path, meta)
    (draft_path / "draft_settings").write_text(
        "[General]\ndraft_create_time=0\ndraft_last_edit_time=0\nreal_edit_keys=1\nreal_edit_seconds=0\n",
        encoding="utf-8",
    )
    _write_json(draft_path / "key_value.json", {})


def validate_draft(
    draft_path: str | Path,
    plan: Mapping[str, Any],
    bgm_path: str | Path,
) -> dict[str, Any]:
    draft_path = Path(draft_path).resolve()
    bgm_path = Path(bgm_path).resolve()
    width, height, fps, duration_us = _project_settings(plan)
    content_path = draft_path / "draft_content.json"
    if not content_path.is_file():
        return {"passed": False, "errors": [f"missing {content_path}"], "checks": {}}
    data = _load_json(content_path)
    tracks = data.get("tracks") if isinstance(data.get("tracks"), list) else []
    video_tracks = [track for track in tracks if track.get("type") == "video"]
    audio_tracks = [track for track in tracks if track.get("type") == "audio"]
    video_segments = video_tracks[0].get("segments", []) if len(video_tracks) == 1 else []
    audio_segments = audio_tracks[0].get("segments", []) if len(audio_tracks) == 1 else []
    checks: dict[str, bool] = {
        "draft_folder_exists": draft_path.is_dir(),
        "draft_info_compat_exists": (draft_path / "draft_info.json").is_file(),
        "canvas_matches": data.get("canvas_config", {}).get("width") == width
        and data.get("canvas_config", {}).get("height") == height,
        "fps_matches": round(float(data.get("fps", fps))) == fps,
        "duration_matches": int(data.get("duration", -1)) == duration_us,
        "one_video_track": len(video_tracks) == 1,
        "one_bgm_track": len(audio_tracks) == 1,
        "video_segment_count": len(video_segments) == len(plan["shots"]),
        "bgm_segment_count": len(audio_segments) == 1,
    }

    materials = data.get("materials") if isinstance(data.get("materials"), Mapping) else {}
    video_materials = materials.get("videos") if isinstance(materials.get("videos"), list) else []
    audio_materials = materials.get("audios") if isinstance(materials.get("audios"), list) else []
    video_paths = {
        str(item.get("id")): Path(str(item.get("path") or item.get("media_path") or "")).resolve()
        for item in video_materials
    }
    audio_paths = [Path(str(item.get("path") or item.get("media_path") or "")).resolve() for item in audio_materials]
    mismatches: list[dict[str, Any]] = []
    segment_ids: list[str] = []
    for index, (shot, segment) in enumerate(zip(plan["shots"], video_segments)):
        target = segment.get("target_timerange", {})
        source = segment.get("source_timerange", {})
        expected_target_duration = _to_us(shot["duration"])
        expected_source_duration = round(expected_target_duration * float(shot["speed"]))
        actual_source_path = video_paths.get(str(segment.get("material_id")))
        expected_source_path = Path(shot["source_path"]).resolve()
        values = {
            "target_start": (int(target.get("start", 0)), _to_us(shot["timeline_start"])),
            "target_duration": (int(target.get("duration", -1)), expected_target_duration),
            "source_start": (int(source.get("start", 0)), _to_us(shot["source_start"])),
            "source_duration": (int(source.get("duration", -1)), expected_source_duration),
            "speed": (float(segment.get("speed", 1.0)), float(shot["speed"])),
            "source_path": (str(actual_source_path), str(expected_source_path)),
        }
        bad = {
            key: {"actual": actual, "expected": expected}
            for key, (actual, expected) in values.items()
            if actual != expected and not (key == "speed" and abs(float(actual) - float(expected)) < 1e-9)
        }
        if bad:
            mismatches.append({"shot_index": index, "fields": bad})
        segment_ids.append(str(segment.get("id")))

    checks["source_paths_exist"] = bool(video_paths) and all(path.is_file() for path in video_paths.values())
    checks["source_paths_ranges_speeds_match"] = not mismatches
    checks["segments_independently_editable"] = len(segment_ids) == len(plan["shots"]) == len(set(segment_ids))
    checks["bgm_path_matches"] = len(audio_paths) == 1 and audio_paths[0] == bgm_path
    if audio_segments:
        segment = audio_segments[0]
        target = segment.get("target_timerange", {})
        source = segment.get("source_timerange", {})
        checks["bgm_sync_matches"] = (
            int(target.get("start", 0)) == 0
            and int(target.get("duration", -1)) == duration_us
            and int(source.get("start", 0)) == 0
            and int(source.get("duration", -1)) == duration_us
            and abs(float(segment.get("speed", 1.0)) - 1.0) < 1e-9
        )
    else:
        checks["bgm_sync_matches"] = False

    return {
        "passed": all(checks.values()),
        "checks": checks,
        "mismatches": mismatches,
        "video_track_count": len(video_tracks),
        "audio_track_count": len(audio_tracks),
        "video_segment_count": len(video_segments),
        "unique_source_material_count": len(set(video_paths.values())),
        "independent_video_segment_count": len(set(segment_ids)),
        "timeline_duration_us": int(data.get("duration", -1)),
    }


def build_draft(
    edit_decisions_path: str | Path,
    draft_name: str,
    *,
    draft_root: str | Path | None = None,
    bgm_path: str | Path | None = None,
    report_path: str | Path | None = None,
) -> dict[str, Any]:
    import pyJianYingDraft as draft

    plan_path = Path(edit_decisions_path).resolve()
    plan = load_edit_decisions(plan_path, bgm_path=bgm_path)
    bgm = _bgm_from_plan(plan, bgm_path)
    preflight = _preflight(plan, bgm)
    width, height, fps, duration_us = _project_settings(plan)
    root = detect_draft_root(draft_root)
    name = _safe_draft_name(draft_name)
    target = root / name
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite existing JianYing draft: {target}")
    report_dir = Path(report_path).resolve().parent if report_path else plan_path.parent
    backup = _backup_before_write(root, report_dir)
    differences: list[dict[str, Any]] = []

    try:
        folder = draft.DraftFolder(str(root))
        script = folder.create_draft(
            name,
            width,
            height,
            fps,
            maintrack_adsorb=True,
            allow_replace=False,
        )
        main_track = script.append_track(draft.TrackSpec(draft.TrackType.video, "main_video"))
        bgm_track = script.append_track(draft.TrackSpec(draft.TrackType.audio, "bgm"))
        material_cache: dict[str, Any] = {}
        for shot in plan["shots"]:
            source = str(Path(shot["source_path"]).resolve())
            if source not in material_cache:
                material_cache[source] = draft.VideoMaterial(source, material_name=Path(source).stem)
            target_duration = _to_us(shot["duration"])
            settings, shot_differences = _clip_settings(draft, shot)
            differences.extend(shot_differences)
            segment = draft.VideoSegment(
                material_cache[source],
                draft.Timerange(_to_us(shot["timeline_start"]), target_duration),
                source_timerange=draft.Timerange(
                    _to_us(shot["source_start"]),
                    round(target_duration * float(shot["speed"])),
                ),
                speed=float(shot["speed"]),
                volume=0.0,
                change_pitch=False,
                clip_settings=settings,
            )
            script.add_segment(segment, main_track)

        audio = draft.AudioMaterial(str(bgm), material_name=bgm.stem)
        script.add_segment(
            draft.AudioSegment(
                audio,
                draft.Timerange(0, duration_us),
                source_timerange=draft.Timerange(0, duration_us),
                speed=1.0,
                volume=1.0,
                change_pitch=False,
            ),
            bgm_track,
        )
        script.save()
        _write_compatibility_metadata(target, root, name, duration_us)
    except Exception:
        if target.exists():
            shutil.move(str(target), str(backup / "failed_created_draft"))
        raise

    validation = validate_draft(target, plan, bgm)
    jianying = detect_jianying_version()
    package_direct_url = Path(draft.__file__).resolve().parent.parent / "pyjianyingdraft-0.3.0.dist-info" / "direct_url.json"
    direct_url = _load_json(package_direct_url) if package_direct_url.is_file() else {}
    package_commit = str(direct_url.get("vcs_info", {}).get("commit_id") or "")
    report = {
        "passed": validation["passed"],
        "adapter_version": ADAPTER_VERSION,
        "draft_name": name,
        "draft_path": str(target),
        "draft_root": str(root),
        "backup_dir": str(backup),
        "timeline_truth": str(plan_path),
        "edit_schema_version": plan["schema_version"],
        "bgm_path": str(bgm),
        "preflight": preflight,
        "project": {
            "width": width,
            "height": height,
            "fps": fps,
            "duration_seconds": duration_us / US,
            "video_track": "main_video",
            "audio_track": "bgm",
        },
        "compatibility": {
            "jianying": jianying,
            "pyjianyingdraft_version": importlib.metadata.version("pyjianyingdraft"),
            "pyjianyingdraft_commit": package_commit or None,
            "tested_commit": TESTED_PYJIANYINGDRAFT_COMMIT,
            "tested_commit_match": package_commit == TESTED_PYJIANYINGDRAFT_COMMIT,
            "module_path": str(Path(draft.__file__).resolve()),
        },
        "editable_mapping": {
            "source_file_references": "direct references; media not copied",
            "individual_video_segments": "mapped",
            "source_and_timeline_ranges": "mapped in microseconds",
            "constant_speed": "mapped",
            "scale_position_rotation_opacity": "mapped through native ClipSettings",
            "bgm": "independent editable audio track",
            "transitions": "hard cuts preserved",
            "color_grade": "not mapped; FFmpeg grade parameters remain documented in edit_decisions",
        },
        "unmapped_or_approximate": differences,
        "validation": validation,
    }
    target_report = Path(report_path).resolve() if report_path else plan_path.with_name("jianying_draft_report.json")
    _write_json(target_report, report)
    report["report_path"] = str(target_report)
    if not validation["passed"]:
        raise RuntimeError(f"Draft created but structural validation failed: {target_report}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("edit_decisions", type=Path)
    parser.add_argument("--draft-name", required=True)
    parser.add_argument("--draft-root", type=Path)
    parser.add_argument("--bgm", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--validate-existing", type=Path, metavar="DRAFT_PATH")
    args = parser.parse_args()
    plan = load_edit_decisions(args.edit_decisions, bgm_path=args.bgm)
    bgm = _bgm_from_plan(plan, args.bgm)
    result = (
        validate_draft(args.validate_existing, plan, bgm)
        if args.validate_existing
        else build_draft(
            args.edit_decisions,
            args.draft_name,
            draft_root=args.draft_root,
            bgm_path=args.bgm,
            report_path=args.report,
        )
    )
    print(json.dumps({"ok": bool(result.get("passed")), "data": result}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(
            json.dumps(
                {"ok": False, "code": exc.__class__.__name__, "reason": str(exc), "data": {}},
                ensure_ascii=False,
                indent=2,
            ),
            file=sys.stderr,
        )
        raise
