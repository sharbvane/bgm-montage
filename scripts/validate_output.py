#!/usr/bin/env python3
"""Validate rendered media with ffprobe and full FFmpeg decode."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from montage import adjacent_diversity_issues, canonical_source_key, parse_ratio
from music_event_contract import normalize_music_event_contract
from visual_intelligence import evaluate_sequence_consistency


def _run(command: list[str], timeout: int = 600) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )


def probe_media(path: str | Path, ffprobe: str = "ffprobe") -> dict[str, Any]:
    process = _run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(Path(path).resolve()),
        ],
        timeout=60,
    )
    if process.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {process.stderr.strip()}")
    return json.loads(process.stdout)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_detector(stderr: str, pattern: str, fields: tuple[str, ...]) -> list[dict[str, float]]:
    records: list[dict[str, float]] = []
    for line in stderr.splitlines():
        if pattern not in line:
            continue
        record: dict[str, float] = {}
        for field in fields:
            match = re.search(rf"{re.escape(field)}:\s*([0-9.]+)", line)
            if match:
                record[field] = float(match.group(1))
        if record:
            records.append(record)
    return records


def _stream_duration(stream: dict[str, Any]) -> float:
    """Return a stream duration without falling back to container duration.

    Keeping stream and container durations separate catches files whose audio
    continues after the video stream has already ended.
    """

    try:
        value = float(stream.get("duration") or 0.0)
        if value > 0:
            return value
    except (TypeError, ValueError):
        pass
    try:
        duration_ts = float(stream.get("duration_ts") or 0.0)
        numerator, denominator = str(stream.get("time_base") or "0/1").split("/", 1)
        time_base = float(numerator) / float(denominator)
        value = duration_ts * time_base
        return value if value > 0 else 0.0
    except (TypeError, ValueError, ZeroDivisionError):
        return 0.0


def _detector_durations(
    records: list[dict[str, float]],
    start_field: str,
    end_field: str,
    duration_field: str,
    media_duration: float,
) -> list[float]:
    """Normalize detector events, including intervals left open at EOF."""

    durations: list[float] = []
    pending_start: float | None = None
    for record in records:
        if start_field in record:
            pending_start = max(0.0, float(record[start_field]))
        if duration_field in record:
            durations.append(max(0.0, float(record[duration_field])))
            pending_start = None
        elif end_field in record and pending_start is not None:
            durations.append(max(0.0, float(record[end_field]) - pending_start))
            pending_start = None
    if pending_start is not None and media_duration > pending_start:
        durations.append(media_duration - pending_start)
    return durations


def _load_payload(value: str | Path | dict[str, Any] | None) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    if value is None:
        return None
    try:
        payload = json.loads(Path(value).expanduser().resolve().read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else None
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def _fraction(value: Any) -> float:
    try:
        text = str(value or "0/1")
        if "/" in text:
            numerator, denominator = text.split("/", 1)
            return float(numerator) / max(float(denominator), 1e-9)
        return float(text)
    except (TypeError, ValueError, ZeroDivisionError):
        return 0.0


def _event_time(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, dict):
        for key in ("time", "time_seconds", "boundary", "start", "start_seconds"):
            if key in value:
                return _event_time(value[key])
    return None


def _collect_event_times(audiomap: dict[str, Any] | None, duration: float) -> dict[str, list[float]]:
    contract = normalize_music_event_contract(audiomap, duration)
    return {
        group: list(contract.get("groups", {}).get(group, []))
        for group in (
            "beats", "downbeats", "onsets", "accents", "hard_stops",
            "drops", "surges", "climaxes", "phrases", "sections",
        )
    }


def _volume_metrics(stderr: str) -> dict[str, float | None]:
    result: dict[str, float | None] = {"mean_volume_db": None, "max_volume_db": None}
    for key, label in (("mean_volume_db", "mean_volume"), ("max_volume_db", "max_volume")):
        match = re.search(rf"{label}:\s*(-?(?:inf|[0-9.]+))\s*dB", stderr, flags=re.IGNORECASE)
        if match:
            token = match.group(1).lower()
            result[key] = float("-inf") if token == "-inf" else float(token)
    return result


def _scene_change_times(stderr: str) -> list[float]:
    """Parse FFmpeg showinfo timestamps emitted by a scene-select filter."""

    values: list[float] = []
    for match in re.finditer(r"\bpts_time:\s*([0-9]+(?:\.[0-9]+)?)", stderr):
        try:
            values.append(float(match.group(1)))
        except ValueError:
            continue
    return sorted({round(value, 6) for value in values})


def _evenly_spaced(values: list[float], limit: int) -> list[float]:
    """Keep deterministic temporal coverage when review-frame work is capped."""

    ordered = sorted(set(values))
    if limit <= 0:
        return []
    if len(ordered) <= limit:
        return ordered
    if limit == 1:
        return [ordered[len(ordered) // 2]]
    indices = sorted({round(index * (len(ordered) - 1) / (limit - 1)) for index in range(limit)})
    return [ordered[index] for index in indices]


def _write_visual_review(
    output_dir: Path,
    *,
    media_sha256: str,
    duration: float,
    event_frames: list[dict[str, Any]],
) -> dict[str, Any]:
    """Write one machine-readable and one human-readable view of extracted evidence."""

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "visual_review.json"
    markdown_path = output_dir / "visual_review.md"
    entries: list[dict[str, Any]] = []
    pairs: dict[str, dict[str, Any]] = {}
    for index, frame in enumerate(event_frames, start=1):
        frame_path = Path(str(frame["path"])).resolve()
        try:
            relative_path = Path(os.path.relpath(frame_path, output_dir)).as_posix()
        except ValueError:
            relative_path = str(frame_path)
        details = [item for item in frame.get("review_items", []) if isinstance(item, dict)]
        evidence_types = sorted(
            {
                f"planned_cut_{item.get('side')}"
                if item.get("type") == "planned_cut" and item.get("side") in {"before", "after"}
                else str(item.get("type") or "event")
                for item in details
            }
        )
        reasons = list(dict.fromkeys(str(item.get("reason") or "Review event frame") for item in details))
        entry = {
            "index": index,
            "time_seconds": round(float(frame["time_seconds"]), 4),
            "evidence_types": evidence_types,
            "event_types": list(frame.get("event_types", [])),
            "reasons": reasons,
            "frame_path": str(frame_path),
            "relative_frame_path": relative_path,
            "details": details,
        }
        entries.append(entry)
        for item in details:
            pair_id = str(item.get("pair_id") or "")
            side = str(item.get("side") or "")
            if not pair_id or side not in {"before", "after"}:
                continue
            pair = pairs.setdefault(
                pair_id,
                {
                    "pair_id": pair_id,
                    "boundary_time_seconds": item.get("boundary_time_seconds"),
                    "left_shot_index": item.get("left_shot_index"),
                    "right_shot_index": item.get("right_shot_index"),
                },
            )
            pair[side] = {
                "time_seconds": entry["time_seconds"],
                "frame_path": entry["frame_path"],
                "relative_frame_path": entry["relative_frame_path"],
            }

    planned_cut_pairs = [pairs[key] for key in sorted(pairs)]
    payload = {
        "schema_version": "1.4.3",
        "artifact_type": "visual_review",
        "media_sha256": media_sha256,
        "duration_seconds": round(duration, 4),
        "artifacts": {"json": str(json_path), "markdown": str(markdown_path)},
        "summary": {
            "evidence_frame_count": len(entries),
            "planned_cut_pair_count": len(planned_cut_pairs),
            "complete_planned_cut_pair_count": sum(
                "before" in pair and "after" in pair for pair in planned_cut_pairs
            ),
        },
        "entries": entries,
        "planned_cut_pairs": planned_cut_pairs,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Visual Review Evidence",
        "",
        f"- Render SHA-256: `{media_sha256}`",
        f"- Duration: {duration:.4f}s",
        f"- Evidence frames: {len(entries)}",
        f"- Complete planned-cut pairs: {payload['summary']['complete_planned_cut_pair_count']}",
        "",
        "## Evidence frames",
        "",
        "| # | Time | Type | Reason | Frame |",
        "|---:|---:|---|---|---|",
    ]
    for entry in entries:
        types = ", ".join(entry["evidence_types"]).replace("|", "\\|")
        reasons = "; ".join(entry["reasons"]).replace("|", "\\|")
        lines.append(
            f"| {entry['index']} | {entry['time_seconds']:.4f}s | {types} | {reasons} | "
            f"![frame {entry['index']}](<{entry['relative_frame_path']}>) |"
        )
    lines.extend(
        [
            "",
            "## Planned cut pairs",
            "",
            "| Cut | Boundary | Before | After |",
            "|---|---:|---|---|",
        ]
    )
    for pair in planned_cut_pairs:
        before = pair.get("before")
        after = pair.get("after")
        before_link = (
            f"{before['time_seconds']:.4f}s ![before](<{before['relative_frame_path']}>)"
            if isinstance(before, dict)
            else "missing"
        )
        after_link = (
            f"{after['time_seconds']:.4f}s ![after](<{after['relative_frame_path']}>)"
            if isinstance(after, dict)
            else "missing"
        )
        lines.append(
            f"| {pair['pair_id']} | {float(pair.get('boundary_time_seconds') or 0.0):.4f}s | "
            f"{before_link} | {after_link} |"
        )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        **payload["artifacts"],
        **payload["summary"],
    }


def _source_interval_overlap_ratio(shots: list[dict[str, Any]]) -> float:
    by_source: dict[str, list[tuple[float, float]]] = {}
    maximum = 0.0
    for shot in shots:
        identity = str(shot.get("canonical_source_key") or canonical_source_key(shot))
        start = float(shot.get("source_start") or 0.0)
        end = float(shot.get("source_end") or start)
        for left, right in by_source.setdefault(identity, []):
            overlap = max(0.0, min(end, right) - max(start, left))
            maximum = max(maximum, overlap / max(1e-6, min(end - start, right - left)))
        by_source[identity].append((start, end))
    return maximum


def _visual_detail(shot: dict[str, Any], key: str) -> tuple[Any, bool]:
    for source in (
        shot.get("visual_features"),
        shot.get("metadata_features"),
        (shot.get("quality") or {}).get("visual_features") if isinstance(shot.get("quality"), dict) else None,
    ):
        if not isinstance(source, dict):
            continue
        details = source.get("feature_details") if isinstance(source.get("feature_details"), dict) else source
        detail = details.get(key) if isinstance(details.get(key), dict) else None
        if isinstance(detail, dict):
            return detail.get("value"), bool(detail.get("available"))
    if key == "shot_scale":
        value = shot.get("source_shot_scale") or shot.get("shot_scale")
        return value, bool(value and str(value).lower() not in {"unknown", "unresolved", "none"})
    return None, False


def _meaningful_visual_value(value: Any) -> bool:
    if isinstance(value, (list, tuple, set)):
        return bool(value)
    return value not in (None, "", "unknown", "unresolved", "none", "n/a")


def _visual_diversity_metrics(shots: list[dict[str, Any]], policy: dict[str, Any]) -> dict[str, Any]:
    pair_count = max(0, len(shots) - 1)
    issue_counts: dict[str, int] = {}
    pair_issue_sets: list[set[str]] = []
    comparable_pairs: dict[str, int] = {"same_shot_scale": 0, "same_motion_direction": 0}
    same_counts = {"same_shot_scale": 0, "same_motion_direction": 0}
    coverage: dict[str, dict[str, Any]] = {}
    for key in ("shot_scale", "world", "time_weather", "camera_language", "motion"):
        available = sum(1 for shot in shots if _visual_detail(shot, key)[1])
        coverage[key] = {
            "available_shots": available,
            "total_shots": len(shots),
            "coverage": round(available / max(1, len(shots)), 4),
        }
    for index in range(1, len(shots)):
        issues = adjacent_diversity_issues(shots[index - 1], shots[index])
        pair_issue_sets.append(set(issues))
        for issue in issues:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1
        for issue in ("same_shot_scale", "same_motion_direction"):
            if issue == "same_shot_scale":
                left = shots[index - 1].get("source_shot_scale") or shots[index - 1].get("shot_scale")
                right = shots[index].get("source_shot_scale") or shots[index].get("shot_scale")
            else:
                left, right = shots[index - 1].get("motion_direction"), shots[index].get("motion_direction")
            comparable = _meaningful_visual_value(left) and _meaningful_visual_value(right)
            if comparable:
                comparable_pairs[issue] += 1
                if str(left).lower() == str(right).lower():
                    same_counts[issue] += 1
    def longest_run(values: list[bool]) -> int:
        longest = current = 0
        for value in values:
            current = current + 1 if value else 0
            longest = max(longest, current)
        return longest

    rates = {
        "same_shot_scale": round(same_counts["same_shot_scale"] / max(1, comparable_pairs["same_shot_scale"]), 4),
        "same_motion_direction": round(same_counts["same_motion_direction"] / max(1, comparable_pairs["same_motion_direction"]), 4),
    }
    all_issue_types = sorted(set(issue_counts) | {"same_shot_scale", "same_motion_direction"})
    issue_summary = {}
    for issue in all_issue_types:
        count = same_counts.get(issue, issue_counts.get(issue, 0))
        denominator = comparable_pairs.get(issue, pair_count)
        issue_flags = [issue in pair_issues for pair_issues in pair_issue_sets]
        issue_summary[issue] = {
            "count": count,
            "pair_count": denominator,
            "rate": round(count / max(1, denominator), 4),
            "longest_run": longest_run(issue_flags),
        }
    scale_policy = {
        "mode": "hard_when_coverage_is_high",
        "coverage_required": 0.90,
        "max_rate": 0.85,
        "source": "v1.4.4_visual_diversity_contract",
    }
    scale_coverage = coverage["shot_scale"]["coverage"]
    scale_hard_fail = (
        comparable_pairs["same_shot_scale"] > 0
        and scale_coverage >= scale_policy["coverage_required"]
        and rates["same_shot_scale"] > scale_policy["max_rate"]
    )
    same_source_hard_fail = issue_counts.get("same_source", 0) > 0
    return {
        "schema_version": "visual-diversity.1",
        "pair_count": pair_count,
        "coverage": coverage,
        "issue_summary": issue_summary,
        "same_shot_scale": {
            **issue_summary["same_shot_scale"],
            "comparable_pairs": comparable_pairs["same_shot_scale"],
            "policy": scale_policy,
        },
        "same_motion_direction": {
            **issue_summary["same_motion_direction"],
            "comparable_pairs": comparable_pairs["same_motion_direction"],
        },
        "policy_decision": {
            "same_source": "hard_fail" if same_source_hard_fail else "pass",
            "same_shot_scale": "hard_fail" if scale_hard_fail else "advisory",
            "status": "hard_fail" if same_source_hard_fail or scale_hard_fail else "pass_or_advisory",
        },
        "passed": not same_source_hard_fail and not scale_hard_fail,
    }


def _duration_stage_report(
    plan_payload: dict[str, Any] | None,
    *,
    target_duration: float,
    container_duration: float,
    video_duration: float,
    audio_duration: float,
    video_stream: dict[str, Any],
    tolerance: float,
) -> dict[str, Any]:
    instrumentation = plan_payload.get("duration_instrumentation") if isinstance(plan_payload, dict) else None
    if not isinstance(instrumentation, dict):
        return {
            "schema_version": "duration-stages.1",
            "status": "not_available",
            "complete": False,
            "root_cause_status": "unverified",
            "missing": ["render_instrumentation"],
        }
    stages = instrumentation.get("stages") if isinstance(instrumentation.get("stages"), dict) else {}
    frame_grid = stages.get("frame_grid") if isinstance(stages.get("frame_grid"), dict) else {}
    filtergraph = stages.get("filtergraph") if isinstance(stages.get("filtergraph"), dict) else {}
    expected_frame_count = frame_grid.get("total_frame_count")
    observed_frame_count = video_stream.get("nb_frames")
    try:
        observed_frame_count = int(observed_frame_count) if observed_frame_count is not None else None
    except (TypeError, ValueError):
        observed_frame_count = None
    expected_filter_duration = float(filtergraph.get("expected_duration_seconds") or 0.0)
    expected_grid_duration = float(frame_grid.get("base_duration_seconds") or 0.0)
    observed = {
        "container_seconds": round(container_duration, 6),
        "video_stream_seconds": round(video_duration, 6),
        "audio_stream_seconds": round(audio_duration, 6),
        "video_frame_count": observed_frame_count,
    }
    deltas = {
        "container_vs_planned_seconds": round(container_duration - target_duration, 6),
        "video_vs_planned_seconds": round(video_duration - target_duration, 6),
        "audio_vs_planned_seconds": round(audio_duration - target_duration, 6),
        "filtergraph_vs_planned_seconds": round(expected_filter_duration - target_duration, 6),
        "frame_grid_vs_planned_seconds": round(expected_grid_duration - target_duration, 6),
    }
    mismatches = [
        name for name, value in (
            ("container", abs(deltas["container_vs_planned_seconds"])),
            ("video_stream", abs(deltas["video_vs_planned_seconds"])),
            ("audio_stream", abs(deltas["audio_vs_planned_seconds"])),
        ) if value > tolerance
    ]
    if expected_frame_count is not None and observed_frame_count is not None and expected_frame_count != observed_frame_count:
        mismatches.append("frame_count")
    missing = [
        name for name, value in (
            ("planned_timeline", stages.get("planned_timeline")),
            ("source_trim", stages.get("source_trim")),
            ("frame_grid", frame_grid),
            ("filtergraph", filtergraph),
            ("ffmpeg_encode", stages.get("ffmpeg_encode")),
        ) if not isinstance(value, dict) or value.get("status") in {"pending", "pending_validation"}
    ]
    complete = not missing and bool(stages.get("encoded_output")) and not isinstance(stages.get("encoded_output"), str)
    return {
        "schema_version": "duration-stages.1",
        "status": "complete" if complete else "incomplete",
        "complete": complete,
        "root_cause_status": "stage_mismatch_observed" if mismatches else "not_reproduced",
        "instrumentation_schema_version": instrumentation.get("schema_version"),
        "planned": stages.get("planned_timeline", {}),
        "source_trim": stages.get("source_trim", {}),
        "frame_grid": frame_grid,
        "filtergraph": filtergraph,
        "observed": observed,
        "deltas": deltas,
        "mismatches": mismatches,
        "missing": missing,
    }


def _cut_alignment(
    shots: list[dict[str, Any]],
    audiomap: dict[str, Any] | None,
    duration: float,
) -> dict[str, Any] | None:
    if not audiomap or len(shots) < 2:
        return None
    contract = normalize_music_event_contract(audiomap, duration)
    mode = str(contract.get("mode") or "phrase_flow")
    allowed = list(contract.get("allowed_times", []))
    tolerance = float(contract.get("tolerance_seconds") or 0.55)
    if not allowed:
        return {
            "mode": mode,
            "available": False,
            "passed": False,
            "reason": "no alignment events",
            "contract_schema_version": contract.get("schema_version"),
            "contract_digest": contract.get("contract_digest"),
            "allowed_event_types": contract.get("allowed_event_types", []),
        }
    offsets = []
    for shot in shots[:-1]:
        boundary = float(shot.get("output_end") or 0.0)
        nearest = min(allowed, key=lambda value: abs(value - boundary))
        offsets.append({"boundary": round(boundary, 5), "nearest_event": nearest, "error_seconds": round(abs(boundary - nearest), 5)})
    errors = [item["error_seconds"] for item in offsets]
    aligned_share = sum(error <= tolerance for error in errors) / max(1, len(errors))
    # A small number of energy-grid cuts between strong anchors is acceptable;
    # requiring every cut to hit a beat would recreate mechanical beat cutting.
    required_share = float(contract.get("required_aligned_share") or (0.70 if mode == "beat_cut" else 0.60))
    return {
        "mode": mode,
        "contract_schema_version": contract.get("schema_version"),
        "contract_digest": contract.get("contract_digest"),
        "allowed_event_types": contract.get("allowed_event_types", []),
        "available": True,
        "tolerance_seconds": tolerance,
        "required_aligned_share": required_share,
        "aligned_share": round(aligned_share, 4),
        "mean_error_seconds": round(sum(errors) / max(1, len(errors)), 5),
        "max_error_seconds": round(max(errors, default=0.0), 5),
        "boundaries": offsets,
        "passed": aligned_share >= required_share,
    }


def _climax_metrics(
    shots: list[dict[str, Any]],
    audiomap: dict[str, Any] | None,
    duration: float,
) -> dict[str, Any] | None:
    if not audiomap or not shots:
        return None

    def counted_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        # ponytail: exclude sub-0.5s bridge microshots from cut-density QA;
        # they are already covered by the dedicated microshot checks and can
        # otherwise make a short calm tail look denser than the musical peak.
        return [item for item in items if float(item.get("output_duration") or 0.0) >= 0.5]

    events = _collect_event_times(audiomap, duration)
    windows: list[tuple[float, float]] = []
    for item in audiomap.get("climaxes", []) if isinstance(audiomap.get("climaxes"), list) else []:
        if isinstance(item, dict) and item.get("start") is not None and item.get("end") is not None:
            windows.append((float(item["start"]), float(item["end"])))
    for time_value in events["climaxes"] + events["drops"]:
        windows.append((max(0.0, time_value - 1.2), min(duration, time_value + 1.8)))
    if not windows:
        counted_calm = counted_items(shots)
        calm_duration = sum(float(item.get("output_duration") or 0.0) for item in counted_calm)
        return {
            "available": True,
            "windows": [],
            "comparison_method": "no_climax_event_window",
            "section_role_event_window_coverage": 0.0,
            "comparison_window_coverage": 0.0,
            "evidence_sufficient": False,
            "status": "insufficient_evidence",
            "climax_shot_count": 0,
            "calm_shot_count": len(shots),
            "counted_climax_shot_count": 0,
            "counted_calm_shot_count": len(counted_calm),
            "excluded_microshots": [
                {
                    "index": int(shot.get("index", position)),
                    "duration_seconds": round(float(shot.get("output_duration") or 0.0), 4),
                    "side": "calm",
                    "reason": "bridge_microshot_excluded_from_density",
                }
                for position, shot in enumerate(shots)
                if float(shot.get("output_duration") or 0.0) < 0.5
            ],
            "climax_cut_density": None,
            "calm_cut_density": round(len(counted_calm) / max(calm_duration, 1e-6), 4) if counted_calm else None,
            "climax_visual_intensity": None,
            "calm_visual_intensity": None,
            "density_passed": None,
            "intensity_passed": None,
            "passed": False,
            "failure_reasons": ["insufficient_comparison_evidence"],
        }
    windows = sorted({(round(left, 4), round(right, 4)) for left, right in windows if right > left})

    def overlaps(shot: dict[str, Any], ranges: list[tuple[float, float]]) -> bool:
        start, end = float(shot.get("output_start") or 0.0), float(shot.get("output_end") or 0.0)
        return any(max(start, left) < min(end, right) for left, right in ranges)

    role_climax = [
        shot
        for shot in shots
        if str(shot.get("audio_section_role") or shot.get("section_role") or "").lower()
        in {"drop", "climax"}
    ]
    role_calm = [
        shot
        for shot in shots
        if str(shot.get("audio_section_role") or shot.get("section_role") or "").lower()
        in {"intro", "break", "outro"}
    ]
    # A section label is only a trustworthy proxy for the actual musical peak
    # when it covers most detected climax/drop windows.  Short tracks can have
    # one early ``drop`` section while later accents inside an ``outro`` carry
    # the strongest energy; blindly preferring section roles makes the QA
    # impossible to satisfy no matter how the later event-driven edit changes.
    role_window_coverage = (
        sum(any(overlaps(shot, [window]) for shot in role_climax) for window in windows)
        / max(1, len(windows))
        if role_climax
        else 0.0
    )
    role_window_coverage = min(1.0, role_window_coverage)
    if role_climax and role_calm and role_window_coverage >= 0.50:
        climax_shots = role_climax
        calm_shots = role_calm
        comparison_method = "audiomap_section_roles"
    else:
        climax_shots = [shot for shot in shots if overlaps(shot, windows)]
        calm_shots = [shot for shot in shots if not overlaps(shot, windows)]
        comparison_method = "drop_and_climax_event_windows"

    def density(items: list[dict[str, Any]]) -> float:
        counted = counted_items(items)
        total = sum(float(item.get("output_duration") or 0.0) for item in counted)
        return len(counted) / max(total, 1e-6)

    def intensity(items: list[dict[str, Any]]) -> float:
        values = []
        for item in items:
            motion = float(item.get("source_motion") or 0.0)
            emphasis = 0.14 if item.get("is_emphasis") else 0.0
            scale = 0.10 if str(item.get("source_shot_scale")) == "wide" else 0.04
            values.append(min(1.0, motion + emphasis + scale))
        return sum(values) / max(1, len(values))

    counted_climax, counted_calm = counted_items(climax_shots), counted_items(calm_shots)
    climax_density, calm_density = density(climax_shots), density(calm_shots)
    climax_intensity, calm_intensity = intensity(counted_climax), intensity(counted_calm)
    window_shot_coverage = sum(
        any(overlaps(shot, [window]) for shot in counted_climax)
        for window in windows
    ) / max(1, len(windows))
    excluded_microshots = [
        {
            "index": int(shot.get("index", position)),
            "duration_seconds": round(float(shot.get("output_duration") or 0.0), 4),
            "side": "climax" if shot in climax_shots else "calm",
            "reason": "bridge_microshot_excluded_from_density",
        }
        for position, shot in enumerate(shots)
        if float(shot.get("output_duration") or 0.0) < 0.5
    ]
    valid_climax_duration = sum(float(item.get("output_duration") or 0.0) for item in counted_climax)
    valid_calm_duration = sum(float(item.get("output_duration") or 0.0) for item in counted_calm)
    evidence_sufficient = (
        bool(counted_climax)
        and bool(counted_calm)
        and valid_climax_duration >= 0.5
        and valid_calm_duration >= 0.5
        and window_shot_coverage >= 0.50
    )
    density_passed = None
    intensity_passed = None
    if evidence_sufficient:
        density_passed = (
            climax_density > calm_density
            if comparison_method == "audiomap_section_roles"
            else climax_density + 0.05 >= calm_density * 0.90
        )
        intensity_passed = climax_intensity + 0.08 >= calm_intensity
    failure_reasons: list[str] = []
    if window_shot_coverage < 0.50:
        failure_reasons.append("insufficient_climax_window_coverage")
    if not evidence_sufficient:
        failure_reasons.append("insufficient_comparison_evidence")
    else:
        if not density_passed:
            failure_reasons.append("climax_density_not_higher")
        if not intensity_passed:
            failure_reasons.append("climax_visual_intensity_not_higher")
    return {
        "available": True,
        "windows": [[round(left, 4), round(right, 4)] for left, right in windows],
        "comparison_method": comparison_method,
        "section_role_event_window_coverage": round(role_window_coverage, 4),
        "comparison_window_coverage": round(window_shot_coverage, 4),
        "evidence_sufficient": evidence_sufficient,
        "status": "evaluated" if evidence_sufficient else "insufficient_evidence",
        "climax_shot_count": len(climax_shots),
        "calm_shot_count": len(calm_shots),
        "counted_climax_shot_count": len(counted_climax),
        "counted_calm_shot_count": len(counted_calm),
        "valid_climax_duration_seconds": round(valid_climax_duration, 4),
        "valid_calm_duration_seconds": round(valid_calm_duration, 4),
        "excluded_microshots": excluded_microshots,
        "climax_cut_density": round(climax_density, 4) if counted_climax else None,
        "calm_cut_density": round(calm_density, 4) if counted_calm else None,
        "climax_visual_intensity": round(climax_intensity, 4) if counted_climax else None,
        "calm_visual_intensity": round(calm_intensity, 4) if counted_calm else None,
        "density_passed": density_passed,
        "intensity_passed": intensity_passed,
        "passed": bool(evidence_sufficient and density_passed and intensity_passed),
        "failure_reasons": failure_reasons,
    }


def validate_output(
    media_path: str | Path,
    expected_duration: float | None = None,
    expected_ratio: str | None = None,
    report_path: str | Path | None = None,
    frames_dir: str | Path | None = None,
    edit_plan: str | Path | dict[str, Any] | None = None,
    ffmpeg: str = "ffmpeg",
    ffprobe: str = "ffprobe",
    audiomap: str | Path | dict[str, Any] | None = None,
    expected_fps: float | None = None,
) -> dict[str, Any]:
    path = Path(media_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if (not shutil.which(ffmpeg) and not Path(ffmpeg).is_file()) or (
        not shutil.which(ffprobe) and not Path(ffprobe).is_file()
    ):
        raise RuntimeError("ffmpeg and ffprobe must be available on PATH")
    probe = probe_media(path, ffprobe)
    streams = probe.get("streams", [])
    video_streams = [stream for stream in streams if stream.get("codec_type") == "video"]
    audio_streams = [stream for stream in streams if stream.get("codec_type") == "audio"]
    video = video_streams[0] if video_streams else {}
    duration = float(probe.get("format", {}).get("duration") or video.get("duration") or 0.0)
    video_duration = _stream_duration(video)
    audio_duration = _stream_duration(audio_streams[0]) if audio_streams else 0.0
    width, height = int(video.get("width") or 0), int(video.get("height") or 0)
    frame_rate = _fraction(video.get("avg_frame_rate") or video.get("r_frame_rate"))

    null_sink = "NUL" if __import__("os").name == "nt" else "/dev/null"
    decode = _run([ffmpeg, "-v", "error", "-i", str(path), "-map", "0:v:0", "-map", "0:a?", "-f", "null", null_sink])
    detectors = _run(
        [
            ffmpeg,
            "-hide_banner",
            "-nostats",
            "-i",
            str(path),
            "-vf",
            "blackdetect=d=0.25:pix_th=0.10,freezedetect=n=-55dB:d=1.5",
            "-af",
            "silencedetect=n=-50dB:d=1.0",
            "-f",
            "null",
            null_sink,
        ]
    )
    volume_process = _run(
        [
            ffmpeg,
            "-hide_banner",
            "-nostats",
            "-i",
            str(path),
            "-map",
            "0:a:0?",
            "-af",
            "volumedetect",
            "-f",
            "null",
            null_sink,
        ]
    )
    scene_process = _run(
        [
            ffmpeg,
            "-hide_banner",
            "-nostats",
            "-i",
            str(path),
            "-vf",
            "select='gt(scene,0.10)',showinfo",
            "-an",
            "-f",
            "null",
            null_sink,
        ]
    )
    volume = _volume_metrics(volume_process.stderr)
    scene_change_times = _scene_change_times(scene_process.stderr)
    black = _parse_detector(detectors.stderr, "black_", ("black_start", "black_end", "black_duration"))
    freeze = _parse_detector(detectors.stderr, "freeze_", ("freeze_start", "freeze_end", "freeze_duration"))
    silence = _parse_detector(detectors.stderr, "silence_", ("silence_start", "silence_end", "silence_duration"))

    black_durations = _detector_durations(black, "black_start", "black_end", "black_duration", duration)
    freeze_durations = _detector_durations(freeze, "freeze_start", "freeze_end", "freeze_duration", duration)
    silence_durations = _detector_durations(silence, "silence_start", "silence_end", "silence_duration", duration)
    # A montage that is short by several frames can hide a truncated terminal
    # shot even though the container is otherwise decodable.  Allow normal
    # codec timestamp rounding, but not a visibly missing tail.
    tolerance_frame_rate = frame_rate if frame_rate > 0 else float(expected_fps or 30.0)
    duration_tolerance = max(0.06, 2.0 / max(tolerance_frame_rate, 1.0))
    target_duration = expected_duration if expected_duration is not None else duration
    stream_duration_ok = target_duration > 0 and video_duration > 0 and abs(video_duration - target_duration) <= duration_tolerance
    audio_duration_ok = target_duration > 0 and audio_duration > 0 and abs(audio_duration - target_duration) <= duration_tolerance
    long_black_limit = max(0.55, duration * 0.12)
    total_black_limit = max(0.80, duration * 0.16)
    long_freeze_limit = max(1.60, duration * 0.18)
    total_freeze_limit = max(2.00, duration * 0.22)
    long_silence_limit = max(1.25, duration * 0.45)
    total_silence_limit = max(1.50, duration * 0.55)
    mean_volume = volume["mean_volume_db"]
    max_volume = volume["max_volume_db"]
    fps_target = float(expected_fps) if expected_fps is not None else None
    terminal_scene_seconds = (
        max(0.0, duration - scene_change_times[-1]) if scene_change_times else duration
    )
    terminal_scene_minimum = max(0.25, 8.0 / max(frame_rate, 24.0))

    checks = {
        "file_exists": path.is_file(),
        "nonempty": path.stat().st_size > 1024,
        "has_video": bool(video_streams),
        "has_audio": bool(audio_streams),
        "full_decode": decode.returncode == 0,
        "detectors_ran": detectors.returncode == 0,
        "scene_detector_ran": scene_process.returncode == 0,
        "duration": expected_duration is None
        or abs(duration - expected_duration) <= duration_tolerance,
        "video_stream_duration": stream_duration_ok,
        "audio_stream_duration": audio_duration_ok,
        "no_long_black": max(black_durations, default=0.0) < long_black_limit
        and sum(black_durations) < total_black_limit,
        "no_long_freeze": max(freeze_durations, default=0.0) < long_freeze_limit
        and sum(freeze_durations) < total_freeze_limit,
        "audio_not_entirely_silent": max(silence_durations, default=0.0) < long_silence_limit
        and sum(silence_durations) < total_silence_limit,
        "frame_rate": frame_rate > 0
        and (abs(frame_rate - fps_target) <= 0.55 if fps_target is not None else 12.0 <= frame_rate <= 120.0),
        "audio_level": bool(audio_streams)
        and mean_volume is not None
        and math.isfinite(float(mean_volume))
        and -42.0 <= float(mean_volume) <= -4.0,
        "audio_peak_safe": bool(audio_streams)
        and max_volume is not None
        and math.isfinite(float(max_volume))
        and -30.0 <= float(max_volume) <= 0.5,
        "no_terminal_microshot": terminal_scene_seconds + 1e-6 >= terminal_scene_minimum,
        "resolution": True,
    }
    non_blocking_checks: list[str] = []
    expected_dimensions: dict[str, int] | None = None
    if expected_ratio:
        spec = parse_ratio(expected_ratio)
        expected_dimensions = {"width": spec.width, "height": spec.height}
        checks["resolution"] = width == spec.width and height == spec.height

    plan_payload = _load_payload(edit_plan)
    audiomap_payload = _load_payload(audiomap)
    duration_stage_metrics = _duration_stage_report(
        plan_payload,
        target_duration=target_duration,
        container_duration=duration,
        video_duration=video_duration,
        audio_duration=audio_duration,
        video_stream=video,
        tolerance=duration_tolerance,
    )
    if edit_plan is not None:
        checks["edit_plan_readable"] = plan_payload is not None
        checks["duration_stage_instrumentation"] = bool(duration_stage_metrics.get("complete"))
    if audiomap is not None:
        checks["audiomap_readable"] = audiomap_payload is not None
    shots: list[dict[str, Any]] = []
    plan_metrics: dict[str, Any] | None = None
    alignment_metrics: dict[str, Any] | None = None
    climax_metrics: dict[str, Any] | None = None
    visual_consistency_metrics: dict[str, Any] | None = None
    visual_diversity_metrics: dict[str, Any] | None = None
    if plan_payload is not None:
        shots = [shot for shot in plan_payload.get("shots", []) if isinstance(shot, dict)]
        policy = plan_payload.get("content_policy", {}) if isinstance(plan_payload.get("content_policy"), dict) else {}
        counts: dict[str, int] = {}
        screen_time: dict[str, float] = {}
        occurrences: dict[str, list[tuple[int, float, float]]] = {}
        prominent_face_seconds = 0.0
        unsafe_crop = 0
        missing_sources: list[str] = []
        for shot in shots:
            identity = str(shot.get("canonical_source_key") or canonical_source_key(shot))
            shot_duration = float(shot.get("output_duration") or 0.0)
            counts[identity] = counts.get(identity, 0) + 1
            screen_time[identity] = screen_time.get(identity, 0.0) + shot_duration
            occurrences.setdefault(identity, []).append(
                (
                    int(shot.get("index", len(occurrences.get(identity, [])))),
                    float(shot.get("output_start") or 0.0),
                    float(shot.get("output_end") or 0.0),
                )
            )
            local = Path(str(shot.get("local_path") or "")).expanduser()
            if not local.is_file():
                missing_sources.append(str(local))
            if float(shot.get("face_content_risk") or 0.0) >= float(policy.get("prominent_face_threshold", 0.65)):
                prominent_face_seconds += shot_duration
            crop = shot.get("crop_plan", {}) if isinstance(shot.get("crop_plan"), dict) else {}
            mode = str(crop.get("mode", ""))
            retention = float(crop.get("retention", 0.0) or 0.0)
            if mode not in {"subject_crop", "blur_fill", "fit"} or (mode == "subject_crop" and retention < 0.85):
                unsafe_crop += 1
        unique = len(counts)
        max_reuse = max(counts.values(), default=0)
        max_share = max(screen_time.values(), default=0.0) / max(duration, 1e-6)
        face_share = prominent_face_seconds / max(duration, 1e-6)
        max_overlap_ratio = _source_interval_overlap_ratio(shots)
        repeat_gap_failures: list[dict[str, Any]] = []
        min_gap_shots = int(policy.get("min_repeat_gap_shots", 0))
        min_gap_seconds = float(policy.get("min_repeat_gap_seconds", 0.0))
        for identity, values in occurrences.items():
            values.sort()
            for previous, current in zip(values, values[1:]):
                if current[0] - previous[0] < min_gap_shots or current[1] - previous[2] < min_gap_seconds:
                    repeat_gap_failures.append(
                        {
                            "canonical_source_key": identity,
                            "previous_index": previous[0],
                            "current_index": current[0],
                            "gap_seconds": round(current[1] - previous[2], 5),
                        }
                    )
        diversity_issues = [
            {
                "left_index": index - 1,
                "right_index": index,
                "issues": adjacent_diversity_issues(shots[index - 1], shots[index]),
            }
            for index in range(1, len(shots))
            if adjacent_diversity_issues(shots[index - 1], shots[index])
        ]
        severe_limit = int(policy.get("max_adjacent_similarity_dimensions", 3))
        visual_diversity_metrics = _visual_diversity_metrics(shots, policy)
        checks["material_repetition"] = (
            unique >= int(policy.get("min_unique_assets", 1))
            and max_reuse <= int(policy.get("max_reuse_per_asset", max_reuse or 1))
            and max_share <= float(policy.get("max_asset_screen_share", 1.0)) + 0.006
        )
        checks["source_paths_exist"] = not missing_sources
        checks["source_intervals_nonoverlap"] = max_overlap_ratio <= float(
            policy.get("max_source_interval_overlap", 0.02)
        ) + 0.006
        checks["repeat_gap"] = not repeat_gap_failures
        checks["adjacent_diversity"] = bool(visual_diversity_metrics.get("passed")) and not any(
            len(item["issues"]) > severe_limit + 2 for item in diversity_issues
        )
        checks["visual_diversity_reported"] = True
        checks["prominent_face_budget"] = face_share <= float(
            policy.get("max_prominent_face_screen_share", 1.0)
        ) + 0.006
        checks["subject_crop_safe"] = bool(shots) and unsafe_crop == 0
        planned_durations = [float(shot.get("output_duration") or 0.0) for shot in shots]
        checks["no_planned_microshots"] = bool(planned_durations) and min(planned_durations) >= 0.24
        planned_terminal_minimum = 0.40
        checks["no_terminal_planned_microshot"] = (
            bool(planned_durations) and planned_durations[-1] >= planned_terminal_minimum
        )
        alignment_metrics = _cut_alignment(shots, audiomap_payload, duration)
        if alignment_metrics is not None:
            checks["music_cut_alignment"] = bool(alignment_metrics.get("passed"))
        climax_metrics = _climax_metrics(shots, audiomap_payload, duration)
        if climax_metrics is not None:
            checks["climax_visual_response"] = bool(climax_metrics.get("passed"))
            checks["climax_evidence_sufficient"] = bool(climax_metrics.get("evidence_sufficient"))
            if climax_metrics.get("status") == "insufficient_evidence":
                # Keep the false evidence/result flags truthful, but do not
                # turn an inapplicable comparison into a whole-video failure.
                non_blocking_checks.extend(
                    ["climax_visual_response", "climax_evidence_sufficient"]
                )
        visual_profile = (
            plan_payload.get("visual_style_profile")
            if isinstance(plan_payload.get("visual_style_profile"), dict)
            else {}
        )
        if str(plan_payload.get("schema_version") or "") == "1.3" or visual_profile:
            visual_consistency_metrics = evaluate_sequence_consistency(shots, visual_profile)
            checks["visual_style_profile_present"] = bool(visual_profile)
            checks["visual_sequence_consistency"] = bool(visual_consistency_metrics.get("passed"))
        plan_metrics = {
            "shot_count": len(shots),
            "unique_asset_count": unique,
            "unique_canonical_source_count": unique,
            "repeat_shot_ratio": round(1.0 - unique / max(1, len(shots)), 4),
            "max_reuse_count": max_reuse,
            "max_asset_screen_share": round(max_share, 4),
            "prominent_face_screen_share": round(face_share, 4),
            "unsafe_crop_count": unsafe_crop,
            "maximum_source_interval_overlap_ratio": round(max_overlap_ratio, 5),
            "repeat_gap_failures": repeat_gap_failures,
            "missing_source_paths": missing_sources,
            "adjacent_diversity_issues": diversity_issues,
            "visual_diversity": visual_diversity_metrics,
            "minimum_planned_shot_seconds": round(min(planned_durations, default=0.0), 4),
            "terminal_planned_shot_seconds": round(planned_durations[-1], 4)
            if planned_durations
            else 0.0,
            "terminal_planned_minimum_seconds": planned_terminal_minimum,
            "visual_sequence_consistency": visual_consistency_metrics,
        }

    representative_frames: list[str] = []
    event_frames: list[dict[str, Any]] = []
    visual_review: dict[str, Any] | None = None
    media_sha256 = _sha256(path)
    if frames_dir:
        frames = Path(frames_dir).expanduser().resolve()
        frames.mkdir(parents=True, exist_ok=True)
        for index, fraction in enumerate((0.1, 0.5, 0.9), start=1):
            timestamp = max(0.0, duration * fraction)
            frame = frames / f"frame_{index}_{timestamp:.2f}s.jpg"
            extraction = _run(
                [ffmpeg, "-hide_banner", "-loglevel", "error", "-ss", f"{timestamp:.4f}", "-i", str(path), "-frames:v", "1", "-q:v", "2", "-y", str(frame)],
                timeout=60,
            )
            if extraction.returncode == 0 and frame.is_file():
                representative_frames.append(str(frame))
        checks["representative_frames"] = len(representative_frames) == 3
        events = _collect_event_times(audiomap_payload, duration)
        requested: dict[float, set[str]] = {}
        review_items: dict[float, list[dict[str, Any]]] = {}
        frame_interval = max(0.04, 1.0 / max(frame_rate, 24.0))
        safe_last_frame = max(0.0, duration - frame_interval)

        def shot_index_at(timestamp: float) -> int | None:
            for fallback, shot in enumerate(shots):
                start = float(shot.get("output_start") or 0.0)
                end = float(shot.get("output_end") or start)
                if start <= timestamp < end or (fallback == len(shots) - 1 and timestamp <= end):
                    return int(shot.get("index", fallback))
            return None

        def request(timestamp: float, label: str, detail: dict[str, Any]) -> float:
            value = round(min(safe_last_frame, max(0.0, timestamp)), 4)
            requested.setdefault(value, set()).add(label)
            review_items.setdefault(value, []).append(detail)
            return value

        request(
            0.0,
            "opening",
            {"type": "opening", "reason": "Pinned opening frame", "shot_index": shot_index_at(0.0)},
        )
        request(
            safe_last_frame,
            "ending",
            {
                "type": "ending",
                "reason": "Pinned final decodable frame",
                "shot_index": shot_index_at(safe_last_frame),
            },
        )
        for group in ("sections", "phrases", "drops", "surges", "climaxes", "hard_stops"):
            for timestamp in events[group]:
                request(
                    timestamp,
                    group,
                    {
                        "type": "music_event",
                        "event": group,
                        "reason": f"BGM {group.replace('_', ' ')} event",
                        "shot_index": shot_index_at(timestamp),
                    },
                )

        cut_indices = list(range(max(0, len(shots) - 1)))
        if len(cut_indices) > 5:
            cut_indices = [int(value) for value in _evenly_spaced([float(value) for value in cut_indices], 5)]
        cut_times: list[float] = []
        for index in cut_indices:
            left, right = shots[index], shots[index + 1]
            boundary = float(left.get("output_end") or right.get("output_start") or 0.0)
            left_start = float(left.get("output_start") or 0.0)
            right_end = float(right.get("output_end") or duration)
            offset = max(0.08, 2.0 / max(frame_rate, 24.0))
            before = max(left_start + frame_interval * 0.5, boundary - offset)
            after = min(safe_last_frame, right_end - frame_interval * 0.5, boundary + offset)
            if before >= boundary or after <= boundary:
                continue
            pair_id = f"cut_{index + 1:03d}"
            common = {
                "type": "planned_cut",
                "pair_id": pair_id,
                "boundary_time_seconds": round(boundary, 4),
                "left_shot_index": left.get("index", index),
                "right_shot_index": right.get("index", index + 1),
            }
            cut_times.append(
                request(
                    before,
                    "planned_cut_before",
                    {**common, "side": "before", "reason": "Frame immediately before planned cut"},
                )
            )
            cut_times.append(
                request(
                    after,
                    "planned_cut_after",
                    {**common, "side": "after", "reason": "Frame immediately after planned cut"},
                )
            )

        if duration > 0:
            randomizer = random.Random(int(media_sha256[:16], 16))
            for _ in range(2):
                request(
                    randomizer.uniform(0.05 * duration, 0.95 * duration),
                    "random",
                    {"type": "coverage_sample", "reason": "Deterministic distributed coverage sample"},
                )

        pinned = [value for value in requested if requested[value] & {"opening", "ending"}]
        selected = sorted(set(pinned + cut_times))
        for label in ("drops", "climaxes", "phrases", "sections", "surges", "hard_stops"):
            if len(selected) >= 24:
                break
            candidates = [
                value for value in requested
                if value not in selected and label in requested[value]
            ]
            selected.extend(_evenly_spaced(candidates, 1))
        for labels in (
            {"drops", "surges", "climaxes", "hard_stops"},
            {"sections", "phrases"},
            {"random"},
        ):
            remaining = 24 - len(selected)
            if remaining <= 0:
                break
            candidates = [
                value for value in requested
                if value not in selected and requested[value] & labels
            ]
            selected.extend(_evenly_spaced(candidates, remaining))
        ordered_times = sorted(set(selected))
        for index, timestamp in enumerate(ordered_times, start=1):
            event_name = "_".join(sorted(requested[timestamp]))
            frame = frames / f"event_{index:02d}_{event_name}_{timestamp:.2f}s.jpg"
            extraction = _run(
                [
                    ffmpeg, "-hide_banner", "-loglevel", "error", "-ss", f"{timestamp:.4f}",
                    "-i", str(path), "-frames:v", "1", "-q:v", "2", "-y", str(frame),
                ],
                timeout=60,
            )
            if extraction.returncode == 0 and frame.is_file():
                event_frames.append(
                    {
                        "time_seconds": timestamp,
                        "event_types": sorted(requested[timestamp]),
                        "path": str(frame),
                        "review_items": review_items[timestamp],
                    }
                )
        checks["event_frames"] = len(event_frames) == len(ordered_times)
        review_root = (
            Path(report_path).expanduser().resolve().parent
            if report_path
            else frames.parent
        )
        visual_review = _write_visual_review(
            review_root,
            media_sha256=media_sha256,
            duration=duration,
            event_frames=event_frames,
        )
        checks["visual_review_artifacts"] = (
            Path(str(visual_review["json"])).is_file()
            and Path(str(visual_review["markdown"])).is_file()
            and bool(event_frames)
        )

    report = {
        "schema_version": "1.3",
        "artifact_type": "render_report",
        "path": str(path),
        "sha256": media_sha256,
        "size_bytes": path.stat().st_size,
        "duration_seconds": round(duration, 4),
        "expected_duration_seconds": round(target_duration, 4),
        "duration_tolerance_seconds": round(duration_tolerance, 6),
        "duration_stages": duration_stage_metrics,
        "video": {
            "codec": video.get("codec_name"),
            "width": width,
            "height": height,
            "pixel_format": video.get("pix_fmt"),
            "frame_rate": video.get("avg_frame_rate"),
            "frame_rate_fps": round(frame_rate, 5),
            "duration_seconds": round(video_duration, 4),
        },
        "audio": {
            "codec": audio_streams[0].get("codec_name") if audio_streams else None,
            "sample_rate": audio_streams[0].get("sample_rate") if audio_streams else None,
            "channels": audio_streams[0].get("channels") if audio_streams else None,
            "duration_seconds": round(audio_duration, 4),
            "mean_volume_db": mean_volume,
            "max_volume_db": max_volume,
        },
        "expected_dimensions": expected_dimensions,
        "checks": checks,
        "passed": all(
            passed
            for name, passed in checks.items()
            if name not in non_blocking_checks
        ),
        "non_blocking_checks": non_blocking_checks,
        "decode_errors": decode.stderr.splitlines()[-20:],
        "detectors": {
            "black": black,
            "freeze": freeze,
            "silence": silence,
            "normalized_durations": {
                "black": [round(value, 6) for value in black_durations],
                "freeze": [round(value, 6) for value in freeze_durations],
                "silence": [round(value, 6) for value in silence_durations],
            },
            "scene_changes": scene_change_times,
            "terminal_scene_seconds": round(terminal_scene_seconds, 6),
            "terminal_scene_minimum_seconds": round(terminal_scene_minimum, 6),
        },
        "representative_frames": representative_frames,
        "event_frames": event_frames,
        "visual_review": visual_review,
        "edit_plan_metrics": plan_metrics,
        "music_cut_alignment": alignment_metrics,
        "climax_visual_response": climax_metrics,
        "visual_sequence_consistency": visual_consistency_metrics,
        "visual_diversity": visual_diversity_metrics,
    }
    if report_path:
        output = Path(report_path).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("media")
    parser.add_argument("--expected-duration", type=float)
    parser.add_argument("--ratio")
    parser.add_argument("--report")
    parser.add_argument("--frames-dir")
    parser.add_argument("--edit-plan")
    parser.add_argument("--audiomap")
    parser.add_argument("--expected-fps", type=float)
    args = parser.parse_args()
    result = validate_output(
        args.media,
        args.expected_duration,
        args.ratio,
        args.report,
        args.frames_dir,
        args.edit_plan,
        audiomap=args.audiomap,
        expected_fps=args.expected_fps,
    )
    print(json.dumps({"passed": result["passed"], "path": result["path"], "checks": result["checks"]}, ensure_ascii=False))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
