from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


SKILL_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = SKILL_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from montage import (  # noqa: E402
    InsufficientMaterialError,
    _choose_boundary,
    _grammar_event_weights,
    build_timeline,
    render_timeline,
    timeline_diversity_issues,
)
from validate_output import _climax_metrics, _visual_diversity_metrics, validate_output  # noqa: E402
from visual_intelligence import build_visual_style_profile, evaluate_sequence_consistency  # noqa: E402


FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")


def test_climax_qa_uses_event_windows_when_early_drop_role_misses_late_peak() -> None:
    shots = [
        {
            "output_start": 0.0, "output_end": 2.0, "output_duration": 2.0,
            "audio_section_role": "drop", "source_motion": 0.30, "source_shot_scale": "wide",
        },
        {
            "output_start": 2.0, "output_end": 5.0, "output_duration": 3.0,
            "audio_section_role": "outro", "source_motion": 0.18, "source_shot_scale": "medium",
        },
        {
            "output_start": 5.0, "output_end": 7.5, "output_duration": 2.5,
            "audio_section_role": "outro", "source_motion": 0.92, "source_shot_scale": "wide",
            "is_emphasis": True,
        },
    ]
    audiomap = {"events": {"climaxes": [6.5], "drops": [6.5]}}
    metrics = _climax_metrics(shots, audiomap, 7.5)
    assert metrics is not None
    assert metrics["comparison_method"] == "drop_and_climax_event_windows"
    assert metrics["section_role_event_window_coverage"] < 0.50
    assert metrics["passed"] is True


def test_climax_comparison_passes_when_climax_is_stronger() -> None:
    shots = [
        {"output_start": 0.0, "output_end": 2.0, "output_duration": 2.0, "section_role": "intro", "source_motion": 0.1},
        {"output_start": 2.0, "output_end": 2.5, "output_duration": 0.5, "section_role": "climax", "source_motion": 0.8, "is_emphasis": True},
        {"output_start": 2.5, "output_end": 3.0, "output_duration": 0.5, "section_role": "climax", "source_motion": 0.9, "is_emphasis": True},
    ]
    metrics = _climax_metrics(shots, {"events": {"climaxes": [2.5], "drops": [2.5]}}, 3.0)
    assert metrics is not None
    assert metrics["evidence_sufficient"] is True
    assert metrics["density_passed"] is True
    assert metrics["intensity_passed"] is True
    assert metrics["passed"] is True


def test_climax_comparison_fails_when_climax_is_weaker() -> None:
    shots = [
        {"output_start": 0.0, "output_end": 0.5, "output_duration": 0.5, "section_role": "intro", "source_motion": 0.9, "is_emphasis": True},
        {"output_start": 0.5, "output_end": 2.5, "output_duration": 2.0, "section_role": "climax", "source_motion": 0.1},
    ]
    metrics = _climax_metrics(shots, {"events": {"climaxes": [1.5], "drops": [1.5]}}, 2.5)
    assert metrics is not None
    assert metrics["evidence_sufficient"] is True
    assert metrics["density_passed"] is False
    assert metrics["intensity_passed"] is False
    assert metrics["passed"] is False


def test_climax_comparison_is_insufficient_without_calm_reference() -> None:
    shots = [
        {"output_start": 0.0, "output_end": 1.0, "output_duration": 1.0, "section_role": "climax", "source_motion": 0.8},
        {"output_start": 1.0, "output_end": 2.0, "output_duration": 1.0, "section_role": "climax", "source_motion": 0.9},
    ]
    metrics = _climax_metrics(shots, {"events": {"climaxes": [1.0], "drops": [1.0]}}, 2.0)
    assert metrics is not None
    assert metrics["status"] == "insufficient_evidence"
    assert metrics["evidence_sufficient"] is False
    assert metrics["calm_shot_count"] == 0
    assert metrics["calm_cut_density"] is None
    assert metrics["calm_visual_intensity"] is None
    assert metrics["density_passed"] is None
    assert metrics["intensity_passed"] is None
    assert metrics["passed"] is False
    assert "insufficient_comparison_evidence" in metrics["failure_reasons"]


def test_climax_comparison_is_insufficient_with_calm_only() -> None:
    shots = [
        {"output_start": 0.0, "output_end": 2.0, "output_duration": 2.0, "section_role": "intro", "source_motion": 0.2},
        {"output_start": 2.0, "output_end": 4.0, "output_duration": 2.0, "section_role": "outro", "source_motion": 0.2},
    ]
    metrics = _climax_metrics(shots, {"events": {}}, 4.0)
    assert metrics is not None
    assert metrics["comparison_method"] == "no_climax_event_window"
    assert metrics["status"] == "insufficient_evidence"
    assert metrics["evidence_sufficient"] is False
    assert metrics["climax_shot_count"] == 0
    assert metrics["climax_cut_density"] is None
    assert metrics["climax_visual_intensity"] is None
    assert metrics["passed"] is False


def test_visual_diversity_reports_same_scale_rate_and_hard_policy() -> None:
    def shot(scale: str, direction: str = "right") -> dict[str, object]:
        return {
            "source_shot_scale": scale,
            "motion_direction": direction,
            "scene_category": scale,
            "subject_label": scale,
            "composition": scale,
            "color_tendency": scale,
            "is_static_like": False,
            "is_aerial": False,
            "visual_features": {
                "feature_details": {
                    "shot_scale": {"value": scale, "available": True},
                    "world": {"value": ["natural"], "available": True},
                    "time_weather": {"value": ["day"], "available": True},
                    "camera_language": {"value": "drift", "available": True},
                    "motion": {"value": "dynamic", "available": True},
                }
            },
        }

    metrics = _visual_diversity_metrics([shot("wide") for _ in range(4)], {})
    assert metrics["pair_count"] == 3
    assert metrics["same_shot_scale"]["count"] == 3
    assert metrics["same_shot_scale"]["rate"] == pytest.approx(1.0)
    assert metrics["policy_decision"]["same_shot_scale"] == "hard_fail"
    assert metrics["passed"] is False


def test_climax_density_reports_microshot_exclusions_and_evidence_coverage() -> None:
    shots = [
        {"index": 0, "output_start": 0.0, "output_end": 2.0, "output_duration": 2.0, "section_role": "intro", "source_motion": 0.2},
        {"index": 1, "output_start": 2.0, "output_end": 3.0, "output_duration": 1.0, "section_role": "drop", "source_motion": 0.7},
        {"index": 2, "output_start": 3.0, "output_end": 3.3, "output_duration": 0.3, "section_role": "drop", "source_motion": 0.9},
        {"index": 3, "output_start": 3.3, "output_end": 5.5, "output_duration": 2.2, "section_role": "drop", "source_motion": 0.95, "is_emphasis": True},
    ]
    metrics = _climax_metrics(shots, {"events": {"drops": [4.0]}}, 5.5)
    assert metrics is not None
    assert metrics["evidence_sufficient"] is True
    assert metrics["comparison_window_coverage"] == pytest.approx(1.0)
    assert metrics["counted_climax_shot_count"] < metrics["climax_shot_count"]
    assert metrics["excluded_microshots"][0]["reason"] == "bridge_microshot_excluded_from_density"


def test_climax_qa_marks_low_window_coverage_as_insufficient_evidence() -> None:
    shots = [
        {"index": 0, "output_start": 0.0, "output_end": 1.0, "output_duration": 1.0, "section_role": "intro", "source_motion": 0.2},
        {"index": 1, "output_start": 1.0, "output_end": 2.0, "output_duration": 1.0, "section_role": "outro", "source_motion": 0.2},
    ]
    metrics = _climax_metrics(shots, {"events": {"climaxes": [4.5]}}, 5.0)
    assert metrics is not None
    assert metrics["evidence_sufficient"] is False
    assert metrics["status"] == "insufficient_evidence"
    assert "insufficient_climax_window_coverage" in metrics["failure_reasons"]


def test_visual_diversity_excludes_unknown_motion_from_comparison() -> None:
    def shot(direction: str) -> dict[str, object]:
        return {
            "source_shot_scale": "wide",
            "motion_direction": direction,
            "is_static_like": False,
            "is_aerial": False,
        }

    metrics = _visual_diversity_metrics([shot("right"), shot("unknown"), shot("right")], {})
    assert metrics["same_motion_direction"]["comparable_pairs"] == 0
    assert metrics["same_motion_direction"]["count"] == 0


def test_sequence_consistency_marks_unavailable_world_and_weather_without_fallback_average() -> None:
    profile = build_visual_style_profile(
        "mountain wilderness",
        {"lighting": "fog overcast", "camera": "drone push in"},
        {"emotion": "moody epic"},
        "remote mountain cliffs and mist",
    )
    quality = {
        "overall_score": 0.8,
        "mean_hsv": {"hue_degrees": 210.0, "saturation": 0.28, "value": 0.46},
        "visual_analysis": {"aesthetic_score": 0.8, "cinematic_score": 0.8, "motion_type": "push_in"},
        "motion_type": "push_in",
    }
    shots = [
        {"quality": quality, "motion_direction": "right"},
        {"quality": quality, "motion_direction": "right"},
    ]
    metrics = evaluate_sequence_consistency(shots, profile)
    assert metrics["coverage"]["world"]["coverage"] == 0.0
    assert metrics["coverage"]["time_weather"]["coverage"] == 0.0
    assert metrics["dimension_available"]["world"] is False
    assert metrics["world_fit_average"] == 0.0
    pair = metrics["pair_scores"][0]
    assert pair["component_availability"]["world"] is False
    assert pair["effective_weight_sum"] < pair["full_weight_sum"]


def _asset(path: Path, index: int, *, fingerprint: str | None = None) -> dict:
    scenes = ("nature", "architecture", "transport", "industrial", "water_coast", "technology")
    scales = ("wide", "medium", "detail")
    directions = ("left", "right", "forward", "up", "backward", "down")
    compositions = ("centered", "rule_of_thirds", "symmetrical", "leading_lines")
    colors = ("warm", "cool", "neutral", "green", "blue", "amber")
    return {
        "pixabay_id": 12000 + index,
        "file_hash": fingerprint or f"sha-{index}",
        "local_path": str(path),
        "duration_seconds": 5.5,
        "width": 640,
        "height": 360,
        "score": 0.78,
        "motion_score": 0.15 + (index % 5) * 0.18,
        "motion_label": "pan" if index % 2 else "push",
        "motion_direction": directions[index % len(directions)],
        "shot_scale": scales[index % len(scales)],
        "scene_category": scenes[index % len(scenes)],
        "subject": f"subject-{index}",
        "composition": compositions[index % len(compositions)],
        "color_tendency": colors[index % len(colors)],
        "tags": f"{scenes[index % len(scenes)]} subject-{index} dynamic environment",
        "face_content_risk": 0.01,
        "stability_score": 0.86,
        "quality_score": 0.88,
        "usable_segments": [
            {"start": 0.55, "end": 4.95, "score": 0.90, "preferred_start": 0.9 + 0.12 * index}
        ],
        "subject_profile": {
            "center": {"x": 0.5, "y": 0.5},
            "bbox": [0.30, 0.20, 0.70, 0.80],
            "confidence": 0.9,
        },
    }


def _slots() -> list[dict]:
    scales = ("wide", "medium", "detail", "wide")
    motions = ("low", "medium", "high", "medium")
    transitions = ("hard_cut", "dissolve", "hard_cut", "hard_cut")
    return [
        {
            "index": index,
            "start": index * 1.5,
            "end": (index + 1) * 1.5,
            "duration": 1.5,
            "section_index": 0 if index < 2 else 1,
            "section_role": "intro" if index < 2 else "climax",
            "rhythm_mode": "beat_cut",
            "mood": "energetic" if index >= 2 else "calm",
            "energy": 0.22 if index < 2 else 0.88,
            "cut_intensity": "high" if index >= 2 else "low",
            "recommended_content": ["environment", "dynamic" if index >= 2 else "calm"],
            "recommended_shot_scale": scales[index],
            "recommended_motion": motions[index],
            "is_emphasis": index in {1, 2},
            "anchor_event": {"type": "accents", "time": (index + 1) * 1.5, "strength": 0.9},
            "transition": transitions[index],
        }
        for index in range(4)
    ]


def _audiomap() -> dict:
    beats = [{"time": value, "strength": 0.8} for value in [x * 0.5 for x in range(1, 12)]]
    return {
        "schema_version": "1.2",
        "duration_seconds": 6.0,
        "tempo": {"bpm": 120.0, "beat_period_seconds": 0.5, "confidence": 0.95},
        "rhythm_mode": {"mode": "beat_cut", "confidence": 0.95},
        "events": {
            "beats": beats,
            "downbeats": [1.5, 3.0, 4.5],
            "accents": [1.5, 3.0, 4.5],
            "section_boundaries": [3.0],
            "drops": [3.0],
            "surges": [4.5],
            "hard_stops": [],
            "climaxes": [3.5],
            "phrase_boundaries": [3.0],
        },
        "climaxes": [{"time": 3.5, "start": 3.0, "end": 4.5}],
        "sections": [
            {"index": 0, "start": 0.0, "end": 3.0, "role": "intro"},
            {"index": 1, "start": 3.0, "end": 6.0, "role": "climax"},
        ],
    }


def _policy(**overrides: object) -> dict:
    return {
        "min_unique_assets": 4,
        "max_reuse_per_asset": 1,
        "max_asset_screen_share": 0.26,
        "min_scene_categories": 3,
        "max_prominent_face_screen_share": 0.10,
        "prominent_face_threshold": 0.65,
        "min_repeat_gap_shots": 3,
        "min_repeat_gap_seconds": 3.0,
        "max_source_interval_overlap": 0.02,
        "max_adjacent_similarity_dimensions": 3,
        "max_soft_transition_share": 0.30,
        **overrides,
    }


def test_canonical_identity_prevents_aliases_from_satisfying_unique_gate(tmp_path: Path) -> None:
    first, second = tmp_path / "first.mp4", tmp_path / "alias.mp4"
    first.write_bytes(b"fixture")
    second.write_bytes(b"fixture")
    assets = [_asset(first, 1, fingerprint="same-source"), _asset(second, 2, fingerprint="same-source")]
    with pytest.raises(InsufficientMaterialError, match="independent assets 1 < 2"):
        build_timeline(
            _audiomap(),
            assets,
            6.0,
            slots=_slots(),
            content_policy=_policy(min_unique_assets=2, min_scene_categories=1),
        )


def test_slots_use_nonzero_nonoverlapping_best_segments_and_real_labels(tmp_path: Path) -> None:
    assets = []
    for index in range(8):
        path = tmp_path / f"asset-{index}.mp4"
        path.write_bytes(b"fixture")
        assets.append(_asset(path, index))
    plan = build_timeline(
        _audiomap(),
        assets,
        6.0,
        slots=_slots(),
        content_policy=_policy(),
        seed="v12-segments",
        theme="dynamic environment",
        ratio="16:9",
    )
    assert plan["schema_version"] == "1.3"
    assert plan["compatible_readers"] == ["1.2", "1.3"]
    assert plan["timeline_plan_applied"] is True
    assert len({shot["canonical_source_key"] for shot in plan["shots"]}) == 4
    assert all(shot["source_start"] >= 0.55 for shot in plan["shots"])
    assert all(0.75 <= shot["speed"] <= 1.35 for shot in plan["shots"])
    assert all(shot["source_motion_label"] in {"pan", "push"} for shot in plan["shots"])
    assert not any("same_source" in item["issues"] for item in timeline_diversity_issues(plan["shots"]))


def test_strong_accent_alias_changes_real_boundary_choice() -> None:
    musical = {"beats": [], "accents": [1.0], "phrases": [1.3], "sections": [], "pauses": []}
    strong = _grammar_event_weights({"event_weights": {"strong_accent": 1.0, "phrase": 0.1}})
    phrase = _grammar_event_weights({"event_weights": {"strong_accent": 0.1, "phrase": 1.0}})
    strong_boundary = _choose_boundary(0.0, 1.2, 3.0, musical, 0.70, strong, {})
    phrase_boundary = _choose_boundary(0.0, 1.2, 3.0, musical, 0.70, phrase, {})
    assert strong["accents"] == pytest.approx(1.0)
    assert phrase["accents"] == pytest.approx(0.1)
    assert strong_boundary == (1.0, "accents")
    assert phrase_boundary == (1.3, "phrases")


def _run(command: list[str]) -> None:
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode:
        raise AssertionError("\n".join(result.stderr.splitlines()[-20:]))


@pytest.mark.skipif(not FFMPEG or not FFPROBE, reason="requires ffmpeg/ffprobe")
def test_v12_real_dissolve_atomic_render_and_event_qa(tmp_path: Path) -> None:
    requested_duration = 6.013
    bgm = tmp_path / "bgm.wav"
    _run(
        [
            str(FFMPEG), "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
            f"sine=frequency=330:sample_rate=44100:duration={requested_duration}",
            "-c:a", "pcm_s16le", str(bgm),
        ]
    )
    assets = []
    for index in range(8):
        path = tmp_path / f"source {index}.mp4"
        _run(
            [
                str(FFMPEG), "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                f"testsrc2=size=640x360:rate=24:duration=5.5,format=yuv420p,hue=H={index * 35}",
                "-an", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "21", str(path),
            ]
        )
        assets.append(_asset(path, index))
    render_slots = _slots()
    boundaries = (0.0, 2.013, 3.513, 5.013, requested_duration)
    for index, slot in enumerate(render_slots):
        slot["start"] = boundaries[index]
        slot["end"] = boundaries[index + 1]
        slot["duration"] = boundaries[index + 1] - boundaries[index]
        slot["anchor_event"]["time"] = boundaries[index + 1]
    audiomap = _audiomap()
    audiomap["duration_seconds"] = requested_duration
    audiomap["events"].update(
        {
            "downbeats": [2.013, 3.513, 5.013],
            "accents": [2.013, 3.513, 5.013],
            "section_boundaries": [3.513],
            "drops": [3.513],
            "surges": [5.013],
            "climaxes": [4.013],
            "phrase_boundaries": [3.513],
        }
    )
    audiomap["climaxes"] = [{"time": 4.013, "start": 3.513, "end": 5.013}]
    audiomap["sections"] = [
        {"index": 0, "start": 0.0, "end": 3.513, "role": "intro"},
        {"index": 1, "start": 3.513, "end": requested_duration, "role": "climax"},
    ]
    plan = build_timeline(
        audiomap, assets, requested_duration, slots=render_slots,
        content_policy=_policy(max_asset_screen_share=0.34),
        seed="v12-render", theme="dynamic environment", ratio="640:360",
    )
    # Force the terminal choice to end one frame before the source EOF.  The
    # renderer must obtain the exact global frame count from decoded motion,
    # not by freezing the final frame when the demuxer's interval rounds down.
    terminal = plan["shots"][-1]
    terminal["source_end"] = 5.49
    terminal["source_start"] = 5.49 - terminal["output_duration"] * terminal["speed"]
    output = tmp_path / "rendered v1.2.mp4"
    render_timeline(plan, bgm, output, "640:360", ffmpeg=str(FFMPEG), fps=24)
    assert output.is_file() and output.stat().st_size > 20_000
    assert not list(tmp_path.glob(".*.part.mp4"))
    report = validate_output(
        output,
        expected_duration=requested_duration,
        expected_ratio="640:360",
        report_path=tmp_path / "render_report.json",
        frames_dir=tmp_path / "event frames",
        edit_plan=plan,
        ffmpeg=str(FFMPEG),
        ffprobe=str(FFPROBE),
        audiomap=audiomap,
        expected_fps=24,
    )
    assert report["passed"], {
        "failed_checks": [name for name, passed in report["checks"].items() if not passed],
        "duration": report.get("duration_seconds"),
        "video": report.get("video"),
        "audio": report.get("audio"),
        "terminal": report.get("detectors", {}).get("terminal_scene_seconds"),
    }
    assert report["checks"]["full_decode"] is True
    assert report["checks"]["duration_stage_instrumentation"] is True
    assert report["duration_stages"]["complete"] is True
    assert report["duration_stages"]["root_cause_status"] in {"not_reproduced", "stage_mismatch_observed"}
    assert report["checks"]["music_cut_alignment"] is True
    assert report["checks"]["source_intervals_nonoverlap"] is True
    assert report["checks"]["event_frames"] is True
    assert report["checks"]["visual_review_artifacts"] is True
    assert report["checks"]["climax_visual_response"] is True
    assert report["checks"]["no_terminal_microshot"] is True
    assert report["checks"]["no_terminal_planned_microshot"] is True
    assert report["video"]["pixel_format"] == "yuv420p"
    assert abs(float(report["video"]["duration_seconds"]) - requested_duration) <= (1.0 / 24.0)
    assert abs(float(report["audio"]["duration_seconds"]) - requested_duration) <= (1.0 / 24.0)
    counted = subprocess.run(
        [
            str(FFPROBE), "-v", "error", "-count_frames", "-select_streams", "v:0",
            "-show_entries", "stream=nb_read_frames", "-of", "default=nw=1:nk=1", str(output),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=True,
    )
    assert int(counted.stdout.strip()) == math.ceil(requested_duration * 24)
    assert report["music_cut_alignment"]["aligned_share"] == pytest.approx(1.0)
    assert report["detectors"]["terminal_scene_seconds"] >= 0.25
    assert len(report["event_frames"]) >= 6
    review = report["visual_review"]
    review_json = Path(review["json"])
    review_markdown = Path(review["markdown"])
    assert review_json.parent == (tmp_path / "render_report.json").parent
    assert review_json.is_file() and review_markdown.is_file()
    evidence = json.loads(review_json.read_text(encoding="utf-8"))
    evidence_types = {
        evidence_type
        for entry in evidence["entries"]
        for evidence_type in entry["evidence_types"]
    }
    assert {"opening", "ending", "music_event", "planned_cut_before", "planned_cut_after"} <= evidence_types
    sampled_events = {event for entry in evidence["entries"] for event in entry["event_types"]}
    assert {"drops", "climaxes", "phrases"} <= sampled_events
    assert all(
        any(detail.get("shot_index") is not None for detail in entry["details"])
        for entry in evidence["entries"]
        if "music_event" in entry["evidence_types"]
    )
    assert evidence["summary"]["complete_planned_cut_pair_count"] >= 1
    for pair in evidence["planned_cut_pairs"]:
        if "before" in pair and "after" in pair:
            assert Path(pair["before"]["frame_path"]).is_file()
            assert Path(pair["after"]["frame_path"]).is_file()
    assert "## Planned cut pairs" in review_markdown.read_text(encoding="utf-8")
    delivered = tmp_path / "delivered.mp4"
    output.replace(delivered)
    assert review_json.is_file() and all(Path(item["frame_path"]).is_file() for item in evidence["entries"])


@pytest.mark.skipif(not FFMPEG or not FFPROBE, reason="requires ffmpeg/ffprobe")
def test_quality_gate_rejects_real_four_frame_terminal_microshot(tmp_path: Path) -> None:
    output = tmp_path / "terminal-microshot.mp4"
    _run(
        [
            str(FFMPEG), "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=24:duration=1.84",
            "-f", "lavfi", "-i", "color=c=white:size=640x360:rate=24:duration=0.16",
            "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100:duration=2",
            "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[v]",
            "-map", "[v]", "-map", "2:a:0", "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-shortest", str(output),
        ]
    )
    report = validate_output(
        output,
        expected_duration=2.0,
        expected_ratio="640:360",
        ffmpeg=str(FFMPEG),
        ffprobe=str(FFPROBE),
        expected_fps=24,
    )
    assert report["checks"]["full_decode"] is True
    assert report["checks"]["duration"] is True
    assert report["checks"]["no_terminal_microshot"] is False
    assert report["passed"] is False
    assert report["detectors"]["terminal_scene_seconds"] < 0.25
