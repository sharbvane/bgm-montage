from __future__ import annotations

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
from validate_output import validate_output  # noqa: E402


FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")


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
    assert report["checks"]["music_cut_alignment"] is True
    assert report["checks"]["source_intervals_nonoverlap"] is True
    assert report["checks"]["event_frames"] is True
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
