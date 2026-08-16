from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace


SKILL_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = SKILL_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import analyze_references as references  # noqa: E402


def test_v11_shot_cache_is_enriched_without_inventing_complex_effects() -> None:
    analysis = {
        "schema_version": "1.1",
        "metadata": {"duration_seconds": 8.0},
        "editing_rhythm": {"detected_cuts": 3, "cuts_per_minute": 22.5},
        "camera_motion": {"distribution": {"static_like": 0.5, "pan_left_like": 0.5}},
        "shots": [
            {
                "index": index,
                "start_seconds": index * 2.0,
                "end_seconds": (index + 1) * 2.0,
                "duration_seconds": 2.0,
                "subject": f"subject-{index % 2}",
                "scene": f"scene-{index}",
                "shot_scale": ("wide_like", "medium_like", "close_up_like", "wide_like")[index],
                "composition": f"composition-{index % 3}",
                "camera_motion": ("static_like", "pan_left_like", "zoom_in_like", "pan_right_like")[index],
                "semantic": {"categories": {"scene": {"confidence": 0.6 + index * 0.05}}},
            }
            for index in range(4)
        ],
    }

    enriched = references._enrich_v12_analysis(analysis)

    assert enriched["schema_version"] == "1.2"
    assert enriched["editing_rhythm"]["shot_duration_distribution_seconds"]["count"] == 4
    assert len(enriched["editing_rhythm"]["pace_over_time"]) == 4
    assert enriched["camera_motion"]["direction_distribution"]["static_like"] == 0.25
    learned = enriched["editing_style_learning"]
    assert learned["adjacent_change_ratios"]["scene"] == 1.0
    assert learned["key_shot_positions"]
    assert "speed-ramp curve reconstruction" in learned["unsupported_effects"]
    assert "speed_ramp" not in learned.get("implemented_scope", [])


def test_scene_targets_are_bounded_and_keep_whole_duration_coverage() -> None:
    duration = 120.0
    uniform = references._sample_times(duration, 30.0, 3600)
    changes = [index * 0.5 for index in range(1, 240)]

    merged, audit = references._merge_scene_sample_times(uniform, changes, duration)

    assert len(merged) <= references.MAX_SAMPLES
    assert merged == sorted(merged)
    assert merged[0] <= 0.25
    assert merged[-1] >= duration - 0.25
    assert audit["scene_change_count"] == len(changes)
    assert audit["selected_scene_target_count"] <= references.MAX_SAMPLES - references.MIN_SAMPLES
    assert audit["merged_target_count"] == len(merged)


def test_scene_scan_parses_deduplicates_and_uses_full_duration_ffmpeg(monkeypatch) -> None:
    observed: dict[str, object] = {}

    def fake_run(command, **kwargs):
        observed["command"] = command
        observed["timeout"] = kwargs["timeout"]
        return SimpleNamespace(
            returncode=0,
            stderr=(
                "[Parsed_showinfo_2] n:1 pts:30 pts_time:1.000 pos:1\n"
                "[Parsed_showinfo_2] n:2 pts:31 pts_time:1.010 pos:2\n"
                "[Parsed_showinfo_2] n:3 pts:75 pts_time:2.500 pos:3\n"
            ),
        )

    monkeypatch.setattr(references.subprocess, "run", fake_run)
    candidates, audit = references._scene_change_times(
        Path("reference.mp4"),
        12.0,
        ffmpeg="ffmpeg",
    )

    command = observed["command"]
    assert candidates == [1.0, 2.5]
    assert audit["status"] == "ok"
    assert audit["raw_candidate_count"] == 3
    assert audit["deduplicated_candidate_count"] == 2
    assert "showinfo" in command[command.index("-vf") + 1]
    assert "-t" not in command
    assert "-vsync" not in command


def test_scene_scan_timeout_preserves_uniform_sampling(monkeypatch) -> None:
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])

    monkeypatch.setattr(references.subprocess, "run", timeout)
    candidates, audit = references._scene_change_times(
        Path("reference.mp4"),
        8.0,
        ffmpeg="ffmpeg",
    )
    uniform = references._sample_times(8.0, 30.0, 240)
    merged, merge_audit = references._merge_scene_sample_times(uniform, candidates, 8.0)

    assert candidates == []
    assert audit["status"] == "timeout"
    assert merged == uniform
    assert merge_audit["selected_uniform_target_count"] == len(uniform)
