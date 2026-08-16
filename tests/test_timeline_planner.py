from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest


SKILL_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = SKILL_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from timeline_planner import build_timeline_slots, plan_timeline_slots  # noqa: E402


def _audiomap(mode: str = "phrase_flow") -> dict[str, object]:
    beats = [{"time": round(index * 0.5, 3), "strength": 0.7} for index in range(1, 20)]
    phrases = [
        {"time": 2.3, "strength": 0.82},
        {"time": 5.15, "strength": 0.88},
        {"time": 8.6, "strength": 0.84},
    ]
    return {
        "schema_version": "1.2",
        "artifact_type": "audiomap",
        "analysis_digest": "fixture-digest",
        "duration_seconds": 10.0,
        "rhythm_mode": {"mode": mode, "confidence": 0.9},
        "events": {
            "beats": beats,
            "downbeats": [beats[index] for index in (1, 5, 9, 13, 17)],
            "onsets": [],
            "accents": [{"time": 4.0, "strength": 0.82}],
            "hard_stops": [],
            "drops": [{"type": "drop", "time": 5.15, "strength": 0.96, "confidence": 0.9}],
            "surges": [],
            "climaxes": [{"type": "climax", "time": 8.6, "strength": 0.92, "confidence": 0.88}],
            "phrase_boundaries": phrases,
            "section_boundaries": [],
        },
        "key_moments": [
            {"type": "drop", "time": 5.15, "strength": 0.96},
            {"type": "climax", "time": 8.6, "strength": 0.92},
        ],
        "sections": [
            {
                "index": 0,
                "start": 0.0,
                "end": 10.0,
                "role": "climax",
                "estimated_mood": "expansive/flowing",
                "rhythm_mode": mode,
                "energy": {"mean": 0.68},
                "edit_guidance": {
                    "rhythm_mode": mode,
                    "cut_intensity": 0.72,
                    "recommended_shot_duration_seconds": [1.8, 3.2],
                },
            }
        ],
    }


def test_phrase_flow_slots_snap_to_irregular_real_events_and_are_not_equal_divisions() -> None:
    profile = _audiomap("phrase_flow")
    plan = plan_timeline_slots(profile)
    repeated = build_timeline_slots(profile)

    assert plan["schema_version"] == "1.2"
    assert plan["artifact_type"] == "timeline"
    assert plan["plan_digest"] == repeated["plan_digest"]
    boundaries = [float(slot["end"]) for slot in plan["slots"][:-1]]
    assert any(abs(value - 2.3) <= 0.01 for value in boundaries), boundaries
    assert any(abs(value - 5.15) <= 0.01 for value in boundaries), boundaries
    assert any(abs(value - 8.6) <= 0.01 for value in boundaries), boundaries
    durations = {round(float(slot["duration"]), 2) for slot in plan["slots"]}
    assert len(durations) >= 3
    assert all(slot["anchor_event"]["type"] != "beat" for slot in plan["slots"][:-1])
    assert plan["metrics"]["event_snap_ratio"] > 0.5

    required = {
        "start",
        "end",
        "section_role",
        "rhythm_mode",
        "mood",
        "energy_level",
        "recommended_content",
        "recommended_shot_scale",
        "recommended_motion",
        "is_emphasis",
        "anchor_event",
        "transition",
    }
    assert all(required <= set(slot) for slot in plan["slots"])
    drop_slot = next(slot for slot in plan["slots"] if slot["emphasis_event"] == "drop")
    assert drop_slot["recommended_shot_scale"] == "wide"
    assert drop_slot["recommended_motion"] == "strong_forward_or_aerial"


def test_beat_cut_can_use_beats_but_phrase_flow_never_falls_back_to_per_beat_cutting() -> None:
    beat_profile = _audiomap("beat_cut")
    phrase_profile = copy.deepcopy(beat_profile)
    phrase_profile["rhythm_mode"] = {"mode": "phrase_flow", "confidence": 0.9}
    phrase_profile["sections"][0]["rhythm_mode"] = "phrase_flow"
    phrase_profile["sections"][0]["edit_guidance"]["rhythm_mode"] = "phrase_flow"

    beat_plan = plan_timeline_slots(beat_profile)
    phrase_plan = plan_timeline_slots(phrase_profile)
    beat_anchors = {slot["anchor_event"]["type"] for slot in beat_plan["slots"]}
    phrase_anchors = {slot["anchor_event"]["type"] for slot in phrase_plan["slots"]}

    assert beat_anchors & {"beat", "downbeat", "accent"}
    assert "beat" not in phrase_anchors
    assert "downbeat" not in phrase_anchors
    assert beat_plan["plan_digest"] != phrase_plan["plan_digest"]


def test_timeline_duration_override_never_exceeds_available_audio() -> None:
    profile = _audiomap("phrase_flow")
    plan = plan_timeline_slots(profile, duration=6.25)
    assert plan["duration_seconds"] == pytest.approx(6.25)
    assert plan["slots"][0]["start"] == 0.0
    assert plan["slots"][-1]["end"] == pytest.approx(6.25)


def test_climax_role_is_denser_than_intro_with_the_same_raw_duration_guidance() -> None:
    profile = _audiomap("beat_cut")
    shared_guidance = {
        "rhythm_mode": "beat_cut",
        "cut_intensity": 0.6,
        "recommended_shot_duration_seconds": [0.6, 1.2],
    }
    profile["sections"] = [
        {
            "index": 0,
            "start": 0.0,
            "end": 5.0,
            "role": "intro",
            "rhythm_mode": "beat_cut",
            "energy": {"mean": 0.7},
            "edit_guidance": dict(shared_guidance),
        },
        {
            "index": 1,
            "start": 5.0,
            "end": 10.0,
            "role": "climax",
            "rhythm_mode": "beat_cut",
            "energy": {"mean": 0.7},
            "edit_guidance": dict(shared_guidance),
        },
    ]

    plan = plan_timeline_slots(profile)
    intro = [slot for slot in plan["slots"] if slot["section_role"] == "intro"]
    climax = [slot for slot in plan["slots"] if slot["section_role"] == "climax"]

    assert len(climax) > len(intro)
    assert sum(slot["duration"] for slot in climax) / len(climax) < (
        sum(slot["duration"] for slot in intro) / len(intro)
    )
    assert min(slot["cut_intensity"] for slot in climax) > max(
        slot["cut_intensity"] for slot in intro
    )


def test_terminal_remainder_is_merged_instead_of_creating_a_flash_shot() -> None:
    profile = _audiomap("beat_cut")
    profile["duration_seconds"] = 2.334
    profile["events"] = {
        "beats": [],
        "downbeats": [],
        "onsets": [],
        "accents": [],
        "hard_stops": [],
        "drops": [],
        "surges": [],
        "climaxes": [],
        "phrase_boundaries": [],
        "section_boundaries": [],
    }
    profile["key_moments"] = []
    profile["sections"] = [
        {
            "index": 0,
            "start": 0.0,
            "end": 2.334,
            "role": "build",
            "rhythm_mode": "beat_cut",
            "energy": {"mean": 0.5},
            "edit_guidance": {
                "rhythm_mode": "beat_cut",
                "cut_intensity": 0.5,
                "recommended_shot_duration_seconds": [0.8, 1.2],
            },
        }
    ]

    plan = plan_timeline_slots(profile)

    assert plan["slots"][-1]["end"] == pytest.approx(2.334)
    assert min(float(slot["duration"]) for slot in plan["slots"]) >= 0.50
    assert plan["metrics"]["minimum_slot_duration_seconds"] >= 0.50


def test_production_planner_applies_editing_grammar_counterfactually() -> None:
    profile = _audiomap("beat_cut")
    fast_detail = {
        "reliability": {"score": 1.0},
        "montage_policy": {
            "boundary_event_weights": {
                "strong_accent": 1.0,
                "weak_beat": 0.15,
                "phrase_boundary": 0.10,
            },
            "boundary_offsets_seconds": {"strong_accent": 0.12},
            "shot_duration_by_energy": {
                "high": {"median_seconds": 0.68, "median_beats": 1.0},
            },
            "scale_transition_matrix": {
                "wide": {"detail": 1.0},
                "detail": {"wide": 1.0},
                "medium": {"detail": 1.0},
            },
            "motion_transition_matrix": {
                "static_like": {"push_in": 1.0},
                "push_in": {"pan_right": 1.0},
                "pan_right": {"push_in": 1.0},
            },
            "transition_distribution": {"hard_cut": 1.0},
            "ending": {
                "last_shot_duration_multiplier": 0.65,
                "preferred_end_event": "strong_accent",
                "fade_out_seconds": 0.15,
                "hold_last_frame": False,
            },
        },
    }
    slow_wide = {
        "reliability": {"score": 1.0},
        "montage_policy": {
            "boundary_event_weights": {
                "strong_accent": 0.10,
                "weak_beat": 0.10,
                "phrase_boundary": 1.0,
            },
            "boundary_offsets_seconds": {"phrase_boundary": -0.08},
            "shot_duration_by_energy": {
                "high": {"median_seconds": 2.5, "median_beats": 4.0},
            },
            "scale_transition_matrix": {
                "wide": {"wide": 1.0},
                "medium": {"wide": 1.0},
                "detail": {"wide": 1.0},
            },
            "motion_transition_matrix": {"static_like": {"static_like": 1.0}},
            "transition_distribution": {"dissolve": 1.0},
            "ending": {
                "last_shot_duration_multiplier": 1.75,
                "preferred_end_event": "phrase_boundary",
                "fade_out_seconds": 0.75,
                "hold_last_frame": True,
            },
        },
    }

    fast = plan_timeline_slots(profile, editing_grammar=fast_detail)
    repeated_fast = plan_timeline_slots(profile, editing_grammar=fast_detail)
    slow = plan_timeline_slots(profile, editing_grammar=slow_wide)

    assert fast["plan_digest"] == repeated_fast["plan_digest"]
    assert fast["plan_digest"] != slow["plan_digest"]
    assert len(fast["slots"]) > len(slow["slots"])
    assert [slot["end"] for slot in fast["slots"][:-1]] != [
        slot["end"] for slot in slow["slots"][:-1]
    ]
    shifted = [
        slot["anchor_event"]
        for slot in fast["slots"]
        if slot["anchor_event"].get("learned_offset_seconds") == pytest.approx(0.12)
    ]
    assert shifted and all(
        anchor["time"] == pytest.approx(anchor["source_event_time"] + 0.12)
        for anchor in shifted
    )
    assert fast["slots"][0]["recommended_shot_scale"] == "detail"
    assert slow["slots"][0]["recommended_shot_scale"] == "wide"
    assert fast["slots"][0]["recommended_motion"] == "push_in"
    assert slow["slots"][0]["recommended_motion"] == "static_like"
    assert {slot["transition"] for slot in fast["slots"][1:-1]} == {"hard_cut"}
    assert {slot["transition"] for slot in slow["slots"][1:-1]} == {"dissolve"}
    assert fast["slots"][-1]["ending_target"]["last_shot_duration_multiplier"] == 0.65
    assert slow["slots"][-1]["ending_target"]["last_shot_duration_multiplier"] == 1.75
    assert fast["slots"][-1]["duration"] < slow["slots"][-1]["duration"]
    assert fast["editing_grammar_applied"] is True
    assert set(fast["editing_grammar_applied_fields"]) == {
        "boundary_event_weights",
        "boundary_offsets_seconds",
        "shot_duration_by_energy",
        "scale_transition_matrix",
        "motion_transition_matrix",
        "transition_distribution",
        "ending",
    }

    no_grammar = plan_timeline_slots(profile)
    empty_grammar = plan_timeline_slots(profile, editing_grammar={})
    assert no_grammar == empty_grammar

    unreliable = plan_timeline_slots(
        profile,
        editing_grammar={**fast_detail, "reliability": {"score": 0.0}},
    )
    assert [slot["end"] for slot in unreliable["slots"]] == [
        slot["end"] for slot in no_grammar["slots"]
    ]
    assert unreliable["slots"][-1]["ending_target"]["last_shot_duration_multiplier"] == 1.0
    assert unreliable["slots"][-1]["ending_target"]["preferred_end_event"] is None
    for slot in unreliable["slots"]:
        influence = slot.get("grammar_influence", {})
        if "boundary_event_weight" in influence:
            assert influence["boundary_event_weight"] == 1.0
        if "boundary_offset_seconds" in influence:
            assert influence["boundary_offset_seconds"] == 0.0
