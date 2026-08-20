from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from edit_schema import normalize_edit_decisions, validate_edit_decisions
from jianying_export import validate_draft
from visual_intelligence import (
    aggregate_video_aesthetics,
    analysis_cache_valid,
    build_visual_style_profile,
    evaluate_sequence_consistency,
    metadata_profile_fit,
    plan_visual_search_queries,
    transition_match,
)
from montage import _colorbalance_filter


def _visual_quality(score: float, motion: str = "push_in") -> dict:
    return {
        "overall_score": score,
        "mean_hsv": {"hue_degrees": 210.0, "saturation": 0.28, "value": 0.46},
        "motion_direction": "right",
        "visual_analysis": {
            "aesthetic_score": score,
            "cinematic_score": score,
            "spatial_depth_score": score,
            "composition_quality_score": score,
            "visual_impact_score": score,
            "lighting_quality_score": score,
            "atmosphere_quality_score": score,
            "intrinsic_color_quality_score": score,
            "ordinary_travelogue_risk": 1.0 - score,
            "motion_type": motion,
        },
    }


def _asset(tags: str, score: float = 0.8, *, direction: str = "right", motion: str = "push_in") -> dict:
    quality = _visual_quality(score, motion)
    quality["motion_direction"] = direction
    return {
        "tags": tags,
        "shot_scale": "wide",
        "motion_direction": direction,
        "quality": quality,
        "mean_hsv": quality["mean_hsv"],
    }


def test_dynamic_query_planning_changes_with_task_and_does_not_inject_place_whitelists() -> None:
    mountain = build_visual_style_profile(
        "高山远征",
        {"mood": "moody", "lighting": "fog dawn", "camera": "drone push in"},
        {"emotion": "epic"},
        "Patagonia cliffs, cold blue grey, large spatial depth",
    )
    kitchen = build_visual_style_profile(
        "手工料理",
        {"mood": "warm intimate", "lighting": "soft studio light", "camera": "macro tracking"},
        {"emotion": "playful"},
        "warm amber, tactile close-up cooking",
    )
    mountain_queries = [item["query"] for item in plan_visual_search_queries(mountain, 0)]
    kitchen_queries = [item["query"] for item in plan_visual_search_queries(kitchen, 0)]
    assert len(mountain_queries) >= 3
    assert len(kitchen_queries) >= 3
    assert mountain_queries != kitchen_queries
    assert "food" in " ".join(kitchen_queries).lower()
    assert all(len(query) <= 100 for query in [*mountain_queries, *kitchen_queries])
    joined = " ".join(mountain_queries).lower()
    assert "patagonia" in joined
    assert not any(place in joined for place in ("iceland", "faroe", "lofoten"))

    cool_profile = build_visual_style_profile(
        "大纵深山地峡谷",
        None,
        {"rapid_energy_changes": True, "section_count": 4},
        "冷蓝灰、低饱和、航拍推进、雾气阴天",
    )
    assert cool_profile["color_profile"]["hue_degrees"] == 205.0
    assert cool_profile["color_profile"]["saturation"] < 0.4
    assert "saturated" not in cool_profile["terms"]["color"]
    assert "true" not in cool_profile["terms"]["audio"]


def test_aesthetic_aggregation_prioritizes_depth_impact_and_cinematic_value() -> None:
    high = {
        "spatial_depth": 0.92,
        "composition_quality": 0.88,
        "visual_impact": 0.94,
        "lighting_quality": 0.86,
        "atmosphere_quality": 0.90,
        "color_quality": 0.82,
        "ordinary_travelogue_risk": 0.08,
    }
    low = {
        "spatial_depth": 0.22,
        "composition_quality": 0.30,
        "visual_impact": 0.24,
        "lighting_quality": 0.32,
        "atmosphere_quality": 0.20,
        "color_quality": 0.38,
        "ordinary_travelogue_risk": 0.88,
    }
    high_result = aggregate_video_aesthetics(
        [high] * 6,
        sharpness=0.9,
        motion_score=0.75,
        stability_score=0.9,
        motion_type="fpv_glide",
        resolution_score=1.0,
    )
    low_result = aggregate_video_aesthetics(
        [low] * 6,
        sharpness=0.7,
        motion_score=0.15,
        stability_score=0.7,
        motion_type="static",
        resolution_score=1.0,
    )
    assert high_result["aesthetic_score"] > low_result["aesthetic_score"] + 0.35
    assert high_result["cinematic_score"] > low_result["cinematic_score"] + 0.35


def test_dynamic_world_filter_transition_matching_and_sequence_consistency() -> None:
    profile = build_visual_style_profile(
        "mountain wilderness",
        {"palette": "cool blue grey", "lighting": "fog overcast", "camera": "drone push in"},
        {"emotion": "moody epic"},
        "remote mountain cliffs and mist, cool muted",
    )
    allowed = metadata_profile_fit("mountain cliff fog aerial cinematic", profile)
    incompatible = metadata_profile_fit("tropical resort palm sunny beach", profile)
    assert allowed["world_fit"] > incompatible["world_fit"]

    first = _asset("mountain cliff fog aerial rock", direction="right", motion="push_in")
    matched = _asset("mountain waterfall mist drone rock", direction="right", motion="push_in")
    mismatched = _asset("tropical palm beach sunny macro", direction="left", motion="static")
    mismatched["quality"]["mean_hsv"] = {
        "hue_degrees": 60.0,
        "saturation": 0.85,
        "value": 0.90,
    }
    matched_score = transition_match(first, matched, profile)
    mismatched_score = transition_match(first, mismatched, profile)
    assert matched_score["total"] > mismatched_score["total"]
    assert matched_score["motion"] > mismatched_score["motion"]
    coherent = evaluate_sequence_consistency([first, matched, _asset("mountain river clouds aerial rock")], profile)
    incoherent = evaluate_sequence_consistency([first, mismatched, _asset("city neon street night macro")], profile)
    assert coherent["score"] > incoherent["score"]


def test_visual_analysis_cache_invalidates_by_schema_engine_and_hash() -> None:
    current = {
        "analysis_cache": {
            "schema_version": 2,
            "engine_version": "1.3.0",
            "file_sha256": "abc",
        }
    }
    assert analysis_cache_valid(current, "abc")
    assert not analysis_cache_valid(current, "def")
    assert not analysis_cache_valid({"analysis_cache": {"schema_version": 1, "engine_version": "1.2"}})


def test_colorbalance_contract_disables_preserve_lightness_for_saturated_subjects() -> None:
    filter_text = _colorbalance_filter(
        {
            "rs": -0.0123,
            "gs": 0.0,
            "bs": 0.0123,
            "rm": -0.0088,
            "gm": 0.0,
            "bm": 0.0088,
            "rh": -0.0052,
            "gh": 0.0,
            "bh": 0.0052,
        }
    )
    assert filter_text.endswith(":pl=0")
    assert ":pl=1" not in filter_text


def test_edit_schema_migrates_v12_and_keeps_one_timeline_truth(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    bgm = tmp_path / "bgm.mp3"
    source.write_bytes(b"source")
    bgm.write_bytes(b"bgm")
    legacy = {
        "schema_version": "1.2",
        "duration_seconds": 2.0,
        "shots": [
            {
                "index": 0,
                "local_path": str(source),
                "output_start": 0.0,
                "output_end": 2.0,
                "output_duration": 2.0,
                "source_start": 1.0,
                "source_end": 3.0,
                "speed": 1.0,
                "crop_plan": {"mode": "fit", "crop_rect_norm": [0.0, 0.0, 1.0, 1.0]},
            }
        ],
    }
    migrated = normalize_edit_decisions(
        legacy,
        bgm_path=bgm,
        ratio="16:9",
        width=1920,
        height=1080,
        fps=30,
    )
    assert migrated["schema_version"] == "1.3"
    assert migrated["migrated_from_schema"] == "1.2"
    assert migrated["shots"][0]["source_path"] == migrated["shots"][0]["local_path"]
    assert migrated["shots"][0]["timeline_start"] == migrated["shots"][0]["output_start"]
    assert migrated["audio_tracks"][0]["source_path"] == str(bgm.resolve())
    assert validate_edit_decisions(migrated, require_sources=True)["passed"]


def test_structural_jianying_validation_uses_raw_sources_and_independent_segments(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    bgm = tmp_path / "bgm.mp3"
    source.write_bytes(b"source")
    bgm.write_bytes(b"bgm")
    plan = normalize_edit_decisions(
        {
            "shots": [
                {
                    "local_path": str(source),
                    "output_start": 0.0,
                    "output_end": 1.0,
                    "output_duration": 1.0,
                    "source_start": 2.0,
                    "source_end": 3.0,
                    "speed": 1.0,
                }
            ]
        },
        bgm_path=bgm,
        width=1920,
        height=1080,
        fps=30,
    )
    draft_dir = tmp_path / "draft"
    draft_dir.mkdir()
    (draft_dir / "draft_info.json").write_text("{}", encoding="utf-8")
    content = {
        "duration": 1_000_000,
        "fps": 30,
        "canvas_config": {"width": 1920, "height": 1080},
        "tracks": [
            {
                "type": "video",
                "name": "main_video",
                "segments": [
                    {
                        "id": "segment-video-1",
                        "material_id": "video-1",
                        "target_timerange": {"start": 0, "duration": 1_000_000},
                        "source_timerange": {"start": 2_000_000, "duration": 1_000_000},
                        "speed": 1.0,
                    }
                ],
            },
            {
                "type": "audio",
                "name": "bgm",
                "segments": [
                    {
                        "id": "segment-audio-1",
                        "material_id": "audio-1",
                        "target_timerange": {"start": 0, "duration": 1_000_000},
                        "source_timerange": {"start": 0, "duration": 1_000_000},
                        "speed": 1.0,
                    }
                ],
            },
        ],
        "materials": {
            "videos": [{"id": "video-1", "path": str(source)}],
            "audios": [{"id": "audio-1", "path": str(bgm)}],
        },
    }
    (draft_dir / "draft_content.json").write_text(json.dumps(content), encoding="utf-8")
    result = validate_draft(draft_dir, plan, bgm)
    assert result["passed"]
    assert result["video_segment_count"] == 1
    assert result["independent_video_segment_count"] == 1
