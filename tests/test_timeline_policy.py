from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest


SKILL_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = SKILL_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import pixabay_pipeline as pipeline  # noqa: E402
from montage import (  # noqa: E402
    InsufficientMaterialError,
    build_timeline,
    parse_ratio,
    plan_subject_crop,
)
from visual_semantics import face_content_risk  # noqa: E402


SCENES = ("nature", "architecture", "transport", "industrial")
SCALES = ("wide", "medium", "detail")


def _assets(
    tmp_path: Path,
    count: int,
    *,
    scenes: tuple[str, ...] = SCENES,
    face_risks: list[float] | None = None,
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for index in range(count):
        path = tmp_path / f"素材-{index}.mp4"
        path.write_bytes(b"test fixture; timeline planning only")
        face_risk = face_risks[index] if face_risks is not None else 0.02
        result.append(
            {
                "pixabay_id": 10_000 + index,
                "local_path": str(path),
                "duration": 12.0,
                "width": 1920,
                "height": 1080,
                "score": 2.0 if face_risk >= 0.65 else 0.8 - index * 0.005,
                "motion_score": 0.15 + (index % 5) * 0.17,
                "shot_scale": SCALES[index % len(SCALES)],
                "scene_category": scenes[index % len(scenes)],
                "face_content_risk": face_risk,
                "tags": scenes[index % len(scenes)],
                "subject_profile": {
                    "center": {"x": 0.5, "y": 0.5},
                    "bbox": [0.30, 0.20, 0.70, 0.80],
                    "confidence": 0.9,
                },
            }
        )
    return result


def _audio(duration: float = 10.0) -> dict[str, object]:
    beats = [round(index * 0.5, 3) for index in range(1, int(duration * 2))]
    return {
        "beat_times_seconds": beats,
        "accent_times_seconds": [value for value in beats if round(value) == value],
        "phrase_boundaries_seconds": [4.0, 8.0],
        "section_boundaries_seconds": [5.0],
        "energy_curve": [
            {"time": 0.0, "value": 0.5},
            {"time": duration, "value": 0.5},
        ],
        "sections": [
            {"index": 0, "start": 0.0, "end": duration / 2, "role": "opening"},
            {"index": 1, "start": duration / 2, "end": duration, "role": "ending"},
        ],
    }


def _relaxed_policy(**overrides: object) -> dict[str, object]:
    return {
        "min_unique_assets": 3,
        "max_reuse_per_asset": 4,
        "max_asset_screen_share": 0.45,
        "min_scene_categories": 2,
        "max_prominent_face_screen_share": 0.15,
        "prominent_face_threshold": 0.65,
        **overrides,
    }


@pytest.mark.parametrize(
    ("assets_count", "scenes", "policy", "message"),
    [
        (
            1,
            ("nature",),
            _relaxed_policy(min_unique_assets=2, min_scene_categories=1),
            "independent assets 1 < 2",
        ),
        (
            5,
            ("nature",),
            _relaxed_policy(min_unique_assets=3, min_scene_categories=2),
            "scene categories 1 < 2",
        ),
    ],
)
def test_independent_asset_or_scene_shortage_fails_explicitly(
    tmp_path: Path,
    assets_count: int,
    scenes: tuple[str, ...],
    policy: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(InsufficientMaterialError, match=message):
        build_timeline(
            _audio(),
            _assets(tmp_path, assets_count, scenes=scenes),
            10.0,
            content_policy=policy,
        )


@pytest.mark.parametrize(
    "guard",
    [
        {"max_reuse_per_asset": 1, "max_asset_screen_share": 1.0},
        {"max_reuse_per_asset": 20, "max_asset_screen_share": 0.15},
    ],
)
def test_reuse_and_single_asset_share_are_hard_failures(
    tmp_path: Path, guard: dict[str, object]
) -> None:
    policy = _relaxed_policy(
        min_unique_assets=1,
        min_scene_categories=1,
        max_prominent_face_screen_share=1.0,
        **guard,
    )
    with pytest.raises(InsufficientMaterialError, match="constraints became infeasible"):
        build_timeline(_audio(8.0), _assets(tmp_path, 1), 8.0, content_policy=policy)


def test_successful_plan_never_exceeds_reuse_share_or_face_budgets(tmp_path: Path) -> None:
    face_risks = [0.95, 0.90] + [0.02] * 10
    policy = _relaxed_policy(
        min_unique_assets=5,
        max_reuse_per_asset=2,
        max_asset_screen_share=0.25,
        min_scene_categories=3,
        max_prominent_face_screen_share=0.16,
    )
    plan = build_timeline(
        _audio(10.0),
        _assets(tmp_path, len(face_risks), face_risks=face_risks),
        10.0,
        content_policy=policy,
        seed="hard-policy",
    )

    by_asset: dict[str, float] = {}
    face_seconds = 0.0
    for shot in plan["shots"]:
        asset_id = str(shot["asset_id"])
        by_asset[asset_id] = by_asset.get(asset_id, 0.0) + float(shot["output_duration"])
        if float(shot["face_content_risk"]) >= 0.65:
            face_seconds += float(shot["output_duration"])

    assert max(plan["asset_usage_counts"].values()) <= 2
    assert max(by_asset.values()) / 10.0 <= 0.255
    assert face_seconds / 10.0 <= 0.165
    assert plan["sufficiency"]["max_asset_screen_share_actual"] <= 0.255
    assert plan["sufficiency"]["prominent_face_screen_share_actual"] <= 0.165


def test_face_budget_is_a_hard_feasibility_constraint(tmp_path: Path) -> None:
    assets = _assets(tmp_path, 8, face_risks=[0.02] * 4 + [0.95] * 4)
    common = _relaxed_policy(
        min_unique_assets=4,
        max_reuse_per_asset=1,
        max_asset_screen_share=0.30,
        min_scene_categories=3,
    )

    # The same pool is feasible when faces are unrestricted, proving that the
    # constrained failure below is specifically caused by the face-time budget.
    unrestricted = build_timeline(
        _audio(10.0),
        assets,
        10.0,
        content_policy={**common, "max_prominent_face_screen_share": 1.0},
        seed="face-budget",
    )
    assert unrestricted["shots"]

    with pytest.raises(InsufficientMaterialError, match="constraints became infeasible"):
        build_timeline(
            _audio(10.0),
            assets,
            10.0,
            content_policy={**common, "max_prominent_face_screen_share": 0.01},
            seed="face-budget",
        )


def test_editing_grammar_counterfactual_changes_timing_scale_path_and_ending(
    tmp_path: Path,
) -> None:
    assets = _assets(tmp_path, 12)
    grammar_fast_detail = {
        "event_weights": {"strong_beat": 1.0, "weak_beat": 0.1, "phrase": 0.1},
        "energy_shot_duration_seconds": {"low": 0.70, "mid": 0.70, "high": 0.55},
        "scale_transition_matrix": {
            "wide": {"detail": 1.0},
            "medium": {"detail": 1.0},
            "detail": {"wide": 1.0},
        },
        "ending": {
            "final_shot_multiplier": 0.65,
            "fade_out_seconds": 0.15,
            "hold_last_frame": False,
        },
    }
    grammar_slow_wide = {
        "event_weights": {"strong_beat": 0.1, "weak_beat": 0.1, "phrase": 1.0},
        "energy_shot_duration_seconds": {"low": 2.60, "mid": 2.60, "high": 2.10},
        "scale_transition_matrix": {
            "wide": {"wide": 1.0},
            "medium": {"wide": 1.0},
            "detail": {"wide": 1.0},
        },
        "ending": {
            "final_shot_multiplier": 1.75,
            "fade_out_seconds": 0.75,
            "hold_last_frame": True,
        },
    }
    kwargs = {
        "audio_profile": _audio(12.0),
        "media_result": assets,
        "duration": 12.0,
        "content_policy": _relaxed_policy(
            min_unique_assets=3,
            max_reuse_per_asset=4,
            max_asset_screen_share=0.45,
            min_scene_categories=2,
        ),
        "seed": "grammar-counterfactual",
    }
    fast = build_timeline(**kwargs, editing_grammar=grammar_fast_detail)
    slow = build_timeline(**kwargs, editing_grammar=grammar_slow_wide)

    fast_boundaries = [shot["output_end"] for shot in fast["shots"][:-1]]
    slow_boundaries = [shot["output_end"] for shot in slow["shots"][:-1]]
    assert fast_boundaries != slow_boundaries
    assert len(fast["shots"]) > len(slow["shots"])
    assert fast["shots"][0]["desired_shot_role"] == "detail"
    assert slow["shots"][0]["desired_shot_role"] == "wide"
    assert fast["shots"][-1]["ending_structure"] != slow["shots"][-1]["ending_structure"]
    assert slow["shots"][-1]["ending_structure"]["hold_last_frame"] is True
    assert slow["shots"][-1]["ending_structure"]["visual_fade_out_seconds"] > 0.0


def test_subject_aware_portrait_crop_retains_left_subject_and_unsafe_cases_blur() -> None:
    spec = parse_ratio("9:16")
    safe_left = {
        "width": 1920,
        "height": 1080,
        "subject_profile": {
            "center": {"x": 0.11, "y": 0.5},
            "bbox": [0.02, 0.20, 0.20, 0.80],
            "confidence": 0.92,
        },
    }
    wide_subject = {
        "width": 1920,
        "height": 1080,
        "subject_profile": {
            "center": {"x": 0.5, "y": 0.5},
            "bbox": [0.04, 0.15, 0.96, 0.85],
            "confidence": 0.95,
        },
    }
    low_confidence = {
        "width": 1920,
        "height": 1080,
        "subject_profile": {
            "center": {"x": 0.5, "y": 0.5},
            "bbox": [0.42, 0.20, 0.58, 0.80],
            "confidence": 0.05,
        },
    }

    safe = plan_subject_crop(safe_left, spec)
    assert safe["mode"] == "subject_crop"
    assert safe["retention"] >= 0.85
    assert safe["crop_rect_norm"][0] == pytest.approx(0.0)
    assert plan_subject_crop(wide_subject, spec)["mode"] == "blur_fill"
    assert plan_subject_crop(low_confidence, spec)["mode"] == "blur_fill"


def _candidate(candidate_id: int, tags: str) -> dict[str, object]:
    return {
        "id": candidate_id,
        "pixabay_id": candidate_id,
        "page_url": f"https://pixabay.example/{candidate_id}",
        "tags": tags,
        "duration": 8.0,
        "likes": 50,
        "views": 5_000,
        "downloads": 1_000,
        "user": f"author-{candidate_id}",
        "matched_queries": ["city architecture"],
        "variant": {"url": "https://cdn.example/video.mp4", "width": 1920, "height": 1080},
        "raw": {"id": candidate_id},
    }


def test_manufacturing_is_not_man_and_face_risk_monotonically_lowers_scores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert pipeline._infer_shot_type("manufacturing factory machinery") != "medium_portrait"
    assert face_content_risk("manufacturing factory machinery", None, None) == 0.0

    tags = [
        "city architecture skyline",
        "city architecture person",
        "city architecture portrait selfie",
    ]
    metadata = [
        pipeline._metadata_score(_candidate(index, value), "city architecture", None, None, 16 / 9, (1280, 720))
        for index, value in enumerate(tags, start=1)
    ]
    metadata_scores = [score for score, _ in metadata]
    metadata_risks = [components["face_content_risk"] for _, components in metadata]
    assert metadata_risks[0] < metadata_risks[1] < metadata_risks[2]
    assert metadata_scores[0] > metadata_scores[1] > metadata_scores[2]

    visual_risks = {11: 0.0, 12: 0.45, 13: 0.95}

    def fake_thumbnail(_session, raw, _variant, _cache, _target):
        return {
            "sharpness_score": 0.8,
            "exposure_score": 0.8,
            "text_watermark_risk": 0.0,
            "color_score": 0.8,
            "face_content_risk": visual_risks[int(raw["id"])],
            "perceptual_hash": None,
        }

    monkeypatch.setattr(pipeline, "_get_thumbnail_signals", fake_thumbnail)
    candidates = [_candidate(index, "city architecture skyline") for index in (11, 12, 13)]
    scored = pipeline._score_candidates(
        object(), copy.deepcopy(candidates), "city architecture", None, None, tmp_path, 16 / 9, (1280, 720), 3
    )
    by_id = {int(item["id"]): item for item in scored}
    assert by_id[11]["face_content_risk"] < by_id[12]["face_content_risk"] < by_id[13]["face_content_risk"]
    assert by_id[11]["pre_score"] > by_id[12]["pre_score"] > by_id[13]["pre_score"]
