from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


SKILL_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = SKILL_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import bgm_montage  # noqa: E402
import youtube_first_pipeline as first  # noqa: E402
import youtube_pipeline as youtube  # noqa: E402


def _quality(score: float = 0.84) -> dict:
    return {
        "passed": True,
        "overall_score": score,
        "scene_category": "general",
        "visual_analysis": {
            "aesthetic_score": score,
            "cinematic_score": score,
            "spatial_depth_score": score,
            "visual_impact_score": score,
        },
    }


def _asset(index: int, provider: str, scene: str) -> dict:
    return {
        "id": f"{provider}-{index}",
        "asset_id": f"{provider}-{index}",
        "youtube_id": f"video{index:03d}" if provider == "youtube" else None,
        "provider": provider,
        "canonical_source_id": f"{provider}:{index}",
        "local_path": f"C:/fixtures/{provider}-{index}.mp4",
        "duration_seconds": 12.0,
        "scene_category": scene,
        "tags": f"arctic glacier cinematic {scene}",
        "quality": {**_quality(), "scene_category": scene},
    }


def test_default_provider_is_youtube_first() -> None:
    args = bgm_montage.build_parser().parse_args([
        "--bgm", "a.mp3", "--theme", "arctic landscape", "--duration", "8",
        "--ratio", "16:9", "--output-dir", "out",
    ])
    assert args.source_provider == "youtube-first"
    assert {"youtube-first", "youtube", "pixabay"} == set(
        next(action for action in bgm_montage.build_parser()._actions if action.dest == "source_provider").choices
    )


def test_queries_are_generated_without_search_query() -> None:
    _, plan = youtube.build_youtube_query_plan("arctic glacier mountains", {}, {}, priority_queries=[])
    assert len(plan) >= 3
    assert all(item["query"] for item in plan)


def test_priority_query_is_additive_and_first() -> None:
    _, plan = youtube.build_youtube_query_plan(
        "ocean coast at night", {}, {}, priority_queries=["moonlit rough ocean handheld"]
    )
    assert plan[0]["query"] == "moonlit rough ocean handheld"
    assert plan[0]["priority"] is True
    assert any(item["priority"] is False for item in plan)


def test_storm_queries_are_dynamic_from_task() -> None:
    _, plan = youtube.build_youtube_query_plan("massive shelf cloud storm over rural road", {}, {})
    joined = " ".join(item["query"] for item in plan).casefold()
    assert "storm" in joined or "shelf cloud" in joined


@pytest.mark.parametrize("theme", [
    "northern arctic glacier mountains",
    "ocean coast moonlight waves",
    "industrial factory machinery",
    "city night traffic neon",
])
def test_non_storm_themes_have_no_storm_contamination(theme: str) -> None:
    _, plan = youtube.build_youtube_query_plan(theme, {}, {})
    joined = " ".join(item["query"] for item in plan).casefold()
    for forbidden in ("shelf cloud", "supercell", "storm front", "wall cloud", "gust front", "severe storm"):
        assert forbidden not in joined


def test_northern_theme_does_not_reward_storm_metadata() -> None:
    profile, _ = youtube.build_youtube_query_plan("northern arctic glacier mountains", {}, {})
    arctic = {"title": "Arctic glacier mountains cinematic 4K", "duration": 40, "query_rank": 1}
    storm = {"title": "Massive supercell storm front over farmland 4K", "duration": 40, "query_rank": 1}
    assert youtube._metadata_score(arctic, profile) > youtube._metadata_score(storm, profile)


def test_license_words_are_score_neutral() -> None:
    profile, _ = youtube.build_youtube_query_plan("arctic glacier", {}, {})
    base = {"title": "Arctic glacier cinematic 4K", "duration": 40, "query_rank": 1}
    assert youtube._metadata_score(base, profile) == youtube._metadata_score(
        {**base, "title": base["title"] + " creative commons public domain no copyright royalty free"}, profile
    )


def test_candidate_pool_gate_is_hard_machine_readable() -> None:
    gate = youtube._candidate_pool_gate(5, 2, {"slots": [{}, {}]}, 3)
    assert gate["passed"] is False
    assert gate["required_candidate_count"] == 6
    assert gate["available_candidate_count"] == 5


def test_youtube_shortfall_triggers_pixabay_and_combined_passes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    yt_assets = [_asset(1, "youtube", "mountain")]
    px_assets = [_asset(2, "pixabay", "coast"), _asset(3, "pixabay", "forest")]
    monkeypatch.setattr(first, "run_youtube_pipeline", lambda *a, **k: {
        "status": "insufficient_material", "selected": yt_assets, "selected_count": 1,
        "candidate_count": 1, "candidate_pool_gate": {"passed": False},
        "sufficiency": {"passed": False, "failures": ["short"]}, "search_rounds": [],
    })
    called = {"pixabay": 0}
    def pixabay(*args, **kwargs):
        called["pixabay"] += 1
        return {"status": "ok", "selected": px_assets, "selected_count": 2, "candidate_count": 7, "search_rounds": []}
    monkeypatch.setattr(first, "run_pixabay_pipeline", pixabay)
    result = first.run_youtube_first_pipeline(
        "arctic glacier cinematic", {}, {}, tmp_path / "materials", tmp_path / "cache",
        2, "16:9", timeline_plan={"slots": [{}, {}]}, candidate_pool_multiplier=2,
    )
    assert called["pixabay"] == 1
    assert result["pixabay_fallback"]["status"] == "completed"
    assert result["sufficiency"]["passed"] is True
    assert len({_asset_id(item) for item in result["selected"]}) == 2


def _asset_id(item: dict) -> str:
    return str(item.get("canonical_source_id") or item.get("id"))


def test_sufficient_youtube_skips_pixabay(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assets = [_asset(1, "youtube", "mountain"), _asset(2, "youtube", "coast")]
    monkeypatch.setattr(first, "run_youtube_pipeline", lambda *a, **k: {
        "status": "ok", "selected": assets, "selected_count": 2, "candidate_count": 6,
        "candidate_pool_gate": {"passed": True}, "sufficiency": {"passed": True, "failures": []},
        "search_rounds": [],
    })
    monkeypatch.setattr(first, "run_pixabay_pipeline", lambda *a, **k: pytest.fail("Pixabay must not run"))
    result = first.run_youtube_first_pipeline(
        "arctic glacier cinematic", {}, {}, tmp_path / "materials", tmp_path / "cache",
        2, "16:9", timeline_plan={"slots": [{}]}, candidate_pool_multiplier=2,
    )
    assert result["pixabay_fallback"]["triggered"] is False
    assert result["sufficiency"]["passed"] is True


def test_merge_deduplicates_and_never_repeats_shots() -> None:
    duplicate_a = _asset(1, "youtube", "mountain")
    duplicate_b = {**_asset(9, "pixabay", "mountain"), "canonical_source_id": duplicate_a["canonical_source_id"]}
    selected, report = first.merge_and_rank_assets(
        [duplicate_a, _asset(2, "youtube", "coast")], [duplicate_b, _asset(3, "pixabay", "forest")],
        3, youtube.build_youtube_query_plan("arctic glacier cinematic", {}, {})[0],
    )
    identities = [_asset_id(item) for item in selected]
    assert len(identities) == len(set(identities))
    assert report["duplicate_count"] == 1


def test_global_cache_reuse_skips_network_and_download(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    library_root = tmp_path / "library"
    monkeypatch.setenv("BGM_MONTAGE_LIBRARY_ROOT", str(library_root))
    assets = []
    for index in range(6):
        video = library_root / "youtube" / "videos" / f"youtube_video{index:03d}.mp4"
        video.parent.mkdir(parents=True, exist_ok=True)
        video.write_bytes(b"x" * (65 * 1024))
        assets.append({
            **_asset(index, "youtube", "mountain" if index % 2 == 0 else "coast"),
            "youtube_id": f"video{index:03d}", "local_path": str(video),
            "title": "Arctic glacier cinematic", "channel": f"camera-{index}",
        })
    index_path = library_root / "youtube" / "asset_index.json"
    index_path.write_text(json.dumps({"schema_version": 1, "assets": assets}), encoding="utf-8")
    monkeypatch.setattr(youtube, "_yt_dlp_executable", lambda: pytest.fail("network search must be skipped"))
    monkeypatch.setattr(youtube, "_download_candidate", lambda *a, **k: pytest.fail("download must be skipped"))
    result = youtube.run_youtube_pipeline(
        "arctic glacier cinematic", {}, {}, tmp_path / "project-b-materials", tmp_path / "project-b-cache",
        1, "16:9", timeline_plan={"slots": [{}]}, candidate_pool_multiplier=6,
    )
    assert result["status"] == "ok"
    assert result["download_candidate_count"] == 0
    assert result["selected"][0]["reuse_mode"] == "global_index"
    assert Path(result["selected"][0]["local_path"]).is_file()
