from __future__ import annotations

import sys
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from pixabay_pipeline import _diversified_order
from visual_semantics import infer_scene_category


def _candidate(asset_id: int, score: float, scene: str) -> dict:
    return {
        "pixabay_id": asset_id,
        "pre_score": score,
        "scene_category": scene,
        "shot_type": "wide",
        "user": f"author-{asset_id}",
        "matched_queries": [f"query-{asset_id}"],
        "thumbnail_signals": {},
    }


def test_diversified_order_frontloads_distinct_scene_categories() -> None:
    candidates = [
        _candidate(1, 0.90, "nature"),
        _candidate(2, 0.88, "nature"),
        _candidate(3, 0.86, "nature"),
        _candidate(4, 0.84, "nature"),
        _candidate(5, 0.70, "architecture"),
        _candidate(6, 0.69, "transport"),
        _candidate(7, 0.68, "water_coast"),
        _candidate(8, 0.75, "food"),
        _candidate(9, 0.74, "people"),
    ]

    ordered = _diversified_order(candidates)
    first_four_scenes = {item["scene_category"] for item in ordered[:4]}

    assert len(first_four_scenes) >= 3
    assert "nature" in first_four_scenes
    assert first_four_scenes & {"architecture", "transport", "water_coast"}
    assert "food" not in first_four_scenes
    assert "people" not in first_four_scenes


def test_natural_scene_taxonomy_preserves_real_visual_variety() -> None:
    assert infer_scene_category("snow mountain canyon aerial landscape") == "mountain_canyon"
    assert infer_scene_category("deep forest trees woodland nature") == "forest_wilderness"
    assert infer_scene_category("arctic glacier iceberg wilderness") == "polar_ice"
    assert infer_scene_category("starry night sky milky way") == "sky_space"
    assert infer_scene_category("scenic road through mountain forest") == "transport"
    assert infer_scene_category("ocean coast waves beach") == "water_coast"
