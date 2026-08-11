from __future__ import annotations

import sys
from pathlib import Path


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

