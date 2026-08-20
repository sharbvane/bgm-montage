from __future__ import annotations

import json
import sys
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from golden_fixture_identity import build_fixture_identity, compare_fixture_identity  # noqa: E402


def test_frozen_fixture_identity_is_content_based_and_detects_slot_drift(tmp_path: Path) -> None:
    bgm = tmp_path / "track.mp3"
    bgm.write_bytes(b"bgm")
    timeline = {
        "schema_version": "1.2",
        "planner": {"name": "planner", "version": "1.4.0"},
        "audiomap_digest": "audio-digest",
        "plan_digest": "plan-digest",
        "duration_seconds": 2.0,
        "rhythm_mode": "beat_cut",
        "slots": [{"index": 0, "start": 0.0, "end": 2.0}],
    }
    audiomap = {
        "schema_version": "1.2",
        "analysis_digest": "audio-digest",
        "duration_seconds": 2.0,
        "rhythm_mode": {"mode": "beat_cut"},
    }
    grammar = {"schema_version": "1.0", "status": "ok"}
    style = {"schema_version": "1.0", "profile_digest": "style-digest"}
    paths = {}
    for name, value in (("timeline", timeline), ("audiomap", audiomap), ("grammar", grammar), ("style", style)):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        paths[name] = path

    expected = build_fixture_identity(
        golden_id="G-TEST",
        bgm_path=bgm,
        timeline_path=paths["timeline"],
        audiomap_path=paths["audiomap"],
        editing_grammar_path=paths["grammar"],
        style_profile_path=paths["style"],
        aspect_ratio="16:9",
        requested_duration_seconds=2.0,
    )
    assert compare_fixture_identity(expected, expected) == []

    timeline["slots"].append({"index": 1, "start": 2.0, "end": 2.0})
    paths["timeline"].write_text(json.dumps(timeline), encoding="utf-8")
    actual = build_fixture_identity(
        golden_id="G-TEST",
        bgm_path=bgm,
        timeline_path=paths["timeline"],
        audiomap_path=paths["audiomap"],
        editing_grammar_path=paths["grammar"],
        style_profile_path=paths["style"],
        aspect_ratio="16:9",
        requested_duration_seconds=2.0,
    )
    fields = {item["field"] for item in compare_fixture_identity(expected, actual)}
    assert {"timeline.raw_sha256", "timeline.slot_count", "timeline.slots_sha256"} <= fields

