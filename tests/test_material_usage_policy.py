from __future__ import annotations

import sys
from pathlib import Path


SKILL_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = SKILL_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import bgm_montage  # noqa: E402
from material_usage_policy import apply_usage_policy, material_usage_policy  # noqa: E402
from youtube_pipeline import _metadata_score  # noqa: E402


def test_local_evaluation_is_default_and_has_zero_rights_weight() -> None:
    args = bgm_montage.build_parser().parse_args(
        [
            "--bgm", "track.mp3",
            "--theme", "storm landscape",
            "--duration", "20",
            "--ratio", "16:9",
            "--output-dir", "renders",
        ]
    )
    assert args.usage_mode == "local_evaluation"
    policy = material_usage_policy(args.usage_mode)
    assert policy["quality_over_source"] is True
    assert policy["license_metadata_used_for_ranking"] is False
    assert policy["license_or_copyright_ranking_weight"] == 0.0
    assert policy["ordinary_youtube_source_penalty"] == 0.0
    assert policy["recurring_rights_warning"] is False


def test_local_manifest_removes_legacy_rights_warning() -> None:
    manifest = apply_usage_policy(
        {"attribution_notice": "legacy recurring warning", "sources": []},
        "local_evaluation",
    )
    assert manifest["usage_mode"] == "local_evaluation"
    assert "attribution_notice" not in manifest
    assert "publication_mode_notice" not in manifest


def test_publish_mode_requires_explicit_task_policy() -> None:
    manifest = apply_usage_policy({}, "publish")
    assert manifest["material_usage_policy"]["task_specific_rights_policy_required"] is True
    assert "publication_mode_notice" in manifest


def test_youtube_metadata_score_ignores_license_status_words() -> None:
    base = {
        "title": "Massive shelf cloud over rural highway 4K",
        "channel": "Storm Camera",
        "duration": 45,
        "query_rank": 1,
        "view_count": 10000,
    }
    licensing_words = " creative commons public domain no copyright royalty free"
    assert _metadata_score(base) == _metadata_score({**base, "title": base["title"] + licensing_words})
