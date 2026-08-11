from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path


SKILL_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = SKILL_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import pixabay_pipeline as pipeline  # noqa: E402
from runtime_paths import migrate_legacy_nested_pixabay_cache, pixabay_cache_root  # noqa: E402


def _candidate() -> dict[str, object]:
    return {
        "id": 4242,
        "pixabay_id": 4242,
        "page_url": "https://pixabay.com/videos/id-4242/",
        "tags": "forest aerial landscape",
        "duration": 8.0,
        "user": "fixture-author",
        "variant": {
            "url": "https://cdn.example.invalid/4242.mp4",
            "width": 1920,
            "height": 1080,
            "name": "large",
        },
        "matched_queries": ["forest aerial"],
        "search_rounds": [1],
        "raw": {},
        "pre_score": 0.9,
        "diversity_adjusted_score": 0.9,
        "score_components": {},
        "thumbnail_signals": {},
        "shot_type": "aerial",
        "shot_scale": "extreme_wide",
        "motion_score_estimate": 0.4,
    }


def _library_entry(origin: Path) -> dict[str, object]:
    file_hash = hashlib.sha256(origin.read_bytes()).hexdigest()
    fingerprint = {
        "sha256": file_hash,
        "perceptual_hashes": [],
        "duration_seconds": 8.0,
        "width": 1920,
        "height": 1080,
        "size_bytes": origin.stat().st_size,
    }
    return {
        "pixabay_id": 4242,
        "author": "fixture-author",
        "page_url": "https://pixabay.com/videos/id-4242/",
        "local_path": str(origin),
        "added_at": "2026-01-01T00:00:00+00:00",
        "fingerprint": fingerprint,
        "file_hash": file_hash,
        "quality": {
            "passed": True,
            "motion_score": 0.4,
            "overall_score": 0.9,
            "analysis_cache": {
                "schema_version": 2,
                "engine_version": "1.3.0",
                "file_sha256": file_hash,
            },
        },
        "media": {
            "duration_seconds": 8.0,
            "width": 1920,
            "height": 1080,
            "fps": 30.0,
            "fingerprint": fingerprint,
        },
    }


def test_pixabay_namespace_is_single_and_legacy_nested_cache_is_migrated(tmp_path: Path) -> None:
    project_cache = tmp_path / ".bgm-montage-cache"
    stage_cache = pixabay_cache_root(project_cache)
    legacy = stage_cache / "pixabay" / "search" / "legacy.json"
    legacy.parent.mkdir(parents=True)
    legacy.write_text('{"cached": true}\n', encoding="utf-8")

    copied = migrate_legacy_nested_pixabay_cache(stage_cache)
    canonical = pipeline._search_cache_path(stage_cache, "forest", 1, 20)

    assert copied["search"] == 1
    assert (stage_cache / "search" / "legacy.json").is_file()
    assert canonical.parent == stage_cache / "search"
    assert "pixabay/pixabay" not in canonical.as_posix().casefold()


def test_known_pixabay_id_is_reused_across_themes_and_projects_without_download(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("PIXABAY_API_KEY", "offline-mock-value")
    monkeypatch.setenv("BGM_MONTAGE_LIBRARY_ROOT", str(tmp_path / "global-library"))
    origin_dir = tmp_path / "materials-a" / "theme-a"
    origin_dir.mkdir(parents=True)
    origin = origin_dir / "01_forest_4242.mp4"
    origin.write_bytes(b"offline-video-fixture")
    entry = _library_entry(origin)

    project_one_cache = tmp_path / "project-one" / ".bgm-montage-cache"
    project_one_cache.mkdir(parents=True)
    (project_one_cache / "video_fingerprints.json").write_text(
        json.dumps({"schema_version": 1, "entries": [entry]}),
        encoding="utf-8",
    )

    download_calls: list[str] = []

    def fail_if_downloaded(*args, **kwargs) -> None:
        download_calls.append("called")
        raise AssertionError("a known Pixabay ID must not be downloaded again")

    monkeypatch.setattr(
        pipeline,
        "_collect_candidates",
        lambda *args, **kwargs: ([copy.deepcopy(_candidate())], [{"round": 1}], []),
    )
    monkeypatch.setattr(
        pipeline,
        "_score_candidates",
        lambda session, candidates, *args, **kwargs: candidates,
    )
    monkeypatch.setattr(pipeline, "_download_video", fail_if_downloaded)

    first = pipeline.run_pixabay_pipeline(
        "theme-b",
        {},
        {},
        tmp_path / "materials-b",
        project_one_cache / "pixabay",
        1,
        "16:9",
    )
    first_path = Path(first["selected"][0]["local_path"])
    assert first_path.samefile(origin)
    assert first["selected"][0]["reuse_mode"] in {
        "hardlink",
        "hardlink_existing",
        "shared_reference",
    }
    assert first["selected"][0]["reuse_origin"]["pixabay_id"] == 4242
    assert not first["rejections"]
    assert (project_one_cache / "pixabay" / "video_fingerprints.json").is_file()
    assert (tmp_path / "global-library" / "material_index.json").is_file()

    project_two_stage = tmp_path / "project-two" / ".bgm-montage-cache" / "pixabay"
    second = pipeline.run_pixabay_pipeline(
        "theme-c",
        {},
        {},
        tmp_path / "materials-c",
        project_two_stage,
        1,
        "16:9",
    )
    second_path = Path(second["selected"][0]["local_path"])
    assert second_path.samefile(origin)
    assert second["selected"][0]["reuse_mode"] in {
        "hardlink",
        "hardlink_existing",
        "shared_reference",
    }
    assert not second["rejections"]
    assert download_calls == []
