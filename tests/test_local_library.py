from __future__ import annotations

import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest


SKILL_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = SKILL_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import local_library as library  # noqa: E402
from pixabay_pipeline import update_usage_intervals  # noqa: E402
from visual_intelligence import FEATURE_METADATA_SCHEMA_VERSION  # noqa: E402


def _fake_quality(path: Path, *_args: object, **_kwargs: object) -> tuple[dict, dict]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    scene = f"scene-{path.stem[-1]}"
    quality = {
        "passed": True,
        "overall_score": 0.82,
        "sharpness_score": 0.8,
        "exposure_score": 0.8,
        "stability_score": 0.8,
        "motion_score": 0.55,
        "motion_direction": "right",
        "scene_category": scene,
        "face_content_risk": 0.0,
        "usable_segments": [{"start": 0.0, "end": 8.0, "duration": 8.0, "score": 0.8}],
        "visual_analysis": {
            "aesthetic_score": 0.8,
            "cinematic_score": 0.8,
            "spatial_depth_score": 0.8,
            "composition_quality_score": 0.8,
            "visual_impact_score": 0.8,
            "lighting_quality_score": 0.8,
            "intrinsic_color_quality_score": 0.8,
            "ordinary_travelogue_risk": 0.1,
        },
        "mean_hsv": {"hue_degrees": 180.0, "saturation": 0.5, "value": 0.5},
        "analysis_cache": {"schema_version": 999, "engine_version": "fixture", "file_sha256": digest},
    }
    media = {
        "width": 1920,
        "height": 1080,
        "duration_seconds": 8.0,
        "fps": 30.0,
        "codec": "h264",
        "fingerprint": {"sha256": digest, "duration_seconds": 8.0, "width": 1920, "height": 1080, "perceptual_hashes": []},
    }
    return quality, media


def _fake_probe(_path: Path) -> dict:
    return {
        "streams": [{"codec_type": "video", "codec_name": "h264", "width": 1920, "height": 1080, "duration": "8.0"}],
        "format": {"duration": "8.0"},
    }


def _fake_light(path: Path, _tags: str) -> dict:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "schema_version": library.LIGHTWEIGHT_SCHEMA_VERSION,
        "engine": "fixture",
        "sample_target": 6,
        "sampled_frame_count": 6,
        "fps": 30.0,
        "decoded_duration_seconds": 8.0,
        "scene_category": f"scene-{digest[0]}",
        "subject_class": "environment",
        "shot_scale": "wide",
        "motion_score": 0.55,
        "motion_label": "gentle",
        "motion_direction": "right",
        "mean_hsv": {"hue_degrees": int(digest[:2], 16) / 255 * 360, "saturation": 0.5, "value": 0.6},
        "perceptual_hashes": [digest[:16]],
        "quality": {
            "overall_score": 0.72,
            "sharpness_score": 0.75,
            "exposure_score": 0.76,
            "stability_score": 0.8,
            "text_watermark_risk": 0.0,
            "face_content_risk": 0.0,
            "motion_score": 0.55,
            "motion_direction": "right",
            "scene_category": f"scene-{digest[0]}",
            "mean_hsv": {"hue_degrees": 180.0, "saturation": 0.5, "value": 0.6},
            "visual_analysis": {
                "aesthetic_score": 0.7, "cinematic_score": 0.7,
                "spatial_depth_score": 0.7, "composition_quality_score": 0.7,
                "visual_impact_score": 0.7, "lighting_quality_score": 0.7,
                "intrinsic_color_quality_score": 0.7, "ordinary_travelogue_risk": 0.2,
            },
        },
    }


def test_incremental_sync_only_deep_analyzes_added_and_changed_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "library"
    cache = tmp_path / "cache"
    root.mkdir()
    for index in range(3):
        (root / f"clip-{index}.mp4").write_bytes(f"video-{index}".encode())

    calls: list[str] = []

    def counted(path: Path, *args: object, **kwargs: object) -> tuple[dict, dict]:
        calls.append(path.name)
        return _fake_quality(path, *args, **kwargs)

    monkeypatch.setattr(library, "_video_quality", counted)
    monkeypatch.setattr(library, "_ffprobe", _fake_probe)
    monkeypatch.setattr(library, "_lightweight_visual", _fake_light)
    monkeypatch.setattr(library, "analysis_cache_valid", lambda *_args: True)

    first = library.sync_local_library(root, cache)
    assert first["sync"]["scanned_files"] == 3
    assert first["sync"]["added"] == 3
    assert first["sync"]["light_profiled_entries"] == 3
    assert first["sync"]["light_analyzed"] == 3
    assert first["sync"]["light_cache_hits"] == 0
    assert all(entry["lightweight_visual"]["sampled_frame_count"] == 6 for entry in first["entries"])
    assert calls == []
    calls.clear()
    second = library.sync_local_library(root, cache)
    assert second["sync"]["deep_analyzed"] == 0
    assert second["sync"]["reused"] == 3
    assert second["sync"]["light_analyzed"] == 0
    assert second["sync"]["light_cache_hits"] == 3
    assert calls == []

    (root / "clip-1.mp4").write_bytes(b"modified-video-1")
    (root / "clip-2.mp4").unlink()
    (root / "clip-3.mp4").write_bytes(b"new-video-3")
    calls.clear()
    third = library.sync_local_library(root, cache)
    assert third["sync"]["added"] == 1
    assert third["sync"]["changed"] == 1
    assert third["sync"]["deleted"] == 1
    assert third["sync"]["reused"] == 1
    assert third["sync"]["light_analyzed"] == 2
    assert third["sync"]["deep_analyzed"] == 0
    assert calls == []
    assert {entry["relative_path"] for entry in third["entries"]} == {
        "clip-0.mp4", "clip-1.mp4", "clip-3.mp4"
    }


def test_two_stage_selection_uses_index_without_redecoding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "冰岛"
    cache = tmp_path / "cache"
    root.mkdir()
    for index in range(20):
        (root / f"iceland-{index}.mp4").write_bytes(f"video-{index}".encode())
    monkeypatch.setattr(library, "_video_quality", _fake_quality)
    monkeypatch.setattr(library, "_ffprobe", _fake_probe)
    monkeypatch.setattr(library, "_lightweight_visual", _fake_light)
    monkeypatch.setattr(library, "analysis_cache_valid", lambda *_args: True)
    monkeypatch.setattr(
        library.requests.Session,
        "get",
        lambda *_args, **_kwargs: pytest.fail("local-library selection must not access the network"),
    )

    first = library.run_local_library_pipeline(
        "iceland cinematic landscape", {}, {}, root, cache, 4, "16:9",
        target_duration=8.0, timeline_plan={"slots": [{"index": index} for index in range(4)]},
    )
    assert first["sync"]["deep_analyzed"] == 16
    assert first["selection"]["coarse_candidate_count"] == 20
    assert first["selection"]["fine_candidate_count"] == 16
    assert first["selection"]["deep_analysis_during_selection"] == 16
    assert len(first["selected"]) == 4
    used = first["selected"][0]
    update_usage_intervals(
        first["sources_manifest"],
        {
            "run_id": "local-run-1",
            "shots": [{
                "asset_id": used["asset_id"],
                "local_path": used["local_path"],
                "output_start": 0.0,
                "output_end": 2.0,
                "source_start": 1.0,
                "source_end": 3.0,
            }],
        },
    )
    after_usage = json.loads(Path(first["library_index"]).read_text(encoding="utf-8"))
    assert after_usage["selection_signature"]
    assert after_usage["library_root"] == str(root.resolve())

    second = library.run_local_library_pipeline(
        "iceland cinematic landscape", {}, {}, root, cache, 4, "16:9",
        target_duration=8.0, timeline_plan={"slots": [{"index": index} for index in range(4)]},
    )
    assert second["sync"]["deep_analyzed"] == 0
    assert second["sync"]["reused"] == 20
    index = json.loads(Path(second["library_index"]).read_text(encoding="utf-8"))
    assert len(index["entries"]) == 20
    persisted = next(entry for entry in index["entries"] if entry["local_path"] == used["local_path"])
    assert persisted["historical_usage_count"] == 1
    assert {item["run_id"] for item in persisted["usage_history"]} == {"local-run-1"}

    (root / "iceland-20.mp4").write_bytes(b"new-video-20")
    third = library.run_local_library_pipeline(
        "iceland cinematic landscape", {}, {}, root, cache, 4, "16:9",
        target_duration=8.0, timeline_plan={"slots": [{"index": index} for index in range(4)]},
    )
    assert third["sync"]["added"] == 1
    assert third["sync"]["reused"] == 20
    assert third["sync"]["light_analyzed"] == 1
    assert third["sync"]["light_profiled_entries"] == 21
    assert 0 <= third["sync"]["deep_analyzed"] <= 16


def test_content_identity_resets_replacement_history_and_survives_move(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, cache = tmp_path / "library", tmp_path / "cache"
    root.mkdir()
    path = root / "A.mp4"
    path.write_bytes(b"old-content")
    monkeypatch.setattr(library, "_ffprobe", _fake_probe)
    monkeypatch.setattr(library, "_lightweight_visual", _fake_light)

    first = library.sync_local_library(root, cache)
    old = first["entries"][0]
    manifest = tmp_path / "sources.json"
    manifest.write_text(json.dumps({
        "material_libraries": {"local": first["index_path"]},
        "sources": [old],
    }), encoding="utf-8")
    shot = {
        "run_id": "old-run",
        "shots": [{"asset_id": old["asset_id"], "local_path": old["local_path"], "output_start": 0, "output_end": 1, "source_start": 0, "source_end": 1}],
    }
    update_usage_intervals(manifest, shot)
    used = library.sync_local_library(root, cache)["entries"][0]
    assert used["historical_usage_count"] == 1

    moved_path = root / "moved.mp4"
    path.rename(moved_path)
    moved = library.sync_local_library(root, cache)
    assert moved["sync"]["moved"] == 1
    assert moved["sync"]["light_analyzed"] == 0
    assert moved["entries"][0]["historical_usage_count"] == 1
    assert moved["entries"][0]["canonical_source_id"] == old["canonical_source_id"]

    moved_path.write_bytes(b"completely-new-content")
    replaced = library.sync_local_library(root, cache)
    current = replaced["entries"][0]
    assert replaced["sync"]["changed"] == 1
    assert current["historical_usage_count"] == 0
    assert current["canonical_source_id"] != old["canonical_source_id"]
    update_usage_intervals(manifest, shot)
    after_stale_update = json.loads(Path(first["index_path"]).read_text(encoding="utf-8"))["entries"][0]
    assert after_stale_update["historical_usage_count"] == 0

    moved_path.unlink()
    deleted = library.sync_local_library(root, cache)
    assert deleted["sync"]["deleted"] == 1
    assert deleted["entries"] == []


def test_concurrent_sync_and_deep_writes_merge_without_duplicate_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, cache = tmp_path / "library", tmp_path / "cache"
    root.mkdir()
    for index in range(30):
        (root / f"clip-{index:03d}.mp4").write_bytes(f"video-{index}".encode())
    light_calls: list[str] = []
    deep_calls: list[str] = []

    def light(path: Path, tags: str) -> dict:
        light_calls.append(path.name)
        return _fake_light(path, tags)

    def deep(path: Path, *args: object, **kwargs: object) -> tuple[dict, dict]:
        deep_calls.append(path.name)
        return _fake_quality(path, *args, **kwargs)

    monkeypatch.setattr(library, "_ffprobe", _fake_probe)
    monkeypatch.setattr(library, "_lightweight_visual", light)
    monkeypatch.setattr(library, "_video_quality", deep)
    monkeypatch.setattr(library, "analysis_cache_valid", lambda *_args: True)
    monkeypatch.setattr(library.requests.Session, "get", lambda *_args, **_kwargs: pytest.fail("network access"))

    with ThreadPoolExecutor(max_workers=4) as pool:
        syncs = list(pool.map(lambda _: library.sync_local_library(root, cache), range(4)))
    assert len(light_calls) == 30
    assert sorted(result["sync"]["added"] for result in syncs) == [0, 0, 0, 30]

    def run() -> dict:
        return library.run_local_library_pipeline(
            "landscape", {}, {"analysis_digest": "same"}, root, cache, 4, "16:9",
            min_resolution=(1, 1), target_duration=8.0,
            timeline_plan={"slots": [{"index": index} for index in range(4)]},
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: run(), range(2)))
    assert len(deep_calls) == 16
    assert sorted(result["sync"]["deep_analyzed"] for result in results) == [0, 16]
    index = json.loads(Path(results[0]["library_index"]).read_text(encoding="utf-8"))
    assert len(index["entries"]) == 30
    assert sum(entry["analysis_status"] == "indexed" for entry in index["entries"]) == 16


def test_deep_analysis_preserves_lightweight_shot_scale_with_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "library"
    root.mkdir()
    path = root / "wide-coast.mp4"
    path.write_bytes(b"wide-video")
    monkeypatch.setattr(library, "_video_quality", _fake_quality)
    monkeypatch.setattr(library, "_ffprobe", _fake_probe)
    previous = {
        "content_fingerprint": library._content_fingerprint(path),
        "lightweight_visual": _fake_light(path, "wide coast"),
    }

    analyzed = library._analyze_entry(root, path, previous)

    assert analyzed["shot_scale"] == "wide"
    assert analyzed["shot_scale_detail"]["available"] is True
    assert analyzed["shot_scale_detail"]["source"] == "lightweight_visual"
    assert analyzed["shot_scale_detail"]["schema_version"] == FEATURE_METADATA_SCHEMA_VERSION
    assert analyzed["quality"]["visual_features"]["shot_scale"] == "wide"


def test_filename_semantics_refresh_after_rename_without_visual_rescan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, cache = tmp_path / "library", tmp_path / "cache"
    root.mkdir()
    path = root / "雾中雪山.mp4"
    path.write_bytes(b"same-video")
    light_calls: list[str] = []

    def light(path: Path, tags: str) -> dict:
        light_calls.append(path.name)
        return _fake_light(path, tags)

    monkeypatch.setattr(library, "_ffprobe", _fake_probe)
    monkeypatch.setattr(library, "_lightweight_visual", light)
    first = library.sync_local_library(root, cache)
    assert {"fog", "mountain"}.issubset(first["entries"][0]["filename_semantics"]["terms"])

    path.rename(root / "航拍海岸.mp4")
    second = library.sync_local_library(root, cache, metadata_refresh_limit=1)
    entry = second["entries"][0]
    assert second["sync"]["metadata_refreshed"] == 1
    assert light_calls == ["雾中雪山.mp4"]
    assert {"aerial", "coast"}.issubset(entry["filename_semantics"]["terms"])
    assert "fog" not in entry["filename_semantics"]["terms"]
    assert entry["shot_scale_detail"]["source"] == "lightweight_visual"


def test_legacy_metadata_migrates_lazily_without_lightweight_rescan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, cache = tmp_path / "library", tmp_path / "cache"
    root.mkdir()
    path = root / "航拍海岸.mp4"
    path.write_bytes(b"legacy-video")
    light_calls: list[str] = []

    def light(path: Path, tags: str) -> dict:
        light_calls.append(path.name)
        return _fake_light(path, tags)

    monkeypatch.setattr(library, "_ffprobe", _fake_probe)
    monkeypatch.setattr(library, "_lightweight_visual", light)
    first = library.sync_local_library(root, cache)
    index_path = Path(first["index_path"])
    legacy = json.loads(index_path.read_text(encoding="utf-8"))
    legacy_entry = legacy["entries"][0]
    legacy_entry.pop("metadata_features", None)
    legacy_entry.pop("filename_semantics", None)
    legacy_entry.pop("metadata_schema_version", None)
    index_path.write_text(json.dumps(legacy, ensure_ascii=False), encoding="utf-8")
    light_calls.clear()

    migrated = library.sync_local_library(root, cache, metadata_refresh_limit=1)
    entry = migrated["entries"][0]
    assert migrated["sync"]["metadata_refreshed"] == 1
    assert migrated["sync"]["light_analyzed"] == 0
    assert light_calls == []
    assert entry["metadata_schema_version"] == FEATURE_METADATA_SCHEMA_VERSION
    assert entry["metadata_features"]["shot_scale"]["available"] is True
