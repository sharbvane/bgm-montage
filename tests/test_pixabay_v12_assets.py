from __future__ import annotations

import copy
import json
import multiprocessing as mp
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from typing import Any

import numpy as np
import pytest


SKILL_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = SKILL_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import pixabay_pipeline as pipeline  # noqa: E402
from runtime_paths import RuntimePaths, normalize_cache_roots  # noqa: E402


def _spawn_asset_lock_worker(
    global_index: str,
    start_event: Any,
    event_queue: Any,
    hold_seconds: float,
) -> None:
    """Spawn-safe worker used to verify the Windows O_EXCL lock."""

    start_event.wait(10.0)
    with pipeline._exclusive_asset_lock(
        Path(global_index),
        7001,
        timeout_seconds=10.0,
        stale_seconds=0.2,
        heartbeat_seconds=0.05,
    ):
        event_queue.put(("enter", os.getpid(), time.time()))
        time.sleep(hold_seconds)
        event_queue.put(("exit", os.getpid(), time.time()))


def _hit(asset_id: int, tags: str = "nature forest landscape") -> dict[str, Any]:
    return {
        "id": asset_id,
        "pageURL": f"https://pixabay.com/videos/{asset_id}/",
        "tags": tags,
        "duration": 9.0,
        "user": f"author-{asset_id}",
        "views": 100,
        "downloads": 50,
        "likes": 10,
        "videos": {
            "large": {
                "url": f"https://cdn.example.invalid/{asset_id}.mp4",
                "width": 1920,
                "height": 1080,
                "size": 100_000,
            }
        },
    }


def _candidate(asset_id: int) -> dict[str, Any]:
    raw = _hit(asset_id)
    return {
        "id": asset_id,
        "pixabay_id": asset_id,
        "page_url": raw["pageURL"],
        "tags": raw["tags"],
        "duration": 9.0,
        "user": raw["user"],
        "variant": {
            "url": raw["videos"]["large"]["url"],
            "width": 1920,
            "height": 1080,
            "size": 100_000,
            "name": "large",
        },
        "matched_queries": ["nature forest"],
        "search_rounds": [1],
        "raw": raw,
        "pre_score": 0.8,
        "diversity_adjusted_score": 0.8,
        "score_components": {},
        "thumbnail_signals": {},
        "shot_type": "wide",
        "shot_scale": "wide",
        "motion_score_estimate": 0.4,
        "scene_category": "nature",
        "semantic_tags": ["nature", "forest", "landscape"],
        "canonical_source_id": f"pixabay:{asset_id}",
    }


def _library_entry(path: Path, asset_id: int) -> dict[str, Any]:
    sha = f"{asset_id:064x}"
    fingerprint = {
        "sha256": sha,
        "perceptual_hashes": [f"{asset_id:016x}"[-16:]],
        "duration_seconds": 9.0,
        "width": 1920,
        "height": 1080,
        "size_bytes": path.stat().st_size,
    }
    return {
        "pixabay_id": asset_id,
        "author": f"author-{asset_id}",
        "page_url": f"https://pixabay.com/videos/{asset_id}/",
        "download_url": f"https://cdn.example.invalid/{asset_id}.mp4",
        "tags": "nature forest landscape",
        "local_path": str(path),
        "fingerprint": fingerprint,
        "quality": {
            "passed": True,
            "scene_category": "nature",
            "motion_score": 0.4,
            "face_content_risk": 0.0,
        },
        "media": {
            "duration_seconds": 9.0,
            "width": 1920,
            "height": 1080,
            "fps": 30.0,
            "fingerprint": fingerprint,
        },
    }


def test_cache_root_and_stage_inputs_collapse_to_one_namespace(tmp_path: Path) -> None:
    project_cache = tmp_path / "project" / ".bgm-montage-cache"
    stage = project_cache / "pixabay"
    duplicated = stage / "pixabay"

    assert normalize_cache_roots(project_cache) == (project_cache.resolve(), stage.resolve())
    assert normalize_cache_roots(stage) == (project_cache.resolve(), stage.resolve())
    assert normalize_cache_roots(duplicated) == (project_cache.resolve(), stage.resolve())
    assert RuntimePaths.build(cache_root=duplicated).pixabay_cache == stage.resolve()


def test_collect_candidates_pages_until_timeline_pool_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pages: list[int] = []

    def fake_search(
        _session: Any,
        _key: str,
        _query: str,
        _cache: Path,
        _per_page: int,
        page: int = 1,
    ) -> tuple[dict[str, Any], bool]:
        pages.append(page)
        start = (page - 1) * 2 + 1
        hits = [_hit(start), _hit(start + 1)]
        return {"total": 20, "totalHits": 20, "hits": hits}, False

    monkeypatch.setattr(pipeline, "_pixabay_search", fake_search)
    candidates, rounds, errors = pipeline._collect_candidates(
        object(),
        "offline-key",
        "nature",
        {},
        {},
        tmp_path / "cache" / "pixabay",
        desired_count=2,
        min_resolution=(1280, 720),
        timeline_plan={"slots": [{"index": 0, "recommended_content": "forest landscape"}]},
        candidate_pool_multiplier=4,
        max_search_pages=3,
    )

    assert not errors
    assert len(candidates) == 4
    assert pages == [1, 2]
    assert rounds[-1]["target_pool"] == 4
    assert rounds[-1]["stop_reason"] == "metadata candidate pool target reached"


def test_local_material_index_can_satisfy_pool_without_api(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entries = []
    for asset_id in range(1, 7):
        path = tmp_path / f"cached-{asset_id}.mp4"
        path.write_bytes(b"cached fixture")
        entries.append(_library_entry(path, asset_id))

    def fail_search(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("local library already satisfies the metadata pool")

    monkeypatch.setattr(pipeline, "_pixabay_search", fail_search)
    candidates, rounds, errors = pipeline._collect_candidates(
        object(),
        "offline-key",
        "nature forest",
        {},
        {},
        tmp_path / "cache" / "pixabay",
        desired_count=2,
        min_resolution=(1280, 720),
        local_entries=entries,
        timeline_plan={"slots": [{"index": 0, "recommended_content": "forest landscape"}]},
        candidate_pool_multiplier=6,
        max_search_pages=3,
    )

    assert not errors
    assert len(candidates) == 6
    assert len(rounds) == 1
    assert rounds[0]["round"] == 0
    assert rounds[0]["stop_reason"] == "local material index satisfied metadata pool"


def test_timeline_candidate_pool_shortage_fails_before_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PIXABAY_API_KEY", "offline-key")
    monkeypatch.setenv("BGM_MONTAGE_LIBRARY_ROOT", str(tmp_path / "library"))
    candidates = [_candidate(index) for index in range(1, 5)]
    monkeypatch.setattr(
        pipeline,
        "_collect_candidates",
        lambda *_args, **_kwargs: (copy.deepcopy(candidates), [{"round": 1}], []),
    )
    monkeypatch.setattr(
        pipeline,
        "_score_candidates",
        lambda _session, values, *_args, **_kwargs: values,
    )
    monkeypatch.setattr(
        pipeline,
        "_download_video",
        lambda *_args, **_kwargs: pytest.fail("pool gate must run before full download"),
    )
    theme = "nature pool shortage"
    with pytest.raises(pipeline.InsufficientMaterialError, match="metadata candidates 4 < 6"):
        pipeline.run_pixabay_pipeline(
            theme,
            {},
            {},
            tmp_path / "materials",
            tmp_path / "cache",
            desired_count=2,
            aspect_ratio="16:9",
            timeline_plan={
                "slots": [
                    {"index": 0, "important": True, "recommended_content": "forest landscape"},
                    {"index": 1, "recommended_content": "nature detail"},
                ]
            },
            candidate_pool_multiplier=3,
        )

    manifest_path = pipeline.material_theme_directory(tmp_path / "materials", theme) / "sources.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "insufficient_material"
    assert manifest["candidate_pool_gate"]["passed"] is False
    assert manifest["candidate_pool_gate"]["required_candidate_count"] == 6
    assert manifest["selected_count"] == 0


def test_canonical_duplicates_do_not_count_as_independent_assets() -> None:
    selected = [
        {
            "pixabay_id": 1,
            "canonical_source_id": "sha256:same",
            "duration_seconds": 8.0,
            "scene_category": "nature",
            "face_content_risk": 0.0,
        },
        {
            "pixabay_id": 2,
            "canonical_source_id": "sha256:same",
            "duration_seconds": 8.0,
            "scene_category": "nature",
            "face_content_risk": 0.0,
        },
    ]
    report = pipeline.evaluate_selected_sufficiency(selected, desired_count=2, target_duration=None)
    assert report["independent_asset_count"] == 1
    assert report["passed"] is False


def test_sampled_usable_segments_avoid_black_frozen_head_and_tail() -> None:
    frames: list[np.ndarray] = []
    signals: list[dict[str, float]] = []
    for index in range(16):
        frame = np.zeros((90, 160, 3), dtype=np.uint8)
        if 2 <= index <= 13:
            frame[:] = 70
            x = 10 + index * 5
            frame[25:65, x : x + 24] = 220
            signals.append({"mean_luma": 90.0, "sharpness_score": 0.8, "exposure_score": 0.8})
        else:
            signals.append({"mean_luma": 0.0, "sharpness_score": 0.0, "exposure_score": 0.0})
        frames.append(frame)

    segments, summary = pipeline._usable_segments_from_samples(frames, signals, duration=8.0)
    assert segments
    best = segments[0]
    assert 0.0 < best["start"] < best["end"] < 8.0
    assert best["duration"] >= 0.75
    assert best["black_frame_ratio"] == 0.0
    assert best["motion_direction"] in {"left", "right", "up", "down", "mixed", "static"}
    assert summary["black_frame_ratio"] > 0.0
    assert summary["freeze_frame_ratio"] > 0.0


def test_usage_history_is_idempotent_and_synced_to_material_indexes(tmp_path: Path) -> None:
    media = tmp_path / "素材 视频.mp4"
    media.write_bytes(b"fixture")
    entry = _library_entry(media, 77)
    project_index = tmp_path / "cache" / "pixabay" / "material_index.json"
    global_index = tmp_path / "global" / "material_index.json"
    fingerprint_index = tmp_path / "cache" / "pixabay" / "video_fingerprints.json"
    for target in (project_index, global_index, fingerprint_index):
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({"schema_version": 3, "entries": [entry]}), encoding="utf-8")

    source = pipeline._complete_asset_record(
        {
            **entry,
            "id": 77,
            "duration_seconds": 9.0,
            "duration": 9.0,
            "width": 1920,
            "height": 1080,
            "fps": 30.0,
            "usage_intervals": [],
            "actual_usage_intervals": [],
        },
        download_status="cached",
    )
    manifest_path = tmp_path / "sources.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 3,
                "manifest_type": "asset_manifest",
                "sources": [source],
                "assets": [source],
                "material_libraries": {
                    "project": str(project_index),
                    "global": str(global_index),
                },
                "cache_layout": {"fingerprints": str(fingerprint_index)},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    plan = {
        "run_id": "fixture-run",
        "shots": [
            {
                "pixabay_id": 77,
                "local_path": str(media),
                "output_start": 0.0,
                "output_end": 2.0,
                "source_start": 1.0,
                "source_end": 3.0,
            }
        ],
    }

    first = pipeline.update_usage_intervals(manifest_path, plan)
    second = pipeline.update_usage_intervals(manifest_path, plan)
    assert first["event_id"] == second["event_id"]
    updated_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    updated_source = updated_manifest["sources"][0]
    assert updated_source["historical_usage_count"] == 1
    assert len(updated_source["usage_history"]) == 1
    assert updated_manifest["assets"] == updated_manifest["sources"]
    for target in (project_index, global_index, fingerprint_index):
        library = json.loads(target.read_text(encoding="utf-8"))
        assert library["entries"][0]["historical_usage_count"] == 1
        assert len(library["entries"][0]["usage_history"]) == 1


def test_manufacturing_scene_is_industrial_not_people() -> None:
    assert pipeline._scene_category("manufacturing") == "industrial"
    assert pipeline._scene_category("advanced manufacturing process") == "industrial"


def test_concurrent_material_catalog_updates_preserve_both_projects(tmp_path: Path) -> None:
    """Two projects writing the shared catalog must merge instead of clobbering."""

    catalog = tmp_path / "共享 素材库" / "material_index.json"
    media_a = tmp_path / "项目 A.mp4"
    media_b = tmp_path / "项目 B.mp4"
    media_a.write_bytes(b"project-a")
    media_b.write_bytes(b"project-b")
    entries = [_library_entry(media_a, 101), _library_entry(media_b, 202)]
    start = Barrier(2)

    def persist(entry: dict[str, Any]) -> None:
        start.wait(timeout=5.0)
        pipeline._persist_material_libraries([catalog], [entry], "fixture-secret")

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(persist, entry) for entry in entries]
        for future in futures:
            future.result(timeout=10.0)

    payload = json.loads(catalog.read_text(encoding="utf-8"))
    assert {int(entry["pixabay_id"]) for entry in payload["entries"]} == {101, 202}
    assert not catalog.with_name(f".{catalog.name}.lock").exists()
    assert not list(catalog.parent.glob(f".{catalog.name}.*.tmp"))


def test_same_pixabay_id_concurrent_projects_download_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both projects start stale; the loser must re-read and reuse under ID lock."""

    monkeypatch.setenv("PIXABAY_API_KEY", "offline-key")
    monkeypatch.setenv("BGM_MONTAGE_LIBRARY_ROOT", str(tmp_path / "共享素材索引"))
    start = Barrier(2)
    download_count = 0
    count_lock = threading.Lock()

    def collect(*_args: Any, **_kwargs: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[Any]]:
        start.wait(timeout=10.0)
        return [copy.deepcopy(_candidate(8801))], [{"round": 1}], []

    def download(_session: Any, _url: str, destination: Path, _secret: str) -> None:
        nonlocal download_count
        with count_lock:
            download_count += 1
        time.sleep(0.20)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"concurrent-video" * 5000)

    fingerprint = {
        "sha256": "8" * 64,
        "perceptual_hashes": ["8" * 16],
        "duration_seconds": 9.0,
        "width": 1920,
        "height": 1080,
        "size_bytes": 80_000,
    }

    def quality(*_args: Any, **_kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        usable = [{"start": 0.5, "end": 8.5, "duration": 8.0, "score": 0.9}]
        return (
            {
                "passed": True,
                "rejection_reasons": [],
                "motion_score": 0.4,
                "scene_category": "nature",
                "face_content_risk": 0.0,
                "subject_profile": {},
                "usable_segments": usable,
            },
            {
                "duration_seconds": 9.0,
                "width": 1920,
                "height": 1080,
                "fps": 30.0,
                "ratio": "16:9",
                "fingerprint": fingerprint,
                "usable_segments": usable,
            },
        )

    monkeypatch.setattr(pipeline, "_collect_candidates", collect)
    monkeypatch.setattr(
        pipeline,
        "_score_candidates",
        lambda _session, values, *_args, **_kwargs: values,
    )
    monkeypatch.setattr(pipeline, "_download_video", download)
    monkeypatch.setattr(pipeline, "_video_quality", quality)

    def run(label: str) -> dict[str, Any]:
        return pipeline.run_pixabay_pipeline(
            theme=f"nature project {label}",
            style_profile={},
            audio_profile={},
            material_root=tmp_path / f"materials-{label}",
            cache_dir=tmp_path / f"project-{label}" / ".bgm-montage-cache" / "pixabay",
            desired_count=1,
            aspect_ratio="16:9",
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = [future.result(timeout=20.0) for future in (pool.submit(run, "A"), pool.submit(run, "B"))]

    assert download_count == 1
    assert all(result["status"] == "ok" for result in results)
    assert all(result["selected_count"] == 1 for result in results)
    assert sorted(source["reuse_mode"] for result in results for source in result["selected"]) == [
        "downloaded",
        "hardlink",
    ]
    global_index = Path(os.environ["BGM_MONTAGE_LIBRARY_ROOT"]) / "material_index.json"
    catalog = json.loads(global_index.read_text(encoding="utf-8"))
    assert [entry["pixabay_id"] for entry in catalog["entries"]] == [8801]


def test_same_theme_concurrent_runs_return_matching_stable_snapshots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PIXABAY_API_KEY", "offline-key")
    monkeypatch.setenv("BGM_MONTAGE_LIBRARY_ROOT", str(tmp_path / "global"))
    start = Barrier(2)

    def collect(
        _session: Any,
        _key: str,
        _theme: str,
        _style: Any,
        _audio: Any,
        cache_root: Path,
        *_args: Any,
        **_kwargs: Any,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[Any]]:
        asset_id = 9101 if "project-A" in str(cache_root) else 9202
        start.wait(timeout=10.0)
        return [copy.deepcopy(_candidate(asset_id))], [{"round": 1}], []

    def download(_session: Any, _url: str, destination: Path, _secret: str) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((destination.name.encode("utf-8") + b"x") * 5000)

    def quality(path: Path, *_args: Any, **_kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        asset_id = 9101 if "9101" in path.name else 9202
        fingerprint = {
            "sha256": f"{asset_id:064x}",
            "perceptual_hashes": [f"{asset_id:016x}"],
            "duration_seconds": 9.0,
            "width": 1920,
            "height": 1080,
            "size_bytes": path.stat().st_size,
        }
        usable = [{"start": 0.5, "end": 8.5, "duration": 8.0, "score": 0.9}]
        return (
            {
                "passed": True,
                "rejection_reasons": [],
                "motion_score": 0.4,
                "scene_category": "nature",
                "face_content_risk": 0.0,
                "subject_profile": {},
                "usable_segments": usable,
            },
            {
                "duration_seconds": 9.0,
                "width": 1920,
                "height": 1080,
                "fps": 30.0,
                "ratio": "16:9",
                "fingerprint": fingerprint,
                "usable_segments": usable,
            },
        )

    monkeypatch.setattr(pipeline, "_collect_candidates", collect)
    monkeypatch.setattr(pipeline, "_score_candidates", lambda _s, values, *_a, **_k: values)
    monkeypatch.setattr(pipeline, "_download_video", download)
    monkeypatch.setattr(pipeline, "_video_quality", quality)

    def run(label: str) -> dict[str, Any]:
        return pipeline.run_pixabay_pipeline(
            theme="共享 主题",
            style_profile={},
            audio_profile={},
            material_root=tmp_path / "shared-materials",
            cache_dir=tmp_path / f"project-{label}" / ".bgm-montage-cache" / "pixabay",
            desired_count=1,
            aspect_ratio="16:9",
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = [future.result(timeout=20.0) for future in (pool.submit(run, "A"), pool.submit(run, "B"))]

    snapshot_paths = [Path(result["sources_manifest"]) for result in results]
    assert snapshot_paths[0] != snapshot_paths[1]
    for result, snapshot_path in zip(results, snapshot_paths):
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        assert [item["pixabay_id"] for item in snapshot["sources"]] == [
            item["pixabay_id"] for item in result["selected"]
        ]
        assert result["shared_sources_manifest"].endswith("sources.json")


def test_usage_event_id_includes_run_id(tmp_path: Path) -> None:
    media = tmp_path / "asset.mp4"
    media.write_bytes(b"fixture")
    source = pipeline._complete_asset_record(
        {
            **_library_entry(media, 44),
            "id": 44,
            "usage_intervals": [],
            "actual_usage_intervals": [],
        },
        download_status="cached",
    )
    manifest_path = tmp_path / "sources.json"
    manifest_path.write_text(
        json.dumps({"schema_version": 3, "sources": [source], "assets": [source]}),
        encoding="utf-8",
    )
    shots = [
        {
            "pixabay_id": 44,
            "output_start": 0.0,
            "output_end": 2.0,
            "source_start": 1.0,
            "source_end": 3.0,
        }
    ]

    run_a = pipeline.update_usage_intervals(manifest_path, {"run_id": "run-A", "shots": shots})
    resume_a = pipeline.update_usage_intervals(manifest_path, {"run_id": "run-A", "shots": shots})
    run_b = pipeline.update_usage_intervals(manifest_path, {"run_id": "run-B", "shots": shots})

    assert run_a["event_id"] == resume_a["event_id"]
    assert run_b["event_id"] != run_a["event_id"]
    updated = json.loads(manifest_path.read_text(encoding="utf-8"))["sources"][0]
    assert updated["historical_usage_count"] == 2
    assert {item["run_id"] for item in updated["usage_history"]} == {"run-A", "run-B"}


def test_concurrent_usage_updates_preserve_both_run_events(tmp_path: Path) -> None:
    media = tmp_path / "asset.mp4"
    media.write_bytes(b"fixture")
    source = pipeline._complete_asset_record(
        {
            **_library_entry(media, 45),
            "id": 45,
            "usage_intervals": [],
            "actual_usage_intervals": [],
        },
        download_status="cached",
    )
    manifest_path = tmp_path / "sources.json"
    manifest_path.write_text(
        json.dumps({"schema_version": 3, "sources": [source], "assets": [source]}),
        encoding="utf-8",
    )
    start = Barrier(2)

    def update(run_id: str) -> dict[str, Any]:
        start.wait(timeout=5.0)
        return pipeline.update_usage_intervals(
            manifest_path,
            {
                "run_id": run_id,
                "shots": [
                    {
                        "pixabay_id": 45,
                        "output_start": 0.0,
                        "output_end": 2.0,
                        "source_start": 1.0,
                        "source_end": 3.0,
                    }
                ],
            },
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = [future.result(timeout=10.0) for future in (pool.submit(update, "run-A"), pool.submit(update, "run-B"))]

    assert results[0]["event_id"] != results[1]["event_id"]
    updated = json.loads(manifest_path.read_text(encoding="utf-8"))["sources"][0]
    assert updated["historical_usage_count"] == 2
    assert {item["run_id"] for item in updated["usage_history"]} == {"run-A", "run-B"}


def test_catalog_corruption_and_permission_errors_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog = tmp_path / "material_index.json"
    catalog.write_text("{broken-json", encoding="utf-8")
    broken_bytes = catalog.read_bytes()
    with pytest.raises(pipeline.PixabayPipelineError, match="Cannot safely read JSON state"):
        pipeline._persist_material_libraries([catalog], [], "fixture-secret")
    assert catalog.read_bytes() == broken_bytes

    valid = json.dumps({"schema_version": 3, "entries": []}).encode("utf-8")
    catalog.write_bytes(valid)
    real_open = Path.open

    def denied(path: Path, *args: Any, **kwargs: Any) -> Any:
        mode = str(args[0]) if args else str(kwargs.get("mode", "r"))
        if path == catalog and "r" in mode:
            raise PermissionError("fixture denied")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", denied)
    with pytest.raises(pipeline.PixabayPipelineError, match="Cannot safely read JSON state"):
        pipeline._persist_material_libraries([catalog], [], "fixture-secret")
    with real_open(catalog, "rb") as handle:
        assert handle.read() == valid


def test_lock_write_failure_closes_descriptor_and_removes_own_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "material_index.json"
    captured: list[int] = []

    def fail_write(descriptor: int, _payload: bytes) -> int:
        captured.append(descriptor)
        raise OSError("fixture write failure")

    monkeypatch.setattr(pipeline.os, "write", fail_write)
    with pytest.raises(OSError, match="fixture write failure"):
        with pipeline._exclusive_catalog_lock(target):
            pytest.fail("lock acquisition must not succeed")
    assert captured
    for descriptor in captured:
        with pytest.raises(OSError):
            os.fstat(descriptor)
    assert not target.with_name(f".{target.name}.lock").exists()


def test_asset_lock_recovers_dead_owner_but_never_deletes_foreign_token(tmp_path: Path) -> None:
    global_index = tmp_path / "global" / "material_index.json"
    lock_path = global_index.parent / ".asset-locks" / "pixabay-77.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(
        json.dumps(
            {
                "owner_token": "dead-owner",
                "pid": 2_147_483_647,
                "created_at_epoch": time.time() - 60.0,
                "purpose": "fixture",
            }
        ),
        encoding="utf-8",
    )
    old = time.time() - 60.0
    os.utime(lock_path, (old, old))

    with pipeline._exclusive_asset_lock(
        global_index,
        77,
        timeout_seconds=2.0,
        stale_seconds=0.05,
        heartbeat_seconds=0.01,
    ) as owner_token:
        assert owner_token != "dead-owner"
        foreign = {
            "owner_token": "successor-token",
            "pid": os.getpid(),
            "created_at_epoch": time.time(),
            "purpose": "fixture successor",
        }
        lock_path.write_text(json.dumps(foreign), encoding="utf-8")

    assert lock_path.is_file(), "old owner must not unlink a successor token"
    assert json.loads(lock_path.read_text(encoding="utf-8"))["owner_token"] == "successor-token"
    lock_path.unlink()


@pytest.mark.skipif(os.name != "nt", reason="Windows spawn regression test")
def test_windows_spawn_asset_lock_serializes_processes(tmp_path: Path) -> None:
    context = mp.get_context("spawn")
    start_event = context.Event()
    event_queue = context.Queue()
    global_index = tmp_path / "spawn global" / "material_index.json"
    processes = [
        context.Process(
            target=_spawn_asset_lock_worker,
            args=(str(global_index), start_event, event_queue, 0.30),
        )
        for _ in range(2)
    ]
    for process in processes:
        process.start()
    start_event.set()
    events = [event_queue.get(timeout=15.0) for _ in range(4)]
    for process in processes:
        process.join(timeout=15.0)
        assert process.exitcode == 0

    ordered = sorted(events, key=lambda item: item[2])
    assert [item[0] for item in ordered] == ["enter", "exit", "enter", "exit"]
    assert ordered[0][1] != ordered[2][1]
    assert ordered[2][2] >= ordered[1][2] - 0.01
