from __future__ import annotations

import copy
import json
import re
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest


SKILL_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = SKILL_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import pixabay_pipeline as pipeline  # noqa: E402


_ENVIRONMENT_HITS = (
    (101, "forest mountain landscape aerial nature", "nature"),
    (102, "city architecture building skyline street", "architecture"),
    (103, "road train traffic transport wide", "transport"),
    (104, "ocean coast beach waves close up", "water_coast"),
)
_FACE_HIT = (105, "selfie portrait close up face person posing", "people")
_PHASHES = {
    101: "0000000000000000",
    102: "ffffffffffffffff",
    103: "aaaaaaaaaaaaaaaa",
    104: "5555555555555555",
    105: "0f0f0f0f0f0f0f0f",
    201: "00ff00ff00ff00ff",
    202: "ff00ff00ff00ff00",
    203: "3333333333333333",
    204: "cccccccccccccccc",
}


class _MockResponse:
    def __init__(
        self,
        *,
        status_code: int = 200,
        payload: dict[str, Any] | None = None,
        content: bytes = b"",
    ) -> None:
        self.status_code = status_code
        self._payload = payload or {}
        self.content = content

    def json(self) -> dict[str, Any]:
        return copy.deepcopy(self._payload)


def _hit(asset_id: int, tags: str) -> dict[str, Any]:
    return {
        "id": asset_id,
        "pageURL": f"https://pixabay.com/videos/mock-{asset_id}/",
        "tags": tags,
        "duration": 8,
        "user": f"mock-author-{asset_id}",
        "user_id": asset_id + 1000,
        "views": 100_000 - asset_id,
        "downloads": 25_000 - asset_id,
        "likes": 500,
        "comments": 3,
        "videos": {
            "large": {
                "url": f"https://video.mock/{asset_id}.mp4",
                "width": 1920,
                "height": 1080,
                "size": 2_000_000,
                "thumbnail": f"https://thumb.mock/{asset_id}.jpg",
            },
            "tiny": {
                "url": f"https://video.mock/{asset_id}-tiny.mp4",
                "width": 640,
                "height": 360,
                "size": 100_000,
            },
        },
    }


def _thumbnail_bytes(seed: int) -> bytes:
    """Return deterministic, face-free, nonidentical JPEG fixtures."""

    height, width = 180, 320
    x = np.linspace(0, 1, width, dtype=np.float32)[None, :, None]
    y = np.linspace(0, 1, height, dtype=np.float32)[:, None, None]
    base = np.array(
        [
            35 + (seed * 19) % 90,
            55 + (seed * 29) % 100,
            70 + (seed * 37) % 110,
        ],
        dtype=np.float32,
    )[None, None, :]
    image = np.clip(base + 55 * x + 35 * y, 0, 255).astype(np.uint8)
    cv2.rectangle(
        image,
        (20 + seed % 30, 25),
        (120 + seed % 40, 145),
        (30 + seed % 80, 180, 80 + seed % 100),
        thickness=-1,
    )
    cv2.line(image, (0, 160 - seed % 30), (319, 50 + seed % 40), (235, 220, 80), 5)
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    assert ok
    return encoded.tobytes()


def _configure_runtime(monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
    monkeypatch.setenv("PIXABAY_API_KEY", "offline-mock-key")
    monkeypatch.setenv("BGM_MONTAGE_PROJECT_ROOT", str(root / "project"))
    monkeypatch.setenv("BGM_MONTAGE_LIBRARY_ROOT", str(root / "global-library"))


def _install_mock_network(
    monkeypatch: pytest.MonkeyPatch,
    hits: list[dict[str, Any]],
) -> dict[str, list[str]]:
    calls: dict[str, list[str]] = {"api": [], "thumbnail": []}
    payload = {
        "total": len(hits),
        "totalHits": len(hits),
        "hits": hits,
    }

    def fake_get(
        _session: Any,
        url: str,
        *args: Any,
        **kwargs: Any,
    ) -> _MockResponse:
        if url == pipeline.PIXABAY_VIDEO_API:
            query = str((kwargs.get("params") or {}).get("q") or "")
            calls["api"].append(query)
            return _MockResponse(payload=payload)
        match = re.fullmatch(r"https://thumb\.mock/(\d+)\.jpg", str(url))
        if match:
            asset_id = int(match.group(1))
            calls["thumbnail"].append(str(asset_id))
            return _MockResponse(content=_thumbnail_bytes(asset_id))
        raise AssertionError(f"unexpected network request in offline test: {url}")

    monkeypatch.setattr(pipeline.requests.Session, "get", fake_get)
    return calls


def _install_mock_video_backend(
    monkeypatch: pytest.MonkeyPatch,
    *,
    scene_by_id: dict[int, str],
    rejected_ids: set[int] | None = None,
) -> list[int]:
    downloads: list[int] = []
    rejected_ids = rejected_ids or set()

    def fake_download(
        _session: Any,
        url: str,
        destination: Path,
        _secret: str,
    ) -> None:
        match = re.search(r"/(\d+)\.mp4$", url)
        assert match, url
        asset_id = int(match.group(1))
        downloads.append(asset_id)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((f"offline-video-{asset_id}\n").encode("ascii") * 8)

    def fake_quality(
        path: Path,
        _style_profile: dict[str, Any] | None,
        _min_resolution: tuple[int, int],
        tags: str = "",
        human_focused: bool = False,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        del tags, human_focused
        match = re.search(r"\.pixabay_(\d+)_", path.name)
        assert match, path
        asset_id = int(match.group(1))
        rejected = asset_id in rejected_ids
        fingerprint = {
            "sha256": f"{asset_id:064x}",
            "perceptual_hashes": [_PHASHES[asset_id]],
            "duration_seconds": 8.0,
            "width": 1920,
            "height": 1080,
            "size_bytes": path.stat().st_size,
        }
        subject = {
            "source_width": 1920,
            "source_height": 1080,
            "preferred_center": {"x": 0.5, "y": 0.5},
            "saliency_bbox_normalized": {"x": 0.25, "y": 0.2, "w": 0.5, "h": 0.6},
            "face_count": 0,
            "face_frame_ratio": 0.0,
            "largest_face_frame_ratio": 0.0,
        }
        quality = {
            "passed": not rejected,
            "rejection_reasons": ["fixture QA rejection"] if rejected else [],
            "overall_score": 0.9 if not rejected else 0.1,
            "resolution_score": 1.0,
            "sharpness_score": 0.9,
            "exposure_score": 0.9,
            "stability_score": 0.9,
            "color_score": 0.8,
            "text_watermark_risk": 0.0,
            "motion_score": 0.45,
            "face_content_risk": 0.05,
            "subject_profile": subject,
            "scene_category": scene_by_id.get(asset_id, "general"),
            "analysis_cache": {
                "schema_version": 2,
                "engine_version": "1.3.0",
                "file_sha256": fingerprint["sha256"],
            },
        }
        media = {
            "duration_seconds": 8.0,
            "width": 1920,
            "height": 1080,
            "fps": 30.0,
            "fingerprint": fingerprint,
        }
        return quality, media

    monkeypatch.setattr(pipeline, "_download_video", fake_download)
    monkeypatch.setattr(pipeline, "_video_quality", fake_quality)
    return downloads


def _run_pipeline(root: Path, theme: str, desired_count: int = 4) -> dict[str, Any]:
    return pipeline.run_pixabay_pipeline(
        theme=theme,
        style_profile={
            "search_hints": {
                "positive_terms": ["nature", "architecture", "transport", "ocean"],
                "negative_terms": ["selfie", "interview", "portrait"],
            }
        },
        audio_profile={"summary": {"duration_seconds": 12.0, "energy": "medium"}},
        material_root=root / "materials",
        cache_dir=root / "project" / ".bgm-montage-cache",
        desired_count=desired_count,
        aspect_ratio="16:9",
        min_resolution=(1280, 720),
        target_duration=12.0,
    )


def test_mock_pipeline_expands_filters_downloads_selected_only_and_reuses_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_runtime(monkeypatch, tmp_path)
    hits = [_hit(asset_id, tags) for asset_id, tags, _scene in _ENVIRONMENT_HITS]
    hits.append(_hit(_FACE_HIT[0], _FACE_HIT[1]))
    # This API result must be filtered before ranking because it has no usable
    # video variant.
    hits.append({"id": 999, "tags": "broken candidate", "videos": {}})
    network_calls = _install_mock_network(monkeypatch, hits)
    scene_by_id = {asset_id: scene for asset_id, _tags, scene in (*_ENVIRONMENT_HITS, _FACE_HIT)}
    downloads = _install_mock_video_backend(monkeypatch, scene_by_id=scene_by_id)

    first = _run_pipeline(tmp_path, "nature architecture transport ocean")
    manifest_path = Path(first["sources_manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert first["status"] == "ok"
    assert len(first["search_rounds"]) == 3
    assert [item["expansion_level"] for item in first["search_rounds"]] == [0, 1, 2]
    assert all(item["queries"] for item in first["search_rounds"])
    assert first["search_rounds"][-1]["stop_reason"] == "maximum bounded expansion reached"
    assert network_calls["api"]
    assert len(network_calls["thumbnail"]) == 4

    # The dynamic world/profile filter rejects the unrelated face candidate
    # before thumbnail ranking; only the four environment clips proceed.
    assert first["candidate_count"] == 4
    assert len(downloads) == 4
    assert set(downloads) == {101, 102, 103, 104}
    assert 105 not in downloads
    assert not any(item["pixabay_id"] == 105 for item in manifest["candidate_log"])

    assert first["selected_count"] == 4
    assert first["sufficiency"]["passed"] is True
    assert first["sufficiency"]["independent_asset_count"] == 4
    assert len(first["sufficiency"]["scene_categories"]) >= 3
    assert (
        first["sufficiency"]["low_face_risk_assets"]
        >= first["sufficiency"]["required_low_face_risk_assets"]
    )
    assert first["sufficiency"]["theoretical_screen_coverage_seconds"] >= 12.0 * 0.95
    assert manifest["status"] == "ok"
    assert manifest["manifest_type"] == "asset_manifest"
    assert manifest["asset_manifest_schema_version"] == 2
    assert manifest["assets"] == manifest["sources"]
    assert manifest["sufficiency"]["passed"] is True

    required_source_fields = {
        "pixabay_id",
        "author",
        "page_url",
        "search_query",
        "search_queries",
        "local_path",
        "actual_usage_intervals",
        "scene_category",
        "face_content_risk",
        "fingerprint",
        "reuse_mode",
        "canonical_source_id",
        "download_url",
        "file_hash",
        "ratio",
        "semantic_tags",
        "download_status",
        "available",
        "failure_reason",
        "historical_usage_count",
        "usage_history",
        "usable_segments",
    }
    for source in manifest["sources"]:
        assert required_source_fields <= source.keys()
        assert source["author"]
        assert source["page_url"].startswith("https://pixabay.com/videos/")
        assert source["search_query"]
        assert Path(source["local_path"]).is_file()
        assert source["face_content_risk"] < 0.65

    first_api_count = len(network_calls["api"])
    first_thumbnail_count = len(network_calls["thumbnail"])

    def fail_network(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("second identical run must use the search and thumbnail caches")

    monkeypatch.setattr(pipeline.requests.Session, "get", fail_network)

    def fail_download(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("second identical run must reuse previously selected material")

    monkeypatch.setattr(pipeline, "_download_video", fail_download)
    second = _run_pipeline(tmp_path, "nature architecture transport ocean")

    assert second["status"] == "ok"
    assert all(
        query["cache_hit"]
        for round_record in second["search_rounds"]
        for query in round_record["queries"]
    )
    assert len(network_calls["api"]) == first_api_count
    assert len(network_calls["thumbnail"]) == first_thumbnail_count
    assert all(source["reused_existing_file"] for source in second["selected"])


def test_mock_pipeline_persists_three_round_failure_when_qa_is_insufficient(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_runtime(monkeypatch, tmp_path)
    fixture_rows = (
        (201, "forest mountain landscape aerial", "nature"),
        (202, "city architecture building street", "architecture"),
        (203, "road train traffic transport wide", "transport"),
        (204, "ocean coast beach waves detail", "water_coast"),
    )
    hits = [_hit(asset_id, tags) for asset_id, tags, _scene in fixture_rows]
    network_calls = _install_mock_network(monkeypatch, hits)
    scene_by_id = {asset_id: scene for asset_id, _tags, scene in fixture_rows}
    downloads = _install_mock_video_backend(
        monkeypatch,
        scene_by_id=scene_by_id,
        rejected_ids={202, 203, 204},
    )
    theme = "nature architecture transport ocean insufficient"

    with pytest.raises(pipeline.InsufficientMaterialError, match="sufficiency gate failed"):
        _run_pipeline(tmp_path, theme)

    manifest_path = pipeline.material_theme_directory(tmp_path / "materials", theme) / "sources.json"
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["status"] == "insufficient_material"
    assert manifest["selected_count"] == 1
    assert manifest["sufficiency"]["passed"] is False
    assert manifest["sufficiency"]["failures"]
    assert len(manifest["search_rounds"]) == 3
    assert [item["round"] for item in manifest["search_rounds"]] == [1, 2, 3]
    assert manifest["search_rounds"][-1]["stop_reason"] == "maximum bounded expansion reached"
    assert len(network_calls["api"]) >= 3
    assert len(downloads) == 4
    assert set(downloads) == {201, 202, 203, 204}
    assert len(manifest["rejections"]) == 3
    assert all(item["stage"] == "post_download_qa" for item in manifest["rejections"])
    assert len(manifest["sources"]) == 1
