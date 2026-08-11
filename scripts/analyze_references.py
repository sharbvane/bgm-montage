#!/usr/bin/env python3
"""Read-only reference-video analysis for the bgm-montage skill.

The v1.2 analyzer combines explainable OpenCV measurements with a real,
pretrained CLIP zero-shot vision model.  Semantic output is always labelled
with its backend and limitations; when the model cannot be loaded the result
is explicitly degraded instead of pretending that heuristics are semantics.

Public API:
    analyze_references(reference_dir, cache_dir, output_path=None, ...)

The source directory is only opened for stat/read operations.  Cache and
profile files are always written below ``cache_dir`` or to ``output_path``.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from runtime_paths import RuntimePaths, discover_project_root
from visual_semantics import (
    ClipSemanticAnalyzer,
    SemanticBackendStatus,
    aggregate_semantics,
    aggregate_subject_regions,
    infer_scene_category,
    load_semantic_analyzer,
    subject_region,
)


ANALYZER_VERSION = "1.2.0"
# v1.2 derives its additional pacing/transition/diversity fields from the
# shot-level records already stored by v1.1.  Reusing that evidence avoids an
# expensive model pass while the existing fingerprint check still forces
# changed and newly added reference files through the full analyzer.
MIGRATABLE_ANALYZER_VERSIONS: set[str] = {"1.1.0"}
CACHE_SCHEMA_VERSION = 2
PROFILE_SCHEMA_VERSION = "1.2"
VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".m4v",
    ".mkv",
    ".webm",
    ".avi",
    ".wmv",
    ".flv",
    ".mts",
    ".m2ts",
    ".ts",
}

# Sampling is intentionally bounded: the reference folder can contain hundreds
# of short-form videos.  Two samples per second still exposes short-form cut
# cadence while keeping first-run analysis practical.
TARGET_SAMPLE_FPS = 2.0
MIN_SAMPLES = 16
MAX_SAMPLES = 96
MAX_SEMANTIC_SAMPLES = 18
ANALYSIS_WIDTH = 384
FINGERPRINT_BLOCK_BYTES = 256 * 1024


class ReferenceAnalysisError(RuntimeError):
    """Base error for a reference-analysis setup or execution failure."""


class ReferenceAnalysisDependencyError(ReferenceAnalysisError):
    """Raised when an external analysis dependency is unavailable."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return number if math.isfinite(number) else default


def _round(value: Any, digits: int = 4) -> float:
    return round(_finite_float(value), digits)


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, _finite_float(value)))


def _mean(values: Iterable[float], default: float = 0.0) -> float:
    cleaned = [_finite_float(value) for value in values]
    return statistics.fmean(cleaned) if cleaned else default


def _median(values: Iterable[float], default: float = 0.0) -> float:
    cleaned = [_finite_float(value) for value in values]
    return statistics.median(cleaned) if cleaned else default


def _percentile(values: Sequence[float], quantile: float, default: float = 0.0) -> float:
    cleaned = sorted(_finite_float(value) for value in values)
    if not cleaned:
        return default
    if len(cleaned) == 1:
        return cleaned[0]
    position = _clamp(quantile) * (len(cleaned) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return cleaned[lower]
    fraction = position - lower
    return cleaned[lower] * (1.0 - fraction) + cleaned[upper] * fraction


def _distribution(labels: Iterable[str]) -> dict[str, float]:
    counts = Counter(label for label in labels if label)
    total = sum(counts.values())
    if not total:
        return {}
    return {
        label: _round(count / total)
        for label, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    }


def _average_distributions(distributions: Iterable[Mapping[str, Any]]) -> dict[str, float]:
    totals: defaultdict[str, float] = defaultdict(float)
    count = 0
    for distribution in distributions:
        if not distribution:
            continue
        count += 1
        for label, value in distribution.items():
            totals[str(label)] += _finite_float(value)
    if not count:
        return {}
    averaged = {label: value / count for label, value in totals.items()}
    denominator = sum(averaged.values()) or 1.0
    return {
        label: _round(value / denominator)
        for label, value in sorted(averaged.items(), key=lambda item: (-item[1], item[0]))
    }


def _dominant(distribution: Mapping[str, Any], default: str = "unresolved") -> str:
    if not distribution:
        return default
    return max(distribution.items(), key=lambda item: (_finite_float(item[1]), item[0]))[0]


def _parse_rate(value: Any) -> float:
    if value in (None, "", "0/0", "N/A"):
        return 0.0
    try:
        return float(Fraction(str(value)))
    except (ValueError, ZeroDivisionError, TypeError):
        return _finite_float(value)


def _atomic_json_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            json.dump(payload, temporary, ensure_ascii=False, indent=2, allow_nan=False)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, path)
    finally:
        if temporary_name and os.path.exists(temporary_name):
            try:
                os.unlink(temporary_name)
            except OSError:
                pass


def _load_json(path: Path) -> tuple[dict[str, Any], str | None]:
    if not path.exists():
        return {}, None
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        if not isinstance(value, dict):
            return {}, "cache root was not a JSON object; ignored"
        return value, None
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return {}, f"cache could not be read and was ignored: {exc}"


def _require_dependencies() -> tuple[Any, Any, str]:
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise ReferenceAnalysisDependencyError(
            "OpenCV is required. Install the skill dependencies (opencv-python-headless)."
        ) from exc
    try:
        import numpy as np  # type: ignore
    except ImportError as exc:
        raise ReferenceAnalysisDependencyError(
            "NumPy is required. Install the skill dependencies (numpy)."
        ) from exc
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise ReferenceAnalysisDependencyError(
            "ffprobe was not found on PATH. Install FFmpeg and make ffprobe available."
        )
    return cv2, np, ffprobe


def _discover_videos(reference_dir: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in reference_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
        ),
        key=lambda path: path.relative_to(reference_dir).as_posix().casefold(),
    )


def _fingerprint(path: Path) -> dict[str, Any]:
    """Build a size/mtime/content-sample fingerprint without modifying *path*."""

    stat = path.stat()
    size = int(stat.st_size)
    block = FINGERPRINT_BLOCK_BYTES
    if size <= block * 3:
        offsets = [0]
        hash_mode = "full_sha256"
    else:
        offsets = [0, max(0, size // 2 - block // 2), max(0, size - block)]
        hash_mode = "first_middle_last_sha256"

    # Keep this a true digest of the bytes read; size/mtime remain separate
    # fingerprint fields. For large files the sampled offsets are deterministic
    # from file size (first, centred middle and last block).
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        if hash_mode == "full_sha256":
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        else:
            for offset in offsets:
                handle.seek(offset)
                data = handle.read(block)
                digest.update(data)
    return {
        "size": size,
        "mtime_ns": int(stat.st_mtime_ns),
        "content_sha256": digest.hexdigest(),
        "hash_algorithm": "sha256",
        "hash_mode": hash_mode,
        "sample_bytes": size if hash_mode == "full_sha256" else min(size, block * len(offsets)),
    }


def _ffprobe(path: Path, executable: str) -> dict[str, Any]:
    command = [
        executable,
        "-v",
        "error",
        "-show_streams",
        "-show_format",
        "-print_format",
        "json",
        os.fspath(path),
    ]
    try:
        result = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=90,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ReferenceAnalysisError(f"ffprobe failed: {exc}") from exc
    if result.returncode != 0:
        message = result.stderr.strip().replace("\r", " ").replace("\n", " ")
        raise ReferenceAnalysisError(f"ffprobe returned {result.returncode}: {message[:500]}")
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise ReferenceAnalysisError("ffprobe returned invalid JSON") from exc

    streams = payload.get("streams") or []
    video_stream = next((item for item in streams if item.get("codec_type") == "video"), None)
    if not video_stream:
        raise ReferenceAnalysisError("no video stream found")
    audio_stream = next((item for item in streams if item.get("codec_type") == "audio"), None)
    format_data = payload.get("format") or {}

    duration = _finite_float(video_stream.get("duration")) or _finite_float(format_data.get("duration"))
    width = int(_finite_float(video_stream.get("width")))
    height = int(_finite_float(video_stream.get("height")))
    rotation = 0
    tags = video_stream.get("tags") or {}
    if "rotate" in tags:
        rotation = int(round(_finite_float(tags.get("rotate")))) % 360
    for side_data in video_stream.get("side_data_list") or []:
        if "rotation" in side_data:
            rotation = int(round(_finite_float(side_data.get("rotation")))) % 360
            break
    display_width, display_height = width, height
    if rotation in (90, 270):
        display_width, display_height = height, width
    aspect_ratio = display_width / display_height if display_height else 0.0
    if aspect_ratio < 0.85:
        orientation = "portrait"
    elif aspect_ratio > 1.18:
        orientation = "landscape"
    else:
        orientation = "square_like"

    fps = _parse_rate(video_stream.get("avg_frame_rate")) or _parse_rate(
        video_stream.get("r_frame_rate")
    )
    frame_count = int(_finite_float(video_stream.get("nb_frames")))
    if not frame_count and fps and duration:
        frame_count = int(round(fps * duration))

    metadata: dict[str, Any] = {
        "duration_seconds": _round(duration, 3),
        "coded_width": width,
        "coded_height": height,
        "display_width": display_width,
        "display_height": display_height,
        "aspect_ratio": _round(aspect_ratio),
        "orientation": orientation,
        "rotation_degrees": rotation,
        "fps": _round(fps, 3),
        "estimated_frame_count": frame_count,
        "video_codec": video_stream.get("codec_name") or "unknown",
        "pixel_format": video_stream.get("pix_fmt") or "unknown",
        "video_bitrate": int(_finite_float(video_stream.get("bit_rate"))),
        "container": format_data.get("format_name") or "unknown",
        "file_bitrate": int(_finite_float(format_data.get("bit_rate"))),
        "has_audio": audio_stream is not None,
    }
    if audio_stream:
        metadata["audio"] = {
            "codec": audio_stream.get("codec_name") or "unknown",
            "sample_rate": int(_finite_float(audio_stream.get("sample_rate"))),
            "channels": int(_finite_float(audio_stream.get("channels"))),
        }
    return metadata


def _resize_frame(frame: Any, cv2: Any) -> Any:
    height, width = frame.shape[:2]
    if width <= ANALYSIS_WIDTH:
        return frame
    scale = ANALYSIS_WIDTH / width
    return cv2.resize(
        frame,
        (ANALYSIS_WIDTH, max(2, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA,
    )


def _text_region_scores(gray: Any, cv2: Any, np: Any) -> dict[str, float]:
    """Estimate text-like horizontal edge groups; this is not OCR."""

    height, width = gray.shape[:2]
    gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    _, binary = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    close_width = max(7, int(round(width * 0.025)))
    connected = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (close_width, 3)),
    )
    contours_data = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours_data[-2]
    region_scores: defaultdict[str, float] = defaultdict(float)
    candidate_count = 0
    frame_area = max(1, width * height)
    for contour in contours:
        x, y, box_width, box_height = cv2.boundingRect(contour)
        if box_height < max(3, height * 0.012) or box_height > height * 0.14:
            continue
        if box_width < max(12, width * 0.04) or box_width > width * 0.96:
            continue
        if box_width / max(1, box_height) < 1.35:
            continue
        box_area = box_width * box_height
        fill = cv2.countNonZero(connected[y : y + box_height, x : x + box_width]) / max(1, box_area)
        if fill < 0.10 or fill > 0.85:
            continue
        candidate_count += 1
        center_x = (x + box_width / 2) / width
        center_y = (y + box_height / 2) / height
        if center_y >= 0.62:
            vertical = "bottom"
        elif center_y <= 0.30:
            vertical = "top"
        else:
            vertical = "middle"
        horizontal = "center" if 0.25 <= center_x <= 0.75 else ("left" if center_x < 0.5 else "right")
        region = f"{vertical}_{horizontal}"
        weight = min(1.0, box_area / (frame_area * 0.035)) * (0.5 + 0.5 * fill)
        region_scores[region] += weight

    # Requiring multiple text-like boxes suppresses many single object edges.
    confidence_scale = min(1.0, candidate_count / 3.0)
    return {
        region: _round(min(1.0, score / 2.0) * confidence_scale)
        for region, score in region_scores.items()
    }


def _focus_footprint(edge_map: Any, np: Any) -> float:
    """Return the area containing the central 70% of edge energy."""

    weights = edge_map.astype("float32") / 255.0
    total = float(weights.sum())
    if total <= 1e-6:
        return 0.0
    rows = weights.sum(axis=1)
    columns = weights.sum(axis=0)

    def bounds(vector: Any) -> tuple[int, int]:
        cumulative = np.cumsum(vector)
        low = int(np.searchsorted(cumulative, total * 0.15))
        high = int(np.searchsorted(cumulative, total * 0.85))
        return low, max(low + 1, high)

    top, bottom = bounds(rows)
    left, right = bounds(columns)
    height, width = edge_map.shape[:2]
    return _clamp(((bottom - top) / height) * ((right - left) / width))


def _load_face_cascade(cv2: Any) -> Any | None:
    try:
        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        if not cascade_path.is_file():
            return None
        load_path = cascade_path
        # OpenCV FileStorage on Windows cannot reliably open non-ASCII paths.
        # A skill venv may live under a Chinese project directory, so mirror
        # the bundled XML to an ASCII-only temp path before loading it.
        if not os.fspath(cascade_path).isascii():
            safe_root = Path(tempfile.gettempdir()) / "bgm_montage_cv2"
            safe_root.mkdir(parents=True, exist_ok=True)
            safe_path = safe_root / cascade_path.name
            if not safe_path.is_file() or safe_path.stat().st_size != cascade_path.stat().st_size:
                temporary = safe_path.with_suffix(".tmp")
                shutil.copyfile(cascade_path, temporary)
                os.replace(temporary, safe_path)
            load_path = safe_path
        cascade = cv2.CascadeClassifier(os.fspath(load_path))
        return None if cascade.empty() else cascade
    except (AttributeError, OSError, cv2.error):
        return None


def _frame_metrics(frame: Any, face_cascade: Any | None, cv2: Any, np: Any) -> dict[str, Any]:
    frame = _resize_frame(frame, cv2)
    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    edges = cv2.Canny(gray, 70, 160)

    brightness = float(gray.mean()) / 255.0
    saturation = float(hsv[:, :, 1].mean()) / 255.0
    contrast = min(1.0, float(gray.std()) / 96.0)
    laplacian_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    sharpness_score = laplacian_variance / (laplacian_variance + 450.0)
    underexposed = float(np.mean(gray <= 20))
    overexposed = float(np.mean(gray >= 235))
    exposure_quality = 1.0 - min(1.0, (underexposed + overexposed) * 2.3)
    edge_density = float(np.mean(edges > 0))

    # Positive values mean red-dominant (warm-like); negative values mean
    # blue-dominant (cool-like).  This is a white-balance proxy, not Kelvin.
    blue = float(frame[:, :, 0].mean())
    red = float(frame[:, :, 2].mean())
    warmth_index = (red - blue) / max(1.0, red + blue)

    histogram = cv2.calcHist([hsv], [0, 1], None, [24, 16], [0, 180, 0, 256])
    cv2.normalize(histogram, histogram, alpha=1.0, norm_type=cv2.NORM_L1)

    l_hist = cv2.calcHist([lab], [0], None, [32], [0, 256]).reshape(-1)
    l_probabilities = l_hist / max(1.0, float(l_hist.sum()))
    nonzero = l_probabilities[l_probabilities > 0]
    entropy = float(-(nonzero * np.log2(nonzero)).sum() / math.log2(32))

    face_count = 0
    largest_face_ratio = 0.0
    if face_cascade is not None and min(height, width) >= 64:
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.15,
            minNeighbors=5,
            minSize=(max(22, width // 18), max(22, height // 18)),
        )
        face_count = len(faces)
        if face_count:
            largest_face_ratio = max((box_width * box_height) / (width * height) for _, _, box_width, box_height in faces)

    footprint = _focus_footprint(edges, np)
    if largest_face_ratio >= 0.075:
        scale_label = "close_up_like"
        scale_method = "face_size"
    elif largest_face_ratio >= 0.018:
        scale_label = "medium_like"
        scale_method = "face_size"
    elif largest_face_ratio > 0:
        scale_label = "wide_like"
        scale_method = "face_size"
    elif footprint >= 0.46 and edge_density >= 0.055:
        scale_label = "close_up_like"
        scale_method = "edge_footprint"
    elif footprint >= 0.20:
        scale_label = "medium_like"
        scale_method = "edge_footprint"
    else:
        scale_label = "wide_like"
        scale_method = "edge_footprint"

    return {
        "frame": frame,
        "gray": gray,
        "histogram": histogram,
        "brightness": brightness,
        "saturation": saturation,
        "contrast": contrast,
        "laplacian_variance": laplacian_variance,
        "sharpness_score": sharpness_score,
        "underexposed_fraction": underexposed,
        "overexposed_fraction": overexposed,
        "exposure_quality": exposure_quality,
        "warmth_index": warmth_index,
        "edge_density": edge_density,
        "luma_entropy": entropy,
        "mean_hue_degrees": float(hsv[:, :, 0].mean()) * 2.0,
        "text_regions": _text_region_scores(gray, cv2, np),
        "face_count": face_count,
        "largest_face_ratio": largest_face_ratio,
        "focus_footprint": footprint,
        "scale_label": scale_label,
        "scale_method": scale_method,
    }


def _transition_metrics(previous: Mapping[str, Any], current: Mapping[str, Any], elapsed: float, cv2: Any, np: Any) -> dict[str, Any]:
    previous_gray = previous["gray"]
    current_gray = current["gray"]
    histogram_distance = float(
        cv2.compareHist(previous["histogram"], current["histogram"], cv2.HISTCMP_BHATTACHARYYA)
    )
    pixel_difference = float(cv2.absdiff(previous_gray, current_gray).mean()) / 255.0
    is_cut = bool(
        (histogram_distance >= 0.64 and pixel_difference >= 0.10)
        or (histogram_distance >= 0.46 and pixel_difference >= 0.21)
    )
    cut_confidence = _clamp(
        (histogram_distance - 0.35) / 0.45 * 0.65 + (pixel_difference - 0.08) / 0.30 * 0.35
    )
    result: dict[str, Any] = {
        "is_cut": is_cut,
        "cut_confidence": cut_confidence,
        "histogram_distance": histogram_distance,
        "pixel_difference": pixel_difference,
        "motion_label": "cut" if is_cut else "unresolved",
        "motion_strength": 0.0,
        "translation_x": 0.0,
        "translation_y": 0.0,
        "zoom_rate": 0.0,
        "rotation_rate": 0.0,
        "tracking_inlier_ratio": 0.0,
        "residual_motion": 0.0,
        "transition_type": "hard_cut" if is_cut else "continuous",
    }
    if is_cut:
        return result

    # Conservative transition hints.  The analyzer intentionally keeps the
    # ``_like`` suffix because sparse sampling cannot prove an authored
    # dissolve/fade.  These labels are style evidence, never a claim that an
    # arbitrary reference effect can be reproduced exactly.
    previous_brightness = _finite_float(previous.get("brightness"))
    current_brightness = _finite_float(current.get("brightness"))
    brightness_delta = current_brightness - previous_brightness
    if (
        abs(brightness_delta) >= 0.16
        and min(previous_brightness, current_brightness) <= 0.22
        and histogram_distance < 0.62
    ):
        result["transition_type"] = "fade_like"
    elif histogram_distance >= 0.30 and pixel_difference >= 0.10:
        result["transition_type"] = "dissolve_like"

    points = cv2.goodFeaturesToTrack(
        previous_gray,
        maxCorners=180,
        qualityLevel=0.015,
        minDistance=7,
        blockSize=7,
    )
    if points is None or len(points) < 8:
        result["motion_label"] = "static_like" if pixel_difference < 0.045 else "complex_or_subject_motion"
        result["motion_strength"] = min(1.0, pixel_difference / max(elapsed, 0.1))
        return result

    tracked, status, _ = cv2.calcOpticalFlowPyrLK(
        previous_gray,
        current_gray,
        points,
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 25, 0.01),
    )
    if tracked is None or status is None:
        result["motion_label"] = "complex_or_subject_motion"
        result["motion_strength"] = min(1.0, pixel_difference / max(elapsed, 0.1))
        return result
    valid = status.reshape(-1) == 1
    source = points.reshape(-1, 2)[valid]
    target = tracked.reshape(-1, 2)[valid]
    if len(source) < 8:
        result["motion_label"] = "complex_or_subject_motion"
        result["motion_strength"] = min(1.0, pixel_difference / max(elapsed, 0.1))
        return result

    transform, inliers = cv2.estimateAffinePartial2D(
        source,
        target,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
        maxIters=1200,
        confidence=0.97,
    )
    height, width = previous_gray.shape[:2]
    diagonal = math.hypot(width, height)
    elapsed = max(0.08, elapsed)
    raw_displacement = np.linalg.norm(target - source, axis=1)
    motion_strength = float(np.median(raw_displacement)) / diagonal / elapsed
    result["motion_strength"] = motion_strength
    if transform is None:
        result["motion_label"] = "complex_or_subject_motion"
        return result

    a, b, translate_x = (float(value) for value in transform[0])
    c, d, translate_y = (float(value) for value in transform[1])
    scale = math.sqrt(max(1e-12, a * a + c * c))
    rotation = math.atan2(c, a)
    normalized_x = translate_x / width / elapsed
    normalized_y = translate_y / height / elapsed
    zoom_rate = math.log(max(scale, 1e-6)) / elapsed
    rotation_rate = math.degrees(rotation) / elapsed
    predicted = cv2.transform(source.reshape(-1, 1, 2), transform).reshape(-1, 2)
    residual = float(np.median(np.linalg.norm(predicted - target, axis=1))) / diagonal / elapsed
    inlier_ratio = float(np.mean(inliers)) if inliers is not None else 0.0

    result.update(
        {
            "translation_x": normalized_x,
            "translation_y": normalized_y,
            "zoom_rate": zoom_rate,
            "rotation_rate": rotation_rate,
            "tracking_inlier_ratio": inlier_ratio,
            "residual_motion": residual,
        }
    )
    if motion_strength < 0.007 and abs(zoom_rate) < 0.012:
        label = "static_like"
    elif abs(zoom_rate) >= 0.025 and inlier_ratio >= 0.35:
        label = "zoom_in_like" if zoom_rate > 0 else "zoom_out_like"
    elif abs(normalized_x) >= 0.012 and abs(normalized_x) > abs(normalized_y) * 1.25 and inlier_ratio >= 0.30:
        label = "pan_right_like" if normalized_x > 0 else "pan_left_like"
    elif abs(normalized_y) >= 0.012 and abs(normalized_y) > abs(normalized_x) * 1.25 and inlier_ratio >= 0.30:
        label = "tilt_down_like" if normalized_y > 0 else "tilt_up_like"
    elif abs(rotation_rate) >= 1.2 or residual >= 0.010 or inlier_ratio < 0.30:
        label = "handheld_or_complex"
    else:
        label = "drift_or_subject_motion"
    result["motion_label"] = label
    return result


def _sample_times(duration: float, fps: float, frame_count: int) -> list[float]:
    if duration <= 0 and fps > 0 and frame_count > 0:
        duration = frame_count / fps
    if duration <= 0:
        return []
    count = int(math.ceil(duration * TARGET_SAMPLE_FPS))
    count = max(MIN_SAMPLES, min(MAX_SAMPLES, count))
    if duration < 0.75:
        count = max(3, min(count, int(max(3, round(duration * max(fps, 8.0))))))
    start = min(duration * 0.025, 0.25)
    end = max(start, duration - min(duration * 0.025, 0.25))
    if count == 1 or end <= start:
        return [duration / 2.0]
    return [start + (end - start) * index / (count - 1) for index in range(count)]


def _tone_label(brightness: float, saturation: float, contrast: float, warmth: float) -> str:
    temperature = "warm" if warmth >= 0.035 else ("cool" if warmth <= -0.035 else "neutral")
    if brightness <= 0.35:
        exposure = "dark"
    elif brightness >= 0.67:
        exposure = "bright"
    else:
        exposure = "midkey"
    chroma = "vivid" if saturation >= 0.52 else ("muted" if saturation <= 0.28 else "balanced")
    contrast_label = "high_contrast" if contrast >= 0.62 else ("soft" if contrast <= 0.34 else "moderate")
    return f"{temperature}_{exposure}_{chroma}_{contrast_label}"


def _mood_label(brightness: float, saturation: float, contrast: float, warmth: float, motion: float, cuts_per_minute: float) -> str:
    if (motion >= 0.035 or cuts_per_minute >= 28) and saturation >= 0.38:
        return "energetic_lively"
    if brightness <= 0.38 and contrast >= 0.50:
        return "dramatic_moody"
    if brightness <= 0.42 and saturation <= 0.32:
        return "subdued_contemplative"
    if warmth >= 0.035 and brightness >= 0.50:
        return "warm_optimistic"
    if warmth <= -0.035 and saturation <= 0.38:
        return "cool_calm"
    if brightness >= 0.65 and saturation >= 0.42:
        return "bright_upbeat"
    return "balanced_neutral"


def _visual_search_hints(
    metadata: Mapping[str, Any],
    tone_label: str,
    mood_label: str,
    scale_distribution: Mapping[str, Any],
    motion_distribution: Mapping[str, Any],
    rhythm_label: str,
    topic_label: str,
) -> dict[str, Any]:
    positive: list[str] = []
    if metadata.get("orientation") == "portrait":
        positive.append("vertical video")
    elif metadata.get("orientation") == "landscape":
        positive.append("cinematic widescreen")

    positive.extend(
        {
            "energetic_lively": ["energetic", "dynamic"],
            "dramatic_moody": ["dramatic lighting", "moody"],
            "subdued_contemplative": ["atmospheric", "calm"],
            "warm_optimistic": ["warm light", "uplifting"],
            "cool_calm": ["cool tones", "serene"],
            "bright_upbeat": ["bright", "vibrant"],
            "balanced_neutral": ["natural lighting"],
        }.get(mood_label, [])
    )
    if tone_label.startswith("warm_") and "warm light" not in positive:
        positive.append("warm color grade")
    elif tone_label.startswith("cool_") and "cool tones" not in positive:
        positive.append("cool color grade")
    if "vivid" in tone_label:
        positive.append("saturated colors")
    elif "muted" in tone_label:
        positive.append("muted colors")

    scale_terms: list[str] = []
    for label in sorted(scale_distribution, key=lambda key: -_finite_float(scale_distribution[key])):
        if _finite_float(scale_distribution[label]) < 0.18:
            continue
        scale_terms.append(
            {
                "wide_like": "wide establishing shot",
                "medium_like": "medium shot",
                "close_up_like": "close up detail",
            }.get(label, label.replace("_", " "))
        )
    motion_terms: list[str] = []
    for label in sorted(motion_distribution, key=lambda key: -_finite_float(motion_distribution[key])):
        if _finite_float(motion_distribution[label]) < 0.16:
            continue
        motion_terms.append(
            {
                "static_like": "locked off shot",
                "pan_left_like": "camera pan",
                "pan_right_like": "camera pan",
                "tilt_up_like": "camera tilt",
                "tilt_down_like": "camera tilt",
                "zoom_in_like": "slow push in",
                "zoom_out_like": "pull out shot",
                "handheld_or_complex": "handheld camera movement",
                "drift_or_subject_motion": "subtle camera movement",
                "complex_or_subject_motion": "dynamic action",
            }.get(label, label.replace("_", " "))
        )

    structural_terms = {
        "human_presence_likely": ["people", "human activity"],
        "graphic_or_interface_like": ["clean graphic composition"],
        "environment_or_establishing_like": ["environment", "establishing view"],
        "water_sky_or_coastal_scenery_like": ["water landscape", "coastal scenery", "open sky"],
        "natural_or_travel_scenery_like": ["nature landscape", "travel scenery", "outdoors"],
        "built_environment_or_street_like": ["urban environment", "architecture", "street scene"],
        "general_environment_like": ["environment", "location atmosphere"],
        "mixed_nature_water_travel_scenery": ["nature landscape", "water scenery", "travel atmosphere"],
        "mixed_environmental_scenery": ["environmental scenery", "varied locations"],
        "detail_or_texture_like": ["detail", "texture macro"],
        "mixed_or_unresolved": [],
    }.get(topic_label, [])
    positive.extend(structural_terms)
    positive = list(dict.fromkeys(positive))
    return {
        "language": "en",
        "positive_terms": positive,
        "shot_scale_terms": list(dict.fromkeys(scale_terms)),
        "camera_terms": list(dict.fromkeys(motion_terms)),
        "rhythm_terms": [
            {
                "rapid": "fast paced montage",
                "moderate": "rhythmic montage",
                "slow": "slow cinematic sequence",
            }.get(rhythm_label, "natural pacing")
        ],
        "avoid_terms": ["watermark", "logo", "burned-in subtitles", "low resolution"],
        "topic_required": True,
        "note": "These are visual-style hints only; combine them with the user-supplied topic and BGM-stage intent.",
    }


def _classify_visual_topic(
    face_frame_ratio: float,
    text_presence: float,
    entropy: float,
    motion_strength: float,
    scale_distribution: Mapping[str, Any],
    saturation: float,
    hue_degrees: float,
    edge_density: float,
) -> tuple[str, float]:
    """Estimate broad visual subject family without a semantic model."""

    wide = _finite_float(scale_distribution.get("wide_like"))
    close = _finite_float(scale_distribution.get("close_up_like"))
    blue_cyan_like = 165.0 <= hue_degrees <= 255.0
    natural_chroma = saturation >= 0.38 and edge_density <= 0.145
    if face_frame_ratio >= 0.30:
        return "human_presence_likely", min(0.85, 0.45 + face_frame_ratio * 0.40)
    if text_presence >= 0.44 and entropy <= 0.72 and motion_strength <= 0.025:
        return "graphic_or_interface_like", 0.52
    if close >= 0.46:
        return "detail_or_texture_like", 0.42
    if blue_cyan_like and saturation >= 0.34 and edge_density <= 0.16:
        return "water_sky_or_coastal_scenery_like", min(0.64, 0.42 + saturation * 0.22 + wide * 0.10)
    if natural_chroma:
        return "natural_or_travel_scenery_like", min(0.62, 0.39 + saturation * 0.22 + wide * 0.10)
    if edge_density >= 0.12 or (text_presence >= 0.26 and saturation <= 0.48):
        return "built_environment_or_street_like", min(0.60, 0.37 + edge_density * 0.8 + text_presence * 0.08)
    if wide >= 0.36:
        return "environment_or_establishing_like", 0.46
    return "general_environment_like", 0.34


def _refresh_topic_and_search_hints(analysis: dict[str, Any]) -> dict[str, Any]:
    """Migrate cached 1.0.1 metrics to the broader 1.0.2 topic taxonomy."""

    topic = analysis.get("topic", {})
    signals = topic.get("signals", {}) if isinstance(topic, Mapping) else {}
    tone = analysis.get("color_tone", {})
    quality = analysis.get("image_quality", {})
    scale = analysis.get("shot_scale", {}).get("distribution", {})
    motion = analysis.get("camera_motion", {})
    visual_mood = analysis.get("visual_mood", {})
    rhythm = analysis.get("editing_rhythm", {})
    label, confidence = _classify_visual_topic(
        _finite_float(signals.get("frames_with_frontal_face_ratio")),
        _finite_float(signals.get("text_like_frame_ratio")),
        _finite_float(tone.get("luma_entropy")),
        _finite_float(motion.get("motion_strength_median_per_second")),
        scale,
        _finite_float(tone.get("saturation", tone.get("saturation_mean"))),
        _finite_float(tone.get("mean_hue_degrees")),
        _finite_float(quality.get("edge_density")),
    )
    analysis["topic"] = {
        "classification": label,
        "confidence": _round(confidence),
        "method": "non_semantic_color_structure_and_framing_heuristic",
        "signals": signals,
        "limitation": "Broad scene-family estimate only; supply the real project theme/product separately.",
    }
    analysis["search_hints"] = _visual_search_hints(
        analysis.get("metadata", {}),
        str(tone.get("classification", "")),
        str(visual_mood.get("classification", "balanced_neutral")),
        scale,
        motion.get("distribution", {}),
        str(rhythm.get("classification", "moderate")),
        label,
    )
    return analysis


def _semantic_positive_terms(semantic: Mapping[str, Any]) -> list[str]:
    """Return compact English phrases suitable for Pixabay query expansion."""

    terms: list[str] = []
    categories = semantic.get("categories", {}) if isinstance(semantic, Mapping) else {}
    if isinstance(categories, Mapping):
        for category in ("subject", "scene", "action", "composition", "emotion"):
            payload = categories.get(category, {})
            if not isinstance(payload, Mapping):
                continue
            label = str(payload.get("label", "")).strip().lower()
            confidence = _finite_float(payload.get("confidence"))
            if label and confidence >= 0.08 and label not in terms:
                terms.append(label)
    for keyword in semantic.get("search_keywords", []) if isinstance(semantic, Mapping) else []:
        value = str(keyword).strip().lower()
        if value and value not in terms:
            terms.append(value)
    return terms[:16]


def _build_shot_semantics(
    frames: Sequence[Mapping[str, Any]],
    decoded_times: Sequence[float],
    transitions: Sequence[Mapping[str, Any]],
    effective_duration: float,
    semantic_records: Sequence[Mapping[str, Any]],
    subject_regions: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Aggregate sampled frame evidence into detected-shot records."""

    cut_indices = {
        index + 1
        for index, transition in enumerate(transitions)
        if bool(transition.get("is_cut"))
    }
    ranges: list[tuple[int, int]] = []
    start = 0
    for index in range(1, len(frames)):
        if index in cut_indices:
            ranges.append((start, index))
            start = index
    ranges.append((start, len(frames)))

    shots: list[dict[str, Any]] = []
    for shot_index, (first, after_last) in enumerate(ranges):
        if after_last <= first:
            continue
        next_time = (
            float(decoded_times[after_last])
            if after_last < len(decoded_times)
            else float(effective_duration)
        )
        start_time = 0.0 if shot_index == 0 else float(decoded_times[first])
        end_time = max(start_time + 0.08, next_time)
        local_frames = frames[first:after_last]
        local_semantics = semantic_records[first:after_last]
        local_regions = subject_regions[first:after_last]
        local_transitions = transitions[max(0, first) : max(0, after_last - 1)]
        semantic = aggregate_semantics(local_semantics)
        subject = aggregate_subject_regions(local_regions)
        scale_distribution = _distribution(str(item.get("scale_label", "unresolved")) for item in local_frames)
        motion_distribution = _distribution(
            str(item.get("motion_label", "unresolved"))
            for item in local_transitions
            if not bool(item.get("is_cut"))
        )
        categories = semantic.get("categories", {}) if isinstance(semantic, Mapping) else {}

        def semantic_label(name: str, fallback: str) -> str:
            value = categories.get(name, {}) if isinstance(categories, Mapping) else {}
            return str(value.get("label", fallback)) if isinstance(value, Mapping) else fallback

        shots.append(
            {
                "index": shot_index,
                "start_seconds": _round(start_time, 3),
                "end_seconds": _round(min(effective_duration, end_time), 3),
                "duration_seconds": _round(max(0.08, min(effective_duration, end_time) - start_time), 3),
                "sample_count": len(local_frames),
                "subject": semantic_label("subject", "unresolved"),
                "scene": semantic_label("scene", "unresolved"),
                "apparent_action": semantic_label("action", "unresolved"),
                "shot_scale": _dominant(scale_distribution, "unresolved"),
                "composition": semantic_label("composition", "unresolved"),
                "camera_motion": _dominant(motion_distribution, "static_like"),
                "visual_mood": semantic_label("emotion", "unresolved"),
                "scene_category": infer_scene_category("", semantic),
                "search_keywords": _semantic_positive_terms(semantic),
                "semantic": semantic,
                "subject_region": subject,
            }
        )
    return shots


def _shot_signature(shot: Mapping[str, Any]) -> tuple[str, ...]:
    """Return a deliberately coarse signature for repetition estimation."""

    return tuple(
        str(shot.get(key, "unresolved")).strip().lower()
        for key in ("subject", "scene", "shot_scale", "composition", "camera_motion")
    )


def _enrich_v12_analysis(analysis: dict[str, Any]) -> dict[str, Any]:
    """Derive the v1.2 style-learning contract from cached shot evidence.

    This function is intentionally deterministic and only consumes fields that
    v1.1 already cached.  It can therefore migrate unchanged reference videos
    without re-running CLIP/OpenCV, while new/modified videos still take the
    complete analysis path.
    """

    analysis["schema_version"] = PROFILE_SCHEMA_VERSION
    raw_shots = analysis.get("shots", [])
    shots = [shot for shot in raw_shots if isinstance(shot, Mapping)]
    duration = max(
        0.0,
        _finite_float(analysis.get("metadata", {}).get("duration_seconds")),
        max((_finite_float(shot.get("end_seconds")) for shot in shots), default=0.0),
    )
    durations = [
        max(0.0, _finite_float(shot.get("duration_seconds")))
        for shot in shots
        if _finite_float(shot.get("duration_seconds")) > 0
    ]
    duration_distribution = {
        "count": len(durations),
        "minimum": _round(min(durations) if durations else 0.0, 3),
        "p10": _round(_percentile(durations, 0.10), 3),
        "p25": _round(_percentile(durations, 0.25), 3),
        "median": _round(_median(durations), 3),
        "mean": _round(_mean(durations), 3),
        "p75": _round(_percentile(durations, 0.75), 3),
        "p90": _round(_percentile(durations, 0.90), 3),
        "maximum": _round(max(durations) if durations else 0.0, 3),
        "buckets": {
            "under_1_5s": sum(value < 1.5 for value in durations),
            "1_5_to_4_5s": sum(1.5 <= value < 4.5 for value in durations),
            "4_5s_and_over": sum(value >= 4.5 for value in durations),
        },
    }

    motion_weights = {
        "static_like": 0.05,
        "drift_or_subject_motion": 0.35,
        "pan_left_like": 0.55,
        "pan_right_like": 0.55,
        "tilt_up_like": 0.60,
        "tilt_down_like": 0.60,
        "zoom_in_like": 0.72,
        "zoom_out_like": 0.68,
        "handheld_or_complex": 0.78,
        "complex_or_subject_motion": 0.75,
    }
    direction_distribution = _distribution(
        str(shot.get("camera_motion", "unresolved")) for shot in shots
    )
    pace: list[dict[str, Any]] = []
    phase_names = ("opening", "early_middle", "late_middle", "ending")
    if duration > 0:
        for index, phase_name in enumerate(phase_names):
            start = duration * index / len(phase_names)
            end = duration * (index + 1) / len(phase_names)
            local = [
                shot
                for shot in shots
                if start <= (
                    _finite_float(shot.get("start_seconds"))
                    + _finite_float(shot.get("end_seconds"))
                ) / 2.0 < end + (1e-6 if index == len(phase_names) - 1 else 0.0)
            ]
            local_durations = [
                _finite_float(shot.get("duration_seconds"))
                for shot in local
                if _finite_float(shot.get("duration_seconds")) > 0
            ]
            local_motion = _mean(
                motion_weights.get(str(shot.get("camera_motion", "")), 0.30)
                for shot in local
            )
            pace.append(
                {
                    "phase": phase_name,
                    "start_seconds": _round(start, 3),
                    "end_seconds": _round(end, 3),
                    "shot_count": len(local),
                    "cuts_per_minute": _round(max(0, len(local) - 1) / max((end - start) / 60.0, 1e-6), 2),
                    "median_shot_duration_seconds": _round(_median(local_durations), 3),
                    "visual_motion_intensity": _round(local_motion),
                }
            )

    adjacency_keys = ("subject", "scene", "shot_scale", "composition", "camera_motion")
    adjacency_change: dict[str, float] = {}
    pair_count = max(0, len(shots) - 1)
    for key in adjacency_keys:
        changes = sum(
            str(shots[index - 1].get(key, "")) != str(shots[index].get(key, ""))
            for index in range(1, len(shots))
        )
        adjacency_change[key] = _round(changes / pair_count if pair_count else 0.0)

    signatures = [_shot_signature(shot) for shot in shots]
    repeated_signature_count = len(signatures) - len(set(signatures))
    key_candidates: list[tuple[float, int, Mapping[str, Any]]] = []
    for index, shot in enumerate(shots):
        categories = shot.get("semantic", {}).get("categories", {}) if isinstance(shot.get("semantic"), Mapping) else {}
        semantic_confidence = max(
            (
                _finite_float(payload.get("confidence"))
                for payload in categories.values()
                if isinstance(payload, Mapping)
            ),
            default=0.0,
        )
        score = motion_weights.get(str(shot.get("camera_motion", "")), 0.30) * 0.55 + semantic_confidence * 0.45
        key_candidates.append((score, index, shot))
    key_shots = []
    for score, index, shot in sorted(key_candidates, key=lambda item: (-item[0], item[1]))[:5]:
        midpoint = (
            _finite_float(shot.get("start_seconds")) + _finite_float(shot.get("end_seconds"))
        ) / 2.0
        key_shots.append(
            {
                "shot_index": int(shot.get("index", index)),
                "time_seconds": _round(midpoint, 3),
                "normalized_position": _round(midpoint / duration if duration else 0.0),
                "score": _round(score),
                "subject": shot.get("subject"),
                "scene": shot.get("scene"),
                "shot_scale": shot.get("shot_scale"),
                "camera_motion": shot.get("camera_motion"),
            }
        )

    rhythm = analysis.setdefault("editing_rhythm", {})
    if not isinstance(rhythm, dict):
        rhythm = {}
        analysis["editing_rhythm"] = rhythm
    transition_distribution = rhythm.get("transition_type_distribution")
    if not isinstance(transition_distribution, Mapping):
        detected_cuts = int(_finite_float(rhythm.get("detected_cuts")))
        transition_distribution = {"hard_cut": detected_cuts} if detected_cuts else {"continuous": 1}
        rhythm["transition_type_distribution"] = transition_distribution
        rhythm["transition_method"] = "migrated_cut_records_only"
    rhythm["shot_duration_distribution_seconds"] = duration_distribution
    rhythm["pace_over_time"] = pace
    rhythm["fastest_phase"] = max(pace, key=lambda item: item["cuts_per_minute"], default={}).get("phase", "unresolved")
    rhythm["key_shot_positions"] = key_shots

    motion = analysis.setdefault("camera_motion", {})
    if isinstance(motion, dict):
        motion["direction_distribution"] = direction_distribution
        motion["adjacent_direction_change_ratio"] = adjacency_change.get("camera_motion", 0.0)

    analysis["editing_style_learning"] = {
        "shot_duration_distribution_seconds": duration_distribution,
        "pace_over_time": pace,
        "transition_type_distribution": dict(transition_distribution),
        "motion_direction_distribution": direction_distribution,
        "adjacent_change_ratios": adjacency_change,
        "key_shot_positions": key_shots,
        "structural_reuse_estimate": {
            "repeated_signature_count": repeated_signature_count,
            "repeated_signature_ratio": _round(repeated_signature_count / len(signatures) if signatures else 0.0),
            "method": "coarse_subject_scene_scale_composition_motion_signature",
            "limitation": "This estimates structural repetition; it does not identify original source-file reuse.",
        },
        "implemented_scope": [
            "shot-duration distribution",
            "four-phase pacing change",
            "conservative hard-cut/fade-like/dissolve-like hints",
            "motion-direction and adjacent visual variation",
            "appearance-based key-shot positions",
        ],
        "unsupported_effects": [
            "effect-specific template reconstruction",
            "mask/compositing reverse engineering",
            "speed-ramp curve reconstruction",
            "source-media reuse identification from the finished reference alone",
        ],
    }
    return analysis


def _analyze_one(
    path: Path,
    relative_path: str,
    metadata: Mapping[str, Any],
    cv2: Any,
    np: Any,
    semantic_analyzer: ClipSemanticAnalyzer | None = None,
    semantic_status: SemanticBackendStatus | None = None,
    semantic_required: bool = False,
) -> dict[str, Any]:
    capture = cv2.VideoCapture(os.fspath(path))
    if not capture.isOpened():
        capture.release()
        raise ReferenceAnalysisError("OpenCV could not open the video")
    face_cascade = _load_face_cascade(cv2)
    duration = _finite_float(metadata.get("duration_seconds"))
    fps = _finite_float(metadata.get("fps")) or _finite_float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(_finite_float(metadata.get("estimated_frame_count"))) or int(
        _finite_float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    )
    targets = _sample_times(duration, fps, frame_count)
    if not targets and fps > 0 and frame_count > 0:
        targets = _sample_times(frame_count / fps, fps, frame_count)
    if not targets:
        capture.release()
        raise ReferenceAnalysisError("duration/frame count unavailable; cannot choose sample frames")

    frames: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []
    decoded_times: list[float] = []
    try:
        for target in targets:
            capture.set(cv2.CAP_PROP_POS_MSEC, target * 1000.0)
            success, frame = capture.read()
            if not success or frame is None:
                continue
            actual_msec = _finite_float(capture.get(cv2.CAP_PROP_POS_MSEC), target * 1000.0)
            actual_time = actual_msec / 1000.0 if actual_msec > 0 else target
            metrics = _frame_metrics(frame, face_cascade, cv2, np)
            if frames:
                elapsed = max(0.08, actual_time - decoded_times[-1])
                transition = _transition_metrics(frames[-1], metrics, elapsed, cv2, np)
                transition["time_seconds"] = actual_time
                transitions.append(transition)
            frames.append(metrics)
            decoded_times.append(actual_time)
    finally:
        capture.release()
    if len(frames) < 3:
        raise ReferenceAnalysisError(f"only {len(frames)} sample frame(s) decoded")

    semantic_records: list[dict[str, Any]] = []
    semantic_runtime_error: str | None = None
    if semantic_analyzer is not None:
        try:
            if len(frames) <= MAX_SEMANTIC_SAMPLES:
                semantic_indices = list(range(len(frames)))
            else:
                semantic_indices = sorted(
                    {
                        round(index * (len(frames) - 1) / (MAX_SEMANTIC_SAMPLES - 1))
                        for index in range(MAX_SEMANTIC_SAMPLES)
                    }
                )
            sampled_semantics = semantic_analyzer.classify_frames(
                [frames[index]["frame"] for index in semantic_indices]
            )
            semantic_records = [
                sampled_semantics[
                    min(
                        range(len(semantic_indices)),
                        key=lambda candidate: abs(semantic_indices[candidate] - frame_index),
                    )
                ]
                for frame_index in range(len(frames))
            ]
        except Exception as exc:
            semantic_runtime_error = f"{type(exc).__name__}: {exc}"
            semantic_records = []
            if semantic_required:
                raise ReferenceAnalysisError(
                    f"pretrained semantic inference failed: {semantic_runtime_error}"
                ) from exc
    subject_regions = [subject_region(frame["frame"], face_cascade) for frame in frames]
    semantic_summary = aggregate_semantics(semantic_records)
    if semantic_analyzer is None or semantic_runtime_error:
        semantic_summary.update(
            {
                "available": False,
                "backend": (semantic_status.backend if semantic_status else "unavailable"),
                "model_id": (semantic_status.model_id if semantic_status else None),
                "error": semantic_runtime_error
                or (semantic_status.error if semantic_status else "semantic backend not configured"),
            }
        )
    else:
        semantic_summary.update(
            {
                "model_id": semantic_status.model_id if semantic_status else semantic_analyzer.model_id,
                "backend": semantic_status.backend if semantic_status else semantic_summary.get("backend"),
                "sampled_frame_count": len(sampled_semantics),
                "propagated_to_structural_samples": len(semantic_records),
            }
        )
    subject_summary = aggregate_subject_regions(subject_regions)

    brightness = _mean(frame["brightness"] for frame in frames)
    saturation = _mean(frame["saturation"] for frame in frames)
    contrast = _mean(frame["contrast"] for frame in frames)
    warmth = _mean(frame["warmth_index"] for frame in frames)
    sharpness_raw = _median([frame["laplacian_variance"] for frame in frames])
    sharpness_score = _median([frame["sharpness_score"] for frame in frames])
    exposure_quality = _mean(frame["exposure_quality"] for frame in frames)
    underexposure = _mean(frame["underexposed_fraction"] for frame in frames)
    overexposure = _mean(frame["overexposed_fraction"] for frame in frames)
    edge_density = _mean(frame["edge_density"] for frame in frames)
    entropy = _mean(frame["luma_entropy"] for frame in frames)
    hue = _mean(frame["mean_hue_degrees"] for frame in frames)

    cuts = [transition for transition in transitions if transition["is_cut"]]
    cut_times = [_finite_float(transition["time_seconds"]) for transition in cuts]
    transition_type_distribution = _distribution(
        str(transition.get("transition_type", "hard_cut" if transition.get("is_cut") else "continuous"))
        for transition in transitions
    )
    effective_duration = duration or (decoded_times[-1] if decoded_times else 0.0)
    boundaries = [0.0] + cut_times + ([effective_duration] if effective_duration > 0 else [])
    shot_durations = [
        boundaries[index + 1] - boundaries[index]
        for index in range(len(boundaries) - 1)
        if boundaries[index + 1] - boundaries[index] >= 0.12
    ]
    cuts_per_minute = len(cuts) / (effective_duration / 60.0) if effective_duration > 0 else 0.0
    median_shot_duration = _median(shot_durations, effective_duration)
    if cuts_per_minute >= 30 or (median_shot_duration and median_shot_duration < 1.8):
        rhythm_label = "rapid"
    elif cuts_per_minute >= 12 or (median_shot_duration and median_shot_duration < 4.5):
        rhythm_label = "moderate"
    else:
        rhythm_label = "slow"

    non_cut_transitions = [transition for transition in transitions if not transition["is_cut"]]
    motion_distribution = _distribution(
        str(transition["motion_label"]) for transition in non_cut_transitions
    )
    motion_strength = _median([transition["motion_strength"] for transition in non_cut_transitions])
    scale_distribution = _distribution(str(frame["scale_label"]) for frame in frames)
    face_frame_ratio = sum(1 for frame in frames if frame["face_count"] > 0) / len(frames)
    face_size_median = _median([frame["largest_face_ratio"] for frame in frames if frame["largest_face_ratio"] > 0])

    text_frame_scores: list[float] = []
    text_region_totals: defaultdict[str, float] = defaultdict(float)
    for frame in frames:
        regions = frame["text_regions"]
        score = max(regions.values(), default=0.0)
        text_frame_scores.append(score)
        for region, value in regions.items():
            text_region_totals[region] += _finite_float(value)
    text_presence = sum(score >= 0.22 for score in text_frame_scores) / len(text_frame_scores)
    text_regions = {
        region: _round(value / len(frames))
        for region, value in sorted(text_region_totals.items(), key=lambda item: (-item[1], item[0]))
    }
    dominant_text_region = _dominant(text_regions, "none_detected") if text_presence >= 0.08 else "none_detected"

    topic_label, topic_confidence = _classify_visual_topic(
        face_frame_ratio,
        text_presence,
        entropy,
        motion_strength,
        scale_distribution,
        saturation,
        hue,
        edge_density,
    )
    if semantic_summary.get("available"):
        topic_label = infer_scene_category("", semantic_summary)
        topic_confidence = max(
            _finite_float(
                semantic_summary.get("categories", {}).get("scene", {}).get("confidence")
            ),
            _finite_float(
                semantic_summary.get("categories", {}).get("subject", {}).get("confidence")
            ),
        )

    shot_type_labels: list[str] = []
    for frame in frames:
        if frame["face_count"] > 0:
            shot_type_labels.append("people_framing_like")
        elif frame["scale_label"] == "close_up_like":
            shot_type_labels.append("detail_insert_like")
        elif frame["scale_label"] == "wide_like":
            shot_type_labels.append("establishing_like")
        else:
            shot_type_labels.append("general_coverage_like")
    shot_type_distribution = _distribution(shot_type_labels)

    tone_label = _tone_label(brightness, saturation, contrast, warmth)
    mood_label = _mood_label(brightness, saturation, contrast, warmth, motion_strength, cuts_per_minute)
    descriptors = [mood_label.replace("_", " "), tone_label.replace("_", " ")]
    descriptors.append(f"{rhythm_label} editing")
    descriptors.append(_dominant(scale_distribution).replace("_", " "))
    descriptors.append(_dominant(motion_distribution, "static_like").replace("_", " "))
    if text_presence >= 0.35:
        descriptors.append(f"persistent text-like overlays near {dominant_text_region.replace('_', ' ')}")

    search_hints = _visual_search_hints(
        metadata,
        tone_label,
        mood_label,
        scale_distribution,
        motion_distribution,
        rhythm_label,
        topic_label,
    )
    semantic_terms = _semantic_positive_terms(semantic_summary)
    existing_terms = list(search_hints.get("positive_terms", []))
    search_hints["positive_terms"] = list(dict.fromkeys(semantic_terms + existing_terms))[:24]
    search_hints["semantic_terms"] = semantic_terms
    median_interval = _median(
        [decoded_times[index] - decoded_times[index - 1] for index in range(1, len(decoded_times))]
    )
    warnings: list[str] = []
    if len(frames) < len(targets) * 0.8:
        warnings.append("Some requested sample frames could not be decoded.")
    if median_interval > 0.85:
        warnings.append("Sampling interval is coarse; very fast cuts may be under-counted.")
    if face_cascade is None:
        warnings.append("OpenCV face cascade unavailable; shot scale uses edge structure only.")
    if not semantic_summary.get("available"):
        warnings.append(
            "Pretrained visual semantics were unavailable; semantic fields are explicitly degraded."
        )

    shots = _build_shot_semantics(
        frames,
        decoded_times,
        transitions,
        effective_duration,
        semantic_records,
        subject_regions,
    )

    analysis = {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "relative_path": relative_path,
        "metadata": dict(metadata),
        "sampling": {
            "method": "uniform_time_seek",
            "requested_frames": len(targets),
            "decoded_frames": len(frames),
            "target_sample_fps": TARGET_SAMPLE_FPS,
            "median_interval_seconds": _round(median_interval, 3),
            "coverage_start_seconds": _round(decoded_times[0], 3),
            "coverage_end_seconds": _round(decoded_times[-1], 3),
        },
        "topic": {
            "classification": topic_label,
            "confidence": _round(topic_confidence),
            "method": (
                "pretrained_clip_zero_shot"
                if semantic_summary.get("available")
                else "non_semantic_color_structure_and_framing_heuristic"
            ),
            "signals": {
                "frames_with_frontal_face_ratio": _round(face_frame_ratio),
                "text_like_frame_ratio": _round(text_presence),
                "dominant_structural_scale": _dominant(scale_distribution),
            },
            "limitation": (
                "Finite zero-shot taxonomy; supply the real project theme/product separately."
                if semantic_summary.get("available")
                else "Broad scene-family estimate only; supply the real project theme/product separately."
            ),
        },
        "semantic_analysis": semantic_summary,
        "subject_profile": subject_summary,
        "shots": shots,
        "visual_mood": {
            "classification": mood_label,
            "method": "brightness_saturation_contrast_temperature_motion_pacing_heuristic",
        },
        "color_tone": {
            "classification": tone_label,
            "brightness": _round(brightness),
            "saturation": _round(saturation),
            "contrast": _round(contrast),
            "brightness_mean": _round(brightness),
            "saturation_mean": _round(saturation),
            "contrast_mean": _round(contrast),
            "warmth_index": _round(warmth),
            "temperature_tendency": "warm" if warmth >= 0.035 else ("cool" if warmth <= -0.035 else "neutral"),
            "mean_hue_degrees": _round(hue, 2),
            "luma_entropy": _round(entropy),
        },
        "image_quality": {
            "sharpness_laplacian_median": _round(sharpness_raw, 2),
            "sharpness_score": _round(sharpness_score),
            "exposure_quality_score": _round(exposure_quality),
            "underexposed_pixel_fraction": _round(underexposure),
            "overexposed_pixel_fraction": _round(overexposure),
            "edge_density": _round(edge_density),
        },
        "shot_types": {
            "dominant": _dominant(shot_type_distribution),
            "distribution": shot_type_distribution,
            "method": "face_presence_and_edge_scale_heuristic",
        },
        "shot_scale": {
            "dominant": _dominant(scale_distribution),
            "distribution": scale_distribution,
            "face_size_median_frame_ratio": _round(face_size_median),
            "method": "frontal_face_size_when_available_else_edge_footprint",
        },
        "camera_motion": {
            "dominant": _dominant(motion_distribution, "unresolved"),
            "distribution": motion_distribution,
            "motion_strength_median_per_second": _round(motion_strength),
            "method": "sparse_optical_flow_global_affine_heuristic",
            "limitation": "Object motion can be confused with camera motion.",
        },
        "editing_rhythm": {
            "classification": rhythm_label,
            "detected_cuts": len(cuts),
            "cuts_per_minute": _round(cuts_per_minute, 2),
            "median_shot_duration_seconds": _round(median_shot_duration, 3),
            "p25_shot_duration_seconds": _round(_percentile(shot_durations, 0.25), 3),
            "p75_shot_duration_seconds": _round(_percentile(shot_durations, 0.75), 3),
            "cut_times_seconds": [_round(value, 3) for value in cut_times],
            "transition_type_distribution": transition_type_distribution,
            "transition_method": "sampled_brightness_histogram_change_heuristic",
            "method": "sampled_histogram_and_pixel_change_detection",
        },
        "subtitle_layout": {
            "possible_text_overlay_frame_ratio": _round(text_presence),
            "dominant_region": dominant_text_region,
            "region_scores": text_regions,
            "method": "text_like_horizontal_edge_grouping_without_ocr",
            "limitation": "Signs, logos, product labels and UI can be mistaken for subtitles.",
        },
        "overall_style": {
            "label": " / ".join(descriptors[:4]),
            "descriptors": descriptors,
        },
        "search_hints": search_hints,
        "warnings": warnings,
    }
    return _enrich_v12_analysis(analysis)


def _video_summary(analysis: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "relative_path": analysis.get("relative_path"),
        "metadata": analysis.get("metadata"),
        "topic": analysis.get("topic"),
        "visual_mood": analysis.get("visual_mood"),
        "color_tone": analysis.get("color_tone"),
        "image_quality": analysis.get("image_quality"),
        "shot_types": analysis.get("shot_types"),
        "shot_scale": analysis.get("shot_scale"),
        "camera_motion": analysis.get("camera_motion"),
        "editing_rhythm": analysis.get("editing_rhythm"),
        "subtitle_layout": analysis.get("subtitle_layout"),
        "overall_style": analysis.get("overall_style"),
        "search_hints": analysis.get("search_hints"),
        "semantic_analysis": analysis.get("semantic_analysis"),
        "subject_profile": analysis.get("subject_profile"),
        "shots": analysis.get("shots", []),
        "editing_style_learning": analysis.get("editing_style_learning", {}),
        "sampling": analysis.get("sampling"),
        "warnings": analysis.get("warnings", []),
    }


def _aggregate_corpus_semantics(analyses: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    available = [
        analysis.get("semantic_analysis", {})
        for analysis in analyses
        if analysis.get("semantic_analysis", {}).get("available")
    ]
    if not available:
        errors = [
            str(analysis.get("semantic_analysis", {}).get("error", ""))
            for analysis in analyses
            if analysis.get("semantic_analysis", {}).get("error")
        ]
        return {
            "available": False,
            "coverage_ratio": 0.0,
            "categories": {},
            "search_keywords": [],
            "errors": list(dict.fromkeys(errors))[:3],
        }
    category_votes: dict[str, defaultdict[str, list[float]]] = {}
    keywords: list[str] = []
    for item in available:
        categories = item.get("categories", {})
        if not isinstance(categories, Mapping):
            continue
        for category, payload in categories.items():
            if not isinstance(payload, Mapping):
                continue
            label = str(payload.get("label", "")).strip()
            if not label:
                continue
            category_votes.setdefault(str(category), defaultdict(list))[label].append(
                _finite_float(payload.get("confidence"))
            )
        for keyword in item.get("search_keywords", []):
            value = str(keyword).strip().lower()
            if value and value not in keywords:
                keywords.append(value)
    aggregated: dict[str, Any] = {}
    for category, labels in category_votes.items():
        ranked = sorted(
            (
                (label, _mean(values), len(values))
                for label, values in labels.items()
            ),
            key=lambda row: (row[2], row[1], row[0]),
            reverse=True,
        )
        aggregated[category] = {
            "label": ranked[0][0],
            "confidence": _round(ranked[0][1]),
            "distribution": {
                label: _round(votes / len(available))
                for label, _, votes in ranked
            },
        }
    return {
        "available": True,
        "backend": "aggregate_pretrained_clip_zero_shot",
        "coverage_ratio": _round(len(available) / max(1, len(analyses))),
        "categories": aggregated,
        "search_keywords": keywords[:20],
    }


def _aggregate(analyses: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not analyses:
        return {
            "topic": {
                "classification": "unresolved",
                "distribution": {},
                "method": "non_semantic_visual_structure_heuristic",
                "limitation": "No successfully analyzed reference videos were available.",
            },
            "visual_mood": {"classification": "unresolved", "distribution": {}},
            "color_tone": {"classification": "unresolved"},
            "shot_types": {"dominant": "unresolved", "distribution": {}},
            "shot_scale": {"dominant": "unresolved", "distribution": {}},
            "camera_motion": {"dominant": "unresolved", "distribution": {}},
            "editing_rhythm": {"classification": "unresolved"},
            "subtitle_layout": {"dominant_region": "none_detected"},
            "overall_style": {"label": "unresolved", "descriptors": []},
            "semantic_profile": {
                "available": False,
                "coverage_ratio": 0.0,
                "categories": {},
                "search_keywords": [],
            },
            "subject_profile": {},
            "search_hints": {
                "language": "en",
                "positive_terms": [],
                "shot_scale_terms": [],
                "camera_terms": [],
                "rhythm_terms": [],
                "avoid_terms": ["watermark", "logo", "burned-in subtitles", "low resolution"],
                "topic_required": True,
            },
        }

    topic_distribution = _distribution(
        str(analysis.get("topic", {}).get("classification", "unresolved")) for analysis in analyses
    )
    corpus_semantics = _aggregate_corpus_semantics(analyses)
    subject_profile = aggregate_subject_regions(
        [
            analysis.get("subject_profile", {})
            for analysis in analyses
            if isinstance(analysis.get("subject_profile"), Mapping)
        ]
    )
    subject_profile["face_frame_ratio"] = _round(
        _mean(
            analysis.get("subject_profile", {}).get("face_frame_ratio", 0.0)
            for analysis in analyses
        )
    )
    mood_distribution = _distribution(
        str(analysis.get("visual_mood", {}).get("classification", "unresolved"))
        for analysis in analyses
    )
    shot_type_distribution = _average_distributions(
        analysis.get("shot_types", {}).get("distribution", {}) for analysis in analyses
    )
    scale_distribution = _average_distributions(
        analysis.get("shot_scale", {}).get("distribution", {}) for analysis in analyses
    )
    motion_distribution = _average_distributions(
        analysis.get("camera_motion", {}).get("distribution", {}) for analysis in analyses
    )
    motion_direction_distribution = _average_distributions(
        analysis.get("camera_motion", {}).get("direction_distribution", {}) for analysis in analyses
    )
    transition_type_distribution = _average_distributions(
        analysis.get("editing_rhythm", {}).get("transition_type_distribution", {}) for analysis in analyses
    )
    all_shot_durations = [
        _finite_float(shot.get("duration_seconds"))
        for analysis in analyses
        for shot in analysis.get("shots", [])
        if isinstance(shot, Mapping) and _finite_float(shot.get("duration_seconds")) > 0
    ]
    corpus_duration_distribution = {
        "count": len(all_shot_durations),
        "minimum": _round(min(all_shot_durations) if all_shot_durations else 0.0, 3),
        "p10": _round(_percentile(all_shot_durations, 0.10), 3),
        "p25": _round(_percentile(all_shot_durations, 0.25), 3),
        "median": _round(_median(all_shot_durations), 3),
        "mean": _round(_mean(all_shot_durations), 3),
        "p75": _round(_percentile(all_shot_durations, 0.75), 3),
        "p90": _round(_percentile(all_shot_durations, 0.90), 3),
        "maximum": _round(max(all_shot_durations) if all_shot_durations else 0.0, 3),
    }
    pace_over_time: list[dict[str, Any]] = []
    for phase in ("opening", "early_middle", "late_middle", "ending"):
        rows = [
            row
            for analysis in analyses
            for row in analysis.get("editing_rhythm", {}).get("pace_over_time", [])
            if isinstance(row, Mapping) and row.get("phase") == phase
        ]
        pace_over_time.append(
            {
                "phase": phase,
                "cuts_per_minute": _round(_mean(_finite_float(row.get("cuts_per_minute")) for row in rows), 2),
                "median_shot_duration_seconds": _round(
                    _median([_finite_float(row.get("median_shot_duration_seconds")) for row in rows]), 3
                ),
                "visual_motion_intensity": _round(
                    _mean(_finite_float(row.get("visual_motion_intensity")) for row in rows)
                ),
            }
        )
    adjacent_change_ratios = {
        key: _round(
            _mean(
                _finite_float(
                    analysis.get("editing_style_learning", {}).get("adjacent_change_ratios", {}).get(key)
                )
                for analysis in analyses
            )
        )
        for key in ("subject", "scene", "shot_scale", "composition", "camera_motion")
    }
    structural_reuse_ratio = _round(
        _mean(
            _finite_float(
                analysis.get("editing_style_learning", {})
                .get("structural_reuse_estimate", {})
                .get("repeated_signature_ratio")
            )
            for analysis in analyses
        )
    )
    brightness = _mean(analysis.get("color_tone", {}).get("brightness_mean", 0.0) for analysis in analyses)
    saturation = _mean(analysis.get("color_tone", {}).get("saturation_mean", 0.0) for analysis in analyses)
    contrast = _mean(analysis.get("color_tone", {}).get("contrast_mean", 0.0) for analysis in analyses)
    warmth = _mean(analysis.get("color_tone", {}).get("warmth_index", 0.0) for analysis in analyses)
    entropy = _mean(analysis.get("color_tone", {}).get("luma_entropy", 0.0) for analysis in analyses)
    sharpness = _median([analysis.get("image_quality", {}).get("sharpness_score", 0.0) for analysis in analyses])
    exposure_quality = _mean(
        analysis.get("image_quality", {}).get("exposure_quality_score", 0.0) for analysis in analyses
    )
    total_duration = sum(_finite_float(analysis.get("metadata", {}).get("duration_seconds")) for analysis in analyses)
    total_cuts = sum(int(_finite_float(analysis.get("editing_rhythm", {}).get("detected_cuts"))) for analysis in analyses)
    cuts_per_minute = total_cuts / (total_duration / 60.0) if total_duration > 0 else 0.0
    median_shot_duration = _median(
        [analysis.get("editing_rhythm", {}).get("median_shot_duration_seconds", 0.0) for analysis in analyses]
    )
    if cuts_per_minute >= 30 or (median_shot_duration and median_shot_duration < 1.8):
        rhythm_label = "rapid"
    elif cuts_per_minute >= 12 or (median_shot_duration and median_shot_duration < 4.5):
        rhythm_label = "moderate"
    else:
        rhythm_label = "slow"

    text_presence = _mean(
        analysis.get("subtitle_layout", {}).get("possible_text_overlay_frame_ratio", 0.0)
        for analysis in analyses
    )
    text_region_maps = [analysis.get("subtitle_layout", {}).get("region_scores", {}) for analysis in analyses]
    text_regions = _average_distributions(text_region_maps)
    dominant_text = _dominant(text_regions, "none_detected") if text_presence >= 0.08 else "none_detected"
    orientation_distribution = _distribution(
        str(analysis.get("metadata", {}).get("orientation", "unknown")) for analysis in analyses
    )
    aggregate_metadata = {"orientation": _dominant(orientation_distribution, "unknown")}
    tone_label = _tone_label(brightness, saturation, contrast, warmth)
    mood_label = _dominant(mood_distribution)
    topic_label = _dominant(topic_distribution)
    scenic_share = _finite_float(topic_distribution.get("natural_or_travel_scenery_like")) + _finite_float(
        topic_distribution.get("water_sky_or_coastal_scenery_like")
    )
    if scenic_share >= 0.45:
        topic_label = "mixed_nature_water_travel_scenery"
    elif topic_distribution and max(topic_distribution.values()) < 0.40:
        topic_label = "mixed_environmental_scenery"
    motion_strength = _median(
        [analysis.get("camera_motion", {}).get("motion_strength_median_per_second", 0.0) for analysis in analyses]
    )
    hints = _visual_search_hints(
        aggregate_metadata,
        tone_label,
        mood_label,
        scale_distribution,
        motion_distribution,
        rhythm_label,
        topic_label,
    )
    semantic_terms = _semantic_positive_terms(corpus_semantics)
    hints["positive_terms"] = list(
        dict.fromkeys(semantic_terms + list(hints.get("positive_terms", [])))
    )[:24]
    hints["semantic_terms"] = semantic_terms

    descriptors = [
        mood_label.replace("_", " "),
        tone_label.replace("_", " "),
        f"{rhythm_label} editing",
        _dominant(scale_distribution).replace("_", " "),
        _dominant(motion_distribution).replace("_", " "),
    ]
    if text_presence >= 0.35:
        descriptors.append(f"persistent text-like overlays near {dominant_text.replace('_', ' ')}")
    return {
        "topic": {
            "classification": topic_label,
            "distribution": topic_distribution,
            "method": "aggregate_of_non_semantic_visual_structure_heuristics",
            "limitation": "This describes visual structure only. Supply the actual theme/product separately for search.",
        },
        "visual_mood": {
            "classification": mood_label,
            "distribution": mood_distribution,
            "method": "aggregate_visual_metric_heuristic",
        },
        "color_tone": {
            "classification": tone_label,
            "brightness": _round(brightness),
            "saturation": _round(saturation),
            "contrast": _round(contrast),
            "brightness_mean": _round(brightness),
            "saturation_mean": _round(saturation),
            "contrast_mean": _round(contrast),
            "warmth_index": _round(warmth),
            "temperature_tendency": "warm" if warmth >= 0.035 else ("cool" if warmth <= -0.035 else "neutral"),
            "luma_entropy_mean": _round(entropy),
        },
        "image_quality": {
            "sharpness_score_median": _round(sharpness),
            "exposure_quality_mean": _round(exposure_quality),
        },
        "shot_types": {
            "dominant": _dominant(shot_type_distribution),
            "distribution": shot_type_distribution,
        },
        "shot_scale": {
            "dominant": _dominant(scale_distribution),
            "distribution": scale_distribution,
        },
        "camera_motion": {
            "dominant": _dominant(motion_distribution),
            "distribution": motion_distribution,
            "direction_distribution": motion_direction_distribution,
            "motion_strength_median_per_second": _round(motion_strength),
        },
        "editing_rhythm": {
            "classification": rhythm_label,
            "detected_cuts": total_cuts,
            "cuts_per_minute": _round(cuts_per_minute, 2),
            "median_shot_duration": _round(median_shot_duration, 3),
            "median_shot_duration_seconds": _round(median_shot_duration, 3),
            "average_shot_duration_seconds": _round(median_shot_duration, 3),
            "median_per_video_shot_duration_seconds": _round(median_shot_duration, 3),
            "shot_duration_distribution_seconds": corpus_duration_distribution,
            "pace_over_time": pace_over_time,
            "fastest_phase": max(pace_over_time, key=lambda item: item["cuts_per_minute"], default={}).get("phase", "unresolved"),
            "transition_type_distribution": transition_type_distribution,
        },
        "subtitle_layout": {
            "possible_text_overlay_frame_ratio": _round(text_presence),
            "dominant_region": dominant_text,
            "region_distribution": text_regions,
            "method": "aggregate_text_like_edge_regions_without_ocr",
        },
        "overall_style": {
            "label": " / ".join(descriptors[:4]),
            "descriptors": descriptors,
            "orientation_distribution": orientation_distribution,
        },
        "semantic_profile": corpus_semantics,
        "subject_profile": subject_profile,
        "editing_style_learning": {
            "shot_duration_distribution_seconds": corpus_duration_distribution,
            "pace_over_time": pace_over_time,
            "transition_type_distribution": transition_type_distribution,
            "motion_direction_distribution": motion_direction_distribution,
            "adjacent_change_ratios": adjacent_change_ratios,
            "structural_reuse_estimate": {
                "repeated_signature_ratio": structural_reuse_ratio,
                "method": "mean coarse per-video visual-structure signature repetition",
                "limitation": "This is not source-file reuse identification.",
            },
            "unsupported_effects": [
                "effect-specific template reconstruction",
                "mask/compositing reverse engineering",
                "speed-ramp curve reconstruction",
            ],
        },
        "search_hints": hints,
    }


def analyze_references(
    reference_dir: str | os.PathLike[str],
    cache_dir: str | os.PathLike[str],
    output_path: str | os.PathLike[str] | None = None,
    *,
    semantic_required: bool = False,
    enable_semantics: bool = True,
    semantic_cache_dir: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Analyze reference videos and return/write an aggregate style profile.

    Args:
        reference_dir: Recursively scanned, read-only source directory.
        cache_dir: Writable directory for ``reference_analysis_cache.json`` and,
            by default, ``style_profile.json``.
        output_path: Optional explicit style-profile JSON path.
        semantic_required: Fail instead of degrading when the pretrained model
            cannot be loaded.
        enable_semantics: Disable model loading explicitly (primarily for
            diagnostics and the documented fallback mode).
        semantic_cache_dir: Optional Hugging Face model-cache directory.

    Unchanged successful entries are reused when analyzer version, size,
    nanosecond mtime and the sampled/full SHA-256 fingerprint all match.
    """

    reference_root = Path(reference_dir).expanduser().resolve(strict=True)
    if not reference_root.is_dir():
        raise NotADirectoryError(f"Reference path is not a directory: {reference_root}")
    cache_root = Path(cache_dir).expanduser().resolve(strict=False)
    profile_path = (
        Path(output_path).expanduser().resolve(strict=False)
        if output_path is not None
        else cache_root / "style_profile.json"
    )
    cache_path = cache_root / "reference_analysis_cache.json"

    # Prevent accidental source-directory writes even if a caller passes a bad
    # cache/output path.  Path.is_relative_to is available in supported Python.
    if cache_root == reference_root or cache_root.is_relative_to(reference_root):
        raise ReferenceAnalysisError("cache_dir must be outside the read-only reference directory")
    if profile_path == reference_root or profile_path.is_relative_to(reference_root):
        raise ReferenceAnalysisError("output_path must be outside the read-only reference directory")

    cache_root.mkdir(parents=True, exist_ok=True)

    cv2, np, ffprobe_executable = _require_dependencies()
    if semantic_required and not enable_semantics:
        raise ReferenceAnalysisError(
            "semantic_required and enable_semantics=False are mutually exclusive"
        )
    if semantic_cache_dir is None:
        project = discover_project_root(reference_root.parent)
        semantic_cache_dir = RuntimePaths.build(project_root=project).cache_root / "models"
    if enable_semantics:
        semantic_analyzer, semantic_status = load_semantic_analyzer(
            cache_dir=semantic_cache_dir,
            required=semantic_required,
        )
    else:
        semantic_analyzer = None
        semantic_status = SemanticBackendStatus(
            available=False,
            model_id="disabled",
            backend="disabled",
            error="disabled by caller",
        )
    videos = _discover_videos(reference_root)
    old_cache, cache_warning = _load_json(cache_path)
    old_analyzer_version = old_cache.get("analyzer_version")
    old_semantic_backend = old_cache.get("semantic_backend", {})
    semantic_cache_compatible = (
        isinstance(old_semantic_backend, Mapping)
        and bool(old_semantic_backend.get("available")) == bool(semantic_status.available)
        and str(old_semantic_backend.get("model_id", "")) == str(semantic_status.model_id)
    )
    migration_mode = old_analyzer_version in MIGRATABLE_ANALYZER_VERSIONS
    cache_compatible = (
        old_cache.get("cache_schema_version") == CACHE_SCHEMA_VERSION
        and (old_analyzer_version == ANALYZER_VERSION or migration_mode)
        and old_cache.get("reference_directory") == os.fspath(reference_root)
        and semantic_cache_compatible
    )
    old_entries = old_cache.get("entries", {}) if cache_compatible else {}
    if not isinstance(old_entries, dict):
        old_entries = {}
        cache_warning = "cache entries were invalid and were ignored"

    report: dict[str, Any] = {
        "discovered": len(videos),
        "analyzed": 0,
        "reused": 0,
        "failed": 0,
        "removed_from_cache": 0,
        "migrated": 0,
        "changed": [],
        "reused_paths": [],
        "errors": [],
    }
    if cache_warning:
        report["cache_warning"] = cache_warning
    elif migration_mode and cache_compatible:
        report["cache_migration"] = (
            f"{old_analyzer_version} -> {ANALYZER_VERSION}; reused shot/semantic signal metrics "
            "and derived v1.2 pacing, transition, motion-direction, key-shot, and diversity fields"
        )
    elif old_cache and not cache_compatible:
        report["cache_warning"] = "cache schema, analyzer version, or source root changed; full reanalysis"

    entries: dict[str, Any] = {}
    analyses: list[dict[str, Any]] = []
    for path in videos:
        relative_path = path.relative_to(reference_root).as_posix()
        try:
            fingerprint = _fingerprint(path)
        except (OSError, PermissionError) as exc:
            report["failed"] += 1
            report["errors"].append({"relative_path": relative_path, "stage": "fingerprint", "error": str(exc)})
            continue

        old_entry = old_entries.get(relative_path)
        if (
            isinstance(old_entry, dict)
            and old_entry.get("status") == "ok"
            and old_entry.get("fingerprint") == fingerprint
            and isinstance(old_entry.get("analysis"), dict)
            and (
                not semantic_required
                or bool(old_entry.get("analysis", {}).get("semantic_analysis", {}).get("available"))
            )
        ):
            analysis = copy.deepcopy(old_entry["analysis"])
            entry = copy.deepcopy(old_entry)
            if migration_mode:
                analysis = _enrich_v12_analysis(analysis)
                entry["analysis"] = analysis
                entry["migrated_at"] = _utc_now()
                report["migrated"] += 1
            entries[relative_path] = entry
            analyses.append(analysis)
            report["reused"] += 1
            report["reused_paths"].append(relative_path)
            continue

        report["analyzed"] += 1
        report["changed"].append(relative_path)
        try:
            metadata = _ffprobe(path, ffprobe_executable)
            analysis = _analyze_one(
                path,
                relative_path,
                metadata,
                cv2,
                np,
                semantic_analyzer,
                semantic_status,
                semantic_required,
            )
            entry = {
                "status": "ok",
                "fingerprint": fingerprint,
                "analyzed_at": _utc_now(),
                "analysis": analysis,
            }
            entries[relative_path] = entry
            analyses.append(analysis)
        except Exception as exc:  # Keep a large corpus useful if one source is corrupt.
            report["failed"] += 1
            error_record = {
                "relative_path": relative_path,
                "stage": "analysis",
                "error": f"{type(exc).__name__}: {exc}",
            }
            report["errors"].append(error_record)
            entries[relative_path] = {
                "status": "error",
                "fingerprint": fingerprint,
                "analyzed_at": _utc_now(),
                "error": error_record["error"],
            }

    current_paths = set(entries)
    report["removed_from_cache"] = len(set(old_entries) - current_paths)
    cache_payload = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "analyzer_version": ANALYZER_VERSION,
        "semantic_backend": {
            "available": semantic_status.available,
            "model_id": semantic_status.model_id,
            "backend": semantic_status.backend,
        },
        "reference_directory": os.fspath(reference_root),
        "updated_at": _utc_now(),
        "entries": entries,
    }
    _atomic_json_write(cache_path, cache_payload)

    style = _aggregate(analyses)
    semantic_used = any(
        bool(item.get("semantic_analysis", {}).get("available"))
        for item in analyses
    )
    profile: dict[str, Any] = {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "analyzer": {
            "name": "bgm-montage reference analyzer",
            "version": ANALYZER_VERSION,
            "semantic_model_used": semantic_used,
            "semantic_backend": {
                "available": semantic_status.available,
                "model_id": semantic_status.model_id,
                "backend": semantic_status.backend,
                "error": semantic_status.error if not semantic_status.available else None,
            },
            "methods": [
                "ffprobe metadata",
                "uniform OpenCV frame sampling",
                "color and exposure statistics",
                "frontal-face and edge-footprint framing heuristics",
                "sparse optical-flow motion heuristic",
                "histogram cut detection",
                "text-like edge-region heuristic without OCR",
                "pretrained CLIP zero-shot visual semantics"
                if semantic_used
                else "explicit semantic fallback (no model output)",
                "spectral-residual saliency and face-aware subject geometry",
            ],
        },
        "reference_directory": os.fspath(reference_root),
        "source_policy": "read_only",
        "cache": {
            "path": os.fspath(cache_path),
            "fingerprint_fields": [
                "size",
                "mtime_ns",
                "content_sha256",
                "hash_algorithm",
                "hash_mode",
            ],
        },
        "run_report": report,
        "corpus": {
            "video_count": len(analyses),
            "total_duration_seconds": _round(
                sum(_finite_float(item.get("metadata", {}).get("duration_seconds")) for item in analyses),
                3,
            ),
        },
        # ``style_profile`` is the stable integration payload consumed by the
        # search/montage stages. ``style`` remains as a concise, intuitive alias.
        "style_profile": style,
        "style": style,
        # Concise summaries remain in the public profile; frame-level working
        # data is not serialized.  Full per-video summaries are in the cache too.
        "videos": [_video_summary(item) for item in analyses],
        "limitations": [
            "CLIP action and emotion labels are appearance-based estimates from sampled frames, not temporal action recognition.",
            "When the semantic backend is unavailable, semantic fields are explicitly marked degraded and only structural heuristics remain.",
            "Sampled analysis can miss cuts or overlays shorter than the sampling interval.",
            "Optical-flow camera labels can include subject motion.",
            "Text-like regions may include signs, logos, UI and product labels.",
        ],
    }
    _atomic_json_write(profile_path, profile)
    return profile


def _default_paths() -> tuple[Path, Path]:
    paths = RuntimePaths.build(project_root=discover_project_root())
    reference_dir = Path(os.environ.get("BGM_MONTAGE_REFERENCE_DIR", paths.project_root / "参考视频"))
    cache_dir = Path(os.environ.get("BGM_MONTAGE_CACHE_DIR", paths.reference_cache))
    return reference_dir, cache_dir


def _build_parser() -> argparse.ArgumentParser:
    default_reference, default_cache = _default_paths()
    parser = argparse.ArgumentParser(
        description="Read-only cached style analysis of a reference-video directory."
    )
    parser.add_argument(
        "--reference-dir",
        default=os.fspath(default_reference),
        help=f"Directory scanned recursively (default: {default_reference})",
    )
    parser.add_argument(
        "--cache-dir",
        default=os.fspath(default_cache),
        help=f"Writable cache directory (default: {default_cache})",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional style_profile.json path (default: CACHE_DIR/style_profile.json)",
    )
    parser.add_argument(
        "--semantic-required",
        action="store_true",
        help="Fail if the pretrained CLIP backend cannot be loaded.",
    )
    parser.add_argument(
        "--no-semantics",
        action="store_true",
        help="Explicitly run the documented structural-only fallback.",
    )
    parser.add_argument(
        "--semantic-cache-dir",
        default=None,
        help="Optional Hugging Face model cache directory.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        profile = analyze_references(
            args.reference_dir,
            args.cache_dir,
            args.output,
            semantic_required=args.semantic_required,
            enable_semantics=not args.no_semantics,
            semantic_cache_dir=args.semantic_cache_dir,
        )
    except (ReferenceAnalysisError, FileNotFoundError, NotADirectoryError) as exc:
        print(f"reference analysis failed: {exc}", file=sys.stderr)
        return 2
    summary = {
        "style_profile": os.fspath(
            Path(args.output).expanduser().resolve(strict=False)
            if args.output
            else Path(args.cache_dir).expanduser().resolve(strict=False) / "style_profile.json"
        ),
        "video_count": profile["corpus"]["video_count"],
        "run_report": profile["run_report"],
        "overall_style": profile["style"]["overall_style"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if profile["corpus"]["video_count"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
