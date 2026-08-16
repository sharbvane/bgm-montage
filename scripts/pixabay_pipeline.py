#!/usr/bin/env python3
"""Pixabay search, ranking, download, QA, deduplication, and attribution.

The public entry point is :func:`run_pixabay_pipeline`.  The module deliberately
keeps the Pixabay credential out of URLs written to disk, cache keys, manifests,
and console output.  All visual labels and QA findings are signal-derived
heuristics; callers should not present them as model-certified facts.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import http.client
import json
import logging
import math
import os
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlparse

import cv2
import imagehash
import numpy as np
import requests
from dotenv import load_dotenv
from PIL import Image

from runtime_paths import RuntimePaths, migrate_legacy_nested_pixabay_cache
from visual_intelligence import (
    ASSET_ANALYSIS_SCHEMA_VERSION,
    ENGINE_VERSION as VISUAL_ENGINE_VERSION,
    aggregate_video_aesthetics,
    analysis_cache_valid,
    asset_visual_features,
    build_visual_style_profile,
    frame_aesthetic_metrics,
    metadata_profile_fit,
    plan_visual_search_queries,
)
from material_usage_policy import USAGE_MODES, apply_usage_policy, normalize_usage_mode
from visual_semantics import (
    aggregate_subject_regions,
    face_content_risk,
    infer_scene_category,
    load_face_detector,
    subject_region,
)


PIXABAY_VIDEO_API = "https://pixabay.com/api/videos/"
CACHE_TTL_SECONDS = 24 * 60 * 60
SCHEMA_VERSION = 4
ASSET_MANIFEST_SCHEMA_VERSION = 2
REQUEST_TIMEOUT = (10, 45)
DOWNLOAD_TIMEOUT = (15, 180)
USER_AGENT = "bgm-montage/1.3 (Pixabay video workflow)"

_FACE_DETECTOR: Any | None = None


def _face_detector() -> Any | None:
    global _FACE_DETECTOR
    if _FACE_DETECTOR is None:
        _FACE_DETECTOR = load_face_detector()
    return _FACE_DETECTOR


def _scene_category(tags: str, semantic: Mapping[str, Any] | None = None) -> str:
    """Return a scene label while avoiding substring-based ``man`` matches.

    The shared v1.1 taxonomy uses substring checks.  Explicit manufacturing
    terms must be classified as industrial even when no second industrial tag
    is present (``manufacturing`` contains the letters ``man``).
    """

    tokens = set(_english_words(tags)) if "_english_words" in globals() else set()
    if tokens & {"manufacturing", "manufacture", "machinery", "industrial", "factory", "production"}:
        return "industrial"
    return infer_scene_category(tags, semantic)


class PixabayPipelineError(RuntimeError):
    """An expected, credential-safe pipeline failure."""


class InsufficientMaterialError(PixabayPipelineError):
    """Raised only after all bounded query expansions and QA are exhausted."""


def _first_numeric(value: Any, names: set[str]) -> float | None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key).lower() in names and isinstance(item, (int, float)):
                number = float(item)
                if math.isfinite(number) and number > 0:
                    return number
        for item in value.values():
            found = _first_numeric(item, names)
            if found is not None:
                return found
    elif isinstance(value, (list, tuple)):
        for item in value:
            found = _first_numeric(item, names)
            if found is not None:
                return found
    return None


def evaluate_selected_sufficiency(
    selected: Sequence[Mapping[str, Any]],
    desired_count: int,
    target_duration: float | None,
) -> dict[str, Any]:
    """Evaluate hard, machine-readable selection gates before rendering."""

    duration = float(target_duration or 0.0)
    if duration > 0:
        min_unique = max(4, min(int(desired_count), math.ceil(duration / 3.5)))
    else:
        # Backward-compatible standalone calls without a target duration can
        # only enforce the caller's requested count.  The unified entry always
        # supplies duration and therefore gets the full four-asset floor.
        min_unique = max(1, min(int(desired_count), 4))
    identities = {
        str(
            item.get("canonical_source_id")
            or item.get("fingerprint", {}).get("sha256")
            or item.get("pixabay_id")
            or item.get("local_path")
        )
        for item in selected
    }
    def quality_of(item: Mapping[str, Any]) -> Mapping[str, Any]:
        value = item.get("quality")
        return value if isinstance(value, Mapping) else {}

    scenes = {
        str(
            item.get("scene_category")
            or quality_of(item).get("scene_category")
            or _scene_category(str(item.get("tags") or ""), None)
        )
        for item in selected
    }
    face_risks = [
        float(
            item.get("face_content_risk")
            or quality_of(item).get("face_content_risk")
            or face_content_risk(str(item.get("tags") or ""), None, None)
        )
        for item in selected
    ]
    low_face_count = sum(risk < 0.65 for risk in face_risks)
    min_scenes = min(3, min_unique)
    required_low_face = max(1, math.ceil(min_unique * 0.65))
    theoretical_coverage = sum(
        min(float(item.get("duration_seconds") or item.get("duration") or 0.0), duration * 0.30)
        for item in selected
    ) if duration > 0 else 0.0
    failures: list[str] = []
    if len(identities) < min_unique:
        failures.append(f"independent assets {len(identities)} < {min_unique}")
    if len(scenes) < min_scenes:
        failures.append(f"scene categories {len(scenes)} < {min_scenes}")
    if low_face_count < required_low_face:
        failures.append(f"low-face-risk assets {low_face_count} < {required_low_face}")
    if duration > 0 and theoretical_coverage < duration * 0.95:
        failures.append(
            f"theoretical screen coverage {theoretical_coverage:.2f}s < {duration * 0.95:.2f}s"
        )
    return {
        "passed": not failures,
        "failures": failures,
        "independent_asset_count": len(identities),
        "required_independent_assets": min_unique,
        "scene_categories": sorted(scenes),
        "required_scene_categories": min_scenes,
        "low_face_risk_assets": low_face_count,
        "required_low_face_risk_assets": required_low_face,
        "theoretical_screen_coverage_seconds": round(theoretical_coverage, 4),
        "target_duration_seconds": round(duration, 4) if duration > 0 else None,
        "max_reuse_per_asset": 2,
        "max_asset_screen_share": 0.30,
        "max_prominent_face_screen_share": 0.15,
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _safe_error(error: BaseException | str, secret: str | None = None) -> str:
    """Remove credentials and authenticated query strings from an error."""

    message = str(error)
    if secret:
        message = message.replace(secret, "[REDACTED]")
    message = re.sub(r"(?i)([?&](?:key|api_key|token)=)[^&\s]+", r"\1[REDACTED]", message)
    message = re.sub(r"(?i)(PIXABAY_API_KEY\s*[=:]\s*)\S+", r"\1[REDACTED]", message)
    return message[:600]


def _strip_secrets(value: Any, secret: str | None = None) -> Any:
    """Recursively sanitize data before it is persisted."""

    if isinstance(value, Mapping):
        clean: dict[str, Any] = {}
        for key, item in value.items():
            if str(key).lower() in {"key", "api_key", "pixabay_api_key", "token"}:
                continue
            clean[str(key)] = _strip_secrets(item, secret)
        return clean
    if isinstance(value, list):
        return [_strip_secrets(item, secret) for item in value]
    if isinstance(value, tuple):
        return [_strip_secrets(item, secret) for item in value]
    if isinstance(value, str):
        return _safe_error(value, secret)
    return value


def _atomic_write_json(path: Path, payload: Any, secret: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sanitized = _strip_secrets(payload, secret)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(sanitized, handle, ensure_ascii=False, indent=2, sort_keys=False)
            handle.write("\n")
        for attempt in range(8):
            try:
                os.replace(temp_name, path)
                break
            except PermissionError:
                if attempt >= 7:
                    raise
                time.sleep(0.04 * (attempt + 1))
    finally:
        try:
            Path(temp_name).unlink(missing_ok=True)
        except OSError:
            pass


def _process_is_alive(pid: int) -> bool:
    """Return whether ``pid`` is alive without sending it a Windows signal."""

    if pid <= 0:
        return False
    if pid == os.getpid():
        return True
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            process_query_limited_information = 0x1000
            still_active = 259
            kernel32 = ctypes.windll.kernel32
            kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
            kernel32.OpenProcess.restype = wintypes.HANDLE
            kernel32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
            kernel32.GetExitCodeProcess.restype = wintypes.BOOL
            kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
            kernel32.CloseHandle.restype = wintypes.BOOL
            handle = kernel32.OpenProcess(process_query_limited_information, False, pid)
            if not handle:
                return False
            try:
                exit_code = wintypes.DWORD()
                return bool(kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))) and exit_code.value == still_active
            finally:
                kernel32.CloseHandle(handle)
        except (AttributeError, OSError, ValueError):
            # A permissions failure must be treated as alive; stealing a live
            # lock is more damaging than waiting for the bounded timeout.
            return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _read_lock_record(lock_path: Path) -> dict[str, Any]:
    try:
        text = lock_path.read_text(encoding="utf-8").strip()
    except OSError:
        return {}
    try:
        payload = json.loads(text)
        if isinstance(payload, Mapping):
            return dict(payload)
    except (TypeError, ValueError):
        pass
    # Read v1.1/v1.2-pre locks so a crashed older process does not leave an
    # unrecoverable catalog lock after an in-place upgrade.
    legacy = re.search(r"pid=(\d+)\s+created=([0-9.]+)", text)
    if legacy:
        return {"pid": int(legacy.group(1)), "created_at_epoch": float(legacy.group(2))}
    return {}


def _lock_record_matches(lock_path: Path, owner_token: str) -> bool:
    return str(_read_lock_record(lock_path).get("owner_token") or "") == owner_token


@contextlib.contextmanager
def _exclusive_lock_file(
    lock_path: Path,
    *,
    purpose: str,
    timeout_seconds: float,
    stale_seconds: float = 15.0,
    heartbeat_seconds: float = 2.0,
) -> Iterable[str]:
    """Acquire a portable O_EXCL lock with PID liveness and an owner token.

    The owner refreshes the lock mtime while work is active.  A dead owner is
    reclaimed immediately; malformed/legacy locks are reclaimed only after the
    grace period.  Release removes the path only when its token is still ours,
    preventing an old owner's ``finally`` from deleting a successor's lock.
    """

    lock_path = lock_path.expanduser().resolve()
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + max(0.1, float(timeout_seconds))
    stale_seconds = max(0.05, float(stale_seconds))
    owner_token = secrets.token_hex(16)
    descriptor: int | None = None
    while descriptor is None:
        created_by_us = False
        try:
            descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            created_by_us = True
            record = {
                "owner_token": owner_token,
                "pid": os.getpid(),
                "created_at_epoch": time.time(),
                "purpose": purpose,
            }
            encoded = (json.dumps(record, sort_keys=True) + "\n").encode("utf-8")
            try:
                offset = 0
                while offset < len(encoded):
                    written = os.write(descriptor, encoded[offset:])
                    if written <= 0:
                        raise OSError("lock owner record write made no progress")
                    offset += written
                os.fsync(descriptor)
            except BaseException:
                # os.open succeeded, so this process owns the still-unpublished
                # lock even if os.write failed.  Close the descriptor and remove
                # that exact creation before propagating the original error.
                try:
                    os.close(descriptor)
                finally:
                    descriptor = None
                    if created_by_us:
                        try:
                            lock_path.unlink(missing_ok=True)
                        except OSError:
                            pass
                raise
        except FileExistsError:
            record = _read_lock_record(lock_path)
            try:
                age = max(0.0, time.time() - lock_path.stat().st_mtime)
            except OSError:
                age = 0.0
            pid_value = record.get("pid")
            try:
                owner_pid = int(pid_value)
            except (TypeError, ValueError):
                owner_pid = 0
            owner_dead = bool(owner_pid) and not _process_is_alive(owner_pid)
            malformed_stale = not owner_pid and age >= stale_seconds
            legacy_stale = not record.get("owner_token") and age >= stale_seconds and owner_dead
            if owner_dead or malformed_stale or legacy_stale:
                observed_token = str(record.get("owner_token") or "")
                current = _read_lock_record(lock_path)
                current_token = str(current.get("owner_token") or "")
                current_pid = int(current.get("pid") or 0) if str(current.get("pid") or "").isdigit() else 0
                if current_token == observed_token and current_pid == owner_pid:
                    try:
                        lock_path.unlink()
                        continue
                    except OSError:
                        pass
            if time.monotonic() >= deadline:
                raise PixabayPipelineError(f"Timed out waiting for {purpose} lock: {lock_path}")
            time.sleep(0.05)
    try:
        os.close(descriptor)
    except BaseException:
        descriptor = None
        if _lock_record_matches(lock_path, owner_token):
            lock_path.unlink(missing_ok=True)
        raise
    descriptor = None

    stop_heartbeat = threading.Event()

    def heartbeat() -> None:
        while not stop_heartbeat.wait(max(0.05, float(heartbeat_seconds))):
            if not _lock_record_matches(lock_path, owner_token):
                return
            try:
                os.utime(lock_path, None)
            except OSError:
                return

    heartbeat_thread = threading.Thread(
        target=heartbeat,
        name=f"bgm-montage-lock-{owner_token[:8]}",
        daemon=True,
    )
    heartbeat_thread.start()
    try:
        yield owner_token
    finally:
        stop_heartbeat.set()
        heartbeat_thread.join(timeout=max(0.2, float(heartbeat_seconds) * 2.0))
        if _lock_record_matches(lock_path, owner_token):
            try:
                lock_path.unlink(missing_ok=True)
            except OSError:
                pass


@contextlib.contextmanager
def _exclusive_catalog_lock(path: Path, timeout_seconds: float = 30.0) -> Iterable[str]:
    """Serialize one catalog merge without blocking unrelated asset downloads."""

    lock_path = path.with_name(f".{path.name}.lock")
    with _exclusive_lock_file(
        lock_path,
        purpose="material catalog",
        timeout_seconds=timeout_seconds,
    ) as owner_token:
        yield owner_token


@contextlib.contextmanager
def _exclusive_asset_lock(
    global_material_index: Path,
    asset_id: str | int,
    timeout_seconds: float = 600.0,
    stale_seconds: float = 15.0,
    heartbeat_seconds: float = 2.0,
) -> Iterable[str]:
    """Serialize acquisition of one Pixabay ID across projects/processes."""

    safe_id = re.sub(r"[^0-9A-Za-z_.-]+", "_", str(asset_id)).strip("._") or "unknown"
    lock_path = global_material_index.parent / ".asset-locks" / f"pixabay-{safe_id}.lock"
    with _exclusive_lock_file(
        lock_path,
        purpose=f"Pixabay asset {safe_id}",
        timeout_seconds=timeout_seconds,
        stale_seconds=stale_seconds,
        heartbeat_seconds=heartbeat_seconds,
    ) as owner_token:
        yield owner_token


@contextlib.contextmanager
def _exclusive_manifest_lock(path: Path, timeout_seconds: float = 60.0) -> Iterable[str]:
    """Protect one shared theme/run manifest read-modify-write transaction."""

    lock_path = path.with_name(f".{path.name}.lock")
    with _exclusive_lock_file(
        lock_path,
        purpose="sources manifest",
        timeout_seconds=timeout_seconds,
    ) as owner_token:
        yield owner_token


def _read_json(path: Path, default: Any) -> Any:
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            return json.load(handle)
    except (OSError, ValueError, TypeError):
        return default


def _read_json_strict(path: Path, missing_default: Any) -> Any:
    """Read JSON without treating corruption or permissions failures as empty."""

    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            return json.load(handle)
    except FileNotFoundError:
        return missing_default
    except (PermissionError, OSError, ValueError, TypeError) as exc:
        raise PixabayPipelineError(f"Cannot safely read JSON state {path}: {_safe_error(exc)}") from None


def _load_environment() -> None:
    """Load only a project-root .env, never a credential file beside the skill."""

    explicit_root = os.environ.get("BGM_MONTAGE_PROJECT_ROOT", "").strip()
    cwd = Path.cwd().resolve()
    skill_root = Path(__file__).resolve().parent.parent
    candidates = [Path(explicit_root).expanduser().resolve() / ".env"] if explicit_root else []
    candidates.extend(candidate / ".env" for candidate in (cwd, *list(cwd.parents)[:3]))
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate.resolve()) if candidate.exists() else str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.parent.resolve() == skill_root:
            continue
        if candidate.is_file():
            load_dotenv(candidate, override=False)
            break


@contextlib.contextmanager
def _suppress_sensitive_http_debug() -> Any:
    """Prevent third-party HTTP debug modes from recording the API-key URL."""

    loggers = [logging.getLogger("urllib3.connectionpool"), logging.getLogger("requests.packages.urllib3.connectionpool")]
    previous_disabled = [logger.disabled for logger in loggers]
    previous_http_debug = http.client.HTTPConnection.debuglevel
    try:
        for logger in loggers:
            logger.disabled = True
        http.client.HTTPConnection.debuglevel = 0
        yield
    finally:
        http.client.HTTPConnection.debuglevel = previous_http_debug
        for logger, disabled in zip(loggers, previous_disabled):
            logger.disabled = disabled


def _parse_aspect_ratio(value: str | float | Sequence[int | float]) -> tuple[float, str]:
    if isinstance(value, (tuple, list)) and len(value) == 2:
        width, height = float(value[0]), float(value[1])
        label = f"{value[0]}:{value[1]}"
    elif isinstance(value, (int, float)):
        width, height = float(value), 1.0
        label = f"{float(value):g}:1"
    else:
        text = str(value).strip().lower().replace("×", "x")
        match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*[:x/]\s*(\d+(?:\.\d+)?)\s*", text)
        if not match:
            raise ValueError(f"Invalid aspect ratio: {value!r}; expected for example 16:9 or 9:16")
        width, height = float(match.group(1)), float(match.group(2))
        label = f"{match.group(1)}:{match.group(2)}"
    if width <= 0 or height <= 0:
        raise ValueError("Aspect ratio components must be positive")
    return width / height, label


def _parse_resolution(value: str | Sequence[int | float]) -> tuple[int, int]:
    if isinstance(value, (tuple, list)) and len(value) == 2:
        width, height = int(value[0]), int(value[1])
    else:
        match = re.fullmatch(r"\s*(\d+)\s*[x×,:]\s*(\d+)\s*", str(value))
        if not match:
            raise ValueError(f"Invalid resolution: {value!r}; expected WIDTHxHEIGHT")
        width, height = int(match.group(1)), int(match.group(2))
    if width <= 0 or height <= 0:
        raise ValueError("Resolution components must be positive")
    return width, height


_CHINESE_TERM_MAP: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("海洋", ("ocean", "waves", "coast")),
    ("大海", ("ocean", "sea", "coast")),
    ("海边", ("coast", "beach", "ocean")),
    ("沙滩", ("beach", "shore", "sand")),
    ("城市", ("city", "urban", "street")),
    ("建筑", ("architecture", "building", "city")),
    ("自然", ("nature", "landscape", "outdoors")),
    ("森林", ("forest", "trees", "nature")),
    ("山", ("mountain", "landscape", "nature")),
    ("天空", ("sky", "clouds", "atmosphere")),
    ("日出", ("sunrise", "golden hour", "morning")),
    ("日落", ("sunset", "golden hour", "dusk")),
    ("科技", ("technology", "digital", "innovation")),
    ("工业", ("industry", "factory", "machinery")),
    ("工厂", ("factory", "manufacturing", "machinery")),
    ("机械", ("machinery", "engineering", "industrial")),
    ("制造", ("manufacturing", "production", "factory")),
    ("焊接", ("welding", "sparks", "workshop")),
    ("农业", ("agriculture", "farm", "fields")),
    ("食物", ("food", "cooking", "ingredients")),
    ("美食", ("food", "cuisine", "cooking")),
    ("旅行", ("travel", "destination", "adventure")),
    ("汽车", ("car", "driving", "automotive")),
    ("运动", ("sports", "athlete", "action")),
    ("人物", ("people", "portrait", "lifestyle")),
    ("女性", ("woman", "portrait", "lifestyle")),
    ("男性", ("man", "portrait", "lifestyle")),
    ("家庭", ("family", "home", "lifestyle")),
    ("儿童", ("children", "family", "play")),
    ("动物", ("animals", "wildlife", "nature")),
    ("宇宙", ("space", "galaxy", "stars")),
    ("足球", ("football", "soccer", "sport")),
    ("篮球", ("basketball", "sport", "athlete")),
    ("安静", ("calm", "quiet", "peaceful")),
    ("治愈", ("serene", "peaceful", "gentle")),
    ("孤独", ("solitude", "lonely", "atmospheric")),
    ("自由", ("freedom", "open", "adventure")),
    ("激情", ("energetic", "dynamic", "powerful")),
    ("浪漫", ("romantic", "dreamy", "warm")),
    ("未来", ("futuristic", "technology", "neon")),
)

_WORD_SYNONYMS: dict[str, tuple[str, ...]] = {
    "ocean": ("sea", "seascape", "coast", "waves"),
    "city": ("urban", "downtown", "metropolis", "street"),
    "nature": ("outdoors", "landscape", "wilderness", "scenery"),
    "industry": ("industrial", "manufacturing", "factory", "engineering"),
    "factory": ("manufacturing", "production line", "workshop", "industrial"),
    "technology": ("digital", "innovation", "futuristic", "electronics"),
    "travel": ("journey", "destination", "adventure", "exploration"),
    "calm": ("serene", "peaceful", "tranquil", "gentle"),
    "energetic": ("dynamic", "powerful", "fast action", "intense"),
    "people": ("lifestyle", "human", "portrait", "community"),
    "food": ("cuisine", "cooking", "ingredients", "kitchen"),
    "sports": ("athlete", "competition", "training", "action"),
    "car": ("automotive", "vehicle", "driving", "road"),
}

_ENGLISH_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in",
    "into", "is", "it", "of", "on", "or", "the", "to", "video", "with", "footage",
    "style", "scene", "shot", "clips", "clip", "overall", "unknown", "none",
    "true", "false",
}


def _english_words(value: Any) -> list[str]:
    text = unicodedata.normalize("NFKC", str(value)).lower()
    words = re.findall(r"[a-z][a-z0-9-]{1,24}", text)
    return [word.replace("-", " ") for word in words if word not in _ENGLISH_STOPWORDS]


def _theme_terms(theme: str) -> list[str]:
    terms = _english_words(theme)
    for chinese, english in _CHINESE_TERM_MAP:
        if chinese in theme:
            terms.extend(english)
    return _unique_terms(terms) or ["cinematic", "visual", "atmosphere"]


def _unique_terms(values: Iterable[str], limit: int | None = None) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = re.sub(r"\s+", " ", str(value).strip().lower())
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
        if limit is not None and len(result) >= limit:
            break
    return result


def _ratio_label(width: Any, height: Any) -> str | None:
    try:
        width_value = int(width)
        height_value = int(height)
    except (TypeError, ValueError):
        return None
    if width_value <= 0 or height_value <= 0:
        return None
    divisor = math.gcd(width_value, height_value)
    return f"{width_value // divisor}:{height_value // divisor}"


def _fingerprint_payload(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    nested = value.get("fingerprint")
    return nested if isinstance(nested, Mapping) else value


def _canonical_source_id(
    value: Mapping[str, Any] | None,
    fallback_pixabay_id: Any = None,
) -> str:
    value = value if isinstance(value, Mapping) else {}
    explicit = value.get("canonical_source_id")
    if explicit:
        return str(explicit)
    fingerprint = _fingerprint_payload(value)
    sha256 = str(fingerprint.get("sha256") or value.get("file_hash") or "").strip().lower()
    if sha256:
        return f"sha256:{sha256}"
    pixabay_id = value.get("pixabay_id", value.get("id", fallback_pixabay_id))
    if pixabay_id not in (None, ""):
        return f"pixabay:{pixabay_id}"
    local_path = str(value.get("local_path") or "").strip()
    if local_path:
        return "path:" + os.path.normcase(str(Path(local_path).expanduser().resolve()))
    return "unknown:" + hashlib.sha256(
        json.dumps(dict(value), ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:20]


def _semantic_tags(
    tags: Any,
    scene_category: Any = None,
    shot_type: Any = None,
    shot_scale: Any = None,
    extra: Iterable[Any] = (),
) -> list[str]:
    values: list[str] = []
    values.extend(_english_words(tags))
    for value in (scene_category, shot_type, shot_scale, *extra):
        if value not in (None, ""):
            values.extend(_english_words(str(value).replace("_", " ")))
    return _unique_terms(values, 30)


def _history_count(history: Any, legacy_intervals: Any = None) -> int:
    if isinstance(history, list):
        total = 0
        for item in history:
            if not isinstance(item, Mapping):
                continue
            intervals = item.get("intervals") or item.get("usage_intervals") or []
            total += len(intervals) if isinstance(intervals, list) else 0
        if total:
            return total
    return len(legacy_intervals) if isinstance(legacy_intervals, list) else 0


def _collect_profile_words(profile: Mapping[str, Any] | None, key_fragments: Sequence[str]) -> list[str]:
    if not isinstance(profile, Mapping):
        return []
    words: list[str] = []

    def visit(value: Any, path: str, depth: int) -> None:
        if depth > 7:
            return
        if isinstance(value, Mapping):
            for key, item in value.items():
                visit(item, f"{path}.{str(key).lower()}", depth + 1)
        elif isinstance(value, (list, tuple)):
            for item in value[:30]:
                visit(item, path, depth + 1)
        elif any(fragment in path for fragment in key_fragments):
            words.extend(_english_words(value))

    visit(profile, "", 0)
    return _unique_terms(words)


def _number_values(profile: Mapping[str, Any] | None, key_fragments: Sequence[str]) -> list[float]:
    values: list[float] = []
    if not isinstance(profile, Mapping):
        return values

    def visit(value: Any, path: str, depth: int) -> None:
        if depth > 7:
            return
        if isinstance(value, Mapping):
            for key, item in value.items():
                visit(item, f"{path}.{str(key).lower()}", depth + 1)
        elif isinstance(value, (list, tuple)):
            for item in value[:50]:
                visit(item, path, depth + 1)
        elif any(fragment in path for fragment in key_fragments):
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                return
            if math.isfinite(numeric):
                values.append(numeric)

    visit(profile, "", 0)
    return values


def _style_payload(profile: Mapping[str, Any] | None) -> Mapping[str, Any]:
    """Accept either the style payload itself or analyze_references' wrapper."""

    if not isinstance(profile, Mapping):
        return {}
    for key in ("style_profile", "style"):
        nested = profile.get(key)
        if isinstance(nested, Mapping):
            return nested
    return profile


def _audio_payload(profile: Mapping[str, Any] | None) -> Mapping[str, Any]:
    """Accept either a BGM profile or a wrapper returned by the analyzer."""

    if not isinstance(profile, Mapping):
        return {}
    for key in ("audio_profile", "bgm_profile", "profile"):
        nested = profile.get(key)
        if isinstance(nested, Mapping):
            return nested
    return profile


def _audio_descriptors(audio_profile: Mapping[str, Any] | None) -> list[str]:
    audio_profile = _audio_payload(audio_profile)
    words = _collect_profile_words(
        audio_profile,
        (
            "estimated_mood", "mood", "emotion", "energy", "timbre", "brightness_label",
            "density_label", "edit_guidance", "stage_profile", "section", "texture", "character",
        ),
    )
    energy_values = _number_values(audio_profile, ("mean_energy", "energy.mean", "energy_curve", "energy", "intensity", "rms"))
    bpm_values = _number_values(audio_profile, ("tempo_bpm_estimate", "bpm", "tempo"))
    if energy_values:
        energy = float(np.median(energy_values))
        if energy > 1.0:
            energy = min(1.0, energy / 100.0)
        words.extend(("energetic", "dynamic") if energy >= 0.62 else ("calm", "atmospheric"))
    if bpm_values:
        bpm = float(np.median([value for value in bpm_values if value > 0] or [0]))
        if bpm >= 120:
            words.extend(("fast motion", "dynamic"))
        elif 0 < bpm <= 85:
            words.extend(("slow motion", "serene"))
    joined = " ".join(words)
    if any(word in joined for word in ("bright", "major", "uplifting")):
        words.extend(("bright", "uplifting"))
    if any(word in joined for word in ("dark", "minor", "melancholy")):
        words.extend(("moody", "dramatic"))
    return _unique_terms(words, 8)


def _clean_query(parts: Iterable[str]) -> str:
    words: list[str] = []
    for part in parts:
        words.extend(_english_words(part))
    query = " ".join(_unique_terms(words))
    if not query:
        query = "cinematic atmosphere"
    while len(query) > 100 and " " in query:
        query = query.rsplit(" ", 1)[0]
    return query[:100].strip()


def _scene_concepts(theme_terms: Sequence[str]) -> list[str]:
    joined = " ".join(theme_terms)
    rules: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
        (("ocean", "sea", "coast", "beach"), ("waves aerial", "shore detail", "coastal landscape", "water reflection")),
        (("factory", "industry", "manufacturing", "machinery"), ("production line", "machine close up", "factory interior", "industrial sparks")),
        (("city", "urban", "street"), ("city skyline", "street life", "urban aerial", "architecture detail")),
        (("nature", "forest", "mountain", "landscape"), ("wide landscape", "nature detail", "aerial wilderness", "sunlight atmosphere")),
        (("technology", "digital", "futuristic"), ("technology close up", "digital network", "futuristic city", "electronics detail")),
        (("food", "cooking", "cuisine"), ("cooking close up", "ingredients detail", "kitchen action", "food table")),
        (("sports", "athlete", "action"), ("athlete close up", "training action", "sports wide shot", "competition detail")),
        (("travel", "journey", "adventure"), ("destination aerial", "traveler lifestyle", "road journey", "landscape detail")),
        (("people", "portrait", "lifestyle", "family"), ("human portrait", "lifestyle detail", "people wide shot", "authentic moment")),
    )
    for needles, concepts in rules:
        if any(needle in joined for needle in needles):
            return list(concepts)
    return ["wide establishing", "close up detail", "aerial view", "authentic atmosphere"]


def _human_focused_theme(theme_terms: Sequence[str]) -> bool:
    tokens = set(_english_words(" ".join(theme_terms)))
    return bool(tokens & {"people", "person", "portrait", "family", "fashion", "wedding", "interview", "human"})


def generate_visual_queries(
    theme: str,
    style_profile: Mapping[str, Any] | None,
    audio_profile: Mapping[str, Any] | None,
    expansion_level: int = 0,
) -> list[str]:
    """Generate task-specific, multi-axis queries without location allowlists."""

    visual_profile = _resolve_visual_profile(theme, style_profile, audio_profile)
    return [
        str(record["query"])
        for record in plan_visual_search_queries(visual_profile, expansion_level)
        if record.get("query")
    ][:6]


def _resolve_visual_profile(
    theme: str,
    style_profile: Mapping[str, Any] | None,
    audio_profile: Mapping[str, Any] | None,
    visual_request: str = "",
) -> dict[str, Any]:
    style = _style_payload(style_profile)
    embedded = style.get("visual_style_profile")
    if isinstance(embedded, Mapping) and str(embedded.get("schema_version") or "") == "1.3":
        return dict(embedded)
    return build_visual_style_profile(theme, style, _audio_payload(audio_profile), visual_request)


def _search_cache_path(cache_dir: Path, query: str, page: int, per_page: int) -> Path:
    identity = json.dumps(
        {"endpoint": PIXABAY_VIDEO_API, "query": query, "page": page, "per_page": per_page},
        sort_keys=True,
        ensure_ascii=True,
    )
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    # ``cache_dir`` is the Pixabay-stage namespace root.  v1.0 appended the
    # namespace here as well as in the unified caller, producing
    # ``pixabay/pixabay/search`` and making the standalone and unified entry
    # points miss each other's cache.  Normalize once at the public boundary
    # and keep all helpers relative to that namespace.
    return cache_dir / "search" / f"{digest}.json"


def _pixabay_search(
    session: requests.Session,
    api_key: str,
    query: str,
    cache_dir: Path,
    per_page: int,
    page: int = 1,
) -> tuple[dict[str, Any], bool]:
    cache_path = _search_cache_path(cache_dir, query, page, per_page)
    now = time.time()
    cached = _read_json(cache_path, {})
    if isinstance(cached, Mapping):
        cached_at = float(cached.get("cached_at_epoch", 0) or 0)
        data = cached.get("response")
        if now - cached_at <= CACHE_TTL_SECONDS and isinstance(data, Mapping):
            return dict(data), True

    params = {
        "key": api_key,
        "q": query,
        "lang": "en",
        "video_type": "all",
        "safesearch": "true",
        "order": "popular",
        "page": page,
        "per_page": max(3, min(200, int(per_page))),
    }
    try:
        with _suppress_sensitive_http_debug():
            response = session.get(PIXABAY_VIDEO_API, params=params, timeout=REQUEST_TIMEOUT)
        status = response.status_code
        if status != 200:
            # Never interpolate response.url: it contains the credential.
            body_hint = ""
            try:
                body = response.json()
                if isinstance(body, Mapping):
                    body_hint = str(body.get("message") or body.get("error") or "")
            except (ValueError, TypeError):
                pass
            raise PixabayPipelineError(
                f"Pixabay API returned HTTP {status}"
                + (f": {_safe_error(body_hint, api_key)}" if body_hint else "")
            )
        payload = response.json()
        if not isinstance(payload, Mapping):
            raise PixabayPipelineError("Pixabay API returned an unexpected response shape")
    except PixabayPipelineError:
        raise
    except requests.RequestException as exc:
        raise PixabayPipelineError(f"Pixabay API request failed: {_safe_error(exc, api_key)}") from None
    except ValueError as exc:
        raise PixabayPipelineError(f"Pixabay API returned invalid JSON: {_safe_error(exc, api_key)}") from None

    clean_payload = _strip_secrets(dict(payload), api_key)
    _atomic_write_json(
        cache_path,
        {
            "schema_version": SCHEMA_VERSION,
            "cached_at": _utc_now(),
            "cached_at_epoch": now,
            "ttl_seconds": CACHE_TTL_SECONDS,
            "query": query,
            "page": page,
            "per_page": per_page,
            "response": clean_payload,
        },
        api_key,
    )
    return dict(clean_payload), False


def _variant_for_hit(hit: Mapping[str, Any], min_resolution: tuple[int, int]) -> dict[str, Any] | None:
    videos = hit.get("videos")
    if not isinstance(videos, Mapping):
        return None
    variants: list[dict[str, Any]] = []
    for name, raw in videos.items():
        if not isinstance(raw, Mapping) or not raw.get("url"):
            continue
        try:
            width = int(raw.get("width") or 0)
            height = int(raw.get("height") or 0)
            size = int(raw.get("size") or 0)
        except (TypeError, ValueError):
            continue
        variants.append(
            {
                "name": str(name),
                "url": str(raw["url"]),
                "width": width,
                "height": height,
                "size": size,
                "thumbnail": raw.get("thumbnail"),
            }
        )
    if not variants:
        return None
    min_long, min_short = sorted(min_resolution, reverse=True)

    def adequate(item: Mapping[str, Any]) -> bool:
        long_side, short_side = sorted((int(item["width"]), int(item["height"])), reverse=True)
        return long_side >= min_long and short_side >= min_short

    hd = [item for item in variants if adequate(item)]
    pool = hd or variants
    # Pixabay's largest available variant is the final HD choice; it is not
    # fetched until the candidate survives all pre-download ranking.
    return max(pool, key=lambda item: (item["width"] * item["height"], item["size"]))


def _thumbnail_url(hit: Mapping[str, Any], variant: Mapping[str, Any]) -> str | None:
    if isinstance(variant.get("thumbnail"), str):
        return str(variant["thumbnail"])
    videos = hit.get("videos")
    if isinstance(videos, Mapping):
        for raw in videos.values():
            if isinstance(raw, Mapping) and isinstance(raw.get("thumbnail"), str):
                return str(raw["thumbnail"])
    picture_id = hit.get("picture_id")
    if picture_id and re.fullmatch(r"[A-Za-z0-9_-]+", str(picture_id)):
        return f"https://i.vimeocdn.com/video/{picture_id}_640x360.jpg"
    return None


def _target_color(style_profile: Mapping[str, Any] | None) -> dict[str, float]:
    style_profile = _style_payload(style_profile)
    words = _collect_profile_words(style_profile, ("color", "colour", "palette", "grade", "tone", "lighting"))
    joined = " ".join(words)
    hue = 30.0
    saturation = 0.48
    value = 0.58
    if any(word in joined for word in ("blue", "cyan", "cool", "teal")):
        hue = 205.0
    elif any(word in joined for word in ("green", "forest", "nature")):
        hue = 115.0
    elif any(word in joined for word in ("purple", "violet", "magenta")):
        hue = 285.0
    elif any(word in joined for word in ("red", "orange", "warm", "golden")):
        hue = 28.0
    if any(word in joined for word in ("monochrome", "muted", "desaturated")):
        saturation = 0.23
    elif any(word in joined for word in ("vibrant", "saturated", "colorful")):
        saturation = 0.78
    if any(word in joined for word in ("dark", "low key", "moody", "night")):
        value = 0.34
    elif any(word in joined for word in ("bright", "high key", "airy")):
        value = 0.76
    brightness_values = _number_values(style_profile, ("brightness",))
    saturation_values = _number_values(style_profile, ("saturation",))
    warmth_values = _number_values(style_profile, ("warmth_index", "warmth"))
    if brightness_values:
        observed = float(np.median(brightness_values))
        value = float(np.clip(observed / 255.0 if observed > 1.0 else observed, 0.08, 0.95))
    if saturation_values:
        observed = float(np.median(saturation_values))
        saturation = float(np.clip(observed / 255.0 if observed > 1.0 else observed, 0.05, 0.95))
    if warmth_values:
        warmth = float(np.median(warmth_values))
        if warmth > 0.12:
            hue = 28.0
        elif warmth < -0.12:
            hue = 205.0
    return {"hue_degrees": hue, "saturation": saturation, "value": value}


def _color_similarity(hue: float, saturation: float, value: float, target: Mapping[str, float]) -> float:
    hue_distance = abs(hue - float(target["hue_degrees"])) % 360.0
    hue_distance = min(hue_distance, 360.0 - hue_distance) / 180.0
    sat_distance = abs(saturation - float(target["saturation"]))
    val_distance = abs(value - float(target["value"]))
    return float(np.clip(1.0 - (0.45 * hue_distance + 0.25 * sat_distance + 0.30 * val_distance), 0.0, 1.0))


def _image_signals(image_bgr: np.ndarray, target_color: Mapping[str, float]) -> dict[str, Any]:
    if image_bgr is None or image_bgr.size == 0:
        return {
            "available": False,
            "sharpness_score": 0.5,
            "exposure_score": 0.5,
            "color_score": 0.5,
            "text_watermark_risk": 0.5,
            "perceptual_hash": None,
            "subject_profile": {},
            "face_content_risk": 0.5,
            "aesthetic_metrics": frame_aesthetic_metrics(np.empty((0, 0, 3), dtype=np.uint8)),
        }
    height, width = image_bgr.shape[:2]
    scale = min(1.0, 640.0 / max(width, height))
    if scale < 1.0:
        image_bgr = cv2.resize(image_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    sharpness_raw = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    sharpness_score = float(np.clip(math.log1p(sharpness_raw) / math.log1p(900.0), 0.0, 1.0))
    mean_luma = float(gray.mean())
    clipped = float(np.mean((gray <= 5) | (gray >= 250)))
    exposure_score = float(np.clip(1.0 - abs(mean_luma - 128.0) / 128.0 - clipped * 1.8, 0.0, 1.0))
    hue = float(np.median(hsv[..., 0])) * 2.0
    saturation = float(np.mean(hsv[..., 1])) / 255.0
    value = float(np.mean(hsv[..., 2])) / 255.0
    color_score = _color_similarity(hue, saturation, value, target_color)
    edges = cv2.Canny(gray, 90, 180) > 0
    h, w = edges.shape
    border = np.zeros_like(edges, dtype=bool)
    border[: max(1, h // 6), :] = True
    border[-max(1, h // 6) :, :] = True
    border[:, : max(1, w // 8)] = True
    border[:, -max(1, w // 8) :] = True
    border_density = float(edges[border].mean()) if np.any(border) else 0.0
    center_density = float(edges[~border].mean()) if np.any(~border) else border_density
    text_risk = float(np.clip((border_density - center_density * 0.75) * 5.0, 0.0, 1.0))
    pil_image = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    phash = str(imagehash.phash(pil_image))
    subject = subject_region(image_bgr, _face_detector())
    image_face_risk = face_content_risk("", None, subject)
    aesthetic_metrics = frame_aesthetic_metrics(image_bgr)
    return {
        "available": True,
        "width": int(image_bgr.shape[1]),
        "height": int(image_bgr.shape[0]),
        "sharpness_raw": round(sharpness_raw, 3),
        "sharpness_score": round(sharpness_score, 4),
        "mean_luma": round(mean_luma, 3),
        "exposure_score": round(exposure_score, 4),
        "mean_hsv": {"hue_degrees": round(hue, 2), "saturation": round(saturation, 4), "value": round(value, 4)},
        "color_score": round(color_score, 4),
        "text_watermark_risk": round(text_risk, 4),
        "perceptual_hash": phash,
        "subject_profile": subject,
        "face_content_risk": round(image_face_risk, 4),
        "aesthetic_metrics": aesthetic_metrics,
    }


def _get_thumbnail_signals(
    session: requests.Session,
    hit: Mapping[str, Any],
    variant: Mapping[str, Any],
    cache_dir: Path,
    target_color: Mapping[str, float],
) -> dict[str, Any]:
    asset_id = str(hit.get("id") or hashlib.sha1(str(hit).encode("utf-8")).hexdigest()[:12])
    thumb_path = cache_dir / "thumbnails" / f"{asset_id}.jpg"
    data: bytes | None = None
    if thumb_path.is_file() and time.time() - thumb_path.stat().st_mtime <= CACHE_TTL_SECONDS:
        try:
            data = thumb_path.read_bytes()
        except OSError:
            data = None
    if data is None:
        url = _thumbnail_url(hit, variant)
        if url:
            try:
                response = session.get(url, timeout=REQUEST_TIMEOUT)
                if response.status_code == 200 and response.content and len(response.content) <= 10 * 1024 * 1024:
                    data = response.content
                    thumb_path.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        thumb_path.write_bytes(data)
                    except OSError:
                        pass
            except requests.RequestException:
                data = None
    if not data:
        return _image_signals(np.empty((0, 0, 3), dtype=np.uint8), target_color)
    array = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    return _image_signals(image, target_color)


def _infer_shot_type(tags: str) -> str:
    text = tags.lower()
    if any(term in text for term in ("aerial", "drone", "bird view", "overhead")):
        return "aerial"
    if any(term in text for term in ("macro", "close up", "close-up", "detail")):
        return "close_up"
    if any(term in text for term in ("wide", "landscape", "panorama", "skyline")):
        return "wide"
    person_pattern = re.compile(r"\b(?:portrait|face|person|people|woman|women|man|men)\b")
    if person_pattern.search(text):
        return "medium_portrait"
    if any(term in text for term in ("pov", "point of view", "tracking")):
        return "pov_tracking"
    return "medium"


def _infer_shot_scale(tags: str) -> str:
    shot_type = _infer_shot_type(tags)
    return {
        "aerial": "extreme_wide",
        "wide": "wide",
        "close_up": "close_up",
        "medium_portrait": "medium",
        "pov_tracking": "medium",
        "medium": "medium",
    }[shot_type]


def _motion_tag_score(tags: str) -> float:
    text = tags.lower()
    strong = ("fast", "action", "running", "driving", "waves", "timelapse", "time lapse", "dynamic", "tracking")
    gentle = ("slow motion", "calm", "still", "quiet", "static")
    score = 0.5 + 0.09 * sum(term in text for term in strong) - 0.08 * sum(term in text for term in gentle)
    return float(np.clip(score, 0.05, 0.95))


def _desired_motion(style_profile: Mapping[str, Any] | None, audio_profile: Mapping[str, Any] | None) -> float:
    style_profile = _style_payload(style_profile)
    audio_profile = _audio_payload(audio_profile)
    words = " ".join(
        _collect_profile_words(style_profile, ("motion", "movement", "pace", "editing"))
        + _audio_descriptors(audio_profile)
    )
    desired = 0.5
    if any(term in words for term in ("fast", "dynamic", "energetic", "intense", "rapid")):
        desired = 0.78
    if any(term in words for term in ("slow", "calm", "gentle", "serene", "static")):
        desired = 0.28
    return desired


def _token_set(values: Iterable[str]) -> set[str]:
    tokens: set[str] = set()
    for value in values:
        tokens.update(_english_words(value))
    return tokens


def _timeline_slots(value: Any) -> list[dict[str, Any]]:
    """Normalize a pre-download timeline plan without imposing one schema."""

    if isinstance(value, (str, os.PathLike)):
        payload = _read_json(Path(value).expanduser().resolve(), None)
        if payload is None:
            raise PixabayPipelineError(f"Timeline plan is missing or invalid JSON: {value}")
        value = payload
    if isinstance(value, Mapping):
        for key in ("slots", "shots", "timeline", "items", "segments"):
            nested = value.get(key)
            if isinstance(nested, list):
                value = nested
                break
    if not isinstance(value, (list, tuple)):
        return []
    slots: list[dict[str, Any]] = []
    for index, raw in enumerate(value):
        if not isinstance(raw, Mapping):
            continue
        slot = dict(raw)
        slot.setdefault("index", index)
        slots.append(slot)
    return slots


def _slot_terms(slot: Mapping[str, Any]) -> list[str]:
    values: list[Any] = []
    for key in (
        "recommended_content", "content", "description", "subject", "scene", "emotion",
        "mood", "shot_scale", "recommended_shot_scale", "motion", "recommended_motion",
        "section", "section_role", "event", "event_type",
    ):
        value = slot.get(key)
        if isinstance(value, (list, tuple)):
            values.extend(value)
        elif value not in (None, ""):
            values.append(value)
    return _unique_terms((word for value in values for word in _english_words(value)), 30)


def _slot_is_important(slot: Mapping[str, Any]) -> bool:
    if any(bool(slot.get(key)) for key in ("important", "is_key", "highlight", "is_highlight", "emphasis")):
        return True
    text = " ".join(_slot_terms(slot))
    return any(term in text for term in ("drop", "climax", "surge", "hard stop", "accent", "impact"))


def _candidate_slot_score(candidate: Mapping[str, Any], slot: Mapping[str, Any]) -> float:
    slot_tokens = set(_slot_terms(slot))
    candidate_tokens = _token_set(
        [
            str(candidate.get("tags") or ""),
            " ".join(str(value) for value in candidate.get("semantic_tags", []) or []),
            str(candidate.get("scene_category") or ""),
            str(candidate.get("shot_type") or ""),
            str(candidate.get("shot_scale") or ""),
        ]
    )
    semantic = (
        len(slot_tokens & candidate_tokens) / max(1, min(8, len(slot_tokens)))
        if slot_tokens
        else 0.35
    )
    desired_scale = str(slot.get("recommended_shot_scale") or slot.get("shot_scale") or "").lower()
    scale_match = 1.0 if desired_scale and desired_scale in str(candidate.get("shot_scale") or "").lower() else 0.35
    desired_motion = str(slot.get("recommended_motion") or slot.get("motion") or "").lower()
    motion_value = float(candidate.get("motion_score_estimate") or candidate.get("motion_score") or 0.5)
    if any(term in desired_motion for term in ("strong", "fast", "dynamic", "high")):
        motion_match = motion_value
    elif any(term in desired_motion for term in ("slow", "calm", "static", "gentle")):
        motion_match = 1.0 - motion_value
    else:
        motion_match = 0.5
    return round(0.68 * semantic + 0.18 * scale_match + 0.14 * motion_match, 6)


def _slot_candidate_coverage(
    slots: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    required_per_important_slot: int = 3,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    failures: list[str] = []
    for slot in slots:
        ranked = sorted(
            (
                (_candidate_slot_score(candidate, slot), candidate)
                for candidate in candidates
            ),
            key=lambda item: (item[0], float(item[1].get("pre_score", item[1].get("metadata_score", 0.0)))),
            reverse=True,
        )
        meaningful = [item for item in ranked if item[0] >= 0.18]
        important = _slot_is_important(slot)
        required = required_per_important_slot if important else 1
        candidate_count = len(meaningful)
        if important and candidate_count < required:
            failures.append(
                f"important slot {slot.get('index')} candidates {candidate_count} < {required}"
            )
        records.append(
            {
                "slot_index": slot.get("index"),
                "important": important,
                "required_candidates": required,
                "candidate_count": candidate_count,
                "top_candidates": [
                    {
                        "pixabay_id": candidate.get("pixabay_id", candidate.get("id")),
                        "canonical_source_id": candidate.get("canonical_source_id"),
                        "score": score,
                    }
                    for score, candidate in meaningful[:8]
                ],
            }
        )
    return {"passed": not failures, "failures": failures, "slots": records}


def _local_library_candidates(
    entries: Sequence[Mapping[str, Any]],
    theme: str,
    slots: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Turn relevant, available material-library rows into rankable candidates."""

    wanted = _token_set([theme, *_theme_terms(theme), *[" ".join(_slot_terms(slot)) for slot in slots]])
    candidates: list[dict[str, Any]] = []
    for raw in entries:
        entry = _normalize_library_entry(raw)
        local_path = Path(str(entry.get("local_path") or ""))
        if not entry.get("available") or not local_path.is_file():
            continue
        tags = str(entry.get("tags") or "")
        entry_tokens = _token_set([tags, " ".join(entry.get("semantic_tags") or [])])
        if wanted and not (wanted & entry_tokens):
            continue
        media = entry.get("media") if isinstance(entry.get("media"), Mapping) else {}
        fingerprint = entry.get("fingerprint") if isinstance(entry.get("fingerprint"), Mapping) else {}
        quality = entry.get("quality") if isinstance(entry.get("quality"), Mapping) else {}
        width = int(media.get("width") or fingerprint.get("width") or entry.get("width") or 0)
        height = int(media.get("height") or fingerprint.get("height") or entry.get("height") or 0)
        duration = float(media.get("duration_seconds") or fingerprint.get("duration_seconds") or entry.get("duration_seconds") or 0.0)
        pixabay_id = entry.get("pixabay_id")
        if pixabay_id in (None, ""):
            continue
        candidates.append(
            {
                "id": pixabay_id,
                "pixabay_id": pixabay_id,
                "page_url": entry.get("page_url") or "",
                "tags": tags,
                "duration": duration,
                "user": entry.get("author") or "",
                "views": 0,
                "downloads": 0,
                "likes": 0,
                "comments": 0,
                "variant": {
                    "name": "cached",
                    "url": entry.get("download_url") or "",
                    "width": width,
                    "height": height,
                    "size": int(fingerprint.get("size_bytes") or 0),
                },
                "matched_queries": ["local material library"],
                "search_rounds": [0],
                "raw": {"id": pixabay_id},
                "local_reuse_entry": entry,
                "canonical_source_id": entry.get("canonical_source_id"),
                "semantic_tags": list(entry.get("semantic_tags") or []),
                "scene_category": quality.get("scene_category") or _scene_category(tags, None),
                "shot_type": entry.get("shot_type") or _infer_shot_type(tags),
                "shot_scale": entry.get("shot_scale") or _infer_shot_scale(tags),
                "motion_score_estimate": quality.get("motion_score", 0.5),
                "face_content_risk": quality.get("face_content_risk", face_content_risk(tags, None, None)),
            }
        )
    return candidates


def _metadata_score(
    candidate: Mapping[str, Any],
    theme: str,
    style_profile: Mapping[str, Any] | None,
    audio_profile: Mapping[str, Any] | None,
    target_ratio: float,
    min_resolution: tuple[int, int],
    prefer_wide_aerial: bool = False,
    visual_cohesion_profile: str = "none",
) -> tuple[float, dict[str, float]]:
    variant = candidate["variant"]
    tags = str(candidate.get("tags") or "")
    semantic_text = " ".join(str(value) for value in candidate.get("semantic_tags", []) if value)
    visual_text = f"{tags} {semantic_text}".strip()
    tag_tokens = _token_set([visual_text])
    style_profile = _style_payload(style_profile)
    audio_profile = _audio_payload(audio_profile)
    visual_profile = _resolve_visual_profile(
        theme,
        style_profile,
        audio_profile,
        "" if visual_cohesion_profile in {"", "none", "auto"} else visual_cohesion_profile.replace("_", " "),
    )
    visual_metadata = metadata_profile_fit(visual_text, visual_profile)
    relevant = _token_set(
        [theme]
        + _theme_terms(theme)
        + _collect_profile_words(
            style_profile,
            ("topic", "subject", "content", "tag", "mood", "positive_terms", "shot_scale_terms"),
        )[:20]
        + list(candidate.get("matched_queries", []))
    )
    if relevant:
        coverage = len(tag_tokens & relevant) / max(1, min(7, len(relevant)))
        query_hits = sum(bool(tag_tokens & _token_set([query])) for query in candidate.get("matched_queries", []))
        relevance = float(np.clip(0.15 + coverage * 0.72 + min(0.13, query_hits * 0.04), 0.0, 1.0))
    else:
        relevance = 0.4
    avoid_tokens = _token_set(
        _collect_profile_words(style_profile, ("avoid_terms", "negative_terms"))
    )
    avoid_matches = len(tag_tokens & avoid_tokens)
    avoid_penalty = float(min(0.6, avoid_matches * 0.15))
    relevance = float(np.clip(relevance * (1.0 - avoid_penalty), 0.0, 1.0))
    width = int(variant.get("width") or 0)
    height = int(variant.get("height") or 0)
    long_side, short_side = sorted((width, height), reverse=True)
    min_long, min_short = sorted(min_resolution, reverse=True)
    resolution = float(np.clip(min(long_side / max(1, min_long), short_side / max(1, min_short)), 0.0, 1.0))
    ratio = width / max(1, height)
    aspect = float(math.exp(-0.85 * abs(math.log(max(0.01, ratio) / max(0.01, target_ratio)))))
    likes = max(0, int(candidate.get("likes") or 0))
    views = max(0, int(candidate.get("views") or 0))
    downloads = max(0, int(candidate.get("downloads") or 0))
    popularity = float(
        np.clip(0.45 * math.log1p(downloads) / math.log(100_001) + 0.35 * math.log1p(views) / math.log(1_000_001) + 0.20 * min(1.0, likes / 1000.0), 0.0, 1.0)
    )
    duration = float(candidate.get("duration") or 0)
    duration_score = float(np.clip(duration / 6.0, 0.0, 1.0) * np.clip((90.0 - duration) / 45.0, 0.3, 1.0))
    try:
        motion = float(candidate.get("motion_score_estimate"))
    except (TypeError, ValueError):
        motion = _motion_tag_score(tags)
    desired = _desired_motion(style_profile, audio_profile)
    motion_style = float(np.clip(1.0 - abs(motion - desired), 0.0, 1.0))
    metadata_face_risk = max(face_content_risk(tags, None, None), float(candidate.get("face_content_risk") or 0.0))
    environment_terms = {
        "environment", "landscape", "nature", "architecture", "building", "road",
        "traffic", "aerial", "drone", "forest", "mountain", "ocean", "factory",
        "machine", "sky", "street", "city", "transport",
    }
    environment_priority = min(1.0, len(tag_tokens & environment_terms) / 2.0)
    human_theme = _human_focused_theme(_theme_terms(theme))
    face_penalty = metadata_face_risk * (0.28 if human_theme else 0.72)
    shot_type = str(candidate.get("shot_type") or _infer_shot_type(tags))
    spatial_scale_priority = {
        "aerial": 1.0,
        "wide": 0.85,
        "pov_tracking": 0.72,
        "medium": 0.22,
        "medium_portrait": 0.08,
        "close_up": 0.0,
    }.get(shot_type, 0.2)
    components = {
        "relevance": relevance,
        "resolution": resolution,
        "aspect": aspect,
        "popularity": popularity,
        "duration": duration_score,
        "motion_style": motion_style,
        "avoid_penalty": avoid_penalty,
        "face_content_risk": metadata_face_risk,
        "face_penalty": face_penalty,
        "environment_priority": environment_priority,
        "spatial_scale_priority": spatial_scale_priority,
        "dynamic_world_fit": float(visual_metadata["world_fit"]),
        "dynamic_profile_relevance": float(visual_metadata["relevance"]),
        "dynamic_profile_allowed": 1.0 if visual_metadata["allowed"] else 0.0,
    }
    score = (
        0.40 * relevance
        + 0.18 * resolution
        + 0.15 * aspect
        + 0.10 * popularity
        + 0.06 * duration_score
        + 0.11 * motion_style
        + 0.06 * environment_priority
        + 0.11 * float(visual_metadata["world_fit"])
        + 0.06 * float(visual_metadata["relevance"])
        - 0.22 * face_penalty
    )
    if prefer_wide_aerial:
        score += 0.18 * spatial_scale_priority + 0.07 * motion
    if not visual_metadata["allowed"]:
        score -= 0.55
    return float(score), {key: round(value, 4) for key, value in components.items()}


def _score_candidates(
    session: requests.Session,
    candidates: list[dict[str, Any]],
    theme: str,
    style_profile: Mapping[str, Any] | None,
    audio_profile: Mapping[str, Any] | None,
    cache_dir: Path,
    target_ratio: float,
    min_resolution: tuple[int, int],
    desired_count: int,
    prefer_wide_aerial: bool = False,
    visual_cohesion_profile: str = "none",
) -> list[dict[str, Any]]:
    for candidate in candidates:
        score, components = _metadata_score(
            candidate,
            theme,
            style_profile,
            audio_profile,
            target_ratio,
            min_resolution,
            prefer_wide_aerial,
            visual_cohesion_profile,
        )
        candidate["metadata_score"] = round(score, 6)
        candidate["score_components"] = components
    candidates.sort(key=lambda item: item["metadata_score"], reverse=True)
    visual_profile = _resolve_visual_profile(
        theme,
        style_profile,
        audio_profile,
        "" if visual_cohesion_profile in {"", "none", "auto"} else visual_cohesion_profile.replace("_", " "),
    )
    target_color = dict(visual_profile.get("color_profile") or _target_color(style_profile))
    inspect_count = min(len(candidates), max(24, desired_count * 8))
    for index, candidate in enumerate(candidates):
        local_entry = candidate.get("local_reuse_entry")
        if isinstance(local_entry, Mapping):
            local_quality = local_entry.get("quality") if isinstance(local_entry.get("quality"), Mapping) else {}
            local_fingerprint = local_entry.get("fingerprint") if isinstance(local_entry.get("fingerprint"), Mapping) else {}
            local_visual = local_quality.get("visual_analysis") if isinstance(local_quality.get("visual_analysis"), Mapping) else {}
            hashes = list(local_fingerprint.get("perceptual_hashes") or [])
            local_hsv = local_entry.get("mean_hsv") if isinstance(local_entry.get("mean_hsv"), Mapping) else local_quality.get("mean_hsv", {})
            try:
                local_color_score = _color_similarity(
                    float(local_hsv.get("hue_degrees")),
                    float(local_hsv.get("saturation")),
                    float(local_hsv.get("value")),
                    target_color,
                )
            except (AttributeError, TypeError, ValueError):
                local_color_score = float(local_quality.get("color_score", 0.65))
            thumbnail = {
                "available": True,
                "sharpness_score": float(local_quality.get("sharpness_score", 0.65)),
                "exposure_score": float(local_quality.get("exposure_score", 0.65)),
                "color_score": local_color_score,
                "text_watermark_risk": float(local_quality.get("text_watermark_risk", 0.0)),
                "perceptual_hash": hashes[0] if hashes else None,
                "subject_profile": local_quality.get("subject_profile", {}),
                "face_content_risk": float(local_quality.get("face_content_risk", candidate.get("face_content_risk", 0.0))),
                "aesthetic_metrics": {
                    "spatial_depth": float(local_visual.get("spatial_depth_score", 0.45)),
                    "composition_quality": float(local_visual.get("composition_quality_score", 0.45)),
                    "visual_impact": float(local_visual.get("visual_impact_score", 0.45)),
                    "lighting_quality": float(local_visual.get("lighting_quality_score", 0.45)),
                    "atmosphere_quality": float(local_visual.get("atmosphere_quality_score", 0.45)),
                    "color_quality": float(local_visual.get("intrinsic_color_quality_score", 0.45)),
                    "ordinary_travelogue_risk": float(local_visual.get("ordinary_travelogue_risk", 0.5)),
                },
            }
        elif index < inspect_count:
            thumbnail = _get_thumbnail_signals(session, candidate["raw"], candidate["variant"], cache_dir, target_color)
        else:
            thumbnail = _image_signals(np.empty((0, 0, 3), dtype=np.uint8), target_color)
        candidate["thumbnail_signals"] = thumbnail
        quality = 0.42 * float(thumbnail["sharpness_score"]) + 0.38 * float(thumbnail["exposure_score"]) + 0.20 * (1.0 - float(thumbnail["text_watermark_risk"]))
        thumbnail_aesthetic = thumbnail.get("aesthetic_metrics")
        if not isinstance(thumbnail_aesthetic, Mapping):
            thumbnail_aesthetic = frame_aesthetic_metrics(np.empty((0, 0, 3), dtype=np.uint8))
        preliminary_aesthetic = (
            0.24 * thumbnail_aesthetic["spatial_depth"]
            + 0.20 * thumbnail_aesthetic["composition_quality"]
            + 0.23 * thumbnail_aesthetic["visual_impact"]
            + 0.18 * thumbnail_aesthetic["lighting_quality"]
            + 0.15 * thumbnail_aesthetic["color_quality"]
            - 0.12 * thumbnail_aesthetic["ordinary_travelogue_risk"]
        )
        visual_face_risk = max(
            float(candidate["score_components"].get("face_content_risk", 0.0)),
            float(thumbnail.get("face_content_risk", 0.0)),
        )
        candidate["score_components"].update(
            {
                "thumbnail_quality": round(quality, 4),
                "color": round(float(thumbnail["color_score"]), 4),
                "visual_face_risk": round(visual_face_risk, 4),
                "preliminary_aesthetic": round(float(np.clip(preliminary_aesthetic, 0.0, 1.0)), 4),
            }
        )
        # Reweight metadata components after thumbnail inspection.
        historical_usage = int(
            (local_entry or {}).get("historical_usage_count", 0)
            if isinstance(local_entry, Mapping)
            else 0
        )
        history_penalty = min(0.12, math.log1p(max(0, historical_usage)) * 0.025)
        pre_score = (
            0.32 * candidate["score_components"]["relevance"]
            + 0.15 * candidate["score_components"]["resolution"]
            + 0.12 * candidate["score_components"]["aspect"]
            + 0.07 * candidate["score_components"]["popularity"]
            + 0.04 * candidate["score_components"]["duration"]
            + 0.09 * candidate["score_components"]["motion_style"]
            + 0.13 * quality
            + 0.08 * float(thumbnail["color_score"])
            + 0.12 * float(np.clip(preliminary_aesthetic, 0.0, 1.0))
            + 0.08 * candidate["score_components"]["dynamic_world_fit"]
            - 0.28 * visual_face_risk
            - history_penalty
        )
        candidate["pre_score"] = round(float(pre_score), 6)
        if not isinstance(local_entry, Mapping):
            candidate["shot_type"] = _infer_shot_type(str(candidate.get("tags") or ""))
            candidate["shot_scale"] = _infer_shot_scale(str(candidate.get("tags") or ""))
            candidate["motion_score_estimate"] = round(_motion_tag_score(str(candidate.get("tags") or "")), 4)
        candidate["face_content_risk"] = round(visual_face_risk, 4)
        if not isinstance(local_entry, Mapping):
            candidate["scene_category"] = _scene_category(str(candidate.get("tags") or ""), None)
        candidate["semantic_tags"] = _semantic_tags(
            candidate.get("tags"),
            candidate.get("scene_category"),
            candidate.get("shot_type"),
            candidate.get("shot_scale"),
            candidate.get("semantic_tags") or [],
        )
        candidate["canonical_source_id"] = candidate.get("canonical_source_id") or _canonical_source_id(
            local_entry if isinstance(local_entry, Mapping) else candidate,
            candidate.get("pixabay_id"),
        )
        candidate["historical_usage_count"] = historical_usage
        candidate["score_components"]["historical_usage_penalty"] = round(history_penalty, 4)
    return _diversified_order(candidates)


def _diversified_order(candidates: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    remaining = list(candidates)
    ordered: list[dict[str, Any]] = []
    shot_counts: Counter[str] = Counter()
    scene_counts: Counter[str] = Counter()
    author_counts: Counter[str] = Counter()
    query_counts: Counter[str] = Counter()
    seen_hashes: list[str] = []
    while remaining:
        best_index = 0
        best_adjusted = -99.0
        for index, item in enumerate(remaining):
            adjusted = float(item.get("pre_score", 0))
            scene = str(item.get("scene_category") or "general")
            # The final sufficiency gate requires multiple scene categories.
            # Give the first two non-human environment candidates in each
            # category enough room to survive full-video QA, then penalize
            # further repeats.  Food and people must not satisfy an unrelated
            # environment montage's diversity requirement merely by being a
            # new category.
            preferred_scenes = {
                "nature",
                "water_coast",
                "mountain_canyon",
                "forest_wilderness",
                "polar_ice",
                "sky_space",
                "architecture",
                "transport",
                "industrial",
                "technology",
            }
            scene_count = scene_counts[scene]
            if scene in preferred_scenes:
                if scene_count == 0:
                    adjusted += 0.12
                elif scene_count == 1:
                    adjusted += 0.05
                else:
                    adjusted -= 0.06 * (scene_count - 1)
            elif scene in {"food", "people"}:
                adjusted -= 0.18
            else:
                adjusted -= 0.04 * scene_count
            adjusted -= 0.055 * shot_counts[str(item.get("shot_type"))]
            adjusted -= 0.025 * author_counts[str(item.get("user") or "")]
            matched = list(item.get("matched_queries") or [])
            if matched:
                adjusted -= 0.012 * min(query_counts[query] for query in matched)
            phash = item.get("thumbnail_signals", {}).get("perceptual_hash")
            if phash and any(_hash_distance(phash, previous) <= 5 for previous in seen_hashes):
                adjusted -= 0.18
            if adjusted > best_adjusted:
                best_index, best_adjusted = index, adjusted
        chosen = remaining.pop(best_index)
        chosen["diversity_adjusted_score"] = round(best_adjusted, 6)
        ordered.append(chosen)
        scene_counts[str(chosen.get("scene_category") or "general")] += 1
        shot_counts[str(chosen.get("shot_type"))] += 1
        author_counts[str(chosen.get("user") or "")] += 1
        for query in chosen.get("matched_queries") or []:
            query_counts[query] += 1
        phash = chosen.get("thumbnail_signals", {}).get("perceptual_hash")
        if phash:
            seen_hashes.append(str(phash))
    return ordered


def _hash_distance(left: str, right: str) -> int:
    try:
        return int(imagehash.hex_to_hash(str(left)) - imagehash.hex_to_hash(str(right)))
    except (ValueError, TypeError):
        return 999


def _collect_candidates(
    session: requests.Session,
    api_key: str,
    theme: str,
    style_profile: Mapping[str, Any] | None,
    audio_profile: Mapping[str, Any] | None,
    cache_dir: Path,
    desired_count: int,
    min_resolution: tuple[int, int],
    local_entries: Sequence[Mapping[str, Any]] = (),
    timeline_plan: Any = None,
    candidate_pool_multiplier: int = 6,
    max_search_pages: int = 3,
    priority_queries: Sequence[str] = (),
    wide_aerial_only: bool = False,
    visual_cohesion_profile: str = "none",
    excluded_pixabay_ids: Sequence[str | int] = (),
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    candidates_by_id: dict[str, dict[str, Any]] = {}
    search_rounds: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    slots = _timeline_slots(timeline_plan)
    target_pool = (
        max(desired_count, len(slots) * max(1, int(candidate_pool_multiplier)))
        if slots
        else max(24, desired_count * max(8, int(candidate_pool_multiplier)))
    )
    per_page = min(100, max(20, desired_count * 5))
    max_search_pages = max(1, min(20, int(max_search_pages)))
    exact_queries = _unique_terms(
        _clean_query([query]) for query in priority_queries if str(query).strip()
    )
    excluded_ids = {str(value).strip() for value in excluded_pixabay_ids if str(value).strip()}
    executed_queries: set[str] = set()
    visual_request = "" if visual_cohesion_profile in {"", "none", "auto"} else str(visual_cohesion_profile).replace("_", " ")
    visual_profile = _resolve_visual_profile(theme, style_profile, audio_profile, visual_request)

    # Explicit location queries are an acquisition boundary. Do not seed this
    # pool with loosely matching generic rows from the local material library.
    local_candidates = [] if exact_queries else _local_library_candidates(local_entries, theme, slots)
    for candidate in local_candidates:
        candidates_by_id[str(candidate["pixabay_id"])] = candidate
    if local_candidates:
        search_rounds.append(
            {
                "round": 0,
                "expansion_level": -1,
                "reason": "local material index retrieval",
                "queries": [],
                "pool_before": 0,
                "pool_after": len(candidates_by_id),
                "new_unique_candidates": len(candidates_by_id),
                "target_pool": target_pool,
                "stop_reason": (
                    "local material index satisfied metadata pool"
                    if len(candidates_by_id) >= target_pool
                    else "local pool below target; continue with Pixabay search"
                ),
            }
        )
        if len(candidates_by_id) >= target_pool:
            return list(candidates_by_id.values()), search_rounds, errors

    round_count = max_search_pages if exact_queries else 3
    for round_index in range(round_count):
        # Always execute the precise first round.  Expansion rounds are bounded
        # and continue until the larger quality-oriented pool is satisfied.
        if len(candidates_by_id) >= target_pool and round_index >= 2:
            break
        query_plan = (
            [{"query": query, "intent": "explicit_user_query"} for query in exact_queries]
            if exact_queries
            else plan_visual_search_queries(visual_profile, round_index)
        )
        queries = [str(record.get("query") or "") for record in query_plan if record.get("query")]
        pages = (round_index + 1,) if exact_queries else range(1, max_search_pages + 1)
        round_record: dict[str, Any] = {
            "round": round_index + 1,
            "expansion_level": -2 if exact_queries else round_index,
            "reason": (
                "explicit priority location queries"
                if exact_queries
                else (
                    "initial style and music queries"
                    if round_index == 0
                    else "candidate pool below target; expanded synonyms, scenes, and visual concepts"
                )
            ),
            "queries": [],
            "query_intents": {str(record.get("query")): record.get("intent") for record in query_plan},
            "pool_before": len(candidates_by_id),
            "target_pool": target_pool,
        }
        for query in queries:
            query_key = re.sub(r"\s+", " ", query.strip().lower())
            if not exact_queries and query_key in executed_queries:
                continue
            for page in pages:
                query_record: dict[str, Any] = {
                    "query": query,
                    "query_length": len(query),
                    "page": page,
                }
                try:
                    payload, cache_hit = _pixabay_search(
                        session, api_key, query, cache_dir, per_page, page=page
                    )
                    hits = payload.get("hits") if isinstance(payload, Mapping) else []
                    if not isinstance(hits, list):
                        hits = []
                    added = 0
                    for raw in hits:
                        if not isinstance(raw, Mapping) or raw.get("id") is None:
                            continue
                        if str(raw.get("id")) in excluded_ids:
                            continue
                        tags = str(raw.get("tags") or "")
                        profile_fit = metadata_profile_fit(f"{tags} {query}", visual_profile)
                        if not profile_fit["allowed"]:
                            continue
                        if wide_aerial_only:
                            lowered_tags = tags.lower()
                            if any(
                                term in lowered_tags
                                for term in (
                                    "abstract", "animation", "animated", "cgi", "3d render",
                                    "rendering", "ai generated", "generative ai", "synthetic",
                                    "illustration", "cartoon", "wallpaper", "background",
                                    "macro", "close up", "close-up", "detail shot",
                                )
                            ):
                                continue
                            if re.search(
                                r"\b(?:man|woman|people|person|boy|girl|child|children|portrait|"
                                r"animal|wildlife|goat|dog|cat|bird|flag|parliament)\b",
                                lowered_tags,
                            ):
                                continue
                            if _infer_shot_type(tags) not in {"aerial", "wide", "pov_tracking"}:
                                continue
                        variant = _variant_for_hit(raw, min_resolution)
                        if variant is None:
                            continue
                        asset_id = str(raw["id"])
                        if asset_id not in candidates_by_id:
                            normalized_id: Any = int(raw["id"]) if str(raw["id"]).isdigit() else asset_id
                            candidates_by_id[asset_id] = {
                                "id": normalized_id,
                                "pixabay_id": normalized_id,
                                "page_url": str(raw.get("pageURL") or raw.get("page_url") or ""),
                                "tags": tags,
                                "duration": float(raw.get("duration") or 0),
                                "user": str(raw.get("user") or ""),
                                "user_id": raw.get("user_id"),
                                "views": int(raw.get("views") or 0),
                                "downloads": int(raw.get("downloads") or 0),
                                "likes": int(raw.get("likes") or 0),
                                "comments": int(raw.get("comments") or 0),
                                "variant": variant,
                                "matched_queries": [query],
                                "search_rounds": [round_index + 1],
                                "raw": dict(raw),
                                "canonical_source_id": f"pixabay:{normalized_id}",
                                "semantic_tags": _semantic_tags(tags),
                                "visual_metadata_fit": profile_fit,
                            }
                            added += 1
                        else:
                            existing = candidates_by_id[asset_id]
                            if query not in existing["matched_queries"]:
                                existing["matched_queries"].append(query)
                            if round_index + 1 not in existing["search_rounds"]:
                                existing["search_rounds"].append(round_index + 1)
                            if not existing.get("page_url"):
                                existing["page_url"] = str(raw.get("pageURL") or raw.get("page_url") or "")
                            if not existing.get("variant", {}).get("url"):
                                existing["variant"] = variant
                                existing["raw"] = dict(raw)
                    query_record.update(
                        {
                            "status": "ok",
                            "cache_hit": cache_hit,
                            "api_total_hits": int(payload.get("totalHits") or payload.get("total") or len(hits)),
                            "returned_hits": len(hits),
                            "new_unique_candidates": added,
                        }
                    )
                except PixabayPipelineError as exc:
                    safe = _safe_error(exc, api_key)
                    query_record.update({"status": "error", "error": safe, "cache_hit": False})
                    errors.append({"round": round_index + 1, "query": query, "page": page, "error": safe})
                round_record["queries"].append(query_record)
                if not exact_queries and round_index >= 1 and len(candidates_by_id) >= target_pool:
                    break
            if not exact_queries:
                executed_queries.add(query_key)
            if not exact_queries and round_index >= 1 and len(candidates_by_id) >= target_pool:
                break
        round_record["pool_after"] = len(candidates_by_id)
        round_record["new_unique_candidates"] = len(candidates_by_id) - round_record["pool_before"]
        search_rounds.append(round_record)
        round_record["stop_reason"] = (
            "metadata candidate pool target reached"
            if len(candidates_by_id) >= target_pool
            else (
                (
                    "candidate pool below target; continue exact location query pages"
                    if exact_queries and round_index < round_count - 1
                    else (
                        "maximum exact-query pages reached"
                        if exact_queries
                        else (
                            "candidate pool below target; expand synonyms, scenes, pages, and concepts"
                            if round_index < 2
                            else "maximum bounded expansion reached"
                        )
                    )
                )
            )
        )

    if not candidates_by_id and errors:
        raise PixabayPipelineError(f"All Pixabay searches failed; first error: {errors[0]['error']}")
    return list(candidates_by_id.values()), search_rounds, errors


def _find_media_executable(name: str) -> str:
    """Find FFmpeg tools on PATH or in the standard Windows winget location."""

    explicit = os.environ.get(f"{name.upper()}_BIN", "").strip()
    if explicit and Path(explicit).is_file():
        return explicit
    found = shutil.which(name)
    if found:
        return found
    local_app_data = Path(os.environ.get("LOCALAPPDATA", ""))
    if local_app_data.is_dir():
        package_root = local_app_data / "Microsoft" / "WinGet" / "Packages"
        matches = sorted(
            package_root.glob(f"Gyan.FFmpeg_*/*/bin/{name}.exe"),
            key=lambda item: item.stat().st_mtime if item.exists() else 0,
            reverse=True,
        )
        if matches:
            return str(matches[0])
    raise PixabayPipelineError(
        f"{name} was not found; add it to PATH or set {name.upper()}_BIN"
    )


def _ffprobe(path: Path) -> dict[str, Any]:
    command = [
        _find_media_executable("ffprobe"), "-v", "error", "-show_entries",
        "stream=index,codec_type,codec_name,width,height,r_frame_rate,avg_frame_rate,duration:format=duration,size,bit_rate",
        "-of", "json", str(path),
    ]
    try:
        process = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=45, check=False)
    except FileNotFoundError:
        raise PixabayPipelineError("ffprobe executable disappeared before media inspection") from None
    except subprocess.TimeoutExpired:
        raise PixabayPipelineError("ffprobe timed out while checking a downloaded video") from None
    if process.returncode != 0:
        raise PixabayPipelineError(f"ffprobe rejected downloaded video: {_safe_error(process.stderr)}")
    try:
        payload = json.loads(process.stdout)
    except ValueError:
        raise PixabayPipelineError("ffprobe returned invalid JSON") from None
    if not isinstance(payload, Mapping):
        raise PixabayPipelineError("ffprobe returned an unexpected response")
    return dict(payload)


def _sample_video_frames(path: Path, max_samples: int = 24) -> tuple[list[np.ndarray], float, float]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return [], 0.0, 0.0
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    duration = frame_count / fps if frame_count > 0 and fps > 0 else 0.0
    samples = max(3, min(max_samples, frame_count if frame_count > 0 else max_samples))
    positions = np.linspace(0, max(0, frame_count - 1), samples).astype(int) if frame_count > 0 else np.arange(samples) * max(1, int(fps or 25))
    frames: list[np.ndarray] = []
    for position in positions:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(position))
        ok, frame = capture.read()
        if ok and frame is not None and frame.size:
            max_side = max(frame.shape[:2])
            if max_side > 640:
                scale = 640.0 / max_side
                frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            frames.append(frame)
    capture.release()
    return frames, fps, duration


def _motion_and_stability(frames: Sequence[np.ndarray]) -> tuple[float, float, dict[str, Any]]:
    if len(frames) < 2:
        return 0.0, 0.5, {
            "mean_flow": 0.0,
            "flow_jitter": 0.0,
            "motion_direction": "unknown",
        }
    flows: list[float] = []
    translations: list[tuple[float, float]] = []
    previous = cv2.cvtColor(cv2.resize(frames[0], (320, 180)), cv2.COLOR_BGR2GRAY)
    for frame in frames[1:]:
        current = cv2.cvtColor(cv2.resize(frame, (320, 180)), cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(previous, current, None, 0.5, 3, 15, 2, 5, 1.1, 0)
        magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        flows.append(float(np.median(magnitude)))
        translations.append((float(np.median(flow[..., 0])), float(np.median(flow[..., 1]))))
        previous = current
    mean_flow = float(np.mean(flows)) if flows else 0.0
    motion_score = float(np.clip(math.log1p(mean_flow) / math.log(8.0), 0.0, 1.0))
    if len(translations) >= 3:
        array = np.asarray(translations, dtype=np.float32)
        acceleration = np.diff(array, axis=0)
        jitter = float(np.mean(np.linalg.norm(acceleration, axis=1)))
    else:
        jitter = 0.0
    stability_score = float(np.clip(math.exp(-jitter / 2.8), 0.0, 1.0))
    direction = "static"
    median_dx = median_dy = consistency = 0.0
    if translations:
        vectors = np.asarray(translations, dtype=np.float32)
        median_dx = float(np.median(vectors[:, 0]))
        median_dy = float(np.median(vectors[:, 1]))
        magnitude = math.hypot(median_dx, median_dy)
        if magnitude >= 0.08:
            horizontal = abs(median_dx) >= abs(median_dy)
            if horizontal:
                direction = "right" if median_dx > 0 else "left"
                signs = np.sign(vectors[:, 0])
                target_sign = 1 if median_dx > 0 else -1
            else:
                direction = "down" if median_dy > 0 else "up"
                signs = np.sign(vectors[:, 1])
                target_sign = 1 if median_dy > 0 else -1
            consistency = float(np.mean(signs == target_sign))
            if consistency < 0.55:
                direction = "mixed"
    return motion_score, stability_score, {
        "mean_flow": round(mean_flow, 4),
        "flow_jitter": round(jitter, 4),
        "median_dx": round(median_dx, 4),
        "median_dy": round(median_dy, 4),
        "direction_consistency": round(consistency, 4),
        "motion_direction": direction,
    }


def _usable_segments_from_samples(
    frames: Sequence[np.ndarray],
    frame_signals: Sequence[Mapping[str, Any]],
    duration: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Find stable continuous windows and reject black/frozen/head-tail spans."""

    count = min(len(frames), len(frame_signals))
    if count < 2 or duration <= 0:
        return [], {
            "black_frame_ratio": 1.0,
            "freeze_frame_ratio": 1.0,
            "head_tail_penalty": 1.0,
            "sample_count": count,
        }
    times = np.linspace(0.0, duration, count)
    step = duration / max(1, count - 1)
    margin = min(0.65, duration * 0.045) if duration >= 2.0 else 0.0
    black: list[bool] = []
    frozen: list[bool] = [False]
    for index in range(count):
        frame = frames[index]
        gray = cv2.cvtColor(cv2.resize(frame, (192, 108)), cv2.COLOR_BGR2GRAY)
        mean_luma = float(frame_signals[index].get("mean_luma", float(gray.mean())))
        dark_share = float(np.mean(gray <= 8))
        black.append(mean_luma <= 11.0 or dark_share >= 0.94)
        if index:
            previous = cv2.cvtColor(cv2.resize(frames[index - 1], (192, 108)), cv2.COLOR_BGR2GRAY)
            difference = float(np.mean(cv2.absdiff(previous, gray)))
            frozen.append(difference <= 0.45)

    good: list[bool] = []
    for index, signal in enumerate(frame_signals[:count]):
        time_value = float(times[index])
        inside_safe_body = margin - 1e-6 <= time_value <= duration - margin + 1e-6
        good.append(
            inside_safe_body
            and not black[index]
            and not frozen[index]
            and float(signal.get("sharpness_score", 0.5)) >= 0.16
            and float(signal.get("exposure_score", 0.5)) >= 0.14
        )

    groups: list[tuple[int, int]] = []
    start: int | None = None
    for index, usable in enumerate(good + [False]):
        if usable and start is None:
            start = index
        elif not usable and start is not None:
            groups.append((start, index - 1))
            start = None

    segments: list[dict[str, Any]] = []
    for first, last in groups:
        segment_start = max(margin, float(times[first]) - step * 0.48)
        segment_end = min(duration - margin, float(times[last]) + step * 0.48)
        segment_duration = segment_end - segment_start
        if segment_duration < 0.75:
            continue
        segment_signals = frame_signals[first : last + 1]
        motion, stability, motion_raw = _motion_and_stability(frames[first : last + 1])
        sharpness = float(np.mean([float(item.get("sharpness_score", 0.5)) for item in segment_signals]))
        exposure = float(np.mean([float(item.get("exposure_score", 0.5)) for item in segment_signals]))
        variation = float(np.mean([not frozen[index] for index in range(first, last + 1)]))
        score = 0.30 * sharpness + 0.27 * exposure + 0.25 * stability + 0.10 * variation + 0.08 * min(1.0, motion + 0.25)
        segments.append(
            {
                "start": round(segment_start, 4),
                "end": round(segment_end, 4),
                "duration": round(segment_duration, 4),
                "score": round(float(score), 4),
                "motion_score": round(motion, 4),
                "motion_direction": motion_raw.get("motion_direction", "unknown"),
                "stability_score": round(stability, 4),
                "black_frame_ratio": round(sum(black[first : last + 1]) / max(1, last - first + 1), 4),
                "freeze_frame_ratio": round(sum(frozen[first : last + 1]) / max(1, last - first + 1), 4),
                "analysis_status": "sampled_continuous_window",
            }
        )
    segments.sort(key=lambda item: (float(item["score"]), float(item["duration"])), reverse=True)
    head_tail_indices = [
        index
        for index, time_value in enumerate(times)
        if time_value < margin or time_value > duration - margin
    ]
    head_tail_bad = sum(black[index] or frozen[index] for index in head_tail_indices)
    summary = {
        "black_frame_ratio": round(sum(black) / count, 4),
        "freeze_frame_ratio": round(sum(frozen) / count, 4),
        "head_tail_penalty": round(head_tail_bad / max(1, len(head_tail_indices)), 4),
        "sample_count": count,
        "sample_interval_seconds": round(step, 4),
    }
    return segments, summary


def _watermark_risk(frames: Sequence[np.ndarray]) -> float:
    if not frames:
        return 0.5
    edge_maps: list[np.ndarray] = []
    for frame in frames[:16]:
        gray = cv2.cvtColor(cv2.resize(frame, (320, 180)), cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200) > 0
        edge_maps.append(edges)
    persistence = np.mean(np.stack(edge_maps, axis=0), axis=0)
    h, w = persistence.shape
    border = np.zeros((h, w), dtype=bool)
    border[: h // 4, :] = True
    border[-h // 4 :, :] = True
    border[:, : w // 5] = True
    border[:, -w // 5 :] = True
    persistent_border = float(np.mean(persistence[border] >= 0.72))
    persistent_center = float(np.mean(persistence[~border] >= 0.72)) if np.any(~border) else 0.0
    # Repeated high-contrast shapes near an edge are a useful but intentionally
    # conservative text/logo heuristic; it does not perform OCR.
    return float(np.clip((persistent_border - persistent_center * 0.55) * 14.0, 0.0, 1.0))


def _video_fingerprint(path: Path, frames: Sequence[np.ndarray], duration: float, width: int, height: int) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    phashes: list[str] = []
    if frames:
        indices = sorted(set((0, len(frames) // 4, len(frames) // 2, (3 * len(frames)) // 4, len(frames) - 1)))
        for index in indices:
            rgb = cv2.cvtColor(frames[index], cv2.COLOR_BGR2RGB)
            phashes.append(str(imagehash.phash(Image.fromarray(rgb))))
    return {
        "sha256": digest.hexdigest(),
        "perceptual_hashes": phashes,
        "duration_seconds": round(float(duration), 4),
        "width": int(width),
        "height": int(height),
        "size_bytes": int(path.stat().st_size),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprint_duplicate(fingerprint: Mapping[str, Any], entries: Sequence[Mapping[str, Any]]) -> tuple[bool, str | None, float | None]:
    sha256 = str(fingerprint.get("sha256") or "")
    hashes = [str(value) for value in fingerprint.get("perceptual_hashes") or []]
    duration = float(fingerprint.get("duration_seconds") or 0.0)
    for entry in entries:
        previous = entry.get("fingerprint") if isinstance(entry.get("fingerprint"), Mapping) else entry
        if sha256 and sha256 == str(previous.get("sha256") or ""):
            return True, str(entry.get("pixabay_id") or entry.get("local_path") or "exact_sha256"), 0.0
        previous_hashes = [str(value) for value in previous.get("perceptual_hashes") or []]
        previous_duration = float(previous.get("duration_seconds") or 0.0)
        if not hashes or not previous_hashes or duration <= 0 or previous_duration <= 0:
            continue
        duration_delta = abs(duration - previous_duration) / max(duration, previous_duration)
        if duration_delta > 0.12:
            continue
        count = min(len(hashes), len(previous_hashes))
        distances = [_hash_distance(hashes[index], previous_hashes[index]) for index in range(count)]
        median_distance = float(np.median(distances)) if distances else 999.0
        if median_distance <= 7.0:
            return True, str(entry.get("pixabay_id") or entry.get("local_path") or "perceptual"), median_distance
    return False, None, None


def _video_quality(
    path: Path,
    style_profile: Mapping[str, Any] | None,
    min_resolution: tuple[int, int],
    tags: str = "",
    human_focused: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    probe = _ffprobe(path)
    video_stream = next(
        (stream for stream in probe.get("streams", []) if isinstance(stream, Mapping) and stream.get("codec_type") == "video"),
        None,
    )
    if not isinstance(video_stream, Mapping):
        raise PixabayPipelineError("Downloaded asset has no video stream")
    width = int(video_stream.get("width") or 0)
    height = int(video_stream.get("height") or 0)
    duration = float(video_stream.get("duration") or probe.get("format", {}).get("duration") or 0.0)
    # Forty-eight distributed frames are enough for stable full-clip signals
    # while keeping one-time v1.3 cache upgrades practical on long 4K sources.
    frames, fps, decoded_duration = _sample_video_frames(path, max_samples=48)
    if duration <= 0:
        duration = decoded_duration
    if len(frames) < 2:
        raise PixabayPipelineError("Downloaded asset could not be decoded into enough frames")
    style_payload = _style_payload(style_profile)
    embedded_visual_profile = style_payload.get("visual_style_profile")
    visual_profile = dict(embedded_visual_profile) if isinstance(embedded_visual_profile, Mapping) else {}
    target_color = dict(visual_profile.get("color_profile") or _target_color(style_profile))
    per_frame = [_image_signals(frame, target_color) for frame in frames]
    sharpness = float(np.mean([item["sharpness_score"] for item in per_frame]))
    exposure = float(np.mean([item["exposure_score"] for item in per_frame]))
    color = float(np.mean([item["color_score"] for item in per_frame]))
    motion, stability, motion_raw = _motion_and_stability(frames)
    direction = str(motion_raw.get("motion_direction") or "unknown")
    motion_type = {
        "left": "lateral_left",
        "right": "lateral_right",
        "up": "rise",
        "down": "dive",
        "static": "static",
        "mixed": "mixed",
    }.get(direction, "unknown")
    usable_segments, segment_summary = _usable_segments_from_samples(frames, per_frame, duration)
    watermark = _watermark_risk(frames)
    subject_profile = aggregate_subject_regions(
        [item.get("subject_profile", {}) for item in per_frame]
    )
    # aggregate_subject_regions expects raw per-frame face_count fields, which
    # are present in each subject profile returned by _image_signals.
    full_face_risk = face_content_risk(tags, None, subject_profile)
    min_long, min_short = sorted(min_resolution, reverse=True)
    long_side, short_side = sorted((width, height), reverse=True)
    resolution_score = float(np.clip(min(long_side / max(1, min_long), short_side / max(1, min_short)), 0.0, 1.0))
    visual_analysis = aggregate_video_aesthetics(
        [item.get("aesthetic_metrics", {}) for item in per_frame],
        sharpness=sharpness,
        motion_score=motion,
        stability_score=stability,
        motion_type=motion_type,
        resolution_score=resolution_score,
    )
    overall = (
        0.23 * sharpness
        + 0.20 * exposure
        + 0.15 * stability
        + 0.13 * color
        + 0.13 * resolution_score
        + 0.08 * (1.0 - watermark)
        + 0.08 * (0.55 + 0.45 * motion)
        + 0.20 * float(visual_analysis["aesthetic_score"])
        + 0.08 * float(visual_analysis["cinematic_score"])
        - (0.05 if human_focused else 0.18) * full_face_risk
        - 0.10 * float(segment_summary.get("black_frame_ratio", 0.0))
        - 0.08 * float(segment_summary.get("freeze_frame_ratio", 0.0))
    )
    reasons: list[str] = []
    if long_side < min_long or short_side < min_short:
        reasons.append(f"resolution below {min_resolution[0]}x{min_resolution[1]}")
    if duration < 0.8:
        reasons.append("duration below 0.8 seconds")
    if sharpness < 0.22:
        reasons.append("low sharpness")
    if exposure < 0.22:
        reasons.append("severe under/over exposure")
    if stability < 0.12:
        reasons.append("strongly unstable motion")
    if watermark > 0.86:
        reasons.append("persistent edge text/logo heuristic triggered")
    if overall < 0.42:
        reasons.append("overall quality score below threshold")
    aesthetic_floor = float((visual_profile.get("quality") or {}).get("aesthetic_floor", 0.40))
    cinematic_floor = float((visual_profile.get("quality") or {}).get("cinematic_floor", 0.36))
    if float(visual_analysis["aesthetic_score"]) < aesthetic_floor:
        reasons.append(f"aesthetic score below dynamic floor {aesthetic_floor:.2f}")
    if float(visual_analysis["cinematic_score"]) < cinematic_floor:
        reasons.append(f"cinematic score below dynamic floor {cinematic_floor:.2f}")
    if full_face_risk >= 0.94 and not human_focused:
        reasons.append("prominent frontal-face content exceeds default policy")
    if float(segment_summary.get("black_frame_ratio", 0.0)) > 0.35:
        reasons.append("long black-frame regions")
    if float(segment_summary.get("freeze_frame_ratio", 0.0)) > 0.72:
        reasons.append("long frozen/static-frame regions")
    if not usable_segments:
        reasons.append("no usable continuous segment")
    quality = {
        "passed": not reasons,
        "rejection_reasons": reasons,
        "overall_score": round(float(overall), 4),
        "resolution_score": round(resolution_score, 4),
        "sharpness_score": round(sharpness, 4),
        "exposure_score": round(exposure, 4),
        "stability_score": round(stability, 4),
        "text_watermark_risk": round(watermark, 4),
        "motion_score": round(motion, 4),
        "color_score": round(color, 4),
        "mean_hsv": per_frame[len(per_frame) // 2].get("mean_hsv", {}),
        "motion_signals": motion_raw,
        "motion_direction": motion_raw.get("motion_direction", "unknown"),
        "motion_type": motion_type,
        "visual_analysis": visual_analysis,
        "usable_segments": usable_segments,
        "segment_analysis_status": "sampled_continuous_windows",
        **segment_summary,
        "sampled_frames": len(frames),
        "subject_profile": subject_profile,
        "face_content_risk": round(full_face_risk, 4),
        "scene_category": _scene_category(tags, None),
        "heuristic_notice": "Signal-derived aesthetic estimates; not a human guarantee and text/logo detection is not OCR.",
    }
    fingerprint = _video_fingerprint(path, frames, duration, width, height)
    quality["analysis_cache"] = {
        "schema_version": ASSET_ANALYSIS_SCHEMA_VERSION,
        "engine_version": VISUAL_ENGINE_VERSION,
        "file_sha256": fingerprint["sha256"],
    }
    quality["visual_features"] = asset_visual_features(
        {
            "tags": tags,
            "shot_scale": _infer_shot_scale(tags),
            "motion_direction": quality["motion_direction"],
            "quality": quality,
        }
    )
    quality["content_semantics"] = _semantic_tags(
        tags,
        quality.get("scene_category"),
        _infer_shot_type(tags),
        _infer_shot_scale(tags),
    )
    media = {
        "width": width,
        "height": height,
        "duration_seconds": round(duration, 4),
        "fps": round(fps, 4),
        "codec": str(video_stream.get("codec_name") or ""),
        "size_bytes": int(path.stat().st_size),
        "fingerprint": fingerprint,
        "ratio": _ratio_label(width, height),
        "usable_segments": usable_segments,
    }
    return quality, media


def _download_video(session: requests.Session, url: str, destination: Path, secret: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with session.get(url, timeout=DOWNLOAD_TIMEOUT, stream=True) as response:
            if response.status_code != 200:
                raise PixabayPipelineError(f"Video download returned HTTP {response.status_code}")
            with destination.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
        if not destination.is_file() or destination.stat().st_size < 64 * 1024:
            raise PixabayPipelineError("Downloaded video is unexpectedly small")
    except PixabayPipelineError:
        destination.unlink(missing_ok=True)
        raise
    except (requests.RequestException, OSError) as exc:
        destination.unlink(missing_ok=True)
        raise PixabayPipelineError(f"Video download failed: {_safe_error(exc, secret)}") from None


_TAG_CHINESE: tuple[tuple[str, str], ...] = (
    ("aerial", "航拍"), ("drone", "航拍"), ("ocean", "海洋"), ("sea", "大海"),
    ("wave", "海浪"), ("beach", "沙滩"), ("coast", "海岸"), ("sunset", "日落"),
    ("sunrise", "日出"), ("mountain", "山峦"), ("forest", "森林"), ("river", "河流"),
    ("waterfall", "瀑布"), ("city", "城市"), ("street", "街道"), ("building", "建筑"),
    ("factory", "工厂"), ("industry", "工业"), ("machine", "机械"), ("welding", "焊接"),
    ("worker", "工人"), ("technology", "科技"), ("digital", "数字科技"), ("car", "汽车"),
    ("road", "道路"), ("train", "列车"), ("airplane", "飞机"), ("woman", "女性"),
    ("man", "男性"), ("people", "人群"), ("family", "家庭"), ("child", "儿童"),
    ("food", "美食"), ("cooking", "烹饪"), ("coffee", "咖啡"), ("sport", "运动"),
    ("football", "足球"), ("basketball", "篮球"), ("animal", "动物"), ("wildlife", "野生动物"),
    ("flower", "花朵"), ("rain", "雨景"), ("snow", "雪景"), ("night", "夜景"),
    ("cloud", "云层"), ("sky", "天空"), ("close up", "特写"), ("macro", "微距"),
    ("slow motion", "慢动作"), ("landscape", "风景"), ("nature", "自然"),
)


def _safe_chinese_slug(value: str, fallback: str = "主题") -> str:
    value = unicodedata.normalize("NFKC", str(value)).strip()
    value = re.sub(r"[<>:\"/\\|?*\x00-\x1f]", "_", value)
    value = re.sub(r"\s+", "_", value)
    value = re.sub(r"_+", "_", value).strip(" ._")
    if not value:
        value = fallback
    reserved = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}
    if value.upper() in reserved:
        value = f"{fallback}_{value}"
    return value[:48].rstrip(" ._") or fallback


def material_theme_directory(material_root: str | os.PathLike[str], theme: str) -> Path:
    """Resolve the exact directory that will receive downloaded theme assets."""

    return Path(material_root).expanduser().resolve() / _safe_chinese_slug(theme, "未命名主题")


def _chinese_description(tags: str, theme: str) -> str:
    text = tags.lower()
    labels: list[str] = []
    for english, chinese in _TAG_CHINESE:
        if english in text and chinese not in labels:
            labels.append(chinese)
        if len(labels) >= 3:
            break
    if not labels:
        chinese_theme = "".join(re.findall(r"[\u3400-\u9fff]+", theme))
        if chinese_theme:
            labels.append(chinese_theme[:12])
        else:
            labels.append("主题画面")
    return _safe_chinese_slug("_".join(labels), "主题画面")


def _extension_from_url(url: str) -> str:
    suffix = Path(urlparse(url).path).suffix.lower()
    return suffix if suffix in {".mp4", ".mov", ".webm", ".m4v"} else ".mp4"


def _legacy_usable_segments(duration: Any) -> list[dict[str, Any]]:
    try:
        duration_value = float(duration)
    except (TypeError, ValueError):
        return []
    if duration_value < 0.8:
        return []
    margin = min(0.45, max(0.0, duration_value * 0.04))
    start = margin if duration_value - 2 * margin >= 0.8 else 0.0
    end = duration_value - margin if duration_value - 2 * margin >= 0.8 else duration_value
    return [
        {
            "start": round(start, 4),
            "end": round(end, 4),
            "duration": round(end - start, 4),
            "score": 0.5,
            "motion_direction": "unknown",
            "analysis_status": "legacy_safe_fallback",
        }
    ]


def _normalize_library_entry(raw: Mapping[str, Any]) -> dict[str, Any]:
    entry = dict(raw)
    fingerprint = dict(entry.get("fingerprint") or {}) if isinstance(entry.get("fingerprint"), Mapping) else {}
    media = dict(entry.get("media") or {}) if isinstance(entry.get("media"), Mapping) else {}
    quality = dict(entry.get("quality") or {}) if isinstance(entry.get("quality"), Mapping) else {}
    local_text = str(entry.get("local_path") or "").strip()
    local_path = Path(local_text).expanduser() if local_text else None
    available = bool(local_path and local_path.is_file())
    width = int(media.get("width") or fingerprint.get("width") or entry.get("width") or 0)
    height = int(media.get("height") or fingerprint.get("height") or entry.get("height") or 0)
    duration = float(
        media.get("duration_seconds")
        or fingerprint.get("duration_seconds")
        or entry.get("duration_seconds")
        or entry.get("duration")
        or 0.0
    )
    history = [dict(item) for item in entry.get("usage_history", []) if isinstance(item, Mapping)]
    legacy_intervals = entry.get("usage_intervals") or entry.get("actual_usage_intervals") or []
    if not quality.get("usable_segments") and available:
        quality["usable_segments"] = _legacy_usable_segments(duration)
        quality.setdefault("segment_analysis_status", "legacy_safe_fallback")
    quality.setdefault("motion_direction", entry.get("motion_direction") or "unknown")
    scene = quality.get("scene_category") or entry.get("scene_category") or _scene_category(str(entry.get("tags") or ""), None)
    semantic_tags = list(entry.get("semantic_tags") or _semantic_tags(entry.get("tags"), scene, entry.get("shot_type"), entry.get("shot_scale")))
    entry.update(
        {
            "canonical_source_id": _canonical_source_id(entry),
            "download_url": str(entry.get("download_url") or ""),
            "file_hash": str(entry.get("file_hash") or fingerprint.get("sha256") or ""),
            "ratio": entry.get("ratio") or _ratio_label(width, height),
            "semantic_tags": semantic_tags,
            "download_status": entry.get("download_status") or ("cached" if available else "missing"),
            "available": available,
            "failure_reason": (
                None
                if available
                else entry.get("failure_reason") or ("local file missing" if local_text else "no local file recorded")
            ),
            "historical_usage_count": max(
                int(entry.get("historical_usage_count") or 0),
                _history_count(history, legacy_intervals),
            ),
            "usage_history": history,
            "quality": quality,
            "media": media,
            "fingerprint": fingerprint,
            "scene_category": scene,
            "motion_direction": quality.get("motion_direction", "unknown"),
        }
    )
    return entry


def _load_fingerprint_library(path: Path, *, strict: bool = False) -> dict[str, Any]:
    payload = _read_json_strict(path, {}) if strict else _read_json(path, {})
    if not isinstance(payload, Mapping):
        if strict:
            raise PixabayPipelineError(f"Material catalog has an invalid root object: {path}")
        payload = {}
    entries = payload.get("entries")
    if not isinstance(entries, list):
        if strict and payload:
            raise PixabayPipelineError(f"Material catalog has an invalid entries array: {path}")
        entries = []
    # Missing rows remain in the catalog with an explicit unavailable state so
    # failed paths are traceable and can later be repaired or re-downloaded.
    cleaned = [_normalize_library_entry(entry) for entry in entries if isinstance(entry, Mapping)]
    return {"schema_version": SCHEMA_VERSION, "updated_at": _utc_now(), "entries": cleaned}


def _library_entry_key(entry: Mapping[str, Any]) -> str:
    """Return a stable identity without treating two Pixabay IDs as one item."""

    pixabay_id = entry.get("pixabay_id")
    if pixabay_id not in (None, ""):
        return f"pixabay:{pixabay_id}"
    fingerprint = entry.get("fingerprint") if isinstance(entry.get("fingerprint"), Mapping) else entry
    sha256 = str(fingerprint.get("sha256") or "")
    if sha256:
        return f"sha256:{sha256}"
    return f"path:{Path(str(entry.get('local_path') or '')).expanduser()}"


def _merge_usage_history(*histories: Any) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for history in histories:
        if not isinstance(history, list):
            continue
        for raw in history:
            if not isinstance(raw, Mapping):
                continue
            item = dict(raw)
            identity = str(item.get("event_id") or "")
            if not identity:
                identity = hashlib.sha256(
                    json.dumps(item, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
                ).hexdigest()[:20]
                item["event_id"] = identity
            if identity in seen:
                continue
            seen.add(identity)
            merged.append(item)
    merged.sort(key=lambda item: (str(item.get("recorded_at") or ""), str(item.get("event_id") or "")))
    return merged


def _merge_library_entries(
    sources: Sequence[tuple[str, Sequence[Mapping[str, Any]]]],
) -> list[dict[str, Any]]:
    """Merge v1.0/project/global catalogs, preferring a usable local file."""

    merged: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for origin, entries in sources:
        for raw in entries:
            if not isinstance(raw, Mapping):
                continue
            entry = _normalize_library_entry(raw)
            entry.setdefault("library_origin", origin)
            key = _library_entry_key(entry)
            previous = merged.get(key)
            if previous is None:
                merged[key] = entry
                order.append(key)
                continue
            previous_path = Path(str(previous.get("local_path") or ""))
            candidate_path = Path(str(entry.get("local_path") or ""))
            previous_exists = bool(previous.get("local_path")) and previous_path.is_file()
            candidate_exists = bool(entry.get("local_path")) and candidate_path.is_file()
            if candidate_exists and not previous_exists:
                replacement = entry
                for field, value in previous.items():
                    replacement.setdefault(field, value)
                replacement["usage_history"] = _merge_usage_history(
                    previous.get("usage_history"), entry.get("usage_history")
                )
                replacement["historical_usage_count"] = _history_count(replacement["usage_history"])
                merged[key] = replacement
            else:
                for field, value in entry.items():
                    if previous.get(field) in (None, "", [], {}):
                        previous[field] = value
                previous["usage_history"] = _merge_usage_history(
                    previous.get("usage_history"), entry.get("usage_history")
                )
                previous["historical_usage_count"] = max(
                    int(previous.get("historical_usage_count") or 0),
                    int(entry.get("historical_usage_count") or 0),
                    _history_count(previous["usage_history"]),
                )
    return [merged[key] for key in order]


def _persist_material_libraries(
    paths: Sequence[Path],
    entries: Sequence[Mapping[str, Any]],
    secret: str,
) -> None:
    """Atomically update every catalog while retaining entries from other runs."""

    seen: set[Path] = set()
    working = [dict(entry) for entry in entries if isinstance(entry, Mapping)]
    for raw_path in paths:
        path = raw_path.expanduser().resolve()
        if path in seen:
            continue
        seen.add(path)
        with _exclusive_catalog_lock(path):
            # Re-read only after acquiring the lock so another project cannot
            # be overwritten by a stale pre-lock snapshot.
            current = _load_fingerprint_library(path, strict=True)
            working = _merge_library_entries(
                (
                    ("current_run", working),
                    (str(path), current.get("entries", [])),
                )
            )
            _atomic_write_json(
                path,
                {
                    "schema_version": SCHEMA_VERSION,
                    "updated_at": _utc_now(),
                    "entries": working,
                },
                secret,
            )


def _reload_material_entries(paths: Sequence[Path]) -> list[dict[str, Any]]:
    """Re-read current catalog state and fail closed on damaged JSON."""

    sources: list[tuple[str, Sequence[Mapping[str, Any]]]] = []
    seen: set[Path] = set()
    for raw_path in paths:
        path = raw_path.expanduser().resolve()
        if path in seen:
            continue
        seen.add(path)
        library = _load_fingerprint_library(path, strict=True)
        sources.append((str(path), library.get("entries", [])))
    return _merge_library_entries(sources)


def _asset_locked_candidates(
    candidates: Sequence[Mapping[str, Any]],
    global_material_index: Path,
    library_paths: Sequence[Path],
) -> Iterable[tuple[Mapping[str, Any], list[dict[str, Any]]]]:
    """Yield each candidate while its ID lock remains held by the consumer.

    The generator resumes (and therefore releases the current lock) only when
    the caller advances to the next candidate.  The post-lock catalog reload is
    the decisive cache check that prevents two projects which started from the
    same stale snapshot from downloading the same Pixabay ID twice.
    """

    for candidate in candidates:
        asset_id = str(candidate.get("pixabay_id", candidate.get("id", "unknown")))
        with _exclusive_asset_lock(global_material_index, asset_id):
            yield candidate, _reload_material_entries(library_paths)


def _same_file(left: Path, right: Path) -> bool:
    try:
        return left.samefile(right)
    except OSError:
        return False


def _reuse_local_path(
    known_path: Path,
    theme_dir: Path,
    candidate: Mapping[str, Any],
    sequence: int,
    theme: str,
) -> tuple[Path, str]:
    """Reuse a prior download via an NTFS/POSIX hard link or direct reference."""

    known_path = known_path.expanduser().resolve()
    theme_dir = theme_dir.expanduser().resolve()
    if known_path.parent == theme_dir:
        return known_path, "same_theme_reference"

    asset_id = str(candidate.get("pixabay_id") or candidate.get("id") or "asset")
    extension = known_path.suffix.lower() or ".mp4"
    # Reuse a hard link made by a previous run even if ranking changed its
    # sequence number; never create a second link for the same inode.
    for existing in sorted(theme_dir.glob(f"*_{asset_id}{extension}")):
        if existing.is_file() and _same_file(existing, known_path):
            return existing.resolve(), "hardlink_existing"

    description = _chinese_description(str(candidate.get("tags") or ""), theme)
    target = theme_dir / (
        _safe_chinese_slug(f"{sequence:02d}_{description}_{asset_id}", f"asset_{asset_id}")
        + extension
    )
    if target.exists():
        if _same_file(target, known_path):
            return target.resolve(), "hardlink_existing"
        # A deterministic collision must not overwrite unrelated user data.
        target = theme_dir / (
            _safe_chinese_slug(f"{sequence:02d}_{description}_reuse_{asset_id}", f"asset_reuse_{asset_id}")
            + extension
        )
        if target.exists():
            if _same_file(target, known_path):
                return target.resolve(), "hardlink_existing"
            return known_path, "shared_reference"
    try:
        os.link(known_path, target)
        return target.resolve(), "hardlink"
    except OSError:
        # Cross-volume links and restricted filesystems are expected fallbacks.
        # Referencing the already downloaded immutable stock clip is preferable
        # to copying it or downloading it again.
        return known_path, "shared_reference"


def _public_candidate(candidate: Mapping[str, Any], decision: str = "ranked") -> dict[str, Any]:
    variant = candidate.get("variant") if isinstance(candidate.get("variant"), Mapping) else {}
    local_entry = candidate.get("local_reuse_entry") if isinstance(candidate.get("local_reuse_entry"), Mapping) else {}
    fingerprint = local_entry.get("fingerprint") if isinstance(local_entry.get("fingerprint"), Mapping) else {}
    width = variant.get("width", 0)
    height = variant.get("height", 0)
    return {
        "pixabay_id": candidate.get("pixabay_id") or candidate.get("id"),
        "author": candidate.get("user") or "",
        "page_url": candidate.get("page_url") or "",
        "tags": candidate.get("tags") or "",
        "duration_seconds": candidate.get("duration") or 0,
        "search_queries": list(candidate.get("matched_queries") or []),
        "search_rounds": list(candidate.get("search_rounds") or []),
        "resolution": {"width": width, "height": height},
        "ratio": _ratio_label(width, height),
        "variant": variant.get("name"),
        "download_url": variant.get("url") or local_entry.get("download_url") or "",
        "canonical_source_id": candidate.get("canonical_source_id") or _canonical_source_id(local_entry or candidate),
        "file_hash": local_entry.get("file_hash") or fingerprint.get("sha256") or "",
        "semantic_tags": list(candidate.get("semantic_tags") or []),
        "pre_score": candidate.get("pre_score", 0),
        "diversity_adjusted_score": candidate.get("diversity_adjusted_score", 0),
        "score_components": candidate.get("score_components", {}),
        "thumbnail_signals": candidate.get("thumbnail_signals", {}),
        "shot_type": candidate.get("shot_type") or "medium",
        "shot_scale": candidate.get("shot_scale") or "medium",
        "motion_score_estimate": candidate.get("motion_score_estimate", 0.5),
        "scene_category": candidate.get("scene_category") or _scene_category(str(candidate.get("tags") or ""), None),
        "face_content_risk": candidate.get("face_content_risk", 0.0),
        "download_status": "cached" if local_entry.get("available") else "not_downloaded",
        "available": bool(local_entry.get("available")),
        "failure_reason": local_entry.get("failure_reason"),
        "historical_usage_count": int(candidate.get("historical_usage_count") or local_entry.get("historical_usage_count") or 0),
        "usage_history": list(local_entry.get("usage_history") or []),
        "decision": decision,
    }


def _complete_asset_record(
    source: Mapping[str, Any],
    candidate: Mapping[str, Any] | None = None,
    library_entry: Mapping[str, Any] | None = None,
    download_status: str | None = None,
) -> dict[str, Any]:
    record = dict(source)
    candidate = candidate if isinstance(candidate, Mapping) else {}
    library_entry = _normalize_library_entry(library_entry) if isinstance(library_entry, Mapping) else {}
    quality = dict(record.get("quality") or {}) if isinstance(record.get("quality"), Mapping) else {}
    fingerprint = dict(record.get("fingerprint") or {}) if isinstance(record.get("fingerprint"), Mapping) else {}
    if not quality.get("usable_segments"):
        quality["usable_segments"] = _legacy_usable_segments(
            record.get("duration_seconds") or record.get("duration")
        )
        quality.setdefault("segment_analysis_status", "legacy_safe_fallback")
    width = int(record.get("width") or candidate.get("variant", {}).get("width") or 0)
    height = int(record.get("height") or candidate.get("variant", {}).get("height") or 0)
    tags = str(record.get("tags") or candidate.get("tags") or library_entry.get("tags") or "")
    scene = record.get("scene_category") or quality.get("scene_category") or _scene_category(tags, None)
    history = _merge_usage_history(library_entry.get("usage_history"), record.get("usage_history"))
    canonical_basis: Mapping[str, Any] = library_entry or {**record, "fingerprint": fingerprint}
    local_text = str(record.get("local_path") or library_entry.get("local_path") or "")
    available = bool(local_text and Path(local_text).is_file())
    variant = candidate.get("variant") if isinstance(candidate.get("variant"), Mapping) else {}
    record.update(
        {
            "canonical_source_id": library_entry.get("canonical_source_id") or _canonical_source_id(
                canonical_basis, record.get("pixabay_id")
            ),
            "download_url": str(
                record.get("download_url")
                or variant.get("url")
                or library_entry.get("download_url")
                or ""
            ),
            "file_hash": str(record.get("file_hash") or fingerprint.get("sha256") or library_entry.get("file_hash") or ""),
            "ratio": record.get("ratio") or _ratio_label(width, height),
            "semantic_tags": list(
                record.get("semantic_tags")
                or candidate.get("semantic_tags")
                or library_entry.get("semantic_tags")
                or _semantic_tags(tags, scene, record.get("shot_type"), record.get("shot_scale"))
            ),
            "download_status": download_status or record.get("download_status") or ("cached" if available else "missing"),
            "available": available,
            "failure_reason": None if available else record.get("failure_reason") or "local file missing",
            "usage_history": history,
            "historical_usage_count": max(
                int(record.get("historical_usage_count") or 0),
                int(library_entry.get("historical_usage_count") or 0),
                _history_count(history, record.get("usage_intervals")),
            ),
            "quality": quality,
            "scene_category": scene,
            "motion_direction": record.get("motion_direction") or quality.get("motion_direction") or "unknown",
            "usable_segments": list(quality.get("usable_segments") or []),
        }
    )
    return record


def _failed_library_entry(
    candidate: Mapping[str, Any],
    stage: str,
    reason: str,
    quality: Mapping[str, Any] | None = None,
    media: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    variant = candidate.get("variant") if isinstance(candidate.get("variant"), Mapping) else {}
    fingerprint = media.get("fingerprint") if isinstance(media, Mapping) and isinstance(media.get("fingerprint"), Mapping) else {}
    width = int((media or {}).get("width") or variant.get("width") or 0)
    height = int((media or {}).get("height") or variant.get("height") or 0)
    return _normalize_library_entry(
        {
            "pixabay_id": candidate.get("pixabay_id", candidate.get("id")),
            "author": candidate.get("user") or "",
            "page_url": candidate.get("page_url") or "",
            "download_url": variant.get("url") or "",
            "tags": candidate.get("tags") or "",
            "local_path": "",
            "canonical_source_id": candidate.get("canonical_source_id") or _canonical_source_id(candidate),
            "file_hash": fingerprint.get("sha256") or "",
            "ratio": _ratio_label(width, height),
            "semantic_tags": candidate.get("semantic_tags") or _semantic_tags(candidate.get("tags")),
            "download_status": f"failed:{stage}",
            "available": False,
            "failure_reason": reason,
            "historical_usage_count": 0,
            "usage_history": [],
            "quality": dict(quality or {}),
            "media": dict(media or {}),
            "fingerprint": dict(fingerprint),
        }
    )


def _existing_usage(
    manifest_path: Path,
    *,
    strict: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    manifest = _read_json_strict(manifest_path, {}) if strict else _read_json(manifest_path, {})
    result: dict[str, list[dict[str, Any]]] = {}
    if isinstance(manifest, Mapping):
        for source in manifest.get("sources", []) or []:
            if not isinstance(source, Mapping):
                continue
            intervals = source.get("usage_intervals") or source.get("actual_usage_intervals") or []
            if not isinstance(intervals, list):
                intervals = []
            for identity in (source.get("pixabay_id"), source.get("id"), source.get("local_path")):
                if identity not in (None, ""):
                    result[str(identity)] = [dict(item) for item in intervals if isinstance(item, Mapping)]
    return result


def _preserve_manifest_usage(
    selected: Sequence[Mapping[str, Any]],
    previous_manifest: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    """Carry committed usage state into a new same-theme sources snapshot."""

    previous_sources = (
        previous_manifest.get("sources", [])
        if isinstance(previous_manifest, Mapping)
        else []
    )
    by_id = {
        str(source.get("pixabay_id", source.get("id"))): source
        for source in previous_sources
        if isinstance(source, Mapping)
        and source.get("pixabay_id", source.get("id")) not in (None, "")
    }
    by_path = {
        os.path.normcase(str(Path(str(source["local_path"])).expanduser().resolve())): source
        for source in previous_sources
        if isinstance(source, Mapping) and source.get("local_path")
    }
    merged: list[dict[str, Any]] = []
    for raw_source in selected:
        source = dict(raw_source)
        previous = by_id.get(str(source.get("pixabay_id", source.get("id"))))
        if previous is None and source.get("local_path"):
            previous = by_path.get(
                os.path.normcase(str(Path(str(source["local_path"])).expanduser().resolve()))
            )
        if isinstance(previous, Mapping):
            source["usage_history"] = _merge_usage_history(
                previous.get("usage_history"), source.get("usage_history")
            )
            source["historical_usage_count"] = max(
                int(previous.get("historical_usage_count") or 0),
                int(source.get("historical_usage_count") or 0),
                _history_count(source["usage_history"]),
            )
            current_intervals = source.get("actual_usage_intervals") or source.get("usage_intervals")
            if not current_intervals:
                prior_intervals = previous.get("actual_usage_intervals") or previous.get("usage_intervals")
                if isinstance(prior_intervals, list):
                    source["usage_intervals"] = [
                        dict(item) for item in prior_intervals if isinstance(item, Mapping)
                    ]
                    source["actual_usage_intervals"] = list(source["usage_intervals"])
        merged.append(_complete_asset_record(source))
    return merged


def run_pixabay_pipeline(
    theme: str,
    style_profile: Mapping[str, Any] | None,
    audio_profile: Mapping[str, Any] | None,
    material_root: str | os.PathLike[str],
    cache_dir: str | os.PathLike[str],
    desired_count: int,
    aspect_ratio: str,
    min_resolution: tuple[int, int] = (1280, 720),
    dry_run: bool = False,
    target_duration: float | None = None,
    timeline_plan: Any = None,
    candidate_pool_multiplier: int = 6,
    max_search_pages: int = 3,
    priority_queries: Sequence[str] = (),
    wide_aerial_only: bool = False,
    visual_cohesion_profile: str = "none",
    excluded_pixabay_ids: Sequence[str | int] = (),
    usage_mode: str = "local_evaluation",
) -> dict[str, Any]:
    """Search, select, download, QA, deduplicate, and attribute Pixabay clips.

    Only candidates that survive API metadata and thumbnail scoring are
    downloaded.  Failed post-download QA causes the next ranked candidate to be
    tried automatically.  Runtime artifacts never contain the API credential.
    """

    usage_mode = normalize_usage_mode(usage_mode)
    if not str(theme).strip():
        raise ValueError("theme must not be empty")
    desired_count = int(desired_count)
    if desired_count <= 0:
        raise ValueError("desired_count must be positive")
    candidate_pool_multiplier = int(candidate_pool_multiplier)
    max_search_pages = int(max_search_pages)
    visual_cohesion_profile = str(visual_cohesion_profile or "auto").strip()
    if candidate_pool_multiplier < 1:
        raise ValueError("candidate_pool_multiplier must be at least 1")
    if max_search_pages < 1:
        raise ValueError("max_search_pages must be at least 1")
    timeline_slots = _timeline_slots(timeline_plan)
    style_profile = _style_payload(style_profile)
    audio_profile = _audio_payload(audio_profile)
    visual_request = "" if visual_cohesion_profile.lower() in {"", "none", "auto"} else visual_cohesion_profile.replace("_", " ")
    visual_style_profile = build_visual_style_profile(
        theme,
        style_profile,
        audio_profile,
        visual_request,
    )
    style_profile = {**style_profile, "visual_style_profile": visual_style_profile}
    if target_duration is None:
        target_duration = _first_numeric(
            audio_profile,
            {"analyzed_duration_seconds", "duration_seconds", "duration"},
        )
    min_resolution = _parse_resolution(min_resolution)
    target_ratio, ratio_label = _parse_aspect_ratio(aspect_ratio)
    _load_environment()
    api_key = os.environ.get("PIXABAY_API_KEY", "").strip()
    if not api_key:
        raise PixabayPipelineError("PIXABAY_API_KEY is not set; put it in the project-root .env or environment")

    # Public contract: ``cache_dir`` is the Pixabay-stage namespace.  Accepting
    # the old project-cache root remains backward compatible via
    # ``pixabay_cache_root`` but every downstream helper sees exactly one
    # ``.../pixabay`` directory.
    runtime_paths = RuntimePaths.build(cache_root=cache_dir)
    cache_root = runtime_paths.pixabay_cache
    theme_dir = material_theme_directory(material_root, theme)
    theme_dir.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)
    nested_migration = migrate_legacy_nested_pixabay_cache(cache_root)
    manifest_path = theme_dir / "sources.json"
    with _exclusive_manifest_lock(manifest_path):
        prior_usage = _existing_usage(manifest_path, strict=True)
    fingerprint_path = cache_root / "video_fingerprints.json"
    legacy_root_fingerprint = cache_root.parent / "video_fingerprints.json"
    project_material_index = cache_root / "material_index.json"
    global_material_index = runtime_paths.global_material_index
    catalog_specs: list[tuple[str, Path]] = [
        ("pixabay_stage", fingerprint_path),
        ("project_material_index", project_material_index),
        ("global_material_index", global_material_index),
    ]
    if legacy_root_fingerprint != fingerprint_path:
        catalog_specs.insert(1, ("legacy_project_cache_root", legacy_root_fingerprint))
    catalog_sources: list[tuple[str, Sequence[Mapping[str, Any]]]] = []
    catalog_counts: dict[str, int] = {}
    for origin, path in catalog_specs:
        library = _load_fingerprint_library(path)
        entries = library.get("entries", [])
        catalog_sources.append((origin, entries))
        catalog_counts[origin] = len(entries)
    existing_entries = _merge_library_entries(catalog_sources)
    fingerprint_library = {
        "schema_version": SCHEMA_VERSION,
        "updated_at": _utc_now(),
        "entries": existing_entries,
    }
    library_paths = [fingerprint_path, project_material_index, global_material_index]
    if existing_entries:
        _persist_material_libraries(library_paths, existing_entries, api_key)
    cache_migration = {
        "legacy_nested_cache_copied": nested_migration,
        "catalog_entries_loaded": catalog_counts,
        "merged_existing_entries": len(existing_entries),
        "legacy_sources_left_in_place": True,
    }

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})
    candidates, search_rounds, search_errors = _collect_candidates(
        session,
        api_key,
        theme,
        style_profile,
        audio_profile,
        cache_root,
        desired_count,
        min_resolution,
        existing_entries,
        timeline_slots,
        candidate_pool_multiplier,
        max_search_pages,
        priority_queries,
        wide_aerial_only,
        visual_cohesion_profile,
        excluded_pixabay_ids,
    )
    ranked = _score_candidates(
        session,
        candidates,
        theme,
        style_profile,
        audio_profile,
        cache_root,
        target_ratio,
        min_resolution,
        desired_count,
        wide_aerial_only,
        visual_cohesion_profile,
    )
    candidate_log = [_public_candidate(candidate) for candidate in ranked]
    log_by_id = {str(item["pixabay_id"]): item for item in candidate_log}
    required_candidate_pool = (
        max(desired_count, len(timeline_slots) * candidate_pool_multiplier)
        if timeline_slots
        else None
    )
    slot_coverage = _slot_candidate_coverage(timeline_slots, ranked) if timeline_slots else {
        "passed": True,
        "failures": [],
        "slots": [],
    }
    candidate_pool_gate = {
        "evaluated": bool(timeline_slots),
        "passed": (
            not timeline_slots
            or (len(ranked) >= int(required_candidate_pool or 0) and slot_coverage["passed"])
        ),
        "planned_slot_count": len(timeline_slots),
        "candidate_pool_multiplier": candidate_pool_multiplier,
        "required_candidate_count": required_candidate_pool,
        "available_candidate_count": len(ranked),
        "failures": [],
        "important_slot_coverage": slot_coverage,
    }
    if timeline_slots and len(ranked) < int(required_candidate_pool or 0):
        candidate_pool_gate["failures"].append(
            f"metadata candidates {len(ranked)} < {required_candidate_pool} for {len(timeline_slots)} planned slots"
        )
    candidate_pool_gate["failures"].extend(slot_coverage.get("failures", []))

    selected: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    known_by_id = {str(entry.get("pixabay_id")): entry for entry in existing_entries if entry.get("pixabay_id") is not None}
    selected_canonical_ids: set[str] = set()

    if not candidate_pool_gate["passed"]:
        status = "insufficient_material"
    elif dry_run:
        for item in candidate_log[:desired_count]:
            item["decision"] = "planned_dry_run"
        status = "dry_run"
    else:
        locked_candidates = _asset_locked_candidates(
            ranked,
            global_material_index,
            library_paths,
        )
        for candidate, concurrent_entries in locked_candidates:
            if len(selected) >= desired_count:
                interim_sufficiency = evaluate_selected_sufficiency(
                    selected, desired_count, target_duration
                )
                if interim_sufficiency["passed"]:
                    break
            asset_id = str(candidate["pixabay_id"])
            log_item = log_by_id[asset_id]
            # This is intentionally after the per-ID lock is acquired.  A
            # concurrent project may have completed the same asset between our
            # initial search snapshot and this selection turn.
            existing_entries = _merge_library_entries(
                (
                    ("current_run", existing_entries),
                    ("post_asset_lock", concurrent_entries),
                )
            )
            fingerprint_library["entries"] = existing_entries
            known_by_id = {
                str(entry.get("pixabay_id")): entry
                for entry in existing_entries
                if entry.get("pixabay_id") is not None
            }
            known = known_by_id.get(asset_id)
            if known and Path(str(known.get("local_path") or "")).is_file():
                known = _normalize_library_entry(known)
                known_path = Path(str(known["local_path"])).resolve()
                if not known.get("file_hash"):
                    file_hash = _sha256_file(known_path)
                    known["file_hash"] = file_hash
                    known.setdefault("fingerprint", {})["sha256"] = file_hash
                    known["canonical_source_id"] = f"sha256:{file_hash}"
                cached_quality = known.get("quality") if isinstance(known.get("quality"), Mapping) else {}
                if not analysis_cache_valid(cached_quality, str(known.get("file_hash") or "")):
                    refreshed_quality, refreshed_media = _video_quality(
                        known_path,
                        style_profile,
                        min_resolution,
                        tags=str(candidate.get("tags") or known.get("tags") or ""),
                        human_focused=_human_focused_theme(_theme_terms(theme)),
                    )
                    known["quality"] = refreshed_quality
                    known["media"] = refreshed_media
                    known["fingerprint"] = refreshed_media.get("fingerprint", known.get("fingerprint", {}))
                    known["file_hash"] = str(known["fingerprint"].get("sha256") or known.get("file_hash") or "")
                    if not refreshed_quality.get("passed"):
                        reasons = list(refreshed_quality.get("rejection_reasons") or ["v1.3 aesthetic reanalysis failed"])
                        rejections.append(
                            {
                                "pixabay_id": candidate["pixabay_id"],
                                "stage": "cached_asset_v13_reanalysis",
                                "reasons": reasons,
                                "quality": refreshed_quality,
                            }
                        )
                        log_item.update(
                            {
                                "decision": "rejected",
                                "rejection_stage": "cached_asset_v13_reanalysis",
                                "reasons": reasons,
                                "available": True,
                                "download_status": "cached_rejected_for_current_profile",
                            }
                        )
                        continue
                known_canonical = str(known.get("canonical_source_id") or _canonical_source_id(known))
                if known_canonical in selected_canonical_ids:
                    reason = "canonical source already selected in this montage"
                    rejection = {
                        "pixabay_id": candidate["pixabay_id"],
                        "canonical_source_id": known_canonical,
                        "stage": "canonical_dedup",
                        "reasons": [reason],
                    }
                    rejections.append(rejection)
                    log_item.update(
                        {
                            "decision": "rejected",
                            "rejection_stage": "canonical_dedup",
                            "download_status": "cached_duplicate",
                            "available": True,
                            "failure_reason": reason,
                            "reasons": [reason],
                        }
                    )
                    continue
                reused_path, reuse_mode = _reuse_local_path(
                    known_path,
                    theme_dir,
                    candidate,
                    len(selected) + 1,
                    theme,
                )
                quality = dict(known.get("quality") or {})
                media = dict(known.get("media") or {})
                usage = prior_usage.get(
                    asset_id,
                    prior_usage.get(str(reused_path), prior_usage.get(str(known_path), [])),
                )
                reuse_origin = {
                    "local_path": str(known_path),
                    "pixabay_id": known.get("pixabay_id", candidate["pixabay_id"]),
                    "library_origin": known.get("library_origin", "material_library"),
                    "added_at": known.get("added_at"),
                }
                source = {
                    "id": candidate["pixabay_id"],
                    "pixabay_id": candidate["pixabay_id"],
                    "author": candidate.get("user") or known.get("author") or "",
                    "page_url": candidate.get("page_url") or known.get("page_url") or "",
                    "tags": candidate.get("tags") or known.get("tags") or "",
                    "search_query": (candidate.get("matched_queries") or [""])[0],
                    "search_queries": list(candidate.get("matched_queries") or []),
                    "local_path": str(reused_path),
                    "duration_seconds": media.get("duration_seconds", candidate.get("duration", 0)),
                    "duration": media.get("duration_seconds", candidate.get("duration", 0)),
                    "width": media.get("width", candidate["variant"].get("width", 0)),
                    "height": media.get("height", candidate["variant"].get("height", 0)),
                    "fps": media.get("fps"),
                    "shot_type": candidate.get("shot_type"),
                    "shot_scale": candidate.get("shot_scale"),
                    "motion_score": quality.get("motion_score", candidate.get("motion_score_estimate", 0.5)),
                    "scene_category": quality.get("scene_category", candidate.get("scene_category", "general")),
                    "face_content_risk": quality.get("face_content_risk", candidate.get("face_content_risk", 0.0)),
                    "subject_profile": quality.get(
                        "subject_profile",
                        candidate.get("thumbnail_signals", {}).get("subject_profile", {}),
                    ),
                    "pre_score": candidate.get("pre_score"),
                    "score_components": candidate.get("score_components"),
                    "quality": quality,
                    "fingerprint": known.get("fingerprint") or media.get("fingerprint"),
                    "usage_intervals": usage,
                    "actual_usage_intervals": usage,
                    "reused_existing_file": True,
                    "reuse_mode": reuse_mode,
                    "reuse_origin": reuse_origin,
                }
                source = _complete_asset_record(
                    source,
                    candidate,
                    known,
                    download_status="reused_cached",
                )
                selected.append(source)
                selected_canonical_ids.add(str(source["canonical_source_id"]))
                refreshed_known = {
                    **known,
                    "author": source["author"],
                    "page_url": source["page_url"],
                    "tags": source["tags"],
                    "local_path": source["local_path"],
                    "canonical_source_id": source["canonical_source_id"],
                    "download_url": source["download_url"],
                    "file_hash": source["file_hash"],
                    "ratio": source["ratio"],
                    "semantic_tags": source["semantic_tags"],
                    "download_status": source["download_status"],
                    "available": source["available"],
                    "failure_reason": source["failure_reason"],
                    "historical_usage_count": source["historical_usage_count"],
                    "usage_history": source["usage_history"],
                    "quality": source["quality"],
                }
                existing_entries.append(refreshed_known)
                known_by_id[asset_id] = refreshed_known
                _persist_material_libraries(library_paths, existing_entries, api_key)
                log_item.update(
                    {
                        "decision": "selected_existing",
                        "local_path": source["local_path"],
                        "reuse_mode": reuse_mode,
                        "reuse_origin": reuse_origin,
                        "canonical_source_id": source["canonical_source_id"],
                        "file_hash": source["file_hash"],
                        "download_url": source["download_url"],
                        "download_status": source["download_status"],
                        "available": source["available"],
                        "failure_reason": source["failure_reason"],
                    }
                )
                continue

            extension = _extension_from_url(str(candidate["variant"]["url"]))
            temp_path = theme_dir / f".pixabay_{asset_id}_{os.getpid()}.part{extension}"
            try:
                _download_video(session, str(candidate["variant"]["url"]), temp_path, api_key)
                quality, media = _video_quality(
                    temp_path,
                    style_profile,
                    min_resolution,
                    tags=str(candidate.get("tags") or ""),
                    human_focused=_human_focused_theme(_theme_terms(theme)),
                )
                if not quality["passed"]:
                    rejection = {
                        "pixabay_id": candidate["pixabay_id"],
                        "stage": "post_download_qa",
                        "reasons": list(quality["rejection_reasons"]),
                        "quality": quality,
                    }
                    rejections.append(rejection)
                    log_item.update(
                        {
                            "decision": "rejected",
                            "rejection_stage": "post_download_qa",
                            "reasons": rejection["reasons"],
                            "quality": quality,
                            "download_status": "rejected_after_download",
                            "available": False,
                            "failure_reason": "; ".join(rejection["reasons"]),
                        }
                    )
                    failed_entry = _failed_library_entry(
                        candidate,
                        "post_download_qa",
                        "; ".join(rejection["reasons"]),
                        quality,
                        media,
                    )
                    existing_entries.append(failed_entry)
                    known_by_id[asset_id] = failed_entry
                    _persist_material_libraries(library_paths, existing_entries, api_key)
                    temp_path.unlink(missing_ok=True)
                    continue
                duplicate, duplicate_of, distance = _fingerprint_duplicate(media["fingerprint"], existing_entries)
                if duplicate:
                    existing_duplicate = next(
                        (
                            entry
                            for entry in existing_entries
                            if str(entry.get("pixabay_id")) == str(duplicate_of)
                            or str(entry.get("local_path")) == str(duplicate_of)
                        ),
                        None,
                    )
                    existing_path = (
                        Path(str(existing_duplicate.get("local_path"))).resolve()
                        if isinstance(existing_duplicate, Mapping) and existing_duplicate.get("local_path")
                        else None
                    )
                    temp_path.unlink(missing_ok=True)
                    if existing_path is None or not existing_path.is_file():
                        reason = f"video fingerprint duplicates unavailable catalog entry {duplicate_of}"
                        rejection = {
                            "pixabay_id": candidate["pixabay_id"],
                            "stage": "post_download_dedup",
                            "reasons": [reason],
                            "perceptual_distance": distance,
                        }
                        rejections.append(rejection)
                        log_item.update({"decision": "rejected", "rejection_stage": "post_download_dedup", "reasons": [reason]})
                        continue
                    existing_duplicate = _normalize_library_entry(existing_duplicate)
                    duplicate_canonical = str(
                        existing_duplicate.get("canonical_source_id")
                        or _canonical_source_id(existing_duplicate)
                    )
                    if duplicate_canonical in selected_canonical_ids:
                        reason = "perceptual duplicate canonical source already selected in this montage"
                        rejection = {
                            "pixabay_id": candidate["pixabay_id"],
                            "canonical_source_id": duplicate_canonical,
                            "stage": "canonical_dedup",
                            "reasons": [reason],
                            "perceptual_distance": distance,
                        }
                        rejections.append(rejection)
                        log_item.update(
                            {
                                "decision": "rejected",
                                "rejection_stage": "canonical_dedup",
                                "download_status": "discarded_perceptual_duplicate",
                                "available": False,
                                "failure_reason": reason,
                                "reasons": [reason],
                            }
                        )
                        continue
                    reused_path, reuse_mode = _reuse_local_path(
                        existing_path, theme_dir, candidate, len(selected) + 1, theme
                    )
                    usage = prior_usage.get(asset_id, prior_usage.get(str(reused_path), []))
                    reused_quality = dict(existing_duplicate.get("quality") or quality)
                    reused_media = dict(existing_duplicate.get("media") or media)
                    reused_fingerprint = dict(existing_duplicate.get("fingerprint") or media["fingerprint"])
                    source = {
                        "id": candidate["pixabay_id"],
                        "pixabay_id": candidate["pixabay_id"],
                        "author": candidate.get("user") or "",
                        "page_url": candidate.get("page_url") or "",
                        "tags": candidate.get("tags") or "",
                        "search_query": (candidate.get("matched_queries") or [""])[0],
                        "search_queries": list(candidate.get("matched_queries") or []),
                        "local_path": str(reused_path),
                        "duration_seconds": reused_media.get("duration_seconds", media["duration_seconds"]),
                        "duration": reused_media.get("duration_seconds", media["duration_seconds"]),
                        "width": reused_media.get("width", media["width"]),
                        "height": reused_media.get("height", media["height"]),
                        "fps": reused_media.get("fps", media["fps"]),
                        "shot_type": candidate.get("shot_type"),
                        "shot_scale": candidate.get("shot_scale"),
                        "motion_score": reused_quality.get("motion_score", quality["motion_score"]),
                        "scene_category": reused_quality.get("scene_category", candidate.get("scene_category", "general")),
                        "face_content_risk": reused_quality.get("face_content_risk", candidate.get("face_content_risk", 0.0)),
                        "subject_profile": reused_quality.get("subject_profile", {}),
                        "pre_score": candidate.get("pre_score"),
                        "score_components": candidate.get("score_components"),
                        "quality": reused_quality,
                        "fingerprint": reused_fingerprint,
                        "usage_intervals": usage,
                        "actual_usage_intervals": usage,
                        "reused_existing_file": True,
                        "reuse_mode": f"perceptual_{reuse_mode}",
                        "reuse_origin": {"local_path": str(existing_path), "duplicate_of": duplicate_of, "perceptual_distance": distance},
                    }
                    source = _complete_asset_record(
                        source,
                        candidate,
                        existing_duplicate,
                        download_status="reused_perceptual_duplicate",
                    )
                    selected.append(source)
                    selected_canonical_ids.add(str(source["canonical_source_id"]))
                    alias_entry = {
                        "pixabay_id": candidate["pixabay_id"],
                        "author": source["author"],
                        "page_url": source["page_url"],
                        "tags": source["tags"],
                        "local_path": source["local_path"],
                        "added_at": _utc_now(),
                        "library_origin": "perceptual_reuse",
                        "canonical_source_id": source["canonical_source_id"],
                        "download_url": source["download_url"],
                        "file_hash": source["file_hash"],
                        "ratio": source["ratio"],
                        "semantic_tags": source["semantic_tags"],
                        "download_status": source["download_status"],
                        "available": True,
                        "failure_reason": None,
                        "historical_usage_count": source["historical_usage_count"],
                        "usage_history": source["usage_history"],
                        "fingerprint": reused_fingerprint,
                        "quality": reused_quality,
                        "media": reused_media,
                    }
                    existing_entries.append(alias_entry)
                    known_by_id[asset_id] = alias_entry
                    _persist_material_libraries(library_paths, existing_entries, api_key)
                    log_item.update(
                        {
                            "decision": "selected_perceptual_reuse",
                            "local_path": source["local_path"],
                            "reuse_mode": source["reuse_mode"],
                            "canonical_source_id": source["canonical_source_id"],
                            "file_hash": source["file_hash"],
                            "download_status": source["download_status"],
                            "available": True,
                            "failure_reason": None,
                        }
                    )
                    continue
                description = _chinese_description(str(candidate.get("tags") or ""), theme)
                final_name = _safe_chinese_slug(
                    f"{len(selected) + 1:02d}_{description}_{asset_id}",
                    f"素材_{asset_id}",
                ) + extension
                final_path = theme_dir / final_name
                if final_path.exists():
                    final_path = theme_dir / (_safe_chinese_slug(f"{len(selected) + 1:02d}_{description}_{asset_id}_{int(time.time())}") + extension)
                os.replace(temp_path, final_path)
                usage = prior_usage.get(asset_id, prior_usage.get(str(final_path), []))
                source = {
                    "id": candidate["pixabay_id"],
                    "pixabay_id": candidate["pixabay_id"],
                    "author": candidate.get("user") or "",
                    "page_url": candidate.get("page_url") or "",
                    "tags": candidate.get("tags") or "",
                    "search_query": (candidate.get("matched_queries") or [""])[0],
                    "search_queries": list(candidate.get("matched_queries") or []),
                    "local_path": str(final_path.resolve()),
                    "duration_seconds": media["duration_seconds"],
                    "duration": media["duration_seconds"],
                    "width": media["width"],
                    "height": media["height"],
                    "fps": media["fps"],
                    "shot_type": candidate.get("shot_type"),
                    "shot_scale": candidate.get("shot_scale"),
                    "motion_score": quality["motion_score"],
                    "scene_category": quality.get("scene_category", candidate.get("scene_category", "general")),
                    "face_content_risk": quality.get("face_content_risk", candidate.get("face_content_risk", 0.0)),
                    "subject_profile": quality.get("subject_profile", {}),
                    "pre_score": candidate.get("pre_score"),
                    "score_components": candidate.get("score_components"),
                    "quality": quality,
                    "fingerprint": media["fingerprint"],
                    "usage_intervals": usage,
                    "actual_usage_intervals": usage,
                    "reused_existing_file": False,
                    "reuse_mode": "downloaded",
                    "reuse_origin": None,
                }
                source = _complete_asset_record(
                    source,
                    candidate,
                    None,
                    download_status="downloaded",
                )
                selected.append(source)
                selected_canonical_ids.add(str(source["canonical_source_id"]))
                library_entry = {
                    "pixabay_id": candidate["pixabay_id"],
                    "author": source["author"],
                    "page_url": source["page_url"],
                    "tags": source["tags"],
                    "local_path": source["local_path"],
                    "added_at": _utc_now(),
                    "library_origin": "current_download",
                    "canonical_source_id": source["canonical_source_id"],
                    "download_url": source["download_url"],
                    "file_hash": source["file_hash"],
                    "ratio": source["ratio"],
                    "semantic_tags": source["semantic_tags"],
                    "download_status": "downloaded",
                    "available": True,
                    "failure_reason": None,
                    "historical_usage_count": source["historical_usage_count"],
                    "usage_history": source["usage_history"],
                    "fingerprint": media["fingerprint"],
                    "quality": quality,
                    "media": media,
                }
                existing_entries.append(library_entry)
                fingerprint_library["entries"] = existing_entries
                fingerprint_library["updated_at"] = _utc_now()
                known_by_id[asset_id] = library_entry
                _persist_material_libraries(library_paths, existing_entries, api_key)
                log_item.update(
                    {
                        "decision": "selected",
                        "local_path": source["local_path"],
                        "quality": quality,
                        "canonical_source_id": source["canonical_source_id"],
                        "file_hash": source["file_hash"],
                        "download_url": source["download_url"],
                        "download_status": "downloaded",
                        "available": True,
                        "failure_reason": None,
                    }
                )
            except PixabayPipelineError as exc:
                temp_path.unlink(missing_ok=True)
                safe = _safe_error(exc, api_key)
                rejection = {"pixabay_id": candidate["pixabay_id"], "stage": "download_or_decode", "reasons": [safe]}
                rejections.append(rejection)
                log_item.update(
                    {
                        "decision": "rejected",
                        "rejection_stage": "download_or_decode",
                        "reasons": [safe],
                        "download_status": "failed",
                        "available": False,
                        "failure_reason": safe,
                    }
                )
                failed_entry = _failed_library_entry(
                    candidate,
                    "download_or_decode",
                    safe,
                )
                existing_entries.append(failed_entry)
                known_by_id[asset_id] = failed_entry
                _persist_material_libraries(library_paths, existing_entries, api_key)
        locked_candidates.close()
        status = "ok" if len(selected) >= desired_count else "partial"

    sufficiency = (
        {
            "passed": True,
            "failures": [],
            "dry_run_not_evaluated": True,
        }
        if dry_run
        else evaluate_selected_sufficiency(selected, desired_count, target_duration)
    )
    if not candidate_pool_gate["passed"]:
        sufficiency["failures"] = [
            *candidate_pool_gate.get("failures", []),
            *sufficiency.get("failures", []),
        ]
        sufficiency["passed"] = False
    if not dry_run and (len(selected) < desired_count or not sufficiency["passed"]):
        if len(selected) < desired_count:
            sufficiency["failures"].insert(
                0, f"selected assets {len(selected)} < requested {desired_count}"
            )
        sufficiency["passed"] = False
        status = "insufficient_material"

    manifest = apply_usage_policy({
        "schema_version": SCHEMA_VERSION,
        "asset_manifest_schema_version": ASSET_MANIFEST_SCHEMA_VERSION,
        "manifest_type": "asset_manifest",
        "generated_at": _utc_now(),
        "status": status,
        "dry_run": bool(dry_run),
        "theme": theme,
        "theme_directory": str(theme_dir.resolve()),
        "requested": {
            "desired_count": desired_count,
            "aspect_ratio": ratio_label,
            "min_resolution": {"width": min_resolution[0], "height": min_resolution[1]},
            "candidate_pool_multiplier": candidate_pool_multiplier,
            "max_search_pages": max_search_pages,
            "priority_queries": list(priority_queries),
            "wide_aerial_only": bool(wide_aerial_only),
            "visual_cohesion_profile": visual_cohesion_profile,
            "visual_style_profile_digest": visual_style_profile["profile_digest"],
            "excluded_pixabay_ids": sorted({str(value) for value in excluded_pixabay_ids}),
        },
        "timeline_plan": {
            "provided": bool(timeline_slots),
            "slot_count": len(timeline_slots),
        },
        "candidate_pool_gate": candidate_pool_gate,
        "visual_style_profile": visual_style_profile,
        "cache_layout": {
            "pixabay_root": str(cache_root),
            "search": str(cache_root / "search"),
            "thumbnails": str(cache_root / "thumbnails"),
            "fingerprints": str(fingerprint_path),
        },
        "cache_migration": cache_migration,
        "material_libraries": {
            "project": str(project_material_index),
            "global": str(global_material_index),
        },
        "search_cache_ttl_seconds": CACHE_TTL_SECONDS,
        "search_rounds": search_rounds,
        "search_errors": search_errors,
        "candidate_count": len(ranked),
        "candidate_log": candidate_log,
        "rejections": rejections,
        "sources": selected,
        "assets": selected,
        "selected_count": len(selected),
        "sufficiency": sufficiency,
        "reuse_summary": dict(
            Counter(str(source.get("reuse_mode") or "unknown") for source in selected)
        ),
        "heuristic_notice": "Content, shot scale, motion, aesthetic quality, and visual consistency labels are sampled signal-derived estimates.",
    }, usage_mode)
    snapshot_token = secrets.token_hex(12)
    snapshot_path = (
        cache_root
        / "run_manifests"
        / _safe_chinese_slug(theme, "未命名主题")
        / f"sources-{int(time.time() * 1000)}-{snapshot_token}.json"
    )
    with _exclusive_manifest_lock(manifest_path):
        previous_manifest = _read_json_strict(manifest_path, {})
        if previous_manifest and not isinstance(previous_manifest, Mapping):
            raise PixabayPipelineError(f"Sources manifest has an invalid root object: {manifest_path}")
        selected = _preserve_manifest_usage(selected, previous_manifest)
        manifest["sources"] = selected
        manifest["assets"] = selected
        manifest["selected_count"] = len(selected)
        manifest["stable_snapshot"] = str(snapshot_path.resolve())
        _atomic_write_json(manifest_path, manifest, api_key)
        # The shared theme manifest remains the compatibility/history target.
        # Callers copy this immutable per-invocation snapshot, so another run of
        # the same theme cannot make asset_manifest.json disagree with the
        # selected objects returned in memory.
        _atomic_write_json(snapshot_path, manifest, api_key)
    if status == "insufficient_material":
        raise InsufficientMaterialError(
            "Pixabay material sufficiency gate failed after all search expansions: "
            + "; ".join(str(item) for item in sufficiency.get("failures", []))
            + f". Search and rejection records: {snapshot_path}"
        )
    result = {
        "status": status,
        "theme": theme,
        "theme_dir": str(theme_dir.resolve()),
        "sources_manifest": str(snapshot_path.resolve()),
        "asset_manifest": str(snapshot_path.resolve()),
        "asset_manifest_path": str(snapshot_path.resolve()),
        "shared_sources_manifest": str(manifest_path.resolve()),
        "fingerprint_library": str(fingerprint_path.resolve()),
        "global_material_library": str(global_material_index.resolve()),
        "cache_root": str(cache_root.resolve()),
        "cache_migration": cache_migration,
        "desired_count": desired_count,
        "selected_count": len(selected),
        "sufficiency": sufficiency,
        "candidate_pool_gate": candidate_pool_gate,
        "selected": selected,
        "search_rounds": search_rounds,
        "search_errors": search_errors,
        "rejections": rejections,
        "candidate_count": len(ranked),
        "priority_queries": list(priority_queries),
        "wide_aerial_only": bool(wide_aerial_only),
        "visual_cohesion_profile": visual_cohesion_profile,
        "visual_style_profile": visual_style_profile,
        "excluded_pixabay_ids": sorted({str(value) for value in excluded_pixabay_ids}),
        "dry_run": bool(dry_run),
    }
    return _strip_secrets(result, api_key)


def _iter_shots(value: Any) -> Iterable[Mapping[str, Any]]:
    """Yield edit-plan shot mappings without mistaking interval dicts for shots."""

    if isinstance(value, Mapping):
        identity_keys = {"pixabay_id", "asset_id", "source_id", "local_path", "source_path", "file"}
        interval_keys = {"output_start", "output_end", "timeline_start", "timeline_end", "source_start", "source_end", "start", "end"}
        keys = set(value)
        if keys & identity_keys and keys & interval_keys:
            yield value
            return
        for key in ("shots", "clips", "segments", "edit_plan", "timeline", "items", "usage"):
            if key in value:
                yield from _iter_shots(value[key])
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_shots(item)


def _normalized_interval(shot: Mapping[str, Any]) -> dict[str, Any] | None:
    def first(*keys: str) -> Any:
        for key in keys:
            if shot.get(key) is not None:
                return shot.get(key)
        return None

    output_start = first("output_start", "timeline_start", "start", "output_in")
    output_end = first("output_end", "timeline_end", "end", "output_out")
    source_start = first("source_start", "source_in", "in_point", "clip_start")
    source_end = first("source_end", "source_out", "out_point", "clip_end")
    try:
        normalized = {
            "output_start": round(float(output_start), 6),
            "output_end": round(float(output_end), 6),
            "source_start": round(float(source_start if source_start is not None else 0.0), 6),
            "source_end": round(float(source_end), 6) if source_end is not None else None,
        }
    except (TypeError, ValueError):
        return None
    if normalized["output_end"] <= normalized["output_start"]:
        return None
    if normalized["source_end"] is None:
        normalized["source_end"] = round(
            normalized["source_start"] + normalized["output_end"] - normalized["output_start"], 6
        )
    for key in ("speed", "section", "reason", "transition"):
        if shot.get(key) is not None:
            normalized[key] = shot[key]
    return normalized


def _apply_usage_update_to_manifest(
    path: Path,
    *,
    by_id: Mapping[str, list[dict[str, Any]]],
    by_path: Mapping[str, list[dict[str, Any]]],
    event_id: str,
    run_id: str,
    recorded_at: str,
    parsed_shots: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], int]:
    """Apply one idempotent usage event under the manifest transaction lock."""

    with _exclusive_manifest_lock(path):
        current_manifest = _read_json_strict(path, None)
        if not isinstance(current_manifest, Mapping):
            raise PixabayPipelineError(f"Sources manifest is missing or invalid: {path}")
        manifest_copy = dict(current_manifest)
        sources = [
            dict(source)
            for source in current_manifest.get("sources", [])
            if isinstance(source, Mapping)
        ]
        updated_sources = 0
        for source_index, source in enumerate(sources):
            intervals = by_id.get(str(source.get("pixabay_id", source.get("id"))))
            if intervals is None and source.get("local_path"):
                try:
                    source_key = os.path.normcase(
                        str(Path(str(source["local_path"])).expanduser().resolve())
                    )
                except OSError:
                    source_key = os.path.normcase(str(source["local_path"]))
                intervals = by_path.get(source_key)
            matched = intervals is not None
            current_intervals = [dict(item) for item in (intervals or [])]
            current_intervals.sort(key=lambda item: (item["output_start"], item["output_end"]))
            source["usage_intervals"] = current_intervals
            source["actual_usage_intervals"] = current_intervals
            if matched:
                history_record = {
                    "event_id": event_id,
                    "run_id": run_id,
                    "recorded_at": recorded_at,
                    "intervals": current_intervals,
                }
                source["usage_history"] = _merge_usage_history(
                    source.get("usage_history"), [history_record]
                )
                source["historical_usage_count"] = _history_count(source["usage_history"])
                updated_sources += 1
            sources[source_index] = _complete_asset_record(source)
        manifest_copy["sources"] = sources
        manifest_copy["assets"] = sources
        manifest_copy["usage_updated_at"] = recorded_at
        manifest_copy["usage_update_summary"] = {
            "parsed_shots": parsed_shots,
            "updated_sources": updated_sources,
            "unmatched_sources": max(0, len(sources) - updated_sources),
            "event_id": event_id,
            "run_id": run_id,
        }
        _atomic_write_json(path, manifest_copy)
        return manifest_copy, sources, updated_sources


def update_usage_intervals(
    manifest_path: str | os.PathLike[str],
    edit_plan_or_shots: str | os.PathLike[str] | Mapping[str, Any] | Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Fill actual output/source intervals by ``pixabay_id`` or local path.

    ``edit_plan_or_shots`` may be an edit-plan JSON path, a nested edit-plan
    mapping, or a list of shots.  Recognized interval names include
    ``output_start/output_end`` and ``timeline_start/timeline_end`` plus
    ``source_start/source_end``.
    """

    path = Path(manifest_path).expanduser().resolve()
    with _exclusive_manifest_lock(path):
        manifest = _read_json_strict(path, None)
        if not isinstance(manifest, Mapping):
            raise PixabayPipelineError(f"Sources manifest is missing or invalid: {path}")
    if isinstance(edit_plan_or_shots, (str, os.PathLike)):
        plan_path = Path(edit_plan_or_shots).expanduser().resolve()
        plan: Any = _read_json(plan_path, None)
        if plan is None:
            raise PixabayPipelineError(f"Edit plan is missing or invalid: {plan_path}")
    else:
        plan = edit_plan_or_shots

    by_id: dict[str, list[dict[str, Any]]] = {}
    by_path: dict[str, list[dict[str, Any]]] = {}
    parsed_shots = 0
    for shot in _iter_shots(plan):
        interval = _normalized_interval(shot)
        if interval is None:
            continue
        parsed_shots += 1
        identity = shot.get("pixabay_id", shot.get("asset_id", shot.get("source_id")))
        local = shot.get("local_path", shot.get("source_path", shot.get("file")))
        if identity not in (None, ""):
            by_id.setdefault(str(identity), []).append(interval)
        if local not in (None, ""):
            try:
                normalized_path = os.path.normcase(str(Path(str(local)).expanduser().resolve()))
            except OSError:
                normalized_path = os.path.normcase(str(local))
            by_path.setdefault(normalized_path, []).append(interval)

    interval_payload = {
        "by_id": {key: value for key, value in sorted(by_id.items())},
        "by_path": {key: value for key, value in sorted(by_path.items())},
    }
    plan_run_id = plan.get("run_id") if isinstance(plan, Mapping) else None
    interval_digest = hashlib.sha256(
        json.dumps(interval_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:24]
    run_id = str(plan_run_id or manifest.get("run_id") or f"usage-{interval_digest}")
    event_payload = {"run_id": run_id, **interval_payload}
    event_id = hashlib.sha256(
        json.dumps(event_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:24]
    recorded_at = _utc_now()
    manifest_copy, sources, updated_sources = _apply_usage_update_to_manifest(
        path,
        by_id=by_id,
        by_path=by_path,
        event_id=event_id,
        run_id=run_id,
        recorded_at=recorded_at,
        parsed_shots=parsed_shots,
    )

    library_paths: list[Path] = []
    material_libraries = manifest_copy.get("material_libraries")
    if isinstance(material_libraries, Mapping):
        for value in material_libraries.values():
            if value:
                library_paths.append(Path(str(value)).expanduser().resolve())
    cache_layout = manifest_copy.get("cache_layout")
    if isinstance(cache_layout, Mapping) and cache_layout.get("fingerprints"):
        library_paths.append(Path(str(cache_layout["fingerprints"])).expanduser().resolve())

    source_by_pixabay = {
        str(source.get("pixabay_id", source.get("id"))): source
        for source in sources
        if source.get("pixabay_id", source.get("id")) not in (None, "")
    }
    source_by_canonical = {
        str(source.get("canonical_source_id")): source
        for source in sources
        if source.get("canonical_source_id")
    }
    source_by_path: dict[str, Mapping[str, Any]] = {}
    for source in sources:
        if source.get("local_path"):
            source_by_path[os.path.normcase(str(Path(str(source["local_path"])).expanduser().resolve()))] = source

    updated_libraries = 0
    seen_library_paths: set[Path] = set()
    for library_path in library_paths:
        if library_path in seen_library_paths:
            continue
        seen_library_paths.add(library_path)
        with _exclusive_catalog_lock(library_path, timeout_seconds=600.0):
            # Usage updates can finish concurrently with another project's
            # acquisition. Re-read under the same lock used by catalog merges.
            if not library_path.is_file():
                continue
            raw_library = _read_json_strict(library_path, {})
            library = _load_fingerprint_library(library_path, strict=True)
            changed = False
            entries: list[dict[str, Any]] = []
            for raw_entry in library.get("entries", []):
                entry = _normalize_library_entry(raw_entry)
                source = source_by_pixabay.get(str(entry.get("pixabay_id")))
                if source is None:
                    source = source_by_canonical.get(str(entry.get("canonical_source_id")))
                if source is None and entry.get("local_path"):
                    normalized_path = os.path.normcase(
                        str(Path(str(entry["local_path"])).expanduser().resolve())
                    )
                    source = source_by_path.get(normalized_path)
                    if (
                        source is not None
                        and entry.get("provider") == "local-library"
                        and str(entry.get("canonical_source_id") or "")
                        != str(source.get("canonical_source_id") or "")
                    ):
                        source = None
                if source is not None:
                    merged_history = _merge_usage_history(
                        entry.get("usage_history"), source.get("usage_history")
                    )
                    if merged_history != entry.get("usage_history"):
                        changed = True
                    entry["usage_history"] = merged_history
                    entry["historical_usage_count"] = max(
                        int(entry.get("historical_usage_count") or 0),
                        _history_count(merged_history),
                    )
                    entry["last_used_at"] = recorded_at
                entries.append(entry)
            if changed:
                updated_library = dict(raw_library) if isinstance(raw_library, Mapping) else {}
                updated_library.update(
                    {
                        "schema_version": library.get("schema_version", SCHEMA_VERSION),
                        "updated_at": recorded_at,
                        "entries": entries,
                    }
                )
                _atomic_write_json(
                    library_path,
                    updated_library,
                )
                updated_libraries += 1
    return {
        "manifest_path": str(path),
        "parsed_shots": parsed_shots,
        "updated_sources": updated_sources,
        "unmatched_sources": max(0, len(sources) - updated_sources),
        "event_id": event_id,
        "run_id": run_id,
        "updated_libraries": updated_libraries,
    }


def _load_profile_argument(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    payload = _read_json(Path(path).expanduser().resolve(), None)
    if not isinstance(payload, Mapping):
        raise PixabayPipelineError(f"Profile is missing or invalid JSON: {path}")
    return dict(payload)


def _default_project_root() -> Path:
    return RuntimePaths.build().project_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Search, quality-rank, download, deduplicate, and attribute Pixabay video material."
    )
    subparsers = parser.add_subparsers(dest="command")
    update = subparsers.add_parser("update-usage", help="write edit-plan usage intervals into sources.json")
    update.add_argument("--manifest", required=True, help="path to sources.json")
    update.add_argument("--edit-plan", required=True, help="path to edit_plan.json or compatible shots JSON")

    parser.add_argument("--theme", help="topic/theme used for English visual search queries")
    parser.add_argument("--style-profile", help="path to reference style_profile.json")
    parser.add_argument("--audio-profile", help="path to BGM profile JSON")
    parser.add_argument("--material-root", default=str(_default_project_root() / "视频素材"))
    parser.add_argument(
        "--cache-dir",
        default=str(RuntimePaths.build().pixabay_cache),
        help="Pixabay cache namespace (default: PROJECT/.bgm-montage-cache/pixabay)",
    )
    parser.add_argument("--desired-count", type=int, default=6)
    parser.add_argument("--aspect-ratio", default="16:9")
    parser.add_argument("--min-resolution", default="1280x720")
    parser.add_argument("--timeline-plan", help="optional pre-download timeline/slot plan JSON")
    parser.add_argument("--candidate-pool-multiplier", type=int, default=6)
    parser.add_argument("--max-search-pages", type=int, default=3)
    parser.add_argument(
        "--search-query",
        dest="priority_queries",
        action="append",
        default=[],
        help="Exact priority Pixabay query; repeat for multiple location-led searches.",
    )
    parser.add_argument(
        "--wide-aerial-only",
        action="store_true",
        help="Exclude abstract/close-up metadata hits and strongly prefer aerial, FPV, and wide footage.",
    )
    parser.add_argument(
        "--visual-style",
        "--visual-cohesion-profile",
        dest="visual_cohesion_profile",
        default="auto",
        help="Free-form task style request; auto derives it from theme, references, and BGM.",
    )
    parser.add_argument(
        "--exclude-pixabay-id",
        dest="excluded_pixabay_ids",
        action="append",
        default=[],
        help="Exclude a visually rejected Pixabay asset ID; repeat after contact-sheet review.",
    )
    parser.add_argument("--dry-run", action="store_true", help="search and rank without downloading full videos")
    parser.add_argument("--result-json", help="optional path for the returned stage result")
    parser.add_argument("--usage-mode", choices=USAGE_MODES, default="local_evaluation")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "update-usage":
            summary = update_usage_intervals(args.manifest, args.edit_plan)
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            return 0
        if not args.theme:
            parser.error("--theme is required unless using the update-usage subcommand")
        result = run_pixabay_pipeline(
            theme=args.theme,
            style_profile=_load_profile_argument(args.style_profile),
            audio_profile=_load_profile_argument(args.audio_profile),
            material_root=args.material_root,
            cache_dir=args.cache_dir,
            desired_count=args.desired_count,
            aspect_ratio=args.aspect_ratio,
            min_resolution=_parse_resolution(args.min_resolution),
            dry_run=args.dry_run,
            timeline_plan=args.timeline_plan,
            candidate_pool_multiplier=args.candidate_pool_multiplier,
            max_search_pages=args.max_search_pages,
            priority_queries=args.priority_queries,
            wide_aerial_only=args.wide_aerial_only,
            visual_cohesion_profile=args.visual_cohesion_profile,
            excluded_pixabay_ids=args.excluded_pixabay_ids,
            usage_mode=args.usage_mode,
        )
        if args.result_json:
            _atomic_write_json(Path(args.result_json).expanduser().resolve(), result)
        console = {
            "status": result["status"],
            "selected_count": result["selected_count"],
            "desired_count": result["desired_count"],
            "sources_manifest": result["sources_manifest"],
            "candidate_count": result["candidate_count"],
            "rejection_count": len(result["rejections"]),
        }
        print(json.dumps(console, ensure_ascii=False, indent=2))
        return 0 if result["status"] in {"ok", "dry_run"} else 2
    except (PixabayPipelineError, ValueError) as exc:
        # Error sanitization is repeated at the outermost boundary so neither
        # requests nor a caller-provided exception can echo the credential.
        print(f"error: {_safe_error(exc, os.environ.get('PIXABAY_API_KEY'))}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
