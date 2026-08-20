"""Deterministic identity checks for frozen Golden fixtures."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


IDENTITY_SCHEMA_VERSION = "v1.4.4-golden-fixture-identity.1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _timeline_identity(path: Path, timeline: Mapping[str, Any]) -> dict[str, Any]:
    slots = timeline.get("slots")
    if not isinstance(slots, list):
        raise ValueError(f"Frozen timeline has no slots list: {path}")
    audiomap_digest = timeline.get("audiomap_digest")
    return {
        "raw_sha256": sha256_file(path),
        "schema_version": timeline.get("schema_version"),
        "planner": timeline.get("planner"),
        "audiomap_digest": audiomap_digest,
        "plan_digest": timeline.get("plan_digest"),
        "duration_seconds": timeline.get("duration_seconds"),
        "rhythm_mode": timeline.get("rhythm_mode"),
        "slot_count": len(slots),
        "slots_sha256": _canonical_sha256(slots),
    }


def build_fixture_identity(
    *,
    golden_id: str,
    bgm_path: Path,
    timeline_path: Path,
    audiomap_path: Path,
    editing_grammar_path: Path,
    style_profile_path: Path,
    aspect_ratio: str,
    requested_duration_seconds: float,
    legacy_manifest_fields: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the identity used before a frozen Golden run.

    Paths are evidence only; identity comparisons use content and contract
    fields so a fixture can move between evidence directories without drifting.
    """

    timeline = read_json(timeline_path)
    audiomap = read_json(audiomap_path)
    grammar = read_json(editing_grammar_path)
    style = read_json(style_profile_path)
    legacy = dict(legacy_manifest_fields or {})
    return {
        "schema_version": IDENTITY_SCHEMA_VERSION,
        "golden_id": golden_id,
        "bgm": {"path": str(bgm_path.resolve()), "sha256": sha256_file(bgm_path)},
        "contract": {
            "aspect_ratio": aspect_ratio,
            "requested_duration_seconds": requested_duration_seconds,
            "target_duration_seconds": timeline.get("duration_seconds"),
            "rhythm_mode": timeline.get("rhythm_mode"),
            "slot_count": len(timeline.get("slots") or []),
        },
        "timeline": _timeline_identity(timeline_path, timeline),
        "audiomap": {
            "raw_sha256": sha256_file(audiomap_path),
            "schema_version": audiomap.get("schema_version"),
            "analysis_digest": audiomap.get("analysis_digest"),
            "duration_seconds": audiomap.get("duration_seconds"),
            "rhythm_mode": (audiomap.get("rhythm_mode") or {}).get("mode"),
        },
        "editing_grammar": {
            "raw_sha256": sha256_file(editing_grammar_path),
            "schema_version": grammar.get("schema_version"),
            "status": grammar.get("status"),
        },
        "style_profile": {
            "raw_sha256": sha256_file(style_profile_path),
            "schema_version": style.get("schema_version"),
            "profile_digest": style.get("profile_digest"),
        },
        "legacy_manifest_fields": {
            "material_input_fingerprint_sha256": legacy.get("material_input_fingerprint_sha256"),
            "timeline_report_fixture_sha256": legacy.get("timeline_report_fixture_sha256"),
        },
    }


_COMPARE_PATHS = (
    ("bgm.sha256", ("bgm", "sha256")),
    ("contract.aspect_ratio", ("contract", "aspect_ratio")),
    ("contract.requested_duration_seconds", ("contract", "requested_duration_seconds")),
    ("contract.target_duration_seconds", ("contract", "target_duration_seconds")),
    ("contract.rhythm_mode", ("contract", "rhythm_mode")),
    ("contract.slot_count", ("contract", "slot_count")),
    ("timeline.raw_sha256", ("timeline", "raw_sha256")),
    ("timeline.plan_digest", ("timeline", "plan_digest")),
    ("timeline.audiomap_digest", ("timeline", "audiomap_digest")),
    ("timeline.duration_seconds", ("timeline", "duration_seconds")),
    ("timeline.rhythm_mode", ("timeline", "rhythm_mode")),
    ("timeline.slot_count", ("timeline", "slot_count")),
    ("timeline.slots_sha256", ("timeline", "slots_sha256")),
    ("audiomap.raw_sha256", ("audiomap", "raw_sha256")),
    ("audiomap.analysis_digest", ("audiomap", "analysis_digest")),
    ("editing_grammar.raw_sha256", ("editing_grammar", "raw_sha256")),
    ("style_profile.raw_sha256", ("style_profile", "raw_sha256")),
)


def _get(value: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = value
    for part in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(part)
    return current


def compare_fixture_identity(
    expected: Mapping[str, Any], actual: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Return mismatches; an empty list is the only identity PASS."""

    mismatches: list[dict[str, Any]] = []
    for label, path in _COMPARE_PATHS:
        expected_value = _get(expected, path)
        actual_value = _get(actual, path)
        if expected_value != actual_value:
            mismatches.append(
                {
                    "field": label,
                    "expected": expected_value,
                    "actual": actual_value,
                }
            )
    return mismatches

