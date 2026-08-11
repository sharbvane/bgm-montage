"""Shared local-evaluation versus publication policy for material acquisition."""

from __future__ import annotations

from typing import Any, Mapping


USAGE_MODES = ("local_evaluation", "publish")


def normalize_usage_mode(value: str | None) -> str:
    mode = str(value or "local_evaluation").strip().lower()
    if mode not in USAGE_MODES:
        raise ValueError(f"usage_mode must be one of {USAGE_MODES}; got {value!r}")
    return mode


def material_usage_policy(value: str | None) -> dict[str, Any]:
    mode = normalize_usage_mode(value)
    local = mode == "local_evaluation"
    return {
        "mode": mode,
        "purpose": (
            "local learning, testing, algorithm validation, and montage quality evaluation"
            if local
            else "explicit public, commercial, or external distribution"
        ),
        "quality_over_source": local,
        "licensing_queries_generated_by_pipeline": False,
        "license_metadata_used_for_ranking": False,
        "license_or_copyright_ranking_weight": 0.0,
        "ordinary_youtube_source_penalty": 0.0,
        "authorization_filter_applied": False,
        "recurring_rights_warning": not local,
        "task_specific_rights_policy_required": not local,
    }


def apply_usage_policy(payload: Mapping[str, Any], value: str | None) -> dict[str, Any]:
    mode = normalize_usage_mode(value)
    updated = dict(payload)
    updated["usage_mode"] = mode
    updated["material_usage_policy"] = material_usage_policy(mode)
    updated.pop("attribution_notice", None)
    updated.pop("publication_mode_notice", None)
    if mode == "publish":
        updated["publication_mode_notice"] = (
            "Publish mode was explicitly selected. Define and apply a task-specific rights policy before distribution."
        )
    return updated
