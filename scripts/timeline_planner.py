#!/usr/bin/env python3
"""Pre-download music-event timeline planning for bgm-montage v1.4.

The planner consumes the deterministic ``audiomap.json`` produced by
``analyze_bgm.py`` and creates semantic shot slots before media search or
download.  It has no network or model dependency and also accepts the v1.1
top-level beat/section aliases.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from montage import (
    _grammar_duration,
    _grammar_event_offsets,
    _grammar_event_weights,
    _mapping_at,
    _normalize_scale,
    _normalize_transition,
    _transition_choice,
)
from music_event_contract import normalize_music_event_contract


SCHEMA_VERSION = "1.2"
PLANNER_VERSION = "1.4.0"

_EVENT_GROUP = {
    "strong_accent": "accents",
    "accent": "accents",
    "downbeat": "accents",
    "onset": "accents",
    "weak_beat": "beats",
    "beat": "beats",
    "phrase": "phrases",
    "phrase_boundary": "phrases",
    "section": "sections",
    "section_boundary": "sections",
    "pause": "pauses",
    "pause_edge": "pauses",
    "hard_stop": "pauses",
    "drop": "drops",
    "surge": "surges",
    "climax": "climaxes",
}

_PREFERRED_EVENT = {
    "strong_accent": "accent",
    "weak_beat": "beat",
    "phrase": "phrase_boundary",
    "section": "section_boundary",
    "pause": "hard_stop",
    "pause_edge": "hard_stop",
}


class TimelinePlanningError(RuntimeError):
    """Raised when an audiomap cannot produce a safe forward timeline."""


def _number(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return result if math.isfinite(result) else default


def _clamp(value: Any, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, _number(value)))


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            temporary = Path(handle.name)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink(missing_ok=True)


def _load_json(path: str | os.PathLike[str] | None) -> dict[str, Any]:
    if not path:
        return {}
    source = Path(path).expanduser().resolve()
    try:
        value = json.loads(source.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise TimelinePlanningError(f"Invalid JSON input: {source}") from exc
    if not isinstance(value, dict):
        raise TimelinePlanningError(f"Expected a JSON object: {source}")
    return value


def _duration(audiomap: Mapping[str, Any], requested: float | None) -> float:
    if requested is not None:
        value = _number(requested, -1.0)
        if value <= 0:
            raise TimelinePlanningError("duration must be a positive finite number")
        available = _number(audiomap.get("duration_seconds"), 0.0)
        if not available:
            available = _number(
                (audiomap.get("input") or {}).get("analysis_window", {}).get("duration"),
                0.0,
            )
        return min(value, available) if available > 0 else value
    value = _number(audiomap.get("duration_seconds"), 0.0)
    if not value:
        value = _number(
            (audiomap.get("input") or {}).get("analysis_window", {}).get("duration"),
            0.0,
        )
    if value <= 0:
        raise TimelinePlanningError("audiomap does not contain a usable duration")
    return value


def _time_item(value: Any, event_type: str) -> dict[str, Any] | None:
    if isinstance(value, (int, float)):
        return {"type": event_type, "time": _number(value), "strength": 0.55}
    if not isinstance(value, Mapping):
        return None
    time_value = None
    for key in ("time", "time_seconds", "start", "start_seconds", "boundary"):
        if value.get(key) is not None:
            time_value = _number(value.get(key), -1.0)
            break
    if time_value is None or time_value < 0:
        return None
    return {
        "type": str(value.get("type") or event_type),
        "time": time_value,
        "strength": _clamp(value.get("strength", value.get("confidence", 0.55))),
        "confidence": _clamp(value.get("confidence", 0.55)),
    }


def _events(audiomap: Mapping[str, Any], duration: float) -> list[dict[str, Any]]:
    return [
        {key: value for key, value in event.items() if key != "group"}
        for event in normalize_music_event_contract(audiomap, duration).get("events", [])
    ]


def _sections(audiomap: Mapping[str, Any], duration: float) -> list[dict[str, Any]]:
    raw = audiomap.get("sections")
    sections: list[dict[str, Any]] = []
    if isinstance(raw, list):
        for index, value in enumerate(raw):
            if not isinstance(value, Mapping):
                continue
            start = max(0.0, _number(value.get("start"), -1.0))
            end = min(duration, _number(value.get("end"), -1.0))
            if start < end:
                sections.append({**dict(value), "index": int(value.get("index", index)), "start": start, "end": end})
    if not sections:
        mode = str((audiomap.get("rhythm_mode") or {}).get("mode") or "phrase_flow")
        return [
            {
                "index": 0,
                "start": 0.0,
                "end": duration,
                "duration": duration,
                "role": "climax",
                "estimated_mood": "balanced/forward",
                "rhythm_mode": mode,
                "energy": {"mean": _number((audiomap.get("global") or {}).get("mean_energy"), 0.5)},
                "edit_guidance": dict(audiomap.get("editing_guidance") or {}),
            }
        ]
    sections.sort(key=lambda item: (item["start"], item["end"]))
    normalized: list[dict[str, Any]] = []
    cursor = 0.0
    for section in sections:
        start = max(cursor, section["start"])
        end = max(start, section["end"])
        if start > cursor + 0.02:
            normalized.append(
                {
                    "index": len(normalized),
                    "start": cursor,
                    "end": start,
                    "role": "build" if normalized else "intro",
                    "estimated_mood": "balanced/forward",
                    "rhythm_mode": str((audiomap.get("rhythm_mode") or {}).get("mode") or "phrase_flow"),
                    "energy": {"mean": 0.5},
                    "edit_guidance": {},
                }
            )
        if end > start + 0.02:
            normalized.append({**section, "index": len(normalized), "start": start, "end": end})
            cursor = end
        if cursor >= duration - 0.02:
            break
    if cursor < duration - 0.02:
        tail = dict(normalized[-1]) if normalized else {}
        normalized.append(
            {
                **tail,
                "index": len(normalized),
                "start": cursor,
                "end": duration,
                "role": "outro",
            }
        )
    return normalized


def _style_shot_target(style_profile: Mapping[str, Any] | None) -> float | None:
    if not isinstance(style_profile, Mapping):
        return None
    wanted = {
        "average_shot_duration",
        "average_shot_duration_seconds",
        "median_shot_duration",
        "median_shot_duration_seconds",
    }

    def visit(value: Any) -> float | None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                if str(key).lower() in wanted:
                    number = _number(item, 0.0)
                    if 0.35 <= number <= 8.0:
                        return number
            for item in value.values():
                result = visit(item)
                if result is not None:
                    return result
        elif isinstance(value, list):
            for item in value:
                result = visit(item)
                if result is not None:
                    return result
        return None

    return visit(style_profile)


def _shot_target(section: Mapping[str, Any], style_target: float | None) -> tuple[float, float, float, float]:
    guidance = section.get("edit_guidance") if isinstance(section.get("edit_guidance"), Mapping) else {}
    recommendation = guidance.get("recommended_shot_duration_seconds")
    low = target = high = 0.0
    if isinstance(recommendation, Mapping):
        low = _number(recommendation.get("min"), 0.0)
        target = _number(recommendation.get("target"), 0.0)
        high = _number(recommendation.get("max"), 0.0)
    elif isinstance(recommendation, (list, tuple)) and len(recommendation) >= 2:
        low, high = _number(recommendation[0]), _number(recommendation[1])
        target = (low + high) / 2.0
    elif isinstance(recommendation, (int, float)):
        target = _number(recommendation)
    energy_root = section.get("energy") if isinstance(section.get("energy"), Mapping) else {}
    energy = _clamp(energy_root.get("mean", section.get("energy", 0.5)))
    mode = str(section.get("rhythm_mode") or guidance.get("rhythm_mode") or "phrase_flow")
    role = str(section.get("role") or "build").strip().lower()
    if target <= 0:
        target = 0.72 + (1.0 - energy) * 1.25 if mode == "beat_cut" else 2.0 + (1.0 - energy) * 2.8
    if style_target is not None:
        target = 0.78 * target + 0.22 * style_target
    low = low if low > 0 else target * (0.62 if mode == "beat_cut" else 0.68)
    high = high if high > 0 else target * (1.65 if mode == "beat_cut" else 1.55)
    # The audio analyzer can legitimately emit similar raw duration ranges for
    # adjacent sections.  Section roles must still create an observable pacing
    # arc: climax/drop sections become denser, while intro/break/outro breathe.
    # Apply this to all three bounds so an explicit analyzer range cannot erase
    # the structural contrast.
    if mode != "beat_cut":
        # phrase_flow must preserve phrase/section anchors instead of forcing a
        # denser pseudo-beat grid; its per-section analyzer guidance remains
        # authoritative.
        role_multiplier = 1.0
    elif role in {"drop", "climax"}:
        role_multiplier = 0.78
    elif role in {"intro", "break", "outro"}:
        role_multiplier = 1.18
    else:
        role_multiplier = 1.0
    low *= role_multiplier
    target *= role_multiplier
    high *= role_multiplier
    low = max(0.36, min(4.0, low))
    high = max(low + 0.12, min(8.0, high))
    target = max(low, min(high, target))
    intensity = _clamp(
        _number(guidance.get("cut_intensity"), 0.18 + 0.70 * energy)
        + (0.12 if role in {"drop", "climax"} else -0.08 if role in {"intro", "break", "outro"} else 0.0)
    )
    return low, target, high, intensity


def _boundary(
    current: float,
    section_end: float,
    target_length: float,
    minimum: float,
    maximum: float,
    mode: str,
    events: Sequence[Mapping[str, Any]],
    event_weights: Mapping[str, float] | None = None,
    event_offsets: Mapping[str, float] | None = None,
    allowed_event_types: Sequence[str] | None = None,
) -> tuple[float, dict[str, Any]]:
    target = min(section_end, current + target_length)
    low = current + minimum
    high = min(section_end, current + maximum)
    if high <= low + 1e-6:
        return section_end, {"type": "section_boundary", "time": section_end, "strength": 0.72}
    priorities = (
        ("drop", "climax", "hard_stop", "surge", "section_boundary", "accent", "downbeat", "beat", "phrase_boundary", "onset")
        if mode == "beat_cut"
        else ("drop", "climax", "hard_stop", "surge", "section_boundary", "phrase_boundary", "accent", "onset")
    )
    rank = {name: index for index, name in enumerate(priorities)}
    allowed_types = set(allowed_event_types or priorities)
    candidates: list[tuple[float, float, Mapping[str, Any]]] = []
    span = max(0.20, high - low)
    elastic_high = min(section_end, current + maximum * 1.25)
    elastic_types = {"drop", "climax", "hard_stop", "surge", "section_boundary", "phrase_boundary"}
    for event in events:
        event_type = str(event.get("type") or "")
        time_value = _number(event.get("time"), -1.0)
        group = _EVENT_GROUP.get(event_type)
        learned_offset = _number((event_offsets or {}).get(group), 0.0) if group else 0.0
        learned_time = time_value + learned_offset
        allowed_high = elastic_high if event_type in elastic_types else high
        if event_type not in rank or event_type not in allowed_types or not low - 1e-6 <= learned_time <= allowed_high + 1e-6:
            continue
        distance = abs(learned_time - target) / span
        if learned_time > high:
            distance += (learned_time - high) / max(0.15, maximum) * 0.18
        score = distance + rank[event_type] * 0.065 - _clamp(event.get("strength", 0.5)) * 0.08
        if event_weights and group:
            score += (1.0 - _clamp(event_weights.get(group, 1.0))) * 0.28
        adjusted = dict(event)
        if learned_offset:
            adjusted.update(
                {
                    "source_time": time_value,
                    "time": learned_time,
                    "learned_offset_seconds": learned_offset,
                }
            )
        candidates.append((score, learned_time, adjusted))
    if candidates:
        _, time_value, event = min(candidates, key=lambda item: (item[0], item[1], str(item[2].get("type"))))
        return time_value, dict(event)
    fallback = max(low, min(high, target))
    return fallback, {"type": "flow_grid", "time": fallback, "strength": 0.25, "confidence": 0.35}


def _visual_intent(role: str, energy: float, emphasis_type: str | None) -> tuple[list[str], str, str]:
    if emphasis_type in {"drop", "climax"} or role in {"drop", "climax"}:
        return (
            ["monumental environment", "high-impact action", "strong spatial reveal"],
            "wide",
            "strong_forward_or_aerial",
        )
    if emphasis_type == "hard_stop":
        return (["clear visual pause", "single readable subject"], "detail", "static_or_decelerating")
    if role == "build" or emphasis_type == "surge":
        return (["progressive action", "approach or reveal", "increasing visual scale"], "medium", "forward_or_lateral")
    if role in {"intro", "break", "outro"} or energy < 0.34:
        return (["atmospheric environment", "calm establishing image"], "wide", "stable_or_gentle")
    return (["theme-relevant environment", "contrasting subject detail"], "medium", "moderate_varied")


def _transition(mode: str, event_type: str, first: bool, last: bool) -> str:
    if first:
        return "fade_in"
    if last:
        return "fade_out"
    if event_type in {"drop", "climax", "hard_stop", "surge", "accent", "downbeat", "beat"}:
        return "hard_cut"
    if mode == "phrase_flow" and event_type in {"phrase_boundary", "section_boundary"}:
        return "short_dissolve"
    return "hard_cut"


def plan_timeline_slots(
    audiomap: Mapping[str, Any],
    duration: float | None = None,
    style_profile: Mapping[str, Any] | None = None,
    editing_grammar: Mapping[str, Any] | None = None,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build deterministic pre-download shot slots snapped to measured events."""

    if not isinstance(audiomap, Mapping):
        raise TimelinePlanningError("audiomap must be a mapping")
    total = _duration(audiomap, duration)
    event_contract = normalize_music_event_contract(audiomap, total)
    events = [
        {key: value for key, value in event.items() if key != "group"}
        for event in event_contract.get("events", [])
    ]
    sections = _sections(audiomap, total)
    style_target = _style_shot_target(style_profile)
    grammar = dict(editing_grammar or {})
    grammar_digest = (
        hashlib.sha256(
            json.dumps(grammar, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if grammar
        else None
    )
    raw_event_weights = _mapping_at(
        grammar, ("event_weights", "boundary_event_weights", "cut_event_weights", "primary_distribution")
    )
    raw_event_offsets = _mapping_at(grammar, ("boundary_offsets_seconds", "event_offsets_seconds"))
    raw_durations = _mapping_at(
        grammar, ("energy_shot_duration_seconds", "shot_duration_by_energy", "energy_duration")
    )
    event_weights = _grammar_event_weights(grammar) if raw_event_weights else {}
    event_offsets = _grammar_event_offsets(grammar) if raw_event_offsets else {}
    scale_matrix = _mapping_at(grammar, ("scale_transition_matrix", "shot_scale_transition_matrix"))
    motion_matrix = _mapping_at(grammar, ("motion_transition_matrix", "camera_motion_transition_matrix"))
    transition_distribution = _mapping_at(
        grammar, ("transition_distribution", "supported_transition_distribution")
    )
    reliability = _mapping_at(grammar, ("reliability",))
    reliability_score = _clamp(reliability.get("score", 1.0)) if reliability else 1.0
    event_weights = {
        key: 1.0 + (value - 1.0) * reliability_score
        for key, value in event_weights.items()
    }
    event_offsets = {key: value * reliability_score for key, value in event_offsets.items()}
    beat_times = sorted(_number(event.get("time")) for event in events if event.get("type") == "beat")
    beat_gaps = [right - left for left, right in zip(beat_times, beat_times[1:]) if 0.18 <= right - left <= 2.0]
    beat_seconds = sorted(beat_gaps)[len(beat_gaps) // 2] if beat_gaps else 0.5
    learned_transition = ""
    if transition_distribution:
        transition_name = max(
            transition_distribution,
            key=lambda key: (_number(transition_distribution.get(key)), str(key)),
        )
        learned_transition = _normalize_transition(transition_name)
    policy = grammar.get("montage_policy") if isinstance(grammar.get("montage_policy"), Mapping) else {}
    ending = (
        dict(policy.get("ending"))
        if isinstance(policy.get("ending"), Mapping)
        else _mapping_at(grammar, ("ending", "ending_structure"))
    )
    applied_fields = [
        name
        for name, value in (
            ("boundary_event_weights", raw_event_weights),
            ("boundary_offsets_seconds", raw_event_offsets),
            ("shot_duration_by_energy", raw_durations),
            ("scale_transition_matrix", scale_matrix),
            ("motion_transition_matrix", motion_matrix),
            ("transition_distribution", transition_distribution),
            ("ending", ending),
        )
        if value
    ]
    global_mode = str(event_contract.get("mode") or "phrase_flow")
    config = dict(config or {})
    minimum_override = _number(config.get("minimum_shot_seconds"), 0.0)
    maximum_override = _number(config.get("maximum_shot_seconds"), 0.0)
    terminal_minimum = max(
        0.32,
        min(1.50, _number(config.get("minimum_terminal_shot_seconds"), 0.50)),
    )
    slots: list[dict[str, Any]] = []
    snapped = 0
    fallback = 0
    previous_scale = "wide"
    previous_motion = "static_like"

    key_types = {"drop", "surge", "climax", "hard_stop"}
    for section in sections:
        start = max(0.0, _number(section.get("start"), 0.0))
        end = min(total, _number(section.get("end"), total))
        if end <= start + 0.02:
            continue
        mode = str(section.get("rhythm_mode") or global_mode)
        if mode not in {"beat_cut", "phrase_flow"}:
            mode = global_mode
        low, target, high, intensity = _shot_target(section, style_target)
        energy_root = section.get("energy") if isinstance(section.get("energy"), Mapping) else {}
        section_energy = _clamp(energy_root.get("mean", section.get("energy", 0.5)))
        learned_target = target
        if raw_durations:
            learned_target = _grammar_duration(grammar, section_energy, target, beat_seconds)
            blended_target = target * (1.0 - reliability_score) + learned_target * reliability_score
            duration_scale = blended_target / max(target, 1e-9)
            low *= duration_scale
            target = blended_target
            high *= duration_scale
            low = max(0.36, min(4.0, low))
            high = max(low + 0.12, min(8.0, high))
            target = max(low, min(high, target))
        if minimum_override > 0:
            low = max(0.24, minimum_override)
        if maximum_override > 0:
            high = max(low + 0.12, maximum_override)
        current = start
        while current < end - 0.02:
            remaining = end - current
            # Never create a terminal flash just to cover a small section
            # remainder.  Merge it into the previous readable shot instead.
            if remaining < terminal_minimum and slots:
                previous = slots[-1]
                previous["end"] = round(end, 4)
                previous["output_end"] = round(end, 4)
                previous["duration"] = round(previous["end"] - previous["start"], 4)
                previous["output_duration"] = previous["duration"]
                current = end
                break
            effective_low = min(low, max(0.24, remaining * 0.48))
            effective_target = min(target, remaining)
            effective_high = min(high, remaining)
            boundary, anchor = _boundary(
                current,
                end,
                effective_target,
                effective_low,
                max(effective_low + 0.12, effective_high),
                mode,
                events,
                event_weights,
                event_offsets,
                event_contract.get("allowed_event_types"),
            )
            boundary = min(end, max(current + 0.24, boundary))
            if end - boundary < terminal_minimum:
                boundary = end
                anchor = {"type": "section_boundary", "time": end, "strength": 0.72, "confidence": 0.65}
            anchor_type = str(anchor.get("type") or "flow_grid")
            if anchor_type == "flow_grid":
                fallback += 1
            else:
                snapped += 1
            start_key = min(
                (
                    event
                    for event in events
                    if str(event.get("type")) in key_types
                    and abs(_number(event.get("time")) - current) <= 0.10
                ),
                key=lambda item: abs(_number(item.get("time")) - current),
                default=None,
            )
            emphasis_type = str(start_key.get("type")) if start_key else (anchor_type if anchor_type in key_types else None)
            energy_root = section.get("energy") if isinstance(section.get("energy"), Mapping) else {}
            energy = _clamp(energy_root.get("mean", section.get("energy", 0.5)))
            role = str(section.get("role") or "build")
            mood = str(section.get("estimated_mood") or section.get("mood") or "balanced/forward")
            content, scale, motion = _visual_intent(role, energy, emphasis_type)
            prior_scale, prior_motion = previous_scale, previous_motion
            if scale_matrix:
                scale = _normalize_scale(_transition_choice(scale_matrix, previous_scale, [scale], len(slots)))
            if motion_matrix:
                motion = _transition_choice(motion_matrix, previous_motion, [motion], len(slots))
            is_last = boundary >= total - 0.02
            transition = _transition(mode, anchor_type, not slots, is_last)
            if learned_transition and slots and not is_last:
                transition = learned_transition
            anchor_payload = {
                "type": anchor_type,
                "time": round(_number(anchor.get("time"), boundary), 4),
                "strength": round(_clamp(anchor.get("strength", 0.5)), 4),
                "confidence": round(_clamp(anchor.get("confidence", 0.5)), 4),
                "delta_seconds": round(boundary - _number(anchor.get("time"), boundary), 4),
            }
            if anchor.get("source_time") is not None:
                anchor_payload["source_event_time"] = round(_number(anchor.get("source_time")), 4)
                anchor_payload["learned_offset_seconds"] = round(
                    _number(anchor.get("learned_offset_seconds")), 4
                )
            slot = {
                "index": len(slots),
                "start": round(current, 4),
                "end": round(boundary, 4),
                "duration": round(boundary - current, 4),
                "output_start": round(current, 4),
                "output_end": round(boundary, 4),
                "output_duration": round(boundary - current, 4),
                "section_index": int(section.get("index", 0)),
                "section_role": role,
                "rhythm_mode": mode,
                "mood": mood,
                "energy": round(energy, 4),
                "energy_level": "high" if energy >= 0.67 else "low" if energy < 0.34 else "medium",
                "cut_intensity": round(intensity, 4),
                "recommended_content": content,
                "recommended_visual_content": content,
                "recommended_shot_scale": scale,
                "recommended_motion": motion,
                "is_emphasis": bool(emphasis_type),
                "is_key_shot": bool(emphasis_type),
                "emphasis_event": emphasis_type,
                "anchor_event": anchor_payload,
                "transition": transition,
                "recommended_transition": transition,
            }
            if grammar:
                grammar_influence = {}
                group = _EVENT_GROUP.get(anchor_type)
                if raw_event_weights and group:
                    grammar_influence["boundary_event_weight"] = round(event_weights.get(group, 1.0), 4)
                if raw_event_offsets and group:
                    grammar_influence["boundary_offset_seconds"] = round(event_offsets.get(group, 0.0), 4)
                if raw_durations:
                    grammar_influence.update(
                        {
                            "reference_duration_target_seconds": round(learned_target, 4),
                            "duration_blend_reliability": round(reliability_score, 4),
                        }
                    )
                if scale_matrix:
                    grammar_influence["scale_transition"] = f"{prior_scale}->{scale}"
                if motion_matrix:
                    grammar_influence["motion_transition"] = f"{prior_motion}->{motion}"
                if learned_transition:
                    grammar_influence["transition_target"] = learned_transition
                slot["grammar_influence"] = grammar_influence
            slots.append(slot)
            previous_scale, previous_motion = scale, motion
            current = boundary

    multiplier_value = ending.get(
        "final_shot_multiplier",
        ending.get("last_shot_multiplier", ending.get("last_shot_duration_multiplier")),
    )
    multiplier_present = multiplier_value is not None
    learned_ending_multiplier = max(0.6, min(2.0, _number(multiplier_value, 1.0)))
    ending_multiplier = 1.0 + (learned_ending_multiplier - 1.0) * reliability_score
    preferred_end_event = str(ending.get("preferred_end_event") or "").strip().lower()
    if grammar and ending and slots:
        ending_target = {
            "last_shot_duration_multiplier": round(ending_multiplier, 4),
            "preferred_end_event": preferred_end_event if reliability_score > 0.0 else None,
            "fade_out_seconds": round(
                max(0.0, _number(ending.get("fade_out_seconds", ending.get("final_fade_seconds")), 0.0)),
                4,
            ),
            "hold_last_frame": bool(ending.get("hold_last_frame", False)),
        }
        slots[-1]["ending_target"] = ending_target
        slots[-1].setdefault("grammar_influence", {})["ending"] = ending_target
    if (
        multiplier_present
        and reliability_score > 0.0
        and len(slots) >= 2
        and slots[-2]["section_index"] == slots[-1]["section_index"]
    ):
        previous, final = slots[-2], slots[-1]
        typical_values = sorted(float(slot["duration"]) for slot in slots[:-1])
        typical = typical_values[len(typical_values) // 2]
        low_boundary = float(previous["start"]) + 0.24
        high_boundary = total - terminal_minimum
        target_boundary = max(low_boundary, min(high_boundary, total - typical * ending_multiplier))
        preferred_type = _PREFERRED_EVENT.get(preferred_end_event, preferred_end_event)
        preferred = [
            event
            for event in events
            if str(event.get("type")) == preferred_type
            and low_boundary <= _number(event.get("time"), -1.0) <= high_boundary
            and abs(_number(event.get("time")) - target_boundary) <= max(0.35, typical * 0.35)
        ]
        chosen = min(preferred, key=lambda event: abs(_number(event.get("time")) - target_boundary), default=None)
        new_boundary = _number(chosen.get("time"), target_boundary) if chosen else target_boundary
        old_boundary = float(previous["end"])
        if abs(new_boundary - old_boundary) > 0.004:
            old_type = str((previous.get("anchor_event") or {}).get("type") or "flow_grid")
            new_type = str(chosen.get("type")) if chosen else "flow_grid"
            if old_type == "flow_grid" and new_type != "flow_grid":
                fallback -= 1
                snapped += 1
            elif old_type != "flow_grid" and new_type == "flow_grid":
                snapped -= 1
                fallback += 1
            previous.update(
                {
                    "end": round(new_boundary, 4),
                    "output_end": round(new_boundary, 4),
                    "duration": round(new_boundary - float(previous["start"]), 4),
                    "output_duration": round(new_boundary - float(previous["start"]), 4),
                    "anchor_event": {
                        "type": new_type,
                        "time": round(new_boundary, 4),
                        "strength": round(_clamp(chosen.get("strength", 0.5)), 4) if chosen else 0.25,
                        "confidence": round(_clamp(chosen.get("confidence", 0.5)), 4) if chosen else 0.35,
                        "delta_seconds": 0.0,
                        "ending_target": True,
                    },
                }
            )
            previous["transition"] = previous["recommended_transition"] = (
                learned_transition
                if learned_transition and int(previous["index"]) > 0
                else _transition(
                    str(previous.get("rhythm_mode") or global_mode),
                    new_type,
                    int(previous["index"]) == 0,
                    False,
                )
            )
            final.update(
                {
                    "start": round(new_boundary, 4),
                    "output_start": round(new_boundary, 4),
                    "duration": round(total - new_boundary, 4),
                    "output_duration": round(total - new_boundary, 4),
                }
            )
            previous.setdefault("grammar_influence", {})["ending_boundary_adjustment_seconds"] = round(
                new_boundary - old_boundary, 4
            )
        slots[-1]["ending_target"]["actual_last_shot_duration_seconds"] = slots[-1]["duration"]

    if not slots or abs(slots[0]["start"]) > 0.02 or abs(slots[-1]["end"] - total) > 0.03:
        raise TimelinePlanningError("planner could not cover the full audiomap duration")
    digest_payload = {
        "audiomap_digest": audiomap.get("analysis_digest"),
        "duration": round(total, 5),
        "style_target": style_target,
        "config": config,
        "slots": slots,
        "music_event_contract_digest": event_contract.get("contract_digest"),
    }
    if grammar_digest:
        digest_payload["editing_grammar_digest"] = grammar_digest
    plan_digest = hashlib.sha256(
        json.dumps(digest_payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    plan = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "timeline",
        "planner": {"name": "bgm-montage pre-download timeline planner", "version": PLANNER_VERSION},
        "audiomap_schema_version": audiomap.get("schema_version"),
        "audiomap_digest": audiomap.get("analysis_digest"),
        "plan_digest": plan_digest,
        "duration_seconds": round(total, 4),
        "rhythm_mode": global_mode,
        "music_event_contract": event_contract,
        "sections": sections,
        "key_moments": [
            dict(item)
            for item in (audiomap.get("key_moments") or [])
            if isinstance(item, Mapping) and 0.0 <= _number(item.get("time"), -1.0) <= total
        ],
        "slots": slots,
        "metrics": {
            "slot_count": len(slots),
            "event_snapped_boundary_count": snapped,
            "fallback_boundary_count": fallback,
            "event_snap_ratio": round(snapped / max(1, snapped + fallback), 4),
            "emphasis_slot_count": sum(bool(slot["is_emphasis"]) for slot in slots),
            "beat_cut_slot_count": sum(slot["rhythm_mode"] == "beat_cut" for slot in slots),
            "phrase_flow_slot_count": sum(slot["rhythm_mode"] == "phrase_flow" for slot in slots),
            "minimum_slot_duration_seconds": round(
                min(float(slot["duration"]) for slot in slots), 4
            ),
        },
        "editing_grammar_provided": bool(editing_grammar),
        "planning_contract": [
            "Slot boundaries prefer measured musical events over equal division.",
            "beat_cut may use ordinary beats; phrase_flow prioritizes phrases, sections and energy events.",
            "Drop and climax slots request wide, high-impact, high-motion visuals.",
        ],
    }
    if grammar:
        plan.update(
            {
                "editing_grammar_digest": grammar_digest,
                "editing_grammar_applied": bool(applied_fields),
                "editing_grammar_applied_fields": applied_fields,
            }
        )
    return plan


def build_timeline_slots(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Compatibility alias for callers that use a build-style verb."""

    return plan_timeline_slots(*args, **kwargs)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audiomap", required=True, help="Input audiomap.json (v1.2 or compatible v1.1 profile)")
    parser.add_argument("--output", required=True, help="Output timeline.json")
    parser.add_argument("--duration", type=float, help="Optional shorter target duration")
    parser.add_argument("--style-profile", help="Optional style_profile.json")
    parser.add_argument("--editing-grammar", help="Optional editing_grammar.json")
    parser.add_argument("--summary-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        audiomap = _load_json(args.audiomap)
        style = _load_json(args.style_profile) if args.style_profile else None
        grammar = _load_json(args.editing_grammar) if args.editing_grammar else None
        plan = plan_timeline_slots(audiomap, args.duration, style, grammar)
        destination = Path(args.output).expanduser().resolve()
        _atomic_json(destination, plan)
    except Exception as exc:
        print(f"timeline_planner: {type(exc).__name__}: {exc}", file=__import__("sys").stderr)
        return 1
    summary = {
        "timeline": str(destination),
        "duration_seconds": plan["duration_seconds"],
        "rhythm_mode": plan["rhythm_mode"],
        "slots": len(plan["slots"]),
        "event_snap_ratio": plan["metrics"]["event_snap_ratio"],
    }
    print(json.dumps(summary if args.summary_only else plan, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
