#!/usr/bin/env python3
"""Dynamic visual style, aesthetic, cohesion, and match-cut intelligence.

The module is deliberately domain-neutral.  It derives a task profile from the
user brief, reference analysis, and BGM descriptors instead of selecting from
location/style white-lists.  All image judgements are sampled heuristics and
remain traceable in manifests and edit decisions.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections import Counter
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np


ENGINE_VERSION = "1.3.0"
PROFILE_SCHEMA_VERSION = "1.3"
ASSET_ANALYSIS_SCHEMA_VERSION = 2


_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in",
    "into", "is", "it", "of", "on", "or", "the", "to", "video", "with", "footage",
    "style", "scene", "shot", "clips", "clip", "overall", "unknown", "none", "auto",
}

_CHINESE_TERMS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("航拍", ("aerial", "drone")), ("穿越", ("fpv", "fly through")),
    ("推进", ("push in", "forward motion")), ("拉远", ("pull out", "reveal")),
    ("俯冲", ("dive", "descending aerial")), ("掠过", ("flyover", "gliding")),
    ("横移", ("lateral tracking",)), ("电影感", ("cinematic",)),
    ("史诗", ("epic", "dramatic")), ("忧郁", ("melancholic", "moody")),
    ("孤独", ("solitary", "remote")), ("治愈", ("healing", "serene")),
    ("轻快", ("uplifting", "bright")), ("梦幻", ("dreamlike", "ethereal")),
    ("冷蓝", ("cool", "blue")), ("蓝灰", ("blue", "grey")),
    ("冷峻", ("cool", "austere")), ("暗暖", ("dark warm",)),
    ("低饱和", ("muted", "desaturated")), ("高饱和", ("vibrant", "saturated")),
    ("暖色", ("warm", "golden")), ("明亮", ("bright", "high key")),
    ("微距", ("macro", "close up")), ("细节", ("detail", "tactile")),
    ("清透", ("clear", "natural light")), ("大纵深", ("deep perspective", "layered depth")),
    ("山", ("mountain",)), ("峡谷", ("canyon",)), ("悬崖", ("cliff",)),
    ("森林", ("forest",)), ("草原", ("grassland",)), ("荒原", ("wilderness", "barren")),
    ("沙漠", ("desert",)), ("冰川", ("glacier",)), ("雪", ("snow", "winter")),
    ("海", ("ocean", "coast")), ("瀑布", ("waterfall",)), ("河流", ("river",)),
    ("城市", ("city", "urban")), ("建筑", ("architecture",)),
    ("工厂", ("factory", "industrial")), ("机械", ("machinery",)),
    ("科技", ("technology", "futuristic")), ("人物", ("people", "human")),
    ("美食", ("food", "culinary")), ("料理", ("food", "culinary")),
    ("食材", ("ingredients", "food")), ("烹饪", ("cooking", "kitchen")),
    ("运动", ("sports", "action")),
    ("日出", ("sunrise", "dawn")), ("日落", ("sunset", "dusk")),
    ("黄昏", ("twilight", "dusk")), ("正午", ("midday",)), ("夜景", ("night",)),
    ("阴天", ("overcast",)), ("暴风", ("storm",)), ("雾", ("fog", "mist")),
    ("雨", ("rain",)), ("云", ("clouds",)), ("阳光", ("sunlight",)),
)

_AXES: dict[str, tuple[str, ...]] = {
    "emotion": (
        "serene", "calm", "peaceful", "uplifting", "joyful", "melancholic", "moody",
        "lonely", "solitary", "epic", "dramatic", "tense", "mysterious", "dreamlike",
        "ethereal", "romantic", "playful", "energetic", "powerful", "austere", "remote",
    ),
    "environment": (
        "mountain", "canyon", "cliff", "valley", "forest", "jungle", "grassland", "desert",
        "dune", "glacier", "snow", "ocean", "coast", "beach", "waterfall", "river", "lake",
        "city", "urban", "street", "architecture", "interior", "factory", "industrial",
        "machinery", "technology", "food", "kitchen", "people", "portrait", "sports", "road",
        "transport", "space", "sky", "wilderness", "barren", "landscape",
    ),
    "weather": (
        "fog", "mist", "haze", "rain", "storm", "snow", "overcast", "cloudy", "clouds",
        "wind", "dust", "sunny", "clear", "lightning",
    ),
    "lighting": (
        "sunrise", "dawn", "morning", "midday", "sunset", "dusk", "twilight", "evening",
        "night", "golden hour", "blue hour", "backlight", "low key", "high key", "natural light",
        "neon", "studio light", "silhouette",
    ),
    "capture": (
        "aerial", "drone", "fpv", "cinematic", "wide angle", "deep perspective", "layered depth", "macro", "handheld", "tripod",
        "pov", "tracking", "timelapse", "slow motion", "documentary",
    ),
    "motion": (
        "push in", "pull out", "dive", "descending aerial", "flyover", "gliding", "fly through",
        "forward motion", "lateral tracking", "orbit", "pan", "tilt", "static", "reveal",
    ),
    "color": (
        "cool", "warm", "dark warm", "blue", "cyan", "teal", "green", "golden", "orange",
        "red", "purple", "grey", "gray", "monochrome", "muted", "desaturated", "vibrant", "saturated",
        "natural", "bright", "dark", "low contrast", "high contrast",
    ),
}

_WORLD_TERMS: dict[str, tuple[str, ...]] = {
    "alpine_rock": ("mountain", "canyon", "cliff", "valley", "peak", "rock", "highland", "gorge"),
    "forest_temperate": ("forest", "woodland", "woods", "tree", "grassland", "meadow"),
    "tropical_lush": ("tropical", "jungle", "palm", "rainforest", "lagoon"),
    "arid_desert": ("desert", "dune", "badlands", "barren", "sand", "dust"),
    "cold_ice": ("glacier", "ice", "iceberg", "snow", "winter", "tundra", "frozen"),
    "water_coast": ("ocean", "sea", "coast", "beach", "wave", "waterfall", "river", "lake"),
    "urban_architecture": ("city", "urban", "street", "architecture", "building", "skyline"),
    "industrial_engineering": ("factory", "industrial", "machine", "machinery", "production", "engineering"),
    "technology_digital": ("technology", "digital", "electronics", "futuristic", "network", "robot"),
    "people_lifestyle": ("people", "person", "portrait", "family", "lifestyle", "human", "fashion"),
    "food_culinary": ("food", "cooking", "kitchen", "cuisine", "ingredients", "coffee"),
    "sports_action": ("sports", "athlete", "training", "competition", "fitness", "action"),
    "interior_domestic": ("interior", "home", "room", "office", "studio", "workshop"),
    "sky_space": ("sky", "cloudscape", "stars", "galaxy", "space", "aurora", "night sky"),
    "abstract_graphic": ("abstract", "animation", "cgi", "render", "graphic", "background"),
}

_WORLD_GROUP: dict[str, str] = {
    "alpine_rock": "natural_temperate", "forest_temperate": "natural_temperate",
    "water_coast": "natural_temperate", "cold_ice": "natural_cold",
    "tropical_lush": "natural_tropical", "arid_desert": "natural_arid",
    "sky_space": "atmospheric", "urban_architecture": "built",
    "industrial_engineering": "built", "technology_digital": "built",
    "interior_domestic": "interior", "people_lifestyle": "human",
    "food_culinary": "human", "sports_action": "human", "abstract_graphic": "abstract",
}

_TEXTURE_TERMS: dict[str, tuple[str, ...]] = {
    "water": ("water", "ocean", "sea", "river", "lake", "waterfall", "wave", "ice", "snow"),
    "vapor": ("cloud", "clouds", "fog", "mist", "haze", "steam", "spray", "smoke"),
    "rock": ("mountain", "cliff", "canyon", "rock", "stone", "dune", "desert"),
    "vegetation": ("forest", "tree", "grass", "meadow", "jungle", "plant"),
    "built_lines": ("road", "bridge", "building", "architecture", "factory", "machine"),
    "human_form": ("person", "people", "portrait", "athlete", "hand", "face"),
    "light": ("sun", "sunrise", "sunset", "neon", "glow", "silhouette", "reflection"),
}


def _clamp(value: Any, low: float = 0.0, high: float = 1.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = low
    if not math.isfinite(number):
        number = low
    return max(low, min(high, number))


def _unique(values: Iterable[Any], limit: int | None = None) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = re.sub(r"\s+", " ", str(value).strip().lower())
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
        if limit is not None and len(result) >= limit:
            break
    return result


def english_terms(value: Any) -> list[str]:
    text = unicodedata.normalize("NFKC", str(value))
    lowered = text.lower()
    words = re.findall(r"[A-Za-z][A-Za-z0-9-]{1,30}", text)
    terms = [word.replace("-", " ").lower() for word in words]
    # Preserve meaningful multi-word camera/light/color phrases before
    # stopword removal (for example, ``push in`` would otherwise become only
    # ``push`` and fail to match the motion axis).
    phrases = {
        term
        for vocabulary in _AXES.values()
        for term in vocabulary
        if " " in term and term in lowered
    }
    terms.extend(sorted(phrases))
    for chinese, mapped in _CHINESE_TERMS:
        if chinese in text:
            terms.extend(mapped)
    return _unique(term for term in terms if term not in _STOPWORDS)


def _proper_location_phrases(text: str) -> list[str]:
    """Keep user-supplied proper names without maintaining a place whitelist."""

    phrases = re.findall(r"\b(?:[A-Z][a-z]{2,})(?:\s+[A-Z][a-z]{2,}){0,2}\b", text)
    return _unique(phrase.lower() for phrase in phrases)


def _reference_geographic_terms(profile: Mapping[str, Any] | None) -> list[str]:
    """Read explicit geographic fields without confusing frame position for place."""

    if not isinstance(profile, Mapping):
        return []
    values: list[str] = []
    geographic_keys = {"place", "places", "region", "regions", "destination", "destinations", "geography", "geographic_location"}
    spatial_labels = {"top", "bottom", "left", "right", "center", "centre", "middle", "foreground", "background"}

    def visit(value: Any, depth: int) -> None:
        if depth > 6:
            return
        if isinstance(value, Mapping):
            for key, item in value.items():
                if str(key).lower() in geographic_keys:
                    values.extend(term for term in english_terms(item) if term not in spatial_labels)
                else:
                    visit(item, depth + 1)
        elif isinstance(value, (list, tuple)):
            for item in value[:40]:
                visit(item, depth + 1)

    visit(profile, 0)
    return _unique(values, 8)


def _collect_profile_terms(profile: Mapping[str, Any] | None, fragments: Sequence[str]) -> list[str]:
    if not isinstance(profile, Mapping):
        return []
    terms: list[str] = []

    def visit(value: Any, path: str, depth: int) -> None:
        if depth > 7:
            return
        if isinstance(value, Mapping):
            for key, item in value.items():
                visit(item, f"{path}.{str(key).lower()}", depth + 1)
        elif isinstance(value, (list, tuple)):
            for item in value[:40]:
                visit(item, path, depth + 1)
        elif isinstance(value, str) and any(fragment in path for fragment in fragments):
            terms.extend(english_terms(value))

    visit(profile, "", 0)
    return _unique(terms)


def _collect_numbers(profile: Mapping[str, Any] | None, fragments: Sequence[str]) -> list[float]:
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
            for item in value[:80]:
                visit(item, path, depth + 1)
        elif any(fragment in path for fragment in fragments):
            try:
                number = float(value)
            except (TypeError, ValueError):
                return
            if math.isfinite(number):
                values.append(number)

    visit(profile, "", 0)
    return values


def _axis_terms(terms: Sequence[str], axis: str) -> list[str]:
    vocabulary = _AXES[axis]
    joined = " ".join(terms)
    return _unique(word for word in vocabulary if _contains_term(joined, word))


def _contains_term(text: str, term: str) -> bool:
    """Match English descriptors as tokens, not as accidental substrings.

    This keeps ``desaturated`` from also activating ``saturated`` and keeps
    short world terms such as ``ice`` from matching unrelated words such as
    ``office``.
    """

    return bool(
        re.search(
            rf"(?<![a-z0-9]){re.escape(str(term).lower())}(?![a-z0-9])",
            str(text).lower(),
        )
    )


def _world_families(terms: Sequence[str] | str) -> list[str]:
    joined = " ".join(terms) if not isinstance(terms, str) else terms
    joined = joined.lower()
    scores = {
        family: sum(_contains_term(joined, term) for term in vocabulary)
        for family, vocabulary in _WORLD_TERMS.items()
    }
    return [family for family, score in sorted(scores.items(), key=lambda item: (-item[1], item[0])) if score > 0]


def _target_color(terms: Sequence[str], reference_profile: Mapping[str, Any] | None) -> dict[str, Any]:
    joined = " ".join(terms)
    hue, saturation, value = 30.0, 0.46, 0.56
    if any(_contains_term(joined, term) for term in ("blue", "cyan", "cool", "teal")):
        hue = 205.0
    elif _contains_term(joined, "green"):
        hue = 115.0
    elif any(_contains_term(joined, term) for term in ("purple", "violet", "magenta")):
        hue = 285.0
    elif any(_contains_term(joined, term) for term in ("red", "orange", "warm", "golden")):
        hue = 28.0
    if any(_contains_term(joined, term) for term in ("grey", "gray", "monochrome", "muted", "desaturated")):
        saturation = 0.24
    elif any(_contains_term(joined, term) for term in ("vibrant", "saturated", "colorful")):
        saturation = 0.76
    if any(_contains_term(joined, term) for term in ("dark", "low key", "moody", "night")):
        value = 0.36
    elif any(_contains_term(joined, term) for term in ("bright", "high key", "airy", "clear")):
        value = 0.74
    descriptive = any(_contains_term(joined, term) for term in _AXES["color"])
    brightness = _collect_numbers(reference_profile, ("brightness", "luma_mean"))
    saturations = _collect_numbers(reference_profile, ("saturation",))
    warmth = _collect_numbers(reference_profile, ("warmth",))
    evidence = 0
    if brightness:
        observed = float(np.median(brightness))
        observed = _clamp(observed / 255.0 if observed > 1.0 else observed, 0.08, 0.95)
        value = _clamp(value * 0.75 + observed * 0.25) if descriptive else observed
        evidence += 1
    if saturations:
        observed = float(np.median(saturations))
        observed = _clamp(observed / 255.0 if observed > 1.0 else observed, 0.04, 0.95)
        saturation = _clamp(saturation * 0.75 + observed * 0.25) if descriptive else observed
        evidence += 1
    if warmth:
        observed = float(np.median(warmth))
        if not descriptive:
            hue = 28.0 if observed > 0.12 else 205.0 if observed < -0.12 else hue
        evidence += 1
    return {
        "hue_degrees": round(hue, 3),
        "saturation": round(saturation, 4),
        "value": round(value, 4),
        "confidence": round(_clamp(0.28 + evidence * 0.18 + (0.18 if descriptive else 0.0)), 4),
    }


def build_visual_style_profile(
    theme: str,
    reference_profile: Mapping[str, Any] | None,
    audio_profile: Mapping[str, Any] | None,
    visual_request: str = "",
) -> dict[str, Any]:
    """Build one task-specific profile without a style or location allowlist."""

    explicit = str(visual_request or "").strip()
    theme_terms = english_terms(theme)
    request_terms = english_terms(explicit)
    reference_terms = _unique(
        _collect_profile_terms(
            reference_profile,
            ("topic", "subject", "content", "positive_terms", "search", "mood", "tone", "palette", "lighting", "motion", "camera", "shot_scale"),
        ),
        32,
    )
    audio_terms = _unique(
        _collect_profile_terms(audio_profile, ("mood", "emotion", "energy", "texture", "character", "edit_guidance", "section")),
        16,
    )
    combined = _unique([*request_terms, *theme_terms, *reference_terms, *audio_terms], 64)
    locations = _unique(
        [
            *_proper_location_phrases(explicit),
            *_proper_location_phrases(theme),
            *_reference_geographic_terms(reference_profile),
        ],
        8,
    )
    environments = _unique(
        [*_axis_terms(request_terms, "environment"), *_axis_terms(theme_terms, "environment"), *_axis_terms(reference_terms, "environment"), *theme_terms[:4]],
        12,
    )
    emotion = _unique(
        [*_axis_terms(request_terms, "emotion"), *_axis_terms(audio_terms, "emotion"), *_axis_terms(reference_terms, "emotion")],
        8,
    )
    weather = _unique([*_axis_terms(request_terms, "weather"), *_axis_terms(reference_terms, "weather")], 6)
    lighting = _unique([*_axis_terms(request_terms, "lighting"), *_axis_terms(reference_terms, "lighting")], 6)
    capture = _unique([*_axis_terms(request_terms, "capture"), *_axis_terms(reference_terms, "capture")], 6)
    motion = _unique([*_axis_terms(request_terms, "motion"), *_axis_terms(reference_terms, "motion")], 6)
    request_color = _axis_terms(request_terms, "color")
    reference_color = _axis_terms(reference_terms, "color")
    color = _unique([*request_color, *reference_color], 6)
    explicit_worlds = _world_families([*request_terms, *theme_terms])
    worlds = (explicit_worlds or _world_families([*environments, *combined]))[:5]
    explicit_signal_count = sum(bool(values) for values in (locations, emotion, weather, lighting, capture, motion, color))
    confidence = _clamp(0.22 + 0.08 * explicit_signal_count + (0.18 if explicit else 0.0))
    profile = {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "engine_version": ENGINE_VERSION,
        "source": {
            "theme": theme,
            "visual_request": explicit,
            "uses_reference_profile": bool(reference_profile),
            "uses_audio_profile": bool(audio_profile),
        },
        "terms": {
            "subject": _unique([*theme_terms, *request_terms], 12),
            "emotion": emotion,
            "environment": environments,
            "location": locations,
            "weather": weather,
            "lighting": lighting,
            "capture": capture,
            "motion": motion,
            "color": color,
            "reference": reference_terms[:16],
            "audio": audio_terms[:8],
            "avoid": _collect_profile_terms(reference_profile, ("avoid_terms", "negative_terms"))[:12],
        },
        "world_model": {
            "preferred_families": worlds,
            "preferred_groups": _unique(_WORLD_GROUP.get(family, family) for family in worlds),
            "confidence": round(confidence, 4),
            "policy": "compatible journey, not single-location lock",
        },
        "color_profile": _target_color(
            [*request_color, *request_terms, *theme_terms]
            if request_color
            else [*reference_color, *theme_terms, *audio_terms],
            reference_profile,
        ),
        "quality": {
            "aesthetic_floor": round(0.46 if confidence >= 0.58 else 0.42, 4),
            "cinematic_floor": round(0.42 if confidence >= 0.58 else 0.38, 4),
            "expand_search_when_short": True,
        },
        "sequence": {
            "profile_confidence": round(confidence, 4),
            "world_weight": 0.25,
            "color_weight": 0.20,
            "time_weather_weight": 0.14,
            "camera_language_weight": 0.13,
            "visual_match_weight": 0.28,
            "minimum_consistency": round(0.55 if confidence >= 0.58 else 0.48, 4),
        },
    }
    digest_payload = json.dumps(profile, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    profile["profile_digest"] = hashlib.sha256(digest_payload.encode("utf-8")).hexdigest()
    return profile


def _clean_query(parts: Iterable[Any]) -> str:
    terms: list[str] = []
    for part in parts:
        terms.extend(english_terms(part))
    unique_terms = _unique(terms)
    phrases = [term for term in unique_terms if " " in term]
    unique_terms = [
        term
        for term in unique_terms
        if " " in term or not any(re.search(rf"\b{re.escape(term)}\b", phrase) for phrase in phrases)
    ]
    query = " ".join(unique_terms)
    while len(query) > 100 and " " in query:
        query = query.rsplit(" ", 1)[0]
    return query[:100].strip()


def plan_visual_search_queries(profile: Mapping[str, Any], expansion_level: int) -> list[dict[str, Any]]:
    """Compose theme + mood + place + light/weather + capture + motion queries."""

    terms = profile.get("terms", {}) if isinstance(profile.get("terms"), Mapping) else {}
    subject = list(terms.get("subject") or [])
    environment = list(terms.get("environment") or [])
    location = list(terms.get("location") or [])
    emotion = list(terms.get("emotion") or [])
    weather = list(terms.get("weather") or [])
    lighting = list(terms.get("lighting") or [])
    capture = list(terms.get("capture") or [])
    motion = list(terms.get("motion") or [])
    anchor = location[:1] or environment[:1] or subject[:1] or ["cinematic"]
    setting = environment[:2] or subject[:2]
    atmosphere = weather[:1] + lighting[:1] + emotion[:1]
    camera = capture[:1] + motion[:1]
    if not camera:
        camera = ["cinematic"]
    if expansion_level <= 0:
        specs = (
            ([*anchor, *setting[:1], *atmosphere[:2], *camera[:2]], "precision_world_atmosphere_camera"),
            ([*anchor, *setting[1:2], *lighting[:1], *(capture[:1] or ["aerial"])], "precision_location_light_capture"),
            ([*setting[:2], *emotion[:1], *weather[:1], *camera[:1]], "precision_theme_emotion_weather"),
            ([*subject[:2], *lighting[:1], *motion[:1], "cinematic"], "precision_subject_motion"),
        )
    elif expansion_level == 1:
        specs = (
            ([*anchor, *setting[:2], *(capture[:1] or ["drone"])], "adjacent_world_capture"),
            ([*setting[:1], *atmosphere[:3], "cinematic"], "adjacent_atmosphere"),
            ([*subject[:2], *(motion[:1] or ["tracking"]), "wide"], "adjacent_motion_scale"),
            ([*environment[1:3], *emotion[:1], *(lighting[1:2] or weather[1:2])], "adjacent_environment"),
        )
    else:
        specs = (
            ([*anchor, *(capture[:1] or ["aerial"]), "cinematic"], "quality_recall_anchor"),
            ([*setting[:1], *atmosphere[:1], "wide view"], "quality_recall_setting"),
            ([*subject[:1], *(motion[:1] or ["camera movement"])], "quality_recall_motion"),
            ([*environment[:1], *emotion[:1], "authentic atmosphere"], "quality_recall_emotion"),
        )
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for parts, intent in specs:
        query = _clean_query(parts)
        if not query or query in seen:
            continue
        seen.add(query)
        records.append({"query": query, "intent": intent, "expansion_level": int(expansion_level)})
    return records


def color_profile_fit(mean_hsv: Mapping[str, Any] | None, profile: Mapping[str, Any]) -> float:
    if not isinstance(mean_hsv, Mapping):
        return 0.45
    target = profile.get("color_profile", {}) if isinstance(profile.get("color_profile"), Mapping) else {}
    try:
        hue = float(mean_hsv.get("hue_degrees")) % 360.0
        saturation = float(mean_hsv.get("saturation"))
        value = float(mean_hsv.get("value"))
    except (TypeError, ValueError):
        return 0.45
    target_hue = float(target.get("hue_degrees", 30.0)) % 360.0
    hue_distance = min(abs(hue - target_hue), 360.0 - abs(hue - target_hue)) / 180.0
    hue_weight = min(0.34, max(0.04, min(saturation, float(target.get("saturation", 0.46))) * 0.55))
    return _clamp(
        1.0
        - hue_weight * hue_distance
        - 0.34 * abs(saturation - float(target.get("saturation", 0.46)))
        - 0.48 * abs(value - float(target.get("value", 0.56)))
    )


def metadata_profile_fit(tags: str, profile: Mapping[str, Any]) -> dict[str, Any]:
    tokens = set(english_terms(tags))
    joined = " ".join(tokens)
    terms = profile.get("terms", {}) if isinstance(profile.get("terms"), Mapping) else {}
    desired = set(_unique([*(terms.get("subject") or []), *(terms.get("environment") or []), *(terms.get("location") or [])]))
    desired_tokens = set(english_terms(" ".join(desired)))
    relevance = len(tokens & desired_tokens) / max(1, min(8, len(desired_tokens))) if desired_tokens else 0.45
    candidate_worlds = _world_families(joined)
    preferred = list((profile.get("world_model") or {}).get("preferred_families") or [])
    preferred_groups = {_WORLD_GROUP.get(item, item) for item in preferred}
    candidate_groups = {_WORLD_GROUP.get(item, item) for item in candidate_worlds}
    if not preferred:
        world_fit = 0.62
    elif set(candidate_worlds) & set(preferred):
        world_fit = 1.0
    elif preferred_groups & candidate_groups:
        world_fit = 0.72
    elif not candidate_worlds:
        world_fit = 0.42
    else:
        world_fit = 0.12
    avoid = set(english_terms(" ".join(terms.get("avoid") or [])))
    avoid_hits = sorted(tokens & avoid)
    confidence = float((profile.get("world_model") or {}).get("confidence", 0.0))
    allowed = not avoid_hits and not (confidence >= 0.72 and preferred and world_fit < 0.18)
    return {
        "allowed": allowed,
        "relevance": round(_clamp(0.15 + relevance * 0.85), 4),
        "world_fit": round(world_fit, 4),
        "world_families": candidate_worlds,
        "avoid_hits": avoid_hits,
    }


def frame_aesthetic_metrics(image_bgr: np.ndarray) -> dict[str, float]:
    """Estimate composition/depth/light/impact without claiming semantic truth."""

    if image_bgr is None or image_bgr.size == 0:
        return {
            "spatial_depth": 0.45, "composition_quality": 0.45, "visual_impact": 0.45,
            "lighting_quality": 0.45, "atmosphere_quality": 0.45, "color_quality": 0.45,
            "ordinary_travelogue_risk": 0.5,
        }
    frame = image_bgr
    scale = min(1.0, 640.0 / max(frame.shape[:2]))
    if scale < 1.0:
        frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    edges = cv2.Canny(gray, 70, 150).astype(np.float32) / 255.0
    h, w = gray.shape
    thirds = [edges[: h // 3], edges[h // 3 : 2 * h // 3], edges[2 * h // 3 :]]
    band_density = [float(np.mean(band)) if band.size else 0.0 for band in thirds]
    layer_variation = float(np.std(band_density) / max(0.04, np.mean(band_density) + 0.02))
    horizontal = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    horizon_strength = float(np.percentile(np.mean(np.abs(horizontal), axis=1), 92) / 255.0)
    spatial_depth = _clamp(0.28 + 0.36 * min(1.0, layer_variation) + 0.36 * min(1.0, horizon_strength * 4.0))

    energy = cv2.GaussianBlur(edges, (0, 0), 7)
    total = float(energy.sum())
    if total > 1e-6:
        yy, xx = np.mgrid[0:h, 0:w]
        cx = float((energy * xx).sum() / total) / max(1, w - 1)
        cy = float((energy * yy).sum() / total) / max(1, h - 1)
        third_distance = min(
            math.hypot(cx - tx, cy - ty)
            for tx in (1 / 3, 2 / 3)
            for ty in (1 / 3, 2 / 3)
        )
        thirds_fit = _clamp(1.0 - third_distance / 0.48)
        left_energy = float(energy[:, : w // 2].sum()) / total
        balance = _clamp(1.0 - abs(left_energy - 0.5) * 1.6)
    else:
        thirds_fit, balance = 0.35, 0.45
    composition = _clamp(0.58 * thirds_fit + 0.42 * balance)

    p02, p10, p50, p90, p98 = [float(value) for value in np.percentile(gray, (2, 10, 50, 90, 98))]
    dynamic_range = _clamp((p98 - p02) / 205.0)
    mid_contrast = _clamp((p90 - p10) / 170.0)
    clipping = float(np.mean((gray <= 4) | (gray >= 251)))
    lighting = _clamp(0.48 * dynamic_range + 0.42 * mid_contrast + 0.10 - clipping * 1.7)
    saturation = hsv[..., 1].astype(np.float32) / 255.0
    sat_mean = float(np.mean(saturation))
    sat_spread = float(np.std(saturation))
    color_quality = _clamp(0.48 + 0.42 * min(1.0, sat_spread * 4.0) - max(0.0, sat_mean - 0.78) * 0.8)
    top = gray[: max(1, h // 3)]
    top_texture = float(cv2.Laplacian(top, cv2.CV_32F).var())
    haze = _clamp((float(np.mean(top)) / 255.0) * (1.0 - min(1.0, top_texture / 500.0)))
    atmosphere = _clamp(0.36 + 0.30 * haze + 0.34 * min(1.0, float(np.std(top)) / 55.0))
    edge_density = float(np.mean(edges))
    visual_impact = _clamp(0.31 * spatial_depth + 0.26 * lighting + 0.23 * composition + 0.20 * min(1.0, edge_density / 0.12))
    flatness = _clamp(1.0 - dynamic_range) * _clamp(1.0 - spatial_depth)
    travelogue = _clamp(0.12 + 0.55 * flatness + 0.20 * (1.0 - composition) + 0.13 * clipping)
    return {
        "spatial_depth": round(spatial_depth, 4),
        "composition_quality": round(composition, 4),
        "visual_impact": round(visual_impact, 4),
        "lighting_quality": round(lighting, 4),
        "atmosphere_quality": round(atmosphere, 4),
        "color_quality": round(color_quality, 4),
        "ordinary_travelogue_risk": round(travelogue, 4),
    }


def aggregate_video_aesthetics(
    frame_metrics: Sequence[Mapping[str, Any]],
    *,
    sharpness: float,
    motion_score: float,
    stability_score: float,
    motion_type: str,
    resolution_score: float,
) -> dict[str, Any]:
    def mean(key: str, default: float = 0.45) -> float:
        values = [_clamp(item.get(key, default)) for item in frame_metrics]
        return float(np.mean(values)) if values else default

    movement_value = {
        "push_in": 0.96, "pull_out": 0.92, "dive": 0.98, "rise": 0.90,
        "fpv_glide": 0.95, "lateral_left": 0.82, "lateral_right": 0.82,
        "tilt_up": 0.70, "tilt_down": 0.70, "drift": 0.54, "mixed": 0.42,
        "static": 0.24, "unknown": 0.40,
    }.get(str(motion_type), 0.46)
    depth = mean("spatial_depth")
    composition = mean("composition_quality")
    impact = mean("visual_impact")
    lighting = mean("lighting_quality")
    atmosphere = mean("atmosphere_quality")
    color = mean("color_quality")
    travelogue = mean("ordinary_travelogue_risk", 0.5)
    movement = _clamp(0.55 * movement_value + 0.30 * motion_score + 0.15 * stability_score)
    aesthetic = _clamp(
        0.17 * depth + 0.16 * composition + 0.18 * impact + 0.14 * lighting
        + 0.08 * atmosphere + 0.09 * color + 0.09 * movement
        + 0.05 * sharpness + 0.04 * resolution_score - 0.13 * travelogue
    )
    cinematic = _clamp(
        0.25 * depth + 0.22 * impact + 0.16 * lighting + 0.12 * atmosphere
        + 0.12 * movement + 0.08 * composition + 0.05 * color - 0.12 * travelogue
    )
    return {
        "schema_version": ASSET_ANALYSIS_SCHEMA_VERSION,
        "engine_version": ENGINE_VERSION,
        "aesthetic_score": round(aesthetic, 4),
        "cinematic_score": round(cinematic, 4),
        "spatial_depth_score": round(depth, 4),
        "composition_quality_score": round(composition, 4),
        "visual_impact_score": round(impact, 4),
        "lighting_quality_score": round(lighting, 4),
        "atmosphere_quality_score": round(atmosphere, 4),
        "intrinsic_color_quality_score": round(color, 4),
        "camera_movement_value": round(movement, 4),
        "ordinary_travelogue_risk": round(travelogue, 4),
        "motion_type": str(motion_type),
        "sampled_frame_count": len(frame_metrics),
        "heuristic_notice": "Sampled visual heuristics; not a human aesthetic guarantee.",
    }


def analysis_cache_valid(quality: Mapping[str, Any] | None, file_sha256: str = "") -> bool:
    if not isinstance(quality, Mapping):
        return False
    cache = quality.get("analysis_cache") if isinstance(quality.get("analysis_cache"), Mapping) else {}
    if int(cache.get("schema_version") or 0) != ASSET_ANALYSIS_SCHEMA_VERSION:
        return False
    if str(cache.get("engine_version") or "") != ENGINE_VERSION:
        return False
    cached_hash = str(cache.get("file_sha256") or "").lower()
    return not file_sha256 or not cached_hash or cached_hash == str(file_sha256).lower()


def asset_visual_features(asset: Mapping[str, Any]) -> dict[str, Any]:
    quality = asset.get("quality") if isinstance(asset.get("quality"), Mapping) else {}
    cached = quality.get("visual_features") if isinstance(quality.get("visual_features"), Mapping) else {}
    if cached and all(key in cached for key in ("world_families", "motion_type", "mean_hsv", "aesthetic_score")):
        return dict(cached)
    visual = quality.get("visual_analysis") if isinstance(quality.get("visual_analysis"), Mapping) else {}
    tags = " ".join(
        str(value)
        for value in (asset.get("tags"), asset.get("scene_category"), asset.get("semantic_tags"), asset.get("search_queries"))
        if value not in (None, "")
    )
    tokens = english_terms(tags)
    joined = " ".join(tokens)
    time_terms = [term for term in _AXES["lighting"] if term in joined]
    weather_terms = [term for term in _AXES["weather"] if term in joined]
    capture_terms = [term for term in _AXES["capture"] if term in joined]
    motion_type = str(
        visual.get("motion_type")
        or quality.get("motion_type")
        or quality.get("motion_signals", {}).get("motion_type")
        or asset.get("motion_label")
        or "unknown"
    ).lower()
    motion_direction = str(
        asset.get("motion_direction")
        or quality.get("motion_direction")
        or quality.get("motion_signals", {}).get("motion_direction")
        or "unknown"
    ).lower()
    hsv = asset.get("mean_hsv") if isinstance(asset.get("mean_hsv"), Mapping) else quality.get("mean_hsv", {})
    texture = [name for name, vocab in _TEXTURE_TERMS.items() if any(term in joined for term in vocab)]
    return {
        "world_families": _world_families(joined),
        "world_groups": _unique(_WORLD_GROUP.get(item, item) for item in _world_families(joined)),
        "time_of_day": _unique(time_terms),
        "weather": _unique(weather_terms),
        "capture": _unique(capture_terms),
        "motion_type": motion_type,
        "motion_direction": motion_direction,
        "shot_scale": str(asset.get("source_shot_scale") or asset.get("shot_scale") or "unknown").lower(),
        "mean_hsv": dict(hsv) if isinstance(hsv, Mapping) else {},
        "texture_families": texture,
        "aesthetic_score": _clamp(visual.get("aesthetic_score", quality.get("overall_score", asset.get("quality_score", 0.5)))),
        "cinematic_score": _clamp(visual.get("cinematic_score", 0.5)),
        "spatial_depth_score": _clamp(visual.get("spatial_depth_score", 0.45)),
        "composition_quality_score": _clamp(visual.get("composition_quality_score", 0.45)),
        "visual_impact_score": _clamp(visual.get("visual_impact_score", 0.45)),
        "ordinary_travelogue_risk": _clamp(visual.get("ordinary_travelogue_risk", 0.45)),
    }


def asset_profile_fit(asset: Mapping[str, Any], profile: Mapping[str, Any]) -> dict[str, float]:
    features = asset_visual_features(asset)
    preferred = set((profile.get("world_model") or {}).get("preferred_families") or [])
    preferred_groups = set((profile.get("world_model") or {}).get("preferred_groups") or [])
    worlds = set(features["world_families"])
    groups = set(features["world_groups"])
    if not preferred:
        world = 0.62
    elif preferred & worlds:
        world = 1.0
    elif preferred_groups & groups:
        world = 0.72
    elif not worlds:
        world = 0.42
    else:
        world = 0.12
    desired_terms = profile.get("terms", {}) if isinstance(profile.get("terms"), Mapping) else {}
    desired_time = set(desired_terms.get("lighting") or [])
    desired_weather = set(desired_terms.get("weather") or [])
    time_weather = 0.62
    if desired_time or desired_weather:
        overlap = bool(desired_time & set(features["time_of_day"])) or bool(desired_weather & set(features["weather"]))
        time_weather = 1.0 if overlap else 0.28 if features["time_of_day"] or features["weather"] else 0.46
    desired_capture = set(desired_terms.get("capture") or [])
    desired_motion = set(desired_terms.get("motion") or [])
    camera = 0.62
    if desired_capture or desired_motion:
        camera = 1.0 if desired_capture & set(features["capture"]) else 0.48
        if any(term.replace(" ", "_") in features["motion_type"] for term in desired_motion):
            camera = max(camera, 0.92)
    color = color_profile_fit(features["mean_hsv"], profile)
    aesthetic = features["aesthetic_score"]
    total = _clamp(0.27 * world + 0.20 * color + 0.14 * time_weather + 0.13 * camera + 0.26 * aesthetic)
    return {
        "total": round(total, 5), "world": round(world, 5), "color": round(color, 5),
        "time_weather": round(time_weather, 5), "camera_language": round(camera, 5),
        "aesthetic": round(aesthetic, 5), "cinematic": round(features["cinematic_score"], 5),
    }


def _pair_color_match(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    try:
        lh, ls, lv = float(left.get("hue_degrees")), float(left.get("saturation")), float(left.get("value"))
        rh, rs, rv = float(right.get("hue_degrees")), float(right.get("saturation")), float(right.get("value"))
    except (TypeError, ValueError):
        return 0.45
    hue_distance = min(abs(lh - rh), 360.0 - abs(lh - rh)) / 180.0
    hue_weight = min(0.30, max(0.04, min(ls, rs) * 0.50))
    return _clamp(1.0 - hue_weight * hue_distance - 0.38 * abs(ls - rs) - 0.48 * abs(lv - rv))


def transition_match(previous: Mapping[str, Any] | None, current: Mapping[str, Any], profile: Mapping[str, Any]) -> dict[str, float]:
    if previous is None:
        return {"total": 0.55, "motion": 0.55, "scale": 0.55, "color": 0.55, "world": 0.55, "texture": 0.55, "composition": 0.55}
    left, right = asset_visual_features(previous), asset_visual_features(current)
    ld, rd = left["motion_direction"], right["motion_direction"]
    lt, rt = left["motion_type"], right["motion_type"]
    if ld not in {"unknown", "mixed"} and ld == rd:
        motion = 1.0
    elif lt == rt and lt not in {"unknown", "static"}:
        motion = 0.92
    elif {lt, rt} <= {"push_in", "fpv_glide", "drift"} or {lt, rt} <= {"pull_out", "rise", "drift"}:
        motion = 0.78
    elif "static" in {lt, rt}:
        motion = 0.32
    else:
        motion = 0.50
    scale_order = {"extreme_wide": 0, "wide": 1, "medium": 2, "close_up": 3, "extreme_close_up": 4}
    ls, rs = scale_order.get(left["shot_scale"], 2), scale_order.get(right["shot_scale"], 2)
    scale = _clamp(1.0 - abs(ls - rs) * 0.22)
    color = _pair_color_match(left["mean_hsv"], right["mean_hsv"])
    left_world, right_world = set(left["world_families"]), set(right["world_families"])
    left_group, right_group = set(left["world_groups"]), set(right["world_groups"])
    world = 1.0 if left_world & right_world else 0.76 if left_group & right_group else 0.42 if not left_world or not right_world else 0.12
    textures_left, textures_right = set(left["texture_families"]), set(right["texture_families"])
    texture = 1.0 if textures_left & textures_right else 0.48 if not textures_left or not textures_right else 0.22
    composition = _clamp(1.0 - abs(left["composition_quality_score"] - right["composition_quality_score"]) * 0.75)
    total = _clamp(0.24 * motion + 0.13 * scale + 0.21 * color + 0.20 * world + 0.13 * texture + 0.09 * composition)
    return {
        "total": round(total, 5), "motion": round(motion, 5), "scale": round(scale, 5),
        "color": round(color, 5), "world": round(world, 5), "texture": round(texture, 5),
        "composition": round(composition, 5),
    }


def evaluate_sequence_consistency(shots: Sequence[Mapping[str, Any]], profile: Mapping[str, Any]) -> dict[str, Any]:
    if not shots:
        return {"evaluated": False, "passed": False, "failures": ["empty sequence"]}
    features = [asset_visual_features(shot) for shot in shots]
    pairs = [transition_match(shots[index - 1], shots[index], profile) for index in range(1, len(shots))]
    profile_fits = [asset_profile_fit(shot, profile) for shot in shots]
    pair_average = float(np.mean([item["total"] for item in pairs])) if pairs else 1.0
    world_average = float(np.mean([item["world"] for item in profile_fits]))
    color_average = float(np.mean([item["color"] for item in profile_fits]))
    time_average = float(np.mean([item["time_weather"] for item in profile_fits]))
    camera_average = float(np.mean([item["camera_language"] for item in profile_fits]))
    overall = _clamp(0.28 * pair_average + 0.25 * world_average + 0.20 * color_average + 0.14 * time_average + 0.13 * camera_average)
    threshold = float((profile.get("sequence") or {}).get("minimum_consistency", 0.48))
    confidence = float((profile.get("sequence") or {}).get("profile_confidence", 0.0))
    evaluated = confidence >= 0.35 and any(item["world_families"] or item["mean_hsv"] for item in features)
    failures: list[str] = []
    if evaluated and overall < threshold:
        failures.append(f"sequence consistency {overall:.4f} < {threshold:.4f}")
    severe_pairs = [index + 1 for index, item in enumerate(pairs) if item["total"] < 0.24]
    if evaluated and len(severe_pairs) > max(1, math.ceil(len(pairs) * 0.18)):
        failures.append("too many severe visual discontinuities")
    world_counts = Counter(family for item in features for family in item["world_families"][:1])
    return {
        "evaluated": evaluated,
        "passed": not failures,
        "score": round(overall, 4),
        "threshold": round(threshold, 4),
        "pair_match_average": round(pair_average, 4),
        "world_fit_average": round(world_average, 4),
        "color_fit_average": round(color_average, 4),
        "time_weather_fit_average": round(time_average, 4),
        "camera_language_fit_average": round(camera_average, 4),
        "severe_pair_right_indices": severe_pairs,
        "dominant_world_families": [name for name, _ in world_counts.most_common(5)],
        "failures": failures,
        "pair_scores": pairs,
        "heuristic_notice": "Sequence-level sampled metadata/signal estimate; review the final video visually.",
    }


def build_light_grade(
    shot: Mapping[str, Any],
    profile: Mapping[str, Any],
    base_brightness: float,
    base_saturation: float,
    base_contrast: float,
) -> dict[str, Any]:
    """Return a bounded per-shot normalization, never a heavy rescue filter."""

    features = asset_visual_features(shot)
    source = features["mean_hsv"]
    target = profile.get("color_profile", {}) if isinstance(profile.get("color_profile"), Mapping) else {}
    try:
        source_hue = float(source.get("hue_degrees", target.get("hue_degrees", 30.0))) % 360.0
        source_sat = float(source.get("saturation", target.get("saturation", 0.46)))
        source_val = float(source.get("value", target.get("value", 0.56)))
    except (TypeError, ValueError):
        source_hue, source_sat, source_val = 30.0, 0.46, 0.56
    target_hue = float(target.get("hue_degrees", source_hue)) % 360.0
    target_sat = float(target.get("saturation", source_sat))
    target_val = float(target.get("value", source_val))
    confidence = float(target.get("confidence", 0.0))
    strength = 0.0 if confidence < 0.35 else min(0.55, 0.24 + confidence * 0.34)
    brightness = max(-0.065, min(0.065, base_brightness + (target_val - source_val) * 0.13 * strength))
    saturation_ratio = target_sat / max(0.08, source_sat)
    saturation = max(0.76, min(1.16, base_saturation * (1.0 + (saturation_ratio - 1.0) * strength)))
    contrast = max(0.94, min(1.12, base_contrast + (0.025 if target_val < 0.48 else 0.0) * strength))
    signed_hue = ((target_hue - source_hue + 540.0) % 360.0) - 180.0
    warm_shift = max(-1.0, min(1.0, -signed_hue / 150.0)) * 0.028 * strength
    balance = {
        "rs": round(warm_shift, 5), "gs": 0.0, "bs": round(-warm_shift, 5),
        "rm": round(warm_shift * 0.72, 5), "gm": 0.0, "bm": round(-warm_shift * 0.72, 5),
        "rh": round(warm_shift * 0.42, 5), "gh": 0.0, "bh": round(-warm_shift * 0.42, 5),
    }
    return {
        "profile": "dynamic_v1.3",
        "source_hsv": {"hue_degrees": round(source_hue, 4), "saturation": round(source_sat, 4), "value": round(source_val, 4)},
        "target_hsv": {"hue_degrees": round(target_hue, 4), "saturation": round(target_sat, 4), "value": round(target_val, 4)},
        "strength": round(strength, 4),
        "brightness": round(brightness, 5), "saturation": round(saturation, 5), "contrast": round(contrast, 5),
        "colorbalance": balance,
        "policy": "bounded normalization only; incompatible clips must be rejected upstream",
    }
