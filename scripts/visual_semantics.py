#!/usr/bin/env python3
"""Real CLIP semantics plus explainable subject/saliency geometry.

CLIP is loaded lazily.  If the model is unavailable callers receive an
explicit degraded result and must not claim semantic recognition succeeded.
Saliency, face and crop geometry remain available without the model.
"""

from __future__ import annotations

import math
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np
from PIL import Image


SEMANTIC_MODEL_ID = os.environ.get(
    "BGM_MONTAGE_SEMANTIC_MODEL", "openai/clip-vit-base-patch32"
)

CATEGORY_LABELS: dict[str, tuple[str, ...]] = {
    "subject": (
        "natural landscape", "ocean or coast", "forest or trees", "mountains",
        "sky or clouds", "city architecture", "urban street", "road or highway",
        "vehicle or transportation", "industrial machinery", "factory production",
        "technology equipment", "food or cooking", "animal or wildlife",
        "single person", "group of people", "abstract graphics",
    ),
    "scene": (
        "outdoor nature", "coastal landscape", "forest landscape", "mountain landscape",
        "urban exterior", "architectural interior", "factory or workshop",
        "road transportation", "home interior", "office or studio", "night scene",
        "minimal abstract background",
    ),
    "action": (
        "mostly static scene", "walking", "running", "driving or traffic movement",
        "working with tools", "operating machinery", "talking or interviewing",
        "looking at the camera", "aerial camera movement", "water or waves moving",
        "general active movement",
    ),
    "composition": (
        "centered subject composition", "rule of thirds composition", "symmetrical composition",
        "leading lines composition", "layered depth composition", "minimal composition",
        "busy detailed composition", "wide establishing composition",
    ),
    "emotion": (
        "calm and serene", "contemplative and quiet", "energetic and exciting",
        "tense and dramatic", "joyful and uplifting", "melancholic and lonely",
        "warm and intimate", "neutral documentary mood",
    ),
    "human_framing": (
        "no prominent person", "person seen from behind", "small distant person",
        "overhead view of people", "prominent frontal face", "portrait close-up",
        "selfie", "interview framing", "posed person",
    ),
}


def _softmax(values: np.ndarray) -> np.ndarray:
    values = values - np.max(values)
    exponent = np.exp(values)
    return exponent / max(1e-12, float(np.sum(exponent)))


@dataclass
class SemanticBackendStatus:
    available: bool
    model_id: str
    backend: str
    error: str | None = None


class ClipSemanticAnalyzer:
    """Zero-shot image semantics backed by a pretrained CLIP model."""

    def __init__(
        self,
        model_id: str = SEMANTIC_MODEL_ID,
        cache_dir: str | os.PathLike[str] | None = None,
        local_files_only: bool | None = None,
    ) -> None:
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as exc:
            raise RuntimeError(
                "Real semantic analysis requires torch and transformers; install the v1.2 lock file."
            ) from exc

        self.torch = torch
        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        local_only = (
            os.environ.get("BGM_MONTAGE_SEMANTIC_OFFLINE") == "1"
            if local_files_only is None
            else bool(local_files_only)
        )
        load_kwargs: dict[str, Any] = {"local_files_only": local_only}
        if cache_dir is not None:
            load_kwargs["cache_dir"] = str(Path(cache_dir).expanduser().resolve())
        self.processor = CLIPProcessor.from_pretrained(model_id, use_fast=False, **load_kwargs)
        self.model = CLIPModel.from_pretrained(model_id, **load_kwargs).eval().to(self.device)
        self._labels: list[str] = []
        self._category_ranges: dict[str, tuple[int, int]] = {}
        prompts: list[str] = []
        for category, labels in CATEGORY_LABELS.items():
            start = len(self._labels)
            self._labels.extend(labels)
            self._category_ranges[category] = (start, len(self._labels))
            prompts.extend(f"a cinematic video frame showing {label}" for label in labels)
        with torch.inference_mode():
            text_inputs = self.processor(text=prompts, return_tensors="pt", padding=True)
            text_inputs = {key: value.to(self.device) for key, value in text_inputs.items()}
            text_features = self.model.get_text_features(**text_inputs)
            self.text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    @property
    def status(self) -> SemanticBackendStatus:
        return SemanticBackendStatus(True, self.model_id, f"transformers-clip/{self.device}")

    def classify_frames(self, frames_bgr: Sequence[np.ndarray], batch_size: int = 8) -> list[dict[str, Any]]:
        if not frames_bgr:
            return []
        output: list[dict[str, Any]] = []
        torch = self.torch
        for offset in range(0, len(frames_bgr), max(1, batch_size)):
            batch = frames_bgr[offset : offset + batch_size]
            images = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in batch]
            inputs = self.processor(images=images, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(self.device)
            with torch.inference_mode():
                image_features = self.model.get_image_features(pixel_values=pixel_values)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                similarity = (image_features @ self.text_features.T).detach().cpu().numpy()
            for row in similarity:
                record: dict[str, Any] = {}
                for category, (start, end) in self._category_ranges.items():
                    category_scores = _softmax(row[start:end] * 10.0)
                    order = np.argsort(category_scores)[::-1][:3]
                    top = [
                        {
                            "label": self._labels[start + int(index)],
                            "confidence": round(float(category_scores[int(index)]), 4),
                        }
                        for index in order
                    ]
                    record[category] = {"label": top[0]["label"], "confidence": top[0]["confidence"], "top": top}
                output.append(record)
        return output


def load_semantic_analyzer(
    cache_dir: str | os.PathLike[str] | None = None,
    required: bool = False,
    local_files_only: bool | None = None,
) -> tuple[ClipSemanticAnalyzer | None, SemanticBackendStatus]:
    if os.environ.get("BGM_MONTAGE_DISABLE_SEMANTICS") == "1":
        status = SemanticBackendStatus(False, SEMANTIC_MODEL_ID, "disabled", "disabled by environment")
        if required:
            raise RuntimeError(status.error)
        return None, status
    try:
        analyzer = ClipSemanticAnalyzer(
            model_id=SEMANTIC_MODEL_ID,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        return analyzer, analyzer.status
    except Exception as exc:
        status = SemanticBackendStatus(False, SEMANTIC_MODEL_ID, "unavailable", f"{type(exc).__name__}: {exc}")
        if required:
            raise RuntimeError(status.error) from exc
        return None, status


def _ascii_safe_cascade_path() -> Path | None:
    try:
        source = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    except AttributeError:
        return None
    if not source.is_file():
        return None
    if os.fspath(source).isascii():
        return source
    target_root = Path(tempfile.gettempdir()) / "bgm_montage_cv2"
    target_root.mkdir(parents=True, exist_ok=True)
    target = target_root / source.name
    if not target.is_file() or target.stat().st_size != source.stat().st_size:
        temporary = target.with_suffix(".tmp")
        shutil.copyfile(source, temporary)
        os.replace(temporary, target)
    return target


def load_face_detector() -> Any | None:
    try:
        path = _ascii_safe_cascade_path()
        if path is None:
            return None
        detector = cv2.CascadeClassifier(os.fspath(path))
        return None if detector.empty() else detector
    except (OSError, cv2.error):
        return None


def detect_faces(frame_bgr: np.ndarray, detector: Any | None = None) -> list[tuple[int, int, int, int]]:
    if detector is None:
        detector = load_face_detector()
    if detector is None or frame_bgr is None or frame_bgr.size == 0:
        return []
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.12,
        minNeighbors=5,
        minSize=(max(20, width // 24), max(20, height // 24)),
    )
    return [tuple(int(value) for value in box) for box in faces]


def spectral_residual_saliency(frame_bgr: np.ndarray) -> np.ndarray:
    """Return a normalized spectral-residual saliency map."""

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    target_width = min(256, gray.shape[1])
    scale = target_width / max(1, gray.shape[1])
    resized = cv2.resize(gray, (target_width, max(32, int(round(gray.shape[0] * scale)))), interpolation=cv2.INTER_AREA)
    spectrum = np.fft.fft2(resized)
    log_amplitude = np.log(np.abs(spectrum) + 1e-8)
    phase = np.angle(spectrum)
    average = cv2.blur(log_amplitude, (3, 3))
    residual = log_amplitude - average
    reconstructed = np.fft.ifft2(np.exp(residual + 1j * phase))
    saliency = np.abs(reconstructed) ** 2
    saliency = cv2.GaussianBlur(saliency.astype(np.float32), (9, 9), 2.5)
    saliency -= float(saliency.min())
    maximum = float(saliency.max())
    if maximum > 1e-9:
        saliency /= maximum
    return cv2.resize(saliency, (frame_bgr.shape[1], frame_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)


def subject_region(frame_bgr: np.ndarray, detector: Any | None = None) -> dict[str, Any]:
    """Estimate subject/salient geometry in normalized coordinates."""

    height, width = frame_bgr.shape[:2]
    saliency = spectral_residual_saliency(frame_bgr)
    threshold = float(np.percentile(saliency, 82.0))
    mask = saliency >= threshold
    yy, xx = np.mgrid[0:height, 0:width]
    weights = np.where(mask, saliency, 0.0).astype(np.float64)
    faces = detect_faces(frame_bgr, detector)
    for x, y, box_width, box_height in faces:
        weights[y : y + box_height, x : x + box_width] += 2.5
    total = float(weights.sum())
    if total <= 1e-9:
        center_x = center_y = 0.5
        bbox = [0.18, 0.18, 0.82, 0.82]
        confidence = 0.0
    else:
        center_x = float((weights * xx).sum() / total) / max(1, width - 1)
        center_y = float((weights * yy).sum() / total) / max(1, height - 1)
        active_y, active_x = np.where(weights >= np.percentile(weights[weights > 0], 35.0))
        if len(active_x):
            bbox = [
                float(np.percentile(active_x, 5)) / width,
                float(np.percentile(active_y, 5)) / height,
                float(np.percentile(active_x, 95)) / width,
                float(np.percentile(active_y, 95)) / height,
            ]
        else:
            bbox = [0.18, 0.18, 0.82, 0.82]
        concentration = float(np.mean(saliency[mask])) if np.any(mask) else 0.0
        confidence = min(1.0, 0.35 + 0.45 * concentration + (0.2 if faces else 0.0))
    largest_face = max((box_width * box_height) / (width * height) for _, _, box_width, box_height in faces) if faces else 0.0
    return {
        "center": {"x": round(center_x, 5), "y": round(center_y, 5)},
        "bbox": [round(max(0.0, min(1.0, value)), 5) for value in bbox],
        "confidence": round(confidence, 4),
        "face_count": len(faces),
        "largest_face_frame_ratio": round(float(largest_face), 5),
    }


def aggregate_subject_regions(regions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not regions:
        return {
            "center": {"x": 0.5, "y": 0.5},
            "bbox": [0.15, 0.15, 0.85, 0.85],
            "confidence": 0.0,
            "face_frame_ratio": 0.0,
            "largest_face_frame_ratio": 0.0,
        }
    centers_x = [float(item.get("center", {}).get("x", 0.5)) for item in regions]
    centers_y = [float(item.get("center", {}).get("y", 0.5)) for item in regions]
    boxes = [item.get("bbox", [0.15, 0.15, 0.85, 0.85]) for item in regions]
    return {
        "center": {"x": round(float(np.median(centers_x)), 5), "y": round(float(np.median(centers_y)), 5)},
        "bbox": [
            round(float(np.percentile([float(box[index]) for box in boxes], 10 if index < 2 else 90)), 5)
            for index in range(4)
        ],
        "confidence": round(float(np.mean([float(item.get("confidence", 0.0)) for item in regions])), 4),
        "face_frame_ratio": round(sum(int(item.get("face_count", 0)) > 0 for item in regions) / len(regions), 4),
        "largest_face_frame_ratio": round(max(float(item.get("largest_face_frame_ratio", 0.0)) for item in regions), 5),
        "center_spread": {
            "x": round(float(np.std(centers_x)), 5),
            "y": round(float(np.std(centers_y)), 5),
        },
        "method": "spectral_residual_saliency_plus_frontal_face_geometry",
    }


def aggregate_semantics(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not records:
        return {
            "available": False,
            "backend": "unavailable",
            "categories": {},
            "search_keywords": [],
        }
    categories: dict[str, Any] = {}
    keywords: list[str] = []
    for category in CATEGORY_LABELS:
        scores: dict[str, list[float]] = {}
        for record in records:
            value = record.get(category)
            if not isinstance(value, Mapping):
                continue
            for item in value.get("top", []):
                if isinstance(item, Mapping) and item.get("label"):
                    scores.setdefault(str(item["label"]), []).append(float(item.get("confidence", 0.0)))
        ranked = sorted(
            ((label, float(np.mean(values)), len(values)) for label, values in scores.items()),
            key=lambda item: (item[1], item[2]),
            reverse=True,
        )
        if ranked:
            categories[category] = {
                "label": ranked[0][0],
                "confidence": round(ranked[0][1], 4),
                "distribution": [
                    {"label": label, "mean_confidence": round(score, 4), "frame_votes": votes}
                    for label, score, votes in ranked[:5]
                ],
            }
            if category in {"subject", "scene", "action"}:
                keywords.extend(ranked[0][0].split())
    cleaned_keywords: list[str] = []
    for keyword in keywords:
        normalized = keyword.lower().strip(" ,.-")
        if len(normalized) > 2 and normalized not in cleaned_keywords and normalized not in {"or", "and", "the"}:
            cleaned_keywords.append(normalized)
    return {
        "available": True,
        "backend": "pretrained_clip_zero_shot",
        "categories": categories,
        "search_keywords": cleaned_keywords[:12],
        "limitations": [
            "Action and emotion labels are appearance-based CLIP estimates from sampled frames, not temporal action recognition.",
            "Labels are selected from the declared finite taxonomy and can miss uncommon subjects.",
        ],
    }


def infer_scene_category(tags: str = "", semantic: Mapping[str, Any] | None = None) -> str:
    text = tags.lower()
    if isinstance(semantic, Mapping):
        categories = semantic.get("categories", {})
        for key in ("scene", "subject"):
            value = categories.get(key, {}) if isinstance(categories, Mapping) else {}
            if isinstance(value, Mapping):
                text += " " + str(value.get("label", "")).lower()
    rules = (
        ("architecture", ("architecture", "building", "interior", "city", "urban")),
        ("transport", ("road", "car", "vehicle", "traffic", "train", "airplane")),
        ("industrial", ("factory", "machinery", "workshop", "industrial", "production")),
        ("people", ("person", "people", "portrait", "interview", "selfie", "woman", "man")),
        ("food", ("food", "cooking", "kitchen", "ingredients")),
        ("technology", ("technology", "digital", "electronics", "computer")),
        ("abstract", ("abstract", "graphic", "background")),
        ("polar_ice", ("polar", "arctic", "antarctic", "glacier", "iceberg", "ice cave")),
        ("sky_space", ("starry", "stars", "galaxy", "milky way", "aurora", "night sky", "cloudscape")),
        ("water_coast", ("ocean", "sea", "coast", "beach", "water", "waves")),
        ("mountain_canyon", ("mountain", "canyon", "cliff", "valley", "alpine", "peak", "summit", "gorge")),
        ("forest_wilderness", ("forest", "woods", "woodland", "jungle", "trees")),
        ("nature", ("nature", "landscape", "wildlife", "wilderness", "outdoors")),
    )
    for category, needles in rules:
        if any(needle in text for needle in needles):
            return category
    return "general"


def face_content_risk(tags: str, semantic: Mapping[str, Any] | None, subject: Mapping[str, Any] | None) -> float:
    text = tags.lower()
    risk = 0.0
    tag_weights = {
        "selfie": 0.95, "interview": 0.85, "portrait": 0.75, "face": 0.75,
        "close up person": 0.8, "woman": 0.35, "man": 0.35, "people": 0.28,
        "person": 0.28, "model": 0.55, "posing": 0.55,
    }
    risk = max(
        [
            weight
            for term, weight in tag_weights.items()
            if re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", text)
        ]
        or [0.0]
    )
    if isinstance(semantic, Mapping):
        framing = semantic.get("categories", {}).get("human_framing", {}) if isinstance(semantic.get("categories"), Mapping) else {}
        label = str(framing.get("label", "")).lower() if isinstance(framing, Mapping) else ""
        confidence = float(framing.get("confidence", 0.0)) if isinstance(framing, Mapping) else 0.0
        semantic_risk = {
            "prominent frontal face": 0.95,
            "portrait close-up": 0.92,
            "selfie": 1.0,
            "interview framing": 0.9,
            "posed person": 0.75,
            "small distant person": 0.15,
            "person seen from behind": 0.12,
            "overhead view of people": 0.12,
            "no prominent person": 0.0,
        }.get(label, 0.25)
        risk = max(risk, semantic_risk * max(0.45, confidence))
    if isinstance(subject, Mapping):
        face_ratio = float(subject.get("face_frame_ratio", 0.0))
        largest = float(subject.get("largest_face_frame_ratio", 0.0))
        risk = max(risk, min(1.0, face_ratio * 0.65 + math.sqrt(max(0.0, largest)) * 1.25))
    return round(min(1.0, risk), 4)
