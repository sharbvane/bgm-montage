#!/usr/bin/env python3
"""Signal-derived BGM structure analysis for the bgm-montage skill.

The analyzer intentionally describes beats, downbeats, sections, vocals and
emotion as estimates.  Every estimate is derived from the decoded waveform;
the script does not infer structure from filenames or container metadata.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence


SCHEMA_VERSION = "1.2"
ANALYZER_VERSION = "1.2.1"
DEFAULT_SAMPLE_RATE = 22_050
RHYTHM_MODES = ("beat_cut", "phrase_flow")
ANALYSIS_CONFIG_VERSION = "audiomap-v1.2-defaults-2"


def _finite_float(value: Any, digits: int = 5) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(number):
        return 0.0
    return round(number, digits)


def _fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "algorithm": "sha256",
        "sha256": digest.hexdigest(),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _cache_key(fingerprint: dict[str, Any], target_duration: Optional[float]) -> str:
    payload = {
        "sha256": fingerprint["sha256"],
        "target_duration": None if target_duration is None else round(float(target_duration), 6),
        "analyzer_version": ANALYZER_VERSION,
        "schema_version": SCHEMA_VERSION,
        "analysis_config_version": ANALYSIS_CONFIG_VERSION,
        "sample_rate": DEFAULT_SAMPLE_RATE,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _analysis_digest(profile: dict[str, Any]) -> str:
    """Hash only deterministic, source-signal-derived fields."""

    canonical_keys = (
        "duration_seconds",
        "tempo",
        "rhythm_mode",
        "events",
        "curves",
        "intervals",
        "phrases",
        "sections",
        "loop_groups",
        "key_moments",
        "timbre",
        "vocal",
        "editing_guidance",
        "reliability",
        "analysis_parameters",
    )
    canonical = {key: profile.get(key) for key in canonical_keys}
    return hashlib.sha256(
        json.dumps(
            canonical,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".tmp",
            prefix=f".{path.name}.",
            dir=str(path.parent),
            delete=False,
        ) as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            temporary = Path(handle.name)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink(missing_ok=True)


def _probe_audio(path: Path) -> dict[str, Any]:
    """Read only technical stream metadata, never musical metadata."""
    result: dict[str, Any] = {
        "duration_seconds": None,
        "channels": None,
        "source_sample_rate": None,
        "probe_method": None,
    }
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        command = [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=sample_rate,channels:format=duration",
            "-of",
            "json",
            str(path),
        ]
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=True,
            )
            payload = json.loads(completed.stdout)
            stream = (payload.get("streams") or [{}])[0]
            duration = (payload.get("format") or {}).get("duration")
            result.update(
                {
                    "duration_seconds": float(duration) if duration is not None else None,
                    "channels": int(stream["channels"]) if stream.get("channels") else None,
                    "source_sample_rate": (
                        int(stream["sample_rate"]) if stream.get("sample_rate") else None
                    ),
                    "probe_method": "ffprobe",
                }
            )
            return result
        except (OSError, ValueError, subprocess.SubprocessError, json.JSONDecodeError):
            pass

    try:
        import soundfile as sf

        info = sf.info(str(path))
        result.update(
            {
                "duration_seconds": float(info.duration),
                "channels": int(info.channels),
                "source_sample_rate": int(info.samplerate),
                "probe_method": "soundfile",
            }
        )
    except Exception:
        pass
    return result


def _load_audio(path: Path, duration: Optional[float], librosa: Any) -> tuple[Any, int, str]:
    load_duration = None if duration is None else float(duration)
    try:
        waveform, sample_rate = librosa.load(
            str(path), sr=DEFAULT_SAMPLE_RATE, mono=True, duration=load_duration
        )
        return waveform, int(sample_rate), "librosa/soundfile"
    except Exception as direct_error:
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError(
                "Audio decoding failed and ffmpeg is unavailable. Install ffmpeg and the "
                "Python audio dependencies."
            ) from direct_error

        temp_path: Optional[Path] = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
                temp_path = Path(handle.name)
            command = [ffmpeg, "-v", "error", "-y", "-i", str(path), "-vn"]
            if duration is not None:
                command.extend(["-t", f"{float(duration):.6f}"])
            command.extend(
                ["-ac", "1", "-ar", str(DEFAULT_SAMPLE_RATE), "-c:a", "pcm_f32le", str(temp_path)]
            )
            subprocess.run(command, capture_output=True, check=True)
            waveform, sample_rate = librosa.load(
                str(temp_path), sr=DEFAULT_SAMPLE_RATE, mono=True
            )
            return waveform, int(sample_rate), "ffmpeg-pcm"
        except (OSError, subprocess.SubprocessError, ValueError) as ffmpeg_error:
            raise RuntimeError(
                "Could not decode the BGM with either librosa/soundfile or ffmpeg."
            ) from ffmpeg_error
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)


def _smooth(values: Any, width: int, np: Any) -> Any:
    values = np.asarray(values, dtype=float)
    if values.size < 3 or width <= 1:
        return values.copy()
    width = max(1, min(int(width), int(values.size)))
    if width % 2 == 0:
        width = max(1, width - 1)
    if width <= 1:
        return values.copy()
    pad = width // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    kernel = np.ones(width, dtype=float) / float(width)
    return np.convolve(padded, kernel, mode="valid")


def _robust_normalize(values: Any, np: Any, low_q: float = 10.0, high_q: float = 95.0) -> Any:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values.copy()
    low = float(np.percentile(values, low_q))
    high = float(np.percentile(values, high_q))
    if high - low < 1e-9:
        return np.zeros_like(values)
    return np.clip((values - low) / (high - low), 0.0, 1.0)


def _nearest_index(sorted_values: Any, value: float, np: Any) -> int:
    if len(sorted_values) == 0:
        return 0
    position = int(np.searchsorted(sorted_values, value))
    if position <= 0:
        return 0
    if position >= len(sorted_values):
        return len(sorted_values) - 1
    before = position - 1
    return position if abs(float(sorted_values[position]) - value) < abs(float(sorted_values[before]) - value) else before


def _merge_boolean_regions(
    mask: Any,
    times: Any,
    duration: float,
    np: Any,
    min_duration: float,
    max_gap: float,
    score: Optional[Any] = None,
) -> list[dict[str, Any]]:
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return []
    frame_step = float(np.median(np.diff(times))) if len(times) > 1 else duration
    raw: list[tuple[int, int]] = []
    start = previous = int(indices[0])
    gap_frames = max(0, int(round(max_gap / max(frame_step, 1e-6))))
    for current_raw in indices[1:]:
        current = int(current_raw)
        if current - previous <= gap_frames + 1:
            previous = current
            continue
        raw.append((start, previous))
        start = previous = current
    raw.append((start, previous))

    regions: list[dict[str, Any]] = []
    for start_index, end_index in raw:
        start_time = max(0.0, float(times[start_index]) - frame_step / 2.0)
        end_time = min(duration, float(times[end_index]) + frame_step / 2.0)
        if end_time - start_time + 1e-9 < min_duration:
            continue
        region = {
            "start": _finite_float(start_time, 4),
            "end": _finite_float(end_time, 4),
            "duration": _finite_float(end_time - start_time, 4),
        }
        if score is not None:
            region["mean_likelihood"] = _finite_float(
                np.mean(score[start_index : end_index + 1]), 4
            )
            region["peak_likelihood"] = _finite_float(
                np.max(score[start_index : end_index + 1]), 4
            )
        regions.append(region)
    return regions


def _estimate_meter(beat_strengths: Any, np: Any) -> tuple[int, int, float]:
    """Return estimated beats/bar, the downbeat phase and a weak confidence."""
    strengths = np.asarray(beat_strengths, dtype=float)
    if strengths.size < 8:
        return 4, 0, 0.0
    scale = float(np.std(strengths)) + 1e-6
    candidates: list[tuple[float, int, int]] = []
    for meter in (3, 4):
        if strengths.size < meter * 2:
            continue
        phase_means = np.asarray(
            [np.mean(strengths[phase::meter]) for phase in range(meter)], dtype=float
        )
        phase = int(np.argmax(phase_means))
        other = np.delete(phase_means, phase)
        contrast = (float(phase_means[phase]) - float(np.mean(other))) / scale
        groups = strengths[phase::meter]
        consistency = 1.0 / (1.0 + float(np.std(groups)))
        candidates.append((contrast + 0.15 * consistency, meter, phase))
    if not candidates:
        return 4, 0, 0.0
    candidates.sort(reverse=True)
    best_score, meter, phase = candidates[0]
    runner_up = candidates[1][0] if len(candidates) > 1 else 0.0
    confidence = max(0.0, min(1.0, 0.35 + 0.3 * best_score + 0.2 * (best_score - runner_up)))
    if confidence < 0.35:
        return 4, phase % 4, confidence
    return meter, phase, confidence


def _pick_peak_indices(values: Any, distance: int, height: float, prominence: float, np: Any) -> Any:
    try:
        from scipy.signal import find_peaks

        peaks, _ = find_peaks(
            values,
            distance=max(1, int(distance)),
            height=float(height),
            prominence=max(0.0, float(prominence)),
        )
        return peaks.astype(int)
    except Exception:
        candidates = []
        last = -max(1, int(distance))
        for index in range(1, len(values) - 1):
            if (
                values[index] >= height
                and values[index] >= values[index - 1]
                and values[index] > values[index + 1]
                and index - last >= distance
            ):
                candidates.append(index)
                last = index
        return np.asarray(candidates, dtype=int)


def _feature_summary(
    start_index: int,
    end_index: int,
    centroid: Any,
    rolloff: Any,
    bandwidth: Any,
    zcr: Any,
    flatness: Any,
    mfcc: Any,
    np: Any,
) -> dict[str, Any]:
    end_index = max(start_index + 1, end_index)
    sl = slice(start_index, end_index)
    return {
        "spectral_centroid_hz": {
            "mean": _finite_float(np.mean(centroid[sl]), 2),
            "std": _finite_float(np.std(centroid[sl]), 2),
        },
        "spectral_rolloff_85_hz": {
            "mean": _finite_float(np.mean(rolloff[sl]), 2),
            "std": _finite_float(np.std(rolloff[sl]), 2),
        },
        "spectral_bandwidth_hz": {
            "mean": _finite_float(np.mean(bandwidth[sl]), 2),
            "std": _finite_float(np.std(bandwidth[sl]), 2),
        },
        "zero_crossing_rate": {
            "mean": _finite_float(np.mean(zcr[sl]), 5),
            "std": _finite_float(np.std(zcr[sl]), 5),
        },
        "spectral_flatness": {
            "mean": _finite_float(np.mean(flatness[sl]), 5),
            "std": _finite_float(np.std(flatness[sl]), 5),
        },
        "mfcc": [
            {
                "index": int(index + 1),
                "mean": _finite_float(np.mean(mfcc[index, sl]), 4),
                "std": _finite_float(np.std(mfcc[index, sl]), 4),
            }
            for index in range(min(13, mfcc.shape[0]))
        ],
    }


def _mood_label(
    energy: float,
    onset_density: float,
    brightness: float,
    vocal_likelihood: float,
    tempo: float,
) -> tuple[str, list[str]]:
    traits: list[str] = []
    traits.append("high-energy" if energy >= 0.68 else "low-energy" if energy < 0.34 else "mid-energy")
    traits.append("bright" if brightness >= 0.62 else "warm" if brightness < 0.36 else "balanced-tone")
    traits.append("dense-rhythm" if onset_density >= 2.2 else "sparse-rhythm" if onset_density < 0.75 else "steady-rhythm")
    if vocal_likelihood >= 0.62:
        traits.append("vocal-likely")

    if energy >= 0.7 and onset_density >= 1.6:
        return "driving/intense", traits
    if energy < 0.34 and onset_density < 0.9:
        return "calm/atmospheric", traits
    if brightness >= 0.63 and tempo >= 105:
        return "bright/uplifting", traits
    if vocal_likelihood >= 0.62 and energy < 0.58:
        return "intimate/expressive", traits
    if energy >= 0.55 and onset_density < 1.2:
        return "expansive/flowing", traits
    return "balanced/forward", traits


def _classify_rhythm_mode(
    *,
    tempo: float,
    beat_confidence: float,
    interval_cv: float,
    pulse_strength: float,
    onset_density: float,
    transient_salience: float,
    beat_count: int,
    duration: float,
) -> dict[str, Any]:
    """Choose beat-led cuts only when the measured pulse is genuinely reliable.

    The decision is deliberately conservative.  A plausible BPM alone is not
    enough: the beat intervals must be regular, the tracked grid must cover a
    useful part of the analysis window, and the onset/pulse evidence must be
    present.  This keeps ambient and rubato material in ``phrase_flow`` mode.
    """

    tempo_valid = 30.0 <= tempo <= 260.0
    regularity = max(0.0, min(1.0, 1.0 - interval_cv / 0.42))
    expected_beats = duration * tempo / 60.0 if tempo_valid else 0.0
    coverage = min(1.0, beat_count / max(4.0, expected_beats * 0.72)) if expected_beats else 0.0
    density_score = min(1.0, max(0.0, onset_density) / 2.2)
    pulse = max(0.0, min(1.0, pulse_strength))
    transient = max(0.0, min(1.0, transient_salience))
    confidence = max(0.0, min(1.0, beat_confidence))
    score = (
        0.29 * confidence
        + 0.22 * regularity
        + 0.16 * coverage
        + 0.10 * pulse
        + 0.05 * density_score
        + 0.18 * transient
    )
    minimum_beats = max(4, int(math.floor(duration * 0.30)))
    beat_led = bool(
        tempo_valid
        and beat_count >= minimum_beats
        and confidence >= 0.42
        and regularity >= 0.52
        and coverage >= 0.45
        and transient >= 0.14
        and score >= 0.56
    )
    mode = "beat_cut" if beat_led else "phrase_flow"
    reasons: list[str] = []
    if beat_led:
        reasons.extend(("stable measured beat intervals", "usable beat-grid coverage"))
        if transient >= 0.45 or pulse >= 0.45:
            reasons.append("clear transient or pulse evidence")
    else:
        if not tempo_valid or confidence < 0.42:
            reasons.append("low beat-tracking confidence")
        if regularity < 0.52:
            reasons.append("unstable beat intervals")
        if coverage < 0.45 or beat_count < minimum_beats:
            reasons.append("insufficient beat-grid coverage")
        if transient < 0.14:
            reasons.append("low transient salience despite a plausible tempo")
        elif pulse < 0.30 and density_score < 0.30:
            reasons.append("sparse transient evidence")
        if not reasons:
            reasons.append("phrase continuity is safer than per-beat cutting")
    distance = abs(score - 0.56) / 0.44
    return {
        "mode": mode,
        "confidence": _finite_float(min(1.0, 0.52 + 0.48 * distance), 4),
        "suitability_score": _finite_float(score, 4),
        "beat_interval_stability": _finite_float(regularity, 4),
        "beat_grid_coverage": _finite_float(coverage, 4),
        "onset_density_score": _finite_float(density_score, 4),
        "transient_salience": _finite_float(transient, 4),
        "reasons": reasons,
    }


def _mean_between(values: Any, times: Any, start: float, end: float, np: Any) -> float:
    left = int(np.searchsorted(times, start, side="left"))
    right = int(np.searchsorted(times, end, side="right"))
    left = max(0, min(len(values) - 1, left))
    right = max(left + 1, min(len(values), right))
    return float(np.mean(values[left:right]))


def _detect_energy_events(
    *,
    times: Any,
    energy: Any,
    density: Any,
    onset: Any,
    accent: Any,
    silence_intervals: Sequence[dict[str, Any]],
    low_energy_intervals: Sequence[dict[str, Any]],
    duration: float,
    frame_rate: float,
    np: Any,
) -> dict[str, list[dict[str, Any]]]:
    """Detect deterministic edit-significant energy events from signal curves."""

    if len(times) == 0:
        return {"hard_stops": [], "drops": [], "surges": [], "climaxes": []}
    window = max(1, int(round(0.34 * frame_rate)))
    before = np.roll(energy, window)
    after = np.roll(energy, -window)
    before[:window] = energy[0]
    after[-window:] = energy[-1]
    change = after - before
    rise_floor = max(0.105, float(np.percentile(change, 88)))
    rise_peaks = _pick_peak_indices(
        change,
        distance=max(1, int(round(0.85 * frame_rate))),
        height=rise_floor,
        prominence=0.025,
        np=np,
    )
    drops: list[dict[str, Any]] = []
    surges: list[dict[str, Any]] = []
    for index in rise_peaks:
        delta = float(change[index])
        if delta < 0.105:
            continue
        time_value = float(times[index])
        if time_value <= 0.08 or time_value >= duration - 0.08:
            continue
        after_level = _mean_between(
            energy, times, time_value, min(duration, time_value + 0.55), np
        )
        before_level = _mean_between(
            energy, times, max(0.0, time_value - 0.55), time_value, np
        )
        local_onset = float(onset[index])
        local_density = float(density[index])
        strength = max(
            0.0,
            min(
                1.0,
                0.48 * min(1.0, delta / 0.42)
                + 0.24 * after_level
                + 0.16 * local_onset
                + 0.12 * local_density,
            ),
        )
        record = {
            "time": _finite_float(time_value, 4),
            "strength": _finite_float(strength, 4),
            "confidence": _finite_float(0.42 + 0.50 * strength, 4),
            "energy_before": _finite_float(before_level, 4),
            "energy_after": _finite_float(after_level, 4),
            "energy_delta": _finite_float(after_level - before_level, 4),
            "onset_strength": _finite_float(local_onset, 4),
            "density": _finite_float(local_density, 4),
        }
        if delta >= 0.19 and after_level >= 0.54 and (
            local_onset >= 0.48 or local_density >= 0.46
        ):
            record["type"] = "drop"
            drops.append(record)
        else:
            record["type"] = "surge"
            surges.append(record)

    rise_event_limit = max(2, int(math.ceil(duration / 4.0)))
    combined_rises = drops + surges
    if len(combined_rises) > rise_event_limit:
        selected_rises = sorted(
            combined_rises,
            key=lambda item: (
                float(item.get("strength", 0.0)) + (0.06 if item.get("type") == "drop" else 0.0),
                -float(item.get("time", 0.0)),
            ),
            reverse=True,
        )[:rise_event_limit]
        selected_ids = {id(item) for item in selected_rises}
        drops = [item for item in drops if id(item) in selected_ids]
        surges = [item for item in surges if id(item) in selected_ids]
        drops.sort(key=lambda item: float(item["time"]))
        surges.sort(key=lambda item: float(item["time"]))

    hard_stops: list[dict[str, Any]] = []
    stop_candidates = sorted(
        {
            round(float(region.get("start", -1.0)), 4)
            for region in (*silence_intervals, *low_energy_intervals)
            if 0.08 < float(region.get("start", -1.0)) < duration - 0.05
        }
    )
    for time_value in stop_candidates:
        before_level = _mean_between(
            energy, times, max(0.0, time_value - 0.42), time_value, np
        )
        after_level = _mean_between(
            energy, times, time_value, min(duration, time_value + 0.32), np
        )
        delta = before_level - after_level
        if before_level < 0.28 or delta < 0.22:
            continue
        strength = max(0.0, min(1.0, 0.58 * delta / 0.55 + 0.42 * before_level))
        hard_stops.append(
            {
                "type": "hard_stop",
                "time": _finite_float(time_value, 4),
                "strength": _finite_float(strength, 4),
                "confidence": _finite_float(0.45 + 0.48 * strength, 4),
                "energy_before": _finite_float(before_level, 4),
                "energy_after": _finite_float(after_level, 4),
                "energy_delta": _finite_float(-delta, 4),
            }
        )

    impact = _smooth(
        0.60 * energy + 0.24 * density + 0.16 * accent,
        max(1, int(round(0.45 * frame_rate))),
        np,
    )
    climaxes: list[dict[str, Any]] = []
    if impact.size:
        usable = np.flatnonzero((times >= min(0.20, duration * 0.05)) & (times <= duration - min(0.20, duration * 0.05)))
        if usable.size:
            ranked = sorted(usable, key=lambda index: float(impact[index]), reverse=True)
            selected: list[int] = []
            for index in ranked:
                if float(impact[index]) < 0.40:
                    break
                if all(abs(float(times[index]) - float(times[other])) >= 2.0 for other in selected):
                    selected.append(int(index))
                if len(selected) >= 2:
                    break
            for index in sorted(selected):
                climaxes.append(
                    {
                        "type": "climax",
                        "time": _finite_float(times[index], 4),
                        "strength": _finite_float(impact[index], 4),
                        "confidence": _finite_float(0.45 + 0.48 * float(impact[index]), 4),
                        "energy": _finite_float(energy[index], 4),
                        "density": _finite_float(density[index], 4),
                    }
                )
    return {
        "hard_stops": hard_stops,
        "drops": drops,
        "surges": surges,
        "climaxes": climaxes,
    }


def _adaptive_energy_points(
    times: Any,
    energy: Any,
    local_energy: Any,
    rms_db: Any,
    onset: Any,
    accent_indices: Iterable[int],
    boundary_times: Sequence[float],
    duration: float,
    np: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(times) == 0:
        return [], {"base_step_seconds": 0.0, "preserved_events": []}
    base_step = 0.25 if duration <= 60 else 0.5 if duration <= 180 else 1.0
    frame_step = float(np.median(np.diff(times))) if len(times) > 1 else base_step
    stride = max(1, int(round(base_step / max(frame_step, 1e-6))))
    selected = set(range(0, len(times), stride))
    selected.update({0, len(times) - 1})

    derivative = np.abs(np.diff(energy, prepend=energy[0]))
    if derivative.size:
        threshold = float(np.percentile(derivative, 94))
        selected.update(int(index) for index in np.flatnonzero(derivative >= threshold))
    for index in accent_indices:
        selected.add(int(index))
    for boundary in boundary_times:
        selected.add(_nearest_index(times, float(boundary), np))

    ordered = sorted(index for index in selected if 0 <= index < len(times))
    max_points = 900
    if len(ordered) > max_points:
        mandatory = {
            0,
            len(times) - 1,
            *[int(index) for index in accent_indices],
            *[_nearest_index(times, float(boundary), np) for boundary in boundary_times],
        }
        remaining_slots = max(0, max_points - len(mandatory))
        optional = [index for index in ordered if index not in mandatory]
        if remaining_slots and optional:
            positions = np.linspace(0, len(optional) - 1, remaining_slots, dtype=int)
            mandatory.update(optional[int(position)] for position in positions)
        ordered = sorted(index for index in mandatory if 0 <= index < len(times))

    points = [
        {
            "time": _finite_float(min(duration, max(0.0, float(times[index]))), 4),
            "level": _finite_float(energy[index], 4),
            # ``energy`` is a compatibility alias used by the montage planner;
            # ``level`` remains the canonical schema name.
            "energy": _finite_float(energy[index], 4),
            "local_level": _finite_float(local_energy[index], 4),
            "rms_db_relative_to_peak": _finite_float(rms_db[index], 3),
            "onset_strength": _finite_float(onset[index], 4),
        }
        for index in ordered
    ]
    return points, {
        "base_step_seconds": base_step,
        "point_count": len(points),
        "preserved_events": ["rapid_energy_changes", "accents", "section_boundaries"],
        "normalization": "global robust percentiles blended with an 8-second local baseline",
    }


def _analyze_signal(
    waveform: Any,
    sample_rate: int,
    analyzed_duration: float,
    librosa: Any,
    np: Any,
) -> dict[str, Any]:
    if analyzed_duration <= 0 or len(waveform) < 32:
        raise ValueError("The decoded BGM is empty or too short to analyze.")

    y = np.asarray(waveform, dtype=np.float32)
    y = np.nan_to_num(y, copy=False)
    y = y - float(np.mean(y))
    if len(y) < 512:
        y = np.pad(y, (0, 512 - len(y)))
    n_fft = 2048 if len(y) >= 2048 else 512
    hop_length = min(512, n_fft // 4)
    frame_rate = float(sample_rate) / float(hop_length)

    spectrum = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length, center=True))
    power = np.maximum(spectrum**2, 1e-12)
    rms = librosa.feature.rms(S=spectrum, frame_length=n_fft, hop_length=hop_length)[0]
    onset_raw = librosa.onset.onset_strength(y=y, sr=sample_rate, hop_length=hop_length)
    centroid = librosa.feature.spectral_centroid(S=spectrum, sr=sample_rate)[0]
    rolloff = librosa.feature.spectral_rolloff(S=spectrum, sr=sample_rate, roll_percent=0.85)[0]
    bandwidth = librosa.feature.spectral_bandwidth(S=spectrum, sr=sample_rate)[0]
    zcr = librosa.feature.zero_crossing_rate(
        y, frame_length=n_fft, hop_length=hop_length, center=True
    )[0]
    flatness = librosa.feature.spectral_flatness(S=spectrum)[0]
    log_power = librosa.power_to_db(power, ref=np.max)
    mfcc = librosa.feature.mfcc(S=log_power, sr=sample_rate, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(S=spectrum, sr=sample_rate)

    feature_lengths = [
        spectrum.shape[1],
        len(rms),
        len(onset_raw),
        len(centroid),
        len(rolloff),
        len(bandwidth),
        len(zcr),
        len(flatness),
        mfcc.shape[1],
        chroma.shape[1],
    ]
    frame_count = max(1, min(feature_lengths))
    times = librosa.frames_to_time(
        np.arange(frame_count), sr=sample_rate, hop_length=hop_length
    )
    valid_count = max(1, int(np.searchsorted(times, analyzed_duration + 1e-9, side="right")))
    frame_count = min(frame_count, valid_count)
    times = times[:frame_count]
    spectrum = spectrum[:, :frame_count]
    rms = rms[:frame_count]
    onset_raw = onset_raw[:frame_count]
    centroid = centroid[:frame_count]
    rolloff = rolloff[:frame_count]
    bandwidth = bandwidth[:frame_count]
    zcr = zcr[:frame_count]
    flatness = flatness[:frame_count]
    mfcc = mfcc[:, :frame_count]
    chroma = chroma[:, :frame_count]

    rms_db = 20.0 * np.log10(np.maximum(rms, 1e-10) / max(float(np.max(rms)), 1e-10))
    rms_db = np.clip(rms_db, -80.0, 0.0)
    global_energy = _robust_normalize(rms_db, np, 8, 94)
    local_width = max(3, int(round(8.0 * frame_rate)))
    local_baseline = _smooth(rms_db, local_width, np)
    local_energy = np.clip(0.5 + (rms_db - local_baseline) / 18.0, 0.0, 1.0)
    energy = _smooth(0.68 * global_energy + 0.32 * local_energy, max(1, int(0.22 * frame_rate)), np)
    onset = _robust_normalize(onset_raw, np, 20, 97)

    positive_delta = np.maximum(np.diff(energy, prepend=energy[0]), 0.0)
    positive_delta = _robust_normalize(positive_delta, np, 20, 98)
    spectral_delta = np.diff(spectrum, axis=1, prepend=spectrum[:, :1])
    spectral_flux = np.sqrt(np.sum(np.maximum(spectral_delta, 0.0) ** 2, axis=0))
    spectral_flux = _robust_normalize(spectral_flux, np, 15, 96)
    accent_curve = _smooth(
        0.55 * onset + 0.25 * spectral_flux + 0.20 * positive_delta,
        max(1, int(0.06 * frame_rate)),
        np,
    )

    try:
        tempo_raw, beat_frames_raw = librosa.beat.beat_track(
            onset_envelope=onset_raw,
            sr=sample_rate,
            hop_length=hop_length,
            trim=False,
        )
        tempo = float(np.asarray(tempo_raw).reshape(-1)[0])
        beat_frames = np.asarray(beat_frames_raw, dtype=int)
    except Exception:
        tempo = 0.0
        beat_frames = np.asarray([], dtype=int)
    beat_frames = beat_frames[(beat_frames >= 0) & (beat_frames < frame_count)]
    beat_times = times[beat_frames] if beat_frames.size else np.asarray([], dtype=float)
    beat_strengths = accent_curve[beat_frames] if beat_frames.size else np.asarray([], dtype=float)
    if beat_times.size >= 2:
        intervals = np.diff(beat_times)
        median_interval = float(np.median(intervals))
        inferred_tempo = 60.0 / median_interval if median_interval > 1e-6 else tempo
        if not 30.0 <= tempo <= 260.0:
            tempo = inferred_tempo
        interval_cv = float(np.std(intervals) / max(np.mean(intervals), 1e-6))
    else:
        interval_cv = 1.0
    if not math.isfinite(tempo) or tempo < 0:
        tempo = 0.0
    pulse_strength = float(np.mean(beat_strengths)) if beat_strengths.size else 0.0
    beat_confidence = max(
        0.0,
        min(
            1.0,
            0.45 * pulse_strength
            + 0.35 * max(0.0, 1.0 - min(1.0, interval_cv))
            + 0.20 * min(1.0, beat_times.size / 16.0),
        ),
    )
    meter, downbeat_phase, meter_confidence = _estimate_meter(beat_strengths, np)
    downbeat_indices = [
        index for index in range(len(beat_times)) if (index - downbeat_phase) % meter == 0
    ]
    downbeat_times = [float(beat_times[index]) for index in downbeat_indices]

    accent_height = float(np.percentile(accent_curve, 78)) if accent_curve.size else 1.0
    accent_peaks = _pick_peak_indices(
        accent_curve,
        distance=max(1, int(round(0.18 * frame_rate))),
        height=accent_height,
        prominence=0.06,
        np=np,
    )
    if len(accent_peaks) > max(64, int(analyzed_duration * 4)):
        ranked = sorted(accent_peaks, key=lambda index: float(accent_curve[index]), reverse=True)
        accent_peaks = np.asarray(
            sorted(ranked[: max(64, int(analyzed_duration * 4))]), dtype=int
        )

    onset_height = float(np.percentile(onset, 62)) if onset.size else 1.0
    onset_peaks = _pick_peak_indices(
        onset,
        distance=max(1, int(round(0.085 * frame_rate))),
        height=max(0.08, onset_height),
        prominence=0.025,
        np=np,
    )
    if len(onset_peaks) > max(96, int(analyzed_duration * 8)):
        ranked_onsets = sorted(onset_peaks, key=lambda index: float(onset[index]), reverse=True)
        onset_peaks = np.asarray(
            sorted(ranked_onsets[: max(96, int(analyzed_duration * 8))]), dtype=int
        )
    onset_markers = np.zeros(frame_count, dtype=float)
    onset_markers[onset_peaks] = 1.0
    density_window = max(1, min(frame_count, int(round(1.0 * frame_rate))))
    density_per_second = np.convolve(
        onset_markers,
        np.ones(density_window, dtype=float),
        mode="same",
    ) * frame_rate / density_window
    density_normalized = _smooth(
        _robust_normalize(density_per_second, np, 5, 95),
        max(1, int(round(0.20 * frame_rate))),
        np,
    )

    beats: list[dict[str, Any]] = []
    for index, beat_time in enumerate(beat_times):
        beats.append(
            {
                "time": _finite_float(beat_time, 4),
                "strength": _finite_float(beat_strengths[index], 4),
                "beat_in_bar": int(((index - downbeat_phase) % meter) + 1),
                "downbeat_estimate": bool(index in downbeat_indices),
            }
        )

    accents: list[dict[str, Any]] = []
    for frame_index in accent_peaks:
        time_value = float(times[frame_index])
        strength = float(accent_curve[frame_index])
        nearest_delta: Optional[float] = None
        if beat_times.size:
            beat_index = _nearest_index(beat_times, time_value, np)
            nearest_delta = time_value - float(beat_times[beat_index])
        accents.append(
            {
                "time": _finite_float(time_value, 4),
                "strength": _finite_float(strength, 4),
                "level": "strong" if strength >= 0.78 else "medium",
                "nearest_beat_delta_seconds": (
                    None if nearest_delta is None else _finite_float(nearest_delta, 4)
                ),
            }
        )

    onsets: list[dict[str, Any]] = []
    for frame_index in onset_peaks:
        time_value = float(times[frame_index])
        strength = float(onset[frame_index])
        nearest_delta: Optional[float] = None
        if beat_times.size:
            beat_index = _nearest_index(beat_times, time_value, np)
            nearest_delta = time_value - float(beat_times[beat_index])
        onsets.append(
            {
                "time": _finite_float(time_value, 4),
                "strength": _finite_float(strength, 4),
                "level": "strong" if strength >= 0.72 else "medium" if strength >= 0.42 else "soft",
                "nearest_beat_delta_seconds": (
                    None if nearest_delta is None else _finite_float(nearest_delta, 4)
                ),
            }
        )

    # A vocal-likelihood heuristic: harmonic dominance + energy in the common
    # singing/speech band + non-noisy tonality. Instrumental leads can trigger it.
    try:
        harmonic, percussive = librosa.decompose.hpss(spectrum)
        harmonic_ratio = np.sum(harmonic, axis=0) / np.maximum(
            np.sum(harmonic + percussive, axis=0), 1e-9
        )
    except Exception:
        harmonic_ratio = np.full(frame_count, 0.5, dtype=float)
    frequencies = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)
    vocal_band = (frequencies >= 180.0) & (frequencies <= 4_500.0)
    band_ratio = np.sum(spectrum[vocal_band, :], axis=0) / np.maximum(
        np.sum(spectrum, axis=0), 1e-9
    )
    tonal_score = np.clip(1.0 - flatness * 8.0, 0.0, 1.0)
    vocal_score = np.clip(
        0.43 * harmonic_ratio + 0.29 * band_ratio + 0.16 * tonal_score + 0.12 * energy,
        0.0,
        1.0,
    )
    vocal_score = _smooth(vocal_score, max(1, int(round(0.7 * frame_rate))), np)
    energy_weights = np.maximum(energy, 0.05)
    overall_vocal = float(np.average(vocal_score, weights=energy_weights))
    vocal_intervals = _merge_boolean_regions(
        (vocal_score >= 0.60) & (energy >= 0.12),
        times,
        analyzed_duration,
        np,
        min_duration=0.8,
        max_gap=0.35,
        score=vocal_score,
    )

    pause_threshold_db = max(-55.0, min(-28.0, float(np.percentile(rms_db, 18)) - 3.0))
    pauses = _merge_boolean_regions(
        rms_db <= pause_threshold_db,
        times,
        analyzed_duration,
        np,
        min_duration=0.18,
        max_gap=0.12,
    )
    for pause in pauses:
        start_i = _nearest_index(times, float(pause["start"]), np)
        end_i = _nearest_index(times, float(pause["end"]), np) + 1
        pause["minimum_db_relative_to_peak"] = _finite_float(np.min(rms_db[start_i:end_i]), 2)

    low_energy_mask = (energy <= 0.22) | (
        rms_db <= min(-18.0, float(np.percentile(rms_db, 22)))
    )
    low_energy_intervals = _merge_boolean_regions(
        low_energy_mask,
        times,
        analyzed_duration,
        np,
        min_duration=0.45,
        max_gap=0.20,
    )
    for interval in low_energy_intervals:
        start_i = _nearest_index(times, float(interval["start"]), np)
        end_i = _nearest_index(times, float(interval["end"]), np) + 1
        interval["mean_energy"] = _finite_float(np.mean(energy[start_i:end_i]), 4)
        interval["minimum_db_relative_to_peak"] = _finite_float(
            np.min(rms_db[start_i:end_i]), 2
        )

    energy_events = _detect_energy_events(
        times=times,
        energy=energy,
        density=density_normalized,
        onset=onset,
        accent=accent_curve,
        silence_intervals=pauses,
        low_energy_intervals=low_energy_intervals,
        duration=analyzed_duration,
        frame_rate=frame_rate,
        np=np,
    )
    overall_onset_density = len(onsets) / max(analyzed_duration, 1e-9)
    onset_raw_median = float(np.median(onset_raw)) if onset_raw.size else 0.0
    onset_raw_p95 = float(np.percentile(onset_raw, 95)) if onset_raw.size else 0.0
    onset_peak_ratio = onset_raw_p95 / max(1e-6, onset_raw_median)
    transient_salience = max(
        0.0,
        min(1.0, (math.log10(1.0 + onset_peak_ratio) - 0.60) / 2.0),
    )
    rhythm_mode = _classify_rhythm_mode(
        tempo=tempo,
        beat_confidence=beat_confidence,
        interval_cv=interval_cv,
        pulse_strength=pulse_strength,
        onset_density=overall_onset_density,
        transient_salience=transient_salience,
        beat_count=len(beats),
        duration=analyzed_duration,
    )

    # Section novelty compares feature means on both sides of each point.
    normalized_mfcc = np.vstack(
        [_robust_normalize(mfcc[index], np, 5, 95) for index in range(1, min(9, mfcc.shape[0]))]
    )
    normalized_chroma = np.vstack(
        [_robust_normalize(chroma[index], np, 5, 95) for index in range(chroma.shape[0])]
    )
    structure_features = np.vstack(
        [normalized_mfcc, normalized_chroma, energy[None, :], onset[None, :]]
    )
    comparison_width = max(2, int(round(min(2.5, max(0.75, analyzed_duration / 30.0)) * frame_rate)))
    novelty = np.zeros(frame_count, dtype=float)
    for index in range(comparison_width, frame_count - comparison_width):
        before = np.mean(structure_features[:, index - comparison_width : index], axis=1)
        after = np.mean(structure_features[:, index : index + comparison_width], axis=1)
        novelty[index] = float(np.linalg.norm(after - before))
    novelty = _smooth(_robust_normalize(novelty, np, 20, 98), max(1, int(0.35 * frame_rate)), np)
    min_section_seconds = min(8.0, max(3.0, analyzed_duration / 10.0))
    boundary_peak_indices = _pick_peak_indices(
        novelty,
        distance=max(1, int(round(min_section_seconds * frame_rate))),
        height=float(np.percentile(novelty, 72)) if novelty.size else 1.0,
        prominence=0.08,
        np=np,
    )
    max_internal_boundaries = max(0, min(10, int(analyzed_duration // max(6.0, min_section_seconds))))
    ranked_boundaries = sorted(
        boundary_peak_indices, key=lambda index: float(novelty[index]), reverse=True
    )[:max_internal_boundaries]
    raw_boundary_times = sorted(float(times[index]) for index in ranked_boundaries)
    aligned_boundary_times: list[float] = []
    for candidate in raw_boundary_times:
        aligned = candidate
        if downbeat_times:
            nearest_downbeat = min(downbeat_times, key=lambda value: abs(value - candidate))
            if abs(nearest_downbeat - candidate) <= 1.25:
                aligned = nearest_downbeat
        if aligned <= 1.0 or analyzed_duration - aligned <= 1.0:
            continue
        if aligned_boundary_times and aligned - aligned_boundary_times[-1] < min_section_seconds * 0.65:
            previous = aligned_boundary_times[-1]
            if novelty[_nearest_index(times, aligned, np)] > novelty[_nearest_index(times, previous, np)]:
                aligned_boundary_times[-1] = aligned
            continue
        aligned_boundary_times.append(aligned)
    if aligned_boundary_times and analyzed_duration - aligned_boundary_times[-1] < min_section_seconds * 0.65:
        aligned_boundary_times.pop()
    section_boundaries = [0.0, *aligned_boundary_times, analyzed_duration]

    boundary_records = []
    for boundary in aligned_boundary_times:
        index = _nearest_index(times, boundary, np)
        boundary_records.append(
            {
                "time": _finite_float(boundary, 4),
                "novelty": _finite_float(novelty[index], 4),
                "aligned_to_downbeat": any(abs(boundary - value) < 1e-4 for value in downbeat_times),
                "confidence": _finite_float(0.35 + 0.6 * novelty[index], 4),
            }
        )

    global_centroid_normalized = _robust_normalize(centroid, np, 5, 95)
    accent_times = np.asarray([float(times[index]) for index in accent_peaks], dtype=float)
    onset_times = np.asarray([float(times[index]) for index in onset_peaks], dtype=float)
    sections: list[dict[str, Any]] = []
    section_vectors: list[Any] = []
    for section_index, (start, end) in enumerate(zip(section_boundaries[:-1], section_boundaries[1:])):
        start_i = int(np.searchsorted(times, start, side="left"))
        end_i = int(np.searchsorted(times, end, side="left"))
        end_i = min(frame_count, max(start_i + 1, end_i))
        section_energy = float(np.mean(energy[start_i:end_i]))
        section_peak = float(np.max(energy[start_i:end_i]))
        quarter = max(1, (end_i - start_i) // 4)
        energy_trend = float(
            np.mean(energy[end_i - quarter : end_i]) - np.mean(energy[start_i : start_i + quarter])
        )
        accent_count = int(np.sum((accent_times >= start) & (accent_times < end)))
        accent_density = accent_count / max(end - start, 1e-6)
        section_onset_count = int(np.sum((onset_times >= start) & (onset_times < end)))
        section_onset_density = section_onset_count / max(end - start, 1e-6)
        section_density = float(np.mean(density_normalized[start_i:end_i]))
        section_brightness = float(np.mean(global_centroid_normalized[start_i:end_i]))
        section_vocal = float(np.average(vocal_score[start_i:end_i], weights=energy_weights[start_i:end_i]))
        section_vectors.append(np.mean(structure_features[:, start_i:end_i], axis=1))
        mood, mood_traits = _mood_label(
            section_energy, section_onset_density, section_brightness, section_vocal, tempo
        )
        section_rhythm_mode = (
            "beat_cut"
            if rhythm_mode["mode"] == "beat_cut"
            and section_onset_density >= 0.32
            and float(np.mean(onset[start_i:end_i])) >= 0.10
            else "phrase_flow"
        )
        cut_intensity = max(
            0.05,
            min(
                1.0,
                0.14
                + 0.50 * section_energy
                + 0.26 * section_density
                + 0.10 * max(0.0, min(1.0, energy_trend + 0.2)),
            ),
        )
        if section_rhythm_mode == "phrase_flow":
            cut_intensity *= 0.76
        if tempo > 0 and section_rhythm_mode == "beat_cut":
            beat_seconds = 60.0 / tempo
            if section_energy >= 0.68 or section_onset_density >= 2.0:
                shot_range = [max(0.35, beat_seconds), max(0.7, beat_seconds * 2.0)]
                cut_style = "accent-and-beat cuts"
                motion = "high"
            elif section_energy < 0.36 and section_onset_density < 1.0:
                shot_range = [beat_seconds * 4.0, beat_seconds * 8.0]
                cut_style = "phrase-led flowing cuts"
                motion = "low"
            else:
                shot_range = [beat_seconds * 2.0, beat_seconds * 4.0]
                cut_style = "downbeat-led cuts"
                motion = "medium"
        else:
            phrase_target = max(1.35, min(6.0, 2.0 + (1.0 - section_energy) * 2.8))
            shot_range = [max(0.85, phrase_target * 0.72), min(7.5, phrase_target * 1.38)]
            cut_style = "phrase-flow cuts"
            motion = "low" if section_energy < 0.40 else "medium"
        sections.append(
            {
                "index": section_index,
                "start": _finite_float(start, 4),
                "end": _finite_float(end, 4),
                "duration": _finite_float(end - start, 4),
                "role": "unassigned",
                "role_confidence": 0.0,
                "energy": {
                    "mean": _finite_float(section_energy, 4),
                    "peak": _finite_float(section_peak, 4),
                    "trend": _finite_float(energy_trend, 4),
                    "trend_label": "rising" if energy_trend > 0.08 else "falling" if energy_trend < -0.08 else "steady",
                },
                "rhythm": {
                    "accent_count": accent_count,
                    "accent_density_per_second": _finite_float(accent_density, 4),
                    "onset_count": section_onset_count,
                    "onset_density_per_second": _finite_float(section_onset_density, 4),
                    "density": _finite_float(section_density, 4),
                    "density_label": "dense" if section_onset_density >= 2.2 else "sparse" if section_onset_density < 0.75 else "moderate",
                    "pulse_strength": _finite_float(np.mean(onset[start_i:end_i]), 4),
                },
                "timbre": {
                    "brightness": _finite_float(section_brightness, 4),
                    "brightness_label": "bright" if section_brightness >= 0.62 else "warm" if section_brightness < 0.36 else "balanced",
                    "spectral_centroid_hz_mean": _finite_float(np.mean(centroid[start_i:end_i]), 2),
                    "spectral_rolloff_hz_mean": _finite_float(np.mean(rolloff[start_i:end_i]), 2),
                    "zero_crossing_rate_mean": _finite_float(np.mean(zcr[start_i:end_i]), 5),
                },
                "vocal_likelihood": _finite_float(section_vocal, 4),
                "estimated_mood": mood,
                "mood_traits": mood_traits,
                "rhythm_mode": section_rhythm_mode,
                "edit_guidance": {
                    "cut_style": cut_style,
                    "rhythm_mode": section_rhythm_mode,
                    "cut_intensity": _finite_float(cut_intensity, 4),
                    "recommended_shot_duration_seconds": [
                        _finite_float(shot_range[0], 3),
                        _finite_float(shot_range[1], 3),
                    ],
                    "visual_motion_intensity": motion,
                    "shot_scale_strategy": (
                        "vary wide/medium/close on successive accents"
                        if motion == "high"
                        else "favor wide-to-medium continuity" if motion == "low" else "alternate wide and detail shots"
                    ),
                },
            }
        )

    if len(sections) == 1:
        sections[0]["legacy_role"] = "main"
        sections[0]["role"] = "climax"
        sections[0]["role_confidence"] = 0.42
    else:
        candidate_indices = list(range(1, len(sections) - 1)) or list(range(len(sections)))
        peak_index = max(
            candidate_indices,
            key=lambda index: sections[index]["energy"]["mean"]
            * (1.0 + 0.18 * sections[index]["rhythm"]["density"]),
        )
        drop_times = [float(item["time"]) for item in energy_events["drops"]]
        climax_times = [float(item["time"]) for item in energy_events["climaxes"]]
        previous_energy = float(sections[0]["energy"]["mean"])
        for index, section in enumerate(sections):
            energy_value = float(section["energy"]["mean"])
            trend_value = float(section["energy"]["trend"])
            density_value = float(section["rhythm"]["density"])
            if index == 0:
                legacy_role = "intro"
            elif index == len(sections) - 1:
                legacy_role = "outro"
            elif index == peak_index:
                legacy_role = "peak"
            elif energy_value < 0.34:
                legacy_role = "breakdown"
            elif index < peak_index or trend_value > 0.08:
                legacy_role = "build"
            elif trend_value < -0.08:
                legacy_role = "release"
            else:
                legacy_role = "sustain"

            starts_with_drop = any(
                float(section["start"]) - 0.20 <= time_value <= float(section["start"]) + 1.10
                for time_value in drop_times
            )
            contains_climax = any(
                float(section["start"]) <= time_value < float(section["end"])
                for time_value in climax_times
            )
            energy_jump = energy_value - previous_energy
            if index == 0:
                role, role_confidence = "intro", 0.78
            elif index == len(sections) - 1:
                role, role_confidence = "outro", 0.78
            elif starts_with_drop or (energy_jump >= 0.18 and energy_value >= 0.54):
                role = "drop"
                role_confidence = min(0.94, 0.62 + max(0.0, energy_jump) * 0.8)
            elif index == peak_index or contains_climax:
                role = "climax"
                role_confidence = min(0.94, 0.58 + 0.22 * energy_value + 0.16 * density_value)
            elif energy_value < 0.34 or energy_jump <= -0.17:
                role = "break"
                role_confidence = min(0.90, 0.58 + abs(min(0.0, energy_jump)) * 0.8)
            else:
                role = "build"
                role_confidence = min(0.88, 0.52 + 0.18 * max(0.0, trend_value) + 0.12 * density_value)
            section["legacy_role"] = legacy_role
            section["role"] = role
            section["role_confidence"] = _finite_float(role_confidence, 4)
            section["edit_guidance"]["section_role"] = role
            previous_energy = energy_value

    # Detect recurring/loop-like sections from the same MFCC, chroma, energy
    # and onset representation used for boundary discovery.  This is a
    # similarity estimate, not a claim about compositional intent.
    parent = list(range(len(sections)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    pair_similarity: dict[tuple[int, int], float] = {}
    for left in range(len(section_vectors)):
        for right in range(left + 1, len(section_vectors)):
            left_vector = section_vectors[left]
            right_vector = section_vectors[right]
            denominator = float(np.linalg.norm(left_vector) * np.linalg.norm(right_vector))
            similarity = float(np.dot(left_vector, right_vector) / denominator) if denominator > 1e-9 else 0.0
            duration_ratio = min(sections[left]["duration"], sections[right]["duration"]) / max(
                sections[left]["duration"], sections[right]["duration"], 1e-9
            )
            pair_similarity[(left, right)] = similarity
            if similarity >= 0.90 and duration_ratio >= 0.55:
                union(left, right)

    members_by_root: dict[int, list[int]] = {}
    for index in range(len(sections)):
        members_by_root.setdefault(find(index), []).append(index)
    repeated_groups = [members for members in members_by_root.values() if len(members) >= 2]
    loop_groups: list[dict[str, Any]] = []
    for group_number, members in enumerate(repeated_groups, start=1):
        group_id = f"repeat_{group_number}"
        similarities = [
            pair_similarity.get((min(left, right), max(left, right)), 0.0)
            for position, left in enumerate(members)
            for right in members[position + 1 :]
        ]
        for index in members:
            similar_to = [
                {"section_index": other, "similarity": _finite_float(pair_similarity.get((min(index, other), max(index, other)), 0.0), 4)}
                for other in members
                if other != index
            ]
            sections[index]["repetition"] = {
                "is_repeated": True,
                "group": group_id,
                "similar_to": similar_to,
                "editing_guidance": "change scene, shot scale and motion treatment across repeated passes",
            }
        loop_groups.append(
            {
                "group": group_id,
                "section_indices": members,
                "mean_similarity": _finite_float(np.mean(similarities) if similarities else 0.0, 4),
                "method": "cosine similarity over section MFCC, chroma, energy and onset means",
            }
        )
    for index, section in enumerate(sections):
        section.setdefault(
            "repetition",
            {"is_repeated": False, "group": None, "similar_to": [], "editing_guidance": None},
        )

    phrase_boundaries: list[float] = [0.0]
    downbeat_array = np.asarray(downbeat_times, dtype=float)
    bars_per_phrase = 4
    for index in range(bars_per_phrase, len(downbeat_times), bars_per_phrase):
        candidate = float(downbeat_times[index])
        if candidate - phrase_boundaries[-1] >= 2.0:
            phrase_boundaries.append(candidate)
    for boundary in aligned_boundary_times:
        if all(abs(boundary - existing) > 1.0 for existing in phrase_boundaries):
            phrase_boundaries.append(boundary)
    phrase_boundaries = sorted(phrase_boundaries)
    if analyzed_duration - phrase_boundaries[-1] < 1.0 and len(phrase_boundaries) > 1:
        phrase_boundaries.pop()
    phrase_boundaries.append(analyzed_duration)
    phrases: list[dict[str, Any]] = []
    for index, (start, end) in enumerate(zip(phrase_boundaries[:-1], phrase_boundaries[1:])):
        start_i = int(np.searchsorted(times, start, side="left"))
        end_i = min(frame_count, max(start_i + 1, int(np.searchsorted(times, end, side="left"))))
        phrase_beats = int(np.sum((beat_times >= start) & (beat_times < end)))
        phrases.append(
            {
                "index": index,
                "start": _finite_float(start, 4),
                "end": _finite_float(end, 4),
                "duration": _finite_float(end - start, 4),
                "beat_count": phrase_beats,
                "bar_count_estimate": _finite_float(phrase_beats / max(1, meter), 2),
                "mean_energy": _finite_float(np.mean(energy[start_i:end_i]), 4),
                "energy_trend": _finite_float(energy[end_i - 1] - energy[start_i], 4),
            }
        )

    energy_points, energy_sampling = _adaptive_energy_points(
        times,
        energy,
        local_energy,
        rms_db,
        onset,
        accent_peaks,
        aligned_boundary_times,
        analyzed_duration,
        np,
    )
    density_points: list[dict[str, Any]] = []
    onset_strength_points: list[dict[str, Any]] = []
    for point in energy_points:
        point_time = float(point["time"])
        frame_index = _nearest_index(times, point_time, np)
        density_points.append(
            {
                "time": _finite_float(point_time, 4),
                "onsets_per_second": _finite_float(density_per_second[frame_index], 4),
                "normalized": _finite_float(density_normalized[frame_index], 4),
            }
        )
        onset_strength_points.append(
            {
                "time": _finite_float(point_time, 4),
                "strength": _finite_float(onset[frame_index], 4),
            }
        )

    key_moments: list[dict[str, Any]] = []
    for event_type in ("hard_stops", "drops", "surges", "climaxes"):
        for event in energy_events[event_type]:
            record = dict(event)
            event_time = float(record.get("time", 0.0))
            active_section = next(
                (
                    section
                    for section in sections
                    if float(section["start"]) <= event_time < float(section["end"]) + 1e-6
                ),
                None,
            )
            record["section_index"] = active_section.get("index") if active_section else None
            record["section_role"] = active_section.get("role") if active_section else None
            key_moments.append(record)
    key_moments.sort(key=lambda item: (float(item.get("time", 0.0)), str(item.get("type", ""))))

    phrase_boundary_events = [
        {"time": _finite_float(value, 4), "confidence": _finite_float(0.52 + 0.34 * rhythm_mode["suitability_score"], 4)}
        for value in phrase_boundaries[1:-1]
    ]
    timbre = _feature_summary(
        0,
        frame_count,
        centroid,
        rolloff,
        bandwidth,
        zcr,
        flatness,
        mfcc,
        np,
    )
    dominant_chroma_index = int(np.argmax(np.mean(chroma, axis=1))) if chroma.size else 0
    pitch_classes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    timbre["dominant_chroma_estimate"] = pitch_classes[dominant_chroma_index]
    timbre["description"] = (
        ("bright" if float(np.mean(global_centroid_normalized)) >= 0.62 else "warm" if float(np.mean(global_centroid_normalized)) < 0.36 else "balanced")
        + ", "
        + ("percussive" if float(np.mean(onset)) >= 0.55 else "smooth")
    )

    stage_profile = [
        {
            "stage": section["role"],
            "start": section["start"],
            "end": section["end"],
            "energy": section["energy"]["mean"],
            "energy_trend": section["energy"]["trend_label"],
            "rhythm_density": section["rhythm"]["density_label"],
            "rhythm_mode": section["rhythm_mode"],
            "mood": section["estimated_mood"],
            "timbre": section["timbre"]["brightness_label"],
            "vocal_likelihood": section["vocal_likelihood"],
            "edit_guidance": section["edit_guidance"],
        }
        for section in sections
    ]

    recommended_targets = [
        sum(section["edit_guidance"]["recommended_shot_duration_seconds"])
        / max(1, len(section["edit_guidance"]["recommended_shot_duration_seconds"]))
        for section in sections
    ]
    overall_recommended = _finite_float(
        np.median(recommended_targets) if recommended_targets else 2.5, 3
    )
    tempo_profile = {
        "bpm": _finite_float(tempo, 3),
        "beat_period_seconds": _finite_float(60.0 / tempo, 5) if tempo > 0 else None,
        "confidence": _finite_float(beat_confidence, 4),
        "interval_cv": _finite_float(interval_cv, 4),
        "stability": rhythm_mode["beat_interval_stability"],
        "pulse_strength": _finite_float(pulse_strength, 4),
        "meter": f"{meter}/4",
        "meter_confidence": _finite_float(meter_confidence, 4),
    }
    event_map = {
        "beats": beats,
        "downbeats": [
            {"time": _finite_float(value, 4), "strength": _finite_float(beats[index]["strength"], 4)}
            for index, value in zip(downbeat_indices, downbeat_times)
        ],
        "onsets": onsets,
        "accents": accents,
        "hard_stops": energy_events["hard_stops"],
        "drops": energy_events["drops"],
        "surges": energy_events["surges"],
        "climaxes": energy_events["climaxes"],
        "phrase_boundaries": phrase_boundary_events,
        "section_boundaries": boundary_records,
    }

    return {
        "artifact_type": "audiomap",
        "duration_seconds": _finite_float(analyzed_duration, 5),
        "tempo": tempo_profile,
        "rhythm_mode": rhythm_mode,
        "events": event_map,
        "curves": {
            "energy": {"points": energy_points, "sampling": energy_sampling},
            "density": {
                "points": density_points,
                "unit": "detected_onsets_per_second",
                "window_seconds": 1.0,
            },
            "onset_strength": {"points": onset_strength_points},
        },
        "intervals": {
            "silence": pauses,
            "low_energy": low_energy_intervals,
            "vocal_likely": vocal_intervals,
        },
        "onsets": onsets,
        "density_curve": density_points,
        "silence_intervals": pauses,
        "low_energy_intervals": low_energy_intervals,
        "hard_stops": energy_events["hard_stops"],
        "drops": energy_events["drops"],
        "surges": energy_events["surges"],
        "climaxes": energy_events["climaxes"],
        "key_moments": key_moments,
        "editing_guidance": {
            "rhythm_mode": rhythm_mode["mode"],
            "recommended_shot_duration_seconds": overall_recommended,
            "cut_intensity": _finite_float(
                np.mean([section["edit_guidance"]["cut_intensity"] for section in sections])
                if sections
                else 0.4,
                4,
            ),
            "strategy": (
                "snap eligible boundaries to measured beats and strong accents"
                if rhythm_mode["mode"] == "beat_cut"
                else "follow phrases, section changes and energy arcs; do not cut every beat"
            ),
        },
        "reliability": {
            "overall": _finite_float(
                0.46 * beat_confidence
                + 0.24 * rhythm_mode["beat_interval_stability"]
                + 0.18 * min(1.0, len(onsets) / max(4.0, analyzed_duration))
                + 0.12 * min(1.0, len(sections) / 4.0),
                4,
            ),
            "tempo": _finite_float(beat_confidence, 4),
            "meter": _finite_float(meter_confidence, 4),
            "structure": _finite_float(
                np.mean([boundary.get("confidence", 0.0) for boundary in boundary_records])
                if boundary_records
                else 0.0,
                4,
            ),
            "disclosure": "All musical events and roles are deterministic signal-derived estimates.",
        },
        "global": {
            "tempo_bpm_estimate": _finite_float(tempo, 3),
            "bpm": _finite_float(tempo, 3),
            "beat_period_seconds": _finite_float(60.0 / tempo, 5) if tempo > 0 else None,
            "tempo_confidence": _finite_float(beat_confidence, 4),
            "beat_count": len(beats),
            "downbeat_count": len(downbeat_times),
            "meter_estimate": f"{meter}/4",
            "meter_confidence": _finite_float(meter_confidence, 4),
            "mean_energy": _finite_float(np.mean(energy), 4),
            "dynamic_range_db_5_to_95": _finite_float(
                np.percentile(rms_db, 95) - np.percentile(rms_db, 5), 3
            ),
            "accent_density_per_second": _finite_float(len(accents) / analyzed_duration, 4),
            "onset_density_per_second": _finite_float(overall_onset_density, 4),
            "rhythm_mode": rhythm_mode["mode"],
        },
        "beats": beats,
        "downbeats": [_finite_float(value, 4) for value in downbeat_times],
        "phrases": phrases,
        "section_boundaries": boundary_records,
        "sections": sections,
        "loop_groups": loop_groups,
        "stage_profile": stage_profile,
        "energy_curve": energy_points,
        "energy_curve_sampling": energy_sampling,
        "accents": accents,
        "pauses": {
            "threshold_db_relative_to_peak": _finite_float(pause_threshold_db, 2),
            "intervals": pauses,
        },
        "pause_intervals": pauses,
        "timbre": timbre,
        "vocal": {
            "likelihood": _finite_float(overall_vocal, 4),
            "label": "likely" if overall_vocal >= 0.64 else "possible" if overall_vocal >= 0.48 else "unlikely",
            "likely_intervals": vocal_intervals,
            "method": "heuristic from harmonic/percussive balance, 180-4500 Hz energy, spectral flatness and active energy",
            "limitation": "Instrumental lead melodies may resemble vocals; this is not source separation or speech recognition.",
        },
        "analysis_parameters": {
            "sample_rate": sample_rate,
            "mono": True,
            "n_fft": n_fft,
            "hop_length": hop_length,
            "frame_rate_hz": _finite_float(frame_rate, 4),
            "bars_per_phrase_assumption": bars_per_phrase,
            "onset_raw_median": _finite_float(onset_raw_median, 6),
            "onset_raw_p95": _finite_float(onset_raw_p95, 6),
            "transient_salience": _finite_float(transient_salience, 4),
        },
    }


def analyze_bgm(
    bgm_path: str | os.PathLike[str],
    cache_dir: str | os.PathLike[str],
    output_path: Optional[str | os.PathLike[str]] = None,
    target_duration: Optional[float] = None,
) -> dict[str, Any]:
    """Analyze a BGM and return a JSON-serializable signal-derived profile.

    Args:
        bgm_path: Source audio/video path. The source is opened read-only.
        cache_dir: Directory for content-fingerprint keyed JSON profiles.
        output_path: Optional additional JSON profile destination.
        target_duration: If set, analyze only [0, target_duration] seconds.

    Beat, downbeat, phrase, section, vocal and mood fields are estimates; they
    are calculated from the decoded signal and include confidence/limitation
    metadata where appropriate.
    """
    source = Path(bgm_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"BGM file does not exist: {source}")
    if target_duration is not None:
        target_duration = float(target_duration)
        if not math.isfinite(target_duration) or target_duration <= 0:
            raise ValueError("target_duration must be a positive finite number of seconds.")

    fingerprint = _fingerprint(source)
    key = _cache_key(fingerprint, target_duration)
    cache_root = Path(cache_dir).expanduser().resolve()
    cache_file = cache_root / f"bgm_{key}.json"
    destination = Path(output_path).expanduser().resolve() if output_path is not None else None

    if cache_file.is_file():
        try:
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
            if (
                cached.get("schema_version") == SCHEMA_VERSION
                and (cached.get("analyzer") or {}).get("version") == ANALYZER_VERSION
                and ((cached.get("input") or {}).get("fingerprint") or {}).get("sha256")
                == fingerprint["sha256"]
            ):
                result = copy.deepcopy(cached)
                result["input"]["path"] = str(source)
                result["input"]["fingerprint"] = fingerprint
                result["cache"]["hit"] = True
                result["cache"]["cache_file"] = str(cache_file)
                if destination is not None:
                    _write_json(destination, result)
                return result
        except (OSError, ValueError, json.JSONDecodeError, KeyError, TypeError):
            pass

    try:
        import librosa
        import numpy as np
    except ImportError as error:
        raise RuntimeError(
            "BGM analysis requires numpy, scipy, librosa and soundfile. Install the skill requirements first."
        ) from error

    probe = _probe_audio(source)
    probed_duration = probe.get("duration_seconds")
    decode_duration = None
    if target_duration is not None:
        decode_duration = target_duration
        if probed_duration is not None:
            decode_duration = min(target_duration, float(probed_duration))
    waveform, sample_rate, decode_method = _load_audio(source, decode_duration, librosa)
    if target_duration is not None:
        waveform = waveform[: int(round(target_duration * sample_rate))]
    analyzed_duration = float(len(waveform)) / float(sample_rate)
    if probed_duration is None:
        try:
            probed_duration = float(librosa.get_duration(path=str(source)))
        except Exception:
            probed_duration = analyzed_duration
    original_duration = max(float(probed_duration), analyzed_duration)

    signal_profile = _analyze_signal(
        waveform=waveform,
        sample_rate=sample_rate,
        analyzed_duration=analyzed_duration,
        librosa=librosa,
        np=np,
    )
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "analyzer": {
            "name": "bgm-montage signal analyzer",
            "version": ANALYZER_VERSION,
            "schema_version": SCHEMA_VERSION,
            "analysis_config_version": ANALYSIS_CONFIG_VERSION,
            "deterministic": True,
            "basis": "decoded audio waveform",
            "disclosure": (
                "Tempo, beats, downbeats, phrases, sections, vocal likelihood, moods and edit "
                "guidance are signal-derived estimates, not ground-truth musical annotations."
            ),
        },
        "input": {
            "path": str(source),
            "fingerprint": fingerprint,
            "original_duration_seconds": _finite_float(original_duration, 5),
            "target_duration_seconds": (
                None if target_duration is None else _finite_float(target_duration, 5)
            ),
            "analysis_window": {
                "start": 0.0,
                "end": _finite_float(analyzed_duration, 5),
                "duration": _finite_float(analyzed_duration, 5),
                "trimmed_to_target": bool(
                    target_duration is not None and original_duration > analyzed_duration + 0.01
                ),
                "target_fully_available": bool(
                    target_duration is None or original_duration + 0.01 >= target_duration
                ),
            },
            "technical_probe": {
                "channels": probe.get("channels"),
                "source_sample_rate": probe.get("source_sample_rate"),
                "probe_method": probe.get("probe_method"),
                "decode_method": decode_method,
            },
        },
        "cache": {
            "hit": False,
            "cache_key": key,
            "cache_file": str(cache_file),
            "invalidation": "SHA-256 file content + target_duration + analyzer/schema/config version + sample rate",
        },
        **signal_profile,
        "estimation_notes": [
            "BPM and beats use onset-envelope beat tracking.",
            "Downbeats approximate a 3/4 or 4/4 accent phase; meter confidence should be respected.",
            "Phrases group four estimated bars and also preserve strong section boundaries.",
            "Sections come from MFCC, chroma, energy and onset novelty, aligned to nearby estimated downbeats.",
            "beat_cut is enabled only when confidence, interval stability, grid coverage and pulse evidence pass conservative gates.",
            "Drops, surges, hard stops and climaxes are deterministic energy/onset/density estimates for editing, not musicological labels.",
            "Mood and editing guidance are deterministic heuristics over measured energy, rhythm and timbre.",
        ],
    }
    result["analysis_digest"] = _analysis_digest(result)
    _write_json(cache_file, result)
    if destination is not None:
        _write_json(destination, result)
    return result


def _build_parser() -> argparse.ArgumentParser:
    try:
        from runtime_paths import RuntimePaths

        default_cache = RuntimePaths.build().bgm_cache
    except Exception:
        explicit = os.environ.get("BGM_MONTAGE_PROJECT_ROOT", "").strip()
        if explicit:
            project_root = Path(explicit).expanduser().resolve()
        else:
            skill_root = Path(__file__).resolve().parent.parent
            if skill_root.parent.name == "skills" and skill_root.parent.parent.name == ".agents":
                project_root = skill_root.parent.parent.parent.resolve()
            else:
                project_root = Path.cwd().resolve()
        default_cache = project_root / ".bgm-montage-cache" / "bgm"
    parser = argparse.ArgumentParser(
        description=(
            "Estimate BGM tempo, beats, phrases, sections, energy, timbre, vocals, pauses and accents "
            "from the decoded audio signal."
        )
    )
    parser.add_argument("bgm_path", nargs="?", help="BGM audio/video file")
    parser.add_argument("--bgm", dest="bgm_option", help="BGM audio/video file (alias)")
    parser.add_argument(
        "--cache-dir",
        default=str(default_cache),
        help=f"Fingerprint-cache directory (default: {default_cache})",
    )
    parser.add_argument("--output", help="Optional JSON output path")
    parser.add_argument(
        "--target-duration",
        type=float,
        help="Analyze only the first N seconds; never pads beyond source duration",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Print a concise JSON summary instead of the complete profile",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    bgm = args.bgm_option or args.bgm_path
    if not bgm:
        parser.error("provide a BGM path positionally or with --bgm")
    try:
        result = analyze_bgm(
            bgm_path=bgm,
            cache_dir=args.cache_dir,
            output_path=args.output,
            target_duration=args.target_duration,
        )
    except Exception as error:
        print(f"analyze_bgm: {error}", file=sys.stderr)
        return 1

    if args.summary_only or args.output:
        summary = {
            "output": str(Path(args.output).expanduser().resolve()) if args.output else None,
            "cache_hit": result["cache"]["hit"],
            "cache_file": result["cache"]["cache_file"],
            "analyzed_duration_seconds": result["input"]["analysis_window"]["duration"],
            "tempo_bpm_estimate": result["global"]["tempo_bpm_estimate"],
            "beats": result["global"]["beat_count"],
            "sections": len(result["sections"]),
            "vocal_likelihood": result["vocal"]["likelihood"],
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
