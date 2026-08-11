from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf


SKILL_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = SKILL_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from analyze_bgm import _build_parser, analyze_bgm  # noqa: E402


def _click_signal(duration: float, sample_rate: int = 22_050, *, stop_gap: bool = False) -> np.ndarray:
    sample_count = int(round(duration * sample_rate))
    times = np.arange(sample_count, dtype=np.float64) / sample_rate
    signal = 0.035 * np.sin(2.0 * np.pi * 180.0 * times)
    for beat_index, beat_time in enumerate(np.arange(0.50, duration - 0.10, 0.50)):
        if stop_gap and 3.0 <= beat_time < 4.0:
            continue
        start = int(round(beat_time * sample_rate))
        length = min(int(0.055 * sample_rate), sample_count - start)
        if length <= 0:
            continue
        envelope = np.exp(-np.arange(length) / max(1.0, 0.010 * sample_rate))
        amplitude = 0.95 if beat_index % 4 == 0 else 0.62
        if stop_gap and beat_time >= 4.0:
            amplitude = min(1.0, amplitude * 1.28)
        click = amplitude * envelope * np.sin(2.0 * np.pi * 1250.0 * np.arange(length) / sample_rate)
        signal[start : start + length] += click
    if stop_gap:
        signal[int(3.0 * sample_rate) : int(4.0 * sample_rate)] = 0.0
        signal[int(4.0 * sample_rate) : int(7.2 * sample_rate)] *= 1.22
        signal[int(7.2 * sample_rate) :] = 0.0
    return np.asarray(np.clip(signal, -0.98, 0.98), dtype=np.float32)


def _ambient_signal(duration: float, sample_rate: int = 22_050) -> np.ndarray:
    times = np.arange(int(round(duration * sample_rate)), dtype=np.float64) / sample_rate
    envelope = 0.55 + 0.30 * np.sin(2.0 * np.pi * 0.035 * times)
    fade = np.minimum(1.0, np.minimum(times / 1.8, (duration - times) / 1.8))
    signal = (
        0.10 * np.sin(2.0 * np.pi * 196.0 * times)
        + 0.055 * np.sin(2.0 * np.pi * 293.66 * times + 0.3)
        + 0.025 * np.sin(2.0 * np.pi * 392.0 * times + 0.8)
    )
    return np.asarray(signal * envelope * np.clip(fade, 0.0, 1.0), dtype=np.float32)


def _write(path: Path, signal: np.ndarray, sample_rate: int = 22_050) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, signal, sample_rate, subtype="PCM_16")
    return path


def test_audiomap_v12_is_deterministic_beat_cut_and_legacy_compatible(tmp_path: Path) -> None:
    bgm = _write(tmp_path / "中文 音频" / "强节奏 测试.wav", _click_signal(12.0))
    cache = tmp_path / "缓存 空格" / "bgm"
    first = analyze_bgm(bgm, cache, target_duration=12.0)
    second = analyze_bgm(bgm, cache, target_duration=12.0)

    assert first["schema_version"] == "1.2"
    assert first["artifact_type"] == "audiomap"
    assert first["duration_seconds"] == pytest.approx(12.0, abs=0.03)
    assert first["rhythm_mode"]["mode"] == "beat_cut", first["rhythm_mode"]
    assert 105.0 <= first["tempo"]["bpm"] <= 135.0
    assert first["analysis_digest"] == second["analysis_digest"]
    assert first["cache"]["hit"] is False
    assert second["cache"]["hit"] is True

    canonical_keys = {
        "tempo",
        "rhythm_mode",
        "events",
        "curves",
        "intervals",
        "onsets",
        "density_curve",
        "silence_intervals",
        "low_energy_intervals",
        "hard_stops",
        "drops",
        "surges",
        "climaxes",
        "key_moments",
        "editing_guidance",
        "reliability",
    }
    assert canonical_keys <= set(first)
    # v1.1 consumers keep their exact top-level entry points.
    assert {"global", "beats", "downbeats", "phrases", "sections", "energy_curve", "accents", "pauses", "pause_intervals"} <= set(first)
    assert len(first["events"]["beats"]) >= 12
    assert len(first["events"]["onsets"]) >= 10
    assert all(section["role"] in {"intro", "build", "drop", "break", "climax", "outro"} for section in first["sections"])
    assert all(section["rhythm_mode"] in {"beat_cut", "phrase_flow"} for section in first["sections"])
    assert all(0.0 <= section["edit_guidance"]["cut_intensity"] <= 1.0 for section in first["sections"])


def test_ambient_music_uses_phrase_flow_instead_of_mechanical_beat_cut(tmp_path: Path) -> None:
    bgm = _write(tmp_path / "氛围 音乐" / "舒缓长音.wav", _ambient_signal(12.0))
    profile = analyze_bgm(bgm, tmp_path / "cache" / "bgm", target_duration=12.0)

    assert profile["rhythm_mode"]["mode"] == "phrase_flow", profile["rhythm_mode"]
    assert profile["editing_guidance"]["rhythm_mode"] == "phrase_flow"
    assert "do not cut every beat" in profile["editing_guidance"]["strategy"]
    assert all(section["rhythm_mode"] == "phrase_flow" for section in profile["sections"])


def test_signal_events_include_silence_low_energy_hard_stop_drop_and_climax(tmp_path: Path) -> None:
    bgm = _write(tmp_path / "事件 测试.wav", _click_signal(9.0, stop_gap=True))
    profile = analyze_bgm(bgm, tmp_path / "event-cache", target_duration=9.0)

    assert profile["silence_intervals"]
    assert profile["low_energy_intervals"]
    assert profile["hard_stops"], profile["key_moments"]
    assert profile["drops"] or profile["surges"], profile["key_moments"]
    assert profile["climaxes"], profile["key_moments"]
    types = {item["type"] for item in profile["key_moments"]}
    assert "hard_stop" in types
    assert "climax" in types
    assert all(math.isfinite(float(item["time"])) for item in profile["key_moments"])


def test_standalone_default_cache_matches_project_runtime_namespace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("BGM_MONTAGE_PROJECT_ROOT", str(tmp_path))
    parser = _build_parser()
    parsed = parser.parse_args([str(tmp_path / "unused.wav"), "--summary-only"])
    assert Path(parsed.cache_dir) == (tmp_path / ".bgm-montage-cache" / "bgm").resolve()
