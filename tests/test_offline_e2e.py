from __future__ import annotations

import json
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


SKILL_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = SKILL_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from analyze_bgm import analyze_bgm  # noqa: E402
from analyze_editing_grammar import analyze_editing_grammar  # noqa: E402
from analyze_references import analyze_references  # noqa: E402
from montage import (  # noqa: E402
    build_timeline,
    parse_ratio,
    plan_subject_crop,
    render_timeline,
)
from validate_output import validate_output  # noqa: E402


FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")


def _run(command: list[str], timeout: int = 120) -> subprocess.CompletedProcess[str]:
    process = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
    )
    if process.returncode != 0:
        tail = "\n".join(process.stderr.splitlines()[-30:])
        raise AssertionError(f"Command failed ({process.returncode}): {command[0]}\n{tail}")
    return process


def _make_segmented_click_bgm(path: Path, duration: float = 8.0) -> Path:
    """Create a real click-bearing, four-energy-stage BGM entirely with FFmpeg."""

    path.parent.mkdir(parents=True, exist_ok=True)
    source = (
        "aevalsrc='0.06*sin(2*PI*220*t)"
        "+if(lt(mod(t\\,0.5)\\,0.035)\\,0.8*sin(2*PI*1200*t)\\,0)'"
        f":s=44100:d={duration}"
    )
    # The four two-second stages deliberately have different actual waveform
    # energy.  The 0.5-second impulse supplies stable beat/accent evidence.
    volume = (
        "volume=if(lt(t\\,2)\\,0.15\\,"
        "if(lt(t\\,4)\\,0.35\\,"
        "if(lt(t\\,6)\\,0.85\\,0.30))),alimiter=limit=0.95"
    )
    _run(
        [
            str(FFMPEG),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            source,
            "-af",
            volume,
            "-c:a",
            "pcm_s16le",
            str(path),
        ]
    )
    return path


def _make_visual_asset(
    path: Path,
    *,
    hue: int,
    subject_rect: tuple[int, int, int, int],
    mirror: str = "",
    duration: float = 5.5,
) -> Path:
    """Create an animated HD-shaped fixture with a visible off-centre subject."""

    path.parent.mkdir(parents=True, exist_ok=True)
    x, y, width, height = subject_rect
    filters = [f"hue=H={hue}:s=1.25"]
    if mirror:
        filters.append(mirror)
    filters.append(
        f"drawbox=x={x}:y={y}:w={width}:h={height}:color=white@0.96:t=fill"
    )
    filters.append(
        f"drawbox=x={x + 10}:y={y + 10}:w={max(8, width - 20)}:"
        f"h={max(8, height - 20)}:color=0x202020@0.78:t=4"
    )
    _run(
        [
            str(FFMPEG),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"testsrc2=size=640x360:rate=24:duration={duration}",
            "-vf",
            ",".join(filters),
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ]
    )
    return path


def _make_reference_video(path: Path, assets: list[Path], bgm: Path) -> Path:
    """Join four visibly different two-second shots and the constructed BGM."""

    path.parent.mkdir(parents=True, exist_ok=True)
    command = [str(FFMPEG), "-hide_banner", "-loglevel", "error", "-y"]
    for asset in assets[:4]:
        command.extend(["-i", str(asset)])
    command.extend(["-i", str(bgm)])
    trims = [
        f"[{index}:v]trim=start=0:end=2,setpts=PTS-STARTPTS[v{index}]"
        for index in range(4)
    ]
    trims.append("[v0][v1][v2][v3]concat=n=4:v=1:a=0[vout]")
    command.extend(
        [
            "-filter_complex",
            ";".join(trims),
            "-map",
            "[vout]",
            "-map",
            "4:a:0",
            "-t",
            "8",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "19",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "160k",
            "-movflags",
            "+faststart",
            str(path),
        ]
    )
    _run(command)
    return path


def _fixture_assets(project: Path) -> list[dict[str, object]]:
    specifications = [
        # scene, hue, visible subject rectangle, normalized subject geometry,
        # motion, scale, optional frame transform
        ("nature", 0, (24, 92, 108, 126), [0.03, 0.24, 0.22, 0.62], 0.22, "wide", ""),
        ("architecture", 55, (510, 74, 105, 132), [0.79, 0.20, 0.97, 0.62], 0.38, "medium", "hflip"),
        ("transport", 110, (42, 35, 124, 92), [0.05, 0.08, 0.27, 0.39], 0.72, "detail", "vflip"),
        ("water_coast", 170, (476, 215, 132, 108), [0.73, 0.57, 0.97, 0.94], 0.58, "wide", ""),
        ("industrial", 230, (270, 86, 118, 150), [0.40, 0.20, 0.61, 0.70], 0.46, "medium", "hflip"),
        # A deliberately broad salient region cannot survive a severe 16:9 to
        # 9:16 crop.  The planner must choose blurred containment, not crop it.
        ("abstract", 290, (30, 58, 580, 245), [0.04, 0.12, 0.96, 0.88], 0.80, "detail", "vflip"),
    ]
    assets: list[dict[str, object]] = []
    media_dir = project / "模拟 Pixabay 素材"
    for index, (scene, hue, box, bbox, motion, scale, mirror) in enumerate(specifications, start=1):
        path = _make_visual_asset(
            media_dir / f"素材 {index:02d} {scene}.mp4",
            hue=hue,
            subject_rect=box,
            mirror=mirror,
        )
        center = {
            "x": round((bbox[0] + bbox[2]) / 2.0, 4),
            "y": round((bbox[1] + bbox[3]) / 2.0, 4),
        }
        assets.append(
            {
                "pixabay_id": 9000 + index,
                "local_path": str(path),
                "duration_seconds": 5.5,
                "width": 640,
                "height": 360,
                "score": 0.96 - index * 0.01,
                "motion_score": motion,
                "shot_scale": scale,
                "scene_category": scene,
                "tags": f"{scene} environment no people cinematic",
                "face_content_risk": 0.02,
                "subject_profile": {
                    "center": center,
                    "bbox": bbox,
                    "confidence": 0.94,
                    "center_spread": {"x": 0.015, "y": 0.012},
                },
                "page_url": f"https://pixabay.example.invalid/videos/{9000 + index}/",
                "search_query": f"offline {scene} environment",
            }
        )
    return assets


@pytest.mark.skipif(not FFMPEG or not FFPROBE, reason="offline render test requires ffmpeg and ffprobe")
def test_offline_reference_bgm_grammar_timeline_render_and_full_decode(tmp_path: Path) -> None:
    """Exercise the real v1.2 pipeline without network or external media."""

    project = tmp_path / "中文 空格 项目"
    reference_dir = project / "参考 视频"
    output_dir = project / "输出 成片" / "run_离线_001"
    cache_dir = project / ".bgm-montage-cache"
    bgm = _make_segmented_click_bgm(project / "音乐 素材" / "8秒 分段节拍.wav")
    assets = _fixture_assets(project)
    reference = _make_reference_video(
        reference_dir / "参考 样片 01.mp4",
        [Path(str(item["local_path"])) for item in assets],
        bgm,
    )
    assert reference.is_file()
    reference_sha256_before = hashlib.sha256(reference.read_bytes()).hexdigest()

    bgm_profile_path = cache_dir / "bgm" / "bgm_profile.json"
    bgm_profile = analyze_bgm(
        bgm,
        cache_dir / "bgm",
        output_path=bgm_profile_path,
        target_duration=8.0,
    )
    assert bgm_profile_path.is_file()
    assert bgm_profile["input"]["analysis_window"]["duration"] == pytest.approx(8.0, abs=0.05)
    assert bgm_profile["global"]["beat_count"] >= 6
    assert len(bgm_profile["energy_curve"]) >= 8
    assert len(bgm_profile["accents"]) >= 4
    assert len(bgm_profile["sections"]) >= 1

    style_path = cache_dir / "references" / "style_profile.json"
    style_profile = analyze_references(
        reference_dir,
        cache_dir / "references",
        output_path=style_path,
        enable_semantics=False,
    )
    assert style_path.is_file()
    assert style_profile["schema_version"] == "1.2"
    assert style_profile["corpus"]["video_count"] == 1
    assert style_profile["run_report"]["analyzed"] == 1
    assert style_profile["source_policy"] == "read_only"
    reference_summary = style_profile["videos"][0]
    semantic = reference_summary["semantic_analysis"]
    assert semantic["available"] is False
    assert semantic["backend"] == "disabled"
    assert isinstance(reference_summary["subject_profile"], dict)
    assert reference_summary["shots"]
    # Even in the explicitly documented no-model fallback, the machine schema
    # must retain every semantic/editing field.  Values may honestly be
    # "unresolved"; their existence must never be fabricated away.
    for shot in reference_summary["shots"]:
        assert {
            "subject",
            "scene",
            "apparent_action",
            "shot_scale",
            "composition",
            "camera_motion",
            "visual_mood",
            "search_keywords",
            "subject_region",
        } <= set(shot)

    # A second pass proves the actual fingerprint cache is reusable with a
    # Unicode/space-bearing Windows path.
    cached_style = analyze_references(
        reference_dir,
        cache_dir / "references",
        output_path=style_path,
        enable_semantics=False,
    )
    assert cached_style["run_report"]["analyzed"] == 0
    assert cached_style["run_report"]["reused"] == 1

    grammar_path = cache_dir / "references" / "editing_grammar.json"
    editing_grammar = analyze_editing_grammar(
        reference_dir,
        style_profile,
        cache_dir / "references",
        output_path=grammar_path,
    )
    assert grammar_path.is_file()
    assert editing_grammar["schema_version"] == "1.1"
    assert editing_grammar["corpus"]["audio_analyzed_video_count"] == 1
    assert editing_grammar["cut_alignment"]["total_reference_cuts"] >= 2
    assert editing_grammar["montage_policy"]["event_weights"]
    assert editing_grammar["montage_policy"]["shot_duration_by_energy"]
    assert "scale_transition_matrix" in editing_grammar["visual_transition_grammar"]

    content_policy = {
        "min_unique_assets": 4,
        "max_reuse_per_asset": 2,
        "max_asset_screen_share": 0.30,
        "min_scene_categories": 4,
        "max_prominent_face_screen_share": 0.10,
        "prominent_face_threshold": 0.65,
    }
    plan = build_timeline(
        bgm_profile,
        {"selected": assets},
        duration=8.0,
        style_profile=style_profile,
        editing_grammar=editing_grammar,
        content_policy=content_policy,
        seed="offline-e2e-v1.2",
        ratio="360:640",
    )
    assert plan["editing_grammar_applied"] is True
    assert plan["editing_grammar_digest"]
    assert plan["sufficiency"]["passed"] is True
    assert plan["sufficiency"]["used_unique_assets"] >= 4
    assert len(plan["sufficiency"]["used_scene_categories"]) >= 4
    assert max(plan["asset_usage_counts"].values()) <= 2
    assert plan["sufficiency"]["max_asset_screen_share_actual"] <= 0.305
    assert plan["sufficiency"]["prominent_face_screen_share_actual"] <= 0.105
    assert all(shot["grammar_influence"] for shot in plan["shots"])

    crop_modes = {shot["crop_plan"]["mode"] for shot in plan["shots"]}
    assert crop_modes <= {"subject_crop", "blur_fill", "fit"}
    assert "subject_crop" in crop_modes
    for shot in plan["shots"]:
        crop = shot["crop_plan"]
        assert crop["retention"] >= 0.85
        if crop["mode"] == "subject_crop":
            left, top, right, bottom = crop["crop_rect_norm"]
            assert 0.0 <= left < right <= 1.0
            assert 0.0 <= top < bottom <= 1.0

    unsafe_asset = assets[-1]
    unsafe_crop = plan_subject_crop(unsafe_asset, parse_ratio("360:640"))
    assert unsafe_crop["mode"] == "blur_fill"
    assert unsafe_crop["reason"] == "unsafe subject retention"

    output_dir.mkdir(parents=True, exist_ok=False)
    plan_path = output_dir / "edit_plan.json"
    plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    movie = render_timeline(
        plan,
        bgm,
        output_dir / "离线 测试成片.mp4",
        ratio="360:640",
        style_profile=style_profile,
        ffmpeg=str(FFMPEG),
        fps=24,
        overwrite=False,
    )
    assert movie.is_file() and movie.stat().st_size > 20_000

    report_path = output_dir / "validation_report.json"
    report = validate_output(
        movie,
        expected_duration=8.0,
        expected_ratio="360:640",
        report_path=report_path,
        frames_dir=output_dir / "验收 帧",
        edit_plan=plan,
        ffmpeg=str(FFMPEG),
        ffprobe=str(FFPROBE),
    )
    assert report_path.is_file()
    assert report["passed"], json.dumps(report, ensure_ascii=False, indent=2)
    assert report["video"]["codec"] == "h264"
    assert report["video"]["width"] == 360
    assert report["video"]["height"] == 640
    assert report["audio"]["codec"] == "aac"
    assert report["duration_seconds"] == pytest.approx(8.0, abs=0.35)
    assert report["checks"]["full_decode"] is True
    assert report["checks"]["no_long_black"] is True
    assert report["checks"]["no_long_freeze"] is True
    assert report["checks"]["audio_not_entirely_silent"] is True
    assert report["checks"]["material_repetition"] is True
    assert report["checks"]["subject_crop_safe"] is True
    assert report["checks"]["representative_frames"] is True
    metrics = report["edit_plan_metrics"]
    assert metrics["unique_asset_count"] >= 4
    assert metrics["max_reuse_count"] <= 2
    assert metrics["max_asset_screen_share"] <= 0.306
    assert metrics["unsafe_crop_count"] == 0
    assert len(report["representative_frames"]) == 3
    assert hashlib.sha256(reference.read_bytes()).hexdigest() == reference_sha256_before
