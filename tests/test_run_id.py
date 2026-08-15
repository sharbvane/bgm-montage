from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pytest


SKILL_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = SKILL_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import bgm_montage as entry  # noqa: E402


def _args(root: Path, run_id: str | None = None) -> argparse.Namespace:
    return argparse.Namespace(
        bgm=str(root / "music.wav"),
        theme="测试主题",
        duration=8.0,
        ratio="16:9",
        output_dir=str(root / "output"),
        project_name="run-id-test",
        run_id=run_id,
        reference_dir=str(root / "references"),
        material_dir=str(root / "materials"),
        cache_dir=str(root / "cache"),
        assets=4,
        min_width=1280,
        min_height=720,
        allow_semantic_fallback=True,
    )


def _stub_pipeline(monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
    monkeypatch.setattr(entry, "PROJECT_ROOT", root)
    monkeypatch.setenv("PIXABAY_API_KEY", "fixture-key-never-written")
    monkeypatch.setattr(entry, "analyze_references", lambda *args, **kwargs: {"style_profile": {}, "run_report": {"analyzed": 1, "reused": 0, "failed": 0}})
    monkeypatch.setattr(entry, "analyze_editing_grammar", lambda *args, **kwargs: {"status": "ok", "run_report": {"analyzed": 1, "reused": 0}, "reliability": {"score": 1.0}})
    def analyze_bgm_fixture(
        _bgm: Path, _cache: Path, output_path: Path, _duration: float
    ) -> dict[str, object]:
        result: dict[str, object] = {"audio_profile": {"duration_seconds": 8.0}}
        entry._write_json(Path(output_path), result)
        return result

    monkeypatch.setattr(entry, "analyze_bgm", analyze_bgm_fixture)
    selected = []
    for index in range(4):
        path = root / "materials" / f"asset-{index}.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")
        selected.append({"pixabay_id": index, "local_path": str(path), "duration_seconds": 8.0})
    monkeypatch.setattr(entry, "run_pixabay_pipeline", lambda *args, **kwargs: {"selected": selected, "search_rounds": [], "rejections": []})
    monkeypatch.setattr(entry, "run_youtube_first_pipeline", lambda *args, **kwargs: {"selected": selected, "search_rounds": [], "rejections": [], "candidate_count": 24, "candidate_pool_gate": {"passed": True}})
    plan = {"duration_seconds": 8.0, "shots": []}
    monkeypatch.setattr(entry, "build_timeline", lambda *args, **kwargs: plan)
    monkeypatch.setattr(entry, "write_plan", lambda value, path: entry._write_json(Path(path), value))

    def render(_plan: dict, _bgm: Path, output: Path, *_args: object, **_kwargs: object) -> Path:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"rendered-fixture")
        return output

    monkeypatch.setattr(entry, "render_timeline", render)
    monkeypatch.setattr(entry, "update_usage_intervals", lambda *args, **kwargs: {})
    monkeypatch.setattr(entry, "validate_output", lambda *args, **kwargs: {"passed": True, "checks": {"fixture": True}})


def test_default_run_id_is_unique_and_history_is_not_overwritten(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "references").mkdir()
    (tmp_path / "music.wav").write_bytes(b"audio")
    _stub_pipeline(monkeypatch, tmp_path)

    first = entry.run(_args(tmp_path))
    second = entry.run(_args(tmp_path))

    assert first["run_id"] != second["run_id"]
    first_dir = Path(first["artifacts"]["run_report"]).parent
    second_dir = Path(second["artifacts"]["run_report"]).parent
    assert first_dir != second_dir
    assert first_dir.is_dir() and second_dir.is_dir()


def test_explicit_existing_run_id_fails_before_overwrite(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "references").mkdir()
    (tmp_path / "music.wav").write_bytes(b"audio")
    _stub_pipeline(monkeypatch, tmp_path)
    args = _args(tmp_path, run_id="fixed-run")

    entry.run(args)
    with pytest.raises(FileExistsError, match="not overwritten"):
        entry.run(args)


def test_failed_run_resumes_from_existing_stage_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "references").mkdir()
    (tmp_path / "music.wav").write_bytes(b"audio")
    _stub_pipeline(monkeypatch, tmp_path)
    args = _args(tmp_path, run_id="resume-run")
    working_youtube_first = entry.run_youtube_first_pipeline

    def fail_once(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise RuntimeError("intentional stage interruption")

    monkeypatch.setattr(entry, "run_youtube_first_pipeline", fail_once)
    with pytest.raises(RuntimeError, match="intentional stage interruption"):
        entry.run(args)

    run_dir = tmp_path / "output" / "run-id-test" / "resume-run"
    assert (run_dir / "audiomap.json").is_file()
    assert not (run_dir / "run-id-test_montage.mp4").exists()

    monkeypatch.setattr(entry, "run_youtube_first_pipeline", working_youtube_first)
    args.resume_run = True
    report = entry.run(args)

    assert report["passed"] is True
    assert report["resumed"] is True
    assert report["stages"]["references"]["status"] == "resumed"
    assert report["stages"]["bgm"]["status"] == "resumed"
    assert Path(report["artifacts"]["video"]).is_file()


def test_legacy_cli_names_and_primary_invocation_remain_compatible() -> None:
    parser = entry.build_parser()
    parsed = parser.parse_args(
        [
            "--bgm", "music.mp3",
            "--theme", "city night",
            "--duration", "12.5",
            "--ratio", "16:9",
            "--output-dir", "output",
            "--max-source-reuse", "2",
            "--max-source-share", "0.25",
        ]
    )

    assert parsed.bgm == "music.mp3"
    assert parsed.theme == "city night"
    assert parsed.duration == pytest.approx(12.5)
    assert parsed.ratio == "16:9"
    assert parsed.max_reuse_per_asset == 2
    assert parsed.max_asset_screen_share == pytest.approx(0.25)


def test_passed_attempt_review_artifacts_are_promoted_with_stable_paths(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    attempt_dir = run_dir / "attempts" / "attempt_01"
    frames = attempt_dir / "validation_frames"
    frames.mkdir(parents=True)
    frame = frames / "event.jpg"
    frame.write_bytes(b"jpeg")
    review_json = attempt_dir / "visual_review.json"
    review_json.write_text(
        json.dumps({"artifacts": {"json": str(review_json)}, "entries": [{"frame_path": str(frame)}]}),
        encoding="utf-8",
    )
    (attempt_dir / "visual_review.md").write_text("![frame](<validation_frames/event.jpg>)\n", encoding="utf-8")
    final_output = run_dir / "final.mp4"
    final_output.write_bytes(b"video")

    promoted = entry._promote_validation_artifacts(
        {"path": str(attempt_dir / "attempt.mp4"), "event_frames": [{"path": str(frame)}]},
        attempt_dir,
        run_dir,
        final_output,
    )

    assert promoted["path"] == str(final_output.resolve())
    assert promoted["event_frames"][0]["path"] == str(run_dir.resolve() / "validation_frames" / "event.jpg")
    payload = json.loads((run_dir / "visual_review.json").read_text(encoding="utf-8"))
    assert payload["entries"][0]["frame_path"] == str(run_dir.resolve() / "validation_frames" / "event.jpg")
    assert (run_dir / "visual_review.md").is_file()


def _write_agent_review(request_path: Path, status: str) -> Path:
    request = json.loads(request_path.read_text(encoding="utf-8"))
    observations = [
        {
            "entry_index": target["entry_index"],
            "timestamp_seconds": target["timestamp_seconds"],
            "status": status,
            "reason": "Visible discontinuity requires another edit" if status == "fail" else "Frame is visually coherent",
            "confidence": 0.91,
        }
        for target in request["review_targets"]
    ]
    findings = (
        [{
            "status": "fail",
            "timestamp_seconds": observations[0]["timestamp_seconds"],
            "reason": "Visible discontinuity requires another edit",
            "confidence": 0.91,
            "suggested_action": "rerender",
            "shot_indices": [],
        }]
        if status == "fail"
        else []
    )
    result_path = Path(request["result_json"])
    result_path.write_text(
        json.dumps({
            "schema_version": "1.4.1",
            "artifact_type": "agent_visual_review",
            "attempt": request["attempt"],
            "media_sha256": request["media_sha256"],
            "reviewer": {"type": "visual_agent", "model": "test-vision-model", "reviewed_with_images": True},
            "status": status,
            "summary": "Retry required" if status == "fail" else "Visual review passed",
            "confidence": 0.91,
            "observations": observations,
            "findings": findings,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result_path


def test_agent_visual_review_failure_triggers_existing_rework_loop_then_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "references").mkdir()
    (tmp_path / "music.wav").write_bytes(b"audio")
    _stub_pipeline(monkeypatch, tmp_path)
    render_calls: list[str] = []

    def render(_plan: dict, _bgm: Path, output: Path, *_args: object, **_kwargs: object) -> Path:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(f"render-{output.parent.name}".encode())
        render_calls.append(output.parent.name)
        return output

    def validate(media: Path, *args: object, **kwargs: object) -> dict:
        media = Path(media)
        frames = Path(kwargs["frames_dir"])
        frames.mkdir(parents=True, exist_ok=True)
        frame = frames / "event.jpg"
        frame.write_bytes(b"jpeg")
        attempt_dir = frames.parent
        review_path = attempt_dir / "visual_review.json"
        digest = hashlib.sha256(media.read_bytes()).hexdigest()
        review_path.write_text(json.dumps({
            "schema_version": "1.4.1",
            "artifact_type": "visual_review",
            "media_sha256": digest,
            "entries": [{
                "index": 1,
                "time_seconds": 1.0,
                "frame_path": str(frame.resolve()),
                "event_types": ["climaxes"],
                "details": [{"type": "music_event", "event": "climaxes", "shot_index": 0}],
            }],
            "planned_cut_pairs": [],
        }), encoding="utf-8")
        (attempt_dir / "visual_review.md").write_text("![frame](<validation_frames/event.jpg>)\n", encoding="utf-8")
        return {
            "passed": True,
            "checks": {"programmatic_fixture": True},
            "sha256": digest,
            "path": str(media),
            "visual_review": {"json": str(review_path), "markdown": str(attempt_dir / "visual_review.md")},
        }

    monkeypatch.setattr(entry, "render_timeline", render)
    monkeypatch.setattr(entry, "validate_output", validate)
    args = _args(tmp_path, run_id="agent-loop")
    args.agent_visual_review = "required"

    pending_first = entry.run(args)
    run_dir = Path(pending_first["artifacts"]["run_report"]).parent
    assert pending_first["status"] == "awaiting_agent_visual_review"
    _write_agent_review(run_dir / "attempts" / "attempt_01" / "agent_visual_review_request.json", "fail")

    args.resume_run = True
    pending_second = entry.run(args)
    assert pending_second["status"] == "awaiting_agent_visual_review"
    assert [item["status"] for item in pending_second["stages"]["render_attempts"]] == [
        "failed_agent_visual_review",
        "awaiting_agent_visual_review",
    ]
    _write_agent_review(run_dir / "attempts" / "attempt_02" / "agent_visual_review_request.json", "pass")

    completed = entry.run(args)
    assert completed["passed"] is True
    assert render_calls == ["attempt_01", "attempt_02"]
    final_report = json.loads(Path(completed["artifacts"]["render_report"]).read_text(encoding="utf-8"))
    assert final_report["checks"]["agent_visual_review"] is True
    assert final_report["agent_visual_review"]["status"] == "pass"
    assert final_report["rework"]["automatic_rework_count"] == 1
    assert Path(completed["artifacts"]["agent_visual_review_result"]).is_file()
