from __future__ import annotations

import importlib.util
import sys
import zipfile
from pathlib import Path, PurePosixPath

import pytest


SKILL_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = SKILL_ROOT / "scripts" / "package_skill.py"
SPEC = importlib.util.spec_from_file_location("bgm_montage_package_skill", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
package_skill = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = package_skill
SPEC.loader.exec_module(package_skill)


def _write(path: Path, text: str = "fixture\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def _make_fixture_skill(base: Path) -> Path:
    root = base / ".agents" / "skills" / "bgm-montage"
    _write(root / ".env.example", "PIXABAY_API_KEY=your_pixabay_api_key_here\n")
    _write(
        root / "SKILL.md",
        "---\nname: bgm-montage\ndescription: Test fixture.\n---\n\n# BGM Montage v1.3\n",
    )
    _write(root / "requirements.txt", "requests>=2,<3\n")
    _write(root / "requirements.lock.txt", "requests==2.34.2\n")
    _write(root / "requirements-jianying.lock.txt", "# optional JianYing adapter lock\n")
    _write(root / "CHANGELOG.md", "# Changelog\n\n## v1.3\n")
    _write(root / "TEST_REPORT.md", "# Test Report\n")
    _write(root / "agents" / "openai.yaml", 'interface:\n  display_name: "BGM Montage"\n')
    _write(root / "references" / "usage.md", "# Usage\n")
    for name in package_skill.REQUIRED_SCRIPTS:
        if name == "package_skill.py":
            (root / "scripts").mkdir(parents=True, exist_ok=True)
            (root / "scripts" / name).write_bytes(MODULE_PATH.read_bytes())
        else:
            _write(root / "scripts" / name, "from __future__ import annotations\n")
    _write(root / "tests" / "test_smoke.py", "def test_smoke():\n    assert True\n")
    return root


def test_windows_unicode_paths_and_portable_allowlisted_zip(tmp_path: Path) -> None:
    root = _make_fixture_skill(tmp_path / "含 空格的造球项目")

    # These are deliberately present but must never enter the strict allowlist.
    _write(root / ".env", "PIXABAY_API_KEY=do-not-package-this-real-looking-value\n")
    _write(root / ".venv" / "Lib" / "site-packages" / "secret.txt")
    _write(root / ".bgm-montage-cache" / "pixabay" / "search.json")
    _write(root / "scripts" / "__pycache__" / "tool.pyc")
    _write(root / "references" / "debug.log")
    (root / "tests" / "render.mp4").write_bytes(b"not media")

    destination = tmp_path / "发布 包" / "bgm-montage-v1.3.zip"
    report = package_skill.build_package(root, destination)
    assert Path(report["output"]) == destination.resolve()
    assert report["path_separator"] == "/"
    assert report["sensitive_files_included"] is False

    with zipfile.ZipFile(destination, "r") as archive:
        names = archive.namelist()
        assert archive.testzip() is None
        assert names == sorted(names)
        assert all("\\" not in name for name in names)
        assert all(name.startswith(".agents/skills/bgm-montage/") for name in names)
        assert ".agents/skills/bgm-montage/.env.example" in names
        assert ".agents/skills/bgm-montage/requirements.lock.txt" in names
        assert ".agents/skills/bgm-montage/requirements-jianying.lock.txt" in names
        assert ".agents/skills/bgm-montage/scripts/timeline_planner.py" in names
        assert ".agents/skills/bgm-montage/tests/test_smoke.py" in names
        assert not any(name.endswith("/.env") or "/.venv/" in name for name in names)
        assert not any(
            {part.casefold() for part in PurePosixPath(name).parts}
            & {part.casefold() for part in package_skill.FORBIDDEN_PARTS}
            for name in names
        )
        assert not any(name.endswith((".mp4", ".pyc", ".log")) for name in names)

        extracted = tmp_path / "解压 目标"
        archive.extractall(extracted)
    installed = extracted / ".agents" / "skills" / "bgm-montage"
    assert (installed / "SKILL.md").is_file()
    assert (installed / "scripts" / "bgm_montage.py").is_file()


def test_package_rejects_secret_embedded_in_allowlisted_source(tmp_path: Path) -> None:
    root = _make_fixture_skill(tmp_path / "项目")
    _write(root / "scripts" / "leak.py", "PIXABAY_API_KEY=12345678-secret-material\n")
    with pytest.raises(package_skill.PackagingError, match="Non-placeholder"):
        package_skill.build_package(root, tmp_path / "bad.zip")


def test_package_requires_dependency_lock(tmp_path: Path) -> None:
    root = _make_fixture_skill(tmp_path / "项目")
    (root / "requirements.lock.txt").unlink()
    with pytest.raises(package_skill.PackagingError, match="requirements.lock.txt"):
        package_skill.build_package(root, tmp_path / "missing-lock.zip")


def test_package_requires_preserved_timeline_planner(tmp_path: Path) -> None:
    root = _make_fixture_skill(tmp_path / "项目")
    (root / "scripts" / "timeline_planner.py").unlink()
    with pytest.raises(package_skill.PackagingError, match="timeline_planner.py"):
        package_skill.build_package(root, tmp_path / "missing-timeline.zip")


def test_required_symlink_is_rejected_before_zip_collection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _make_fixture_skill(tmp_path / "项目")
    required = (root / "SKILL.md").resolve()
    original = package_skill.Path.is_symlink

    def pretend_required_is_link(self: Path) -> bool:
        return self.resolve() == required or original(self)

    monkeypatch.setattr(package_skill.Path, "is_symlink", pretend_required_is_link)
    with pytest.raises(package_skill.PackagingError, match="SKILL.md"):
        package_skill.build_package(root, tmp_path / "symlink.zip")


def test_package_refuses_silent_overwrite(tmp_path: Path) -> None:
    root = _make_fixture_skill(tmp_path / "项目")
    destination = tmp_path / "release.zip"
    package_skill.build_package(root, destination)
    original = destination.read_bytes()
    with pytest.raises(FileExistsError, match="--force"):
        package_skill.build_package(root, destination)
    assert destination.read_bytes() == original
    package_skill.build_package(root, destination, force=True)
    assert destination.is_file()
