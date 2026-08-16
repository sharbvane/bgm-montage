#!/usr/bin/env python3
"""Build runtime or development bgm-montage ZIPs from strict allowlists."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Iterable, Sequence


SKILL_NAME = "bgm-montage"
ARCHIVE_PREFIX = PurePosixPath(".agents") / "skills" / SKILL_NAME

RUNTIME_ROOT_FILES = (
    ".env.example",
    "SKILL.md",
    "requirements.txt",
    "requirements.lock.txt",
    "requirements-jianying.lock.txt",
)
DEVELOPMENT_ROOT_FILES = ("CHANGELOG.md", "TEST_REPORT.md")
OPTIONAL_ROOT_FILES = (
    "requirements-dev.txt",
    "requirements-dev.lock.txt",
    "pytest.ini",
    "pyproject.toml",
)
RUNTIME_SCRIPTS = (
    "analyze_bgm.py",
    "analyze_editing_grammar.py",
    "analyze_references.py",
    "bgm_montage.py",
    "edit_schema.py",
    "jianying_export.py",
    "local_library.py",
    "montage.py",
    "pixabay_pipeline.py",
    "runtime_paths.py",
    "timeline_planner.py",
    "validate_output.py",
    "visual_intelligence.py",
    "visual_semantics.py",
    "material_usage_policy.py",
    "youtube_pipeline.py",
    "youtube_first_pipeline.py",
)
DEVELOPMENT_SCRIPTS = ("package_skill.py",)
REQUIRED_EXACT_FILES = (
    Path("agents/openai.yaml"),
    Path("references/usage.md"),
)

FORBIDDEN_PARTS = {
    ".bgm-montage-cache",
    ".cache",
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "cache",
    "caches",
    "downloads",
    "output",
    "outputs",
    "test-output",
    "test_output",
    "venv",
    "成片",
    "视频素材",
}
FORBIDDEN_SUFFIXES = {
    ".7z",
    ".avi",
    ".flac",
    ".jpeg",
    ".jpg",
    ".log",
    ".m4a",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp3",
    ".mp4",
    ".part",
    ".png",
    ".pyc",
    ".pyo",
    ".tmp",
    ".wav",
    ".webm",
    ".zip",
}
PLACEHOLDER_VALUES = {
    "change_me",
    "placeholder",
    "replace_me",
    "your_api_key",
    "your_pixabay_api_key",
    "your_pixabay_api_key_here",
}
KEY_ASSIGNMENT = re.compile(r"(?im)^\s*PIXABAY_API_KEY\s*=\s*([^\s#]+)")


class PackagingError(RuntimeError):
    """Raised when the release package would be incomplete or unsafe."""


def _is_forbidden(relative: Path) -> bool:
    parts = {part.casefold() for part in relative.parts}
    if parts & {part.casefold() for part in FORBIDDEN_PARTS}:
        return True
    if relative.name.casefold() == ".env":
        return True
    return relative.suffix.casefold() in FORBIDDEN_SUFFIXES


def _safe_regular_file(path: Path, skill_root: Path) -> Path | None:
    """Return a relative path only for an allowed, non-link regular file."""

    if path.is_symlink() or not path.is_file():
        return None
    relative = path.relative_to(skill_root)
    return None if _is_forbidden(relative) else relative


def collect_allowlisted_files(
    skill_root: str | os.PathLike[str], profile: str = "runtime"
) -> list[Path]:
    """Return deterministic paths for one explicit release profile."""

    root = Path(skill_root).expanduser().resolve(strict=True)
    if not root.is_dir() or root.name != SKILL_NAME:
        raise PackagingError(f"Skill root must be a directory named {SKILL_NAME}: {root}")
    if profile not in {"runtime", "development"}:
        raise PackagingError(f"Unknown package profile: {profile}")

    required_relatives = [
        *map(Path, RUNTIME_ROOT_FILES),
        *REQUIRED_EXACT_FILES,
        *(Path("scripts") / name for name in RUNTIME_SCRIPTS),
    ]
    if profile == "development":
        required_relatives.extend(map(Path, DEVELOPMENT_ROOT_FILES))
        required_relatives.extend(Path("scripts") / name for name in DEVELOPMENT_SCRIPTS)
    missing: list[str] = []
    for relative in required_relatives:
        required_path = root / relative
        if not required_path.is_file() or required_path.is_symlink():
            missing.append(relative.as_posix())
    if missing:
        raise PackagingError("Required package files are missing: " + ", ".join(sorted(missing)))

    selected: dict[str, Path] = {}

    def admit(path: Path) -> None:
        relative = _safe_regular_file(path, root)
        if relative is not None:
            selected[relative.as_posix()] = relative

    root_files = RUNTIME_ROOT_FILES + (
        DEVELOPMENT_ROOT_FILES + OPTIONAL_ROOT_FILES if profile == "development" else ()
    )
    for name in root_files:
        path = root / name
        if path.exists():
            admit(path)

    admit(root / "agents" / "openai.yaml")
    admit(root / "references" / "usage.md")
    for name in RUNTIME_SCRIPTS + (DEVELOPMENT_SCRIPTS if profile == "development" else ()):
        admit(root / "scripts" / name)
    if profile == "development":
        tests_root = root / "tests"
        if tests_root.is_dir():
            for path in sorted(tests_root.rglob("*")):
                if path.suffix.casefold() in {".py", ".json", ".txt", ".yaml", ".yml"}:
                    admit(path)
        if not any(relative.parts and relative.parts[0] == "tests" for relative in selected.values()):
            raise PackagingError("Development package requires at least one automated test under tests/")
    selected_names = set(selected)
    omitted_required = sorted(
        relative.as_posix()
        for relative in required_relatives
        if relative.as_posix() not in selected_names
    )
    if omitted_required:
        raise PackagingError(
            "Required package files were not admitted to the ZIP allowlist: "
            + ", ".join(omitted_required)
        )
    return [selected[key] for key in sorted(selected)]


def _project_root_for(skill_root: Path) -> Path:
    """Resolve the containing project for a canonical .agents/skills Skill."""

    if skill_root.parent.name == "skills" and skill_root.parent.parent.name == ".agents":
        return skill_root.parent.parent.parent.resolve()
    return Path.cwd().resolve()


def _configured_secrets(skill_root: Path, extra: Iterable[str] = ()) -> list[str]:
    """Read secret values for scanning without returning or logging their content."""

    values = [value.strip() for value in extra if value and value.strip()]
    environment_value = os.environ.get("PIXABAY_API_KEY", "").strip()
    if environment_value:
        values.append(environment_value)
    env_path = _project_root_for(skill_root) / ".env"
    try:
        env_text = env_path.read_text(encoding="utf-8-sig")
    except OSError:
        env_text = ""
    match = KEY_ASSIGNMENT.search(env_text)
    if match and match.group(1).strip():
        values.append(match.group(1).strip().strip("'\""))
    return sorted(set(values), key=len, reverse=True)


def _validate_source_contents(skill_root: Path, relatives: Sequence[Path], secrets: Sequence[str]) -> None:
    for relative in relatives:
        path = skill_root / relative
        try:
            text = path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError as exc:
            raise PackagingError(f"Allowlisted file is not UTF-8 text: {relative.as_posix()}") from exc

        for secret in secrets:
            if len(secret) >= 8 and secret in text:
                raise PackagingError(f"Configured secret found in allowlisted file: {relative.as_posix()}")

        for match in KEY_ASSIGNMENT.finditer(text):
            value = match.group(1).strip().strip("'\"")
            normalized = value.casefold()
            is_placeholder = normalized in PLACEHOLDER_VALUES or normalized.startswith("your_")
            if not is_placeholder:
                raise PackagingError(
                    f"Non-placeholder PIXABAY_API_KEY assignment found in allowlisted file: {relative.as_posix()}"
                )


def _archive_name(relative: Path) -> str:
    return (ARCHIVE_PREFIX / PurePosixPath(relative.as_posix())).as_posix()


def _zip_info(source: Path, arcname: str) -> zipfile.ZipInfo:
    timestamp = datetime.fromtimestamp(source.stat().st_mtime)
    year = min(2107, max(1980, timestamp.year))
    info = zipfile.ZipInfo(
        filename=arcname,
        date_time=(year, timestamp.month, timestamp.day, timestamp.hour, timestamp.minute, timestamp.second),
    )
    info.create_system = 3
    executable = arcname.endswith(".py") and "/scripts/" in arcname
    mode = 0o100755 if executable else 0o100644
    info.external_attr = mode << 16
    info.compress_type = zipfile.ZIP_DEFLATED
    return info


def _validate_archive(path: Path, expected_names: Sequence[str], secrets: Sequence[str]) -> None:
    expected = list(expected_names)
    with zipfile.ZipFile(path, "r") as archive:
        names = archive.namelist()
        if names != expected or len(names) != len(set(names)):
            raise PackagingError("ZIP members do not exactly match the deterministic allowlist")
        if archive.testzip() is not None:
            raise PackagingError("ZIP CRC validation failed")
        for info in archive.infolist():
            name = info.filename
            pure = PurePosixPath(name)
            if "\\" in name or pure.is_absolute() or ".." in pure.parts:
                raise PackagingError(f"Non-portable ZIP member name: {name}")
            if not name.startswith(f"{ARCHIVE_PREFIX.as_posix()}/"):
                raise PackagingError(f"ZIP member escaped canonical Skill prefix: {name}")
            data = archive.read(info)
            for secret in secrets:
                if len(secret) >= 8 and secret.encode("utf-8") in data:
                    raise PackagingError(f"Configured secret found in ZIP member: {name}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_package(
    skill_root: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    *,
    profile: str = "runtime",
    force: bool = False,
    extra_secrets: Iterable[str] = (),
) -> dict[str, object]:
    """Create and validate one profiled ZIP, refusing silent overwrites."""

    root = Path(skill_root).expanduser().resolve(strict=True)
    destination = Path(output_path).expanduser().resolve(strict=False)
    if destination.suffix.casefold() != ".zip":
        raise PackagingError(f"Output path must end in .zip: {destination}")
    try:
        destination.relative_to(root)
    except ValueError:
        pass
    else:
        raise PackagingError("Output ZIP must be outside the Skill source directory")
    if destination.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite existing package without --force: {destination}")

    relatives = collect_allowlisted_files(root, profile)
    secrets = _configured_secrets(root, extra_secrets)
    _validate_source_contents(root, relatives, secrets)
    archive_names = [_archive_name(relative) for relative in relatives]

    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.stem}.", suffix=".tmp.zip", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for relative, arcname in zip(relatives, archive_names):
                source = root / relative
                archive.writestr(_zip_info(source, arcname), source.read_bytes())
        _validate_archive(temporary, archive_names, secrets)
        if destination.exists() and not force:
            raise FileExistsError(f"Refusing to overwrite existing package without --force: {destination}")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)

    return {
        "skill": SKILL_NAME,
        "profile": profile,
        "output": str(destination),
        "sha256": _sha256(destination),
        "size_bytes": destination.stat().st_size,
        "entry_count": len(archive_names),
        "entries": archive_names,
        "path_separator": "/",
        "sensitive_files_included": False,
    }


def _default_output(skill_root: Path, version: str, profile: str) -> Path:
    project_root = _project_root_for(skill_root)
    return project_root.parent / "skills" / project_root.name / f"{SKILL_NAME}-v{version}-{profile}.zip"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skill-root",
        default=str(Path(__file__).resolve().parent.parent),
        help="Canonical bgm-montage Skill directory",
    )
    parser.add_argument("--output", help="Destination .zip path")
    parser.add_argument("--version", default="1.4.3", help="Version used by the default ZIP filename")
    parser.add_argument(
        "--profile",
        choices=("runtime", "development"),
        default="runtime",
        help="runtime excludes tests/reports/packager; development includes them",
    )
    parser.add_argument("--force", action="store_true", help="Explicitly replace an existing ZIP")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    skill_root = Path(args.skill_root).expanduser().resolve(strict=True)
    output = (
        Path(args.output).expanduser()
        if args.output
        else _default_output(skill_root, args.version, args.profile)
    )
    try:
        report = build_package(skill_root, output, profile=args.profile, force=args.force)
    except (PackagingError, FileExistsError, FileNotFoundError, OSError) as exc:
        print(f"package_skill: {exc}", file=__import__("sys").stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
