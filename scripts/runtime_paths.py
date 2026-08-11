#!/usr/bin/env python3
"""Shared path contracts for bgm-montage v1.1.

Every entry point uses the same project cache layout.  The Pixabay stage is
always rooted at ``<cache>/pixabay``; helpers accept the project cache root for
backward compatibility but never append ``pixabay`` twice.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


SKILL_DIR = Path(__file__).resolve().parent.parent


def _candidate_roots(start: Path | None = None) -> Iterable[Path]:
    current = (start or Path.cwd()).expanduser().resolve()
    seen: set[Path] = set()
    for candidate in (current, *current.parents, *SKILL_DIR.parents):
        if candidate not in seen:
            seen.add(candidate)
            yield candidate


def discover_project_root(start: Path | None = None) -> Path:
    """Find the data project without assuming where the skill is installed."""

    explicit = os.environ.get("BGM_MONTAGE_PROJECT_ROOT", "").strip()
    if explicit:
        return Path(explicit).expanduser().resolve()
    for candidate in _candidate_roots(start):
        if (candidate / "参考视频").is_dir() and (
            (candidate / "视频素材").exists() or (candidate / ".env").is_file()
        ):
            return candidate
        if (candidate / ".agents" / "skills" / "bgm-montage" / "SKILL.md").is_file():
            return candidate
    # A standard project-local install is <project>/.agents/skills/bgm-montage.
    if SKILL_DIR.parent.name == "skills" and SKILL_DIR.parent.parent.name == ".agents":
        return SKILL_DIR.parent.parent.parent.resolve()
    return (start or Path.cwd()).expanduser().resolve()


def default_library_root() -> Path:
    """Return the machine-wide catalog root used for cross-project reuse."""

    explicit = os.environ.get("BGM_MONTAGE_LIBRARY_ROOT", "").strip()
    if explicit:
        return Path(explicit).expanduser().resolve()
    if os.name == "nt" and os.environ.get("LOCALAPPDATA"):
        return (Path(os.environ["LOCALAPPDATA"]) / "bgm-montage" / "material-library").resolve()
    cache_home = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return (cache_home / "bgm-montage" / "material-library").expanduser().resolve()


def pixabay_cache_root(cache_dir: str | os.PathLike[str]) -> Path:
    """Normalize either the project cache root or its Pixabay stage root."""

    path = Path(cache_dir).expanduser().resolve()
    return path if path.name.casefold() == "pixabay" else path / "pixabay"


@dataclass(frozen=True)
class RuntimePaths:
    project_root: Path
    cache_root: Path
    reference_cache: Path
    bgm_cache: Path
    pixabay_cache: Path
    pixabay_search_cache: Path
    pixabay_thumbnail_cache: Path
    project_material_index: Path
    library_root: Path
    global_material_index: Path

    @classmethod
    def build(
        cls,
        project_root: str | os.PathLike[str] | None = None,
        cache_root: str | os.PathLike[str] | None = None,
        library_root: str | os.PathLike[str] | None = None,
    ) -> "RuntimePaths":
        project = (
            Path(project_root).expanduser().resolve()
            if project_root is not None
            else discover_project_root()
        )
        cache = (
            Path(cache_root).expanduser().resolve()
            if cache_root is not None
            else project / ".bgm-montage-cache"
        )
        pixabay = pixabay_cache_root(cache)
        library = (
            Path(library_root).expanduser().resolve()
            if library_root is not None
            else default_library_root()
        )
        return cls(
            project_root=project,
            cache_root=cache,
            reference_cache=cache / "references",
            bgm_cache=cache / "bgm",
            pixabay_cache=pixabay,
            pixabay_search_cache=pixabay / "search",
            pixabay_thumbnail_cache=pixabay / "thumbnails",
            project_material_index=pixabay / "material_index.json",
            library_root=library,
            global_material_index=library / "material_index.json",
        )


def migrate_legacy_nested_pixabay_cache(stage_root: Path) -> dict[str, int]:
    """Copy useful v1.0 ``pixabay/pixabay`` cache entries into v1.1 paths.

    The legacy tree is intentionally left untouched so migration is reversible.
    """

    stage_root = stage_root.expanduser().resolve()
    legacy = stage_root / "pixabay"
    copied = {"search": 0, "thumbnails": 0}
    for folder, pattern in (("search", "*.json"), ("thumbnails", "*")):
        source = legacy / folder
        destination = stage_root / folder
        if not source.is_dir():
            continue
        destination.mkdir(parents=True, exist_ok=True)
        for item in source.glob(pattern):
            if not item.is_file():
                continue
            target = destination / item.name
            if target.exists():
                continue
            shutil.copy2(item, target)
            copied[folder] += 1
    return copied
