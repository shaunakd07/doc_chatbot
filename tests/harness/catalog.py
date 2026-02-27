from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, List

from .types import FileSelector


def _natural_key(value: str) -> List[object]:
    parts = re.split(r"(\d+)", value.lower())
    out: List[object] = []
    for part in parts:
        if part.isdigit():
            out.append(int(part))
        else:
            out.append(part)
    return out


def _sorted_paths(items: Iterable[Path]) -> List[Path]:
    return sorted(items, key=lambda p: _natural_key(p.as_posix()))


def _resolve_file(path: Path) -> List[Path]:
    return [path] if path.is_file() else []


def _resolve_paths(paths: List[Path]) -> List[Path]:
    out: List[Path] = []
    for path in paths:
        if path.is_file():
            out.append(path)
    return _sorted_paths(out)


def _resolve_glob(root: Path, pattern: str, recursive: bool) -> List[Path]:
    iterator = root.rglob(pattern) if recursive else root.glob(pattern)
    return _sorted_paths(path for path in iterator if path.is_file())


def resolve_selector(selector: FileSelector, repo_root: Path) -> List[Path]:
    kind = selector.kind.strip().lower()
    if kind == "file":
        if not selector.path:
            raise RuntimeError("Selector kind=file requires `path`.")
        path = (repo_root / selector.path).resolve()
        return _resolve_file(path)

    if kind == "paths":
        paths = [(repo_root / p).resolve() for p in selector.paths]
        return _resolve_paths(paths)

    if kind in {"glob", "dir_all", "dir_first_n"}:
        root = (repo_root / str(selector.root or ".")).resolve()
        pattern = str(selector.pattern or "*")
        recursive = bool(selector.recursive)
        items = _resolve_glob(root, pattern, recursive)
        if kind == "dir_first_n":
            if selector.n is None:
                raise RuntimeError("Selector kind=dir_first_n requires `n`.")
            n = max(0, int(selector.n))
            return items[:n]
        return items

    raise RuntimeError(f"Unsupported selector kind: {selector.kind}")


def resolve_dataset(
    name: str,
    selectors: Dict[str, FileSelector],
    repo_root: Path,
) -> List[Path]:
    selector = selectors.get(name)
    if selector is None:
        raise RuntimeError(f"Unknown dataset selector: {name}")
    return resolve_selector(selector, repo_root)

