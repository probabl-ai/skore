"""Installed-skill sidecars and per-target catalog files."""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from skore._cli.skills.catalog import GITHUB_REPO

SIDECAR = ".skore-skill.json"
LOCAL_CATALOG = ".catalog.json"
LEGACY_CATALOG = "catalog.json"


class SkillStoreError(Exception):
    """A local skills catalog or sidecar could not be read."""


def skill_repository(sidecar: dict[str, Any]) -> str:
    """Return the GitHub ``owner/name`` recorded on an installed skill."""
    return sidecar.get("repository") or GITHUB_REPO


def local_catalog_path(target: Path) -> Path | None:
    """Return the on-disk catalog file, preferring the hidden name."""
    hidden = target / LOCAL_CATALOG
    if hidden.is_file():
        return hidden
    legacy = target / LEGACY_CATALOG
    if legacy.is_file():
        return legacy
    return None


def envelope_from_catalog_data(data: Any) -> dict[str, Any]:
    """Normalize a catalog file into the ``sources`` envelope.

    A file without ``sources`` is the previous single-repository snapshot.
    """
    if isinstance(data, dict) and "sources" in data:
        return data
    repo = GITHUB_REPO
    if isinstance(data, dict):
        repo = data.get("repository") or GITHUB_REPO
    return {"sources": {repo: data}}


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise SkillStoreError(f"could not parse {path}: {error}") from error


def read_local_catalog(target: Path) -> dict[str, Any]:
    """Load the per-target catalog envelope, or an empty sources map."""
    path = local_catalog_path(target)
    if path is None:
        return {"sources": {}}
    data = _load_json(path)
    if not isinstance(data, dict):
        raise SkillStoreError(f"could not read {path}: expected a JSON object.")
    return envelope_from_catalog_data(data)


def _remove_legacy_catalog(target: Path) -> None:
    """Delete a leftover ``catalog.json`` after migrating to ``.catalog.json``."""
    legacy = target / LEGACY_CATALOG
    if legacy.is_file():
        legacy.unlink()


def write_local_catalog(target: Path, envelope: dict[str, Any]) -> None:
    """Write the per-target catalog envelope to ``.catalog.json``."""
    target.mkdir(parents=True, exist_ok=True)
    (target / LOCAL_CATALOG).write_text(json.dumps(envelope, indent=2) + "\n")
    _remove_legacy_catalog(target)


def persist_source_catalog(
    target: Path, repo: str, tag: str, catalog: dict[str, Any]
) -> None:
    """Merge ``catalog`` for ``repo`` into ``target``'s local ``.catalog.json``."""
    envelope = read_local_catalog(target)
    envelope.setdefault("sources", {})[repo] = {
        **catalog,
        "repository": repo,
        "release": tag,
    }
    write_local_catalog(target, envelope)


def installed(target: Path) -> Iterator[tuple[Path, dict[str, Any]]]:
    """Yield ``(directory, sidecar)`` for each skill installed under ``target``."""
    if not target.is_dir():
        return

    for child in sorted(target.iterdir()):
        if not child.is_dir():
            continue
        sidecar = child / SIDECAR
        if sidecar.is_file():
            payload = _load_json(sidecar)
            if not isinstance(payload, dict) or "id" not in payload:
                raise SkillStoreError(
                    f"could not read {sidecar}: expected an object with an id."
                )
            yield child, payload


def prune_local_catalog(target: Path) -> None:
    """Drop catalog sources that no longer have an installed skill in ``target``."""
    remaining = {skill_repository(sidecar) for _, sidecar in installed(target)}
    envelope = read_local_catalog(target)
    sources = {
        repo: entry
        for repo, entry in envelope.get("sources", {}).items()
        if repo in remaining
    }
    path = target / LOCAL_CATALOG
    if not sources:
        if path.is_file():
            path.unlink()
        _remove_legacy_catalog(target)
        return
    envelope["sources"] = sources
    write_local_catalog(target, envelope)


def write_sidecar(destination: Path, skill: dict, tag: str, repo: str) -> None:
    """Write ``.skore-skill.json`` for an installed skill directory."""
    sidecar = {
        "id": skill["id"],
        "release": tag,
        "hash": skill["hash"],
        "repository": repo,
    }
    (destination / SIDECAR).write_text(json.dumps(sidecar, indent=2))


def install_skill(skill: dict, root: Path, target: Path, tag: str, repo: str) -> None:
    """Copy ``skill`` from the extracted release into ``target`` and write a sidecar."""
    source = root / skill["path"]
    destination = target / skill["id"]
    staging = destination.with_name(f".{destination.name}.skore-tmp")
    backup = destination.with_name(f".{destination.name}.skore-bak")
    if staging.exists():
        shutil.rmtree(staging)
    try:
        shutil.copytree(source, staging)
        write_sidecar(staging, skill, tag, repo)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    if backup.exists():
        shutil.rmtree(backup)
    replaced = destination.exists()
    if replaced:
        destination.rename(backup)
    try:
        staging.rename(destination)
    except Exception:
        if replaced:
            backup.rename(destination)
        raise
    if replaced:
        shutil.rmtree(backup, ignore_errors=True)
