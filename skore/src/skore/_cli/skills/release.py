"""Fetch a skills release and remove its temporary extraction directory."""

from __future__ import annotations

import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from skore._cli.skills.catalog import GITHUB_REPO, fetch_release


class ReleaseError(Exception):
    """The latest skills release could not be fetched or parsed."""


def load_release(repo: str = GITHUB_REPO) -> tuple[str, Path, dict[str, Any]]:
    """Download the latest release of ``repo``.

    Raises
    ------
    ReleaseError
        GitHub could not be reached, or the release payload could not be parsed.
    """
    try:
        return fetch_release(repo)
    except (OSError, ValueError, KeyError) as error:
        raise ReleaseError(
            f"Could not fetch the latest skills release from GitHub ({repo}): {error}"
        ) from error


def cleanup_release(root: Path | None) -> None:
    """Remove the temporary directory that contains ``root``.

    ``download_release`` extracts into a dedicated temporary directory and
    returns its single child, so ``root.parent`` is that temporary directory.
    """
    if root is None:
        return
    shutil.rmtree(root.parent, ignore_errors=True)


@contextmanager
def open_release(
    repo: str = GITHUB_REPO,
) -> Iterator[tuple[str, Path, dict[str, Any]]]:
    """Yield a fetched release and delete its extraction directory afterwards."""
    tag, root, catalog = load_release(repo)
    try:
        yield tag, root, catalog
    finally:
        cleanup_release(root)
