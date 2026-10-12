"""Fetch and parse the skill catalog from a GitHub skills release."""

from __future__ import annotations

import json
import tarfile
from io import BytesIO
from pathlib import Path
from tempfile import mkdtemp
from typing import Any
from urllib.request import Request, urlopen

GITHUB_REPO = "probabl-ai/skills"
_USER_AGENT = "skore-skills-cli"
CATALOG_FILENAMES = (".catalog.json", "catalog.json")


def normalize_github_repo(value: str) -> str:
    """Return a canonical ``owner/name`` GitHub repository identifier.

    Parameters
    ----------
    value : str
        User-supplied repository string, possibly with extra whitespace.

    Returns
    -------
    str
        ``owner/name`` with surrounding slashes and whitespace stripped.

    Raises
    ------
    ValueError
        If ``value`` is not exactly ``owner/name`` with both parts non-empty.
    """
    repo = value.strip().strip("/")
    owner, sep, name = repo.partition("/")
    if not sep or "/" in name or not owner or not name:
        raise ValueError(f"GitHub repository must be owner/name, got {value!r}")
    return f"{owner}/{name}"


def _fetch_bytes(url: str) -> bytes:
    """Download ``url`` and return its raw content.

    Parameters
    ----------
    url : str
        The URL to download.

    Returns
    -------
    bytes
        The raw content of the response.
    """
    request = Request(url, headers={"User-Agent": _USER_AGENT})
    with urlopen(request) as response:
        return response.read()


def latest_release_tag(repo: str = GITHUB_REPO) -> str:
    """Return the tag name of the latest release of ``repo``.

    Parameters
    ----------
    repo : str, default="probabl-ai/skills"
        The ``owner/name`` of the GitHub repository.

    Returns
    -------
    str
        The ``tag_name`` of the latest release.
    """
    url = f"https://api.github.com/repos/{repo}/releases/latest"
    payload = json.loads(_fetch_bytes(url))
    return payload["tag_name"]


def download_release(tag: str, repo: str = GITHUB_REPO) -> Path:
    """Download and extract the source tarball of ``repo`` at ``tag``.

    Parameters
    ----------
    tag : str
        The release tag to download.
    repo : str, default="probabl-ai/skills"
        The ``owner/name`` of the GitHub repository.

    Returns
    -------
    Path
        The path to the extracted repository root.
    """
    url = f"https://api.github.com/repos/{repo}/tarball/{tag}"
    archive = _fetch_bytes(url)
    destination = Path(mkdtemp(prefix="skore-skills-"))

    with tarfile.open(fileobj=BytesIO(archive), mode="r:gz") as tar:
        tar.extractall(destination, filter="data")

    # GitHub source tarballs always contain a single top-level directory.
    (root,) = destination.iterdir()
    return root


def load_catalog(root: Path) -> dict[str, Any]:
    """Load the catalog JSON from an extracted repository ``root``.

    Prefers ``.catalog.json`` and falls back to ``catalog.json`` so older
    releases remain readable.

    Parameters
    ----------
    root : Path
        The path to the extracted repository root.

    Returns
    -------
    dict
        The parsed content of the catalog file.

    Raises
    ------
    FileNotFoundError
        If neither catalog filename is present under ``root``.
    """
    for name in CATALOG_FILENAMES:
        path = root / name
        if path.is_file():
            return json.loads(path.read_text())
    raise FileNotFoundError(
        f"No catalog file found in {root} (tried {', '.join(CATALOG_FILENAMES)})"
    )


def fetch_release(repo: str = GITHUB_REPO) -> tuple[str, Path, dict[str, Any]]:
    """Resolve, download and parse the latest release of ``repo``.

    Parameters
    ----------
    repo : str, default="probabl-ai/skills"
        The ``owner/name`` of the GitHub repository.

    Returns
    -------
    tag : str
        The tag name of the latest release.
    root : Path
        The path to the extracted repository root.
    catalog : dict
        The parsed content of the catalog file.
    """
    tag = latest_release_tag(repo)
    root = download_release(tag, repo)
    catalog = load_catalog(root)
    return tag, root, catalog
