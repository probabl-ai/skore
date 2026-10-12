import copy
import io
import json
import re
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from skore._cli.skills import catalog as _catalog

CATALOG = {
    "skills": [
        {
            "id": "alpha",
            "path": "skills/alpha",
            "title": "Alpha",
            "summary": "The alpha skill",
            "category": "tooling",
            "hash": "hash-alpha-1",
        },
        {
            "id": "beta",
            "path": "skills/beta",
            "title": "Beta",
            "summary": "The beta skill",
            "category": "reference",
            "hash": "hash-beta-1",
        },
    ],
    "workflows": [
        {
            "id": "flow",
            "title": "Flow",
            "summary": "Bundle of alpha and beta",
            "includes": ["alpha", "beta"],
        },
    ],
}

_REPO_URL = re.compile(r"https://api\.github\.com/repos/([^/]+/[^/]+)/")


def _build_tarball(catalog, *, catalog_name=".catalog.json"):
    buffer = io.BytesIO()
    prefix = "probabl-ai-skills-test"

    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:

        def add_bytes(name, data):
            info = tarfile.TarInfo(f"{prefix}/{name}")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))

        add_bytes(catalog_name, json.dumps(catalog).encode())
        for skill in catalog["skills"]:
            content = f"# {skill['title']}\n".encode()
            add_bytes(f"{skill['path']}/SKILL.md", content)

    return buffer.getvalue()


@pytest.fixture
def catalog_dict():
    return copy.deepcopy(CATALOG)


@pytest.fixture
def release_tarball(catalog_dict):
    return _build_tarball(catalog_dict)


@pytest.fixture
def legacy_release_tarball(catalog_dict):
    """Serve a release that still ships the pre-migration ``catalog.json``."""
    return _build_tarball(catalog_dict, catalog_name="catalog.json")


@pytest.fixture
def release(monkeypatch):
    """Serve fake GitHub skills releases from in-memory tarballs."""
    state = {
        "tag": "0.1.0",
        "catalog": copy.deepcopy(CATALOG),
        "by_repo": {},
        "urls": [],
    }

    def fake_fetch(url):
        state["urls"].append(url)
        match = _REPO_URL.match(url)
        repo = match.group(1) if match else _catalog.GITHUB_REPO
        entry = state["by_repo"].get(
            repo, {"tag": state["tag"], "catalog": state["catalog"]}
        )
        if url.endswith("/releases/latest"):
            return json.dumps({"tag_name": entry["tag"]}).encode()
        if "/tarball/" in url:
            return _build_tarball(entry["catalog"])
        raise AssertionError(f"unexpected url: {url}")

    monkeypatch.setattr(_catalog, "_fetch_bytes", fake_fetch)
    return state


@pytest.fixture
def workspace(monkeypatch, tmp_path):
    """Provide isolated home and project directories for skill installs."""
    home = tmp_path / "home"
    project = tmp_path / "project"
    home.mkdir()
    project.mkdir()

    monkeypatch.setattr(Path, "home", lambda: home)
    monkeypatch.chdir(project)
    for var in (
        "CLAUDECODE",
        "CURSOR_AGENT",
        "CURSOR_CLI",
        "GEMINI_CLI",
        "CODEX_SANDBOX",
        "PI_CODING_AGENT",
        "OPENCODE_CLIENT",
        "CI",
    ):
        monkeypatch.delenv(var, raising=False)

    return SimpleNamespace(home=home, project=project)


@pytest.fixture(autouse=True)
def monkeypatch_tmpdir(monkeypatch, tmp_path):
    """
    Change ``TMPDIR`` used by ``tempfile.gettempdir()`` to point to ``tmp_path``, so
    that it is automatically deleted after use, with no impact on user's environment.

    Force the reload of the ``tempfile`` module to change the cached return of
    ``tempfile.gettempdir()``.

    https://docs.python.org/3/library/tempfile.html#tempfile.gettempdir
    """
    import importlib
    import tempfile

    monkeypatch.setenv("TMPDIR", str(tmp_path))
    importlib.reload(tempfile)


@pytest.fixture
def monkeypatch_home(monkeypatch, tmp_path):
    """
    Change ``HOME`` used by ``os.path.expanduser()`` to point to ``tmp_path``, so
    that it is automatically deleted after use, with no impact on user's environment.

    https://docs.python.org/3/library/os.path.html#os.path.expanduser
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))


@pytest.fixture
def monkeypatch_keyring(monkeypatch):
    """
    Make the system keyring unavailable, so that tests can't read and write into the
    user's real keyring.

    Force a re-init of the keyring backend, ``keyring.get_keyring()`` caching the
    backend in ``keyring.core``.

    https://github.com/jaraco/keyring#disabling-keyring
    https://github.com/jaraco/keyring/blob/7603e7cadc254b4c6e3fc2b2f0916a005e78087d/keyring/core.py#L32
    """
    import keyring.core

    monkeypatch.setenv("PYTHON_KEYRING_BACKEND", "keyring.backends.null.Keyring")
    keyring.core.init_backend()


@pytest.fixture(autouse=True)
def monkeypatch_sdk_env(monkeypatch):
    """Unset the SDK credential variables."""
    monkeypatch.delenv("SKORE_HUB_API_KEY", raising=False)
    monkeypatch.delenv("SKORE_HUB_URI", raising=False)
