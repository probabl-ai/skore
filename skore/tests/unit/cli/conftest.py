import copy
import io
import json
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from skore._cli.skills import _catalog

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


def _build_tarball(catalog):
    buffer = io.BytesIO()
    prefix = "probabl-ai-skills-test"

    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:

        def add_bytes(name, data):
            info = tarfile.TarInfo(f"{prefix}/{name}")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))

        add_bytes("catalog.json", json.dumps(catalog).encode())
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
def release(monkeypatch):
    """Serve a fake ``probabl-ai/skills`` release from an in-memory tarball."""
    state = {"tag": "0.1.0", "catalog": copy.deepcopy(CATALOG)}

    def fake_fetch(url):
        if url.endswith("/releases/latest"):
            return json.dumps({"tag_name": state["tag"]}).encode()
        if "/tarball/" in url:
            return _build_tarball(state["catalog"])
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

    return SimpleNamespace(home=home, project=project)
