import json
import tarfile
from io import BytesIO
from urllib.request import Request

import pytest

from skore._cli.skills import catalog as _catalog
from skore._cli.skills.release import ReleaseError, load_release
from skore._cli.skills.store import persist_source_catalog, skill_repository


def test_fetch_bytes_sends_user_agent(monkeypatch):
    captured: dict[str, Request] = {}
    timeouts: list[float | None] = []

    class FakeResponse:
        def read(self):
            return b"payload"

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

    def fake_urlopen(request, timeout=None):
        captured["request"] = request
        timeouts.append(timeout)
        return FakeResponse()

    monkeypatch.setattr(_catalog, "urlopen", fake_urlopen)

    assert _catalog._fetch_bytes("https://example.com") == b"payload"
    assert captured["request"].headers["User-agent"] == "skore-skills-cli"
    assert timeouts == [30]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("probabl-ai/skore-skills", "probabl-ai/skore-skills"),
        ("  acme/skills  ", "acme/skills"),
        ("/acme/skills/", "acme/skills"),
    ],
)
def test_normalize_github_repo(value, expected):
    assert _catalog.normalize_github_repo(value) == expected


@pytest.mark.parametrize("value", ["", "acme", "acme/", "/skills", "a/b/c"])
def test_normalize_github_repo_rejects_invalid(value):
    with pytest.raises(ValueError, match="owner/name"):
        _catalog.normalize_github_repo(value)


def test_skill_repository_follows_renamed_repository():
    assert skill_repository({"repository": "probabl-ai/skills"}) == _catalog.GITHUB_REPO
    assert skill_repository({}) == _catalog.GITHUB_REPO


def test_latest_release_tag(release):
    assert _catalog.latest_release_tag() == "0.1.0"


def test_latest_release_tag_requests_latest_endpoint(monkeypatch):
    captured: list[str] = []

    def fake_fetch(url):
        captured.append(url)
        return json.dumps({"tag_name": "1.2.3"}).encode()

    monkeypatch.setattr(_catalog, "_fetch_bytes", fake_fetch)

    assert _catalog.latest_release_tag() == "1.2.3"
    assert captured == [
        "https://api.github.com/repos/probabl-ai/skore-skills/releases/latest"
    ]


def test_latest_release_tag_custom_repo(monkeypatch):
    captured: list[str] = []

    def fake_fetch(url):
        captured.append(url)
        return json.dumps({"tag_name": "9.9.9"}).encode()

    monkeypatch.setattr(_catalog, "_fetch_bytes", fake_fetch)

    assert _catalog.latest_release_tag("acme/skills") == "9.9.9"
    assert captured == ["https://api.github.com/repos/acme/skills/releases/latest"]


def test_download_release(release_tarball, monkeypatch):
    monkeypatch.setattr(_catalog, "_fetch_bytes", lambda url: release_tarball)

    root = _catalog.download_release("0.1.0")

    assert root.is_dir()
    assert (root / ".catalog.json").is_file()
    assert (root / "skills" / "alpha" / "SKILL.md").is_file()


def test_load_catalog_prefers_hidden_file(tmp_path, catalog_dict):
    root = tmp_path / "repo"
    root.mkdir()
    hidden = {**catalog_dict, "name": "hidden"}
    (root / ".catalog.json").write_text(json.dumps(hidden))
    (root / "catalog.json").write_text(json.dumps(catalog_dict))

    assert _catalog.load_catalog(root) == hidden


def test_load_catalog_without_any_catalog_file(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()

    with pytest.raises(FileNotFoundError, match="No catalog file found"):
        _catalog.load_catalog(root)


def test_load_catalog_falls_back_to_catalog_json(tmp_path, catalog_dict):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "catalog.json").write_text(json.dumps(catalog_dict))

    assert _catalog.load_catalog(root) == catalog_dict


def test_fetch_release(release, catalog_dict):
    tag, root, catalog = _catalog.fetch_release()

    assert tag == "0.1.0"
    assert (root / ".catalog.json").is_file()
    assert catalog == {**catalog_dict, "version": tag}


def test_fetch_release_custom_repo(release, catalog_dict):
    other = {**catalog_dict, "name": "acme"}
    release["by_repo"]["acme/skills"] = {"tag": "2.0.0", "catalog": other}

    tag, root, catalog = _catalog.fetch_release("acme/skills")

    assert tag == "2.0.0"
    assert catalog == {**other, "version": tag}
    assert any("/repos/acme/skills/" in url for url in release["urls"])
    assert (root / ".catalog.json").is_file()


def test_fetch_release_persists_tag_when_catalog_has_no_version(
    release, catalog_dict, tmp_path
):
    tag, _root, catalog = _catalog.fetch_release()

    assert "version" not in catalog_dict
    persist_source_catalog(tmp_path, "probabl-ai/skore-skills", tag, catalog)
    stored = json.loads((tmp_path / ".catalog.json").read_text())

    assert stored["sources"]["probabl-ai/skore-skills"]["version"] == tag


def test_fetch_release_persists_tag_over_stale_catalog_version(release, tmp_path):
    release["catalog"]["version"] = "0.0.1"
    tag, _root, catalog = _catalog.fetch_release()

    assert catalog["version"] == tag == "0.1.0"
    persist_source_catalog(tmp_path, "probabl-ai/skore-skills", tag, catalog)
    stored = json.loads((tmp_path / ".catalog.json").read_text())

    assert stored["sources"]["probabl-ai/skore-skills"]["version"] == tag


def test_load_catalog_legacy_tarball(catalog_dict, legacy_release_tarball, monkeypatch):
    monkeypatch.setattr(_catalog, "_fetch_bytes", lambda url: legacy_release_tarball)

    root = _catalog.download_release("0.1.0")
    assert _catalog.load_catalog(root) == catalog_dict


def _gzip_tarball(members: dict[str, bytes]) -> bytes:
    buffer = BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, BytesIO(data))
    return buffer.getvalue()


def test_load_release_rejects_a_non_tar_body(monkeypatch, tmp_path):
    extract = tmp_path / "extract"

    def fake_fetch(url):
        if url.endswith("/releases/latest"):
            return b'{"tag_name": "0.1.0"}'
        return b"not a tarball"

    def fake_mkdtemp(**kwargs):
        extract.mkdir()
        return str(extract)

    monkeypatch.setattr(_catalog, "_fetch_bytes", fake_fetch)
    monkeypatch.setattr(_catalog, "mkdtemp", fake_mkdtemp)

    with pytest.raises(ReleaseError, match="Could not fetch"):
        load_release("owner/name")

    assert not extract.exists()


def test_fetch_release_removes_temp_dir_when_catalog_is_missing(monkeypatch, tmp_path):
    extract = tmp_path / "extract"
    archive = _gzip_tarball({"repo-1.0/README.md": b"hello"})

    def fake_fetch(url):
        if url.endswith("/releases/latest"):
            return b'{"tag_name": "0.1.0"}'
        return archive

    def fake_mkdtemp(**kwargs):
        extract.mkdir()
        return str(extract)

    monkeypatch.setattr(_catalog, "_fetch_bytes", fake_fetch)
    monkeypatch.setattr(_catalog, "mkdtemp", fake_mkdtemp)

    with pytest.raises(FileNotFoundError, match="No catalog file"):
        _catalog.fetch_release("owner/name")

    assert not extract.exists()
