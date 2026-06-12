import json
from urllib.request import Request

from skore._cli.skills import _catalog


def test_fetch_bytes_sends_user_agent(monkeypatch):
    captured: dict[str, Request] = {}

    class FakeResponse:
        def read(self):
            return b"payload"

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

    def fake_urlopen(request):
        captured["request"] = request
        return FakeResponse()

    monkeypatch.setattr(_catalog, "urlopen", fake_urlopen)

    assert _catalog._fetch_bytes("https://example.com") == b"payload"
    assert captured["request"].headers["User-agent"] == "skore-skills-cli"


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
        "https://api.github.com/repos/probabl-ai/skills/releases/latest"
    ]


def test_download_release(release_tarball, monkeypatch):
    monkeypatch.setattr(_catalog, "_fetch_bytes", lambda url: release_tarball)

    root = _catalog.download_release("0.1.0")

    assert root.is_dir()
    assert (root / "catalog.json").is_file()
    assert (root / "skills" / "alpha" / "SKILL.md").is_file()


def test_load_catalog(tmp_path, catalog_dict):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "catalog.json").write_text(json.dumps(catalog_dict))

    assert _catalog.load_catalog(root) == catalog_dict


def test_fetch_release(release, catalog_dict):
    tag, root, catalog = _catalog.fetch_release()

    assert tag == "0.1.0"
    assert (root / "catalog.json").is_file()
    assert catalog == catalog_dict
