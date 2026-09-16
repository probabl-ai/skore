from json import loads

from pytest import raises

from skore._plugins.hub.authentication.api_key import API_key, APIKeyError, Registry
from skore._plugins.hub.authentication.uri import DEFAULT as DEFAULT_URI


def test_api_key(monkeypatch):
    monkeypatch.setenv("SKORE_HUB_API_KEY", "<api-key>")

    assert API_key()() == {"X-API-Key": "<api-key>"}


def test_api_key_missing(monkeypatch):
    monkeypatch.delenv("SKORE_HUB_API_KEY", raising=False)

    with raises(APIKeyError):
        API_key()


class TestRegistry:
    def test_filepath_creates_empty_registry(self, tmp_path):
        filepath = Registry().filepath

        assert filepath == tmp_path / ".skore.hub" / "credentials.json"
        assert filepath.read_text() == "[]"

    def test_filepath_does_not_overwrite_existing_registry(self, tmp_path):
        filepath = tmp_path / ".skore.hub" / "credentials.json"
        filepath.parent.mkdir()
        filepath.write_text('[{"host": "h"}]')

        registry = Registry()

        assert registry.filepath == filepath
        assert registry.filepath.read_text() == '[{"host": "h"}]'

    def test_iter(self):
        registry = Registry()
        registry.persist(uri="https://a.example", workspace="w1", api_key="k1")
        registry.persist(uri="https://b.example", workspace="w2", api_key="k2")

        assert list(registry) == [
            ("https://a.example", "w1"),
            ("https://b.example", "w2"),
        ]

    def test_get(self):
        registry = Registry()
        registry.persist(uri="https://a.example", workspace="w1", api_key="k1")
        registry.persist(uri="https://a.example", workspace="w2", api_key="k2")

        assert registry.get(uri="https://a.example", workspace="w1") == "k1"
        assert registry.get(uri="https://a.example", workspace="w2") == "k2"

    def test_get_missing(self):
        registry = Registry()

        with raises(APIKeyError):
            registry.get(uri="https://a.example", workspace="missing")

    def test_persist(self):
        registry = Registry()
        registry.persist(uri="https://a.example", workspace="workspace", api_key="k1")

        assert loads(registry.filepath.read_text()) == [
            {
                "host": "https://a.example",
                "workspace": "workspace",
                "api_key": "k1",
            }
        ]

    def test_persist_replaces_matching_credential(self):
        registry = Registry()
        registry.persist(uri="https://a.example", workspace="w1", api_key="k1")
        registry.persist(uri="https://b.example", workspace="w1", api_key="k2")
        registry.persist(uri="https://c.example", workspace="w2", api_key="k3")

        assert loads(registry.filepath.read_text()) == [
            {
                "host": "https://a.example",
                "workspace": "w1",
                "api_key": "k1",
            },
            {
                "host": "https://b.example",
                "workspace": "w1",
                "api_key": "k2",
            },
            {
                "host": "https://c.example",
                "workspace": "w2",
                "api_key": "k3",
            },
        ]

        registry.persist(uri="https://a.example", workspace="w1", api_key="k1-updated")

        assert loads(registry.filepath.read_text()) == [
            {
                "host": "https://b.example",
                "workspace": "w1",
                "api_key": "k2",
            },
            {
                "host": "https://c.example",
                "workspace": "w2",
                "api_key": "k3",
            },
            {
                "host": "https://a.example",
                "workspace": "w1",
                "api_key": "k1-updated",
            },
        ]

    def test_persist_without_uri_uses_default_uri(self, monkeypatch):
        monkeypatch.delenv("SKORE_HUB_URI", raising=False)

        registry = Registry()
        registry.persist(workspace="workspace", api_key="k1")

        assert loads(registry.filepath.read_text()) == [
            {
                "host": DEFAULT_URI,
                "workspace": "workspace",
                "api_key": "k1",
            }
        ]

    def test_persist_without_uri_uses_environment_uri(self, monkeypatch):
        monkeypatch.setenv("SKORE_HUB_URI", "https://custom.example")

        registry = Registry()
        registry.persist(workspace="workspace", api_key="k1")

        assert registry.get(uri="https://custom.example", workspace="workspace") == "k1"
