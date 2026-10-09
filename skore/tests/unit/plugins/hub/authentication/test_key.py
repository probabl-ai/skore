from types import SimpleNamespace

from pytest import fixture, raises

from skore._plugins.hub.authentication import key as key_module
from skore._plugins.hub.authentication.key import KeyExistsError
from skore._plugins.hub.authentication.registry import distant, local
from skore._plugins.hub.authentication.uri import DEFAULT as DEFAULT_URI


@fixture
def hub(monkeypatch):
    generated = []
    revoked = []

    def generate(**kwargs):
        generated.append(kwargs)
        return distant.Key(id=len(generated), key=f"k{len(generated)}")

    def revoke(**kwargs):
        revoked.append(kwargs)

    monkeypatch.setattr(distant, "generate", generate)
    monkeypatch.setattr(distant, "revoke", revoke)

    return SimpleNamespace(generated=generated, revoked=revoked)


def test_generate_stores_key(hub):
    key_module.generate(
        host="https://a.example",
        workspace="w1",
        name="laptop",
        expires="3",
        timeout=10,
    )

    assert hub.generated == [
        {
            "host": "https://a.example",
            "workspace": "w1",
            "name": "laptop",
            "expires": "3",
            "timeout": 10,
        }
    ]
    assert hub.revoked == []
    assert key_module.get(host="https://a.example", workspace="w1") == "k1"
    assert list(key_module.keys()) == [(1, "https://a.example", "w1")]


def test_generate_normalizes_host(hub):
    key_module.generate(host="HTTPS://A.Example/", workspace="w1")

    assert hub.generated[0]["host"] == "https://a.example"
    assert key_module.get(host="https://a.example", workspace="w1") == "k1"


def test_generate_without_host_uses_environment_uri(monkeypatch, hub):
    monkeypatch.setenv("SKORE_HUB_URI", "https://custom.example")

    key_module.generate(workspace="w1")

    assert hub.generated[0]["host"] == "https://custom.example"
    assert hub.generated[0]["name"] is None
    assert hub.generated[0]["expires"] == "never"
    assert hub.generated[0]["timeout"] == 600
    assert key_module.get(workspace="w1") == "k1"


def test_generate_without_host_uses_default_uri(monkeypatch, hub):
    monkeypatch.delenv("SKORE_HUB_URI", raising=False)

    key_module.generate(workspace="w1")

    assert hub.generated[0]["host"] == DEFAULT_URI


def test_generate_refuses_to_overwrite(hub):
    key_module.generate(host="https://a.example", workspace="w1", name="first")

    with raises(KeyExistsError, match="already exists"):
        key_module.generate(host="https://a.example", workspace="w1", name="second")

    assert len(hub.generated) == 1
    assert hub.revoked == []
    assert key_module.get(host="https://a.example", workspace="w1") == "k1"


def test_generate_force_replaces_existing_key(hub):
    key_module.generate(host="https://a.example", workspace="w1", name="first")
    key_module.generate(
        host="https://a.example", workspace="w1", name="second", force=True
    )

    assert hub.revoked == [{"host": "https://a.example", "id": 1, "timeout": 600}]
    assert hub.generated[1]["name"] == "second"
    assert key_module.get(host="https://a.example", workspace="w1") == "k2"
    assert list(key_module.keys()) == [(2, "https://a.example", "w1")]


def test_get_missing():
    assert key_module.get(host="https://a.example", workspace="missing") is None


def test_get_without_stored_secret(monkeypatch):
    monkeypatch.setattr(
        local,
        "get",
        lambda **kwargs: local.Key(id=1, host="h", workspace="w", key=None),
    )

    assert key_module.get(host="h", workspace="w") is None


def test_revoke(hub):
    key_module.generate(host="https://a.example", workspace="w1")
    key_module.generate(host="https://b.example", workspace="w2")
    key_module.revoke(host="https://a.example", workspace="w1", timeout=10)

    assert hub.revoked == [{"host": "https://a.example", "id": 1, "timeout": 10}]
    assert key_module.get(host="https://a.example", workspace="w1") is None
    assert key_module.get(host="https://b.example", workspace="w2") == "k2"
    assert list(key_module.keys()) == [(2, "https://b.example", "w2")]


def test_revoke_missing_is_a_noop(hub):
    key_module.revoke(host="https://a.example", workspace="missing")

    assert hub.revoked == []


def test_revoke_without_host_uses_environment_uri(monkeypatch, hub):
    monkeypatch.setenv("SKORE_HUB_URI", "https://custom.example")

    key_module.generate(host="https://custom.example", workspace="w1")
    key_module.generate(host="https://other.example", workspace="w2")
    key_module.revoke(workspace="w1")

    assert hub.revoked == [{"host": "https://custom.example", "id": 1, "timeout": 600}]
    assert key_module.get(workspace="w1") is None
    assert key_module.get(host="https://other.example", workspace="w2") == "k2"
