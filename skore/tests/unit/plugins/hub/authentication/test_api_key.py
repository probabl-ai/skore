from json import loads

from pytest import fixture, raises

from skore._plugins.hub.authentication.api_key import (
    ENV_VAR_NAME,
    KEYRING_SERVICE,
    API_key,
    APIKeyError,
    Registry,
)
from skore._plugins.hub.authentication.uri import DEFAULT as DEFAULT_URI


@fixture
def plaintext_keyring(monkeypatch):
    from keyring.backends.fail import Keyring as FailBackend

    monkeypatch.setattr(
        "skore._plugins.hub.authentication.api_key.get_keyring",
        lambda: FailBackend(),
    )


@fixture
def secret_keyring(monkeypatch):
    passwords = {}

    monkeypatch.setattr(
        "skore._plugins.hub.authentication.api_key.get_keyring",
        lambda: object(),
    )
    monkeypatch.setattr(
        "skore._plugins.hub.authentication.api_key.recommended",
        lambda backend: True,
    )
    monkeypatch.setattr(
        "skore._plugins.hub.authentication.api_key.set_password",
        lambda service, username, password: passwords.__setitem__(
            (service, username), password
        ),
    )
    monkeypatch.setattr(
        "skore._plugins.hub.authentication.api_key.get_password",
        lambda service, username: passwords.get((service, username)),
    )

    def delete_password(service, username):
        try:
            del passwords[(service, username)]
        except KeyError:
            from keyring.errors import PasswordDeleteError

            raise PasswordDeleteError() from None

    monkeypatch.setattr(
        "skore._plugins.hub.authentication.api_key.delete_password",
        delete_password,
    )
    return passwords


def test_api_key(monkeypatch):
    monkeypatch.setenv(ENV_VAR_NAME, "<api-key>")

    assert API_key()() == {"X-API-Key": "<api-key>"}


def test_api_key_reads_environment_on_call(monkeypatch):
    monkeypatch.setenv(ENV_VAR_NAME, "<api-key>")
    credentials = API_key()

    monkeypatch.setenv(ENV_VAR_NAME, "<updated-api-key>")

    assert credentials() == {"X-API-Key": "<updated-api-key>"}


def test_api_key_missing(monkeypatch):
    monkeypatch.delenv(ENV_VAR_NAME, raising=False)

    with raises(APIKeyError):
        API_key()


class TestRegistryPlaintext:
    @fixture(autouse=True)
    def _plaintext_keyring(self, plaintext_keyring):
        pass

    def test_filepath_creates_empty_registry(self, tmp_path):
        filepath = Registry().filepath

        assert filepath == tmp_path / ".skore.hub" / "credentials.json"
        assert loads(filepath.read_text()) == {"type": "plaintext", "keys": []}

    def test_filepath_does_not_overwrite_existing_registry(self, tmp_path):
        filepath = tmp_path / ".skore.hub" / "credentials.json"
        filepath.parent.mkdir()
        filepath.write_text('{"type": "plaintext", "keys": [{"host": "h"}]}')

        registry = Registry()

        assert registry.filepath == filepath
        assert loads(registry.filepath.read_text()) == {
            "type": "plaintext",
            "keys": [{"host": "h"}],
        }

    def test_type(self):
        assert Registry().type == "plaintext"

    def test_iter_empty(self):
        assert list(Registry()) == []

    def test_iter(self):
        registry = Registry()
        registry.set(uri="https://a.example", workspace="w1", api_key="k1")
        registry.set(uri="https://b.example", workspace="w2", api_key="k2")

        assert list(registry) == [
            ("https://a.example", "w1"),
            ("https://b.example", "w2"),
        ]

    def test_get(self):
        registry = Registry()
        registry.set(uri="https://a.example", workspace="w1", api_key="k1")
        registry.set(uri="https://a.example", workspace="w2", api_key="k2")

        assert registry.get(uri="https://a.example", workspace="w1") == "k1"
        assert registry.get(uri="https://a.example", workspace="w2") == "k2"

    def test_get_missing(self):
        registry = Registry()
        registry.set(uri="https://a.example", workspace="w1", api_key="k1")

        with raises(APIKeyError):
            registry.get(uri="https://a.example", workspace="missing")

        with raises(APIKeyError):
            registry.get(uri="https://missing.example", workspace="w1")

    def test_set(self):
        registry = Registry()
        registry.set(uri="https://a.example", workspace="workspace", api_key="k1")

        assert loads(registry.filepath.read_text()) == {
            "type": "plaintext",
            "keys": [
                {
                    "host": "https://a.example",
                    "workspace": "workspace",
                    "key": "k1",
                }
            ],
        }

    def test_set_replaces_matching_credential(self):
        registry = Registry()
        registry.set(uri="https://a.example", workspace="w1", api_key="k1")
        registry.set(uri="https://b.example", workspace="w1", api_key="k2")
        registry.set(uri="https://c.example", workspace="w2", api_key="k3")
        registry.set(uri="https://a.example", workspace="w1", api_key="k1-updated")

        assert loads(registry.filepath.read_text()) == {
            "type": "plaintext",
            "keys": [
                {
                    "host": "https://b.example",
                    "workspace": "w1",
                    "key": "k2",
                },
                {
                    "host": "https://c.example",
                    "workspace": "w2",
                    "key": "k3",
                },
                {
                    "host": "https://a.example",
                    "workspace": "w1",
                    "key": "k1-updated",
                },
            ],
        }

    def test_set_without_uri_uses_default_uri(self, monkeypatch):
        monkeypatch.delenv("SKORE_HUB_URI", raising=False)

        registry = Registry()
        registry.set(workspace="workspace", api_key="k1")

        assert loads(registry.filepath.read_text()) == {
            "type": "plaintext",
            "keys": [
                {
                    "host": DEFAULT_URI,
                    "workspace": "workspace",
                    "key": "k1",
                }
            ],
        }

    def test_set_without_uri_uses_environment_uri(self, monkeypatch):
        monkeypatch.setenv("SKORE_HUB_URI", "https://custom.example")

        registry = Registry()
        registry.set(workspace="workspace", api_key="k1")

        assert registry.get(uri="https://custom.example", workspace="workspace") == "k1"

    def test_get_without_uri_uses_environment_uri(self, monkeypatch):
        monkeypatch.setenv("SKORE_HUB_URI", "https://custom.example")

        registry = Registry()
        registry.set(uri="https://custom.example", workspace="workspace", api_key="k1")

        assert registry.get(workspace="workspace") == "k1"

    def test_delete_without_uri_uses_environment_uri(self, monkeypatch):
        monkeypatch.setenv("SKORE_HUB_URI", "https://custom.example")

        registry = Registry()
        registry.set(uri="https://custom.example", workspace="w1", api_key="k1")
        registry.set(uri="https://other.example", workspace="w2", api_key="k2")
        registry.delete(workspace="w1")

        assert loads(registry.filepath.read_text()) == {
            "type": "plaintext",
            "keys": [
                {
                    "host": "https://other.example",
                    "workspace": "w2",
                    "key": "k2",
                }
            ],
        }

    def test_delete(self):
        registry = Registry()
        registry.set(uri="https://a.example", workspace="w1", api_key="k1")
        registry.set(uri="https://b.example", workspace="w2", api_key="k2")
        registry.delete(uri="https://a.example", workspace="w1")

        assert loads(registry.filepath.read_text()) == {
            "type": "plaintext",
            "keys": [
                {
                    "host": "https://b.example",
                    "workspace": "w2",
                    "key": "k2",
                }
            ],
        }

    def test_delete_missing(self):
        registry = Registry()
        registry.set(uri="https://a.example", workspace="w1", api_key="k1")

        with raises(APIKeyError):
            registry.delete(uri="https://a.example", workspace="missing")


class TestRegistrySecret:
    @fixture(autouse=True)
    def _secret_keyring(self, secret_keyring):
        pass

    def test_filepath_creates_empty_registry(self, tmp_path):
        filepath = Registry().filepath

        assert filepath == tmp_path / ".skore.hub" / "credentials.json"
        assert loads(filepath.read_text()) == {"type": "secret", "keys": []}

    def test_type(self):
        assert Registry().type == "secret"

    def test_set_does_not_store_key_on_disk(self, secret_keyring):
        registry = Registry()
        registry.set(uri="https://a.example", workspace="w1", api_key="k1")

        assert loads(registry.filepath.read_text()) == {
            "type": "secret",
            "keys": [{"host": "https://a.example", "workspace": "w1"}],
        }
        assert secret_keyring[(KEYRING_SERVICE, "https://a.example:w1")] == "k1"

    def test_get(self):
        registry = Registry()
        registry.set(uri="https://a.example", workspace="w1", api_key="k1")
        registry.set(uri="https://a.example", workspace="w2", api_key="k2")

        assert registry.get(uri="https://a.example", workspace="w1") == "k1"
        assert registry.get(uri="https://a.example", workspace="w2") == "k2"

    def test_get_missing_in_keyring(self, secret_keyring):
        registry = Registry()
        registry.set(uri="https://a.example", workspace="w1", api_key="k1")
        secret_keyring.clear()

        with raises(APIKeyError):
            registry.get(uri="https://a.example", workspace="w1")

    def test_set_replaces_matching_credential(self, secret_keyring):
        registry = Registry()
        registry.set(uri="https://a.example", workspace="w1", api_key="k1")
        registry.set(uri="https://b.example", workspace="w2", api_key="k2")
        registry.set(uri="https://a.example", workspace="w1", api_key="k1-updated")

        assert loads(registry.filepath.read_text()) == {
            "type": "secret",
            "keys": [
                {"host": "https://b.example", "workspace": "w2"},
                {"host": "https://a.example", "workspace": "w1"},
            ],
        }
        assert secret_keyring[(KEYRING_SERVICE, "https://a.example:w1")] == "k1-updated"
        assert secret_keyring[(KEYRING_SERVICE, "https://b.example:w2")] == "k2"

    def test_delete(self, secret_keyring):
        registry = Registry()
        registry.set(uri="https://a.example", workspace="w1", api_key="k1")
        registry.set(uri="https://b.example", workspace="w2", api_key="k2")
        registry.delete(uri="https://a.example", workspace="w1")

        assert loads(registry.filepath.read_text()) == {
            "type": "secret",
            "keys": [{"host": "https://b.example", "workspace": "w2"}],
        }
        assert (KEYRING_SERVICE, "https://a.example:w1") not in secret_keyring
        assert secret_keyring[(KEYRING_SERVICE, "https://b.example:w2")] == "k2"

    def test_delete_missing_in_keyring(self, secret_keyring):
        registry = Registry()
        registry.set(uri="https://a.example", workspace="w1", api_key="k1")
        secret_keyring.clear()

        registry.delete(uri="https://a.example", workspace="w1")

        assert loads(registry.filepath.read_text()) == {"type": "secret", "keys": []}
