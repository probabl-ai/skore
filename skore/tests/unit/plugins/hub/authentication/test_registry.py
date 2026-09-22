from json import loads
from stat import S_IMODE
from sys import platform

from pytest import fixture, mark

from skore._plugins.hub.authentication import registry
from skore._plugins.hub.authentication.uri import DEFAULT as DEFAULT_URI


def assert_credentials_permissions(filepath):
    if platform == "win32":
        return

    assert S_IMODE(filepath.stat().st_mode) == 0o600
    assert S_IMODE(filepath.parent.stat().st_mode) == 0o700


@fixture
def plaintext_keyring(monkeypatch):
    from keyring.backends.fail import Keyring as FailBackend

    monkeypatch.setattr(
        "skore._plugins.hub.authentication.registry.get_keyring",
        lambda: FailBackend(),
    )


@fixture
def secret_keyring(monkeypatch):
    passwords = {}

    monkeypatch.setattr(
        "skore._plugins.hub.authentication.registry.get_keyring",
        lambda: object(),
    )
    monkeypatch.setattr(
        "skore._plugins.hub.authentication.registry.recommended",
        lambda backend: True,
    )
    monkeypatch.setattr(
        "skore._plugins.hub.authentication.registry.set_password",
        lambda service, username, password: passwords.__setitem__(
            (service, username), password
        ),
    )
    monkeypatch.setattr(
        "skore._plugins.hub.authentication.registry.get_password",
        lambda service, username: passwords.get((service, username)),
    )

    def delete_password(service, username):
        try:
            del passwords[(service, username)]
        except KeyError:
            from keyring.errors import PasswordDeleteError

            raise PasswordDeleteError() from None

    monkeypatch.setattr(
        "skore._plugins.hub.authentication.registry.delete_password",
        delete_password,
    )

    return passwords


def content():
    return loads(registry.setup().read_text())


@mark.usefixtures("plaintext_keyring")
class TestPlaintext:
    def test_setup_creates_empty_registry(self, tmp_path):
        filepath = registry.setup()

        assert filepath == tmp_path / ".skore.hub" / "credentials.json"
        assert loads(filepath.read_text()) == {"type": "plaintext", "keys": []}
        assert_credentials_permissions(filepath)

    def test_setup_does_not_overwrite_existing_registry(self, tmp_path):
        filepath = tmp_path / ".skore.hub" / "credentials.json"
        filepath.parent.mkdir()
        filepath.write_text('{"type": "plaintext", "keys": [{"host": "h"}]}')

        assert registry.setup() == filepath
        assert loads(filepath.read_text()) == {
            "type": "plaintext",
            "keys": [{"host": "h"}],
        }

    @mark.skipif(platform == "win32", reason="POSIX permission bits")
    def test_setup_does_not_change_existing_modes(self, tmp_path):
        filepath = registry.setup()
        filepath.chmod(0o660)
        filepath.parent.chmod(0o770)

        registry.setup()

        assert S_IMODE(filepath.stat().st_mode) == 0o660
        assert S_IMODE(filepath.parent.stat().st_mode) == 0o770

    @mark.skipif(platform == "win32", reason="POSIX permission bits")
    @mark.parametrize("operation", ["set", "delete"])
    def test_write_preserves_existing_file_mode(self, tmp_path, operation):
        filepath = registry.setup()
        filepath.chmod(0o660)

        if operation == "set":
            registry.set(host="https://a.example", workspace="w1", api_key="k1")
        else:
            registry.delete(host="https://a.example", workspace="w1")

        assert S_IMODE(filepath.stat().st_mode) == 0o660

    def test_keys_empty(self):
        assert list(registry.keys()) == []

    def test_keys(self):
        registry.set(host="https://a.example", workspace="w1", api_key="k1")
        registry.set(host="https://b.example", workspace="w2", api_key="k2")

        assert list(registry.keys()) == [
            ("https://a.example", "w1"),
            ("https://b.example", "w2"),
        ]

    def test_get(self):
        registry.set(host="https://a.example", workspace="w1", api_key="k1")
        registry.set(host="https://a.example", workspace="w2", api_key="k2")

        assert registry.get(host="https://a.example", workspace="w1") == "k1"
        assert registry.get(host="https://a.example", workspace="w2") == "k2"

    def test_get_missing(self):
        registry.set(host="https://a.example", workspace="w1", api_key="k1")

        assert registry.get(host="https://a.example", workspace="missing") is None
        assert registry.get(host="https://missing.example", workspace="w1") is None

    def test_set(self):
        registry.set(host="https://a.example", workspace="workspace", api_key="k1")

        assert content() == {
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
        registry.set(host="https://a.example", workspace="w1", api_key="k1")
        registry.set(host="https://b.example", workspace="w1", api_key="k2")
        registry.set(host="https://c.example", workspace="w2", api_key="k3")
        registry.set(host="https://a.example", workspace="w1", api_key="k1-updated")

        assert content() == {
            "type": "plaintext",
            "keys": [
                {
                    "host": "https://a.example",
                    "workspace": "w1",
                    "key": "k1-updated",
                },
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
            ],
        }

    def test_set_without_host_uses_default_uri(self, monkeypatch):
        monkeypatch.delenv("SKORE_HUB_URI", raising=False)

        registry.set(workspace="workspace", api_key="k1")

        assert content() == {
            "type": "plaintext",
            "keys": [
                {
                    "host": DEFAULT_URI,
                    "workspace": "workspace",
                    "key": "k1",
                }
            ],
        }

    def test_set_without_host_uses_environment_uri(self, monkeypatch):
        monkeypatch.setenv("SKORE_HUB_URI", "https://custom.example")

        registry.set(workspace="workspace", api_key="k1")

        assert (
            registry.get(host="https://custom.example", workspace="workspace") == "k1"
        )

    def test_get_without_host_uses_environment_uri(self, monkeypatch):
        monkeypatch.setenv("SKORE_HUB_URI", "https://custom.example")

        registry.set(host="https://custom.example", workspace="workspace", api_key="k1")

        assert registry.get(workspace="workspace") == "k1"

    def test_delete(self):
        registry.set(host="https://a.example", workspace="w1", api_key="k1")
        registry.set(host="https://b.example", workspace="w2", api_key="k2")
        registry.delete(host="https://a.example", workspace="w1")

        assert content() == {
            "type": "plaintext",
            "keys": [
                {
                    "host": "https://b.example",
                    "workspace": "w2",
                    "key": "k2",
                }
            ],
        }

    def test_delete_without_host_uses_environment_uri(self, monkeypatch):
        monkeypatch.setenv("SKORE_HUB_URI", "https://custom.example")

        registry.set(host="https://custom.example", workspace="w1", api_key="k1")
        registry.set(host="https://other.example", workspace="w2", api_key="k2")
        registry.delete(workspace="w1")

        assert content() == {
            "type": "plaintext",
            "keys": [
                {
                    "host": "https://other.example",
                    "workspace": "w2",
                    "key": "k2",
                }
            ],
        }

    def test_host_trailing_slash_and_case_are_the_same_credential(self):
        registry.set(host="HTTPS://A.example/", workspace="w1", api_key="k1")

        assert registry.get(host="https://a.example", workspace="w1") == "k1"
        assert registry.get(host="https://a.example/", workspace="w1") == "k1"
        assert list(registry.keys()) == [("https://a.example", "w1")]
        assert content()["keys"][0]["host"] == "https://a.example"

        registry.set(host="https://a.example/", workspace="w1", api_key="k1-updated")

        assert content() == {
            "type": "plaintext",
            "keys": [
                {
                    "host": "https://a.example",
                    "workspace": "w1",
                    "key": "k1-updated",
                }
            ],
        }

    def test_delete_matches_unnormalized_stored_host(self):
        filepath = registry.setup()
        filepath.write_text(
            '{"type": "plaintext", "keys": ['
            '{"host": "https://a.example/", "workspace": "w1", "key": "k1"}]}'
        )

        assert registry.get(host="https://a.example", workspace="w1") == "k1"

        registry.delete(host="https://a.example/", workspace="w1")

        assert content() == {"type": "plaintext", "keys": []}
        registry.set(host="https://a.example", workspace="w1", api_key="k1")
        registry.delete(host="https://a.example", workspace="missing")

        assert content() == {
            "type": "plaintext",
            "keys": [
                {
                    "host": "https://a.example",
                    "workspace": "w1",
                    "key": "k1",
                }
            ],
        }


@mark.usefixtures("secret_keyring")
class TestSecret:
    def test_setup_creates_empty_registry(self, tmp_path):
        filepath = registry.setup()

        assert filepath == tmp_path / ".skore.hub" / "credentials.json"
        assert loads(filepath.read_text()) == {"type": "secret", "keys": []}
        assert_credentials_permissions(filepath)

    def test_set_does_not_store_key_on_disk(self, secret_keyring):
        registry.set(host="https://a.example", workspace="w1", api_key="k1")

        assert content() == {
            "type": "secret",
            "keys": [{"host": "https://a.example", "workspace": "w1"}],
        }
        assert secret_keyring[("skore", "https://a.example:w1")] == "k1"

    def test_get(self):
        registry.set(host="https://a.example", workspace="w1", api_key="k1")
        registry.set(host="https://a.example", workspace="w2", api_key="k2")

        assert registry.get(host="https://a.example", workspace="w1") == "k1"
        assert registry.get(host="https://a.example", workspace="w2") == "k2"

    def test_get_missing_in_keyring(self, secret_keyring):
        registry.set(host="https://a.example", workspace="w1", api_key="k1")
        secret_keyring.clear()

        assert registry.get(host="https://a.example", workspace="w1") is None

    def test_set_replaces_matching_credential(self, secret_keyring):
        registry.set(host="https://a.example", workspace="w1", api_key="k1")
        registry.set(host="https://b.example", workspace="w2", api_key="k2")
        registry.set(host="https://a.example", workspace="w1", api_key="k1-updated")

        assert content() == {
            "type": "secret",
            "keys": [
                {"host": "https://a.example", "workspace": "w1"},
                {"host": "https://b.example", "workspace": "w2"},
            ],
        }
        assert secret_keyring[("skore", "https://a.example:w1")] == "k1-updated"
        assert secret_keyring[("skore", "https://b.example:w2")] == "k2"

    def test_delete(self, secret_keyring):
        registry.set(host="https://a.example", workspace="w1", api_key="k1")
        registry.set(host="https://b.example", workspace="w2", api_key="k2")
        registry.delete(host="https://a.example", workspace="w1")

        assert content() == {
            "type": "secret",
            "keys": [{"host": "https://b.example", "workspace": "w2"}],
        }
        assert ("skore", "https://a.example:w1") not in secret_keyring
        assert secret_keyring[("skore", "https://b.example:w2")] == "k2"

    def test_host_trailing_slash_uses_same_keyring_entry(self, secret_keyring):
        registry.set(host="https://a.example/", workspace="w1", api_key="k1")

        assert registry.get(host="https://a.example", workspace="w1") == "k1"
        assert content() == {
            "type": "secret",
            "keys": [{"host": "https://a.example", "workspace": "w1"}],
        }
        assert secret_keyring[("skore", "https://a.example:w1")] == "k1"

    def test_delete_missing_in_keyring(self, secret_keyring):
        registry.set(host="https://a.example", workspace="w1", api_key="k1")
        secret_keyring.clear()

        registry.delete(host="https://a.example", workspace="w1")

        assert content() == {"type": "secret", "keys": []}
