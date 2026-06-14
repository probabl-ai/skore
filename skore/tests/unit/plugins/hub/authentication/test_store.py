import json
import os
import stat
import sys
from pathlib import Path

from pytest import fixture, mark

from skore._plugins.hub.authentication import store


@fixture
def credentials(monkeypatch, tmp_path):
    """Point the token cache at an isolated file inside ``tmp_path``."""
    file = tmp_path / "hub.json"
    monkeypatch.setenv("SKORE_HUB_CREDENTIALS", str(file))
    return file


def test_path_honors_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("SKORE_HUB_CREDENTIALS", str(tmp_path / "creds.json"))

    assert store.path() == tmp_path / "creds.json"


def test_path_expands_user(monkeypatch):
    monkeypatch.setenv("SKORE_HUB_CREDENTIALS", "~/somewhere/hub.json")

    result = store.path()

    assert "~" not in str(result)
    assert result == Path.home() / "somewhere" / "hub.json"


def test_path_default_location(monkeypatch, tmp_path):
    monkeypatch.delenv("SKORE_HUB_CREDENTIALS", raising=False)
    monkeypatch.setattr("platformdirs.user_config_dir", lambda app: str(tmp_path / app))

    assert store.path() == tmp_path / "skore" / "hub.json"


def test_load_missing_file_returns_none(credentials):
    assert not credentials.exists()
    assert store.load() is None


def test_load_invalid_json_returns_none(credentials):
    credentials.write_text("{not valid json")

    assert store.load() is None


def test_load_non_dict_json_returns_none(credentials):
    credentials.write_text("[]")

    assert store.load() is None


def test_save_then_load_roundtrip(credentials):
    token = {
        "uri": "https://hub",
        "access_token": "A",
        "refresh_token": "B",
        "expires_at": "C",
    }

    saved = store.save(token)

    assert saved == credentials
    assert store.load() == token


def test_save_creates_missing_parent_dirs(monkeypatch, tmp_path):
    file = tmp_path / "nested" / "dir" / "hub.json"
    monkeypatch.setenv("SKORE_HUB_CREDENTIALS", str(file))

    store.save({"access_token": "A"})

    assert file.is_file()


@mark.skipif(sys.platform == "win32", reason="POSIX file permissions")
def test_save_sets_owner_only_permissions(credentials):
    store.save({"access_token": "A"})

    mode = stat.S_IMODE(os.stat(credentials).st_mode)
    assert mode == 0o600


def test_clear_removes_existing_file(credentials):
    store.save({"access_token": "A"})
    assert credentials.is_file()

    cleared = store.clear()

    assert cleared == credentials
    assert not credentials.exists()


def test_clear_when_absent_returns_none(credentials):
    assert not credentials.exists()

    assert store.clear() is None


def test_save_writes_indented_json_with_trailing_newline(credentials):
    store.save({"access_token": "A"})

    content = credentials.read_text()
    assert content.endswith("\n")
    assert json.loads(content) == {"access_token": "A"}
