"""Tests for the ``skore hub`` command group."""

from __future__ import annotations

import os

import pytest
from click.testing import CliRunner

from skore._cli import cli
from skore._plugins.hub.authentication import key as hub_key
from skore._plugins.hub.authentication.registry import local

pytestmark = pytest.mark.usefixtures("monkeypatch_home", "monkeypatch_keyring")


def _invoke(args: list[str]):
    return CliRunner().invoke(cli, args)


def _patch_generate(monkeypatch):
    calls = []

    def generate(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(hub_key, "generate", generate)
    return calls


def test_hub_no_subcommand_shows_help():
    result = _invoke(["hub"])

    assert result.exit_code == 0
    assert "api-key" in result.output


def test_hub_api_key_no_subcommand_shows_help():
    result = _invoke(["hub", "api-key"])

    assert result.exit_code == 0
    assert "generate" in result.output
    assert "revoke" in result.output
    assert "list" in result.output


def test_hub_api_key_revoke_requires_workspace():
    result = _invoke(["hub", "api-key", "revoke"])

    assert result.exit_code != 0
    assert "workspace" in result.output.lower()


def test_hub_api_key_revoke_stored_key(monkeypatch):
    rows = [(7, "http://hub.test", "team")]
    revoked = []

    def keys():
        yield from rows

    def revoke(*, host, workspace, timeout):
        revoked.append((host, workspace, timeout, os.environ.get("SKORE_HUB_URI")))
        rows.clear()

    monkeypatch.setattr(hub_key, "keys", keys)
    monkeypatch.setattr(hub_key, "revoke", revoke)

    result = _invoke(
        [
            "hub",
            "api-key",
            "revoke",
            "--host=http://hub.test",
            "--workspace=team",
            "--login-timeout=42",
        ]
    )

    assert result.exit_code == 0, result.output
    assert "No API key stored." not in result.output
    assert revoked == [("http://hub.test", "team", 42, "http://hub.test")]


def test_hub_api_key_revoke_without_host_forwards_none(monkeypatch):
    revoked = []
    monkeypatch.setattr(hub_key, "keys", lambda: iter(()))
    monkeypatch.setattr(
        hub_key,
        "revoke",
        lambda *, host, workspace, timeout: revoked.append((host, workspace, timeout)),
    )

    result = _invoke(["hub", "api-key", "revoke", "--workspace=team"])

    assert result.exit_code == 0, result.output
    assert revoked == [(None, "team", 600)]
    assert "No API key stored." in result.output


def test_hub_api_key_revoke_reports_when_nothing_stored():
    result = _invoke(
        ["hub", "api-key", "revoke", "--host=http://hub.test", "--workspace=team"]
    )

    assert result.exit_code == 0, result.output
    assert "No API key stored." in result.output


def test_hub_api_key_list_empty():
    result = _invoke(["hub", "api-key", "list"])

    assert result.exit_code == 0
    assert "No API keys stored." in result.output


def test_hub_api_key_list_shows_host_and_workspace():
    local.set(id=1, host="http://a.test", workspace="w1", key="k1")
    local.set(id=2, host="http://b.test", workspace="w2", key="k2")

    result = _invoke(["hub", "api-key", "list"])

    assert result.exit_code == 0
    assert "http://a.test" in result.output
    assert "w1" in result.output
    assert "http://b.test" in result.output
    assert "w2" in result.output
    assert "k1" not in result.output
    assert "k2" not in result.output


def test_hub_api_key_generate_requires_workspace():
    result = _invoke(["hub", "api-key", "generate"])

    assert result.exit_code != 0
    assert "workspace" in result.output.lower()


def test_hub_api_key_generate_delegates_to_skore(monkeypatch):
    calls = _patch_generate(monkeypatch)

    result = _invoke(
        [
            "hub",
            "api-key",
            "generate",
            "--host=http://hub.test",
            "--workspace=team",
            "--login-timeout=42",
        ]
    )

    assert result.exit_code == 0, result.output
    assert calls == [
        {
            "host": "http://hub.test",
            "workspace": "team",
            "name": None,
            "expires": "never",
            "timeout": 42,
            "force": False,
        }
    ]
    assert os.environ["SKORE_HUB_URI"] == "http://hub.test"
    assert "expires " not in result.output
    assert "team" in result.output


def test_hub_api_key_generate_errors_when_key_already_stored():
    local.set(id=9, host="http://hub.test", workspace="team", key="old-secret")

    result = _invoke(
        ["hub", "api-key", "generate", "--host=http://hub.test", "--workspace=team"]
    )

    assert result.exit_code != 0
    assert "already stored" in result.output
    assert "--force" in result.output
    assert hub_key.get(host="http://hub.test", workspace="team") == "old-secret"


def test_hub_api_key_generate_forwards_force(monkeypatch):
    calls = _patch_generate(monkeypatch)

    result = _invoke(["hub", "api-key", "generate", "--workspace=team", "--force"])

    assert result.exit_code == 0, result.output
    assert calls[0]["force"] is True
    assert calls[0]["host"] is None
    assert calls[0]["workspace"] == "team"


def test_hub_api_key_generate_name_override(monkeypatch):
    calls = _patch_generate(monkeypatch)

    result = _invoke(
        ["hub", "api-key", "generate", "--workspace=team", "--name=ui-laptop"]
    )

    assert result.exit_code == 0, result.output
    assert calls[0]["name"] == "ui-laptop"


def test_hub_api_key_generate_expires_in_three_months(monkeypatch):
    calls = _patch_generate(monkeypatch)

    result = _invoke(
        ["hub", "api-key", "generate", "--workspace=team", "--expires", "3"]
    )

    assert result.exit_code == 0, result.output
    assert calls[0]["expires"] == "3"
    assert "expires in 3 months" in result.output


def test_hub_api_key_generate_rejects_unknown_expires():
    result = _invoke(
        ["hub", "api-key", "generate", "--workspace=team", "--expires", "12"]
    )

    assert result.exit_code != 0
    assert "expires" in result.output.lower()


def test_hub_api_key_generate_unknown_workspace(monkeypatch):
    def generate(**kwargs):
        raise PermissionError("You are not member of 'other'")

    monkeypatch.setattr(hub_key, "generate", generate)

    result = _invoke(["hub", "api-key", "generate", "--workspace=other"])

    assert result.exit_code != 0
    assert "not member" in result.output
