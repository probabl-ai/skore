"""Tests for the CLI wiring."""

from __future__ import annotations

import os

import pytest

from skore._cli import skore as _skore
from skore._plugins.hub.authentication import HOST_ENV_VAR_NAME


def test_cli_exposes_builtin_commands():
    from skore._cli import cli
    from skore._cli.skills import skills

    assert {"skills", "agent", "hub", "sync"} == set(cli.commands)
    assert {"install", "list", "remove", "tools", "update"} == set(skills.commands)


def test_cli_without_subcommand_shows_plain_help():
    from click.testing import CliRunner

    from skore._cli import cli

    result = CliRunner().invoke(cli, [])

    assert result.exit_code == 0
    assert "Skore command-line interface." in result.output
    assert "agent" in result.output
    assert "hub" in result.output
    assert "skills" in result.output
    assert "Quick start:" in result.output


# --------------------------------------------------------------------------- #
# Agent detection: help output
# --------------------------------------------------------------------------- #

_AGENT_ENV_VARS = (
    "CLAUDECODE",
    "CURSOR_AGENT",
    "CURSOR_CLI",
    "GEMINI_CLI",
    "CODEX_SANDBOX",
    "PI_CODING_AGENT",
    "OPENCODE_CLIENT",
    "CI",
)


def _clear_agent_envs(monkeypatch):
    for var in _AGENT_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def test_cli_help_no_agent_shows_generic_quick_start(monkeypatch):
    from click.testing import CliRunner

    from skore._cli import cli

    _clear_agent_envs(monkeypatch)
    result = CliRunner().invoke(cli, [])

    assert result.exit_code == 0
    assert "Detected:" not in result.output
    assert "Install all skills" in result.output
    assert "Configure and launch" in result.output


def test_cli_interactive_without_subcommand_prints_help(monkeypatch):
    from click.testing import CliRunner

    from skore._cli import cli

    monkeypatch.setattr("skore._cli.is_non_interactive", lambda: False)
    result = CliRunner().invoke(cli, [])

    assert result.exit_code == 0
    assert "Usage:" in result.output


def test_module_entry_point_exposes_cli():
    import skore._cli.__main__ as entry
    from skore._cli import cli

    assert entry.cli is cli


def test_cli_help_for_agent_without_a_harness(monkeypatch):
    from click.testing import CliRunner

    from skore._cli import cli

    _clear_agent_envs(monkeypatch)
    monkeypatch.setenv("GEMINI_CLI", "1")
    result = CliRunner().invoke(cli, [])

    assert result.exit_code == 0
    assert "Detected: Gemini CLI" in result.output
    assert "Harness:" not in result.output


def test_cli_help_claude_code_detected(monkeypatch):
    from click.testing import CliRunner

    from skore._cli import cli

    _clear_agent_envs(monkeypatch)
    monkeypatch.setenv("CLAUDECODE", "1")
    result = CliRunner().invoke(cli, [])

    assert result.exit_code == 0
    assert "Detected: Claude Code" in result.output
    assert "Skills target: .claude/skills" in result.output
    assert "Harness: Claude CLI" in result.output
    assert ".claude/skills" in result.output
    assert "Configure Claude CLI with the Skore Hub provider" in result.output


def test_cli_help_cursor_detected(monkeypatch):
    from click.testing import CliRunner

    from skore._cli import cli

    _clear_agent_envs(monkeypatch)
    monkeypatch.setenv("CURSOR_AGENT", "1")
    result = CliRunner().invoke(cli, [])

    assert result.exit_code == 0
    assert "Detected: Cursor" in result.output
    assert "Skills target: .cursor/skills" in result.output
    assert "Harness: Cursor IDE" in result.output
    assert "Configure Cursor IDE with the Skore Hub provider" in result.output


def test_cli_help_opencode_detected(monkeypatch):
    from click.testing import CliRunner

    from skore._cli import cli

    _clear_agent_envs(monkeypatch)
    monkeypatch.setenv("OPENCODE_CLIENT", "1")
    result = CliRunner().invoke(cli, [])

    assert result.exit_code == 0
    assert "Detected: OpenCode" in result.output
    assert "Harness: OpenCode" in result.output
    assert "Skills target: .agents/skills" in result.output
    assert "Configure OpenCode with the Skore Hub provider" in result.output


def test_cli_help_pi_detected(monkeypatch):
    from click.testing import CliRunner

    from skore._cli import cli

    _clear_agent_envs(monkeypatch)
    monkeypatch.setenv("PI_CODING_AGENT", "true")
    result = CliRunner().invoke(cli, [])

    assert result.exit_code == 0
    assert "Detected: Pi" in result.output
    assert "Harness: Pi" in result.output
    assert "Skills target: .agents/skills" in result.output


def test_cli_subcommand_help_omits_banner():
    from click.testing import CliRunner

    from skore._cli import cli
    from skore._cli.style import SKORE_BANNER

    result = CliRunner().invoke(cli, ["skills"])

    assert result.exit_code == 0
    assert SKORE_BANNER.rstrip("\n") not in result.output
    assert "Usage" in result.output


# --------------------------------------------------------------------------- #
# Hub URL resolution
# --------------------------------------------------------------------------- #


def test_resolve_hub_uri_without_url_doesnt_set_url(monkeypatch):
    monkeypatch.delenv(HOST_ENV_VAR_NAME, raising=False)

    with pytest.raises(ValueError, match="You must provide an host"):
        _skore.resolve_hub_uri(None)

    assert HOST_ENV_VAR_NAME not in os.environ


def test_resolve_hub_uri_sets_explicit_url(monkeypatch):
    monkeypatch.delenv(HOST_ENV_VAR_NAME, raising=False)
    monkeypatch.setattr(_skore, "_discover_api_url", lambda url: None)

    assert _skore.resolve_hub_uri("https://hub.test") == "https://hub.test"
    assert os.environ[HOST_ENV_VAR_NAME] == "https://hub.test"


def test_resolve_hub_uri_discovers_api_url_from_frontend(monkeypatch):
    monkeypatch.delenv(HOST_ENV_VAR_NAME, raising=False)
    monkeypatch.setattr(
        _skore, "_discover_api_url", lambda url: "https://api.discovered.test"
    )

    assert (
        _skore.resolve_hub_uri("https://frontend.test") == "https://api.discovered.test"
    )
    assert os.environ[HOST_ENV_VAR_NAME] == "https://api.discovered.test"


def test_discover_api_url_returns_api_url_on_success(monkeypatch):
    import json

    class FakeResponse:
        status = 200

        def read(self):
            return json.dumps({"api_url": "https://api.hub.test"}).encode()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    urls = []

    def fake_urlopen(url, *, timeout):
        urls.append(url)
        return FakeResponse()

    monkeypatch.setattr(_skore.urllib.request, "urlopen", fake_urlopen)

    assert _skore._discover_api_url("https://frontend.test") == "https://api.hub.test"
    assert urls == ["https://frontend.test/.well-known/skore-hub.json"]


def test_discover_api_url_returns_none_on_404(monkeypatch):
    import io
    from urllib.error import HTTPError

    body = io.BytesIO()

    def fake_urlopen(url, *, timeout):
        raise HTTPError(url, 404, "Not Found", {}, body)

    monkeypatch.setattr(_skore.urllib.request, "urlopen", fake_urlopen)

    assert _skore._discover_api_url("https://frontend.test") is None
    assert body.closed


def test_discover_api_url_returns_none_on_non_200(monkeypatch):
    class FakeResponse:
        status = 404

        def read(self):
            return b""

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    monkeypatch.setattr(
        _skore.urllib.request, "urlopen", lambda url, *, timeout: FakeResponse()
    )

    assert _skore._discover_api_url("https://frontend.test") is None


def test_discover_api_url_returns_none_on_network_error(monkeypatch):
    from urllib.error import URLError

    def fake_urlopen(url, *, timeout):
        raise URLError("no route")

    monkeypatch.setattr(_skore.urllib.request, "urlopen", fake_urlopen)

    assert _skore._discover_api_url("https://frontend.test") is None


def test_discover_api_url_returns_none_when_json_missing_api_url(monkeypatch):
    import json

    class FakeResponse:
        status = 200

        def read(self):
            return json.dumps({"other": "data"}).encode()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    monkeypatch.setattr(
        _skore.urllib.request, "urlopen", lambda url, *, timeout: FakeResponse()
    )

    assert _skore._discover_api_url("https://frontend.test") is None
