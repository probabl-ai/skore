"""Tests for the ``skore agent`` command and ``.skore`` persistence."""

from __future__ import annotations

import json
import os
import re
from types import SimpleNamespace

import pytest
import rich_click as click
from click.testing import CliRunner

from skore._cli.agent import app as _agent_app
from skore._cli.agent import commands as _commands
from skore._cli.agent.commands import agent
from skore._cli.agents import AGENTS, DEFAULT_MODEL_ID, HARNESS_NAMES
from skore._cli.credentials import HubError
from skore._cli.harnesses import bob as _bob
from skore._cli.harnesses import claude as _claude
from skore._cli.harnesses import process as _process
from skore._cli.harnesses.context import HarnessContext
from skore._cli.project import (
    SKORE_FILENAME,
    SkoreConfig,
    SkoreConfigError,
    ensure_gitignore_entry,
)
from skore._plugins.hub.authentication import key as hub_key
from skore._plugins.hub.authentication import login as hub_login
from skore._plugins.hub.authentication.registry import distant
from skore._plugins.hub.authentication.registry.distant import Workspace

_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")


def _plain_output(output: str) -> str:
    """Strip ANSI codes from rich-click panels for stable assertions."""
    return _ANSI_ESCAPE.sub("", output)


def _mock_harness_on_path(monkeypatch, name: str) -> None:
    """Pretend the harness binary is installed without relying on the real PATH."""
    monkeypatch.setattr(
        _process.shutil,
        "which",
        lambda cmd: f"/usr/bin/{cmd}" if cmd == name else None,
    )


def _membership(public_id: str = "ws-1", workspace_id: int = 1):
    return Workspace(id=workspace_id, public_id=public_id, permissions=frozenset())


def _stub_login(monkeypatch):
    monkeypatch.setattr(
        hub_login,
        "login",
        lambda *, host, timeout: SimpleNamespace(access="tok"),
    )


def _stub_identity(monkeypatch, workspaces):
    monkeypatch.setattr(
        distant,
        "identity",
        lambda *, host, token: SimpleNamespace(workspaces=workspaces),
    )


@pytest.fixture(autouse=True)
def stub_registry(monkeypatch):
    """Serve a stored API key so runs never reach the real credential registry."""
    monkeypatch.setattr(hub_key, "get", lambda **kwargs: "new-secret")


@pytest.fixture(autouse=True)
def default_hub_uri(monkeypatch):
    """Seed the hub URI so first-run `URI()` does not hit the public hub."""
    monkeypatch.setenv("SKORE_HUB_URI", "http://hub.test")


def _stub_resolve_hub_uri(monkeypatch):
    monkeypatch.setattr(
        _commands,
        "resolve_hub_uri",
        lambda url, *args, **kwargs: url or "http://hub.test",
    )


def _write_skore(directory, **overrides):
    payload = {
        "hub_url": "http://hub.test",
        "workspace": "ws-1",
        "workspace_id": 1,
        "harness": "opencode",
    }
    payload.update(overrides)
    (directory / SKORE_FILENAME).write_text(json.dumps(payload) + "\n")
    return payload


def test_skore_config_round_trip(tmp_path):
    config = SkoreConfig(
        hub_url="http://hub.test",
        workspace="ws-1",
        workspace_id=1,
        harness="pi",
    )
    path = config.save(tmp_path)
    loaded = SkoreConfig.load(tmp_path)
    assert path.name == SKORE_FILENAME
    assert loaded == config


def test_skore_config_load_does_not_set_hub_uri_env(tmp_path, monkeypatch):
    monkeypatch.delenv("SKORE_HUB_URI", raising=False)
    _write_skore(tmp_path)
    loaded = SkoreConfig.load(tmp_path)
    assert loaded is not None
    assert "SKORE_HUB_URI" not in os.environ


def test_skore_config_load_invalid_raises(tmp_path):
    (tmp_path / SKORE_FILENAME).write_text("{ not json")
    with pytest.raises(SkoreConfigError, match="could not parse"):
        SkoreConfig.load(tmp_path)


def test_skore_config_load_normalizes_legacy_claude_code_harness(tmp_path):
    _write_skore(tmp_path, harness="claude-code")
    loaded = SkoreConfig.load(tmp_path)
    assert loaded is not None
    assert loaded.harness == "claude"


def test_skore_config_load_returns_none_when_absent(tmp_path):
    assert SkoreConfig.load(tmp_path) is None


def test_skore_config_load_rejects_missing_required_field(tmp_path):
    (tmp_path / SKORE_FILENAME).write_text(
        json.dumps({"hub_url": "http://hub.test"}) + "\n"
    )
    with pytest.raises(SkoreConfigError, match="missing hub_url"):
        SkoreConfig.load(tmp_path)


def test_skore_config_load_rejects_non_object(tmp_path):
    (tmp_path / SKORE_FILENAME).write_text("[]\n")
    with pytest.raises(SkoreConfigError, match="JSON object"):
        SkoreConfig.load(tmp_path)


def test_skore_config_load_rejects_non_integer_workspace_id(tmp_path):
    (tmp_path / SKORE_FILENAME).write_text(
        json.dumps(
            {"hub_url": "http://hub.test", "workspace": "ws-1", "workspace_id": "nope"}
        )
    )
    with pytest.raises(SkoreConfigError, match="not an integer"):
        SkoreConfig.load(tmp_path)


def test_agent_refuses_unreadable_skore_without_replacing_it(tmp_path, monkeypatch):
    original = "{ not json"
    (tmp_path / SKORE_FILENAME).write_text(original)
    monkeypatch.setattr(
        hub_login, "login", lambda **kwargs: (_ for _ in ()).throw(AssertionError)
    )

    result = CliRunner().invoke(agent, ["--workspace", str(tmp_path)])

    assert result.exit_code != 0
    assert "could not parse" in result.output
    assert (tmp_path / SKORE_FILENAME).read_text() == original


def test_skore_config_save_omits_the_api_key(tmp_path):
    SkoreConfig(hub_url="http://hub.test", workspace="ws-1", workspace_id=1).save(
        tmp_path
    )
    payload = json.loads((tmp_path / SKORE_FILENAME).read_text())
    assert "api_key" not in payload


def test_skore_config_save_omits_none_harness(tmp_path):
    config = SkoreConfig(
        hub_url="http://hub.test",
        workspace="ws-1",
        workspace_id=1,
    )
    config.save(tmp_path)
    payload = json.loads((tmp_path / SKORE_FILENAME).read_text())
    assert "harness" not in payload


def test_ensure_gitignore_appends_to_existing_file(tmp_path):
    (tmp_path / ".gitignore").write_text("*.log\n__pycache__/\n")
    ensure_gitignore_entry(tmp_path)
    lines = (tmp_path / ".gitignore").read_text().splitlines()
    assert lines[-1] == ".skore"
    assert "*.log" in lines


def test_ensure_gitignore_inserts_blank_line_when_missing_trailing_newline(tmp_path):
    (tmp_path / ".gitignore").write_text("*.log")
    ensure_gitignore_entry(tmp_path)
    assert (tmp_path / ".gitignore").read_text() == "*.log\n\n.skore\n"


def test_ensure_gitignore_keeps_existing_trailing_blank_line(tmp_path):
    (tmp_path / ".gitignore").write_text("*.log\n\n")
    ensure_gitignore_entry(tmp_path)
    assert (tmp_path / ".gitignore").read_text() == "*.log\n\n.skore\n"


def test_ensure_gitignore_appends_entry(tmp_path):
    ensure_gitignore_entry(tmp_path)
    assert (tmp_path / ".gitignore").read_text().strip() == ".skore"

    ensure_gitignore_entry(tmp_path)
    assert (tmp_path / ".gitignore").read_text().count(".skore") == 1


def test_opencode_writer_embeds_api_key(tmp_path):
    AGENTS["opencode"].configure(
        HarnessContext(
            workspace=tmp_path,
            hub_url="http://hub.test",
            api_key="secret-key",
        )
    )
    config = json.loads((tmp_path / "opencode.json").read_text())
    provider = config["provider"]["skore"]
    assert config["model"] == "skore/skore-agent"
    assert provider["options"]["baseURL"] == "http://hub.test/v1"
    assert provider["options"]["apiKey"] == "secret-key"


def test_agent_nonexistent_workspace_errors(tmp_path):
    missing = tmp_path / "missing"
    result = CliRunner().invoke(agent, ["--workspace", str(missing)])
    assert result.exit_code != 0
    assert "workspace does not exist" in result.output


def test_agent_invalid_harness_lists_all_supported_harnesses(tmp_path):
    result = CliRunner().invoke(
        agent, ["--workspace", str(tmp_path), "--harness", "does-not-exist"]
    )
    assert result.exit_code != 0
    output = _plain_output(result.output)
    for name in HARNESS_NAMES:
        assert name in output


def test_agent_reports_hub_errors(tmp_path, monkeypatch):
    _write_skore(tmp_path)

    def boom(**kwargs):
        raise HubError("mint failed")

    monkeypatch.setattr(_commands, "require_workspace_api_key", boom)

    result = CliRunner().invoke(
        agent, ["--workspace", str(tmp_path), "--harness", "opencode"]
    )

    assert result.exit_code != 0
    assert "mint failed" in result.output
    assert "Traceback" not in result.output


def test_agent_reports_configure_errors(tmp_path, monkeypatch):
    _write_skore(tmp_path, harness="opencode")

    def boom(ctx):
        raise RuntimeError("configure broke")

    monkeypatch.setattr(
        _commands,
        "get_harness",
        lambda name: SimpleNamespace(
            harness_name=name,
            harness_display_name="OpenCode",
            configure=boom,
        ),
    )

    result = CliRunner().invoke(
        agent, ["--workspace", str(tmp_path), "--harness", "opencode"]
    )

    assert result.exit_code != 0
    assert "configure broke" in result.output
    assert "Traceback" not in result.output


def test_agent_reports_launch_errors(tmp_path, monkeypatch):
    _write_skore(tmp_path, harness="opencode")
    _mock_harness_on_path(monkeypatch, "opencode")

    def boom(*args, **kwargs):
        raise RuntimeError("launch broke")

    monkeypatch.setattr(_commands, "launch_harness", boom)

    result = CliRunner().invoke(
        agent, ["--workspace", str(tmp_path), "--harness", "opencode"]
    )

    assert result.exit_code != 0
    assert "launch broke" in result.output
    assert "Traceback" not in result.output


def test_agent_uses_existing_skore_config(tmp_path, monkeypatch):
    _write_skore(tmp_path)
    _mock_harness_on_path(monkeypatch, "opencode")
    launched: list[str] = []
    monkeypatch.setattr(
        _commands,
        "launch_harness",
        lambda selected, workspace, model_id=DEFAULT_MODEL_ID, **k: launched.append(
            selected.harness_name
        ),
    )

    result = CliRunner().invoke(
        agent,
        ["--workspace", str(tmp_path), "--harness", "opencode"],
    )

    assert result.exit_code == 0, result.output
    assert launched == ["opencode"]
    assert json.loads((tmp_path / "opencode.json").read_text())["provider"]["skore"]


def test_agent_hub_url_param_updates_skore_file(tmp_path, monkeypatch):
    """Passing a different --hub-url rewrites .skore for that hub."""
    _write_skore(tmp_path)
    _mock_harness_on_path(monkeypatch, "opencode")
    _stub_resolve_hub_uri(monkeypatch)
    _stub_login(monkeypatch)
    _stub_identity(monkeypatch, [_membership()])
    monkeypatch.setattr(
        _commands,
        "launch_harness",
        lambda selected, workspace, model_id=DEFAULT_MODEL_ID, **k: None,
    )

    result = CliRunner().invoke(
        agent,
        [
            "--workspace",
            str(tmp_path),
            "--hub-url",
            "http://other.hub",
            "--harness",
            "opencode",
        ],
    )

    assert result.exit_code == 0, result.output
    saved = json.loads((tmp_path / SKORE_FILENAME).read_text())
    assert saved["hub_url"] == "http://other.hub"
    assert saved["workspace"] == "ws-1"
    assert "api_key" not in saved
    config = json.loads((tmp_path / "opencode.json").read_text())
    assert config["provider"]["skore"]["options"]["baseURL"] == "http://other.hub/v1"


def test_agent_hub_url_switch_uses_new_workspace_membership(tmp_path, monkeypatch):
    """Switching hubs drops the saved workspace and resolves membership anew."""
    _write_skore(tmp_path, workspace="ws-old", workspace_id=1)
    _mock_harness_on_path(monkeypatch, "opencode")
    _stub_resolve_hub_uri(monkeypatch)
    _stub_login(monkeypatch)
    _stub_identity(monkeypatch, [_membership("ws-new", workspace_id=9)])
    monkeypatch.setattr(
        _commands,
        "launch_harness",
        lambda selected, workspace, model_id=DEFAULT_MODEL_ID, **k: None,
    )

    result = CliRunner().invoke(
        agent,
        [
            "--workspace",
            str(tmp_path),
            "--hub-url",
            "http://other.hub",
            "--harness",
            "opencode",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "is not in your memberships" not in _plain_output(result.output)
    saved = json.loads((tmp_path / SKORE_FILENAME).read_text())
    assert saved["hub_url"] == "http://other.hub"
    assert saved["workspace"] == "ws-new"
    assert saved["workspace_id"] == 9
    assert "api_key" not in saved


def test_agent_hub_url_switch_asks_for_workspace(tmp_path, monkeypatch):
    """Switching hubs picks a new workspace interactively across memberships."""
    _write_skore(tmp_path, workspace="ws-old", workspace_id=1)
    _mock_harness_on_path(monkeypatch, "opencode")
    _stub_resolve_hub_uri(monkeypatch)
    _stub_login(monkeypatch)
    _stub_identity(
        monkeypatch,
        [_membership("ws-a"), _membership("ws-b", workspace_id=2)],
    )
    monkeypatch.setattr(_commands, "is_non_interactive", lambda: False)
    monkeypatch.setattr(_commands, "_pick_workspace", lambda m: m[1])
    monkeypatch.setattr(
        _commands,
        "launch_harness",
        lambda selected, workspace, model_id=DEFAULT_MODEL_ID, **k: None,
    )

    result = CliRunner().invoke(
        agent,
        [
            "--workspace",
            str(tmp_path),
            "--hub-url",
            "http://other.hub",
            "--harness",
            "opencode",
        ],
    )

    assert result.exit_code == 0, result.output
    saved = json.loads((tmp_path / SKORE_FILENAME).read_text())
    assert saved["workspace"] == "ws-b"
    assert "api_key" not in saved


def test_agent_hub_url_param_matching_saved_reuses_config(tmp_path, monkeypatch):
    """Passing --hub-url equal to the saved one reuses the config as-is."""
    _write_skore(tmp_path)
    _mock_harness_on_path(monkeypatch, "opencode")
    logins = []
    monkeypatch.setattr(
        hub_login, "login", lambda *, host, timeout: logins.append(host)
    )
    monkeypatch.setattr(
        _commands,
        "launch_harness",
        lambda selected, workspace, model_id=DEFAULT_MODEL_ID, **k: None,
    )

    result = CliRunner().invoke(
        agent,
        [
            "--workspace",
            str(tmp_path),
            "--hub-url",
            "http://hub.test",
            "--harness",
            "opencode",
        ],
    )

    assert result.exit_code == 0, result.output
    assert logins == []
    saved = json.loads((tmp_path / SKORE_FILENAME).read_text())
    assert saved["hub_url"] == "http://hub.test"
    assert "api_key" not in saved


def test_agent_hub_url_trailing_slash_reuses_config(tmp_path, monkeypatch):
    """A --hub-url equal modulo a trailing slash reuses the saved config."""
    _write_skore(tmp_path)
    _mock_harness_on_path(monkeypatch, "opencode")
    logins = []
    monkeypatch.setattr(
        hub_login, "login", lambda *, host, timeout: logins.append(host)
    )
    monkeypatch.setattr(
        _commands,
        "launch_harness",
        lambda selected, workspace, model_id=DEFAULT_MODEL_ID, **k: None,
    )

    result = CliRunner().invoke(
        agent,
        [
            "--workspace",
            str(tmp_path),
            "--hub-url",
            "http://hub.test/",
            "--harness",
            "opencode",
        ],
    )

    assert result.exit_code == 0, result.output
    assert logins == []
    saved = json.loads((tmp_path / SKORE_FILENAME).read_text())
    assert saved["hub_url"] == "http://hub.test"
    assert "api_key" not in saved


def test_agent_hub_switch_non_interactive_reuses_saved_harness(tmp_path, monkeypatch):
    """Switching hubs non-interactively without --harness keeps the saved one."""
    _write_skore(tmp_path)
    _mock_harness_on_path(monkeypatch, "opencode")
    _stub_resolve_hub_uri(monkeypatch)
    _stub_login(monkeypatch)
    _stub_identity(monkeypatch, [_membership()])
    monkeypatch.setattr(_commands, "is_non_interactive", lambda: True)
    monkeypatch.setattr(_commands, "detect_agent", lambda: None)
    monkeypatch.setattr(
        _commands,
        "launch_harness",
        lambda selected, workspace, model_id=DEFAULT_MODEL_ID, **k: None,
    )

    result = CliRunner().invoke(
        agent,
        ["--workspace", str(tmp_path), "--hub-url", "http://other.hub"],
    )

    assert result.exit_code == 0, result.output
    saved = json.loads((tmp_path / SKORE_FILENAME).read_text())
    assert saved["hub_url"] == "http://other.hub"
    assert saved["harness"] == "opencode"
    assert "api_key" not in saved


def test_agent_creates_skore_on_first_run(tmp_path, monkeypatch):
    _mock_harness_on_path(monkeypatch, "opencode")
    _stub_login(monkeypatch)
    _stub_identity(monkeypatch, [_membership()])
    monkeypatch.setattr(
        _commands,
        "launch_harness",
        lambda selected, workspace, model_id=DEFAULT_MODEL_ID, **k: None,
    )

    result = CliRunner().invoke(
        agent,
        ["--workspace", str(tmp_path), "--harness", "opencode"],
    )

    assert result.exit_code == 0, result.output
    saved = json.loads((tmp_path / SKORE_FILENAME).read_text())
    assert saved["workspace"] == "ws-1"
    assert saved["hub_url"] == "http://hub.test"
    assert "api_key" not in saved
    assert ".skore" in (tmp_path / ".gitignore").read_text().splitlines()


def test_agent_saves_claude_plugin_ide(tmp_path, monkeypatch):
    cursor = tmp_path / "cursor"
    vscode = tmp_path / "vscode"
    (cursor / "anthropic.claude-code-1.0.0").mkdir(parents=True)
    (vscode / "anthropic.claude-code-1.0.0").mkdir(parents=True)
    monkeypatch.setattr(
        _claude,
        "ide_extension_hosts",
        lambda: (
            ("cursor", cursor, "cursor"),
            ("code", vscode, "vscode"),
        ),
    )
    monkeypatch.setattr(_claude, "is_non_interactive", lambda: False)
    monkeypatch.setattr(_claude, "_prompt_ide", lambda options: "cursor")
    _stub_login(monkeypatch)
    _stub_identity(monkeypatch, [_membership()])
    monkeypatch.setattr(
        _commands,
        "launch_harness",
        lambda selected, workspace, model_id=DEFAULT_MODEL_ID, **k: None,
    )

    result = CliRunner().invoke(
        agent,
        ["--workspace", str(tmp_path), "--harness", "claude-plugin"],
    )

    assert result.exit_code == 0, result.output
    saved = json.loads((tmp_path / SKORE_FILENAME).read_text())
    assert saved["harness"] == "claude-plugin-cursor"


def test_agent_claude_plugin_ide_flag_skips_the_menu(tmp_path, monkeypatch):
    cursor = tmp_path / "cursor"
    vscode = tmp_path / "vscode"
    (cursor / "anthropic.claude-code-1.0.0").mkdir(parents=True)
    (vscode / "anthropic.claude-code-1.0.0").mkdir(parents=True)
    monkeypatch.setattr(
        _claude,
        "ide_extension_hosts",
        lambda: (
            ("cursor", cursor, "cursor"),
            ("code", vscode, "vscode"),
        ),
    )
    prompted: list[object] = []
    monkeypatch.setattr(
        _claude, "_prompt_ide", lambda options: prompted.append(options)
    )
    _stub_login(monkeypatch)
    _stub_identity(monkeypatch, [_membership()])
    monkeypatch.setattr(
        _commands,
        "launch_harness",
        lambda selected, workspace, model_id=DEFAULT_MODEL_ID, **k: None,
    )

    result = CliRunner().invoke(
        agent,
        ["--workspace", str(tmp_path), "--harness", "claude-plugin-cursor"],
    )

    assert result.exit_code == 0, result.output
    assert prompted == []
    saved = json.loads((tmp_path / SKORE_FILENAME).read_text())
    assert saved["harness"] == "claude-plugin-cursor"


def test_agent_reuses_saved_claude_plugin_ide(tmp_path, monkeypatch):
    cursor = tmp_path / "cursor"
    vscode = tmp_path / "vscode"
    (cursor / "anthropic.claude-code-1.0.0").mkdir(parents=True)
    (vscode / "anthropic.claude-code-1.0.0").mkdir(parents=True)
    _write_skore(tmp_path, harness="claude-plugin-cursor")
    monkeypatch.setattr(
        _claude,
        "ide_extension_hosts",
        lambda: (
            ("cursor", cursor, "cursor"),
            ("code", vscode, "vscode"),
        ),
    )
    prompted: list[object] = []
    monkeypatch.setattr(
        _claude, "_prompt_ide", lambda options: prompted.append(options)
    )
    launched: list[str | None] = []
    monkeypatch.setattr(
        _commands,
        "launch_harness",
        lambda selected, workspace, model_id=DEFAULT_MODEL_ID, **k: launched.append(
            selected.harness_name
        ),
    )

    result = CliRunner().invoke(agent, ["--workspace", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert prompted == []
    assert launched == ["claude-plugin"]
    saved = json.loads((tmp_path / SKORE_FILENAME).read_text())
    assert saved["harness"] == "claude-plugin-cursor"


def test_agent_accepts_claude_cli_alias(tmp_path, monkeypatch):
    _write_skore(tmp_path, harness="opencode")
    _mock_harness_on_path(monkeypatch, "claude")
    launched: list[str] = []
    monkeypatch.setattr(
        _commands,
        "launch_harness",
        lambda selected, workspace, model_id=DEFAULT_MODEL_ID, **k: launched.append(
            selected.harness_name
        ),
    )

    result = CliRunner().invoke(
        agent,
        ["--workspace", str(tmp_path), "--harness", "claude-cli"],
    )

    assert result.exit_code == 0, result.output
    assert launched == ["claude"]
    assert json.loads((tmp_path / SKORE_FILENAME).read_text())["harness"] == "claude"


@pytest.mark.parametrize("harness_flag", ["bob-ide", "bobide"])
def test_agent_bob_ide_first_run_on_linux_writes_mcp_config(
    tmp_path, monkeypatch, harness_flag
):
    """Bob IDE on Linux installs a ``bobide`` binary on PATH (not ``bob-ide``);
    the run must detect it and still write ``.bob/mcp.json`` after saving
    ``.skore`` (skore-hub#1894). ``--harness bobide`` is accepted as an alias
    and stored under the canonical ``bob-ide`` name."""
    monkeypatch.setattr(_process.sys, "platform", "linux")
    _mock_harness_on_path(monkeypatch, "bobide")
    _stub_login(monkeypatch)
    _stub_identity(monkeypatch, [_membership()])
    monkeypatch.setattr(
        _commands,
        "launch_harness",
        lambda selected, workspace, model_id=DEFAULT_MODEL_ID, **k: None,
    )

    result = CliRunner().invoke(
        agent, ["--workspace", str(tmp_path), "--harness", harness_flag]
    )

    assert result.exit_code == 0, result.output
    assert "not installed or not on PATH" not in _plain_output(result.output)
    saved = json.loads((tmp_path / SKORE_FILENAME).read_text())
    assert saved["harness"] == "bob-ide"
    mcp = json.loads((tmp_path / ".bob" / "mcp.json").read_text())
    assert mcp["mcpServers"]["skore"]["url"] == "http://hub.test/mcp"


def test_agent_non_interactive_without_harness_errors(tmp_path, monkeypatch):
    _clear_agent_envs(monkeypatch)
    _stub_login(monkeypatch)
    _stub_identity(monkeypatch, [_membership()])
    monkeypatch.setattr(_commands, "is_non_interactive", lambda: True)

    result = CliRunner().invoke(agent, ["--workspace", str(tmp_path)])

    assert result.exit_code != 0
    assert "pass --harness" in _plain_output(result.output)


def test_agent_generates_api_key_when_none_is_stored(tmp_path, monkeypatch):
    _mock_harness_on_path(monkeypatch, "opencode")
    _stub_login(monkeypatch)
    _stub_identity(monkeypatch, [_membership()])
    monkeypatch.setattr(
        _commands,
        "launch_harness",
        lambda selected, workspace, model_id=DEFAULT_MODEL_ID: None,
    )

    stored: dict[tuple[str, str], str] = {}
    generated = []

    def fake_generate(*, host, workspace, name, expires, timeout, force):
        generated.append((host, workspace))
        stored[(host, workspace)] = "minted-secret"

    monkeypatch.setattr(
        hub_key,
        "get",
        lambda *, host, workspace: stored.get((host, workspace)),
    )
    monkeypatch.setattr(hub_key, "generate", fake_generate)

    result = CliRunner().invoke(
        agent, ["--workspace", str(tmp_path), "--harness", "opencode"]
    )

    assert result.exit_code == 0, result.output
    assert generated == [("http://hub.test", "ws-1")]
    provider = json.loads((tmp_path / "opencode.json").read_text())["provider"]
    assert provider["skore"]["options"]["apiKey"] == "minted-secret"


# --------------------------------------------------------------------------- #
# is_non_interactive
# --------------------------------------------------------------------------- #


def test_is_non_interactive_with_tty(monkeypatch):
    from skore._cli import runtime as _runtime

    _clear_agent_envs(monkeypatch)
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.setattr(_runtime.sys, "stdin", SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr(_runtime.sys, "stdout", SimpleNamespace(isatty=lambda: True))
    assert _commands.is_non_interactive() is False


def test_is_non_interactive_without_tty(monkeypatch):
    from skore._cli import runtime as _runtime

    _clear_agent_envs(monkeypatch)
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.setattr(_runtime.sys, "stdin", SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr(_runtime.sys, "stdout", SimpleNamespace(isatty=lambda: False))
    assert _commands.is_non_interactive() is True


# --------------------------------------------------------------------------- #
# _pick_workspace
# --------------------------------------------------------------------------- #


class _FakePicker:
    """Stand-in for the Textual pickers: ``run()`` is a no-op, ``result`` is set."""

    def __init__(self, result):
        self._result = result

    def __call__(self, *args, **kwargs):
        self.result = self._result
        return self

    def run(self):
        return None


def test_pick_workspace_returns_selected_membership(monkeypatch):
    memberships = [_membership("ws-1"), _membership("ws-2", workspace_id=2)]
    monkeypatch.setattr(_agent_app, "WorkspacePicker", _FakePicker("ws-2"))

    chosen = _commands._pick_workspace(memberships)

    assert chosen.public_id == "ws-2"


def test_pick_workspace_aborts_when_cancelled(monkeypatch):
    monkeypatch.setattr(_agent_app, "WorkspacePicker", _FakePicker(None))
    with pytest.raises(click.Abort):
        _commands._pick_workspace([_membership()])


# --------------------------------------------------------------------------- #
# _pick_harness
# --------------------------------------------------------------------------- #


def test_pick_harness_returns_selection(tmp_path, monkeypatch):
    monkeypatch.setattr(_claude, "ide_extension_hosts", lambda: ())
    monkeypatch.setattr(
        _process.shutil,
        "which",
        lambda name: "/usr/bin/pi" if name == "pi" else None,
    )
    monkeypatch.setattr(_agent_app, "HarnessPicker", _FakePicker("pi"))

    assert _commands._pick_harness(tmp_path) == "pi"


def test_pick_harness_lists_detected_then_undetected(tmp_path, monkeypatch):
    captured: dict[str, object] = {}

    class _CapturePicker:
        def __init__(self, rows, *, preselect=0):
            captured["rows"] = rows
            captured["preselect"] = preselect
            self.result = "cursor"

        def run(self):
            return None

    monkeypatch.setattr(_bob, "BOB_IDE_APP_PATH", tmp_path / "absent.app")
    monkeypatch.setattr(_claude, "CLAUDE_UI_APP_PATH", tmp_path / "absent.app")
    monkeypatch.setattr(_claude, "ide_extension_hosts", lambda: ())
    monkeypatch.setattr(
        _process.shutil,
        "which",
        lambda name: "/usr/bin/cursor" if name == "cursor" else None,
    )
    monkeypatch.setattr(_agent_app, "HarnessPicker", _CapturePicker)

    assert _commands._pick_harness(tmp_path) == "cursor"
    rows = captured["rows"]
    assert rows[0] == ("cursor", "Cursor IDE", True)
    assert [name for name, _, detected in rows if detected] == ["cursor"]
    assert [name for name, _, detected in rows if not detected] == [
        name for name in HARNESS_NAMES if name != "cursor"
    ]
    assert captured["preselect"] == 0


def test_pick_harness_lists_undetected_when_none_detected(tmp_path, monkeypatch):
    captured: dict[str, object] = {}

    class _CapturePicker:
        def __init__(self, rows, *, preselect=0):
            captured["rows"] = rows
            self.result = "pi"

        def run(self):
            return None

    monkeypatch.setattr(_bob, "BOB_IDE_APP_PATH", tmp_path / "absent.app")
    monkeypatch.setattr(_claude, "CLAUDE_UI_APP_PATH", tmp_path / "absent.app")
    monkeypatch.setattr(_claude, "ide_extension_hosts", lambda: ())
    monkeypatch.setattr(_process.shutil, "which", lambda name: None)
    monkeypatch.setattr(_agent_app, "HarnessPicker", _CapturePicker)

    assert _commands._pick_harness(tmp_path) == "pi"
    rows = captured["rows"]
    assert [name for name, _, _ in rows] == HARNESS_NAMES
    assert all(detected is False for _, _, detected in rows)


def test_pick_harness_aborts_when_cancelled(tmp_path, monkeypatch):
    monkeypatch.setattr(_claude, "ide_extension_hosts", lambda: ())
    monkeypatch.setattr(
        _process.shutil,
        "which",
        lambda name: "/usr/bin/opencode" if name == "opencode" else None,
    )
    monkeypatch.setattr(_agent_app, "HarnessPicker", _FakePicker(None))
    with pytest.raises(click.Abort):
        _commands._pick_harness(tmp_path)


# --------------------------------------------------------------------------- #
# _resolve_membership
# --------------------------------------------------------------------------- #


def test_resolve_membership_single_membership_auto_selected():
    only = _membership("ws-1")
    assert _commands._resolve_membership([only]) is only


def test_resolve_membership_multiple_non_interactive_errors(monkeypatch):
    monkeypatch.setattr(_commands, "is_non_interactive", lambda: True)
    memberships = [_membership("ws-1"), _membership("ws-2", workspace_id=2)]
    with pytest.raises(click.UsageError, match="run interactively"):
        _commands._resolve_membership(memberships)


def test_resolve_membership_multiple_interactive_picks(monkeypatch):
    monkeypatch.setattr(_commands, "is_non_interactive", lambda: False)
    memberships = [_membership("ws-1"), _membership("ws-2", workspace_id=2)]
    monkeypatch.setattr(_commands, "_pick_workspace", lambda m: m[1])
    assert _commands._resolve_membership(memberships).public_id == "ws-2"


# --------------------------------------------------------------------------- #
# agent command: extra branches
# --------------------------------------------------------------------------- #


def test_agent_no_memberships_errors(tmp_path, monkeypatch):
    _stub_login(monkeypatch)
    _stub_identity(monkeypatch, [])

    result = CliRunner().invoke(
        agent, ["--workspace", str(tmp_path), "--harness", "opencode"]
    )

    assert result.exit_code != 0
    assert "not a member of any hub workspace" in _plain_output(result.output)


def test_agent_errors_when_harness_not_installed(tmp_path, monkeypatch):
    _write_skore(tmp_path)
    monkeypatch.setattr(_process.shutil, "which", lambda cmd: None)
    result = CliRunner().invoke(
        agent, ["--workspace", str(tmp_path), "--harness", "opencode"]
    )

    assert result.exit_code != 0
    assert "not installed or not on PATH" in _plain_output(result.output)
    assert (tmp_path / "opencode.json").is_file()
    saved = json.loads((tmp_path / SKORE_FILENAME).read_text())
    assert saved["harness"] == "opencode"


def test_agent_valid_config_without_harness_non_interactive_errors(
    tmp_path, monkeypatch
):
    # Config is complete (hub_url + workspace) but no harness was ever saved.
    _clear_agent_envs(monkeypatch)
    payload = {
        "hub_url": "http://hub.test",
        "workspace": "ws-1",
        "workspace_id": 1,
    }
    (tmp_path / SKORE_FILENAME).write_text(json.dumps(payload) + "\n")
    monkeypatch.setattr(_commands, "is_non_interactive", lambda: True)

    result = CliRunner().invoke(agent, ["--workspace", str(tmp_path)])

    assert result.exit_code != 0
    assert "pass --harness" in _plain_output(result.output)


def test_agent_valid_config_without_harness_picks_interactively(tmp_path, monkeypatch):
    payload = {
        "hub_url": "http://hub.test",
        "workspace": "ws-1",
        "workspace_id": 1,
    }
    (tmp_path / SKORE_FILENAME).write_text(json.dumps(payload) + "\n")
    _mock_harness_on_path(monkeypatch, "opencode")
    monkeypatch.setattr(_commands, "is_non_interactive", lambda: False)
    monkeypatch.setattr(_commands, "_pick_harness", lambda workspace: "opencode")
    monkeypatch.setattr(
        _commands,
        "launch_harness",
        lambda selected, workspace, model_id=DEFAULT_MODEL_ID, **k: None,
    )

    result = CliRunner().invoke(agent, ["--workspace", str(tmp_path)])

    assert result.exit_code == 0, result.output
    saved = json.loads((tmp_path / SKORE_FILENAME).read_text())
    assert saved["harness"] == "opencode"


def test_agent_first_run_picks_harness_interactively(tmp_path, monkeypatch):
    _mock_harness_on_path(monkeypatch, "opencode")
    _stub_login(monkeypatch)
    _stub_identity(monkeypatch, [_membership()])
    monkeypatch.setattr(_commands, "is_non_interactive", lambda: False)
    monkeypatch.setattr(_commands, "_pick_harness", lambda workspace: "opencode")
    monkeypatch.setattr(
        _commands,
        "launch_harness",
        lambda selected, workspace, model_id=DEFAULT_MODEL_ID, **k: None,
    )

    result = CliRunner().invoke(agent, ["--workspace", str(tmp_path)])

    assert result.exit_code == 0, result.output
    saved = json.loads((tmp_path / SKORE_FILENAME).read_text())
    assert saved["harness"] == "opencode"


def test_agent_rewrites_config_when_harness_changes(tmp_path, monkeypatch):
    _write_skore(tmp_path, harness="claude")
    _mock_harness_on_path(monkeypatch, "opencode")
    monkeypatch.setattr(
        _commands,
        "launch_harness",
        lambda selected, workspace, model_id=DEFAULT_MODEL_ID, **k: None,
    )

    result = CliRunner().invoke(
        agent, ["--workspace", str(tmp_path), "--harness", "opencode"]
    )

    assert result.exit_code == 0, result.output
    saved = json.loads((tmp_path / SKORE_FILENAME).read_text())
    assert saved["harness"] == "opencode"


# --------------------------------------------------------------------------- #
# Agent detection: non-interactive harness auto-selection
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


def test_agent_non_interactive_auto_selects_claude(tmp_path, monkeypatch):
    _clear_agent_envs(monkeypatch)
    monkeypatch.setenv("CLAUDECODE", "1")
    _mock_harness_on_path(monkeypatch, "claude")
    _stub_login(monkeypatch)
    _stub_identity(monkeypatch, [_membership()])
    monkeypatch.setattr(_commands, "is_non_interactive", lambda: True)
    launched = []
    monkeypatch.setattr(
        _commands,
        "launch_harness",
        lambda selected, workspace, model_id=DEFAULT_MODEL_ID, **k: launched.append(
            selected.harness_name
        ),
    )

    result = CliRunner().invoke(agent, ["--workspace", str(tmp_path)])

    assert result.exit_code == 0, result.output
    saved = json.loads((tmp_path / SKORE_FILENAME).read_text())
    assert saved["harness"] == "claude"
    assert launched == []
    assert "configured with the Skore Hub" in _plain_output(result.output)


def test_agent_non_interactive_auto_selects_opencode(tmp_path, monkeypatch):
    _clear_agent_envs(monkeypatch)
    monkeypatch.setenv("OPENCODE_CLIENT", "1")
    _mock_harness_on_path(monkeypatch, "opencode")
    _stub_login(monkeypatch)
    _stub_identity(monkeypatch, [_membership()])
    monkeypatch.setattr(_commands, "is_non_interactive", lambda: True)
    launched = []
    monkeypatch.setattr(
        _commands,
        "launch_harness",
        lambda selected, workspace, model_id=DEFAULT_MODEL_ID, **k: launched.append(
            selected.harness_name
        ),
    )

    result = CliRunner().invoke(agent, ["--workspace", str(tmp_path)])

    assert result.exit_code == 0, result.output
    saved = json.loads((tmp_path / SKORE_FILENAME).read_text())
    assert saved["harness"] == "opencode"
    assert launched == []


def test_agent_non_interactive_auto_selects_pi(tmp_path, monkeypatch):
    _clear_agent_envs(monkeypatch)
    monkeypatch.setenv("PI_CODING_AGENT", "true")
    _mock_harness_on_path(monkeypatch, "pi")
    _stub_login(monkeypatch)
    _stub_identity(monkeypatch, [_membership()])
    monkeypatch.setattr(_commands, "is_non_interactive", lambda: True)
    launched = []
    monkeypatch.setattr(
        _commands,
        "launch_harness",
        lambda selected, workspace, model_id=DEFAULT_MODEL_ID, **k: launched.append(
            selected.harness_name
        ),
    )

    result = CliRunner().invoke(agent, ["--workspace", str(tmp_path)])

    assert result.exit_code == 0, result.output
    saved = json.loads((tmp_path / SKORE_FILENAME).read_text())
    assert saved["harness"] == "pi"
    assert launched == []


def test_agent_non_interactive_detected_harness_not_on_path_errors(
    tmp_path, monkeypatch
):
    _clear_agent_envs(monkeypatch)
    monkeypatch.setenv("CLAUDECODE", "1")
    monkeypatch.setattr(_process.shutil, "which", lambda cmd: None)
    _stub_login(monkeypatch)
    _stub_identity(monkeypatch, [_membership()])
    monkeypatch.setattr(_commands, "is_non_interactive", lambda: True)

    result = CliRunner().invoke(agent, ["--workspace", str(tmp_path)])

    assert result.exit_code != 0
    assert "pass --harness" in _plain_output(result.output)


def test_agent_detected_different_harness_still_launches(tmp_path, monkeypatch):
    """Claude detected but --harness opencode explicitly -> launches opencode."""
    _clear_agent_envs(monkeypatch)
    monkeypatch.setenv("CLAUDECODE", "1")
    _mock_harness_on_path(monkeypatch, "opencode")
    _write_skore(tmp_path)
    launched = []
    monkeypatch.setattr(
        _commands,
        "launch_harness",
        lambda selected, workspace, model_id=DEFAULT_MODEL_ID, **k: launched.append(
            selected.harness_name
        ),
    )

    result = CliRunner().invoke(
        agent, ["--workspace", str(tmp_path), "--harness", "opencode"]
    )

    assert result.exit_code == 0, result.output
    assert launched == ["opencode"]


def test_agent_reuse_path_auto_selects_detected_harness(tmp_path, monkeypatch):
    """Config exists but no harness; non-interactive + detected."""
    _clear_agent_envs(monkeypatch)
    monkeypatch.setenv("OPENCODE_CLIENT", "1")
    payload = {
        "hub_url": "http://hub.test",
        "workspace": "ws-1",
        "workspace_id": 1,
    }
    (tmp_path / SKORE_FILENAME).write_text(json.dumps(payload) + "\n")
    _mock_harness_on_path(monkeypatch, "opencode")
    monkeypatch.setattr(_commands, "is_non_interactive", lambda: True)
    launched = []
    monkeypatch.setattr(
        _commands,
        "launch_harness",
        lambda selected, workspace, model_id=DEFAULT_MODEL_ID, **k: launched.append(
            selected.harness_name
        ),
    )

    result = CliRunner().invoke(agent, ["--workspace", str(tmp_path)])

    assert result.exit_code == 0, result.output
    saved = json.loads((tmp_path / SKORE_FILENAME).read_text())
    assert saved["harness"] == "opencode"
    assert launched == []


def test_agent_translates_harness_runtime_errors(tmp_path, monkeypatch):
    _write_skore(tmp_path)
    _mock_harness_on_path(monkeypatch, "opencode")

    def unknown(name):
        raise RuntimeError("harness broke")

    monkeypatch.setattr(_commands, "get_harness", unknown)

    result = CliRunner().invoke(
        agent, ["--workspace", str(tmp_path), "--harness", "opencode"]
    )

    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit)
    assert "harness broke" in result.output
    assert "Traceback" not in result.output


def test_agent_translates_unknown_harness_keys(tmp_path, monkeypatch):
    _write_skore(tmp_path)
    _mock_harness_on_path(monkeypatch, "opencode")
    monkeypatch.setattr(
        _commands, "get_harness", lambda name: (_ for _ in ()).throw(KeyError(name))
    )

    result = CliRunner().invoke(
        agent, ["--workspace", str(tmp_path), "--harness", "opencode"]
    )

    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit)
    assert "unknown harness: opencode" in result.output
    assert "Traceback" not in result.output


def test_agent_leaves_harness_abort_uncaught_as_click_exception(tmp_path, monkeypatch):
    _write_skore(tmp_path)
    _mock_harness_on_path(monkeypatch, "opencode")
    monkeypatch.setattr(
        _commands, "get_harness", lambda name: (_ for _ in ()).throw(click.Abort())
    )

    result = CliRunner().invoke(
        agent, ["--workspace", str(tmp_path), "--harness", "opencode"]
    )

    assert result.exit_code == 1
    assert "Aborted." in result.output
    assert "Traceback" not in result.output
