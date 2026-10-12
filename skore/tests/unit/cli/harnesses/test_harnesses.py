"""Tests for the harness registry, detection and config writers."""

from __future__ import annotations

import json
import os
import tomllib
from typing import Any

import pytest
import rich_click as click

from skore._cli import agents as _agents
from skore._cli.agents import (
    AGENTS,
    installed_harnesses,
    is_harness_installed,
)
from skore._cli.harnesses import bob as _bob
from skore._cli.harnesses import claude as _claude
from skore._cli.harnesses import copilot as _copilot
from skore._cli.harnesses import cursor as _cursor
from skore._cli.harnesses import process as _process
from skore._cli.harnesses.context import HarnessContext


def _stub_hub_registry(monkeypatch, api_key):
    from skore._plugins.hub.authentication import key

    monkeypatch.setattr(key, "get", lambda **k: api_key)


def _ctx(workspace, **kwargs):
    return HarnessContext(
        workspace=workspace,
        hub_url="http://hub.test",
        api_key="secret-key",
        **kwargs,
    )


@pytest.fixture(autouse=True)
def no_bob_ide_app(tmp_path, monkeypatch):
    """Keep detection off the real machine: Bob IDE is found by its bundle on
    macOS and by the ``bobide`` binary on other platforms."""
    monkeypatch.setattr(_bob, "BOB_IDE_APP_PATH", tmp_path / "absent.app")
    monkeypatch.setattr(_claude, "CLAUDE_UI_APP_PATH", tmp_path / "absent.app")
    monkeypatch.setattr(_claude, "ide_extension_hosts", lambda: ())
    for name in ("CURSOR_AGENT", "CURSOR_CLI"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(_process.subprocess, "run", lambda *a, **k: None)
    _real_which = _process.shutil.which

    def _which(name):
        if name in {"bobide", "agent", "cursor-agent"}:
            return None
        return _real_which(name)

    monkeypatch.setattr(_process.shutil, "which", _which)


def test_opencode_installed_by_binary(monkeypatch):
    monkeypatch.setattr(
        _process.shutil,
        "which",
        lambda name: "/usr/bin/opencode" if name == "opencode" else None,
    )
    assert is_harness_installed(AGENTS["opencode"]) is True
    assert [agent.harness_name for agent in installed_harnesses()] == ["opencode"]


def test_claude_installed_by_binary(monkeypatch):
    monkeypatch.setattr(
        _process.shutil,
        "which",
        lambda name: "/usr/bin/claude" if name == "claude" else None,
    )
    assert [agent.harness_name for agent in installed_harnesses()] == ["claude"]
    assert is_harness_installed(AGENTS["claude-ui"]) is False


def test_claude_ui_installed_by_app_bundle(tmp_path, monkeypatch):
    monkeypatch.setattr(_process.sys, "platform", "darwin")
    bundle = tmp_path / "Claude.app"
    bundle.mkdir()
    monkeypatch.setattr(_claude, "CLAUDE_UI_APP_PATH", bundle)
    monkeypatch.setattr(_process.shutil, "which", lambda name: None)
    assert is_harness_installed(AGENTS["claude-ui"]) is True
    assert [agent.harness_name for agent in installed_harnesses()] == ["claude-ui"]


def test_claude_ui_not_detected_by_binary_on_non_darwin(monkeypatch):
    monkeypatch.setattr(_process.sys, "platform", "linux")
    monkeypatch.setattr(
        _process.shutil,
        "which",
        lambda name: "/usr/bin/claude-ui" if name == "claude-ui" else None,
    )
    assert is_harness_installed(AGENTS["claude-ui"]) is False
    assert installed_harnesses() == []


def test_pi_installed_by_binary(monkeypatch):
    monkeypatch.setattr(
        _process.shutil,
        "which",
        lambda name: "/usr/bin/pi" if name == "pi" else None,
    )
    assert [agent.harness_name for agent in installed_harnesses()] == ["pi"]


def test_cursor_installed_by_binary(monkeypatch):
    monkeypatch.setattr(
        _process.shutil,
        "which",
        lambda name: "/usr/local/bin/cursor" if name == "cursor" else None,
    )
    assert is_harness_installed(AGENTS["cursor"]) is True
    assert is_harness_installed(AGENTS["cursor-cli"]) is False
    assert [agent.harness_name for agent in installed_harnesses()] == ["cursor"]


def test_cursor_cli_installed_by_agent_binary(monkeypatch):
    monkeypatch.setattr(
        _process.shutil,
        "which",
        lambda name: "/usr/local/cursor/bin/agent" if name == "agent" else None,
    )
    assert is_harness_installed(AGENTS["cursor-cli"]) is True
    assert [agent.harness_name for agent in installed_harnesses()] == ["cursor-cli"]


def test_cursor_cli_installed_by_cursor_agent_binary(monkeypatch):
    monkeypatch.setattr(
        _process.shutil,
        "which",
        lambda name: "/usr/local/bin/cursor-agent" if name == "cursor-agent" else None,
    )
    assert is_harness_installed(AGENTS["cursor-cli"]) is True
    assert [agent.harness_name for agent in installed_harnesses()] == ["cursor-cli"]


def test_claude_plugin_installed_by_cursor_extension(tmp_path, monkeypatch):
    extensions = tmp_path / ".cursor" / "extensions"
    (extensions / "anthropic.claude-code-2.1.0-darwin-arm64").mkdir(parents=True)
    monkeypatch.setattr(
        _claude,
        "ide_extension_hosts",
        lambda: (("cursor", extensions, "cursor"),),
    )
    monkeypatch.setattr(_process.shutil, "which", lambda name: None)
    assert is_harness_installed(AGENTS["claude-plugin"]) is True
    assert [agent.harness_name for agent in installed_harnesses()] == ["claude-plugin"]


def test_claude_plugin_installed_by_unversioned_extension_dir(tmp_path, monkeypatch):
    extensions = tmp_path / ".vscode" / "extensions"
    (extensions / "anthropic.claude-code").mkdir(parents=True)
    monkeypatch.setattr(
        _claude,
        "ide_extension_hosts",
        lambda: (("code", extensions, "vscode"),),
    )
    monkeypatch.setattr(_process.shutil, "which", lambda name: None)
    assert is_harness_installed(AGENTS["claude-plugin"]) is True


def test_claude_plugin_ignores_unrelated_extensions(tmp_path, monkeypatch):
    extensions = tmp_path / ".vscode" / "extensions"
    (extensions / "openai.chatgpt-1.0.0").mkdir(parents=True)
    monkeypatch.setattr(
        _claude,
        "ide_extension_hosts",
        lambda: (("code", extensions, "vscode"),),
    )
    monkeypatch.setattr(_process.shutil, "which", lambda name: None)
    assert is_harness_installed(AGENTS["claude-plugin"]) is False
    assert installed_harnesses() == []


def test_bob_shell_installed_by_binary(monkeypatch):
    monkeypatch.setattr(
        _process.shutil,
        "which",
        lambda name: "/usr/local/bin/bob" if name == "bob" else None,
    )
    assert is_harness_installed(AGENTS["bob"]) is True
    assert [agent.harness_name for agent in installed_harnesses()] == ["bob"]


def test_bob_ide_installed_by_app_bundle(tmp_path, monkeypatch):
    """On macOS the IDE installs no command, so detection uses the bundle."""
    monkeypatch.setattr(_process.sys, "platform", "darwin")
    bundle = tmp_path / "IBM Bob.app"
    bundle.mkdir()
    monkeypatch.setattr(_bob, "BOB_IDE_APP_PATH", bundle)
    monkeypatch.setattr(_process.shutil, "which", lambda name: None)
    assert is_harness_installed(AGENTS["bob-ide"]) is True
    assert [agent.harness_name for agent in installed_harnesses()] == ["bob-ide"]


def test_bob_ide_installed_by_binary_on_non_darwin(tmp_path, monkeypatch):
    """On non-macOS the IDE installs a ``bobide`` command on PATH."""
    monkeypatch.setattr(_process.sys, "platform", "linux")
    monkeypatch.setattr(
        _process.shutil,
        "which",
        lambda name: "/usr/bin/bobide" if name == "bobide" else None,
    )
    assert is_harness_installed(AGENTS["bob-ide"]) is True
    assert [agent.harness_name for agent in installed_harnesses()] == ["bob-ide"]


def test_installed_harnesses_excludes_missing(monkeypatch):
    monkeypatch.setattr(_process.shutil, "which", lambda name: None)
    assert installed_harnesses() == []


def test_opencode_config_matches_hub_ui(tmp_path):
    AGENTS["opencode"].configure(_ctx(tmp_path))
    config = json.loads((tmp_path / "opencode.json").read_text())
    assert config["$schema"] == "https://opencode.ai/config.json"
    assert config["model"] == "skore/skore-agent"
    assert config["provider"]["skore"]["name"] == "Skore Hub"
    model = config["provider"]["skore"]["models"]["skore-agent"]
    assert model["name"] == "Skore Agent"
    assert model["limit"] == {"context": 200000, "output": 8192}
    assert "cost" not in model


def test_opencode_writes_session_plugin(tmp_path):
    AGENTS["opencode"].configure(_ctx(tmp_path))
    plugin = tmp_path / ".opencode" / "plugins" / "skore-session.js"
    source = plugin.read_text()
    assert "chat.headers" in source
    assert "X-Skore-Session-Id" in source
    assert "sessionID" in source
    ignored = (tmp_path / ".gitignore").read_text().splitlines()
    assert ".opencode/plugins/skore-session.js" in ignored


def test_pi_config_matches_hub_ui(tmp_path):
    AGENTS["pi"].configure(_ctx(tmp_path))
    config = json.loads((tmp_path / ".pi" / "agent" / "models.json").read_text())
    model = config["providers"]["skore"]["models"][0]
    assert model["id"] == "skore-agent"
    assert model["contextWindow"] == 200000
    assert "cost" not in model
    compat = config["providers"]["skore"]["compat"]
    assert compat["sendSessionAffinityHeaders"] is True
    assert compat["sessionAffinityFormat"] == "openrouter"


def test_cursor_config_points_at_the_mcp_endpoint(tmp_path):
    AGENTS["cursor"].configure(_ctx(tmp_path))

    config = json.loads((tmp_path / ".cursor" / "mcp.json").read_text())
    assert config["mcpServers"]["skore"] == {
        "type": "http",
        "url": "http://hub.test/mcp",
        "headers": {"Authorization": "Bearer secret-key"},
    }

    permissions = json.loads((tmp_path / ".cursor" / "permissions.json").read_text())
    assert permissions["mcpAllowlist"] == ["skore:*"]
    assert permissions["autoRun"]["allow_instructions"] == [
        "curl fetching a Skore materialize URL under /v1/materialize/ to write "
        "a template or script into the workspace",
        "calls to the skore MCP server's skore_agent tool",
    ]


def test_cursor_config_prints_a_prominent_enable_hint(tmp_path):
    """The manual enable step must stand out, not hide in a muted note."""
    from skore._cli import style as _style

    with _style.console.capture() as capture:
        AGENTS["cursor"].configure(_ctx(tmp_path))

    output = capture.get()
    assert "Action required" in output
    assert "Customize" in output


def test_bob_shell_config_declares_the_streamable_http_url(tmp_path):
    """Bob Shell reads ``httpURL``; a plain ``url`` would mean legacy SSE."""
    AGENTS["bob"].configure(_ctx(tmp_path))

    config = json.loads((tmp_path / ".bob" / "mcp.json").read_text())
    assert config["mcpServers"]["skore"] == {
        "httpURL": "http://hub.test/mcp",
        "headers": {"Authorization": "Bearer secret-key"},
        "alwaysAllow": ["skore_agent"],
        "disabled": False,
    }


def test_bob_ide_config_declares_the_transport_type(tmp_path):
    """Bob IDE reads the same file but wants ``type`` alongside ``url``."""
    AGENTS["bob-ide"].configure(_ctx(tmp_path))

    config = json.loads((tmp_path / ".bob" / "mcp.json").read_text())
    assert config["mcpServers"]["skore"] == {
        "type": "streamable-http",
        "url": "http://hub.test/mcp",
        "headers": {"Authorization": "Bearer secret-key"},
        "alwaysAllow": ["skore_agent"],
        "disabled": False,
    }


def test_bob_config_preserves_what_the_user_already_had(tmp_path):
    config_dir = tmp_path / ".bob"
    config_dir.mkdir()
    (config_dir / "mcp.json").write_text(
        json.dumps({"mcpServers": {"other": {"command": "node"}}})
    )

    AGENTS["bob"].configure(_ctx(tmp_path))

    config = json.loads((config_dir / "mcp.json").read_text())
    assert set(config["mcpServers"]) == {"other", "skore"}
    assert config["mcpServers"]["other"] == {"command": "node"}


def test_bob_config_is_idempotent(tmp_path):
    AGENTS["bob"].configure(_ctx(tmp_path))
    AGENTS["bob"].configure(_ctx(tmp_path))

    config = json.loads((tmp_path / ".bob" / "mcp.json").read_text())
    assert config["mcpServers"]["skore"]["alwaysAllow"] == ["skore_agent"]
    assert (tmp_path / ".gitignore").read_text().count(".bob/mcp.json") == 1


def test_bob_config_refuses_to_overwrite_what_it_cannot_read(tmp_path):
    config_dir = tmp_path / ".bob"
    config_dir.mkdir()
    (config_dir / "mcp.json").write_text("{not json")

    with pytest.raises(RuntimeError, match="mcp.json"):
        AGENTS["bob"].configure(_ctx(tmp_path))

    assert (config_dir / "mcp.json").read_text() == "{not json"


@pytest.mark.parametrize(
    "name, entry",
    [
        ("opencode", "opencode.json"),
        ("claude-code", ".claude/settings.local.json"),
        ("claude-ui", ".claude/settings.local.json"),
        ("claude-plugin", ".claude/settings.local.json"),
        ("cursor-cli", ".cursor/mcp.json"),
        ("pi", ".pi/agent/models.json"),
        ("cursor", ".cursor/mcp.json"),
        ("bob", ".bob/mcp.json"),
        ("bob-ide", ".bob/mcp.json"),
    ],
)
def test_config_embedding_the_api_key_is_gitignored(tmp_path, name, entry):
    AGENTS[name].configure(_ctx(tmp_path))

    assert "secret-key" in (tmp_path / entry).read_text()
    assert entry in (tmp_path / ".gitignore").read_text().splitlines()


def test_cursor_permissions_are_not_gitignored(tmp_path):
    """They hold no secret, and a team may well want them committed."""
    AGENTS["cursor"].configure(_ctx(tmp_path))
    gitignore = (tmp_path / ".gitignore").read_text()
    assert "permissions.json" not in gitignore


def test_cursor_config_preserves_what_the_user_already_had(tmp_path):
    config_dir = tmp_path / ".cursor"
    config_dir.mkdir()
    (config_dir / "mcp.json").write_text(
        json.dumps({"mcpServers": {"other": {"url": "http://elsewhere"}}})
    )
    (config_dir / "permissions.json").write_text(
        json.dumps(
            {
                "mcpAllowlist": ["other:*"],
                "allow": ["Read(**)"],
                "autoRun": {"allow_instructions": ["anything the user allowed"]},
            }
        )
    )

    AGENTS["cursor"].configure(_ctx(tmp_path))

    config = json.loads((config_dir / "mcp.json").read_text())
    assert set(config["mcpServers"]) == {"other", "skore"}
    assert config["mcpServers"]["other"] == {"url": "http://elsewhere"}

    permissions = json.loads((config_dir / "permissions.json").read_text())
    assert permissions["mcpAllowlist"] == ["other:*", "skore:*"]
    assert permissions["allow"] == ["Read(**)"]
    assert permissions["autoRun"]["allow_instructions"][0] == (
        "anything the user allowed"
    )
    assert len(permissions["autoRun"]["allow_instructions"]) == 3


def test_cursor_config_is_idempotent(tmp_path):
    AGENTS["cursor"].configure(_ctx(tmp_path))
    AGENTS["cursor"].configure(_ctx(tmp_path))

    permissions = json.loads((tmp_path / ".cursor" / "permissions.json").read_text())
    assert permissions["mcpAllowlist"] == ["skore:*"]
    assert len(permissions["autoRun"]["allow_instructions"]) == 2


@pytest.mark.parametrize("content", ["{not json", "// a comment\n{}", "[]"])
def test_cursor_config_refuses_to_overwrite_what_it_cannot_read(tmp_path, content):
    """Cursor accepts comments in these files; wiping one would lose real config."""
    config_dir = tmp_path / ".cursor"
    config_dir.mkdir()
    (config_dir / "mcp.json").write_text(content)

    with pytest.raises(RuntimeError, match="mcp.json"):
        AGENTS["cursor"].configure(_ctx(tmp_path))

    assert (config_dir / "mcp.json").read_text() == content


def test_cursor_config_writes_nothing_when_permissions_cannot_be_read(tmp_path):
    config_dir = tmp_path / ".cursor"
    config_dir.mkdir()
    (config_dir / "permissions.json").write_text("{not json")

    with pytest.raises(RuntimeError, match="permissions.json"):
        AGENTS["cursor"].configure(_ctx(tmp_path))

    assert not (config_dir / "mcp.json").exists()


def test_launch_opencode_passes_model_flag(tmp_path, monkeypatch):
    captured: dict[str, list[str]] = {}

    def fake_exec(name, argv, *, env=None):
        captured["argv"] = argv

    monkeypatch.setattr(
        _process.shutil,
        "which",
        lambda cmd: "/usr/bin/opencode" if cmd == "opencode" else None,
    )
    monkeypatch.setattr(_process, "_exec_harness", fake_exec)
    _agents.launch_harness(AGENTS["opencode"], tmp_path, model_id="skore-agent")
    assert captured["argv"] == ["opencode", "-m", "skore/skore-agent"]


def test_launch_pi_passes_provider_and_model(tmp_path, monkeypatch):
    captured: dict[str, object] = {}

    def fake_exec(name, argv, *, env=None):
        captured["argv"] = argv
        captured["env"] = env

    monkeypatch.setattr(
        _process.shutil,
        "which",
        lambda cmd: "/usr/bin/pi" if cmd == "pi" else None,
    )
    monkeypatch.setattr(_process, "_exec_harness", fake_exec)
    _agents.launch_harness(AGENTS["pi"], tmp_path, model_id="skore-agent")
    assert captured["argv"] == ["pi", "--provider", "skore", "--model", "skore-agent"]
    assert "PI_CODING_AGENT_DIR" in captured["env"]


def test_launch_cursor_opens_the_workspace(tmp_path, monkeypatch):
    captured: dict[str, object] = {}

    def fake_exec(name, argv, *, env=None):
        captured["argv"] = argv
        captured["env"] = env

    monkeypatch.setattr(
        _process.shutil,
        "which",
        lambda cmd: "/usr/local/bin/cursor" if cmd == "cursor" else None,
    )
    monkeypatch.setattr(_process, "_exec_harness", fake_exec)
    _agents.launch_harness(AGENTS["cursor"], tmp_path)
    assert captured["argv"] == ["cursor", str(tmp_path)]
    # The key lives in mcp.json, so no environment has to reach the app and a
    # window opened any other way works just as well.
    assert captured["env"] is None


def test_cursor_cli_ignores_generic_agent_binary(monkeypatch):
    monkeypatch.setattr(
        _process.shutil,
        "which",
        lambda name: "/usr/local/bin/agent" if name == "agent" else None,
    )
    assert is_harness_installed(AGENTS["cursor-cli"]) is False


def test_launch_cursor_cli_uses_agent(tmp_path, monkeypatch):
    captured: dict[str, object] = {}

    def fake_exec(name, argv, *, env=None):
        captured["argv"] = argv
        captured["cwd"] = os.getcwd()

    monkeypatch.setattr(
        _process.shutil,
        "which",
        lambda cmd: "/usr/local/cursor/bin/agent" if cmd == "agent" else None,
    )
    monkeypatch.setattr(_process, "_exec_harness", fake_exec)
    _agents.launch_harness(AGENTS["cursor-cli"], tmp_path)
    assert captured["argv"] == ["agent"]
    assert captured["cwd"] == str(tmp_path)


def test_launch_cursor_cli_prefers_cursor_agent_when_both_exist(tmp_path, monkeypatch):
    captured: dict[str, object] = {}

    def fake_exec(name, argv, *, env=None):
        captured["argv"] = argv

    monkeypatch.setattr(
        _process.shutil,
        "which",
        lambda cmd: (
            "/usr/local/bin/cursor-agent"
            if cmd == "cursor-agent"
            else "/usr/local/bin/agent"
            if cmd == "agent"
            else None
        ),
    )
    monkeypatch.setattr(_process, "_exec_harness", fake_exec)
    _agents.launch_harness(AGENTS["cursor-cli"], tmp_path)
    assert captured["argv"] == ["cursor-agent"]


def test_launch_cursor_cli_falls_back_to_cursor_agent(tmp_path, monkeypatch):
    captured: dict[str, object] = {}

    def fake_exec(name, argv, *, env=None):
        captured["argv"] = argv

    monkeypatch.setattr(
        _process.shutil,
        "which",
        lambda cmd: "/usr/local/bin/cursor-agent" if cmd == "cursor-agent" else None,
    )
    monkeypatch.setattr(_process, "_exec_harness", fake_exec)
    _agents.launch_harness(AGENTS["cursor-cli"], tmp_path)
    assert captured["argv"] == ["cursor-agent"]


def test_launch_claude_plugin_opens_cursor_uri_in_cursor(tmp_path, monkeypatch):
    captured: dict[str, object] = {}

    def fake_exec(name, argv, *, env=None):
        captured["argv"] = argv

    extensions = tmp_path / "extensions"
    (extensions / "anthropic.claude-code-1.0.0").mkdir(parents=True)
    monkeypatch.setenv("CURSOR_AGENT", "1")
    monkeypatch.setattr(_process.sys, "platform", "darwin")
    monkeypatch.setattr(
        _claude,
        "ide_extension_hosts",
        lambda: (("cursor", extensions, "cursor"),),
    )
    monkeypatch.setattr(
        _process.shutil,
        "which",
        lambda cmd: "/usr/bin/cursor" if cmd == "cursor" else None,
    )
    opened: list[list[str]] = []
    monkeypatch.setattr(
        _process.subprocess,
        "run",
        lambda argv, check=False: opened.append(list(argv)),
    )
    monkeypatch.setattr(_process, "_exec_harness", fake_exec)
    _agents.launch_harness(AGENTS["claude-plugin"], tmp_path)
    assert opened == [["/usr/bin/cursor", str(tmp_path)]]
    assert captured["argv"] == ["open", "cursor://anthropic.claude-code/open"]


def test_missing_harness_message_claude_plugin_outside_ide():
    assert (
        _agents.missing_harness_message(AGENTS["claude-plugin"])
        == "Claude Plugin is not installed in VS Code or Cursor."
    )


def test_missing_harness_message_claude_ui_off_darwin(monkeypatch):
    monkeypatch.setattr(_process.sys, "platform", "linux")
    assert (
        _agents.missing_harness_message(AGENTS["claude-ui"])
        == "Claude UI is only supported on macOS."
    )


def test_launch_claude_ui_errors_when_app_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(_process.sys, "platform", "darwin")
    monkeypatch.setattr(_claude, "CLAUDE_UI_APP_PATH", tmp_path / "absent.app")
    with pytest.raises(RuntimeError, match="Claude UI is not installed"):
        _claude._launch_claude_ui(tmp_path, "skore-agent")


def test_open_uri_uses_xdg_open_on_linux(monkeypatch):
    captured: dict[str, object] = {}

    def fake_exec(name, argv, *, env=None):
        captured["argv"] = argv

    monkeypatch.setattr(_process.sys, "platform", "linux")
    monkeypatch.setattr(_process, "_exec_harness", fake_exec)
    _claude._open_uri("vscode://anthropic.claude-code/open")
    assert captured["argv"] == ["xdg-open", "vscode://anthropic.claude-code/open"]


def test_open_uri_uses_cmd_on_windows(monkeypatch):
    captured: dict[str, object] = {}

    def fake_exec(name, argv, *, env=None):
        captured["argv"] = argv

    monkeypatch.setattr(_process.sys, "platform", "win32")
    monkeypatch.setattr(_process, "_exec_harness", fake_exec)
    _claude._open_uri("vscode://anthropic.claude-code/open")
    assert captured["argv"] == [
        "cmd",
        "/c",
        "start",
        "",
        "vscode://anthropic.claude-code/open",
    ]


def test_open_workspace_in_ide_skips_missing_binary(tmp_path, monkeypatch):
    ran: list[object] = []
    monkeypatch.setattr(_process.shutil, "which", lambda name: None)
    monkeypatch.setattr(
        _process.subprocess, "run", lambda *args, **kwargs: ran.append(args)
    )
    _claude._open_workspace_in_ide("code", tmp_path)
    assert ran == []


def test_launch_claude_plugin_errors_when_host_missing(tmp_path):
    with pytest.raises(RuntimeError, match="Claude Plugin is not installed"):
        _agents.launch_harness(AGENTS["claude-plugin"], tmp_path)


def test_launch_cursor_cli_errors_when_binary_missing(tmp_path):
    with pytest.raises(RuntimeError, match="Cursor CLI is not installed"):
        _cursor._launch_cursor_cli(tmp_path, "skore-agent")


def test_launch_cursor_cli_restores_cwd_when_exec_fails(tmp_path, monkeypatch):
    previous = os.getcwd()
    monkeypatch.setattr(_cursor, "_resolve_cursor_cli_binary", lambda: "cursor-agent")
    monkeypatch.setattr(
        _process,
        "_exec_harness",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    with pytest.raises(RuntimeError, match="boom"):
        _cursor._launch_cursor_cli(tmp_path, "skore-agent")
    assert os.getcwd() == previous


def test_launch_copilot_cli_restores_cwd_when_exec_fails(tmp_path, monkeypatch):
    previous = os.getcwd()
    (tmp_path / ".skore").write_text(
        json.dumps(
            {
                "hub_url": "http://hub.test",
                "workspace": "ws-1",
                "workspace_id": 1,
                "harness": "copilot-cli",
            }
        )
        + "\n"
    )
    _stub_hub_registry(monkeypatch, "secret-key")
    monkeypatch.setattr(
        _process,
        "_exec_harness",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    with pytest.raises(RuntimeError, match="boom"):
        _copilot._launch_copilot_cli(tmp_path, "skore-agent")
    assert os.getcwd() == previous


def _plugin_hosts(tmp_path, *binaries: str) -> tuple[tuple[str, object, str], ...]:
    hosts = []
    for binary, scheme in (
        ("cursor", "cursor"),
        ("code", "vscode"),
        ("code-insiders", "vscode-insiders"),
    ):
        folder = tmp_path / binary
        if binary in binaries:
            (folder / "anthropic.claude-code-1.0.0").mkdir(parents=True, exist_ok=True)
        hosts.append((binary, folder, scheme))
    return tuple(hosts)


def test_claude_plugin_opens_the_only_installed_ide(tmp_path, monkeypatch):
    captured: dict[str, object] = {}

    def fake_exec(name, argv, *, env=None):
        captured["argv"] = argv

    monkeypatch.setenv("CURSOR_AGENT", "1")
    monkeypatch.setattr(_process.sys, "platform", "darwin")
    monkeypatch.setattr(
        _claude,
        "ide_extension_hosts",
        lambda: _plugin_hosts(tmp_path, "code"),
    )
    monkeypatch.setattr(
        _process.shutil,
        "which",
        lambda cmd: "/usr/bin/code" if cmd == "code" else None,
    )
    opened: list[list[str]] = []
    monkeypatch.setattr(
        _process.subprocess,
        "run",
        lambda argv, check=False: opened.append(list(argv)),
    )
    prompted: list[object] = []
    monkeypatch.setattr(
        _claude,
        "_prompt_ide",
        lambda options: prompted.append(options),
    )
    monkeypatch.setattr(_process, "_exec_harness", fake_exec)
    assert is_harness_installed(AGENTS["claude-plugin"]) is True
    _agents.launch_harness(AGENTS["claude-plugin"], tmp_path)
    assert prompted == []
    assert opened == [["/usr/bin/code", str(tmp_path)]]
    assert captured["argv"] == ["open", "vscode://anthropic.claude-code/open"]


def test_launch_claude_plugin_prompts_when_several_ides(tmp_path, monkeypatch):
    captured: dict[str, object] = {}

    def fake_exec(name, argv, *, env=None):
        captured["argv"] = argv

    monkeypatch.setattr(_process.sys, "platform", "darwin")
    monkeypatch.setattr(_claude, "is_non_interactive", lambda: False)
    monkeypatch.setattr(
        _claude,
        "ide_extension_hosts",
        lambda: _plugin_hosts(tmp_path, "cursor", "code"),
    )
    monkeypatch.setattr(
        _process.shutil,
        "which",
        lambda cmd: f"/usr/bin/{cmd}" if cmd in {"cursor", "code"} else None,
    )
    opened: list[list[str]] = []
    monkeypatch.setattr(
        _process.subprocess,
        "run",
        lambda argv, check=False: opened.append(list(argv)),
    )
    prompts: list[list[tuple[str, str]]] = []

    def fake_prompt(options):
        prompts.append(list(options))
        return "cursor"

    monkeypatch.setattr(_claude, "_prompt_ide", fake_prompt)
    monkeypatch.setattr(_process, "_exec_harness", fake_exec)
    _agents.launch_harness(AGENTS["claude-plugin"], tmp_path)
    assert prompts == [[("cursor", "Cursor"), ("code", "VS Code")]]
    assert opened == [["/usr/bin/cursor", str(tmp_path)]]
    assert captured["argv"] == ["open", "cursor://anthropic.claude-code/open"]


def test_launch_claude_plugin_reuses_saved_ide(tmp_path, monkeypatch):
    captured: dict[str, object] = {}

    def fake_exec(name, argv, *, env=None):
        captured["argv"] = argv

    (tmp_path / ".skore").write_text(
        json.dumps(
            {
                "hub_url": "http://hub.test",
                "workspace": "ws-1",
                "workspace_id": 1,
                "api_key": "secret-key",
                "harness": "claude-plugin-cursor",
            }
        )
        + "\n"
    )
    monkeypatch.setattr(_process.sys, "platform", "darwin")
    monkeypatch.setattr(_claude, "is_non_interactive", lambda: False)
    monkeypatch.setattr(
        _claude,
        "ide_extension_hosts",
        lambda: _plugin_hosts(tmp_path, "cursor", "code"),
    )
    monkeypatch.setattr(
        _process.shutil,
        "which",
        lambda cmd: f"/usr/bin/{cmd}" if cmd in {"cursor", "code"} else None,
    )
    opened: list[list[str]] = []
    monkeypatch.setattr(
        _process.subprocess,
        "run",
        lambda argv, check=False: opened.append(list(argv)),
    )
    prompted: list[object] = []
    monkeypatch.setattr(
        _claude, "_prompt_ide", lambda options: prompted.append(options)
    )
    monkeypatch.setattr(_process, "_exec_harness", fake_exec)
    _agents.launch_harness(AGENTS["claude-plugin"], tmp_path)
    assert prompted == []
    assert opened == [["/usr/bin/cursor", str(tmp_path)]]
    assert captured["argv"] == ["open", "cursor://anthropic.claude-code/open"]


def test_launch_claude_plugin_lists_ides_when_non_interactive(tmp_path, monkeypatch):
    monkeypatch.setattr(_claude, "is_non_interactive", lambda: True)
    monkeypatch.setattr(
        _claude,
        "ide_extension_hosts",
        lambda: _plugin_hosts(tmp_path, "cursor", "code", "code-insiders"),
    )
    with pytest.raises(
        RuntimeError,
        match="Cursor, VS Code, and VS Code Insiders",
    ):
        _agents.launch_harness(AGENTS["claude-plugin"], tmp_path)


def test_prompt_ide_returns_the_picker_choice(monkeypatch):
    from skore._cli.agent import app as agent_app

    class _Picker:
        def __init__(self, options):
            self.result = "code"

        def run(self):
            return None

    monkeypatch.setattr(agent_app, "IdePicker", _Picker)

    assert _claude._prompt_ide([("code", "VS Code")]) == "code"


def test_prompt_ide_aborts_when_cancelled(monkeypatch):
    from skore._cli.agent import app as agent_app

    class _Picker:
        def __init__(self, options):
            self.result = None

        def run(self):
            return None

    monkeypatch.setattr(agent_app, "IdePicker", _Picker)

    with pytest.raises(click.Abort):
        _claude._prompt_ide([("code", "VS Code")])


def test_format_ide_list_joins_two_names():
    assert _claude._format_ide_list(["Cursor", "VS Code"]) == "Cursor and VS Code"


def test_claude_plugin_ide_rejects_an_unknown_suffix():
    assert _agents.claude_plugin_ide("claude-plugin-notepad") is None


def test_choose_claude_plugin_host_errors_when_none_installed(monkeypatch):
    monkeypatch.setattr(_claude, "ide_extension_hosts", lambda: ())

    with pytest.raises(RuntimeError, match="Claude Plugin is not installed"):
        _claude._choose_claude_plugin_host()


def test_choose_claude_plugin_host_errors_when_preferred_is_missing(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        _claude,
        "ide_extension_hosts",
        lambda: _plugin_hosts(tmp_path, "cursor", "code"),
    )

    with pytest.raises(RuntimeError, match="Claude Plugin is not installed"):
        _claude._choose_claude_plugin_host("code-insiders")


def test_choose_claude_plugin_host_accepts_the_second_ide(tmp_path, monkeypatch):
    monkeypatch.setattr(_claude, "is_non_interactive", lambda: False)
    monkeypatch.setattr(
        _claude,
        "ide_extension_hosts",
        lambda: _plugin_hosts(tmp_path, "cursor", "code"),
    )
    monkeypatch.setattr(_claude, "_prompt_ide", lambda options: "code")

    binary, scheme = _claude._choose_claude_plugin_host()

    assert (binary, scheme) == ("code", "vscode")


def test_choose_claude_plugin_host_rejects_an_unknown_choice(tmp_path, monkeypatch):
    monkeypatch.setattr(_claude, "is_non_interactive", lambda: False)
    monkeypatch.setattr(
        _claude,
        "ide_extension_hosts",
        lambda: _plugin_hosts(tmp_path, "cursor", "code"),
    )
    monkeypatch.setattr(_claude, "_prompt_ide", lambda options: "nope")

    with pytest.raises(RuntimeError, match="Unknown IDE"):
        _claude._choose_claude_plugin_host()


def test_launch_bob_shell_takes_the_workspace_from_the_cwd(tmp_path, monkeypatch):
    captured: dict[str, object] = {}

    def fake_exec(name, argv, *, env=None):
        captured["argv"] = argv

    monkeypatch.setattr(
        _process.shutil,
        "which",
        lambda cmd: "/usr/local/bin/bob" if cmd == "bob" else None,
    )
    monkeypatch.setattr(_process, "_exec_harness", fake_exec)
    _agents.launch_harness(AGENTS["bob"], tmp_path)
    assert captured["argv"] == ["bob"]


def test_launch_bob_ide_opens_the_app_bundle(tmp_path, monkeypatch):
    monkeypatch.setattr(_process.sys, "platform", "darwin")
    captured: dict[str, object] = {}

    def fake_exec(name, argv, *, env=None):
        captured["argv"] = argv

    bundle = tmp_path / "IBM Bob.app"
    bundle.mkdir()
    monkeypatch.setattr(_bob, "BOB_IDE_APP_PATH", bundle)
    monkeypatch.setattr(_process, "_exec_harness", fake_exec)
    _agents.launch_harness(AGENTS["bob-ide"], tmp_path)
    assert captured["argv"] == ["open", "-a", str(bundle), str(tmp_path)]


def test_launch_claude_ui_opens_the_app_bundle(tmp_path, monkeypatch):
    monkeypatch.setattr(_process.sys, "platform", "darwin")
    captured: dict[str, object] = {}

    def fake_exec(name, argv, *, env=None):
        captured["argv"] = argv

    bundle = tmp_path / "Claude.app"
    bundle.mkdir()
    monkeypatch.setattr(_claude, "CLAUDE_UI_APP_PATH", bundle)
    monkeypatch.setattr(_process, "_exec_harness", fake_exec)
    _agents.launch_harness(AGENTS["claude-ui"], tmp_path)
    assert captured["argv"] == ["open", "-a", str(bundle), str(tmp_path)]


def test_launch_claude_ui_errors_off_darwin(tmp_path, monkeypatch):
    monkeypatch.setattr(_process.sys, "platform", "linux")
    bundle = tmp_path / "Claude.app"
    bundle.mkdir()
    monkeypatch.setattr(_claude, "CLAUDE_UI_APP_PATH", bundle)
    with pytest.raises(RuntimeError, match="only supported on macOS"):
        _claude._launch_claude_ui(tmp_path, "skore-agent")


def test_launch_bob_ide_uses_binary_on_non_darwin(tmp_path, monkeypatch):
    captured: dict[str, object] = {}

    def fake_exec(name, argv, *, env=None):
        captured["argv"] = argv

    monkeypatch.setattr(_process.sys, "platform", "linux")
    monkeypatch.setattr(
        _process.shutil,
        "which",
        lambda name: "/usr/bin/bobide" if name == "bobide" else None,
    )
    monkeypatch.setattr(_process, "_exec_harness", fake_exec)
    _agents.launch_harness(AGENTS["bob-ide"], tmp_path)
    assert captured["argv"] == ["bobide", str(tmp_path)]


def test_launch_errors_when_binary_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(
        _process.shutil,
        "which",
        lambda name: "/usr/bin/opencode" if name == "opencode" else None,
    )
    monkeypatch.setattr(
        _process.os,
        "execve",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    with pytest.raises(RuntimeError):
        _agents.launch_harness(AGENTS["opencode"], tmp_path)


def test_claude_config_matches_hub_ui(tmp_path):
    AGENTS["claude-code"].configure(_ctx(tmp_path))
    config = json.loads((tmp_path / ".claude" / "settings.local.json").read_text())
    env = config["env"]
    assert env["ANTHROPIC_BASE_URL"] == "http://hub.test"
    assert env["ANTHROPIC_AUTH_TOKEN"] == "secret-key"
    assert env["ANTHROPIC_MODEL"] == "skore-agent"


def test_launch_claude_loads_settings_env(tmp_path, monkeypatch):
    AGENTS["claude-code"].configure(_ctx(tmp_path))
    captured: dict[str, object] = {}

    def fake_exec(name, argv, *, env=None):
        captured["argv"] = argv
        captured["env"] = env

    monkeypatch.setattr(
        _process.shutil,
        "which",
        lambda cmd: "/usr/bin/claude" if cmd == "claude" else None,
    )
    monkeypatch.setattr(_process, "_exec_harness", fake_exec)
    _agents.launch_harness(AGENTS["claude-code"], tmp_path, model_id="skore-agent")

    assert captured["argv"] == ["claude"]
    assert captured["env"]["ANTHROPIC_AUTH_TOKEN"] == "secret-key"


def test_launch_claude_without_settings_uses_process_env(tmp_path, monkeypatch):
    captured: dict[str, object] = {}

    def fake_exec(name, argv, *, env=None):
        captured["env"] = env

    monkeypatch.setattr(
        _process.shutil,
        "which",
        lambda cmd: "/usr/bin/claude" if cmd == "claude" else None,
    )
    monkeypatch.setattr(_process, "_exec_harness", fake_exec)
    _agents.launch_harness(AGENTS["claude-code"], tmp_path)

    assert "ANTHROPIC_AUTH_TOKEN" not in captured["env"]


def _prepare_claude_launch(tmp_path, monkeypatch):
    captured: dict[str, Any] = {}

    def fake_exec(name, argv, *, env=None):
        captured["env"] = env

    monkeypatch.setattr(
        _process.shutil,
        "which",
        lambda cmd: "/usr/bin/claude" if cmd == "claude" else None,
    )
    monkeypatch.setattr(_process, "_exec_harness", fake_exec)
    monkeypatch.delenv("SKORE_HUB_URI", raising=False)
    return captured


def _write_skore_file(workspace, **overrides):
    payload = {
        "hub_url": "http://hub.test",
        "workspace": "acme",
        "workspace_id": 1,
    }
    payload.update(overrides)
    (workspace / ".skore").write_text(json.dumps(payload))


def test_launch_exports_sdk_uri_from_skore_file(tmp_path, monkeypatch):
    captured = _prepare_claude_launch(tmp_path, monkeypatch)
    _write_skore_file(tmp_path)

    _agents.launch_harness(AGENTS["claude-code"], tmp_path)

    assert captured["env"]["SKORE_HUB_URI"] == "http://hub.test"
    assert "SKORE_HUB_API_KEY" not in captured["env"]


def test_launch_keeps_existing_sdk_uri(tmp_path, monkeypatch):
    captured = _prepare_claude_launch(tmp_path, monkeypatch)
    _write_skore_file(tmp_path)
    monkeypatch.setenv("SKORE_HUB_URI", "http://user.hub")

    _agents.launch_harness(AGENTS["claude-code"], tmp_path)

    assert captured["env"]["SKORE_HUB_URI"] == "http://user.hub"


def test_launch_exports_nothing_without_a_skore_file(tmp_path, monkeypatch):
    captured = _prepare_claude_launch(tmp_path, monkeypatch)

    _agents.launch_harness(AGENTS["claude-code"], tmp_path)

    assert "SKORE_HUB_URI" not in captured["env"]


def test_launch_claude_rejects_invalid_settings(tmp_path):
    settings = tmp_path / ".claude"
    settings.mkdir()
    (settings / "settings.local.json").write_text("{")

    with pytest.raises(RuntimeError, match="not valid JSON"):
        _claude._launch_claude(tmp_path, "skore-agent")


def test_launch_claude_rejects_non_object_settings(tmp_path):
    settings = tmp_path / ".claude"
    settings.mkdir()
    (settings / "settings.local.json").write_text("[]")

    with pytest.raises(RuntimeError, match="JSON object"):
        _claude._launch_claude(tmp_path, "skore-agent")


def test_launch_claude_rejects_non_object_env(tmp_path):
    settings = tmp_path / ".claude"
    settings.mkdir()
    (settings / "settings.local.json").write_text(json.dumps({"env": []}))

    with pytest.raises(RuntimeError, match="env is not a JSON object"):
        _claude._launch_claude(tmp_path, "skore-agent")


def test_claude_ui_unavailable_message(monkeypatch):
    monkeypatch.setattr(_process.sys, "platform", "linux")
    assert _claude.claude_ui_unavailable() == "Claude UI is only supported on macOS."

    monkeypatch.setattr(_process.sys, "platform", "darwin")
    assert (
        _claude.claude_ui_unavailable() == "Claude UI is not installed or not on PATH."
    )


def test_launch_rejects_an_incomplete_skore_file(tmp_path, monkeypatch):
    _prepare_claude_launch(tmp_path, monkeypatch)
    (tmp_path / ".skore").write_text(json.dumps({"hub_url": "http://hub.test"}))

    with pytest.raises(RuntimeError, match="missing hub_url"):
        _agents.launch_harness(AGENTS["claude-code"], tmp_path)


def test_launch_harness_errors_when_not_detected(tmp_path, monkeypatch):
    monkeypatch.setattr(_process.shutil, "which", lambda cmd: None)
    with pytest.raises(RuntimeError, match="not installed or not on PATH"):
        _agents.launch_harness(AGENTS["opencode"], tmp_path)


def test_exec_harness_errors_when_executable_missing(monkeypatch):
    monkeypatch.setattr(_process.shutil, "which", lambda cmd: None)
    with pytest.raises(RuntimeError, match="not installed or not on PATH"):
        _process._exec_harness("opencode", ["opencode"])


def test_exec_harness_invokes_execve(monkeypatch):
    recorded: dict[str, object] = {}

    monkeypatch.setattr(_process.shutil, "which", lambda cmd: f"/usr/bin/{cmd}")

    def fake_execve(executable, argv, env):
        recorded["executable"] = executable
        recorded["argv"] = argv
        recorded["env"] = env

    monkeypatch.setattr(_process.os, "execve", fake_execve)
    _process._exec_harness("opencode", ["opencode", "-m", "x"], env={"A": "B"})

    assert recorded["executable"] == "/usr/bin/opencode"
    assert recorded["argv"] == ["opencode", "-m", "x"]
    assert recorded["env"] == {"A": "B"}


def test_detect_copilot_by_code_binary(monkeypatch):
    monkeypatch.setattr(
        _process.shutil,
        "which",
        lambda name: "/usr/bin/code" if name == "code" else None,
    )
    assert is_harness_installed(AGENTS["github-copilot"]) is True
    assert [agent.harness_name for agent in installed_harnesses()] == ["copilot"]


def test_detect_copilot_by_code_insiders(monkeypatch):
    monkeypatch.setattr(
        _process.shutil,
        "which",
        lambda name: "/usr/bin/code-insiders" if name == "code-insiders" else None,
    )
    assert [agent.harness_name for agent in installed_harnesses()] == ["copilot"]


def test_detect_copilot_cli_by_binary(monkeypatch):
    monkeypatch.setattr(
        _process.shutil,
        "which",
        lambda name: "/usr/bin/copilot" if name == "copilot" else None,
    )
    assert is_harness_installed(AGENTS["github-copilot-cli"]) is True
    assert [agent.harness_name for agent in installed_harnesses()] == ["copilot-cli"]


def test_configure_copilot_cli_does_not_write_vscode_config(tmp_path):
    AGENTS["github-copilot-cli"].configure(_ctx(tmp_path))
    assert not (tmp_path / ".vscode").exists()


def test_launch_copilot_cli_sets_provider_env(tmp_path, monkeypatch):
    (tmp_path / ".skore").write_text(
        json.dumps(
            {
                "hub_url": "http://hub.test",
                "workspace": "ws-1",
                "workspace_id": 1,
                "harness": "copilot-cli",
            }
        )
        + "\n"
    )
    _stub_hub_registry(monkeypatch, "secret-key")
    captured: dict[str, object] = {}

    def fake_exec(name, argv, *, env=None):
        captured["argv"] = argv
        captured["env"] = env
        captured["cwd"] = os.getcwd()

    monkeypatch.setattr(
        _process.shutil,
        "which",
        lambda cmd: "/usr/bin/copilot" if cmd == "copilot" else None,
    )
    monkeypatch.setattr(_process, "_exec_harness", fake_exec)
    _agents.launch_harness(
        AGENTS["github-copilot-cli"], tmp_path, model_id="skore-agent"
    )
    assert captured["argv"] == ["copilot"]
    assert captured["cwd"] == str(tmp_path)
    env = captured["env"]
    assert env["COPILOT_PROVIDER_TYPE"] == "openai"
    assert env["COPILOT_PROVIDER_BASE_URL"] == "http://hub.test/v1"
    assert env["COPILOT_PROVIDER_WIRE_API"] == "completions"
    assert env["COPILOT_MODEL"] == "skore-agent"
    assert env["COPILOT_PROVIDER_HEADERS"] == "X-API-Key: secret-key"
    assert env["COPILOT_PROVIDER_MAX_OUTPUT_TOKENS"] == "8192"
    assert env["COPILOT_PROVIDER_MAX_PROMPT_TOKENS"] == "200000"


def test_launch_copilot_cli_errors_without_skore(tmp_path, monkeypatch):
    monkeypatch.setattr(
        _process.shutil,
        "which",
        lambda cmd: "/usr/bin/copilot" if cmd == "copilot" else None,
    )
    with pytest.raises(RuntimeError, match="missing .skore"):
        _agents.launch_harness(AGENTS["github-copilot-cli"], tmp_path)


def test_launch_copilot_cli_errors_without_stored_key(tmp_path, monkeypatch):
    (tmp_path / ".skore").write_text(
        json.dumps(
            {
                "hub_url": "http://hub.test",
                "workspace": "ws-1",
                "workspace_id": 1,
                "harness": "copilot-cli",
            }
        )
        + "\n"
    )
    _stub_hub_registry(monkeypatch, None)
    monkeypatch.setattr(
        _process.shutil,
        "which",
        lambda cmd: "/usr/bin/copilot" if cmd == "copilot" else None,
    )
    with pytest.raises(RuntimeError, match="missing Hub API key"):
        _agents.launch_harness(AGENTS["github-copilot-cli"], tmp_path)


def test_copilot_config_matches_custom_endpoint(tmp_path):
    AGENTS["github-copilot"].configure(_ctx(tmp_path))
    config_path = tmp_path / ".vscode" / "chatLanguageModels.json"
    providers = json.loads(config_path.read_text())
    assert len(providers) == 1
    provider = providers[0]
    assert provider["name"] == "Skore Agent"
    assert provider["vendor"] == "customendpoint"
    assert provider["apiKey"] == "skore"
    assert provider["apiType"] == "chat-completions"
    model = provider["models"][0]
    assert model["id"] == "skore-agent"
    assert model["url"] == "http://hub.test/v1/chat/completions"
    assert model["toolCalling"] is True
    assert model["vision"] is False
    assert model["maxInputTokens"] == 200000
    assert model["maxOutputTokens"] == 8192
    assert model["requestHeaders"] == {"X-API-Key": "secret-key"}
    gitignore = (tmp_path / ".gitignore").read_text().splitlines()
    assert ".vscode/chatLanguageModels.json" in gitignore


def test_configure_copilot_preserves_other_entries(tmp_path):
    config_path = tmp_path / ".vscode" / "chatLanguageModels.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        json.dumps(
            [{"name": "My Other Endpoint", "vendor": "customendpoint", "models": []}]
        )
        + "\n"
    )
    AGENTS["github-copilot"].configure(_ctx(tmp_path))
    providers = json.loads(config_path.read_text())
    assert [entry["name"] for entry in providers] == [
        "My Other Endpoint",
        "Skore Agent",
    ]


def test_launch_copilot_prefers_code_and_opens_workspace(tmp_path, monkeypatch):
    AGENTS["github-copilot"].configure(_ctx(tmp_path))
    user_root = tmp_path / "vscode-user"
    captured: dict[str, list[str]] = {}

    def fake_exec(name, argv, *, env=None):
        captured["argv"] = argv

    monkeypatch.setattr(
        _process.shutil,
        "which",
        lambda cmd: f"/usr/bin/{cmd}" if cmd in {"code", "code-insiders"} else None,
    )
    monkeypatch.setattr(
        _copilot,
        "_copilot_user_config_path",
        lambda binary, home=None: user_root / binary / "chatLanguageModels.json",
    )
    monkeypatch.setattr(_process, "_exec_harness", fake_exec)

    _agents.launch_harness(AGENTS["github-copilot"], tmp_path, model_id="skore-agent")
    assert captured["argv"] == ["code", str(tmp_path)]
    user_config = user_root / "code" / "chatLanguageModels.json"
    providers = json.loads(user_config.read_text())
    assert providers[-1]["name"] == "Skore Agent"
    assert providers[-1]["apiKey"] == "skore"
    assert providers[-1]["models"][0]["requestHeaders"] == {"X-API-Key": "secret-key"}


def test_launch_copilot_falls_back_to_insiders(tmp_path, monkeypatch):
    AGENTS["github-copilot"].configure(_ctx(tmp_path))
    user_root = tmp_path / "vscode-user"
    captured: dict[str, list[str]] = {}

    def fake_exec(name, argv, *, env=None):
        captured["argv"] = argv

    monkeypatch.setattr(
        _process.shutil,
        "which",
        lambda cmd: "/usr/bin/code-insiders" if cmd == "code-insiders" else None,
    )
    monkeypatch.setattr(
        _copilot,
        "_copilot_user_config_path",
        lambda binary, home=None: user_root / binary / "chatLanguageModels.json",
    )
    monkeypatch.setattr(_process, "_exec_harness", fake_exec)
    _agents.launch_harness(AGENTS["github-copilot"], tmp_path, model_id="skore-agent")
    assert captured["argv"] == ["code-insiders", str(tmp_path)]
    user_config = user_root / "code-insiders" / "chatLanguageModels.json"
    assert user_config.is_file()


def test_launch_copilot_errors_when_binary_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(_process.shutil, "which", lambda cmd: None)
    with pytest.raises(RuntimeError, match="not installed or not on PATH"):
        _copilot._launch_copilot(tmp_path, "skore-agent")


def test_launch_copilot_errors_when_project_config_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(
        _process.shutil,
        "which",
        lambda cmd: "/usr/bin/code" if cmd == "code" else None,
    )
    with pytest.raises(RuntimeError, match="missing .vscode/chatLanguageModels.json"):
        _copilot._launch_copilot(tmp_path, "skore-agent")


def test_launch_copilot_errors_when_provider_missing(tmp_path, monkeypatch):
    config_path = tmp_path / ".vscode" / "chatLanguageModels.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(json.dumps([{"name": "Other"}]))
    monkeypatch.setattr(
        _process.shutil,
        "which",
        lambda cmd: "/usr/bin/code" if cmd == "code" else None,
    )

    with pytest.raises(RuntimeError, match="has no Skore Agent provider"):
        _copilot._launch_copilot(tmp_path, "skore-agent")


def test_launch_copilot_errors_when_project_config_unparsable(tmp_path, monkeypatch):
    config_path = tmp_path / ".vscode" / "chatLanguageModels.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("{not json\n")
    monkeypatch.setattr(
        _process.shutil,
        "which",
        lambda cmd: "/usr/bin/code" if cmd == "code" else None,
    )
    with pytest.raises(RuntimeError, match="could not parse"):
        _copilot._launch_copilot(tmp_path, "skore-agent")


def test_launch_copilot_finds_provider_among_others(tmp_path, monkeypatch):
    config_path = tmp_path / ".vscode" / "chatLanguageModels.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        json.dumps(
            [
                {"name": "My Other Endpoint", "vendor": "customendpoint", "models": []},
                {
                    "name": "Skore Agent",
                    "vendor": "customendpoint",
                    "apiKey": "skore",
                    "models": [
                        {
                            "id": "skore-agent",
                            "requestHeaders": {"X-API-Key": "secret-key"},
                        }
                    ],
                },
            ]
        )
        + "\n"
    )
    user_root = tmp_path / "vscode-user"
    captured: dict[str, list[str]] = {}

    def fake_exec(name, argv, *, env=None):
        captured["argv"] = argv

    monkeypatch.setattr(
        _process.shutil,
        "which",
        lambda cmd: "/usr/bin/code" if cmd == "code" else None,
    )
    monkeypatch.setattr(
        _copilot,
        "_copilot_user_config_path",
        lambda binary, home=None: user_root / binary / "chatLanguageModels.json",
    )
    monkeypatch.setattr(_process, "_exec_harness", fake_exec)

    _agents.launch_harness(AGENTS["github-copilot"], tmp_path, model_id="skore-agent")
    assert captured["argv"] == ["code", str(tmp_path)]
    user_config = user_root / "code" / "chatLanguageModels.json"
    providers = json.loads(user_config.read_text())
    assert providers[-1]["name"] == "Skore Agent"
    assert providers[-1]["models"][0]["requestHeaders"] == {"X-API-Key": "secret-key"}


def test_upsert_copilot_provider_preserves_other_entries(tmp_path):
    user_config = tmp_path / "User" / "chatLanguageModels.json"
    user_config.parent.mkdir(parents=True)
    user_config.write_text(
        json.dumps(
            [
                {"name": "Other", "vendor": "openai", "models": []},
                {
                    "name": "Skore Agent",
                    "vendor": "customendpoint",
                    "apiKey": "old-key",
                    "models": [],
                },
            ]
        )
        + "\n"
    )
    provider = {
        "name": "Skore Agent",
        "vendor": "customendpoint",
        "apiKey": "new-key",
        "apiType": "chat-completions",
        "models": [{"id": "skore-agent", "name": "Skore Agent"}],
    }
    _copilot._upsert_copilot_provider(user_config, provider)
    providers = json.loads(user_config.read_text())
    assert len(providers) == 2
    assert providers[0]["name"] == "Other"
    assert providers[1]["apiKey"] == "new-key"


def test_upsert_copilot_provider_errors_on_unparsable_file(tmp_path):
    user_config = tmp_path / "User" / "chatLanguageModels.json"
    user_config.parent.mkdir(parents=True)
    user_config.write_text("// comments are not valid JSON\n[]\n")
    with pytest.raises(RuntimeError, match="could not parse"):
        _copilot._upsert_copilot_provider(user_config, {"name": "Skore Agent"})


def test_upsert_copilot_provider_errors_on_non_list_file(tmp_path):
    user_config = tmp_path / "User" / "chatLanguageModels.json"
    user_config.parent.mkdir(parents=True)
    user_config.write_text('{"name": "not a list"}\n')
    with pytest.raises(RuntimeError, match="could not parse"):
        _copilot._upsert_copilot_provider(user_config, {"name": "Skore Agent"})


@pytest.mark.parametrize(
    ("platform", "expected"),
    [
        ("darwin", ("Library", "Application Support", "Code", "User")),
        ("linux", (".config", "Code", "User")),
    ],
)
def test_copilot_user_config_path_per_platform(
    tmp_path, monkeypatch, platform, expected
):
    monkeypatch.setattr(_process.sys, "platform", platform)
    path = _copilot._copilot_user_config_path("code", home=tmp_path)
    assert path == tmp_path.joinpath(*expected, "chatLanguageModels.json")
    insiders = _copilot._copilot_user_config_path("code-insiders", home=tmp_path)
    assert "Code - Insiders" in str(insiders)


def test_copilot_user_config_path_windows_uses_appdata(tmp_path, monkeypatch):
    monkeypatch.setattr(_process.sys, "platform", "win32")
    monkeypatch.setenv("APPDATA", str(tmp_path / "Roaming"))
    path = _copilot._copilot_user_config_path("code", home=tmp_path)
    assert path == tmp_path / "Roaming" / "Code" / "User" / "chatLanguageModels.json"

    monkeypatch.delenv("APPDATA")
    fallback = _copilot._copilot_user_config_path("code", home=tmp_path)
    assert fallback == tmp_path.joinpath(
        "AppData", "Roaming", "Code", "User", "chatLanguageModels.json"
    )


def test_detect_codex_by_binary(monkeypatch):
    monkeypatch.setattr(
        _process.shutil,
        "which",
        lambda name: "/usr/bin/codex" if name == "codex" else None,
    )
    assert is_harness_installed(AGENTS["codex"]) is True
    assert [agent.harness_name for agent in installed_harnesses()] == ["codex"]


def _codex_home(tmp_path, monkeypatch):
    """Point ``Path.home`` at a scratch dir to detect any global write."""
    monkeypatch.delenv("CODEX_HOME", raising=False)
    home = tmp_path / "home"
    monkeypatch.setattr(_process.Path, "home", classmethod(lambda cls: home))
    return home


def _prepare_codex_launch(tmp_path, monkeypatch) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    def fake_exec(name, argv, *, env=None):
        captured["argv"] = argv
        captured["env"] = env

    monkeypatch.setattr(
        _process.shutil,
        "which",
        lambda cmd: "/usr/bin/codex" if cmd == "codex" else None,
    )
    monkeypatch.setattr(_process, "_exec_harness", fake_exec)
    return captured


def _parse_config_overrides(argv: list[str]) -> dict[str, object]:
    """Parse the ``--config key=value`` tail of a codex argv as one TOML doc."""
    overrides = argv[4::2]
    assert argv[3::2] == ["--config"] * len(overrides)
    return tomllib.loads("\n".join(overrides))


def test_configure_codex_writes_project_config_only(tmp_path, monkeypatch):
    home = _codex_home(tmp_path, monkeypatch)

    AGENTS["codex"].configure(_ctx(tmp_path))

    project = tomllib.loads((tmp_path / ".codex" / "skore-provider.toml").read_text())
    assert project["model"] == "skore-agent"
    assert project["model_provider"] == "skore"
    assert project["base_url"] == "http://hub.test/v1"
    assert project["api_key"] == "secret-key"
    gitignore = (tmp_path / ".gitignore").read_text().splitlines()
    assert ".codex/skore-provider.toml" in gitignore
    assert not home.exists()


def test_configure_codex_overwrites_previous_project_config(tmp_path):
    stale = HarnessContext(
        workspace=tmp_path, hub_url="http://old.test", api_key="old-key"
    )
    AGENTS["codex"].configure(stale)
    AGENTS["codex"].configure(_ctx(tmp_path))
    project = tomllib.loads((tmp_path / ".codex" / "skore-provider.toml").read_text())
    assert project["base_url"] == "http://hub.test/v1"
    assert project["api_key"] == "secret-key"


def test_configure_codex_escapes_toml_special_characters(tmp_path):
    api_key = 'sec"ret\\key'
    ctx = HarnessContext(workspace=tmp_path, hub_url="http://hub.test", api_key=api_key)

    AGENTS["codex"].configure(ctx)

    project = tomllib.loads((tmp_path / ".codex" / "skore-provider.toml").read_text())
    assert project["api_key"] == api_key


def test_launch_codex_passes_provider_as_runtime_overrides(tmp_path, monkeypatch):
    home = _codex_home(tmp_path, monkeypatch)
    captured = _prepare_codex_launch(tmp_path, monkeypatch)
    AGENTS["codex"].configure(_ctx(tmp_path))

    _agents.launch_harness(AGENTS["codex"], tmp_path, model_id="skore-agent")

    argv = captured["argv"]
    assert argv[:3] == ["codex", "--model", "skore-agent"]
    data = _parse_config_overrides(argv)
    assert data["model_provider"] == "skore"
    provider = data["model_providers"]["skore"]
    assert provider["name"] == "Skore Agent"
    assert provider["base_url"] == "http://hub.test/v1"
    assert provider["wire_api"] == "responses"
    assert provider["env_http_headers"] == {"X-API-Key": "SKORE_AGENT_API_KEY"}
    assert "http_headers" not in provider
    assert "env_key" not in provider
    assert "secret-key" not in argv  # the key never rides the command line
    assert captured["env"]["SKORE_AGENT_API_KEY"] == "secret-key"
    assert not home.exists()


def test_launch_codex_prefers_project_model(tmp_path, monkeypatch):
    captured = _prepare_codex_launch(tmp_path, monkeypatch)
    config = tmp_path / ".codex" / "skore-provider.toml"
    config.parent.mkdir(parents=True)
    config.write_text(
        'model = "custom-model"\n'
        'model_provider = "skore"\n'
        'base_url = "http://hub.test/v1"\n'
        'api_key = "secret-key"\n'
    )

    _agents.launch_harness(AGENTS["codex"], tmp_path, model_id="skore-agent")

    assert captured["argv"][:3] == ["codex", "--model", "custom-model"]


def test_launch_codex_falls_back_to_model_argument(tmp_path, monkeypatch):
    captured = _prepare_codex_launch(tmp_path, monkeypatch)
    config = tmp_path / ".codex" / "skore-provider.toml"
    config.parent.mkdir(parents=True)
    config.write_text('base_url = "http://hub.test/v1"\napi_key = "secret-key"\n')

    _agents.launch_harness(AGENTS["codex"], tmp_path, model_id="fallback-model")

    assert captured["argv"][:3] == ["codex", "--model", "fallback-model"]


def test_launch_codex_errors_when_project_config_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(
        _process.shutil,
        "which",
        lambda cmd: "/usr/bin/codex" if cmd == "codex" else None,
    )
    with pytest.raises(RuntimeError, match="missing .codex/skore-provider.toml"):
        _agents.launch_harness(AGENTS["codex"], tmp_path)


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        (
            'model = "skore-agent"\nmodel_provider = "skore"\napi_key = "secret-key"\n',
            "missing a valid base_url",
        ),
        (
            'model = "skore-agent"\n'
            'model_provider = "skore"\n'
            'base_url = ""\n'
            'api_key = "secret-key"\n',
            "missing a valid base_url",
        ),
        (
            'model = "skore-agent"\n'
            'model_provider = "skore"\n'
            'base_url = "http://hub.test/v1"\n',
            "missing a valid api_key",
        ),
        (
            'model = "skore-agent"\n'
            'model_provider = "skore"\n'
            'base_url = "http://hub.test/v1"\n'
            'api_key = ""\n',
            "missing a valid api_key",
        ),
        ("model = [unterminated\n", "could not parse"),
    ],
)
def test_launch_codex_errors_when_project_config_invalid(
    tmp_path, monkeypatch, payload, match
):
    config = tmp_path / ".codex" / "skore-provider.toml"
    config.parent.mkdir(parents=True)
    config.write_text(payload)
    monkeypatch.setattr(
        _process.shutil,
        "which",
        lambda cmd: "/usr/bin/codex" if cmd == "codex" else None,
    )
    with pytest.raises(RuntimeError, match=match):
        _agents.launch_harness(AGENTS["codex"], tmp_path)
