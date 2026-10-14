from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from skore._cli import agents as _agents
from skore._cli.agents import (
    AGENTS,
    DEFAULT_AGENT,
    HARNESS_CHOICES,
    HARNESS_NAMES,
    SKILL_AGENT_NAMES,
    Agent,
    _missing_skills_directory_message,
    get_harness,
    installed_harnesses,
    is_harness_installed,
    launch_harness,
    normalize_harness_name,
    resolve_skill_agent,
    resolve_targets,
)
from skore._cli.harnesses import bob as _bob
from skore._cli.harnesses import claude as _claude
from skore._cli.harnesses import process as _process
from skore._cli.harnesses.context import HarnessContext
from skore._cli.runtime import is_non_interactive


def test_default_agent():
    assert DEFAULT_AGENT == "agents"


def test_agent_names_match_registry():
    assert SKILL_AGENT_NAMES == [
        "agents",
        "claude-code",
        "cursor",
        "codex",
        "gemini",
        "windsurf",
        "cline",
        "roo",
        "amp",
        "opencode",
        "pi",
        "github-copilot",
        "bob",
        "bob-ide",
    ]


def test_harness_names_come_from_registry():
    assert HARNESS_NAMES == [
        "claude",
        "claude-ui",
        "claude-plugin",
        "cursor",
        "cursor-cli",
        "codex",
        "opencode",
        "pi",
        "copilot",
        "copilot-cli",
        "bob",
        "bob-ide",
    ]


def test_harness_choices_include_canonical_names_and_aliases():
    assert HARNESS_CHOICES[-1] == "bobide"
    assert "bob-ide" in HARNESS_CHOICES
    assert HARNESS_CHOICES.count("bob-ide") == 1
    assert HARNESS_CHOICES.count("bobide") == 1
    assert "claude" in HARNESS_CHOICES
    assert "claude-cli" in HARNESS_CHOICES
    assert "claude-ui" in HARNESS_CHOICES
    assert "claude-plugin" in HARNESS_CHOICES
    assert HARNESS_NAMES.count("claude-plugin") == 1
    assert "claude-plugin-cursor" in HARNESS_CHOICES
    assert "claude-plugin-code" in HARNESS_CHOICES
    assert "claude-plugin-code-insiders" in HARNESS_CHOICES
    assert "claude-plugin-cursor" not in HARNESS_NAMES
    assert "cursor-cli" in HARNESS_CHOICES
    assert "copilot-cli" in HARNESS_CHOICES
    assert "github-copilot-cli" in HARNESS_CHOICES
    assert all(name in HARNESS_CHOICES for name in HARNESS_NAMES)


def test_harness_rows_are_complete():
    harnesses = [agent for agent in AGENTS.values() if agent.harness_name]
    assert all(agent.configure is not None for agent in harnesses)
    assert all(agent.launch is not None for agent in harnesses)


def test_registry_values_are_agents():
    assert all(isinstance(agent, Agent) for agent in AGENTS.values())


def test_agent_is_frozen():
    agent = AGENTS["agents"]
    with pytest.raises(FrozenInstanceError):
        agent.name = "other"  # type: ignore[misc]


def test_resolve_targets_project_scope(tmp_path):
    home = tmp_path / "home"
    project = tmp_path / "project"
    home.mkdir()
    project.mkdir()

    targets = resolve_targets(["agents"], global_=False, home=home, cwd=project)

    assert targets == [("agents", project / ".agents" / "skills")]


def test_resolve_targets_global_scope(tmp_path):
    home = tmp_path / "home"
    project = tmp_path / "project"
    home.mkdir()
    project.mkdir()

    targets = resolve_targets(["cursor"], global_=True, home=home, cwd=project)

    assert targets == [("cursor", home / ".cursor" / "skills")]


def test_resolve_targets_gemini_differs_by_scope(tmp_path):
    home = tmp_path / "home"
    project = tmp_path / "project"
    home.mkdir()
    project.mkdir()

    local = resolve_targets(["gemini"], global_=False, home=home, cwd=project)
    global_ = resolve_targets(["gemini"], global_=True, home=home, cwd=project)

    assert local == [("gemini", project / ".agents" / "skills")]
    assert global_ == [("gemini", home / ".gemini" / "skills")]


def test_resolve_targets_deduplicates_by_directory(tmp_path):
    home = tmp_path / "home"
    project = tmp_path / "project"
    home.mkdir()
    project.mkdir()

    targets = resolve_targets(
        ["agents", "codex"], global_=False, home=home, cwd=project
    )

    assert targets == [("agents", project / ".agents" / "skills")]


def test_resolve_targets_windsurf_differs_by_scope(tmp_path):
    home = tmp_path / "home"
    project = tmp_path / "project"
    home.mkdir()
    project.mkdir()

    local = resolve_targets(["windsurf"], global_=False, home=home, cwd=project)
    global_ = resolve_targets(["windsurf"], global_=True, home=home, cwd=project)

    assert local == [("windsurf", project / ".windsurf" / "skills")]
    assert global_ == [("windsurf", home / ".codeium" / "windsurf" / "skills")]


def test_resolve_targets_amp_differs_by_scope(tmp_path):
    home = tmp_path / "home"
    project = tmp_path / "project"
    home.mkdir()
    project.mkdir()

    local = resolve_targets(["amp"], global_=False, home=home, cwd=project)
    global_ = resolve_targets(["amp"], global_=True, home=home, cwd=project)

    assert local == [("amp", project / ".agents" / "skills")]
    assert global_ == [("amp", home / ".config" / "agents" / "skills")]


def test_resolve_targets_bob_and_bob_ide_deduplicate(tmp_path):
    home = tmp_path / "home"
    project = tmp_path / "project"
    home.mkdir()
    project.mkdir()

    targets = resolve_targets(["bob", "bob-ide"], global_=False, home=home, cwd=project)

    assert targets == [("bob", project / ".bob" / "skills")]


def test_resolve_targets_copilot_project_only(tmp_path):
    home = tmp_path / "home"
    project = tmp_path / "project"
    home.mkdir()
    project.mkdir()

    local = resolve_targets(["github-copilot"], global_=False, home=home, cwd=project)

    assert local == [("github-copilot", project / ".github" / "skills")]
    with pytest.raises(ValueError, match="Copilot in VSCode has no user-level"):
        resolve_targets(["github-copilot"], global_=True, home=home, cwd=project)


def test_resolve_targets_skips_copilot_global_when_requested(tmp_path):
    home = tmp_path / "home"
    project = tmp_path / "project"
    home.mkdir()
    project.mkdir()

    targets = resolve_targets(
        ["github-copilot", "cursor"],
        global_=True,
        home=home,
        cwd=project,
        skip_missing=True,
    )

    assert targets == [("cursor", home / ".cursor" / "skills")]


def test_resolve_targets_uses_defaults(monkeypatch, tmp_path):
    home = tmp_path / "home"
    project = tmp_path / "project"
    home.mkdir()
    project.mkdir()

    monkeypatch.setattr(Path, "home", lambda: home)
    monkeypatch.setattr(Path, "cwd", lambda: project)

    targets = resolve_targets(["agents"], global_=False)

    assert targets == [("agents", project / ".agents" / "skills")]


def test_harness_context_normalizes_hub_url():
    context = HarnessContext(Path("."), "http://hub.test///", "secret")

    assert context.base_url == "http://hub.test/v1"


def test_is_non_interactive_when_agent_is_detected(monkeypatch):
    monkeypatch.delenv("CI", raising=False)
    for name in (
        "CLAUDECODE",
        "CURSOR_AGENT",
        "CURSOR_CLI",
        "GEMINI_CLI",
        "CODEX_SANDBOX",
        "PI_CODING_AGENT",
        "OPENCODE_CLIENT",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPENCODE_CLIENT", "1")

    assert is_non_interactive() is True


def test_agent_env_vars_match_registry():
    from skore._cli.runtime import AGENT_ENV_VARS

    registered = {
        agent.env_var for agent in AGENTS.values() if agent.env_var is not None
    }
    assert registered == set(AGENT_ENV_VARS)


def test_is_non_interactive_in_ci(monkeypatch):
    monkeypatch.setenv("CI", "1")

    assert is_non_interactive() is True


def test_harness_display_name_uses_label():
    assert AGENTS["claude-code"].harness_display_name == "Claude CLI"
    assert AGENTS["claude-ui"].harness_display_name == "Claude UI"
    assert AGENTS["claude-plugin"].harness_display_name == "Claude Plugin"
    assert AGENTS["cursor"].harness_display_name == "Cursor IDE"
    assert AGENTS["cursor-cli"].harness_display_name == "Cursor CLI"
    assert AGENTS["codex"].harness_display_name == "Codex CLI"
    assert AGENTS["github-copilot"].harness_display_name == "Copilot in VSCode"
    assert AGENTS["github-copilot-cli"].harness_display_name == "Copilot CLI"
    assert AGENTS["agents"].harness_display_name == "Agents"


def test_ide_extension_hosts_uses_home(tmp_path, monkeypatch):
    monkeypatch.setattr(_process.Path, "home", lambda: tmp_path)
    assert _claude.ide_extension_hosts() == (
        ("cursor", tmp_path / ".cursor" / "extensions", "cursor"),
        ("code", tmp_path / ".vscode" / "extensions", "vscode"),
        (
            "code-insiders",
            tmp_path / ".vscode-insiders" / "extensions",
            "vscode-insiders",
        ),
    )


def test_harness_registry_helpers(monkeypatch, tmp_path):
    monkeypatch.setattr(_bob, "BOB_IDE_APP_PATH", tmp_path / "absent.app")
    monkeypatch.setattr(_claude, "CLAUDE_UI_APP_PATH", tmp_path / "absent.app")
    monkeypatch.setattr(_claude, "ide_extension_hosts", lambda: ())
    monkeypatch.setattr(
        _process.shutil,
        "which",
        lambda name: "/usr/bin/opencode" if name == "opencode" else None,
    )

    assert get_harness("opencode") is AGENTS["opencode"]
    with pytest.raises(KeyError):
        get_harness("missing")
    assert get_harness("claude-plugin") is AGENTS["claude-plugin"]
    assert get_harness("claude-plugin-cursor") is AGENTS["claude-plugin"]
    assert normalize_harness_name(None) is None
    assert normalize_harness_name("claude-code") == "claude"
    assert normalize_harness_name("claude-cli") == "claude"
    assert normalize_harness_name("claude-ui") == "claude-ui"
    assert normalize_harness_name("claude-plugin") == "claude-plugin"
    assert normalize_harness_name("claude-plugin-cursor") == "claude-plugin-cursor"
    assert normalize_harness_name("cursor-cli") == "cursor-cli"
    assert normalize_harness_name("bobide") == "bob-ide"
    assert normalize_harness_name("unknown") == "unknown"
    assert is_harness_installed(AGENTS["opencode"])
    assert not is_harness_installed(AGENTS["claude-code"])
    assert installed_harnesses() == [AGENTS["opencode"]]


def test_resolve_skill_agent_requires_skill_directories():
    agent = Agent(name="broken", label="Broken")

    with pytest.raises(ValueError, match="has no skills target"):
        resolve_skill_agent(agent)


def test_missing_skills_directory_message_plural():
    assert (
        _missing_skills_directory_message(["Copilot in VSCode", "Other"], global_=True)
        == "Copilot in VSCode, Other have no user-level skills directories."
    )


def test_launch_harness_validates_installation(monkeypatch, tmp_path):
    monkeypatch.setattr(_agents, "is_harness_installed", lambda agent: False)
    with pytest.raises(RuntimeError, match="Claude CLI is not installed"):
        launch_harness(AGENTS["claude-code"], tmp_path)


def test_launch_harness_requires_launcher(monkeypatch, tmp_path):
    agent = Agent(name="fake", label="Fake", harness_name="fake")
    monkeypatch.setattr(_agents, "is_harness_installed", lambda selected: True)

    with pytest.raises(RuntimeError, match="has no harness launcher"):
        launch_harness(agent, tmp_path)


def test_launch_harness_delegates_to_launcher(monkeypatch, tmp_path):
    calls = []
    agent = Agent(
        name="fake",
        label="Fake",
        harness_name="fake",
        launch=lambda workspace, model_id: calls.append((workspace, model_id)),
    )
    monkeypatch.setattr(_agents, "is_harness_installed", lambda selected: True)

    launch_harness(agent, tmp_path, model_id="custom-model")

    assert calls == [(tmp_path, "custom-model")]
