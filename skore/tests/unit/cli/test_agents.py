from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from skore._cli.skills._agents import (
    AGENT_NAMES,
    AGENTS,
    DEFAULT_AGENT,
    Agent,
    resolve_targets,
)


def test_default_agent():
    assert DEFAULT_AGENT == "agents"


def test_agent_names_match_registry():
    assert list(AGENTS) == AGENT_NAMES


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


def test_resolve_targets_uses_defaults(monkeypatch, tmp_path):
    home = tmp_path / "home"
    project = tmp_path / "project"
    home.mkdir()
    project.mkdir()

    monkeypatch.setattr(Path, "home", lambda: home)
    monkeypatch.setattr(Path, "cwd", lambda: project)

    targets = resolve_targets(["agents"], global_=False)

    assert targets == [("agents", project / ".agents" / "skills")]
