"""Tests for agent environment-variable detection."""

from __future__ import annotations

import pytest

from skore._cli.agents import AGENTS, detect_agent, resolve_skill_agent


def _clear_agent_envs(monkeypatch):
    for var in (
        "CLAUDECODE",
        "CURSOR_AGENT",
        "CURSOR_CLI",
        "GEMINI_CLI",
        "CODEX_SANDBOX",
        "PI_CODING_AGENT",
        "OPENCODE_CLIENT",
    ):
        monkeypatch.delenv(var, raising=False)


def test_no_env_vars_returns_none(monkeypatch):
    _clear_agent_envs(monkeypatch)
    assert detect_agent() is None


def test_claude_code_detected(monkeypatch):
    _clear_agent_envs(monkeypatch)
    monkeypatch.setenv("CLAUDECODE", "1")
    result = detect_agent()
    assert result is AGENTS["claude-code"]


def test_cursor_detected(monkeypatch):
    _clear_agent_envs(monkeypatch)
    monkeypatch.setenv("CURSOR_AGENT", "1")
    result = detect_agent()
    assert result is AGENTS["cursor"]


def test_cursor_cli_detected(monkeypatch):
    _clear_agent_envs(monkeypatch)
    monkeypatch.setenv("CURSOR_CLI", "1")
    result = detect_agent()
    assert result is AGENTS["cursor-cli"]


def test_priority_cursor_ide_before_cursor_cli(monkeypatch):
    _clear_agent_envs(monkeypatch)
    monkeypatch.setenv("CURSOR_AGENT", "1")
    monkeypatch.setenv("CURSOR_CLI", "1")
    assert detect_agent().name == "cursor"


def test_gemini_detected(monkeypatch):
    _clear_agent_envs(monkeypatch)
    monkeypatch.setenv("GEMINI_CLI", "1")
    result = detect_agent()
    assert result is AGENTS["gemini"]


def test_codex_detected(monkeypatch):
    _clear_agent_envs(monkeypatch)
    monkeypatch.setenv("CODEX_SANDBOX", "seatbelt")
    result = detect_agent()
    assert result is AGENTS["codex"]


def test_pi_detected_any_nonempty(monkeypatch):
    _clear_agent_envs(monkeypatch)
    monkeypatch.setenv("PI_CODING_AGENT", "true")
    assert detect_agent() is AGENTS["pi"]


def test_pi_detected_session_id(monkeypatch):
    _clear_agent_envs(monkeypatch)
    monkeypatch.setenv("PI_CODING_AGENT", "some-session-id")
    assert detect_agent() is AGENTS["pi"]


def test_opencode_detected(monkeypatch):
    _clear_agent_envs(monkeypatch)
    monkeypatch.setenv("OPENCODE_CLIENT", "1")
    result = detect_agent()
    assert result is AGENTS["opencode"]


def test_empty_value_not_detected(monkeypatch):
    _clear_agent_envs(monkeypatch)
    monkeypatch.setenv("PI_CODING_AGENT", "")
    assert detect_agent() is None


def test_priority_claude_before_cursor(monkeypatch):
    _clear_agent_envs(monkeypatch)
    monkeypatch.setenv("CLAUDECODE", "1")
    monkeypatch.setenv("CURSOR_AGENT", "1")
    assert detect_agent().name == "claude-code"


def test_priority_cursor_before_gemini(monkeypatch):
    _clear_agent_envs(monkeypatch)
    monkeypatch.setenv("CURSOR_AGENT", "1")
    monkeypatch.setenv("GEMINI_CLI", "1")
    assert detect_agent().name == "cursor"


def test_priority_gemini_before_codex(monkeypatch):
    _clear_agent_envs(monkeypatch)
    monkeypatch.setenv("GEMINI_CLI", "1")
    monkeypatch.setenv("CODEX_SANDBOX", "seatbelt")
    assert detect_agent().name == "gemini"


def test_priority_codex_before_pi(monkeypatch):
    _clear_agent_envs(monkeypatch)
    monkeypatch.setenv("CODEX_SANDBOX", "seatbelt")
    monkeypatch.setenv("PI_CODING_AGENT", "true")
    assert detect_agent().name == "codex"


def test_priority_pi_before_opencode(monkeypatch):
    _clear_agent_envs(monkeypatch)
    monkeypatch.setenv("PI_CODING_AGENT", "true")
    monkeypatch.setenv("OPENCODE_CLIENT", "1")
    assert detect_agent().name == "pi"


@pytest.mark.parametrize(
    "env_var,value,expected",
    [
        ("CLAUDECODE", "1", "claude"),
        ("CURSOR_AGENT", "1", "cursor"),
        ("CURSOR_CLI", "1", "cursor-cli"),
        ("GEMINI_CLI", "1", None),
        ("CODEX_SANDBOX", "seatbelt", "codex"),
        ("PI_CODING_AGENT", "true", "pi"),
        ("OPENCODE_CLIENT", "1", "opencode"),
    ],
)
def test_harness_mapping(monkeypatch, env_var, value, expected):
    _clear_agent_envs(monkeypatch)
    monkeypatch.setenv(env_var, value)
    detected = detect_agent()
    assert detected.harness_name == expected


@pytest.mark.parametrize(
    "env_var,value,expected",
    [
        ("CLAUDECODE", "1", "claude-code"),
        ("CURSOR_AGENT", "1", "cursor"),
        ("GEMINI_CLI", "1", "gemini"),
        ("CODEX_SANDBOX", "seatbelt", "codex"),
        ("PI_CODING_AGENT", "true", "agents"),
        ("OPENCODE_CLIENT", "1", "agents"),
    ],
)
def test_skill_agent_mapping(monkeypatch, env_var, value, expected):
    _clear_agent_envs(monkeypatch)
    monkeypatch.setenv(env_var, value)
    detected = detect_agent()
    assert resolve_skill_agent(detected).name == expected
