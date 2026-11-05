"""Agent registry: detection, skill targets, and harness lookup."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from skore._cli.harnesses import process as _process
from skore._cli.harnesses.bob import (
    _configure_bob_ide,
    _configure_bob_shell,
    _launch_bob_ide,
    _launch_bob_shell,
    bob_ide_installed,
)
from skore._cli.harnesses.claude import (
    IDE_PLUGIN_LABELS,
    _configure_claude,
    _launch_claude,
    _launch_claude_plugin,
    _launch_claude_ui,
    claude_plugin_ide,
    claude_plugin_installed,
    claude_plugin_unavailable,
    claude_ui_installed,
    claude_ui_unavailable,
)
from skore._cli.harnesses.codex import (
    _configure_codex,
    _launch_codex,
)
from skore._cli.harnesses.context import (
    DEFAULT_AGENT,
    DEFAULT_MODEL_ID,
    Configure,
    Launch,
)
from skore._cli.harnesses.copilot import (
    COPILOT_BINARIES,
    COPILOT_CLI_BINARIES,
    _configure_copilot,
    _configure_copilot_cli,
    _launch_copilot,
    _launch_copilot_cli,
)
from skore._cli.harnesses.cursor import (
    _configure_cursor,
    _launch_cursor,
    _launch_cursor_cli,
    cursor_cli_installed,
)
from skore._cli.harnesses.opencode import (
    _configure_opencode,
    _launch_opencode,
)
from skore._cli.harnesses.pi import _configure_pi, _launch_pi


@dataclass(frozen=True)
class Agent:
    """An agent and its optional detection, skills, and harness capabilities."""

    name: str
    label: str
    env_var: str | None = None
    detection_priority: int | None = None
    skill_target: str | None = None
    user_skills_dir: str | None = None
    project_skills_dir: str | None = None
    harness_name: str | None = None
    harness_label: str | None = None
    harness_binaries: tuple[str, ...] = ()
    harness_aliases: tuple[str, ...] = ()
    configure: Configure | None = None
    launch: Launch | None = None
    installed: Callable[[], bool] | None = None
    unavailable: Callable[[], str] | None = None

    @property
    def harness_display_name(self) -> str:
        """Return the harness label shown to users."""
        return self.harness_label or self.label


def missing_harness_message(agent: Agent) -> str:
    """Return why ``agent`` cannot be launched on this machine."""
    if agent.unavailable is not None:
        return agent.unavailable()
    return f"{agent.harness_display_name} is not installed or not on PATH."


AGENTS: dict[str, Agent] = {
    "agents": Agent(
        name="agents",
        label="Agents",
        user_skills_dir=".agents/skills",
        project_skills_dir=".agents/skills",
    ),
    "claude-code": Agent(
        name="claude-code",
        label="Claude Code",
        env_var="CLAUDECODE",
        detection_priority=0,
        user_skills_dir=".claude/skills",
        project_skills_dir=".claude/skills",
        harness_name="claude",
        harness_label="Claude CLI",
        harness_aliases=("claude-cli",),
        configure=_configure_claude,
        launch=_launch_claude,
    ),
    "claude-ui": Agent(
        name="claude-ui",
        label="Claude UI",
        harness_name="claude-ui",
        harness_label="Claude UI",
        configure=_configure_claude,
        launch=_launch_claude_ui,
        installed=claude_ui_installed,
        unavailable=claude_ui_unavailable,
    ),
    "claude-plugin": Agent(
        name="claude-plugin",
        label="Claude Plugin",
        harness_name="claude-plugin",
        harness_label="Claude Plugin",
        configure=_configure_claude,
        launch=_launch_claude_plugin,
        installed=claude_plugin_installed,
        unavailable=claude_plugin_unavailable,
    ),
    "cursor": Agent(
        name="cursor",
        label="Cursor",
        env_var="CURSOR_AGENT",
        detection_priority=1,
        user_skills_dir=".cursor/skills",
        project_skills_dir=".cursor/skills",
        harness_name="cursor",
        harness_label="Cursor IDE",
        configure=_configure_cursor,
        launch=_launch_cursor,
    ),
    "cursor-cli": Agent(
        name="cursor-cli",
        label="Cursor CLI",
        env_var="CURSOR_CLI",
        detection_priority=6,
        harness_name="cursor-cli",
        harness_label="Cursor CLI",
        configure=_configure_cursor,
        launch=_launch_cursor_cli,
        installed=cursor_cli_installed,
    ),
    "codex": Agent(
        name="codex",
        label="Codex CLI",
        env_var="CODEX_SANDBOX",
        detection_priority=3,
        user_skills_dir=".agents/skills",
        project_skills_dir=".agents/skills",
        harness_name="codex",
        harness_label="Codex CLI",
        configure=_configure_codex,
        launch=_launch_codex,
    ),
    "gemini": Agent(
        name="gemini",
        label="Gemini CLI",
        env_var="GEMINI_CLI",
        detection_priority=2,
        user_skills_dir=".gemini/skills",
        project_skills_dir=".agents/skills",
    ),
    "windsurf": Agent(
        name="windsurf",
        label="Windsurf",
        user_skills_dir=".codeium/windsurf/skills",
        project_skills_dir=".windsurf/skills",
    ),
    "cline": Agent(
        name="cline",
        label="Cline",
        user_skills_dir=".cline/skills",
        project_skills_dir=".cline/skills",
    ),
    "roo": Agent(
        name="roo",
        label="Roo Code",
        user_skills_dir=".roo/skills",
        project_skills_dir=".roo/skills",
    ),
    "amp": Agent(
        name="amp",
        label="Amp",
        user_skills_dir=".config/agents/skills",
        project_skills_dir=".agents/skills",
    ),
    "opencode": Agent(
        name="opencode",
        label="OpenCode",
        env_var="OPENCODE_CLIENT",
        detection_priority=5,
        skill_target=DEFAULT_AGENT,
        harness_name="opencode",
        configure=_configure_opencode,
        launch=_launch_opencode,
    ),
    "pi": Agent(
        name="pi",
        label="Pi",
        env_var="PI_CODING_AGENT",
        detection_priority=4,
        skill_target=DEFAULT_AGENT,
        harness_name="pi",
        configure=_configure_pi,
        launch=_launch_pi,
    ),
    "github-copilot": Agent(
        name="github-copilot",
        label="Copilot in VSCode",
        project_skills_dir=".github/skills",
        harness_name="copilot",
        harness_binaries=COPILOT_BINARIES,
        configure=_configure_copilot,
        launch=_launch_copilot,
    ),
    "github-copilot-cli": Agent(
        name="github-copilot-cli",
        label="GitHub Copilot CLI",
        harness_name="copilot-cli",
        harness_label="Copilot CLI",
        harness_binaries=COPILOT_CLI_BINARIES,
        harness_aliases=("github-copilot-cli",),
        configure=_configure_copilot_cli,
        launch=_launch_copilot_cli,
    ),
    "bob": Agent(
        name="bob",
        label="Bob Shell",
        user_skills_dir=".bob/skills",
        project_skills_dir=".bob/skills",
        harness_name="bob",
        harness_label="Bob Shell",
        configure=_configure_bob_shell,
        launch=_launch_bob_shell,
    ),
    "bob-ide": Agent(
        name="bob-ide",
        label="Bob IDE",
        user_skills_dir=".bob/skills",
        project_skills_dir=".bob/skills",
        harness_name="bob-ide",
        harness_label="Bob IDE",
        harness_binaries=("bobide",),
        harness_aliases=("bobide",),
        configure=_configure_bob_ide,
        launch=_launch_bob_ide,
        installed=bob_ide_installed,
    ),
}

SKILL_AGENTS = {
    name: agent
    for name, agent in AGENTS.items()
    if agent.project_skills_dir is not None
    or agent.user_skills_dir is not None
    or agent.skill_target is not None
}
SKILL_AGENT_NAMES = list(SKILL_AGENTS)
HARNESS_BY_NAME = {
    agent.harness_name: agent
    for agent in AGENTS.values()
    if agent.harness_name is not None
}
HARNESS_NAMES = list(HARNESS_BY_NAME)
HARNESS_ALIASES: dict[str, str] = {}
for _harness in HARNESS_BY_NAME.values():
    canonical = _harness.harness_name
    assert canonical is not None
    for _token in (canonical, _harness.name, *_harness.harness_aliases):
        HARNESS_ALIASES.setdefault(_token, canonical)
# ``--harness claude-plugin-{ide}`` selects that IDE and is stored as-is.
# It is not a picker row and not an alias: aliases collapse to the canonical name.
CLAUDE_PLUGIN_IDE_CHOICES = tuple(
    f"claude-plugin-{binary}" for binary in IDE_PLUGIN_LABELS
)
# Canonical harness names plus their aliases (e.g. ``bobide`` for Bob IDE, the
# command it installs). Accepted by ``--harness``; still stored as the canonical
# name in ``.skore``.
HARNESS_CHOICES = [
    name
    for agent in AGENTS.values()
    if agent.harness_name is not None
    for name in (
        (agent.harness_name, *CLAUDE_PLUGIN_IDE_CHOICES, *agent.harness_aliases)
        if agent.harness_name == "claude-plugin"
        else (agent.harness_name, *agent.harness_aliases)
    )
]


def detect_agent() -> Agent | None:
    """Return the calling agent detected from the environment, if any."""
    detectable = sorted(
        (agent for agent in AGENTS.values() if agent.env_var is not None),
        key=lambda agent: (
            agent.detection_priority
            if agent.detection_priority is not None
            else len(AGENTS)
        ),
    )
    for agent in detectable:
        if os.environ.get(agent.env_var or "", ""):
            return agent
    return None


def resolve_skill_agent(agent: Agent) -> Agent:
    """Return the registry row that owns ``agent``'s skills directories."""
    target = AGENTS[agent.skill_target] if agent.skill_target else agent
    if target.user_skills_dir is None and target.project_skills_dir is None:
        raise ValueError(f"{agent.name} has no skills target")
    return target


def _missing_skills_directory_message(labels: list[str], *, global_: bool) -> str:
    """Return an error for agents that have no directory in the given scope."""
    scope = "user-level" if global_ else "project"
    if len(labels) == 1:
        return f"{labels[0]} has no {scope} skills directory."
    return f"{', '.join(labels)} have no {scope} skills directories."


def resolve_targets(
    agent_names: list[str],
    *,
    global_: bool,
    home: Path | None = None,
    cwd: Path | None = None,
    skip_missing: bool = False,
) -> list[tuple[str, Path]]:
    """Resolve skill agent names to unique global or project directories.

    Parameters
    ----------
    skip_missing : bool, default=False
        When True, agents with no directory for the requested scope are
        skipped. When False, those agents raise ``ValueError``.
    """
    home = home or Path.home()
    cwd = cwd or Path.cwd()

    targets: list[tuple[str, Path]] = []
    seen: set[Path] = set()
    missing: list[str] = []
    for name in agent_names:
        agent = resolve_skill_agent(AGENTS[name])
        subdir = agent.user_skills_dir if global_ else agent.project_skills_dir
        if subdir is None:
            if skip_missing:
                continue
            missing.append(agent.label)
            continue
        path = (home if global_ else cwd) / subdir
        if path in seen:
            continue
        seen.add(path)
        targets.append((name, path))
    if missing:
        raise ValueError(_missing_skills_directory_message(missing, global_=global_))
    return targets


def get_harness(name: str) -> Agent:
    """Return the agent that provides the named harness."""
    if claude_plugin_ide(name):
        name = "claude-plugin"
    try:
        return HARNESS_BY_NAME[name]
    except KeyError:
        raise KeyError(name) from None


def normalize_harness_name(name: str | None) -> str | None:
    """Return the current harness name for a persisted agent or harness name."""
    if name is None:
        return None
    return HARNESS_ALIASES.get(name, name)


def is_harness_installed(agent: Agent) -> bool:
    """Return whether the harness is on PATH or, on macOS, its app bundle."""
    if agent.installed is not None:
        return agent.installed()
    binaries = agent.harness_binaries or (
        (agent.harness_name,) if agent.harness_name else ()
    )
    return any(_process.shutil.which(binary) for binary in binaries)


def installed_harnesses() -> list[Agent]:
    """Return harness-capable agents whose executables are on ``PATH``."""
    return [
        agent
        for agent in AGENTS.values()
        if agent.harness_name is not None and is_harness_installed(agent)
    ]


def launch_harness(
    agent: Agent, workspace: Path, *, model_id: str = DEFAULT_MODEL_ID
) -> None:
    """Launch ``agent``'s harness in ``workspace``."""
    from skore._cli.skore import apply_hub_uri
    from skore._cli.style import console
    from skore._plugins.hub.authentication import HOST_ENV_VAR_NAME

    if not is_harness_installed(agent):
        raise RuntimeError(missing_harness_message(agent))
    if agent.launch is None:
        raise RuntimeError(f"{agent.name} has no harness launcher.")
    console.print(
        f"[skore.ok]Launching[/] [skore.skill]{agent.harness_display_name}[/] ..."
    )
    from skore._cli.project import SkoreConfig, SkoreConfigError

    if HOST_ENV_VAR_NAME not in os.environ:
        try:
            config = SkoreConfig.load(workspace)
        except SkoreConfigError as error:
            raise RuntimeError(str(error)) from error
        if config is not None:
            apply_hub_uri(config.hub_url)
    agent.launch(workspace, model_id)
