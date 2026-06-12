"""Agent registry and target-directory resolution following agentskills.io."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DEFAULT_AGENT = "agents"


@dataclass(frozen=True)
class Agent:
    """An AI agent and the directories where its skills live.

    Attributes
    ----------
    name : str
        The identifier of the agent.
    home_marker : str
        The home-relative directory whose presence signals the agent is
        installed (e.g. ``.claude``).
    user_subdir : str
        The home-relative skills directory used for a global installation.
    project_subdir : str
        The cwd-relative skills directory used for a project installation.
    """

    name: str
    home_marker: str
    user_subdir: str
    project_subdir: str


AGENTS: dict[str, Agent] = {
    "agents": Agent("agents", ".agents", ".agents/skills", ".agents/skills"),
    "claude-code": Agent("claude-code", ".claude", ".claude/skills", ".claude/skills"),
    "cursor": Agent("cursor", ".cursor", ".cursor/skills", ".cursor/skills"),
    "codex": Agent("codex", ".codex", ".agents/skills", ".agents/skills"),
    "gemini": Agent("gemini", ".gemini", ".gemini/skills", ".agents/skills"),
}

AGENT_NAMES = list(AGENTS)


def resolve_targets(
    agent_names: list[str],
    *,
    global_: bool,
    home: Path | None = None,
    cwd: Path | None = None,
) -> list[tuple[str, Path]]:
    """Resolve agent names to their skills directories.

    Parameters
    ----------
    agent_names : list of str
        The agents to resolve.
    global_ : bool
        Whether to target the user-level (``True``) or project-level
        (``False``) directories.
    home : Path or None, default=None
        The home directory. Defaults to ``Path.home()``.
    cwd : Path or None, default=None
        The current working directory. Defaults to ``Path.cwd()``.

    Returns
    -------
    list of (str, Path)
        Pairs of agent name and skills directory, deduplicated by directory.
    """
    home = home or Path.home()
    cwd = cwd or Path.cwd()

    targets: list[tuple[str, Path]] = []
    seen: set[Path] = set()
    for name in agent_names:
        agent = AGENTS[name]
        path = home / agent.user_subdir if global_ else cwd / agent.project_subdir

        if path in seen:
            continue

        seen.add(path)
        targets.append((name, path))

    return targets
