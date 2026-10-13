"""Process mode for the CLI, without the harness registry."""

from __future__ import annotations

import os
import sys

# Non-empty values mean a coding agent is driving this process. Kept beside
# ``is_non_interactive`` so styling does not import harness configuration.
# These names match ``Agent.env_var`` in the agent registry.
AGENT_ENV_VARS = (
    "CLAUDECODE",
    "CURSOR_AGENT",
    "CURSOR_CLI",
    "GEMINI_CLI",
    "CODEX_SANDBOX",
    "PI_CODING_AGENT",
    "OPENCODE_CLIENT",
)


def is_non_interactive() -> bool:
    """Return whether the CLI should avoid TUIs and emit plain-text output."""
    if os.environ.get("CI"):
        return True
    if any(os.environ.get(name) for name in AGENT_ENV_VARS):
        return True
    return not (sys.stdin.isatty() and sys.stdout.isatty())
