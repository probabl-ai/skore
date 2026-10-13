"""Command-line interface for skore."""

from __future__ import annotations

import rich_click as click

import skore._cli.style  # noqa: F401  (applies the CLI palette and rich-click config)

# These command modules defer their heavy `skore` imports into the callbacks, so
# importing them (to build the CLI / show `--help`) stays instant. Each merges its
# own `rich_click.COMMAND_GROUPS` entry, so import order does not matter.
from skore._cli.agent import agent
from skore._cli.agents import Agent, detect_agent, resolve_skill_agent
from skore._cli.hub import hub
from skore._cli.runtime import is_non_interactive
from skore._cli.skills import skills
from skore._cli.style import SKORE_BANNER, apply_interaction_style, console
from skore._cli.sync import sync

click.rich_click.COMMAND_GROUPS = {
    **getattr(click.rich_click, "COMMAND_GROUPS", {}),
    "cli": [
        {"name": "Agent", "commands": ["agent"]},
        {"name": "Hub", "commands": ["hub"]},
        {"name": "Projects", "commands": ["sync"]},
        {"name": "Skills", "commands": ["skills"]},
    ],
}

_COMMANDS = [
    ("agent", "Authenticate, configure and launch a Skore Hub agent harness."),
    ("hub", "Manage Skore Hub credentials."),
    ("skills", "Install and manage Agent Skills from the probabl-ai/skills release."),
    ("sync", "Synchronize report projects across local storage, Skore Hub and MLflow."),
]


def _render_help(detected: Agent | None) -> str:
    """Build a plain-text help page, agent-flavored when ``detected`` is set."""
    lines = ["Skore command-line interface.", ""]

    if detected is not None:
        lines.append(f"Detected: {detected.label}")
        target = resolve_skill_agent(detected).project_skills_dir
        lines.append(f"Skills target: {target}")
        if detected.harness_name:
            lines.append(f"Harness: {detected.harness_display_name}")
        lines.append("")

    lines.append("Quick start:")

    if detected is not None:
        skill_target = resolve_skill_agent(detected).project_skills_dir
        lines.append(
            f"  skore skills install all      Install all skills to {skill_target}"
        )
    else:
        lines.append("  skore skills install all      Install all skills")
    lines.append("  skore skills install <ids>    Install specific skills by id")

    if detected is not None and detected.harness_name:
        label = detected.harness_display_name
        lines.append(
            f"  skore agent                   "
            f"Configure {label} with the Skore Hub provider"
        )
    else:
        lines.append(
            "  skore agent                   "
            "Configure and launch a Skore Hub agent harness"
        )

    lines.append("")
    lines.append("Commands:")
    for name, desc in _COMMANDS:
        lines.append(f"  {name:<7} {desc}")

    return "\n".join(lines)


class _CLI(click.RichGroup):
    """Root group that restyles help from the process mode at startup."""

    def main(self, *args, **kwargs):
        apply_interaction_style()
        return super().main(*args, **kwargs)


@click.group(invoke_without_command=True, cls=_CLI)
@click.pass_context
@click.version_option(package_name="skore")
def cli(ctx) -> None:
    """Skore command-line interface.

    Use ``skore agent`` to connect a project to the Skore Hub agent, ``skore hub``
    to manage Hub credentials, ``skore sync`` to synchronize report projects, and
    ``skore skills`` to install probabl-skills.
    """
    if ctx.invoked_subcommand is None:
        if is_non_interactive():
            click.echo(_render_help(detect_agent()))
        else:
            console.print(f"[bold cyan]{SKORE_BANNER}[/]")
            click.echo(ctx.get_help())


cli.add_command(skills)
cli.add_command(agent)
cli.add_command(hub)
cli.add_command(sync)
