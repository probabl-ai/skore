"""The ``skore skills`` command group to manage Agent Skills."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

import rich_click as click
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.tree import Tree

from skore._cli._style import console
from skore._cli.skills._agents import AGENT_NAMES, DEFAULT_AGENT, resolve_targets
from skore._cli.skills._catalog import fetch_release
from skore._cli.skills.app import ProbablSkillsFinder, ProbablSkillsInstaller

SIDECAR = ".skore-skill.json"

click.rich_click.COMMAND_GROUPS = {
    "cli skills": [
        {"name": "Discover", "commands": ["find", "list"]},
        {"name": "Manage", "commands": ["install", "update", "remove"]},
    ],
}


def _agent_option(func):
    return click.option(
        "--agent",
        "-a",
        multiple=True,
        type=click.Choice(AGENT_NAMES),
        help="Target agent(s). Defaults to the .agents/ cross-client directory.",
    )(func)


def _global_option(func):
    return click.option(
        "--global",
        "-g",
        "global_",
        is_flag=True,
        help="Target the user directory instead of the current project.",
    )(func)


def _index(catalog: dict[str, Any]) -> tuple[dict, dict]:
    skills = {skill["id"]: skill for skill in catalog.get("skills", [])}
    workflows = {workflow["id"]: workflow for workflow in catalog.get("workflows", [])}
    return skills, workflows


def _expand(ids: list[str], skills: dict, workflows: dict) -> list[dict]:
    selected: dict[str, dict] = {}
    for identifier in ids:
        if identifier in workflows:
            for skill_id in workflows[identifier]["includes"]:
                selected[skill_id] = skills[skill_id]
        elif identifier in skills:
            selected[identifier] = skills[identifier]
        else:
            raise click.ClickException(f"Unknown skill or workflow: {identifier!r}")
    return list(selected.values())


def _install_skill(skill: dict, root: Path, target: Path, tag: str) -> None:
    source = root / skill["path"]
    destination = target / skill["id"]

    if destination.exists():
        shutil.rmtree(destination)

    shutil.copytree(source, destination)

    sidecar = {"id": skill["id"], "release": tag, "hash": skill["hash"]}
    (destination / SIDECAR).write_text(json.dumps(sidecar, indent=2))


def _installed(target: Path):
    if not target.is_dir():
        return

    for child in sorted(target.iterdir()):
        sidecar = child / SIDECAR
        if sidecar.is_file():
            yield child, json.loads(sidecar.read_text())


def _matches(entry: dict, query: str) -> bool:
    fields = (
        entry.get("id"),
        entry.get("title"),
        entry.get("summary"),
        entry.get("category"),
    )
    return any(field and query in field.lower() for field in fields)


def _is_interactive() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def _render_catalog(
    catalog: dict[str, Any],
    *,
    query: str | None = None,
    ids: list[str] | None = None,
) -> None:
    workflows = catalog.get("workflows", [])
    skills = catalog.get("skills", [])

    if ids is not None:
        wanted = set(ids)
        workflows = [entry for entry in workflows if entry["id"] in wanted]
        skills = [entry for entry in skills if entry["id"] in wanted]
    elif query:
        needle = query.lower()
        workflows = [entry for entry in workflows if _matches(entry, needle)]
        skills = [entry for entry in skills if _matches(entry, needle)]

    if workflows:
        table = Table(title="Workflows (recommended)")
        table.add_column("id", style="cyan")
        table.add_column("summary")
        for workflow in workflows:
            table.add_row(workflow["id"], workflow.get("summary", ""))
        console.print(table)

    if skills:
        table = Table(title="Skills")
        table.add_column("id", style="cyan")
        table.add_column("summary")
        for skill in skills:
            table.add_row(skill["id"], skill.get("summary", ""))
        console.print(table)

    if not workflows and not skills:
        console.print("No skill or workflow matches the query.")


def _resolve_agent_names(agent: tuple[str, ...]) -> list[str]:
    return list(agent) if agent else [DEFAULT_AGENT]


def _interactive_install_options(
    catalog: dict[str, Any],
    *,
    agent: tuple[str, ...],
    default_global: bool,
) -> tuple[list[dict], list[str], bool] | None:
    """Run the tabbed Textual wizard to choose skills, agents and scope.

    Parameters
    ----------
    catalog : dict
        The parsed ``catalog.json`` content.
    agent : tuple of str
        Agents passed on the command line; when non-empty the agent step is
        skipped.
    default_global : bool
        The pre-selected scope (``True`` for the user-level directory).

    Returns
    -------
    tuple or None
        ``(selected_skills, agent_names, global_)`` or ``None`` when the user
        aborts or selects nothing.
    """
    skills_by_id, workflows_by_id = _index(catalog)

    app = ProbablSkillsInstaller(catalog, agent=agent, default_global=default_global)
    app.run()

    if app.result is None:
        return None

    selected_ids, agent_names, global_ = app.result
    if not selected_ids or not agent_names:
        return None

    selected = _expand(selected_ids, skills_by_id, workflows_by_id)
    return selected, agent_names, global_


def _interactive_find(catalog: dict[str, Any]) -> None:
    """Pick catalog entries with the Textual finder and render their details."""
    app = ProbablSkillsFinder(catalog)
    app.run()

    if not app.result:
        return

    _render_catalog(catalog, ids=app.result)


@click.group()
def skills() -> None:
    """Install and manage Agent Skills from the probabl-ai/skills release."""


@skills.command("install")
@click.argument("ids", nargs=-1)
@_agent_option
@_global_option
@click.option(
    "--all",
    "all_",
    is_flag=True,
    help="Install every skill from the latest release (non-interactive).",
)
@click.option(
    "--list",
    "-l",
    "list_only",
    is_flag=True,
    help="List available skills and workflows without installing.",
)
def install(ids, agent, global_, all_, list_only) -> None:
    """Install skill(s) or workflow(s) from the latest release.

    Run without arguments to launch the interactive installer (a tabbed wizard
    for the selection, target agent and install scope).

    Pass skill or workflow ids (or ``--all``) to install non-interactively,
    optionally with ``--agent`` and ``--global`` to choose the targets and
    scope. ``--agent``/``--global`` require an explicit selection.
    """
    with console.status("Fetching latest skills release...", spinner="dots"):
        tag, root, catalog = fetch_release()

    if list_only:
        _render_catalog(catalog)
        return

    skills_by_id, workflows_by_id = _index(catalog)
    ids = list(ids)

    if ids or all_ or agent or global_:
        if not (ids or all_):
            raise click.UsageError(
                "Specify skill/workflow ids or --all to install non-interactively."
            )
        selected = (
            list(skills_by_id.values())
            if all_
            else _expand(ids, skills_by_id, workflows_by_id)
        )
        agent_names = _resolve_agent_names(agent)
    else:
        options = _interactive_install_options(
            catalog, agent=agent, default_global=global_
        )
        if not options:
            console.print("Nothing selected.")
            return
        selected, agent_names, global_ = options

    targets = resolve_targets(agent_names, global_=global_)

    tree = Tree(f"Installing {len(selected)} skill(s) from release {tag}")
    for _, target in targets:
        branch = tree.add(f"[skore.path]{target}[/]")
        for skill in selected:
            branch.add(
                f"[skore.skill]{skill['id']}[/]  [skore.muted]{skill['summary']}[/]"
            )
    console.print(tree)

    with Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Installing", total=len(selected) * len(targets))
        for _, target in targets:
            for skill in selected:
                _install_skill(skill, root, target, tag)
                progress.advance(task)

    console.print(
        f"[skore.ok]+[/] installed [skore.skill]{len(selected)}[/] skill(s) "
        f"into {len(targets)} location(s) from release {tag}"
    )


@skills.command("list")
@_global_option
def list_skills(global_) -> None:
    """List installed skills."""
    targets = resolve_targets([DEFAULT_AGENT], global_=global_)

    table = Table(title="Installed skills")
    table.add_column("id", style="skore.skill")
    table.add_column("release")
    table.add_column("location")

    found = False
    for _, target in targets:
        for _, sidecar in _installed(target):
            found = True
            table.add_row(sidecar["id"], sidecar.get("release", ""), str(target))

    if found:
        console.print(table)
    else:
        console.print("No skills installed.")


@skills.command("find")
@click.argument("query", required=False)
def find(query) -> None:
    """Search available skills and workflows in the latest release.

    Without a query in an interactive terminal, an interactive picker is
    launched and the chosen entries are rendered; otherwise the matching
    catalog entries are listed.
    """
    with console.status("Fetching latest skills release...", spinner="dots"):
        _, _, catalog = fetch_release()

    if query is None and _is_interactive():
        _interactive_find(catalog)
        return

    _render_catalog(catalog, query=query)


@skills.command("update")
@click.argument("ids", nargs=-1)
@_global_option
@click.option("--all", "all_", is_flag=True, help="Update every installed skill.")
def update(ids, global_, all_) -> None:
    """Update installed skills to the latest release.

    Pass skill ids to update, or ``--all`` to update every installed skill.
    Run ``skore skills find`` to discover ids.
    """
    if not all_ and not ids:
        raise click.UsageError(
            "Specify skill ids to update or pass --all. "
            "Run `skore skills find` to discover ids."
        )

    with console.status("Fetching latest skills release...", spinner="dots"):
        tag, root, catalog = fetch_release()
    skills_by_id, _ = _index(catalog)
    targets = resolve_targets([DEFAULT_AGENT], global_=global_)

    requested = set(ids)
    updated = []
    for _, target in targets:
        for _, sidecar in _installed(target):
            skill_id = sidecar["id"]
            if not all_ and skill_id not in requested:
                continue

            skill = skills_by_id.get(skill_id)
            if skill is None or sidecar.get("hash") == skill["hash"]:
                continue

            _install_skill(skill, root, target, tag)
            updated.append((skill_id, target))

    if updated:
        for skill_id, target in updated:
            console.print(
                f"[skore.ok]^[/] updated [skore.skill]{skill_id}[/] -> "
                f"[skore.path]{target}[/]"
            )
    else:
        console.print("All skills are up to date.")


@skills.command("remove")
@click.argument("ids", nargs=-1)
@_global_option
@click.option("--all", "all_", is_flag=True, help="Remove every installed skill.")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompts.")
def remove(ids, global_, all_, yes) -> None:
    """Remove installed skills.

    Pass skill ids to remove, or ``--all`` to remove every installed skill.
    Run ``skore skills find`` to discover ids.
    """
    if not all_ and not ids:
        raise click.UsageError(
            "Specify skill ids to remove or pass --all. "
            "Run `skore skills find` to discover ids."
        )

    targets = resolve_targets([DEFAULT_AGENT], global_=global_)

    to_remove = []
    for _, target in targets:
        if all_:
            for skill_dir, _ in _installed(target):
                to_remove.append(skill_dir)
        else:
            for skill_id in ids:
                skill_dir = target / skill_id
                if (skill_dir / SIDECAR).is_file():
                    to_remove.append(skill_dir)

    if not to_remove:
        console.print("Nothing to remove.")
        return

    if not yes:
        locations = ", ".join(str(skill_dir) for skill_dir in to_remove)
        console.print(f"Removing {locations}")
        if not click.confirm("Proceed?", default=True):
            return

    for skill_dir in to_remove:
        shutil.rmtree(skill_dir)
        console.print(f"[skore.ok]-[/] removed [skore.path]{skill_dir}[/]")
