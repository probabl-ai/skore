"""The ``skore skills`` command group to manage Agent Skills."""

from __future__ import annotations

import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import Any

import rich_click as click
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.tree import Tree

from skore._cli.agents import (
    DEFAULT_AGENT,
    SKILL_AGENT_NAMES,
    detect_agent,
    resolve_skill_agent,
    resolve_targets,
)
from skore._cli.runtime import is_non_interactive
from skore._cli.skills.app import (
    InstalledSkillsPicker,
    ProbablSkillsInstaller,
)
from skore._cli.skills.catalog import GITHUB_REPO
from skore._cli.skills.release import ReleaseError, cleanup_release, load_release
from skore._cli.skills.render import render_catalog
from skore._cli.skills.store import (
    SIDECAR,
    SkillStoreError,
    install_skill,
    installed,
    persist_source_catalog,
    prune_local_catalog,
    skill_repository,
    write_sidecar,
)
from skore._cli.style import console

click.rich_click.COMMAND_GROUPS = {
    **getattr(click.rich_click, "COMMAND_GROUPS", {}),
    "cli skills": [
        {"name": "Discover", "commands": ["list"]},
        {"name": "Manage", "commands": ["install", "update", "remove"]},
    ],
}


def _agent_option(func):
    return click.option(
        "--agent",
        "-a",
        multiple=True,
        type=click.Choice(SKILL_AGENT_NAMES),
        help="Target agent(s). Defaults to the ``.agents/`` cross-client directory.",
    )(func)


def _global_option(func):
    return click.option(
        "--global",
        "-g",
        "global_",
        is_flag=True,
        help="Target the user directory instead of the current project.",
    )(func)


def _manage_agent_option(func):
    return click.option(
        "--agent",
        "-a",
        multiple=True,
        type=click.Choice(SKILL_AGENT_NAMES),
        help="Restrict to specific agent(s). Defaults to all known agents.",
    )(func)


def _manage_targets(agent: tuple[str, ...], *, global_: bool) -> list[tuple[str, Path]]:
    """Resolve the directories scanned by ``list``/``update``/``remove``.

    With no explicit ``--agent`` every known agent is scanned so that skills
    installed into any client directory remain discoverable and manageable.
    """
    agent_names = list(agent) if agent else SKILL_AGENT_NAMES
    try:
        return resolve_targets(agent_names, global_=global_, skip_missing=not agent)
    except ValueError as error:
        raise click.ClickException(str(error)) from error


def _report_store_errors(func):
    """Translate local catalog failures into Click errors."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except SkillStoreError as error:
            raise click.ClickException(str(error)) from error

    return wrapper


@contextmanager
def _release(repo: str = GITHUB_REPO) -> Iterator[tuple[str, Path, dict[str, Any]]]:
    """Fetch the latest release and clean up its extracted files afterwards.

    Surfaces network/parsing failures as a :class:`click.ClickException` rather
    than a raw traceback, and removes the temporary extraction directory once
    the wrapped command is done with it.
    """
    with console.status(
        f"Fetching latest skills release from {repo}...", spinner="dots"
    ):
        try:
            tag, root, catalog = load_release(repo)
        except ReleaseError as error:
            raise click.ClickException(str(error)) from error

    try:
        yield tag, root, catalog
    finally:
        cleanup_release(root)


def _format_sources(repos: list[str]) -> str:
    """Join repository names for user-facing messages."""
    return ", ".join(repos)


def _index(catalog: dict[str, Any]) -> tuple[dict, dict]:
    skills = {skill["id"]: skill for skill in catalog.get("skills", [])}
    workflows = {workflow["id"]: workflow for workflow in catalog.get("workflows", [])}
    return skills, workflows


def _expand(ids: list[str], skills: dict, workflows: dict) -> list[dict]:
    selected: dict[str, dict] = {}
    for identifier in ids:
        if identifier in workflows:
            for skill_id in workflows[identifier].get("includes", []):
                skill = skills.get(skill_id)
                if skill is None:
                    raise click.ClickException(
                        f"Workflow {identifier!r} references unknown skill "
                        f"{skill_id!r}."
                    )
                selected[skill_id] = skill
        elif identifier in skills:
            selected[identifier] = skills[identifier]
        else:
            raise click.ClickException(f"Unknown skill or workflow: {identifier!r}")
    return list(selected.values())


def _installed_skill_ids(
    targets: list[tuple[str, Path]],
) -> list[str]:
    """Return sorted unique installed skill ids across the given targets."""
    ids: set[str] = set()
    for _, target in targets:
        for _, sidecar in installed(target):
            ids.add(sidecar["id"])
    return sorted(ids)


def _installed_skill_sources(
    targets: list[tuple[str, Path]],
) -> dict[str, str]:
    """Map each installed skill id to a display string of its source repo(s)."""
    sources: dict[str, set[str]] = {}
    for _, target in targets:
        for _, sidecar in installed(target):
            sources.setdefault(sidecar["id"], set()).add(skill_repository(sidecar))
    return {
        skill_id: _format_sources(sorted(repos)) for skill_id, repos in sources.items()
    }


def _interactive_manage_picker(
    skill_ids: list[str],
    *,
    title: str,
    sources: dict[str, str] | None = None,
) -> list[str] | None:
    """Run the installed-skills picker and return the chosen ids."""
    if not skill_ids:
        console.print("No skills installed.")
        return None

    app = InstalledSkillsPicker(skill_ids, title=title, sources=sources)
    app.run()
    return app.result


def _resolve_agent_names(agent: tuple[str, ...]) -> list[str]:
    if agent:
        return list(agent)
    detected = detect_agent()
    if detected:
        return [resolve_skill_agent(detected).name]
    return [DEFAULT_AGENT]


def _interactive_install_options(
    *,
    agent: tuple[str, ...],
    default_global: bool,
    default_repo: str,
) -> tuple[list[dict], list[str], bool, str, str, Path, dict[str, Any]] | None:
    """Run the tabbed Textual wizard to choose source, skills, agents and scope.

    Parameters
    ----------
    agent : tuple of str
        Agents passed on the command line; when non-empty the agent step is
        skipped.
    default_global : bool
        The pre-selected scope (``True`` for the user-level directory).
    default_repo : str
        GitHub ``owner/name`` pre-filled in the source step.

    Returns
    -------
    tuple or None
        ``(selected_skills, agent_names, global_, repo, tag, root, catalog)``
        or ``None`` when the user aborts or selects nothing. ``root`` is an
        extracted release the caller must delete.
    """
    app = ProbablSkillsInstaller(
        agent=agent, default_global=default_global, default_repo=default_repo
    )
    app.run()

    if app.result is None:
        cleanup_release(app.root)
        return None

    selected_ids, agent_names, global_, repo = app.result
    if (
        not selected_ids
        or not agent_names
        or app.catalog is None
        or app.root is None
        or app.tag is None
    ):
        cleanup_release(app.root)
        return None

    skills_by_id, workflows_by_id = _index(app.catalog)
    selected = _expand(selected_ids, skills_by_id, workflows_by_id)
    return selected, agent_names, global_, repo, app.tag, app.root, app.catalog


def _copy_selected_skills(
    selected: list[dict],
    agent_names: list[str],
    *,
    global_: bool,
    repo: str,
    tag: str,
    root: Path,
    catalog: dict[str, Any],
) -> int:
    """Copy ``selected`` skills into the resolved targets and persist catalogs."""
    try:
        targets = resolve_targets(agent_names, global_=global_)
    except ValueError as error:
        raise click.ClickException(str(error)) from error

    tree = Tree(f"Installing {len(selected)} skill(s) from {repo} release {tag}")
    for _, target in targets:
        branch = tree.add(f"[skore.path]{target}[/]")
        for skill in selected:
            branch.add(
                f"[skore.skill]{skill['id']}[/]  "
                f"[skore.muted]{skill.get('summary', '')}[/]"
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
                install_skill(skill, root, target, tag, repo)
                persist_source_catalog(target, repo, tag, catalog)
                progress.advance(task)

    return len(targets)


@click.group(invoke_without_command=True)
@click.pass_context
def skills(ctx) -> None:
    """Install and manage Agent Skills from the ``probabl-ai/skore-skills`` release."""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@skills.command("install")
@click.argument("ids", nargs=-1)
@_agent_option
@_global_option
@click.option(
    "--repo",
    "-r",
    default=GITHUB_REPO,
    show_default=True,
    help="GitHub ``owner/name`` of the skills catalog to install from.",
)
@click.option(
    "--all",
    "all_",
    is_flag=True,
    help="Install every skill from the latest release (non-interactive).",
)
@_report_store_errors
def install(ids, agent, global_, repo, all_) -> None:
    """Install skill(s) or workflow(s) from the latest release.

    Run without arguments to launch the interactive installer (a tabbed wizard
    for the GitHub source, skill selection, target agent and install scope).

    Pass skill or workflow ids (or ``--all``) to install non-interactively,
    optionally with ``--agent`` and ``--global`` to choose the targets and
    scope. ``--agent``/``--global`` require an explicit selection.

    ``all`` can be passed as a positional argument instead of ``--all``:
    ``skore skills install all``.

    Use ``--repo owner/name`` to install from a catalog other than the default.
    In the interactive wizard that value is pre-filled and can still be edited.
    """
    ids = list(ids)
    if "all" in ids:
        all_ = True
        ids = [i for i in ids if i != "all"]

    interactive = not (ids or all_ or agent or global_)
    if interactive:
        if is_non_interactive():
            with _release(repo) as (_tag, _root, catalog):
                render_catalog(catalog)
                console.print(
                    "Run [skore.cmd]skore skills install <ids>[/] to install "
                    "specific skills, or [skore.cmd]skore skills install all[/] "
                    "to install everything."
                )
            return
        options = _interactive_install_options(
            agent=agent, default_global=global_, default_repo=repo
        )
        if not options:
            console.print("Nothing selected.")
            return
        selected, agent_names, global_, repo, tag, root, catalog = options
        try:
            n_targets = _copy_selected_skills(
                selected,
                agent_names,
                global_=global_,
                repo=repo,
                tag=tag,
                root=root,
                catalog=catalog,
            )
        finally:
            cleanup_release(root)
        console.print(
            f"[skore.ok]+[/] installed [skore.skill]{len(selected)}[/] skill(s) "
            f"into {n_targets} location(s) from {repo} release {tag}"
        )
        return

    if not (ids or all_):
        raise click.UsageError(
            "Specify skill/workflow ids or --all to install non-interactively."
        )

    with _release(repo) as (tag, root, catalog):
        skills_by_id, workflows_by_id = _index(catalog)
        selected = (
            list(skills_by_id.values())
            if all_
            else _expand(ids, skills_by_id, workflows_by_id)
        )
        agent_names = _resolve_agent_names(agent)
        n_targets = _copy_selected_skills(
            selected,
            agent_names,
            global_=global_,
            repo=repo,
            tag=tag,
            root=root,
            catalog=catalog,
        )

    console.print(
        f"[skore.ok]+[/] installed [skore.skill]{len(selected)}[/] skill(s) "
        f"into {n_targets} location(s) from {repo} release {tag}"
    )


@skills.command("list")
@_manage_agent_option
@_global_option
@_report_store_errors
def list_skills(agent, global_) -> None:
    """List installed skills.

    Scans every known agent by default; pass ``--agent`` to restrict the scan.
    """
    targets = _manage_targets(agent, global_=global_)

    table = Table(title="Installed skills")
    table.add_column("id", style="skore.skill")
    table.add_column("release")
    table.add_column("source")
    table.add_column("location")

    found = False
    for _, target in targets:
        for _, sidecar in installed(target):
            found = True
            table.add_row(
                sidecar["id"],
                sidecar.get("release", ""),
                skill_repository(sidecar),
                str(target),
            )

    if found:
        console.print(table)
    else:
        console.print("No skills installed.")


@skills.command("update")
@click.argument("ids", nargs=-1)
@_manage_agent_option
@_global_option
@click.option("--all", "all_", is_flag=True, help="Update every installed skill.")
@_report_store_errors
def update(ids, agent, global_, all_) -> None:
    """Update installed skills to the latest release of their recorded source.

    Pass skill ids to update, or ``--all`` to update every installed skill.
    Each skill is fetched from the GitHub repository stored at install time.
    Scans every known agent by default; pass ``--agent`` to restrict the scan.
    Run ``skore skills list`` to see installed ids.
    """
    targets = _manage_targets(agent, global_=global_)

    if not all_ and not ids:
        if not is_non_interactive():
            selected = _interactive_manage_picker(
                _installed_skill_ids(targets),
                title="Select skills to update.",
                sources=_installed_skill_sources(targets),
            )
            if not selected:
                console.print("Nothing selected.")
                return
            ids = selected
        else:
            raise click.UsageError(
                "Specify skill ids to update or pass --all. "
                "Run `skore skills list` to discover ids."
            )

    requested = set(ids)
    selected_items: list[tuple[Path, dict[str, Any]]] = []
    for _, target in targets:
        for _, sidecar in installed(target):
            if not all_ and sidecar["id"] not in requested:
                continue
            selected_items.append((target, sidecar))

    updated: list[tuple[str, str, Path]] = []
    missing: list[str] = []
    considered_repos = sorted(
        {skill_repository(sidecar) for _, sidecar in selected_items}
    )
    for repo in considered_repos:
        with _release(repo) as (tag, root, catalog):
            skills_by_id, _ = _index(catalog)
            for target, sidecar in selected_items:
                if skill_repository(sidecar) != repo:
                    continue
                skill = skills_by_id.get(sidecar["id"])
                if skill is None:
                    missing.append(sidecar["id"])
                    continue
                if sidecar.get("hash") != skill["hash"]:
                    install_skill(skill, root, target, tag, repo)
                    updated.append((sidecar["id"], repo, target))
                else:
                    write_sidecar(target / sidecar["id"], skill, tag, repo)
                persist_source_catalog(target, repo, tag, catalog)

    if missing:
        skipped = ", ".join(sorted(set(missing)))
        console.print(
            f"[skore.warn]![/] skipped [skore.skill]{skipped}[/]: "
            "not in the latest release."
        )
    if updated:
        for skill_id, repo, target in updated:
            console.print(
                f"[skore.ok]^[/] updated [skore.skill]{skill_id}[/] "
                f"({repo}) -> [skore.path]{target}[/]"
            )
    elif not missing and considered_repos:
        console.print(
            f"All skills are up to date ({_format_sources(considered_repos)})."
        )
    elif not missing:
        console.print("All skills are up to date.")


@skills.command("remove")
@click.argument("ids", nargs=-1)
@_manage_agent_option
@_global_option
@click.option("--all", "all_", is_flag=True, help="Remove every installed skill.")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompts.")
@_report_store_errors
def remove(ids, agent, global_, all_, yes) -> None:
    """Remove installed skills.

    Pass skill ids to remove, or ``--all`` to remove every installed skill.
    Scans every known agent by default; pass ``--agent`` to restrict the scan.
    Run ``skore skills list`` to see installed ids.
    """
    targets = _manage_targets(agent, global_=global_)

    if not all_ and not ids:
        if not is_non_interactive():
            selected = _interactive_manage_picker(
                _installed_skill_ids(targets),
                title="Select skills to remove.",
                sources=_installed_skill_sources(targets),
            )
            if not selected:
                console.print("Nothing selected.")
                return
            ids = selected
        else:
            raise click.UsageError(
                "Specify skill ids to remove or pass --all. "
                "Run `skore skills list` to discover ids."
            )

    to_remove = []
    for _, target in targets:
        if all_:
            for skill_dir, _ in installed(target):
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

    for target in {skill_dir.parent for skill_dir in to_remove}:
        prune_local_catalog(target)
