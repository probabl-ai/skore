"""Plain and table renderers for a skills catalog."""

from __future__ import annotations

from typing import Any

from rich.table import Table

from skore._cli.runtime import is_non_interactive
from skore._cli.style import console


def _matches(entry: dict, query: str) -> bool:
    fields = (
        entry.get("id"),
        entry.get("title"),
        entry.get("summary"),
        entry.get("category"),
    )
    return any(field and query in field.lower() for field in fields)


def render_catalog(
    catalog: dict[str, Any],
    *,
    query: str | None = None,
    ids: list[str] | None = None,
) -> None:
    """Print workflows and skills from ``catalog``, optionally filtered."""
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

    if is_non_interactive():
        _render_catalog_plain(workflows, skills)
        return

    if workflows:
        table = Table(title="Workflows (recommended)")
        table.add_column("id", style="skore.skill")
        table.add_column("summary")
        for workflow in workflows:
            table.add_row(workflow["id"], workflow.get("summary", ""))
        console.print(table)

    if skills:
        table = Table(title="Skills")
        table.add_column("id", style="skore.skill")
        table.add_column("summary")
        for skill in skills:
            table.add_row(skill["id"], skill.get("summary", ""))
        console.print(table)

    if not workflows and not skills:
        console.print("No skill or workflow matches the query.")


def _render_catalog_plain(
    workflows: list[dict[str, Any]], skills: list[dict[str, Any]]
) -> None:
    """Render the catalog as plain text (no boxes) for non-interactive use."""
    if workflows:
        console.print("Workflows:")
        for wf in workflows:
            summary = wf.get("summary", "")
            console.print(f"  {wf['id']:<26} {summary}")

    if skills:
        if workflows:
            console.print("")
        console.print("Skills:")
        for skill in skills:
            summary = skill.get("summary", "")
            console.print(f"  {skill['id']:<26} {summary}")

    if not workflows and not skills:
        console.print("No skill or workflow matches the query.")
