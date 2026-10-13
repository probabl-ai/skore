"""The ``skore sync`` command."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import rich_click as click

from skore._cli.skore import URI_ENV, resolve_hub_uri

MODES = ("local", "hub", "mlflow")
Mode = Literal["local", "hub", "mlflow"]


def _endpoint_options(
    mode: str,
    *,
    workspace: str | None,
    tracking_uri: str | None,
    prefix: str,
) -> dict[str, Any]:
    """Validate one endpoint's mode-specific options."""
    workspace_option = f"--{prefix}-workspace"

    if mode == "local":
        return (
            {}
            if workspace is None
            else {"workspace": Path(workspace).expanduser().resolve()}
        )

    if mode == "hub":
        if workspace is None:
            raise click.UsageError(f"{workspace_option} is required for Hub.")
        return {"workspace": workspace}

    if workspace is not None:
        raise click.UsageError(f"{workspace_option} is not valid for MLflow.")
    return {} if tracking_uri is None else {"tracking_uri": tracking_uri}


def _render_result(result, *, dry_run: bool) -> None:
    """Render synchronization results."""
    if result.empty:
        click.echo("No reports to synchronize.")
    else:
        click.echo(result.to_string())
    if dry_run:
        click.echo("Dry run complete. No reports were transferred.")


@click.command()
@click.argument("source_project")
@click.option(
    "--from",
    "from_mode",
    type=click.Choice(MODES),
    default=None,
    help="Storage mode of the source project. Defaults to ``local``.",
)
@click.option(
    "--from-workspace",
    help=(
        "Source workspace. Required for ``hub``; optional local directory for "
        "``local`` storage; invalid for ``mlflow``."
    ),
)
@click.option(
    "--to",
    "to_mode",
    type=click.Choice(MODES),
    default=None,
    help="Storage mode of the destination project. Defaults to ``local``.",
)
@click.option(
    "--to-project",
    default=None,
    help="Destination project name. Defaults to ``SOURCE_PROJECT``.",
)
@click.option(
    "--to-workspace",
    help=(
        "Destination workspace. Required for ``hub``; optional local directory "
        "for ``local`` storage; invalid for ``mlflow``."
    ),
)
@click.option(
    "--tracking-uri",
    help=(
        "MLflow tracking URI. Applied to every MLflow endpoint; required only "
        "when its default tracking URI is unsuitable."
    ),
)
@click.option(
    "--hub-url",
    default=None,
    help=(
        "Hub URL (e.g. ``https://skore.probabl.ai``). "
        f"Defaults to the ``{URI_ENV}`` environment variable or the public Hub."
    ),
)
@click.option(
    "--both", is_flag=True, help="Synchronize missing reports in both directions."
)
@click.option("--dry-run", is_flag=True, help="Show the synchronization plan only.")
def sync(
    source_project: str,
    from_mode: Mode | None,
    from_workspace: str | None,
    to_mode: Mode | None,
    to_project: str | None,
    to_workspace: str | None,
    tracking_uri: str | None,
    hub_url: str | None,
    both: bool,
    dry_run: bool,
) -> None:
    """Synchronize report projects across local, Hub, and MLflow storage.

    ``SOURCE_PROJECT`` is the source project name. Pass ``--from`` or ``--to``
    (or both) to select the endpoint modes; an omitted endpoint mode defaults
    to ``local`` storage.
    """
    if from_mode is None and to_mode is None:
        raise click.UsageError("Pass --from or --to.")

    source_mode = from_mode or "local"
    destination_mode = to_mode or "local"
    destination_project = to_project or source_project

    if tracking_uri is not None and "mlflow" not in (source_mode, destination_mode):
        raise click.UsageError("--tracking-uri requires an MLflow endpoint.")

    source_options = _endpoint_options(
        source_mode,
        workspace=from_workspace,
        tracking_uri=tracking_uri,
        prefix="from",
    )
    destination_options = _endpoint_options(
        destination_mode,
        workspace=to_workspace,
        tracking_uri=tracking_uri,
        prefix="to",
    )
    if (
        source_mode == destination_mode
        and source_project == destination_project
        and source_options == destination_options
    ):
        raise click.UsageError("Source and destination identify the same project.")

    uses_hub = "hub" in (source_mode, destination_mode)
    if hub_url is not None and not uses_hub:
        raise click.UsageError("--hub-url requires a Hub endpoint.")

    from skore import Project
    from skore._exceptions import (
        InvalidProjectNameError,
        ProjectCreationError,
        ProjectLoadError,
        ProjectPermissionError,
    )

    try:
        if uses_hub and hub_url:
            resolve_hub_uri(hub_url)
        source = Project(source_project, mode=source_mode, **source_options)
        destination = Project(
            destination_project,
            mode=destination_mode,
            **destination_options,
        )
        result = source.sync(destination, bidirectional=both, dry_run=dry_run)
    except click.ClickException:
        raise
    except (
        InvalidProjectNameError,
        ProjectCreationError,
        ProjectLoadError,
        ProjectPermissionError,
        OSError,
        ValueError,
        RuntimeError,
    ) as error:
        raise click.ClickException(str(error)) from error

    _render_result(result, dry_run=dry_run)
