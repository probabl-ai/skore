"""The ``skore hub`` command group to manage Skore Hub credentials."""

from __future__ import annotations

import rich_click as click
from rich.table import Table

from skore._cli.credentials import (
    HubError,
    iter_api_keys,
    mint_api_key,
    revoke_api_key,
)
from skore._cli.skore import apply_hub_uri
from skore._cli.style import console

click.rich_click.COMMAND_GROUPS = {
    **getattr(click.rich_click, "COMMAND_GROUPS", {}),
    "cli hub": [
        {"name": "Credentials", "commands": ["api-key"]},
    ],
    "cli hub api-key": [
        {"name": "Manage", "commands": ["generate", "revoke", "list"]},
    ],
}


@click.group(invoke_without_command=True)
@click.pass_context
def hub(ctx) -> None:
    """Manage Skore Hub credentials."""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@hub.group("api-key", invoke_without_command=True)
@click.pass_context
def api_key(ctx) -> None:
    """Generate, revoke and list Hub API keys."""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@api_key.command("generate")
@click.option("--host", default=None, help="Hub host URL.")
@click.option("--workspace", required=True, help="Hub workspace to mint the key for.")
@click.option(
    "--name",
    default=None,
    help="Name stored on the hub for this key. A random name is used when omitted.",
)
@click.option(
    "--login-timeout",
    default=600,
    show_default=True,
    help="Seconds to wait for interactive device login.",
)
@click.option(
    "--expires",
    type=click.Choice(["1", "3", "6", "never"]),
    default="never",
    show_default=True,
    help="Lifetime in months (1, 3, or 6 months), or never.",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Replace a stored key: revoke it on the hub, then mint a new one.",
)
def generate(
    host: str | None,
    workspace: str,
    name: str | None,
    login_timeout: int,
    expires: str,
    force: bool,
) -> None:
    """Mint a workspace-scoped Hub API key and store it locally."""
    if host:
        host = apply_hub_uri(host, discover=True)
    try:
        mint_api_key(
            host=host,
            workspace=workspace,
            name=name,
            expires=expires,
            timeout=login_timeout,
            force=force,
        )
    except HubError as error:
        raise click.ClickException(str(error)) from error
    message = (
        f"[skore.ok]+[/] generated API key for workspace [skore.skill]{workspace}[/]"
    )
    if expires != "never":
        unit = "month" if expires == "1" else "months"
        message = f"{message} (expires in {expires} {unit})"
    console.print(message, highlight=False)


@api_key.command("revoke")
@click.option("--host", default=None, help="Hub host URL.")
@click.option("--workspace", required=True, help="Hub workspace whose key to revoke.")
@click.option(
    "--login-timeout",
    default=600,
    show_default=True,
    help="Seconds to wait for interactive device login.",
)
def revoke(host: str | None, workspace: str, login_timeout: int) -> None:
    """Revoke a stored Hub API key locally and on the hub."""
    if host:
        host = apply_hub_uri(host, discover=True)
    if not revoke_api_key(host=host, workspace=workspace, timeout=login_timeout):
        console.print("No API key stored.")


@api_key.command("list")
def list_keys() -> None:
    """List stored Hub API keys."""
    rows = list(iter_api_keys())
    if not rows:
        console.print("No API keys stored.")
        return

    table = Table(title="Hub API keys")
    table.add_column("host")
    table.add_column("workspace")
    for _key_id, stored_host, stored_workspace in rows:
        table.add_row(stored_host, stored_workspace)
    console.print(table)
