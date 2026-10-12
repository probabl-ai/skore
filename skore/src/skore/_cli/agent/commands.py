"""The ``skore agent`` command: authenticate, configure and launch a harness."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import rich_click as click

from skore._cli.agents import (
    AGENTS,
    DEFAULT_MODEL_ID,
    HARNESS_CHOICES,
    HARNESS_NAMES,
    detect_agent,
    get_harness,
    installed_harnesses,
    is_harness_installed,
    launch_harness,
    missing_harness_message,
    normalize_harness_name,
)
from skore._cli.credentials import HubError, require_workspace_api_key
from skore._cli.harnesses.claude import (
    claude_plugin_ide_ready,
    resolve_claude_plugin_harness,
)
from skore._cli.harnesses.context import HarnessContext
from skore._cli.project import SkoreConfig, SkoreConfigError, ensure_gitignore_entry
from skore._cli.runtime import is_non_interactive
from skore._cli.skore import URI_ENV, apply_hub_uri, resolve_hub_uri
from skore._cli.style import action_required, console

if TYPE_CHECKING:
    from skore._plugins.hub.authentication.registry.distant import Workspace


def _pick_workspace(memberships: list[Workspace]) -> Workspace:
    """Launch the Textual workspace picker and return the chosen membership."""
    from skore._cli.agent.app import WorkspacePicker

    rows = [(membership.public_id, membership.public_id) for membership in memberships]
    app = WorkspacePicker(rows)
    app.run()
    if app.result is None:
        raise click.Abort()
    return next(m for m in memberships if m.public_id == app.result)


def _pick_harness(workspace: Path) -> str:
    """Launch the Textual harness picker.

    Detected harnesses are listed first. Harnesses that are not installed
    on this machine are listed after them.
    """
    from skore._cli.agent.app import HarnessPicker

    installed = {
        harness.harness_name
        for harness in installed_harnesses()
        if harness.harness_name is not None
    }
    harnesses = [agent for agent in AGENTS.values() if agent.harness_name is not None]
    ordered = [agent for agent in harnesses if agent.harness_name in installed]
    ordered += [agent for agent in harnesses if agent.harness_name not in installed]
    rows = [
        (
            agent.harness_name,
            agent.harness_display_name,
            agent.harness_name in installed,
        )
        for agent in ordered
        if agent.harness_name is not None
    ]
    app = HarnessPicker(rows, preselect=0)
    app.run()
    if app.result is None:
        raise click.Abort()
    return app.result


def _api_key_for(config: SkoreConfig, *, login_timeout: int) -> str:
    """Return the workspace API key, minting one when none is stored."""
    try:
        api_key, minted = require_workspace_api_key(
            host=config.hub_url,
            workspace=config.workspace,
            login_timeout=login_timeout,
        )
    except HubError as error:
        raise click.ClickException(str(error)) from error
    if minted:
        console.print(
            "[skore.ok]+[/] generated API key for workspace "
            f"[skore.skill]{config.workspace}[/]"
        )
    return api_key


def _resolve_membership(memberships: list[Workspace]) -> Workspace:
    if len(memberships) == 1:
        return memberships[0]
    if is_non_interactive():
        raise click.UsageError(
            "pass a saved workspace in .skore or run interactively to pick one."
        )
    return _pick_workspace(memberships)


@click.command()
@click.option(
    "--workspace",
    "-w",
    default=".",
    type=click.Path(file_okay=False, path_type=Path),
    help="Project directory to configure (default: current directory).",
)
@click.option(
    "--hub-url",
    default=None,
    help=(
        "Hub URL (API or frontend, e.g. https://skore.probabl.ai). "
        "Frontend URLs are auto-resolved via /.well-known/skore-hub.json. "
        f"Defaults to the {URI_ENV} env var or the public hub."
    ),
)
@click.option(
    "--harness",
    "-H",
    "harness_name",
    type=click.Choice(HARNESS_CHOICES),
    default=None,
    help="Harness to use non-interactively (omit to pick one).",
)
@click.option(
    "--model-id",
    default=DEFAULT_MODEL_ID,
    show_default=True,
    help="Model id advertised by the hub.",
)
@click.option(
    "--login-timeout",
    default=600,
    show_default=True,
    help="Seconds to wait for interactive device login.",
)
def agent(
    workspace: Path,
    hub_url: str | None,
    harness_name: str | None,
    model_id: str,
    login_timeout: int,
) -> None:
    """Authenticate, configure and launch a Skore Hub agent harness.

    On the first run, ``skore agent`` logs in to the hub (when needed), lets
    you pick a workspace and harness, and saves the workspace and hub URI to
    ``.skore``. The API key itself comes from ``skore hub api-key generate``,
    which is run automatically when no key is stored for that workspace.
    Later runs reuse ``.skore``; passing ``--hub-url`` or ``--harness`` updates
    the recorded values. A different ``--hub-url`` re-authenticates against
    that hub and uses the key stored for it.

    Supported harnesses: Bob Shell, Bob IDE, Claude CLI, Claude UI, Claude
    Plugin (Cursor, VS Code, or VS Code Insiders), Cursor IDE, Cursor CLI,
    OpenCode, Pi, Copilot in VSCode, Copilot CLI and Codex CLI.
    Detection uses ``PATH``, the application bundle on macOS for Bob IDE and
    Claude UI, or the Claude Code IDE extension for Claude Plugin. Claude CLI
    also accepts ``--harness claude-cli``. ``--harness claude-plugin`` asks
    which IDE to open; ``claude-plugin-cursor``, ``claude-plugin-code``, and
    ``claude-plugin-code-insiders`` select that IDE. Bob IDE also accepts
    ``--harness bobide``.
    """
    harness_name = normalize_harness_name(harness_name)
    workspace = workspace.resolve()
    if not workspace.is_dir():
        raise click.ClickException(f"workspace does not exist: {workspace}")

    try:
        config = SkoreConfig.load(workspace)
    except SkoreConfigError as error:
        raise click.ClickException(str(error)) from error
    saved_harness = config.harness if config is not None else None

    # The saved hub_url is stored already resolved, so a raw match modulo a
    # trailing slash means the same hub and reuses the saved workspace.
    if config is not None and (
        hub_url is None or hub_url.rstrip("/") == config.hub_url.rstrip("/")
    ):
        apply_hub_uri(config.hub_url)
        harness_name = harness_name or config.harness
    else:
        from skore._plugins.hub.authentication.login import login
        from skore._plugins.hub.authentication.registry import distant

        resolved_hub_url = resolve_hub_uri(hub_url)
        token = login(host=resolved_hub_url, timeout=login_timeout)
        memberships = distant.identity(host=resolved_hub_url, token=token).workspaces
        if not memberships:
            raise click.ClickException(
                "you are not a member of any hub workspace; create or join one first."
            )

        membership = _resolve_membership(memberships)
        config = SkoreConfig(
            hub_url=resolved_hub_url,
            workspace=membership.public_id,
            workspace_id=membership.id,
            harness=harness_name,
        )
        config_path = config.save(workspace)
        ensure_gitignore_entry(workspace)
        console.print(f"[skore.ok]+[/] saved [skore.path]{config_path}[/]")

    api_key = _api_key_for(config, login_timeout=login_timeout)

    if harness_name is None:
        if is_non_interactive():
            detected = detect_agent()
            if detected and detected.harness_name and is_harness_installed(detected):
                harness_name = detected.harness_name
            elif saved_harness:
                # The harness is hub-independent: fall back to the saved one
                # before erroring out on a hub switch.
                harness_name = saved_harness
            else:
                raise click.UsageError(
                    f"pass --harness <name> (one of: {', '.join(HARNESS_NAMES)})."
                )
        else:
            harness_name = _pick_harness(workspace)

    try:
        if harness_name == "claude-plugin" or claude_plugin_ide_ready(harness_name):
            harness_name = resolve_claude_plugin_harness(harness_name)
        harness = get_harness(harness_name)
    except click.Abort:
        raise
    except KeyError as error:
        raise click.ClickException(f"unknown harness: {harness_name}") from error
    except RuntimeError as error:
        raise click.ClickException(str(error)) from error

    if config.harness != harness_name:
        config = SkoreConfig(
            hub_url=config.hub_url,
            workspace=config.workspace,
            workspace_id=config.workspace_id,
            harness=harness_name,
        )
        config.save(workspace)

    console.print(
        f"Configuring [skore.skill]{harness.harness_display_name}[/] in "
        f"[skore.path]{workspace}[/]"
    )
    assert harness.configure is not None
    try:
        harness.configure(
            HarnessContext(
                workspace=workspace,
                hub_url=config.hub_url,
                api_key=api_key,
                model_id=model_id,
            )
        )
    except RuntimeError as error:
        raise click.ClickException(str(error)) from error
    detected = detect_agent()
    if detected and detected.harness_name == harness_name:
        action_required(
            f"{harness.harness_display_name} is configured with the Skore Hub "
            f"provider. Restart {harness.harness_display_name} or start a new "
            f"session to use it."
        )
        return
    if not is_harness_installed(harness):
        raise click.ClickException(missing_harness_message(harness))
    try:
        launch_harness(harness, workspace, model_id=model_id)
    except RuntimeError as error:
        raise click.ClickException(str(error)) from error
