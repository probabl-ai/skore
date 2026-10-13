"""Cursor IDE and Cursor CLI configuration and launch."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from skore._cli.harnesses import process as _process
from skore._cli.harnesses.context import SKORE_PROVIDER_KEY, HarnessContext
from skore._cli.project import ensure_gitignore_entry
from skore._cli.style import console

CURSOR_ALLOWLIST_ENTRY = "skore:*"

CURSOR_AUTORUN_INSTRUCTIONS = (
    "curl fetching a Skore materialize URL under /v1/materialize/ to write a "
    "template or script into the workspace",
    "calls to the skore MCP server's skore_agent tool",
)


def _load_json_object(path: Path) -> dict[str, Any]:
    """Return the JSON object stored at ``path``, or an empty one when absent.

    Refuses anything else: the caller writes the result back, so treating an
    unreadable file as empty would drop the servers and rules it holds. Cursor
    accepts comments in these files and Python's parser does not.
    """
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text() or "{}")
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"{path} is not valid JSON (comments are not supported here); "
            "fix or move it, then run `skore agent` again."
        ) from error
    if not isinstance(data, dict):
        raise RuntimeError(
            f"{path} does not hold a JSON object; fix or move it, then run "
            "`skore agent` again."
        )
    return data


def _configure_cursor(ctx: HarnessContext) -> dict[str, Any]:
    """Point Cursor's MCP client at the hub and pre-approve the Skore tools.

    These two files belong to Cursor, not to Skore, so they are read-modify-
    written: another MCP server or permission rule the user set up survives.
    """
    from skore._cli.style import action_required

    config_dir = ctx.workspace / ".cursor"
    config_dir.mkdir(parents=True, exist_ok=True)

    config_path = config_dir / "mcp.json"
    permissions_path = config_dir / "permissions.json"
    # Both files are read up front so an unreadable second one cannot leave the
    # workspace half-configured.
    config = _load_json_object(config_path)
    permissions = _load_json_object(permissions_path)

    servers = config.get("mcpServers")
    if not isinstance(servers, dict):
        servers = {}
    servers[SKORE_PROVIDER_KEY] = {
        # Without an explicit transport Cursor probes streamable HTTP and, on
        # any failure, retries the same URL as legacy SSE, which the hub does
        # not serve: the real error is then buried under an SSE 404.
        "type": "http",
        "url": ctx.mcp_url,
        # Written out rather than interpolated from the environment: Cursor
        # resolves ${env:...} against its own process, which is whatever
        # launched the app, not `skore agent`.
        "headers": {"Authorization": f"Bearer {ctx.api_key}"},
    }
    config["mcpServers"] = servers
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    ensure_gitignore_entry(ctx.workspace, ".cursor/mcp.json")

    allowlist = permissions.get("mcpAllowlist")
    if not isinstance(allowlist, list):
        allowlist = []
    if CURSOR_ALLOWLIST_ENTRY not in allowlist:
        allowlist.append(CURSOR_ALLOWLIST_ENTRY)
    permissions["mcpAllowlist"] = allowlist
    auto_run = permissions.get("autoRun")
    if not isinstance(auto_run, dict):
        auto_run = {}
    instructions = auto_run.get("allow_instructions")
    if not isinstance(instructions, list):
        instructions = []
    for instruction in CURSOR_AUTORUN_INSTRUCTIONS:
        if instruction not in instructions:
            instructions.append(instruction)
    auto_run["allow_instructions"] = instructions
    permissions["autoRun"] = auto_run
    permissions_path.write_text(json.dumps(permissions, indent=2) + "\n")

    console.print(f"[skore.ok]+[/] wrote [skore.path]{config_path}[/]")
    console.print(f"[skore.ok]+[/] wrote [skore.path]{permissions_path}[/]")
    action_required(
        f"enable the [skore.warn]{SKORE_PROVIDER_KEY}[/] MCP server in Cursor.\n"
        f"  [skore.muted]Cursor Settings -> Customize -> MCP, then turn[/] "
        f"[skore.warn]{SKORE_PROVIDER_KEY}[/] [skore.muted]on.\n"
        f"  Config: [skore.path]{config_path}[/]"
    )
    return {"config_path": str(config_path)}


def cursor_cli_installed() -> bool:
    """Return whether a Cursor CLI binary is on ``PATH``."""
    return _resolve_cursor_cli_binary() is not None


def _resolve_cursor_cli_binary() -> str | None:
    """Return ``cursor-agent``, or ``agent`` only when the path looks like Cursor."""
    if _process.shutil.which("cursor-agent") is not None:
        return "cursor-agent"
    agent = _process.shutil.which("agent")
    if agent is not None and "cursor" in agent.lower():
        return "agent"
    return None


def _launch_cursor_cli(workspace: Path, _model_id: str) -> None:
    """Start the Cursor CLI (``cursor-agent``, or ``agent`` when that is Cursor)."""
    name = _resolve_cursor_cli_binary()
    if name is None:
        raise RuntimeError("Cursor CLI is not installed or not on PATH.")
    previous = _process.os.getcwd()
    _process.os.chdir(workspace)
    try:
        _process._exec_harness(name, [name])
    except BaseException:
        _process.os.chdir(previous)
        raise


def _launch_cursor(workspace: Path, _model_id: str) -> None:
    _process._exec_harness("cursor", ["cursor", str(workspace)])
