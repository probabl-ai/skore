"""Claude CLI, desktop app, and IDE plugin configuration and launch."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from skore._cli.harnesses import process as _process
from skore._cli.harnesses.context import HarnessContext
from skore._cli.harnesses.cursor import _load_json_object
from skore._cli.project import SkoreConfig, ensure_gitignore_entry
from skore._cli.runtime import is_non_interactive
from skore._cli.style import console

CLAUDE_UI_APP_PATH = Path("/Applications/Claude.app")

CLAUDE_PLUGIN_EXTENSION = "anthropic.claude-code"

IDE_PLUGIN_LABELS = {
    "cursor": "Cursor",
    "code": "VS Code",
    "code-insiders": "VS Code Insiders",
}


def _configure_claude(ctx: HarnessContext) -> dict[str, Any]:
    """Write ``.claude/settings.local.json`` for Claude.

    The file belongs to Claude, so an existing ``permissions``, ``hooks``, or
    ``model`` entry is kept. Only the Skore ``env`` keys are updated.
    """
    config_dir = ctx.workspace / ".claude"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "settings.local.json"
    payload = _load_json_object(config_path)
    env = payload.get("env", {})
    if not isinstance(env, dict):
        raise RuntimeError(
            f"{config_path} env is not a JSON object; fix or move it, "
            "then run `skore agent` again."
        )
    env.update(
        {
            "ANTHROPIC_BASE_URL": ctx.hub_url.rstrip("/"),
            "ANTHROPIC_AUTH_TOKEN": ctx.api_key,
            "ANTHROPIC_MODEL": ctx.model_id,
        }
    )
    payload["env"] = env
    config_path.write_text(json.dumps(payload, indent=2) + "\n")
    ensure_gitignore_entry(ctx.workspace, ".claude/settings.local.json")
    console.print(f"[skore.ok]+[/] wrote [skore.path]{config_path}[/]")
    return {"config_path": str(config_path)}


def _read_claude_settings(path: Path) -> dict[str, Any]:
    """Return Claude's settings object, or raise ``RuntimeError`` when unreadable."""
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


def _launch_claude(workspace: Path, _model_id: str) -> None:
    env = _process.os.environ.copy()
    settings_path = workspace / ".claude" / "settings.local.json"
    if settings_path.is_file():
        settings = _read_claude_settings(settings_path)
        exported = settings.get("env", {})
        if not isinstance(exported, dict):
            raise RuntimeError(
                f"{settings_path} env is not a JSON object; fix or move it, "
                "then run `skore agent` again."
            )
        env.update(exported)
    _process._exec_harness("claude", ["claude"], env=env)


def _launch_claude_ui(workspace: Path, _model_id: str) -> None:
    """Open the Claude desktop app on the workspace."""
    if _process.sys.platform != "darwin":
        raise RuntimeError("Claude UI is only supported on macOS.")
    if not CLAUDE_UI_APP_PATH.is_dir():
        raise RuntimeError("Claude UI is not installed.")
    _process._exec_harness(
        "open", ["open", "-a", str(CLAUDE_UI_APP_PATH), str(workspace)]
    )


def ide_extension_hosts() -> tuple[tuple[str, Path, str], ...]:
    """Return ``(binary, extensions_dir, uri_scheme)`` for VS Code-compatible IDEs."""
    home = _process.Path.home()
    return (
        ("cursor", home / ".cursor" / "extensions", "cursor"),
        ("code", home / ".vscode" / "extensions", "vscode"),
        ("code-insiders", home / ".vscode-insiders" / "extensions", "vscode-insiders"),
    )


def _has_ide_extension(extensions_dir: Path, extension_id: str) -> bool:
    """Return whether ``extensions_dir`` contains ``extension_id``."""
    try:
        entries = list(extensions_dir.iterdir())
    except OSError:
        return False
    prefix = f"{extension_id}-"
    return any(
        path.is_dir() and (path.name == extension_id or path.name.startswith(prefix))
        for path in entries
    )


def _claude_plugin_missing_message() -> str:
    return "Claude Plugin is not installed in VS Code or Cursor."


def claude_ui_installed() -> bool:
    """Return whether the Claude desktop app is present."""
    return _process.sys.platform == "darwin" and CLAUDE_UI_APP_PATH.is_dir()


def claude_ui_unavailable() -> str:
    """Return why Claude UI cannot be launched."""
    if _process.sys.platform != "darwin":
        return "Claude UI is only supported on macOS."
    return "Claude UI is not installed or not on PATH."


def claude_plugin_installed() -> bool:
    """Return whether a supported IDE has the Claude plugin."""
    return bool(_installed_claude_plugin_hosts())


def claude_plugin_unavailable() -> str:
    """Return why the Claude plugin harness cannot be launched."""
    return _claude_plugin_missing_message()


def _installed_claude_plugin_hosts() -> list[tuple[str, str]]:
    """Return ``(binary, uri_scheme)`` for IDEs that have the Claude plugin."""
    return [
        (binary, scheme)
        for binary, folder, scheme in ide_extension_hosts()
        if _has_ide_extension(folder, CLAUDE_PLUGIN_EXTENSION)
    ]


def _prompt_ide(options: list[tuple[str, str]]) -> str:
    """Ask which IDE to open. ``options`` are ``(binary, label)`` pairs."""
    import rich_click as click

    from skore._cli.agent.app import IdePicker

    app = IdePicker(options)
    app.run()
    if app.result is None:
        raise click.Abort()
    return app.result


def _format_ide_list(labels: list[str]) -> str:
    """Return ``labels`` as a phrase, with a comma before the last item."""
    if len(labels) == 2:
        return f"{labels[0]} and {labels[1]}"
    return f"{', '.join(labels[:-1])}, and {labels[-1]}"


def claude_plugin_ide(name: str | None) -> str | None:
    """Return the IDE binary encoded in ``claude-plugin-{ide}``."""
    prefix = "claude-plugin-"
    if not name or not name.startswith(prefix):
        return None
    binary = name.removeprefix(prefix)
    if binary in IDE_PLUGIN_LABELS:
        return binary
    return None


def claude_plugin_ide_ready(name: str | None) -> bool:
    """Return whether the IDE named in ``claude-plugin-{ide}`` has the plugin."""
    ide = claude_plugin_ide(name)
    if ide is None:
        return False
    return any(binary == ide for binary, _scheme in _installed_claude_plugin_hosts())


def _choose_claude_plugin_host(preferred: str | None = None) -> tuple[str, str]:
    """Return the IDE to open, asking when more than one has the plugin."""
    hosts = _installed_claude_plugin_hosts()
    if not hosts:
        raise RuntimeError(_claude_plugin_missing_message())
    if preferred is not None:
        for host in hosts:
            if host[0] == preferred:
                return host
        raise RuntimeError(_claude_plugin_missing_message())
    if len(hosts) == 1:
        return hosts[0]
    options = [
        (binary, IDE_PLUGIN_LABELS.get(binary, binary)) for binary, _scheme in hosts
    ]
    if is_non_interactive():
        raise RuntimeError(
            "Claude Plugin is installed in "
            f"{_format_ide_list([label for _, label in options])}. "
            "Run interactively to choose an IDE."
        )
    chosen = _prompt_ide(options)
    for host in hosts:
        if host[0] == chosen:
            return host
    raise RuntimeError(f"Unknown IDE: {chosen}")


def resolve_claude_plugin_harness(name: str) -> str:
    """Return ``claude-plugin-{ide}`` for a Claude plugin harness name."""
    binary, _scheme = _choose_claude_plugin_host(claude_plugin_ide(name))
    return f"claude-plugin-{binary}"


def _open_uri(uri: str) -> None:
    """Open ``uri`` with the platform handler."""
    if _process.sys.platform == "darwin":
        _process._exec_harness("open", ["open", uri])
        return
    if _process.sys.platform == "win32":
        _process._exec_harness("cmd", ["cmd", "/c", "start", "", uri])
        return
    _process._exec_harness("xdg-open", ["xdg-open", uri])


def _open_workspace_in_ide(binary: str, workspace: Path) -> None:
    """Ask a VS Code-compatible IDE to open ``workspace``."""
    executable = _process.shutil.which(binary)
    if executable is None:
        return
    _process.subprocess.run([executable, str(workspace)], check=False)


def _launch_claude_plugin(workspace: Path, _model_id: str) -> None:
    """Open the workspace, then the Claude Code panel, in a host IDE."""
    config = SkoreConfig.load(workspace)
    preferred = claude_plugin_ide(config.harness) if config is not None else None
    binary, scheme = _choose_claude_plugin_host(preferred)
    from skore._cli.style import console

    console.print(
        f"[skore.muted]  opening in[/] "
        f"[skore.skill]{IDE_PLUGIN_LABELS.get(binary, binary)}[/]"
    )
    _open_workspace_in_ide(binary, workspace)
    _open_uri(f"{scheme}://{CLAUDE_PLUGIN_EXTENSION}/open")
