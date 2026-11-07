"""GitHub Copilot in VS Code and Copilot CLI configuration and launch."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from skore._cli.harnesses import process as _process
from skore._cli.harnesses.context import HarnessContext
from skore._cli.project import SkoreConfig, ensure_gitignore_entry
from skore._cli.style import action_required, console

COPILOT_PROVIDER_NAME = "Skore Agent"

COPILOT_PROJECT_CONFIG = ".vscode/chatLanguageModels.json"

COPILOT_BINARIES = ("code", "code-insiders")

COPILOT_CLI_BINARIES = ("copilot",)


def _copilot_provider(ctx: HarnessContext) -> dict[str, Any]:
    """Build the Custom Endpoint provider entry for VS Code Copilot Chat."""
    return {
        "name": COPILOT_PROVIDER_NAME,
        "vendor": "customendpoint",
        "apiKey": "skore",
        "apiType": "chat-completions",
        "models": [
            {
                "id": ctx.model_id,
                "name": COPILOT_PROVIDER_NAME,
                "url": f"{ctx.base_url}/chat/completions",
                "toolCalling": True,
                "vision": False,
                "maxInputTokens": 200000,
                "maxOutputTokens": 8192,
                "requestHeaders": {"X-API-Key": ctx.api_key},
            }
        ],
    }


def _configure_copilot(ctx: HarnessContext) -> dict[str, Any]:
    """Write ``.vscode/chatLanguageModels.json`` for Copilot in VSCode."""
    config_path = ctx.workspace / COPILOT_PROJECT_CONFIG
    _upsert_copilot_provider(config_path, _copilot_provider(ctx))
    ensure_gitignore_entry(ctx.workspace, COPILOT_PROJECT_CONFIG)
    console.print(f"[skore.ok]+[/] wrote [skore.path]{config_path}[/]")
    return {"config_path": str(config_path)}


def _resolve_copilot_binary() -> str | None:
    """Return the preferred VS Code binary available on PATH."""
    for name in COPILOT_BINARIES:
        if _process.shutil.which(name) is not None:
            return name
    return None


def _vscode_app_dirname(binary: str) -> str:
    return "Code - Insiders" if binary == "code-insiders" else "Code"


def _copilot_user_config_path(binary: str, *, home: Path | None = None) -> Path:
    """Return the user-profile language-model configuration path."""
    home = home or _process.Path.home()
    app = _vscode_app_dirname(binary)
    if _process.sys.platform == "darwin":
        return (
            home
            / "Library"
            / "Application Support"
            / app
            / "User"
            / "chatLanguageModels.json"
        )
    if _process.sys.platform == "win32":
        appdata = _process.os.environ.get("APPDATA")
        base = Path(appdata) if appdata else home / "AppData" / "Roaming"
        return base / app / "User" / "chatLanguageModels.json"
    return home / ".config" / app / "User" / "chatLanguageModels.json"


def _upsert_copilot_provider(config_path: Path, provider: dict[str, Any]) -> None:
    """Upsert the Skore provider into a language-model file."""
    providers: list[Any] = []
    if config_path.is_file():
        try:
            parsed = json.loads(config_path.read_text() or "[]")
        except json.JSONDecodeError as error:
            raise RuntimeError(
                f"could not parse {config_path}; "
                "fix the file or add the Skore Agent provider manually."
            ) from error
        if not isinstance(parsed, list):
            raise RuntimeError(
                f"could not parse {config_path}; "
                "fix the file or add the Skore Agent provider manually."
            )
        providers = parsed
    updated = [
        entry
        for entry in providers
        if not (isinstance(entry, dict) and entry.get("name") == COPILOT_PROVIDER_NAME)
    ]
    updated.append(provider)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(updated, indent=2) + "\n")


def _launch_copilot(workspace: Path, _model_id: str) -> None:

    binary = _resolve_copilot_binary()
    if binary is None:
        raise RuntimeError("Copilot in VSCode is not installed or not on PATH.")

    # VS Code only reads providers from the user profile, so the project config
    # written by ``_configure_copilot`` has to be mirrored there.
    project_config = workspace / COPILOT_PROJECT_CONFIG
    if not project_config.is_file():
        raise RuntimeError(
            f"missing {COPILOT_PROJECT_CONFIG}; "
            "run skore agent --harness copilot first."
        )
    try:
        providers = json.loads(project_config.read_text() or "[]")
    except json.JSONDecodeError as error:
        raise RuntimeError(f"could not parse {COPILOT_PROJECT_CONFIG}.") from error
    provider = next(
        (
            entry
            for entry in providers
            if isinstance(entry, dict) and entry.get("name") == COPILOT_PROVIDER_NAME
        ),
        None,
    )
    if provider is None:
        raise RuntimeError(
            f"{COPILOT_PROJECT_CONFIG} has no {COPILOT_PROVIDER_NAME} provider; "
            "run skore agent --harness copilot first."
        )
    user_config = _copilot_user_config_path(binary)
    _upsert_copilot_provider(user_config, provider)
    console.print(f"[skore.ok]+[/] synced [skore.path]{user_config}[/]")
    action_required(
        "select [skore.warn]Skore Agent[/] in Copilot Chat "
        "(reload VS Code if it is missing)."
    )
    _process._exec_harness(binary, [binary, str(workspace)])


def _configure_copilot_cli(_ctx: HarnessContext) -> dict[str, Any]:
    """Copilot CLI takes the Hub provider from the process environment at launch."""
    console.print(
        "[skore.muted]Copilot CLI uses the Hub key from the credential "
        "registry at launch; no extra project file is written.[/]"
    )
    return {}


def _launch_copilot_cli(workspace: Path, model_id: str) -> None:
    """Start ``copilot`` with the same provider env as ``skore-copilot cli``."""
    from skore._plugins.hub.authentication import key

    config = SkoreConfig.load(workspace)
    if config is None or not config.hub_url:
        raise RuntimeError(
            "missing .skore; run skore agent --harness copilot-cli first."
        )
    api_key = key.get(host=config.hub_url, workspace=config.workspace)
    if not api_key:
        raise RuntimeError(
            "missing Hub API key; run skore agent --harness copilot-cli first."
        )
    hub_url = config.hub_url
    env = _process.os.environ.copy()
    env["COPILOT_PROVIDER_TYPE"] = "openai"
    env["COPILOT_PROVIDER_BASE_URL"] = f"{hub_url.rstrip('/')}/v1"
    env["COPILOT_PROVIDER_WIRE_API"] = "completions"
    env["COPILOT_MODEL"] = model_id
    env["COPILOT_PROVIDER_HEADERS"] = f"X-API-Key: {api_key}"
    env["COPILOT_PROVIDER_MAX_OUTPUT_TOKENS"] = "8192"
    env["COPILOT_PROVIDER_MAX_PROMPT_TOKENS"] = "200000"
    previous = _process.os.getcwd()
    _process.os.chdir(workspace)
    try:
        _process._exec_harness("copilot", ["copilot"], env=env)
    except BaseException:
        _process.os.chdir(previous)
        raise
