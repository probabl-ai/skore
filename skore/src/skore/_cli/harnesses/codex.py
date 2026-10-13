"""Codex CLI configuration and launch."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Any

from skore._cli.harnesses import process as _process
from skore._cli.harnesses.context import HarnessContext
from skore._cli.project import ensure_gitignore_entry
from skore._cli.style import console

CODEX_PROVIDER_KEY = "skore"

CODEX_PROVIDER_NAME = "Skore Agent"

CODEX_PROJECT_CONFIG = ".codex/skore-provider.toml"

CODEX_API_KEY_ENV = "SKORE_AGENT_API_KEY"


def _toml_string(value: str) -> str:
    """Quote ``value`` as a TOML basic string.

    JSON string escapes are a valid subset of TOML basic-string escapes, so
    values containing quotes or backslashes stay parseable.
    """
    return json.dumps(value)


def _codex_config_overrides(base_url: str) -> list[str]:
    """Build the runtime ``--config`` overrides that declare the provider.

    Codex refuses ``model_providers`` in project-local config files, so the
    provider is passed per run instead. The API key is referenced through
    ``env_http_headers`` (header name -> environment variable): it travels in
    the process environment and never appears on the command line.
    """
    provider = f"model_providers.{CODEX_PROVIDER_KEY}"
    return [
        f"model_provider={_toml_string(CODEX_PROVIDER_KEY)}",
        f"{provider}.name={_toml_string(CODEX_PROVIDER_NAME)}",
        f"{provider}.base_url={_toml_string(base_url)}",
        f'{provider}.wire_api="responses"',
        f'{provider}.env_http_headers={{ "X-API-Key" = "{CODEX_API_KEY_ENV}" }}',
    ]


def _configure_codex(ctx: HarnessContext) -> dict[str, Any]:
    """Write the project-local Codex provider file.

    Everything lives in the workspace so each worktree keeps its own hub
    credentials; nothing is written under ``~/.codex`` and plain ``codex``
    runs keep the user's default model.
    """
    config_path = ctx.workspace / CODEX_PROJECT_CONFIG
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        f"model = {_toml_string(ctx.model_id)}\n"
        f"model_provider = {_toml_string(CODEX_PROVIDER_KEY)}\n"
        f"base_url = {_toml_string(ctx.base_url)}\n"
        f"api_key = {_toml_string(ctx.api_key)}\n"
    )
    ensure_gitignore_entry(ctx.workspace, CODEX_PROJECT_CONFIG)
    console.print(f"[skore.ok]+[/] wrote [skore.path]{config_path}[/]")
    console.print(
        "[skore.muted]Note:[/] configuration is project-local and "
        "[skore.path]~/.codex[/] [skore.muted]is untouched. Launch through[/] "
        "[skore.skill]skore agent[/] [skore.muted]to use the Skore hub model.[/]"
    )
    return {"config_path": str(config_path)}


def _launch_codex(workspace: Path, model_id: str) -> None:
    project_config = workspace / CODEX_PROJECT_CONFIG
    if not project_config.is_file():
        raise RuntimeError(
            f"missing {CODEX_PROJECT_CONFIG}; run skore agent --harness codex first."
        )
    try:
        data = tomllib.loads(project_config.read_text())
    except tomllib.TOMLDecodeError as error:
        raise RuntimeError(
            f"could not parse {CODEX_PROJECT_CONFIG}; "
            "run skore agent --harness codex first."
        ) from error
    base_url = data.get("base_url")
    api_key = data.get("api_key")
    if not isinstance(base_url, str) or not base_url:
        raise RuntimeError(f"{CODEX_PROJECT_CONFIG} is missing a valid base_url.")
    if not isinstance(api_key, str) or not api_key:
        raise RuntimeError(f"{CODEX_PROJECT_CONFIG} is missing a valid api_key.")
    model = data.get("model")
    if not isinstance(model, str) or not model:
        model = model_id
    env = _process.os.environ.copy()
    env[CODEX_API_KEY_ENV] = api_key
    argv = ["codex", "--model", model]
    for override in _codex_config_overrides(base_url):
        argv.extend(["--config", override])
    _process._exec_harness("codex", argv, env=env)
