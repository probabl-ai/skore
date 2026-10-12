"""OpenCode project configuration and launch."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from skore._cli.harnesses import process as _process
from skore._cli.harnesses.context import (
    SKORE_PROVIDER_KEY,
    HarnessContext,
)
from skore._cli.project import ensure_gitignore_entry
from skore._cli.style import console

OPENCODE_SCHEMA = "https://opencode.ai/config.json"

OPENCODE_SESSION_PLUGIN = ".opencode/plugins/skore-session.js"

OPENCODE_MODEL = {
    "name": "Skore Agent",
    "limit": {"context": 200_000, "output": 8192},
}

OPENCODE_SESSION_PLUGIN_SOURCE = """\
export const SkoreSessionPlugin = async () => ({
  "chat.headers": async (input, output) => {
    if (input.sessionID) {
      output.headers["X-Skore-Session-Id"] = input.sessionID;
    }
  },
});
"""


def _configure_opencode(ctx: HarnessContext) -> dict[str, Any]:
    """Write ``opencode.json`` with the Skore Hub provider."""
    config_path = ctx.workspace / "opencode.json"
    config: dict[str, Any] = {
        "$schema": OPENCODE_SCHEMA,
        "model": f"{SKORE_PROVIDER_KEY}/{ctx.model_id}",
        "provider": {
            SKORE_PROVIDER_KEY: {
                "npm": "@ai-sdk/openai-compatible",
                "name": "Skore Hub",
                "options": {
                    "baseURL": ctx.base_url,
                    "apiKey": ctx.api_key,
                },
                "models": {ctx.model_id: dict(OPENCODE_MODEL)},
            }
        },
    }
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    ensure_gitignore_entry(ctx.workspace, "opencode.json")
    console.print(f"[skore.ok]+[/] wrote [skore.path]{config_path}[/]")

    plugin_path = ctx.workspace / OPENCODE_SESSION_PLUGIN
    plugin_path.parent.mkdir(parents=True, exist_ok=True)
    plugin_path.write_text(OPENCODE_SESSION_PLUGIN_SOURCE)
    ensure_gitignore_entry(ctx.workspace, OPENCODE_SESSION_PLUGIN)
    console.print(f"[skore.ok]+[/] wrote [skore.path]{plugin_path}[/]")
    return {"config_path": str(config_path), "plugin_path": str(plugin_path)}


def _launch_opencode(_workspace: Path, model_id: str) -> None:
    _process._exec_harness(
        "opencode",
        ["opencode", "-m", f"{SKORE_PROVIDER_KEY}/{model_id}"],
    )
