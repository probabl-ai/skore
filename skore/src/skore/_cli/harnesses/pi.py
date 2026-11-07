"""Pi coding agent configuration and launch."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from skore._cli.harnesses import process as _process
from skore._cli.harnesses.context import SKORE_PROVIDER_KEY, HarnessContext
from skore._cli.harnesses.cursor import _load_json_object
from skore._cli.project import ensure_gitignore_entry
from skore._cli.style import console


def _configure_pi(ctx: HarnessContext) -> dict[str, Any]:
    """Write ``.pi/agent/models.json`` for Pi.

    Other providers already in the file are left in place.
    """
    config_dir = ctx.workspace / ".pi" / "agent"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "models.json"
    payload = _load_json_object(config_path)
    providers = payload.get("providers", {})
    if not isinstance(providers, dict):
        raise RuntimeError(
            f"{config_path} providers is not a JSON object; fix or move it, "
            "then run `skore agent` again."
        )
    providers[SKORE_PROVIDER_KEY] = {
        "baseUrl": ctx.base_url,
        "api": "openai-completions",
        "apiKey": ctx.api_key,
        "authHeader": True,
        "compat": {
            "supportsDeveloperRole": False,
            "supportsReasoningEffort": False,
            "sendSessionAffinityHeaders": True,
            "sessionAffinityFormat": "openrouter",
        },
        "models": [
            {
                "id": ctx.model_id,
                "name": "Skore Agent",
                "reasoning": True,
                "input": ["text"],
                "contextWindow": 200000,
                "maxTokens": 8192,
            }
        ],
    }
    payload["providers"] = providers
    config_path.write_text(json.dumps(payload, indent=2) + "\n")
    ensure_gitignore_entry(ctx.workspace, ".pi/agent/models.json")
    console.print(f"[skore.ok]+[/] wrote [skore.path]{config_path}[/]")
    return {"config_path": str(config_path)}


def _launch_pi(workspace: Path, model_id: str) -> None:
    env = _process.os.environ.copy()
    env["PI_CODING_AGENT_DIR"] = str(workspace / ".pi" / "agent")
    _process._exec_harness(
        "pi",
        ["pi", "--provider", SKORE_PROVIDER_KEY, "--model", model_id],
        env=env,
    )
