"""Inputs shared by every harness configuration writer."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_AGENT = "agents"
DEFAULT_MODEL_ID = "skore-agent"
# Name under which Skore registers in each harness config (model provider or
# MCP server).
SKORE_PROVIDER_KEY = "skore"


@dataclass(frozen=True)
class HarnessContext:
    """Inputs shared by every harness configuration writer."""

    workspace: Path
    hub_url: str
    api_key: str
    model_id: str = DEFAULT_MODEL_ID

    @property
    def base_url(self) -> str:
        """Return the OpenAI-compatible API base URL."""
        return f"{self.hub_url.rstrip('/')}/v1"

    @property
    def mcp_url(self) -> str:
        return f"{self.hub_url.rstrip('/')}/mcp"


Configure = Callable[[HarnessContext], dict[str, Any]]
Launch = Callable[[Path, str], None]
