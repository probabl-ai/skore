"""API key used for ``skore hub`` authentication."""

from __future__ import annotations

from typing import TYPE_CHECKING

from skore._plugins.hub.authentication.api_key import registry

if TYPE_CHECKING:
    from typing import Final

__all__ = ["ENV_VAR_NAME", "registry"]

ENV_VAR_NAME: Final[str] = "SKORE_HUB_API_KEY"
