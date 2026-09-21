"""Module to manage ``skore hub`` authentication via API key."""

from typing import Final

from skore._plugins.hub.authentication import registry
from skore._plugins.hub.authentication.uri import URI

__all__ = ["ENV_VAR_NAME", "URI", "registry"]

ENV_VAR_NAME: Final[str] = "SKORE_HUB_API_KEY"
