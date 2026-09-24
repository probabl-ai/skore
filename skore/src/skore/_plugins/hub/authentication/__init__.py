"""Module to manage ``skore hub`` authentication via API key."""

from typing import Final

__all__ = [
    "API_KEY_ENV_VAR_NAME",
    "HOST_ENV_VAR_NAME",
]

API_KEY_ENV_VAR_NAME: Final[str] = "SKORE_HUB_API_KEY"
HOST_ENV_VAR_NAME: Final[str] = "SKORE_HUB_URI"
