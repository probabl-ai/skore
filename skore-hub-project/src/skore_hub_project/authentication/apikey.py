"""API key used for ``skore hub`` authentication."""

from collections.abc import Callable
from os import environ
from typing import Final

ENVARNAME: Final[str] = "SKORE_HUB_API_KEY"


def APIKey() -> Callable[[], dict[str, str]]:
    """API key used for ``skore hub`` authentication, as HTTP header parameters."""
    return lambda: {"X-API-Key": environ[ENVARNAME]}  # when is evaluate ?!
