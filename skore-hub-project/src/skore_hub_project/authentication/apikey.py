"""API key used for ``skore hub`` authentication."""

from collections.abc import Callable
from os import environ


def APIKey() -> Callable[[], dict[str, str]]:
    """API key used for ``skore hub`` authentication, as HTTP header parameters."""
    return lambda: {"X-API-Key": environ["SKORE_HUB_API_KEY"]}  # when is evaluate ?!
