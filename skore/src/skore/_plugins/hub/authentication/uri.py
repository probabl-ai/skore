"""URI used for ``skore hub`` authentication."""

from os import environ
from typing import Final

DEFAULT: Final[str] = "https://api.skore.probabl.ai"
ENV_VAR_NAME: Final[str] = "SKORE_HUB_URI"


def URI() -> str:
    """URI used for ``skore hub`` authentication."""
    return environ.get(ENV_VAR_NAME, DEFAULT)
