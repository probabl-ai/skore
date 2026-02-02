"""URI used for ``skore hub`` authentication."""

from os import environ
from typing import Final

DEFAULT: Final[str] = "https://api.skore.probabl.ai"
ENV_VAR_NAME: Final[str] = "SKORE_HUB_URI"
URI: Final[str] = environ.get(ENV_VAR_NAME, DEFAULT)
