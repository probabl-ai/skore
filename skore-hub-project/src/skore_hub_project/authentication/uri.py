"""URI used for ``skore hub`` authentication."""

from os import environ
from typing import Final

DEFAULT: Final[str] = "https://api.skore.probabl.ai"
ENVARNAME: Final[str] = "SKORE_HUB_URI"
URI: Final[str] = environ.get(ENVARNAME, DEFAULT)
