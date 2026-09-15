"""Module to manage ``skore hub`` authentication."""

from typing import Callable
from contextlib import suppress


@cache
def credentials(self, *, host: str, workspace: str) -> Callable[[], dict[str, str]]:
    with suppress(APIKeyError):
        return API_key()

    with suppress(TokenError):
        return Token(login=False)

    with suppress(APIKeyRegistryError):
        ...

    raise RuntimeError()
