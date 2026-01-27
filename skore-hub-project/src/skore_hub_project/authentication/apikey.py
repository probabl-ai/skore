from collections.abc import Callable
from os import environ


def APIKey() -> Callable[[], dict[str, str]]:
    return lambda: {"X-API-Key": environ["SKORE_HUB_API_KEY"]}  # when is evaluate ?!
