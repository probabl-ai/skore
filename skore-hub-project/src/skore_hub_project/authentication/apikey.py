from os import environ
from typing import Iterator, Literal


class MissingAPIKey(KeyError):
    ...


class APIKey:
    def __init__(self) -> None:
        if "SKORE_HUB_API_KEY" not in environ:
            raise MissingAPIKey

    def __iter__(self) -> Iterator[tuple[Literal["X-API-Key"], str]]:
        yield ("X-API-Key", environ["SKORE_HUB_API_KEY"])
