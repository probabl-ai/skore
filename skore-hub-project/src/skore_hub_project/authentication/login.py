"""Login to ``skore hub``."""

from __future__ import annotations

from skore_hub_project import console
from skore_hub_project.authentication.token import APIKey, Token

CREDENTIALS: APIKey | Token | None = None


def login(*, timeout: int = 600) -> None:
    """Login to ``skore hub``."""

    global CREDENTIALS

    try:
        CREDENTIALS = APIKey()
    except KeyError:
        console.rule("[cyan]Skore Hub[/cyan]")
        console.print(
            f"",
            soft_wrap=True,
        )

        CREDENTIALS = Token(timeout=timeout)
