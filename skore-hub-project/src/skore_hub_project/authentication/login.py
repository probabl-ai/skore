"""Login to ``skore hub``."""

from collections.abc import Callable

credentials: Callable[[], dict[str, str]] | None = None


def login(*, timeout: int = 600) -> None:
    """Login to ``skore hub``."""
    from skore_hub_project import console
    from skore_hub_project.authentication.apikey import APIKey
    from skore_hub_project.authentication.token import Token

    global credentials

    try:
        credentials = APIKey()
    except KeyError:
        console.rule("[cyan]Skore Hub[/cyan]")
        console.print(f"", soft_wrap=True)

        credentials = Token(timeout=timeout)
