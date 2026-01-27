"""Login to ``skore hub``."""

from collections.abc import Callable

#
# Global variable storing credentials used for authentication by the ``HUBClient``, both
# with an API key or a temporary token.
#
# By default, it is empty and must be initialized by the user by calling explicitly the
# function ``login``.
#
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
        console.print("BAD", soft_wrap=True)

        credentials = Token(timeout=timeout)
    else:
        console.rule("[cyan]Skore Hub[/cyan]")
        console.print("GOOD", soft_wrap=True)
