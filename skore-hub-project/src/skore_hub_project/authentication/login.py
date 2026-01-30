"""Login to ``skore hub``."""

from collections.abc import Callable
from logging import getLogger

logger = getLogger(__name__)

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
    from rich.align import Align
    from rich.console import Group
    from rich.panel import Panel
    from rich.text import Text

    from skore_hub_project import console
    from skore_hub_project.authentication.apikey import APIKey
    from skore_hub_project.authentication.token import Token
    from skore_hub_project.authentication.uri import URI

    global credentials

    if credentials is not None:
        logger.debug(f"Already logged in {URI} with {type(credentials)}.")
        console.print(
            Panel(
                Align.center("[blink]Already logged in."),
                title="[cyan]Login to [bold]Skore Hub",
                border_style="cyan",
                padding=1,
            )
        )

        return

    try:
        credentials = APIKey()
    except KeyError:
        console.print(
            Panel(
                Align.center(
                    "Falling back to interactive authentication for the session.\n"
                    "We recommend that you set up an API key via [url](coming soon) "
                    "and use it to log in."
                ),
                title="[cyan]Login to [bold]Skore Hub",
                subtitle="[dark_orange bold]API key not detected",
                border_style="dark_orange",
                padding=1,
            )
        )

        credentials = Token(timeout=timeout)

        console.print(
            Panel(
                Align.center("[blink]Login successful."),
                title="[cyan]Login to [b]Skore Hub",
                border_style="cyan",
                padding=1,
            )
        )
    else:
        console.print(
            Panel(
                Align.center("[blink]Login successful."),
                title="[cyan]Login to [bold]Skore Hub",
                subtitle="[cyan bold]API key detected",
                border_style="cyan",
                padding=1,
            )
        )
