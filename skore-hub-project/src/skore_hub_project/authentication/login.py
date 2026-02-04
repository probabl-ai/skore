"""Login to ``skore hub``."""

from collections.abc import Callable
from logging import getLogger

from rich.align import Align
from rich.live import Live
from rich.panel import Panel

from skore_hub_project import console
from skore_hub_project.authentication.apikey import APIKey
from skore_hub_project.authentication.token import Token
from skore_hub_project.authentication.uri import URI

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
    global credentials

    if credentials is not None:
        logger.debug(f"Already logged in {URI} with {type(credentials)}.")
        console.print(
            Panel(
                Align.center("Already logged in."),
                title="[cyan]Login to [b]Skore Hub",
                border_style="cyan",
                padding=1,
            )
        )

        return

    try:
        credentials = APIKey()
    except KeyError:
        with Live(console=console, auto_refresh=False) as live:
            credentials = Token(timeout=timeout, live=live)

            live.update(
                Panel(
                    Align.center(
                        "Successfully logged in, using [b]interactive authentication."
                    ),
                    title="[cyan]Login to [bold]Skore Hub",
                    border_style="cyan",
                    padding=1,
                )
            )
            live.refresh()
    else:
        console.print(
            Panel(
                Align.center("Successfully logged in, using [b]API key."),
                title="[cyan]Login to [bold]Skore Hub",
                border_style="cyan",
                padding=1,
            )
        )
