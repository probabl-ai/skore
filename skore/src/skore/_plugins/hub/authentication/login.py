"""Login to ``skore hub``."""

from collections.abc import Callable
from logging import getLogger
from os import environ

from rich.align import Align
from rich.live import Live
from rich.panel import Panel

from skore._plugins.hub.authentication.apikey import APIKey
from skore._plugins.hub.authentication.token import Token
from skore._plugins.hub.authentication.uri import URI

logger = getLogger(__name__)

#
# Global variable storing credentials used for authentication by the ``HUBClient``, both
# with an API key or a temporary token.
#
# By default, it is empty and must be initialized by the user by calling explicitly the
# function ``login``.
#
credentials: Callable[[], dict[str, str]] | None = None


def panel_message(msg):
    return Panel(
        Align.center(msg),
        title="[cyan]Login to [b]Skore Hub",
        border_style="cyan",
        padding=1,
    )


def login(*, timeout: int = 600) -> None:
    """Login to ``skore hub``.

    This function is a no op if SKORE_HUB_JUPYTERLITE is truthy.
    """
    from skore import console

    is_running_in_hub_jupyterlite = environ.get(
        "SKORE_HUB_JUPYTERLITE", ""
    ).lower() in ("1", "true", "yes")
    if is_running_in_hub_jupyterlite:
        return

    global credentials

    if credentials is not None:
        logger.debug(f"Already logged in {URI()} with {credentials.__module__}.")
        console.print(panel_message("Already logged in."))

        return

    try:
        credentials = APIKey()
    except KeyError:
        with Live(console=console, auto_refresh=False) as live:
            credentials = Token(timeout=timeout, live=live)

            live.update(
                panel_message(
                    "Successfully logged in, using [b]interactive authentication."
                )
            )
            live.refresh()
    else:
        console.print(panel_message("Successfully logged in, using [b]API key."))
