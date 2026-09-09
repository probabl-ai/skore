"""Login to ``skore hub``."""

from collections.abc import Callable
from io import StringIO
from logging import getLogger
from os import environ

from rich.align import Align
from rich.console import Console
from rich.live import Live
from rich.panel import Panel

from skore import console
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


def _login_panel(message: str) -> Panel:
    return Panel(
        Align.center(message),
        title="[cyan]Login to [bold]Skore Hub",
        border_style="cyan",
        padding=1,
    )


def _panel_line_count(panel: Panel) -> int:
    buffer = StringIO()
    Console(
        file=buffer,
        width=console.width,
        color_system=None,
        force_terminal=True,
        highlight=False,
    ).print(panel)
    return buffer.getvalue().count("\n")


def login(*, timeout: int = 600) -> None:
    """Login to ``skore hub``.

    This function is a no op if SKORE_HUB_JUPYTERLITE
    """
    is_running_in_hub_jupyterlite = environ.get(
        "SKORE_HUB_JUPYTERLITE", ""
    ).lower() in ("1", "true", "yes")
    if is_running_in_hub_jupyterlite:
        return

    global credentials

    if credentials is not None:
        logger.debug(f"Already logged in {URI()} with {credentials.__module__}.")
        console.print(_login_panel("Already logged in."))

        return

    try:
        credentials = APIKey()
    except KeyError:
        success = "Successfully logged in, using [b]interactive authentication."
        with Live(
            console=console,
            auto_refresh=False,
            redirect_stdout=False,
            redirect_stderr=False,
        ) as live:
            credentials = Token(timeout=timeout, live=live)
            extra = max(
                0,
                live._live_render.last_render_height
                - _panel_line_count(_login_panel(success)),
            )
            live.update(_login_panel(success + extra * "\n"))
            live.refresh()
    else:
        console.print(_login_panel("Successfully logged in, using [b]API key."))
