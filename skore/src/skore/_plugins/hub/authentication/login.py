"""Login to ``skore hub``."""

from os import environ

from rich.align import Align
from rich.live import Live
from rich.panel import Panel

from skore import console
from skore._plugins.hub.authentication.api_key import ENV_VAR_NAME
from skore._plugins.hub.authentication.token import Token, token


def login(*, timeout: int = 600) -> None:
    """Login to ``skore hub``."""
    global token

    if (
        (environ.get("SKORE_HUB_JUPYTERLITE", "").lower() in ("1", "true", "yes"))
        or (token is not None)
        or (ENV_VAR_NAME in environ)
    ):
        return

    with Live(console=console, auto_refresh=False) as live:
        token = Token(timeout=timeout, live=live)

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
