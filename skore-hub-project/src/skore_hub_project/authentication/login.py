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
    from rich.align import Align
    from rich.console import Group
    from rich.panel import Panel
    from rich.text import Text

    from skore_hub_project import console
    from skore_hub_project.authentication.apikey import APIKey
    from skore_hub_project.authentication.token import Token

    global credentials

    try:
        credentials = APIKey()
    except KeyError:
        console.print(
            Panel(
                Group(
                    Panel(
                        Align.center(
                            "API key not detected, fallback to interactive "
                            "authentication for the session."
                        ),
                        style="white on orange_red1",
                    ),
                    Text(),
                    Text(
                        "We recommend that you set up an API key via [url](coming soon) "
                        "and use it to log in.",
                        style="white",
                    ),
                ),
                title="[cyan]Login to [b]Skore Hub",
                style="dark_orange",
            )
        )

        credentials = Token(timeout=timeout)
    else:
        console.print(
            Panel(
                Panel(Align.center("API key detected."), style="white on blue"),
                title="[cyan]Login to [b]Skore Hub",
                style="dark_orange",
            )
        )
    finally:
        console.print(
            Panel(Align.center("[bold blink cyan]Login successful")),
            style="dark_orange",
        )
