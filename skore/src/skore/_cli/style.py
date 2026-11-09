"""Shared CLI styling: theme-agnostic ANSI colors and rich-click config."""

from __future__ import annotations

from typing import cast

import rich.box
import rich_click as click
from rich.console import Console
from rich.theme import Theme

from skore._cli.runtime import is_non_interactive

# Static slant-style wordmark for the top-level `skore` welcome output.
SKORE_BANNER = r"""
   _____ __ __ ____  ____  ______
  / ___// //_// __ \/ __ \/ ____/
  \__ \/ ,<  / / / / /_/ / __/
 ___/ / /| |/ /_/ / _, _/ /___
/____/_/ |_|\____/_/ |_/_____/
"""

_THEME = Theme(
    {
        "skore.skill": "cyan",
        "skore.path": "blue",
        "skore.ok": "green",
        "skore.muted": "dim",
        "skore.warn": "bold yellow",
        "skore.cmd": "bold cyan",
    }
)
_console: Console | None = None


class _ConsoleProxy:
    """Forward to a console built on first use, against the current stdout.

    Building it at import locks the color system of whatever stdout the
    process had then. Tests and Windows workers often replace stdout later.
    Rich uses the console as a context manager, and those methods are not
    looked up through ``__getattr__``.
    """

    def __getattr__(self, name: str):
        return getattr(_get_console(), name)

    def __enter__(self):
        return _get_console().__enter__()

    def __exit__(self, *args):
        return _get_console().__exit__(*args)


def _get_console() -> Console:
    global _console
    if _console is None:
        _console = Console(theme=_THEME)
    return _console


console: Console = cast(Console, _ConsoleProxy())


def action_required(message: str) -> None:
    """Print a prominent, multi-line call to action."""
    console.print()
    console.print(f"[skore.warn]! Action required:[/] {message}")
    console.print()


click.rich_click.STYLE_OPTION = "cyan"
click.rich_click.STYLE_ARGUMENT = "cyan"
click.rich_click.STYLE_COMMAND = "bold cyan"
click.rich_click.STYLE_SWITCH = "green"
click.rich_click.STYLE_METAVAR = "yellow"
click.rich_click.STYLE_OPTIONS_PANEL_BORDER = "dim"
click.rich_click.STYLE_COMMANDS_PANEL_BORDER = "dim"
click.rich_click.STYLE_USAGE = "bold"
click.rich_click.STYLE_HELPTEXT = ""
click.rich_click.TEXT_MARKUP = "rich"
click.rich_click.STYLE_ERRORS_SUGGESTION = "dim"
click.rich_click.HEADER_TEXT = (
    "[bold cyan]skore[/]  [dim]· ML reporting & agent skills[/]"
)
click.rich_click.STYLE_HEADER_TEXT = ""


def apply_interaction_style() -> None:
    """Choose help boxes from the current process mode.

    Called again when a command starts, so a ``CI`` or TTY change after import
    still applies.
    """
    box = rich.box.SIMPLE_HEAD if is_non_interactive() else click.rich_click.FROM_THEME
    click.rich_click.STYLE_OPTIONS_PANEL_BOX = box
    click.rich_click.STYLE_COMMANDS_PANEL_BOX = box


apply_interaction_style()
