"""Shared CLI styling: theme-agnostic ANSI colors and rich-click config."""

from __future__ import annotations

import rich_click as click
from rich.console import Console
from rich.theme import Theme

console = Console(
    theme=Theme(
        {
            "skore.skill": "cyan",
            "skore.path": "blue",
            "skore.ok": "green",
            "skore.muted": "dim",
        }
    )
)

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
