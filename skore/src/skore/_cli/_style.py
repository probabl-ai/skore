"""Shared CLI styling: brand palette, background detection, and rich-click config."""

from __future__ import annotations

import contextlib
import os
import re
import select
import sys
import termios
import tty
from typing import Final

import rich_click as click
from rich.console import Console
from rich.theme import Theme

_DARK_PALETTE: Final = {"blue": "#79C0FF", "orange": "#F8AB53"}
_LIGHT_PALETTE: Final = {"blue": "#3043F0", "orange": "#F97316"}

_OSC11_QUERY = b"\x1b]11;?\x1b\\"
_OSC11_PATTERN = re.compile(
    r"rgb:([0-9a-fA-F]+)/([0-9a-fA-F]+)/([0-9a-fA-F]+)|#([0-9a-fA-F]{6})"
)


def _relative_luminance(red: float, green: float, blue: float) -> float:
    def channel(value: float) -> float:
        value /= 255.0
        if value <= 0.03928:
            return value / 12.92
        return ((value + 0.055) / 1.055) ** 2.4

    return 0.2126 * channel(red) + 0.7152 * channel(green) + 0.0722 * channel(blue)


def _parse_osc11_channels(response: str) -> tuple[float, float, float] | None:
    match = _OSC11_PATTERN.search(response)
    if not match:
        return None
    if match.group(4):
        hex_color = match.group(4)
        return (
            int(hex_color[0:2], 16),
            int(hex_color[2:4], 16),
            int(hex_color[4:6], 16),
        )

    def _channel(hex_part: str) -> float:
        maximum = 16 ** len(hex_part) - 1
        return int(hex_part, 16) / maximum * 255

    return (
        _channel(match.group(1)),
        _channel(match.group(2)),
        _channel(match.group(3)),
    )


def _is_dark_from_colorfgbg() -> bool | None:
    value = os.environ.get("COLORFGBG")
    if not value:
        return None
    try:
        background = int(value.split(";")[-1])
    except (ValueError, IndexError):
        return None
    # xterm: 0–7 dark backgrounds, 8–15 light (e.g. 15 = white)
    return background < 8


def _query_background_luminance() -> float | None:
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return None
    fd = sys.stdin.fileno()
    try:
        original = termios.tcgetattr(fd)
    except termios.error:
        return None
    try:
        tty.setraw(fd)
        sys.stdout.buffer.write(_OSC11_QUERY)
        sys.stdout.buffer.flush()
        response = ""
        while len(response) < 128:
            ready, _, _ = select.select([sys.stdin], [], [], 0.1)
            if not ready:
                break
            char = sys.stdin.read(1)
            if not char:
                break
            response += char
            if response.endswith("\x07") or response.endswith("\x1b\\"):
                break
    except OSError:
        return None
    finally:
        with contextlib.suppress(termios.error):
            termios.tcsetattr(fd, termios.TCSADRAIN, original)
    channels = _parse_osc11_channels(response)
    if channels is None:
        return None
    red, green, blue = channels
    return _relative_luminance(red, green, blue)


def _detect_is_dark_background() -> bool:
    theme = os.environ.get("SKORE_CLI_THEME", "").strip().lower()
    if theme == "dark":
        return True
    if theme == "light":
        return False

    from_colorfgbg = _is_dark_from_colorfgbg()
    if from_colorfgbg is not None:
        return from_colorfgbg

    luminance = _query_background_luminance()
    if luminance is not None:
        return luminance < 0.5

    return True


def _active_palette() -> dict[str, str]:
    return _DARK_PALETTE if _detect_is_dark_background() else _LIGHT_PALETTE


_PALETTE = _active_palette()

console = Console(
    theme=Theme(
        {
            "skore.brand": _PALETTE["blue"],
            "skore.accent": _PALETTE["orange"],
            "skore.skill": _PALETTE["blue"],
            "skore.path": _PALETTE["blue"],
            "skore.ok": _PALETTE["orange"],
            "skore.muted": "dim",
        }
    )
)

click.rich_click.STYLE_OPTION = _PALETTE["blue"]
click.rich_click.STYLE_ARGUMENT = _PALETTE["blue"]
click.rich_click.STYLE_COMMAND = f"bold {_PALETTE['blue']}"
click.rich_click.STYLE_SWITCH = _PALETTE["orange"]
click.rich_click.STYLE_METAVAR = _PALETTE["orange"]
click.rich_click.STYLE_OPTIONS_PANEL_BORDER = "dim"
click.rich_click.STYLE_COMMANDS_PANEL_BORDER = "dim"
click.rich_click.STYLE_USAGE = "bold"
click.rich_click.STYLE_HELPTEXT = ""
click.rich_click.TEXT_MARKUP = "rich"
click.rich_click.STYLE_ERRORS_SUGGESTION = "dim"
click.rich_click.HEADER_TEXT = (
    f"[bold {_PALETTE['blue']}]skore[/]  [dim]· ML reporting & agent skills[/]"
)
click.rich_click.STYLE_HEADER_TEXT = ""
