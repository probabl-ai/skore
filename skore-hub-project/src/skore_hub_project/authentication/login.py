"""Login to ``skore hub``."""

from __future__ import annotations

from contextlib import suppress
from webbrowser import open as open_webbrowser

from httpx import HTTPError
from rich.align import Align
from rich.panel import Panel

from .. import console
from ..client import api
from . import token as Token


def login(*, timeout=600):
    """Login to ``skore hub``."""
    with suppress(HTTPError):
        if Token.exists() and (Token.access() is not None):
            console.print(
                Panel(
                    Align(
                        "Already logged in!",
                        align="center",
                    ),
                    title="[cyan]Skore Hub[/cyan]",
                    border_style="orange1",
                    expand=False,
                    padding=1,
                    title_align="center",
                )
            )
            return

    url, device_code, user_code = api.get_oauth_device_login()

    console.rule("[cyan]Skore Hub[/cyan]")
    console.print(
        f"Opening browser; if this fails, please visit this URL to log in:\n{url}",
        soft_wrap=True,
    )

    open_webbrowser(url)

    api.get_oauth_device_code_probe(device_code, timeout=timeout)
    api.post_oauth_device_callback(device_code, user_code)
    Token.save(*api.get_oauth_device_token(device_code))
