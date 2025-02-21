"""Encapsulate all authentication related stuff."""

from __future__ import annotations

import functools
import json
import os
import pathlib
import tempfile
import time
import webbrowser
from datetime import datetime
from urllib.parse import urljoin

import httpx

URI = os.environ.get("SKORE_HUB_URI", "https://skh.k.probabl.dev")


class AuthenticationError(Exception):
    """An exception dedicated to authentication."""


class AuthenticationToken:
    """Wrap access, refres and expires_at."""

    FILEPATH = pathlib.Path(tempfile.gettempdir(), "skore.token")
    EMPTY = object()

    def __init__(self, access=EMPTY, refreshment=EMPTY, expires_at=EMPTY):
        if all(attr != self.EMPTY for attr in (access, refreshment, expires_at)):
            self.access = access
            self.refreshment = refreshment
            self.expires_at = expires_at

            self.save()
        else:
            try:
                self.access, self.refreshment, self.expires_at = json.loads(
                    self.FILEPATH.read_text()
                )
            except FileNotFoundError:
                self.access = None
                self.refreshment = None
                self.expires_at = None

    def save(self):
        """Save the tokens to the disk."""
        self.FILEPATH.write_text(
            json.dumps((self.access, self.refreshment, self.expires_at))
        )

    def refresh(self):
        """Call the API to get a fresh access token."""
        with httpx.Client() as client:
            response = client.post(
                urljoin(URI, "identity/oauth/token/refresh"),
                json={"refresh_token": self.refreshment},
            )
            response.raise_for_status()

            tokens = response.json()["token"]

            self.access = tokens["access_token"]
            self.refreshment = tokens["refresh_token"]
            self.expires_at = tokens["expires_at"]
            self.save()

    def is_valid(self):
        """Check that all data are present."""
        return (
            self.access is not None
            and self.refresh is not None
            and self.expires_at is not None
        )

    def __repr__(self):
        """Repr implementation."""
        return f"AuthenticationToken('{self.access:.10}[...]')"


class AuthenticatedClient(httpx.Client):
    """Override httpx client to pass our bearer token."""

    @functools.cached_property
    def token(self):
        """Access token."""
        return AuthenticationToken()

    def request(self, method: str, url: httpx.URL | str, **kwargs) -> httpx.Response:
        """Execute request with access token, and refresh the token if needed."""
        # Check if token is well initialized
        if not self.token.is_valid():
            raise AuthenticationError("You are not logged in.")

        # Check if token must be refreshed
        if datetime.fromisoformat(self.token.expires_at) <= datetime.now():
            self.token.refresh()

        # Overload headers with authorization token
        headers = kwargs.pop("headers", None) or {}
        headers |= {"Authorization": f"Bearer {self.token.access}"}

        return super().request(
            method,
            urljoin(URI, str(url)),
            headers=headers,
            **kwargs,
        )


def login(timeout=600):
    """Login to the skore-HUB."""
    from rich.align import Align
    from rich.panel import Panel

    from skore import console

    with httpx.Client() as client:
        # Request a new authorization URL
        response = client.get(urljoin(URI, "identity/oauth/device/login"))
        response.raise_for_status()

        response = response.json()
        authorization_url = response["authorization_url"]
        device_code = response["device_code"]
        user_code = response["user_code"]

        # Display authentication info to the user

        console.print(
            "\n"
            "🌍 Opening your default browser to start the authentication process.\n"
            "❔ If your browser did not open visit our "
            f"[link={authorization_url}]authentication page[/link].\n"
        )
        console.print(
            Panel(
                Align(f"[bold]{user_code}[/bold]", align="center"),
                title="[cyan]Your unique code is[/cyan]",
                border_style="orange1",
                expand=False,
                padding=1,
                title_align="center",
            )
        )

        # Open the default browser
        webbrowser.open(authorization_url)

        # Start polling Skore-Hub, waiting for the token
        tic = datetime.now()

        try:
            while True:
                response = client.get(
                    urljoin(
                        URI,
                        f"identity/oauth/device/token?device_code={device_code}",
                    )
                )

                try:
                    response.raise_for_status()
                except httpx.HTTPError:
                    time.sleep(0.5)

                    if (datetime.now() - tic).total_seconds() > timeout:
                        raise AuthenticationError(
                            "Authentication process timed out."
                        ) from None
                else:
                    response = response.json()
                    tokens = response["token"]

                    return AuthenticationToken(
                        access=tokens["access_token"],
                        refreshment=tokens["refresh_token"],
                        expires_at=tokens["expires_at"],
                    )
        except KeyboardInterrupt:
            console.print("👋 login process interrupted.")
