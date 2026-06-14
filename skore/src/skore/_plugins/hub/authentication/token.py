"""Token used for ``skore hub`` authentication."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from threading import RLock
from time import sleep
from typing import TYPE_CHECKING
from urllib.parse import urljoin
from webbrowser import open as open_webbrowser

from httpx import HTTPError, HTTPStatusError, TimeoutException
from rich.align import Align
from rich.live import Live
from rich.panel import Panel

from skore import console
from skore._plugins.hub.authentication.uri import URI

if TYPE_CHECKING:
    from skore._plugins.hub.authentication import store


def get_oauth_device_login(success_uri: str | None = None) -> tuple[str, str, str]:
    """
    Initiate device OAuth flow.

    Initiates the OAuth device flow.
    Provides the user with a URL and a OTP code to authenticate the device.

    Parameters
    ----------
    success_uri : str, optional
        The URI to redirect to after successful authentication.
        If not provided, defaults to None.

    Returns
    -------
    tuple
        A tuple containing:
        - authorization_url: str
            The URL to which the user needs to navigate
        - device_code: str
            The device code used for authentication
        - user_code: str
            The user code that needs to be entered on the authorization page
    """
    from skore._plugins.hub.client.client import Client

    url = "identity/oauth/device/login"
    params = {"success_uri": success_uri} if success_uri is not None else {}

    with Client() as client:
        response = client.get(urljoin(URI(), url), params=params).json()

        return (
            response["authorization_url"],
            response["device_code"],
            response["user_code"],
        )


def get_oauth_device_code_probe(device_code: str, *, timeout: int = 600) -> None:
    """
    Ensure authorization code is acknowledged.

    Start polling, wait for the authorization code to be acknowledged by the hub.
    This is mandatory to be authorize to exchange with a token.

    Parameters
    ----------
    device_code : str
        The device code to exchange for tokens.
    """
    from skore._plugins.hub.client.client import Client

    url = "identity/oauth/device/code-probe"
    params = {"device_code": device_code}

    with Client() as client:
        start = datetime.now()

        while True:
            try:
                client.get(urljoin(URI(), url), params=params)
            except HTTPStatusError as exc:
                if exc.response.status_code != 400:
                    raise

                if (datetime.now() - start).total_seconds() >= timeout:
                    raise TimeoutException("Authentication timeout") from exc

                sleep(0.5)
            else:
                break


def post_oauth_device_callback(state: str, user_code: str) -> None:
    """
    Validate the user-provided device code.

    This endpoint verifies the code entered by the user during the device auth flow.

    Parameters
    ----------
    state: str
        The unique value identifying the device flow.
    user_code: str
        The code entered by the user.
    """
    from skore._plugins.hub.client.client import Client

    url = "identity/oauth/device/callback"
    data = {"state": state, "user_code": user_code}

    with Client() as client:
        client.post(urljoin(URI(), url), data=data)


def get_oauth_device_token(device_code: str) -> tuple[str, str, str]:
    """
    Exchanges the device code for an access token.

    This endpoint completes the device authorization flow
    by exchanging the validated device code for an access token.

    Parameters
    ----------
    device_code : str
        The device code to exchange for tokens

    Returns
    -------
    tuple
        A tuple containing:
        - access_token : str
            The OAuth access token
        - refresh_token : str
            The OAuth refresh token
        - expires_at : str
            The expiration datetime as ISO 8601 str of the access token
    """
    from skore._plugins.hub.client.client import Client

    url = "identity/oauth/device/token"
    params = {"device_code": device_code}

    with Client() as client:
        response = client.get(urljoin(URI(), url), params=params).json()
        tokens = response["token"]

        return (
            tokens["access_token"],
            tokens["refresh_token"],
            tokens["expires_at"],
        )


def post_oauth_refresh_token(refresh_token: str) -> tuple[str, str, str]:
    """
    Refresh an access token using a provided refresh token.

    This endpoint allows a client to obtain a new access token
    by providing a valid refresh token.

    Parameters
    ----------
    refresh_token : str
        A valid refresh token

    Returns
    -------
    tuple
        A tuple containing:
        - access_token : str
            The OAuth access token
        - refresh_token : str
            The OAuth refresh token
        - expires_at : str
            The expiration datetime as ISO 8601 str of the access token
    """
    from skore._plugins.hub.client.client import Client

    url = "identity/oauth/token/refresh"
    json = {"refresh_token": refresh_token}

    with Client() as client:
        response = client.post(urljoin(URI(), url), json=json).json()

        return (
            response["access_token"],
            response["refresh_token"],
            response["expires_at"],
        )


def interactive_device_login(
    *, timeout: int = 600, open_browser: bool = True
) -> tuple[str, str, str]:
    """Run the interactive OAuth device flow and return the resulting token.

    Performs the full device flow (login -> browser -> probe -> callback ->
    token) and returns ``(access_token, refresh_token, expires_at)``. Shared
    building block behind ``skore hub login`` and the re-login fallback in
    :func:`fresh_token`.
    """
    url, device_code, user_code = get_oauth_device_login()
    console.print(
        f"Opening your browser to authenticate. If it does not open, visit:\n  {url}"
    )
    if open_browser:
        open_webbrowser(url)

    get_oauth_device_code_probe(device_code, timeout=timeout)
    post_oauth_device_callback(device_code, user_code)
    return get_oauth_device_token(device_code)


def post_oauth_logout(access_token: str, refresh_token: str | None = None) -> None:
    """Revoke the access (and refresh) token on the hub.

    Mirrors ``POST identity/oauth/logout``: the tokens are presented as cookies so
    the hub revokes both the access and refresh tokens server-side.
    """
    from skore._plugins.hub.client.client import Client

    cookies = {"access_token": access_token}
    if refresh_token:
        cookies["refresh_token"] = refresh_token

    with Client() as client:
        client.post(urljoin(URI(), "identity/oauth/logout"), cookies=cookies)


def _token_expired(expires_at: str | None, *, margin_seconds: int = 60) -> bool:
    """Return whether ``expires_at`` (ISO 8601) is past, within a safety margin."""
    if not expires_at:
        return True
    try:
        expiration = datetime.fromisoformat(expires_at)
    except ValueError:
        return True
    if expiration.tzinfo is None:
        expiration = expiration.replace(tzinfo=UTC)
    return expiration <= datetime.now(UTC) + timedelta(seconds=margin_seconds)


def fresh_token(*, relogin: bool = True, timeout: int = 600) -> store.Token | None:
    """Return a non-expired stored token, refreshing or re-logging in as needed.

    Mirrors the SDK :class:`Token` lifecycle for the persisted CLI token: reuse it
    while valid, refresh it on the fly when expired, and only relaunch the device
    login when the refresh token is no longer usable. Returns ``None`` when no
    token is stored, so callers do not trigger a surprise browser login.
    """
    from skore._plugins.hub.authentication import store

    token = store.load()
    if not token or not token.get("access_token"):
        return None

    if not _token_expired(token.get("expires_at")):
        return token

    refresh = token.get("refresh_token")
    if refresh:
        try:
            access, refresh, expires_at = post_oauth_refresh_token(refresh)
        except HTTPError:
            pass  # refresh failed (expired/revoked); fall back to interactive login
        else:
            token = {
                "uri": token.get("uri") or URI(),
                "access_token": access,
                "refresh_token": refresh,
                "expires_at": expires_at,
            }
            store.save(token)
            return token

    if relogin:
        access, refresh, expires_at = interactive_device_login(timeout=timeout)
        token = {
            "uri": URI(),
            "access_token": access,
            "refresh_token": refresh,
            "expires_at": expires_at,
        }
        store.save(token)
        return token

    return None


class Token:
    """
    Token used for ``skore hub`` authentication, as HTTP header parameters.

    Notes
    -----
    Refresh the token on-the-fly if necessary.
    """

    def __init__(self, *, timeout: int = 600, live: Live | None = None) -> None:
        url, device_code, user_code = get_oauth_device_login()
        panel = Panel(
            Align.center(
                "[b]API key not detected.[/b]\n\n"
                "Starting interactive authentication for the session.\n"
                "[i]We recommend that you create an API key and use it to log in, "
                "at [link=https://skore.probabl.ai/account]"
                "https://skore.probabl.ai/account[/link].[/i]\n\n"
                "Opening browser for interactive authentication; if this fails, "
                f"please visit:\n[link={url}]{url}[/link]"
            ),
            title="[cyan]Login to [bold]Skore Hub",
            border_style="cyan",
            padding=1,
        )

        if live:
            live.update(panel)
            live.refresh()
        else:
            console.print(panel, soft_wrap=True)

        open_webbrowser(url)

        get_oauth_device_code_probe(device_code, timeout=timeout)
        post_oauth_device_callback(device_code, user_code)

        access, refreshment, expiration = get_oauth_device_token(device_code)

        self.__lock = RLock()
        self.__access = access
        self.__refreshment = refreshment
        self.__expiration = datetime.fromisoformat(expiration)

    def __call__(self) -> dict[str, str]:  # noqa: D102
        with self.__lock:
            if self.__expiration <= datetime.now(UTC):
                access, refreshment, expiration = post_oauth_refresh_token(
                    self.__refreshment
                )

                self.__access = access
                self.__refreshment = refreshment
                self.__expiration = datetime.fromisoformat(expiration)

            return {"Authorization": f"Bearer {self.__access}"}
