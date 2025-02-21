from __future__ import annotations

import functools
import json
import pathlib
import tempfile
import time
import webbrowser
from urllib.parse import urljoin

import httpx


class Client(httpx.Client):
    URI = "https://skh.k.probabl.dev"

    def request(self, method: str, url: httpx.URL | str, **kwargs) -> httpx.Response:
        response = super().request(method, urljoin(self.URI, url), **kwargs)
        response.raise_for_status()

        return response.json()


class AuthenticationToken:
    FILEPATH = pathlib.Path(tempfile.gettempdir(), "skore.token")

    def __init__(self):
        try:
            self.__access, self.__refreshment = json.loads(self.FILEPATH.read_text())
        except FileNotFoundError:
            self.__create()

    def __create(self):
        with Client() as client:
            # Request a new authorization URL
            response = client.get("identity/oauth/device/login")
            authorization_url = response["authorization_url"]
            device_code = response["device_code"]
            user_code = response["user_code"]

            # Display authentication info to the user
            print(
                "Attempting to automatically open the SSO authorization page in your "
                "default browser."
            )
            print(
                "If the browser does not open or you wish to use a different device to "
                "authorize this request, open the following URL:"
            )
            print()
            print(authorization_url)
            print()
            print("Then enter the code:")
            print()
            print(user_code)
            print()

            # Open the default browser
            webbrowser.open(authorization_url)

            # Start polling Skore-Hub, waiting for the token
            while True:
                try:
                    response = client.get(
                        f"identity/oauth/device/token?device_code={device_code}"
                    )
                except httpx.HTTPError:
                    time.sleep(0.5)
                else:
                    tokens = response["token"]

                    self.__access = tokens["access_token"]
                    self.__refreshment = tokens["refresh_token"]
                    self.__save()
                    break

    def __save(self):
        self.FILEPATH.write_text(json.dumps((self.__access, self.__refreshment)))

    def refresh(self):
        with Client() as client:
            response = client.post(
                "identity/oauth/token/refresh",
                headers={
                    "Authorization": f"Bearer {self.__access}",
                    "Content-Type": "application/json",
                },
                json={"refresh_token": self.__refreshment},
            )

            tokens = response["token"]

            self.__access = tokens["access_token"]
            self.__refreshment = tokens["refresh_token"]
            self.__save()

    def __str__(self):
        return self.__access

    def __repr__(self):
        return f"AuthenticationToken('{self.__access:.10}[...]')"


TOKEN = AuthenticationToken()


def authenticate(fn):
    """
    Authenticate current user by requesting token before calling ``fn``.

    Parameters
    ----------
    fn : callable
        Callable that is protected from unauthenticated/unauthorized users.

    Returns
    -------
    wrapper : callable
        Wrapper that authenticate current user and pass ``AuthenticationToken`` as first
        argument to ``fn``.
    """
    return functools.wraps(fn)(functools.partial(fn, token=TOKEN))
