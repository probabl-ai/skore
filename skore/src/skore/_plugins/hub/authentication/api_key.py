"""API key used for ``skore hub`` authentication."""

from __future__ import annotations

from collections.abc import Callable, Generator
from contextlib import suppress
from functools import cached_property, wraps
from json import dump, load
from os import environ
from pathlib import Path
from shutil import move
from tempfile import NamedTemporaryFile, gettempdir
from typing import Any, Final, Literal, ParamSpec, TypeVar, cast

from filelock import FileLock
from keyring import delete_password, get_keyring, get_password, set_password
from keyring.backends.fail import Keyring as FailBackend
from keyring.core import recommended
from keyring.errors import PasswordDeleteError

from skore._plugins.hub.authentication.uri import URI

ENV_VAR_NAME: Final[str] = "SKORE_HUB_API_KEY"
KEYRING_SERVICE: Final[str] = "skore"
P = ParamSpec("P")
R = TypeVar("R")


class APIKeyError(KeyError):
    pass


def API_key() -> Callable[[], dict[str, str]]:
    """Retrieve the API key from the environment as an HTTP header."""
    if ENV_VAR_NAME in environ:
        return lambda: {"X-API-Key": environ[ENV_VAR_NAME]}

    raise APIKeyError()


def locked(method: Callable[P, R]) -> Callable[P, R]:
    """Acquire an exclusive lock around writes to the credentials file."""

    @wraps(method)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        lockfile = Path(gettempdir()) / ".skore_hub_credentials.json.lock"

        with FileLock(lockfile):
            return method(*args, **kwargs)

    return wrapper


class Registry:
    """
    Registry used to persist API keys on disk.

    API keys are stored in ``~/.skore.hub/credentials.json`` as a JSON object:

        {
            "type": "plaintext",
            "keys": [
                {
                    "host": "<host>",
                    "workspace": "<workspace>",
                    "key": "<key>"
                }
            ]
        }

        {
            "type": "secret",
            "keys": [
                {
                    "host": "<host>",
                    "workspace": "<workspace>"
                }
            ]
        }

    The storage mode is chosen automatically when the registry file is created:
    ``secret`` if a recommended `keyring <https://github.com/jaraco/keyring>`_
    backend is available, otherwise ``plaintext``. In ``secret`` mode the API key
    is stored in the system keyring rather than in the JSON file.

    Notes
    -----
    The registry is locked during write operations.
    When ``uri`` is omitted on :meth:`set`, :meth:`get` or :meth:`delete`,
    the URI is derived from :func:`URI`.
    """

    @locked
    def __init__(self) -> None:
        self.filepath = Path.home() / ".skore.hub" / "credentials.json"
        self.filepath.parent.mkdir(exist_ok=True)

        if not self.filepath.exists():
            is_keyring_available = (
                (backend := get_keyring())
                and not isinstance(backend, FailBackend)
                and recommended(backend)
            )

            self.__save_on_disk(
                {
                    "type": (is_keyring_available and "secret") or "plaintext",
                    "keys": [],
                }
            )

    @cached_property
    def type(self) -> Literal["secret", "plaintext"]:
        """Storage backend recorded in the credentials file."""
        with self.filepath.open() as file:
            return cast(Literal["secret", "plaintext"], load(file)["type"])

    def __save_on_disk(self, registry: dict[str, Any], /) -> None:
        """Write ``registry`` atomically so a JSON error cannot truncate the file."""
        with NamedTemporaryFile(mode="w", delete=False) as file:
            dump(registry, file, indent=4)

        move(file.name, self.filepath)

    def __iter__(self) -> Generator[tuple[str, str]]:
        """Yield ``(host, workspace)`` pairs stored in the registry."""
        with self.filepath.open() as file:
            for credential in load(file)["keys"]:
                yield (credential["host"], credential["workspace"])

    @locked
    def set(self, *, uri: str | None = None, workspace: str, api_key: str) -> None:
        """
        Insert or replace the API key for ``uri`` and ``workspace``.

        Parameters
        ----------
        uri : str, optional
            URI associated with the API key. If omitted, :func:`URI` is used.
        workspace : str
            Workspace associated with the API key.
        api_key : str
            API key to persist.
        """
        uri = uri or URI()

        with self.filepath.open() as file:
            registry = load(file)

        keys = [
            credential
            for credential in registry["keys"]
            if credential["host"] != uri or credential["workspace"] != workspace
        ]

        if registry["type"] == "secret":
            set_password(KEYRING_SERVICE, f"{uri}:{workspace}", api_key)
            keys.append({"host": uri, "workspace": workspace})
        else:
            keys.append({"host": uri, "workspace": workspace, "key": api_key})

        registry["keys"] = keys

        self.__save_on_disk(registry)

    def get(self, *, uri: str | None = None, workspace: str) -> str:
        """
        Return the API key for ``uri`` and ``workspace``.

        Parameters
        ----------
        uri : str, optional
            URI associated with the API key. If omitted, :func:`URI` is used.
        workspace : str
            Workspace associated with the API Key.

        Returns
        -------
        str
            The matching API key.

        Raises
        ------
        APIKeyError
            If no credential matches ``uri`` and ``workspace``.
        """
        uri = uri or URI()

        with self.filepath.open() as file:
            registry = load(file)

        for credential in registry["keys"]:
            if credential["host"] == uri and credential["workspace"] == workspace:
                if registry["type"] == "secret":
                    if api_key := get_password(KEYRING_SERVICE, f"{uri}:{workspace}"):
                        return api_key

                    raise APIKeyError()

                return cast(str, credential["key"])

        raise APIKeyError()

    @locked
    def delete(self, *, uri: str | None = None, workspace: str) -> None:
        """
        Remove the API key for ``uri`` and ``workspace``.

        Parameters
        ----------
        uri : str, optional
            URI associated with the API key. If omitted, :func:`URI` is used.
        workspace : str
            Workspace associated with the API key.

        Raises
        ------
        APIKeyError
            If no credential matches ``uri`` and ``workspace``.
        """
        uri = uri or URI()

        with self.filepath.open() as file:
            registry = load(file)

        keys = [
            credential
            for credential in registry["keys"]
            if credential["host"] != uri or credential["workspace"] != workspace
        ]

        if len(keys) == len(registry["keys"]):
            raise APIKeyError()

        if registry["type"] == "secret":
            with suppress(PasswordDeleteError):
                delete_password(KEYRING_SERVICE, f"{uri}:{workspace}")

        registry["keys"] = keys

        self.__save_on_disk(registry)
