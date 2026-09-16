"""
Registry used to persist on disk the API keys used for ``skore hub`` authentication.

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
backend is available, otherwise ``plaintext``. In ``secret`` mode the API key is stored
in the system keyring rather than in the JSON file.

Notes
-----
The registry is locked during write operations.
When ``host`` is omitted on :meth:`set`, :meth:`get` or :meth:`delete`, its value is
derived from :func:`URI`.
"""

from __future__ import annotations

from collections.abc import Callable, Generator
from contextlib import suppress
from functools import wraps
from json import dump, load
from pathlib import Path
from shutil import move
from tempfile import NamedTemporaryFile, gettempdir
from typing import Any, ParamSpec, TypeVar, cast, TYPE_CHECKING, Final

from filelock import FileLock
from keyring import delete_password, get_keyring, get_password, set_password
from keyring.backends.fail import Keyring as FailBackend
from keyring.core import recommended
from keyring.errors import PasswordDeleteError

from skore._plugins.hub.authentication.uri import URI

if TYPE_CHECKING:
    P = ParamSpec("P")
    R = TypeVar("R")


KEYRING_SERVICE: Final[str] = "skore"


def lock(function: Callable[P, R]) -> Callable[P, R]:
    """Acquire an exclusive lock around writes to the credentials file."""

    @wraps(function)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        lockfile = Path(gettempdir()) / ".skore_hub_credentials.json.lock"

        with FileLock(lockfile):
            return function(*args, **kwargs)

    return wrapper


def setup() -> Path:
    filepath = Path.home() / ".skore.hub" / "credentials.json"
    filepath.parent.mkdir(exist_ok=True)

    if not filepath.exists():
        is_keyring_available = (
            (backend := get_keyring())
            and not isinstance(backend, FailBackend)
            and recommended(backend)
        )

        save_on_disk(
            registry={
                "type": (is_keyring_available and "secret") or "plaintext",
                "keys": [],
            },
            filepath=filepath,
        )

    return filepath


def save_on_disk(*, registry: dict[str, Any], filepath: Path) -> None:
    """Write ``registry`` atomically so a JSON error cannot truncate the file."""
    with NamedTemporaryFile(mode="w", delete=False) as file:
        dump(registry, file, indent=4)

    move(file.name, filepath)


def keys() -> Generator[tuple[str, str]]:
    """Yield ``(host, workspace)`` keys stored in the registry."""
    filepath = setup()

    with filepath.open() as file:
        for credential in load(file)["keys"]:
            yield (credential["host"], credential["workspace"])


@lock
def set(*, host: str | None = None, workspace: str, api_key: str) -> None:
    """
    Insert or replace the API key for ``host`` and ``workspace``.

    Parameters
    ----------
    host : str, optional
        URI associated with the API key. If omitted, :func:`URI` is used.
    workspace : str
        Workspace associated with the API key.
    api_key : str
        API key to persist.
    """
    filepath = setup()
    host = host or URI()

    with filepath.open() as file:
        registry = load(file)

    keys = [
        credential
        for credential in registry["keys"]
        if credential["host"] != host or credential["workspace"] != workspace
    ]

    if registry["type"] == "secret":
        set_password(KEYRING_SERVICE, f"{host}:{workspace}", api_key)
        keys.append({"host": host, "workspace": workspace})
    else:
        keys.append({"host": host, "workspace": workspace, "key": api_key})

    registry["keys"] = keys

    save_on_disk(registry=registry, filepath=filepath)


def get(*, host: str | None = None, workspace: str) -> str | None:
    """
    Return the API key for ``host`` and ``workspace``.

    Parameters
    ----------
    host : str, optional
        URI associated with the API key. If omitted, :func:`URI` is used.
    workspace : str
        Workspace associated with the API Key.

    Returns
    -------
    str
        The matching API key.
    """
    filepath = setup()
    host = host or URI()

    with filepath.open() as file:
        registry = load(file)

    for credential in registry["keys"]:
        if credential["host"] == host and credential["workspace"] == workspace:
            if registry["type"] == "secret":
                return get_password(KEYRING_SERVICE, f"{host}:{workspace}")
            return cast(str, credential["key"])
    return None


@lock
def delete(*, host: str | None = None, workspace: str) -> None:
    """
    Remove the API key for ``host`` and ``workspace``.

    Parameters
    ----------
    host : str, optional
        URI associated with the API key. If omitted, :func:`URI` is used.
    workspace : str
        Workspace associated with the API key.
    """
    filepath = setup()
    host = host or URI()

    with filepath.open() as file:
        registry = load(file)

    keys = [
        credential
        for credential in registry["keys"]
        if credential["host"] != host or credential["workspace"] != workspace
    ]

    if len(keys) == len(registry["keys"]):
        return

    if registry["type"] == "secret":
        with suppress(PasswordDeleteError):
            delete_password(KEYRING_SERVICE, f"{host}:{workspace}")

    registry["keys"] = keys

    save_on_disk(registry=registry, filepath=filepath)
