"""
Registry used to persist on disk the API keys used for ``skore hub`` authentication.

API keys are stored in ``~/.skore.hub/credentials.json`` as a JSON object:

    {
        "type": "plaintext",
        "keys": [
            {
                "host": "<host>",
                "workspace": "<workspace>",
                "api_key_id": 42,
                "key": "<key>"
            }
        ]
    }

    {
        "type": "secret",
        "keys": [
            {
                "host": "<host>",
                "workspace": "<workspace>",
                "api_key_id": 42
            }
        ]
    }

The storage mode is chosen automatically when the registry file is created:
``secret`` if a recommended `keyring <https://github.com/jaraco/keyring>`_ backend is
available, otherwise ``plaintext``. In ``secret`` mode the API key is stored in the
system keyring rather than in the JSON file.

The credentials directory is created with mode ``0o700`` and the JSON file with mode
``0o600``; later operations keep existing modes.

Notes
-----
The registry is locked during write operations.

When ``host`` is omitted on :meth:`set`, :meth:`get`, :meth:`get_api_key_id` or
:meth:`delete`, its value is derived from :func:`URI`. Hosts are compared after
:func:`normalize`, so a trailing slash or differences in scheme/host case do not
create a distinct credential.
"""

from __future__ import annotations

from collections.abc import Callable, Generator
from contextlib import suppress
from functools import wraps
from json import dump, load
from pathlib import Path
from shutil import move
from stat import S_IMODE
from tempfile import NamedTemporaryFile, gettempdir
from typing import TYPE_CHECKING, Any, Final, ParamSpec, TypeVar, cast

from filelock import FileLock
from keyring import delete_password, get_keyring, get_password, set_password
from keyring.backends.fail import Keyring as FailBackend
from keyring.core import recommended
from keyring.errors import PasswordDeleteError

from skore._plugins.hub.authentication.uri import URI, normalize

if TYPE_CHECKING:
    P = ParamSpec("P")
    R = TypeVar("R")


KEYRING_SERVICE: Final[str] = "skore"
DIRECTORY_MODE: Final[int] = 0o700
FILE_MODE: Final[int] = 0o600


def lock(function: Callable[P, R]) -> Callable[P, R]:
    """Acquire an exclusive lock around writes to the credentials file."""

    @wraps(function)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        lockfile = Path(gettempdir()) / ".skore_hub_credentials.json.lock"

        with FileLock(lockfile):
            return function(*args, **kwargs)

    return wrapper


def setup() -> Path:
    directory = Path.home() / ".skore.hub"

    if not directory.exists():
        directory.mkdir()
        directory.chmod(DIRECTORY_MODE)

    filepath = directory / "credentials.json"

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
            mode=FILE_MODE,
        )

    return filepath


def save_on_disk(*, registry: dict[str, Any], filepath: Path, mode: int) -> None:
    """Write ``registry`` atomically so a JSON error cannot truncate the file."""
    with NamedTemporaryFile(mode="w", delete=False) as tmpfile:
        dump(registry, tmpfile, indent=4)

    move(tmpfile.name, filepath)
    filepath.chmod(mode)


def keys() -> Generator[tuple[str, str]]:
    """Yield ``(host, workspace)`` keys stored in the registry."""
    filepath = setup()

    with filepath.open() as file:
        for credential in load(file)["keys"]:
            yield (normalize(credential["host"]), credential["workspace"])


@lock
def set(
    *, host: str | None = None, workspace: str, api_key: str, api_key_id: int
) -> None:
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
    api_key_id : int
        Hub id of this API key, used to revoke it later.
    """
    filepath = setup()
    host = normalize(host or URI())
    mode = S_IMODE(filepath.stat().st_mode)

    with filepath.open() as file:
        registry = load(file)

    if registry["type"] == "secret":
        set_password(KEYRING_SERVICE, f"{host}:{workspace}", api_key)
        new = {"host": host, "workspace": workspace, "api_key_id": api_key_id}
    else:
        new = {
            "host": host,
            "workspace": workspace,
            "api_key_id": api_key_id,
            "key": api_key,
        }

    for i, credential in enumerate(registry["keys"]):
        if (
            normalize(credential["host"]) == host
            and credential["workspace"] == workspace
        ):
            registry["keys"][i] = new
            break
    else:
        registry["keys"].append(new)

    save_on_disk(registry=registry, filepath=filepath, mode=mode)


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
    str or None
        The matching API key, or ``None`` if none is stored.
    """
    filepath = setup()
    host = normalize(host or URI())

    with filepath.open() as file:
        registry = load(file)

    for credential in registry["keys"]:
        if (
            normalize(credential["host"]) == host
            and credential["workspace"] == workspace
        ):
            if registry["type"] == "secret":
                return get_password(KEYRING_SERVICE, f"{host}:{workspace}")

            return cast(str, credential["key"])

    return None


def get_api_key_id(*, host: str | None = None, workspace: str) -> int | None:
    """
    Return the Hub API key id for ``host`` and ``workspace``.

    Parameters
    ----------
    host : str, optional
        URI associated with the API key. If omitted, :func:`URI` is used.
    workspace : str
        Workspace associated with the API key.

    Returns
    -------
    int or None
        The matching Hub id, or ``None`` if none is stored.
    """
    filepath = setup()
    host = normalize(host or URI())

    with filepath.open() as file:
        registry = load(file)

    for credential in registry["keys"]:
        if (
            normalize(credential["host"]) == host
            and credential["workspace"] == workspace
        ):
            api_key_id = credential.get("api_key_id")
            return int(api_key_id) if api_key_id is not None else None

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
    host = normalize(host or URI())
    mode = S_IMODE(filepath.stat().st_mode)

    with filepath.open() as file:
        registry = load(file)

    if registry["type"] == "secret":
        with suppress(PasswordDeleteError):
            delete_password(KEYRING_SERVICE, f"{host}:{workspace}")

    registry["keys"] = [
        credential
        for credential in registry["keys"]
        if normalize(credential["host"]) != host or credential["workspace"] != workspace
    ]

    save_on_disk(registry=registry, filepath=filepath, mode=mode)
