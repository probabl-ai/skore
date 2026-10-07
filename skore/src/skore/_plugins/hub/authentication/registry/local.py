"""
Registry used to persist on disk the API keys used for ``skore hub`` authentication.

API keys are stored in ``~/.skore.hub/credentials.json`` as a JSON object:

    {
        "type": "plaintext",
        "keys": [
            {
                "id": 42,
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
                "id": 42,
                "host": "<host>",
                "workspace": "<workspace>",
                "key": null
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

When ``host`` is omitted on :meth:`set`, :meth:`get` or :meth:`delete`, its value is
derived from :func:`URI`. Hosts are compared after :func:`normalize`, so a trailing
slash or differences in scheme/host case do not create a distinct credential.
"""

from __future__ import annotations

from collections.abc import Callable, Generator
from contextlib import suppress
from dataclasses import asdict, dataclass
from functools import wraps
from json import dump, load
from pathlib import Path
from shutil import move
from stat import S_IMODE
from tempfile import NamedTemporaryFile, gettempdir
from typing import TYPE_CHECKING, Any, Final, ParamSpec, TypeVar

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


@dataclass
class Key:
    id: int
    host: str
    workspace: str
    key: str | None = None


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


def keys() -> Generator[tuple[int, str, str]]:
    """Yield ``(id, host, workspace)`` keys stored in the registry."""
    filepath = setup()

    with filepath.open() as file:
        for entry in load(file)["keys"]:
            yield (
                entry["id"],
                normalize(entry["host"]),
                entry["workspace"],
            )


@lock
def set(*, id: int, host: str | None = None, workspace: str, key: str) -> None:
    """
    Insert or replace the API key for ``host`` and ``workspace``.

    Parameters
    ----------
    id : int
        ID of the API key, used to revoke it from the hub.
    host : str, optional
        URI associated with the API key. If omitted, :func:`URI` is used.
    workspace : str
        Workspace associated with the API key.
    key : str
        API key to persist.
    """
    filepath = setup()

    with filepath.open() as file:
        registry = load(file)

    host = normalize(host or URI())
    new = Key(id=id, host=host, workspace=workspace)

    if registry["type"] == "secret":
        set_password(KEYRING_SERVICE, f"{host}:{workspace}", key)
    else:
        new.key = key

    for i, entry in enumerate(registry["keys"]):
        i_key = Key(**entry)

        if (normalize(i_key.host) == host) and (i_key.workspace == workspace):
            registry["keys"][i] = asdict(new)
            break
    else:
        registry["keys"].append(asdict(new))

    save_on_disk(
        registry=registry,
        filepath=filepath,
        mode=S_IMODE(filepath.stat().st_mode),
    )


def get(*, host: str | None = None, workspace: str) -> Key | None:
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
    Key or None
        The matching API key, or ``None`` if none is stored.
    """
    filepath = setup()

    with filepath.open() as file:
        registry = load(file)

    host = normalize(host or URI())

    for entry in registry["keys"]:
        key = Key(**entry)

        if (normalize(key.host) == host) and (key.workspace == workspace):
            if registry["type"] == "secret":
                key.key = get_password(KEYRING_SERVICE, f"{host}:{workspace}")

            return key

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

    with filepath.open() as file:
        registry = load(file)

    host = normalize(host or URI())

    if registry["type"] == "secret":
        with suppress(PasswordDeleteError):
            delete_password(KEYRING_SERVICE, f"{host}:{workspace}")

    registry["keys"] = [
        entry
        for entry in registry["keys"]
        if (normalize(entry["host"]) != host) or (entry["workspace"] != workspace)
    ]

    save_on_disk(
        registry=registry,
        filepath=filepath,
        mode=S_IMODE(filepath.stat().st_mode),
    )
