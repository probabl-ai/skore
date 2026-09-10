"""API key used for ``skore hub`` authentication."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from functools import cached_property
from itertools import chain, filterfalse
from json import dump, load
from os import environ
from pathlib import Path
from shutil import move
from tempfile import NamedTemporaryFile, gettempdir
from typing import TYPE_CHECKING, Final, cast

from filelock import FileLock

from skore._plugins.hub.authentication.uri import URI

if TYPE_CHECKING:
    from collections.abc import Generator

ENV_VAR_NAME: Final[str] = "SKORE_HUB_API_KEY"


class APIKeyError(KeyError):
    pass


def API_key() -> Callable[[], dict[str, str]]:
    """Retrieve the API key from the environment as an HTTP header."""
    if ENV_VAR_NAME in environ:
        return lambda: {"X-API-Key": environ[ENV_VAR_NAME]}

    raise APIKeyError()


class Registry:
    """
    Registry used to persist API keys on disk.

    API keys are stored in ``~/.skore.hub/credentials.json`` as a JSON list:

        [
            {
                "host": "<host>",
                "workspace": "<workspace>",
                "api_key": "<api_key>",
            },
        ]

    Notes
    -----
    Writes are serialized with a file lock. When ``uri`` is omitted on :meth:`persist`,
    the URI is derived from :func:`URI`.
    """

    @cached_property
    def filepath(self) -> Path:
        """Path to the credentials file, creating an empty registry if missing."""
        file = Path.home() / ".skore.hub" / "credentials.json"

        if not file.exists():
            file.parent.mkdir(exist_ok=True)
            file.write_text("[]")

        return file

    @contextmanager
    def lock(self) -> Generator[None]:
        """Acquire an exclusive lock around writes to the credentials file."""
        lockfile = Path(gettempdir()) / ".skore_hub_credentials.json.lock"

        with FileLock(lockfile):
            yield

    def __iter__(self) -> Generator[tuple[str, str]]:
        """Yield ``(host, workspace)`` pairs stored in the registry."""
        with self.filepath.open() as file:
            for credential in load(file):
                yield (
                    credential["host"],
                    credential["workspace"],
                )

    def get(self, *, uri: str, workspace: str) -> str:
        """
        Return the API key for ``uri`` and ``workspace``.

        Parameters
        ----------
        uri : str
            URI associated with the API key.
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
        with self.filepath.open() as file:
            for credential in load(file):
                if credential["host"] == uri and credential["workspace"] == workspace:
                    return cast(str, credential["api_key"])

        raise APIKeyError()

    def persist(self, *, uri: str | None = None, workspace: str, api_key: str) -> None:
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

        with (
            self.lock(),
            open(self.filepath) as credentials_file_reader,
            NamedTemporaryFile(mode="w", delete=False) as credentials_tmpfile_writer,
        ):
            credentials = filterfalse(
                lambda cred: cred["host"] == uri and cred["workspace"] == workspace,
                load(credentials_file_reader),
            )

            # Save the new credentials to the tmpfile, taking care not to truncate the
            # previous credentials in case of JSON/IO error.
            dump(
                list(
                    chain(
                        credentials,
                        [
                            {
                                "host": uri,
                                "workspace": workspace,
                                "api_key": api_key,
                            }
                        ],
                    )
                ),
                credentials_tmpfile_writer,
            )

            credentials_tmpfile_writer.seek(0)
            credentials_tmpfile_writer.flush()

            # Move tmpfile to file
            move(credentials_tmpfile_writer.name, self.filepath)
