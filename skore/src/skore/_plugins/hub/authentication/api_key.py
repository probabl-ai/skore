"""API key used for ``skore hub`` authentication."""

from collections.abc import Callable
from contextlib import contextmanager
from itertools import chain, filterfalse
from json import dump, load
from os import environ
from pathlib import Path
from shutil import move
from tempfile import NamedTemporaryFile, gettempdir
from typing import TYPE_CHECKING, Final, cast

from filelock import FileLock

if TYPE_CHECKING:
    from collections.abc import Generator

ENV_VAR_NAME: Final[str] = "SKORE_HUB_API_KEY"


class APIKeyError(Exception): ...


def API_key() -> Callable[[], dict[str, str]]:
    """Get the API key used for ``skore hub`` authentication.

    In the form of HTTP header parameters.
    """
    if ENV_VAR_NAME in environ:
        return lambda: {"X-API-Key": environ[ENV_VAR_NAME]}

    raise APIKeyError()


class PersistedAPIKey:
    """
    [
        {
            "host": "<host>",
            "workspace": "<workspace>",
            "api_key": "<api_key>",
        },
    ]
    """

    @property
    def filepath(self) -> Path:
        file = Path.home() / ".skore.hub" / "credentials.json"

        if not file.exists():
            file.parent.mkdir(exist_ok=True)
            file.write_text("[]")

        return file

    @contextmanager
    def lock(self) -> Generator[None]:
        lockfile = Path(gettempdir()) / ".skore_hub_credentials.json.lock"

        with FileLock(lockfile):
            yield

    def __iter__(self) -> Generator[tuple[str, str]]:
        with self.filepath.open() as file:
            for credential in load(file):
                yield (
                    credential["host"],
                    credential["workspace"],
                )

    def get(self, *, uri: str, workspace: str) -> str:
        with self.filepath.open() as file:
            for credential in load(file):
                if credential["host"] == uri and credential["workspace"] == workspace:
                    return cast(str, credential["api_key"])

        raise KeyError

    def persist(self, *, uri: str, workspace: str, api_key: str) -> None:
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
                        [{"host": uri, "workspace": workspace, "api_key": api_key}],
                    )
                ),
                credentials_tmpfile_writer,
            )

            credentials_tmpfile_writer.seek(0)
            credentials_tmpfile_writer.flush()

            # Move tmpfile to file
            move(credentials_tmpfile_writer.name, self.filepath)
