from contextlib import contextmanager
from itertools import chain, filterfalse
from json import dump, load
from pathlib import Path
from shutil import move
from sys import modules
from tempfile import NamedTemporaryFile, gettempdir
from types import ModuleType
from typing import TYPE_CHECKING, cast

from filelock import FileLock

if TYPE_CHECKING:
    from collections.abc import Generator


class PersistedCredentials(ModuleType):
    """
    [
        {
            "host": "<host>",
            "workspace": "<workspace>",
            "apikey": "<apikey>",
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
                    return cast(str, credential["apikey"])

        raise KeyError

    def persist(self, *, uri: str, workspace: str, apikey: str) -> None:
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
                        [{"host": uri, "workspace": workspace, "apikey": apikey}],
                    )
                ),
                credentials_tmpfile_writer,
            )

            credentials_tmpfile_writer.seek(0)
            credentials_tmpfile_writer.flush()

            # Move tmpfile to file
            move(credentials_tmpfile_writer.name, self.filepath)


# Hook the current module
modules[__name__].__class__ = PersistedCredentials
