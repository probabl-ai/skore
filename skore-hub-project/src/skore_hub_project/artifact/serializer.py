"""Function definition of the content ``Serializer``."""

from __future__ import annotations

from functools import cached_property
from hashlib import blake2b
from pathlib import Path
from sys import version_info
from tempfile import NamedTemporaryFile
from typing import Any

if version_info >= (3, 11):
    from hashlib import file_digest
else:

    def file_digest(fileobj, digest, /, *, _bufsize=2**18):  # type: ignore
        """Backported from https://github.com/python/cpython/blob/3.14/Lib/hashlib.py#L195."""
        digestobj = digest()

        if hasattr(fileobj, "getbuffer"):
            digestobj.update(fileobj.getbuffer())
            return digestobj

        if not (
            hasattr(fileobj, "readinto")
            and hasattr(fileobj, "readable")
            and fileobj.readable()
        ):
            raise ValueError(
                f"'{fileobj!r}' is not a file-like object in binary reading mode."
            )

        buf = bytearray(_bufsize)  # Reusable buffer to reduce allocations.
        view = memoryview(buf)
        while True:
            size = fileobj.readinto(buf)
            if size is None:
                raise BlockingIOError("I/O operation would block.")
            if size == 0:
                break  # EOF
            digestobj.update(view[:size])

        return digestobj


class Serializer:
    """Serialize a content directly on disk to reduce RAM footprint."""

    def __init__(self, content: str | bytes):
        if isinstance(content, str):
            self.filepath.write_text(content, encoding="utf-8")
        else:
            self.filepath.write_bytes(content)

    def __enter__(self) -> Serializer:  # noqa: D105
        return self

    def __exit__(self, *args: Any) -> None:  # noqa: D105
        self.filepath.unlink(True)

    @cached_property
    def filepath(self) -> Path:
        """The filepath used to serialize the content."""
        with NamedTemporaryFile(mode="w+b", delete=False) as file:
            return Path(file.name)

    @cached_property
    def checksum(self) -> str:
        """The checksum of the serialized content using `BLAKE2b`."""
        with open(self.filepath, "rb") as file:
            digest = file_digest(file, blake2b)

        return f"blake2b-{digest.hexdigest()}"

    @cached_property
    def size(self) -> int:
        """The size of the serialized content, in bytes."""
        return self.filepath.stat().st_size
