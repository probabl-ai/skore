"""Function definition of the content ``Serializer``."""

from __future__ import annotations

from functools import cached_property
from hashlib import blake2b, file_digest
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any


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
