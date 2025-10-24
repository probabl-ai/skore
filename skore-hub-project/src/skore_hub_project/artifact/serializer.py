"""Function definition of the content ``Serializer``."""

from __future__ import annotations

from functools import cached_property
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from blake3 import blake3 as Blake3


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
        """
        The checksum of the serialized content.

        Notes
        -----
        Depending on the size of the serialized content, the checksum can be computed on
        one or more threads:

            Note that this can be slower for inputs shorter than ~1 MB

        https://github.com/oconnor663/blake3-py
        """
        from skore_hub_project import bytes_to_b64_str

        hasher = Blake3(max_threads=(1 if self.size < 1e6 else Blake3.AUTO))
        checksum = hasher.update_mmap(self.filepath).digest()

        return f"blake3-{bytes_to_b64_str(checksum)}"

    @cached_property
    def size(self) -> int:
        """The size of the serialized content, in bytes."""
        return self.filepath.stat().st_size
