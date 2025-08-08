"""Function definition of the object ``Serializer``."""

from __future__ import annotations

from functools import cached_property
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING

from blake3 import blake3 as Blake3
from joblib import dump

from ..item.item import bytes_to_b64_str

if TYPE_CHECKING:
    from typing import Any


class Serializer:
    """Serialize an object using ``joblib``, on disk to reduce RAM footprint."""

    def __init__(self, o: Any):
        with self.filepath.open("wb") as file:
            dump(o, file)

    def __enter__(self):  # noqa: D105
        return self

    def __exit__(self, exc_type, exc_value, traceback):  # noqa: D105
        self.filepath.unlink(True)

    @cached_property
    def filepath(self) -> Path:
        """The filepath used to serialize the object."""
        with NamedTemporaryFile(mode="w+b", delete=False) as file:
            return Path(file.name)

    @cached_property
    def checksum(self) -> str:
        """
        The checksum of the serialized object.

        Notes
        -----
        Depending on the size of the serialized object, the checksum can be computed on
        one or more threads:

            Note that this can be slower for inputs shorter than ~1 MB

        https://github.com/oconnor663/blake3-py
        """
        hasher = Blake3(max_threads=(1 if self.size < 1e6 else Blake3.AUTO))
        checksum = hasher.update_mmap(self.filepath).digest()

        return f"blake3-{bytes_to_b64_str(checksum)}"

    @cached_property
    def size(self) -> int:
        """The size of the serialized object, in bytes."""
        return self.filepath.stat().st_size
