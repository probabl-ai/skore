"""Function definition of the object ``Serializer``."""

from abc import ABC, abstractmethod
from functools import cached_property
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from blake3 import blake3 as Blake3


class Serializer(ABC):  # noqa: B024
    """Interface to serialize an object directly on disk to reduce RAM footprint."""

    @abstractmethod
    def __init__(self, _: Any): ...

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
        from skore_hub_project import bytes_to_b64_str

        hasher = Blake3(max_threads=(1 if self.size < 1e6 else Blake3.AUTO))
        checksum = hasher.update_mmap(self.filepath).digest()

        return f"blake3-{bytes_to_b64_str(checksum)}"

    @cached_property
    def size(self) -> int:
        """The size of the serialized object, in bytes."""
        return self.filepath.stat().st_size


class JoblibSerializer(Serializer):
    def __init__(self, object: Any):
        import joblib

        with self.filepath.open("wb") as file:
            joblib.dump(object, file)


class TxtSerializer(Serializer):
    def __init__(self, txt: str):
        self.filepath.write_text(txt)


class BytesSerializer(Serializer):
    def __init__(self, txt: bytes):
        self.filepath.write_bytes(txt)


class JsonSerializer(Serializer):
    def __init__(self, object: Any):
        import orjson

        self.filepath.write_bytes(
            orjson.dumps(
                object,
                option=(orjson.OPT_NON_STR_KEYS | orjson.OPT_SERIALIZE_NUMPY),
            )
        )
