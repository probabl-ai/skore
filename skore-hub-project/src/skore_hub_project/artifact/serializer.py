"""Function definition of the content ``Serializer``."""

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import cached_property
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from blake3 import blake3 as Blake3
from joblib import dump

from skore_hub_project.protocol import CrossValidationReport, EstimatorReport


class Serializer(ABC):
    """Abstract class to serialize anything on disk."""

    called: bool = False

    @abstractmethod
    def __init__(self, _: Any, /): ...

    def __enter__(self) -> Serializer:  # noqa: D105
        return self

    def __exit__(self, *args: Any) -> None:  # noqa: D105
        self.filepath.unlink(True)

    @abstractmethod
    def __call__(self) -> None:
        """Serialize anything on disk."""

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

        if not self.called:
            raise RuntimeError(
                "You cannot access the checksum of a serializer without explicitly "
                "called it. Please use `serializer()` before."
            )

        # Compute checksum with the appropriate number of threads
        hasher = Blake3(max_threads=(1 if self.size < 1e6 else Blake3.AUTO))
        checksum = hasher.update_mmap(self.filepath).digest()

        return f"blake3-{bytes_to_b64_str(checksum)}"

    @cached_property
    def size(self) -> int:
        """The size of the serialized content, in bytes."""
        if not self.called:
            raise RuntimeError(
                "You cannot access the size of a serializer without explicitly "
                "called it. Please use `serializer()` before."
            )

        return self.filepath.stat().st_size


class TxtSerializer(Serializer):
    """Serialize a str or bytes on disk."""

    def __init__(self, txt: str | bytes, /):
        if isinstance(txt, str):
            txt = txt.encode(encoding="utf-8")

        self.filepath.write_bytes(txt)
        self.called = True

    def __call__(self) -> None:
        """Serialize a str or bytes on disk."""


class ReportSerializer(Serializer):
    """Serialize a report using joblib on disk."""

    def __init__(self, report: CrossValidationReport | EstimatorReport, /):
        self.report = report

    def __call__(self) -> None:
        """Serialize a report using joblib on disk."""
        if self.called:
            return

        with BytesIO() as stream:
            dump(self.report, stream)

            self.filepath.write_bytes(stream.getvalue())
            self.called = True

    @cached_property
    def checksum(self) -> str:
        """The checksum of the serialized report."""
        return f"skore-{self.report.__class__.__name__}-{self.report._hash}"
