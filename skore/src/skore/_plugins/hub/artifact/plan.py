"""ArtifactPlan: a prepared-but-not-yet-uploaded artifact."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from math import ceil
from pathlib import Path


@dataclass(frozen=True)
class ArtifactPlan:
    """One artifact ready to upload.

    Attributes
    ----------
    checksum : str
        BLAKE2b checksum of the content, prefixed ``blake2b-``.

    size : int
        Size of the content, in bytes.

    content_type : str
        MIME type of the content.

    payload : bytes | Path
        The content itself (small, in-memory) or a path to a file containing
        the content (large, on-disk). Disk-backed payloads are read lazily,
        chunk by chunk, by ``iter_chunks``.
    """

    checksum: str
    size: int
    content_type: str
    payload: bytes | Path

    def __post_init__(self) -> None:
        if self.size < 0:
            raise ValueError(f"size must be >= 0, got {self.size}")

    def chunk_count_for(self, chunk_size: int) -> int:
        """Compute the number of chunks needed for a given chunk size."""
        if self.size <= chunk_size:
            return 1
        return ceil(self.size / chunk_size)

    def iter_chunks(self, chunk_size: int) -> Iterator[bytes]:
        """Yield chunks of content in order."""
        if isinstance(self.payload, bytes):
            yield self.payload
            return

        with self.payload.open("rb") as f:
            while chunk := f.read(chunk_size):
                yield chunk
