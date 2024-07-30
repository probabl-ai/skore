from __future__ import annotations

import itertools
import typing

if typing.TYPE_CHECKING:
    from pathlib import PosixPath


class URI:
    def __init__(self, *segments: URI | PosixPath | str):
        """
        *segments: URI | PosixPath | str
            URI, PosixPath or PosixPath-like str.

        Examples
        --------
        >> URI("/", "r", "/", "o", "/", "o", "/", "t")
        URI(r,o,o,t)

        >> URI("/r/o", "/o/t")
        URI(r,o,o,t)

        >> URI("/r/o/o/t")
        URI(r,o,o,t)

        >> URI("r/o/o/t")
        URI(r,o,o,t)

        >> URI("/")
        Traceback (most recent call last):
            ...
        ValueError: You must specify non-empty PosixPath-like str.

        >> URI("")
        Traceback (most recent call last):
            ...
        ValueError: You must specify non-empty PosixPath-like str.
        """
        self.__segments = tuple(
            filter(
                None,
                itertools.chain.from_iterable(
                    str(segment).lower().split("/") for segment in segments
                ),
            )
        )

        if not self.__segments:
            raise ValueError(
                f"Expected a non-empty PosixPath-like string; got {self.__segments}"
            )

    @property
    def segments(self):
        return self.__segments

    @property
    def parent(self) -> URI:
        return URI(*self.__segments[:-1]) if len(self.__segments) > 1 else self

    @property
    def stem(self) -> str:
        return self.__segments[-1]

    def __truediv__(self, segment: URI | PosixPath | str):
        return URI(*self.__segments, segment)

    def __str__(self) -> str:
        return f"/{'/'.join(self.__segments)}"

    def __repr__(self) -> str:
        return f"URI({','.join(self.__segments)})"

    def __hash__(self) -> int:
        return hash(self.__segments)

    def __len__(self) -> int:
        return len(self.__segments)

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, URI) and (self.__segments == other.segments)
