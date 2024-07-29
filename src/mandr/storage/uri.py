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
        >> repr(URI("/", "r", "/", "o", "/", "o", "/", "t"))
        URI(r,o,o,t)

        >> repr(URI("/r/o", "/o/t"))
        URI(r,o,o,t)

        >> repr(URI("/r/o/o/t"))
        URI(r,o,o,t)

        >> repr(URI("r/o/o/t"))
        URI(r,o,o,t)

        >> repr(URI("/"))
        Traceback (most recent call last):
            ...
        ValueError: You must specify non-empty PosixPath-like str.

        >> repr(URI(""))
        Traceback (most recent call last):
            ...
        ValueError: You must specify non-empty PosixPath-like str.
        """
        segments = tuple(
            filter(
                None,
                itertools.chain.from_iterable(
                    str(segment).lower().split("/") for segment in segments
                ),
            )
        )

        if not segments:
            raise ValueError("You must specify non-empty PosixPath-like str.")

        self.__segments = segments

    @property
    def segments(self):
        return self.__segments

    def __str__(self) -> str:
        try:
            return self.__str
        except AttributeError:
            self.__str = f"/{'/'.join(self.segments)}"
            return self.__str

    def __repr__(self) -> str:
        try:
            return self.__repr
        except AttributeError:
            self.__repr = f"URI({','.join(self.segments)})"
            return self.__repr

    def __hash__(self) -> int:
        return hash(str(self))

    def __len__(self) -> int:
        return len(self.segments)

    def __eq__(self, other) -> bool:
        return isinstance(other, URI) and (self.segments == other.segments)
