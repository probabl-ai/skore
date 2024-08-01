"""URI class used to manipulate PosixPath-like str."""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import PosixPath
    from typing import Any


class URI:
    """URI class used to manipulate PosixPath-like str.

    It is mainly based on `pathlib.PurePosixPath`.
    """

    def __init__(self, *segments: URI | PosixPath | str):
        """Initialize URI with segments.

        Parameters
        ----------
        *segments : URI | PosixPath | str
            URI, PosixPath or PosixPath-like str.

        Notes
        -----
        The slashes ("/") can optionally be used to delimit segments in a string.

        Examples
        --------
        >>> URI("/", "r", "/", "o", "/", "o", "/", "t")
        URI(r,o,o,t)

        >>> URI("/r/o", "/o/t")
        URI(r,o,o,t)

        >>> URI("/r/o/o/t")
        URI(r,o,o,t)

        >>> URI("r/o/o/t")
        URI(r,o,o,t)

        >>> URI("r", "o", "o", "t")
        URI(r,o,o,t)

        >>> URI("/")
        Traceback (most recent call last):
            ...
        ValueError: Expected a non-empty PosixPath-like string; got '('/',)'.

        >>> URI("")
        Traceback (most recent call last):
            ...
        ValueError: Expected a non-empty PosixPath-like string; got '('',)'.
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
                f"Expected a non-empty PosixPath-like string; got '{segments}'."
            )

    @property
    def segments(self):
        """Segments composing the URI."""
        return self.__segments

    @property
    def parent(self) -> URI:
        """The logical parent of the URI."""
        if len(self.__segments) < 2:
            raise ValueError(f"{repr(self)} has not parent.")

        return URI(*self.__segments[:-1])

    @property
    def stem(self) -> str:
        """The final URI segment."""
        return self.__segments[-1]

    def __truediv__(self, segment: URI | PosixPath | str):
        """Compose a new URI by appending segment to the URI."""
        return URI(*self.__segments, segment)

    def __str__(self) -> str:
        """Return str(self)."""
        return f"/{'/'.join(self.__segments)}"

    def __repr__(self) -> str:
        """Return repr(self)."""
        return f"URI({','.join(self.__segments)})"

    def __hash__(self) -> int:
        """Return hash(self)."""
        return hash(self.__segments)

    def __len__(self) -> int:
        """Return the number of segments of the URI."""
        return len(self.__segments)

    def __eq__(self, other: Any) -> bool:
        """Return self == other."""
        return isinstance(other, URI) and (self.__segments == other.segments)

    def __contains__(self, other: URI) -> bool:
        """Return True if self is relative to other, else False."""
        return self.__segments[: len(other.segments)] == other.segments
