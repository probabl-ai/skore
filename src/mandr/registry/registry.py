"""Registry used to supervise stores."""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING, Generator

from mandr.store import Store

if TYPE_CHECKING:
    from mandr.storage import Storage


def children(store: Store, /, recursive: bool = False) -> Generator[Store, None, None]:
    """Yield `store` children."""
    compare = operator.ge if recursive else operator.eq

    for uri in store.storage:
        # Remove from `uri` the `keyname`.
        uri = uri.parent

        # Check if `uri` is relative to `store.uri`.
        if uri.segments[: len(store.uri.segments)] == store.uri.segments:
            # Remove from `uri` the relative part to the `store.uri`
            relative_segments = uri.segments[len(store.uri.segments) :]

            # If there is remaining segments, its a child.
            if compare(len(relative_segments), 1):
                yield Store(uri, store.storage)


def parent(store: Store, /) -> Store:
    """Return `store` parent."""
    return Store(store.uri.parent, store.storage)


def stores(storage: Storage, /) -> Generator[Store, None, None]:
    """Yield stores saved in `storage`."""
    for uri in storage:
        yield Store(uri.parent, storage)
