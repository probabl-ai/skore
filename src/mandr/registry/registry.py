from __future__ import annotations

import operator
from typing import TYPE_CHECKING, Generator

from mandr.storage import URI
from mandr.store import Store

if TYPE_CHECKING:
    from mandr.storage import Storage


def children(store: Store, recursive: bool = False) -> Generator[Store, None, None]:
    compare = operator.ge if recursive else operator.eq

    for uri in store.storage.keys():
        uri = uri.parent

        if uri.segments[: len(store.uri.segments)] == store.uri.segments:
            suffix = uri.segments[len(store.uri.segments) :]

            if compare(len(suffix), 1):
                yield Store(uri, store.storage)


def parent(store: Store, /) -> Store:
    return Store(store.uri.parent, store.storage)


def stores(storage: Storage, /) -> Generator[Store, None, None]:
    for uri in storage.keys():
        yield Store(uri.parent, storage)
