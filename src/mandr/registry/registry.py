from __future__ import annotations

import operator
from functools import singledispatch
from typing import TYPE_CHECKING, Generator

from mandr.storage import URI
from mandr.store import Store

if TYPE_CHECKING:
    from mandr.storage import Storage

    StoreGenerator = Generator[Store, None, None]
    URIGenerator = Generator[URI, None, None]


class Registry:
    @singledispatch
    @staticmethod
    def children(store: Store, recursive: bool = False) -> StoreGenerator:
        for uri in Registry.children_from_uri(store.uri, store.storage, recursive):
            yield Store(uri, store.storage)

    @children.register(URI)
    @staticmethod
    def children_from_uri(
        root: URI,
        storage: Storage,
        recursive: bool = False,
    ) -> URIGenerator:
        compare = operator.ge if recursive else operator.eq

        for uri in storage:
            if uri.segments[: len(root.segments)] == root.segments:
                sufix = uri.segments[len(root.segments) :]

                if compare(len(sufix), 1):
                    yield uri

    @singledispatch
    @staticmethod
    def parent(store: Store) -> Store:
        return Store(Registry.parent(store.uri), store.storage)

    @parent.register(URI)
    @staticmethod
    def parent_from_uri(root: URI) -> URI:
        return URI(*root.segments[:-1]) if len(root.segments) > 1 else root
