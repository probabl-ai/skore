import pytest
from mandr import registry
from mandr.storage import NonPersistentStorage, URI
from mandr.store import Store


class TestRegistry:
    @pytest.fixture
    def storage(self):
        return NonPersistentStorage(
            content={
                URI("/r/key"): True,
                URI("/r/o/key"): True,
                URI("/r/o/o/key"): True,
                URI("/r/o/o/t/key"): True,
            }
        )

    def test_children(self, storage):
        ro = Store(URI("/r/o"), storage)
        roo = Store(URI("/r/o/o"), storage)
        root = Store(URI("/r/o/o/t"), storage)

        assert list(registry.children(ro, recursive=False)) == [roo]
        assert list(registry.children(roo, recursive=False)) == [root]
        assert list(registry.children(root, recursive=False)) == []

        assert list(registry.children(ro, recursive=True)) == [roo, root]
        assert list(registry.children(roo, recursive=True)) == [root]
        assert list(registry.children(root, recursive=True)) == []

    def test_parent(self, storage):
        r = Store(URI("/r"), storage)
        ro = Store(URI("/r/o"), storage)

        assert registry.parent(r) == r
        assert registry.parent(ro) == r

    def test_stores(self, storage):
        assert list(registry.stores(storage)) == [
            Store(URI("/r"), storage),
            Store(URI("/r/o"), storage),
            Store(URI("/r/o/o"), storage),
            Store(URI("/r/o/o/t"), storage),
        ]
