import pytest
from skore import registry
from skore.storage import URI, NonPersistentStorage
from skore.store import Store


class TestRegistry:
    @pytest.fixture
    def storage(self):
        return NonPersistentStorage(
            content={
                URI("/r/key"): True,
                URI("/r/key2"): True,
                URI("/r/o/key"): True,
                URI("/r/o/key2"): True,
                URI("/r/o/o/key"): True,
                URI("/r/o/o/key2"): True,
                URI("/r/o/o/t/key"): True,
                URI("/r/o/o/t/key2"): True,
            }
        )

    def test_children(self, storage):
        ro = Store(URI("/r/o"), storage)
        roo = Store(URI("/r/o/o"), storage)
        root = Store(URI("/r/o/o/t"), storage)

        assert list(registry.children(ro)) == [roo, root]
        assert list(registry.children(roo)) == [root]
        assert list(registry.children(root)) == []

    def test_parent(self, storage):
        r = Store(URI("/r"), storage)
        ro = Store(URI("/r/o"), storage)

        assert registry.parent(ro) == r

    def test_stores(self, storage):
        assert list(registry.stores(storage)) == [
            Store(URI("/r"), storage),
            Store(URI("/r/o"), storage),
            Store(URI("/r/o/o"), storage),
            Store(URI("/r/o/o/t"), storage),
        ]

    def test_find_store_by_uri(self, storage):
        assert registry.find_store_by_uri(URI("/r/o/o"), storage) == Store(
            URI("/r/o/o"), storage
        )
        assert registry.find_store_by_uri(URI("/hello"), storage) is None
