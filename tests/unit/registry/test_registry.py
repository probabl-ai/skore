import pytest
from mandr.registry import Registry
from mandr.storage import URI, NonPersistentStorage
from mandr.store import Store


class TestRegistry:
    @pytest.fixture
    def storage(self):
        return NonPersistentStorage(
            content={
                URI("/r"): {},
                URI("/r/o"): {},
                URI("/r/o/o"): {},
                URI("/r/o/o/t"): {},
            }
        )

    def test_children(self, storage):
        parent = Store(URI("/r/o"), storage)
        child = Store(URI("/r/o/o"), storage)
        grandchild = Store(URI("/r/o/o/t"), storage)

        assert list(Registry.children(parent)) == [child]
        assert list(Registry.children(child)) == [grandchild]
        assert list(Registry.children(grandchild)) == []

        assert list(Registry.children(parent, True)) == [child, grandchild]
        assert list(Registry.children(child, True)) == [grandchild]
        assert list(Registry.children(grandchild, True)) == []

    def test_children_from_uri(self, storage):
        r = URI("/r")
        ro = URI("/r/o")
        roo = URI("/r/o/o")
        root = URI("/r/o/o/t")

        assert list(Registry.children_from_uri(r, storage, False)) == [ro]
        assert list(Registry.children_from_uri(ro, storage, False)) == [roo]
        assert list(Registry.children_from_uri(roo, storage, False)) == [root]
        assert list(Registry.children_from_uri(root, storage, False)) == []

        assert list(Registry.children_from_uri(r, storage, True)) == [ro, roo, root]
        assert list(Registry.children_from_uri(ro, storage, True)) == [roo, root]
        assert list(Registry.children_from_uri(roo, storage, True)) == [root]
        assert list(Registry.children_from_uri(root, storage, True)) == []

    def test_parent(self, storage):
        parent = Store(URI("/r"), storage)
        child = Store(URI("/r/o"), storage)

        assert Registry.parent(parent) == parent
        assert Registry.parent(child) == parent

    def test_parent_from_uri(self, storage):
        r = URI("/r")
        ro = URI("/r/o")
        roo = URI("/r/o/o")
        root = URI("/r/o/o/t")

        assert Registry.parent_from_uri(r) == r
        assert Registry.parent_from_uri(ro) == r
        assert Registry.parent_from_uri(roo) == ro
        assert Registry.parent_from_uri(root) == roo
