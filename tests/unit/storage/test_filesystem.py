import pytest

from mandr.item import DisplayType, Item, ItemMetadata
from mandr.storage import FileSystem
from mandr.storage.filesystem import Key


class TestFileSystem:
    @pytest.fixture
    def storage(self, mock_nowstr, tmp_path):
        filesystem = FileSystem(directory=tmp_path)
        filesystem.cache[Key("root", "key")] = Item(
            data="value",
            metadata=ItemMetadata(
                display_type=DisplayType.STRING,
                created_at=mock_nowstr,
                updated_at=mock_nowstr,
            ),
        )

        return filesystem

    def test_iter(self, storage):
        assert list(storage) == ["root"]

    def test_contains(self, storage):
        assert storage.contains("root", "key")
        assert not storage.contains("root", "key1")
        assert not storage.contains("root1", "key")

    def test_getitem(self, storage, mock_nowstr):
        assert storage.getitem("root", "key") == Item(
            data="value",
            metadata=ItemMetadata(
                display_type=DisplayType.STRING,
                created_at=mock_nowstr,
                updated_at=mock_nowstr,
            ),
        )

        with pytest.raises(KeyError):
            storage.getitem("root", "key1")
        with pytest.raises(KeyError):
            storage.getitem("root1", "key")

    def test_setitem(self, storage, mock_nowstr):
        item0 = Item(
            data="value",
            metadata=ItemMetadata(
                display_type=DisplayType.STRING,
                created_at=mock_nowstr,
                updated_at=mock_nowstr,
            ),
        )
        item1 = Item(
            data=1,
            metadata=ItemMetadata(
                display_type=DisplayType.INTEGER,
                created_at=mock_nowstr,
                updated_at=mock_nowstr,
            ),
        )
        item2 = Item(
            data=2,
            metadata=ItemMetadata(
                display_type=DisplayType.INTEGER,
                created_at=mock_nowstr,
                updated_at=mock_nowstr,
            ),
        )

        storage.setitem("root1", "key1", item1)
        storage.setitem("root2", "key2", item2)

        assert {key: storage.cache[key] for key in storage.cache.iterkeys()} == {
            Key("root", "key"): item0,
            Key("root1", "key1"): item1,
            Key("root2", "key2"): item2,
        }

        item3 = Item(
            data=3,
            metadata=ItemMetadata(
                display_type=DisplayType.INTEGER,
                created_at=mock_nowstr,
                updated_at=mock_nowstr,
            ),
        )

        storage.setitem("root", "key", item3)

        assert {key: storage.cache[key] for key in storage.cache.iterkeys()} == {
            Key("root", "key"): item3,
            Key("root1", "key1"): item1,
            Key("root2", "key2"): item2,
        }

    def test_delitem(self, storage):
        storage.delitem("root", "key")

        assert list(storage.cache.iterkeys()) == []

    def test_keys(self, storage):
        assert list(storage.keys("root")) == ["key"]

    def test_items(self, storage, mock_nowstr):
        assert list(storage.items("root")) == [
            (
                "key",
                Item(
                    data="value",
                    metadata=ItemMetadata(
                        display_type=DisplayType.STRING,
                        created_at=mock_nowstr,
                        updated_at=mock_nowstr,
                    ),
                ),
            )
        ]
