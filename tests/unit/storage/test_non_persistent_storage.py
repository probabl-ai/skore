import pytest

from mandr.item import DisplayType, Item, ItemMetadata
from mandr.storage import NonPersistentStorage


class TestNonPersistentStorage:
    @pytest.fixture
    def storage(self, mock_nowstr):
        return NonPersistentStorage(
            content={
                "root": {
                    "key": Item(
                        data="value",
                        metadata=ItemMetadata(
                            display_type=DisplayType.STRING,
                            created_at=mock_nowstr,
                            updated_at=mock_nowstr,
                        ),
                    ),
                }
            }
        )

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

        assert storage.content == {
            "root": { "key": item0 },
            "root1": { "key1": item1 },
            "root2": { "key2": item2 },
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

        assert storage.content == {
            "root": { "key": item3 },
            "root1": { "key1": item1 },
            "root2": { "key2": item2 },
        }

    def test_delitem(self, storage):
        storage.delitem("root", "key")

        assert storage.content == {"root": {}}

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
