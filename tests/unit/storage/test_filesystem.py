import pytest
from skore.item import DisplayType, Item, ItemMetadata
from skore.storage import URI, FileSystem


class TestFileSystem:
    @pytest.fixture
    def storage(self, mock_nowstr, tmp_path):
        filesystem = FileSystem(directory=tmp_path)
        filesystem.cache[URI("root/key")] = Item(
            data="value",
            metadata=ItemMetadata(
                display_type=DisplayType.STRING,
                created_at=mock_nowstr,
                updated_at=mock_nowstr,
            ),
        )

        return filesystem

    def test_contains(self, storage):
        assert URI("root/key") in storage
        assert URI("root/key1") not in storage
        assert URI("root1/key") not in storage

    def test_iter(self, storage):
        assert list(storage) == [URI("root/key")]

    def test_getitem(self, storage, mock_nowstr):
        assert storage.getitem(URI("root/key")) == Item(
            data="value",
            metadata=ItemMetadata(
                display_type=DisplayType.STRING,
                created_at=mock_nowstr,
                updated_at=mock_nowstr,
            ),
        )

        with pytest.raises(KeyError):
            storage.getitem(URI("root/key1"))
        with pytest.raises(KeyError):
            storage.getitem(URI("root1/key"))

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

        storage.setitem(URI("root1/key1"), item1)
        storage.setitem(URI("root2/key2"), item2)

        assert {key: storage.cache[key] for key in storage.cache.iterkeys()} == {
            URI("root/key"): item0,
            URI("root1/key1"): item1,
            URI("root2/key2"): item2,
        }

        item3 = Item(
            data=3,
            metadata=ItemMetadata(
                display_type=DisplayType.INTEGER,
                created_at=mock_nowstr,
                updated_at=mock_nowstr,
            ),
        )

        storage.setitem(URI("root/key"), item3)

        assert {key: storage.cache[key] for key in storage.cache.iterkeys()} == {
            URI("root/key"): item3,
            URI("root1/key1"): item1,
            URI("root2/key2"): item2,
        }

    def test_delitem(self, storage):
        storage.delitem(URI("root/key"))

        assert list(storage.cache.iterkeys()) == []

    def test_keys(self, storage):
        assert list(storage.keys()) == [URI("root/key")]

    def test_items(self, storage, mock_nowstr):
        assert list(storage.items()) == [
            (
                URI("root/key"),
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
