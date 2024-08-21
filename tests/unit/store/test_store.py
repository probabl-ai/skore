import pytest
from mandr.item import DisplayType, Item, ItemMetadata
from mandr.storage import URI, NonPersistentStorage
from mandr.store import Store


class TestStore:
    @pytest.fixture
    def storage(self, monkeypatch, mock_now, mock_nowstr):
        class MockDatetime:
            @staticmethod
            def now(*args, **kwargs):
                return mock_now

        monkeypatch.setattr("mandr.store.store.datetime", MockDatetime)

        return NonPersistentStorage(
            content={
                URI("root1/key1"): Item(
                    data="value",
                    metadata=ItemMetadata(
                        display_type=DisplayType.STRING,
                        created_at=mock_nowstr,
                        updated_at=mock_nowstr,
                    ),
                ),
                URI("root2/key3"): Item(
                    data="value",
                    metadata=ItemMetadata(
                        display_type=DisplayType.STRING,
                        created_at=mock_nowstr,
                        updated_at=mock_nowstr,
                    ),
                ),
            }
        )

    @pytest.fixture
    def store(self, storage):
        return Store("root1", storage=storage)

    def test_eq(self, store):
        assert store == store

    def test_insert(self, monkeypatch, mock_nowstr, storage, store):
        store.insert("key2", 2)
        store.insert("key3", 3, display_type="integer")
        store.insert("key4", 4, display_type=DisplayType.INTEGER)

        assert storage.content == {
            URI("root1/key1"): Item(
                data="value",
                metadata=ItemMetadata(
                    display_type=DisplayType.STRING,
                    created_at=mock_nowstr,
                    updated_at=mock_nowstr,
                ),
            ),
            URI("root1/key2"): Item(
                data=2,
                metadata=ItemMetadata(
                    display_type=DisplayType.INTEGER,
                    created_at=mock_nowstr,
                    updated_at=mock_nowstr,
                ),
            ),
            URI("root1/key3"): Item(
                data=3,
                metadata=ItemMetadata(
                    display_type=DisplayType.INTEGER,
                    created_at=mock_nowstr,
                    updated_at=mock_nowstr,
                ),
            ),
            URI("root1/key4"): Item(
                data=4,
                metadata=ItemMetadata(
                    display_type=DisplayType.INTEGER,
                    created_at=mock_nowstr,
                    updated_at=mock_nowstr,
                ),
            ),
            URI("root2/key3"): Item(
                data="value",
                metadata=ItemMetadata(
                    display_type=DisplayType.STRING,
                    created_at=mock_nowstr,
                    updated_at=mock_nowstr,
                ),
            ),
        }

        with pytest.raises(KeyError):
            store.insert("key2", 2, display_type=DisplayType.INTEGER)

    def test_read(self, store):
        assert store.read("key1") == "value"

        with pytest.raises(KeyError):
            store.read("key2")

    def test_update(self, monkeypatch, mock_nowstr, storage, store):
        store.update("key1", 2)
        store.update("key1", 3, display_type="integer")
        store.update("key1", 4, display_type=DisplayType.INTEGER)

        assert storage.content == {
            URI("root1/key1"): Item(
                data=4,
                metadata=ItemMetadata(
                    display_type=DisplayType.INTEGER,
                    created_at=mock_nowstr,
                    updated_at=mock_nowstr,
                ),
            ),
            URI("root2/key3"): Item(
                data="value",
                metadata=ItemMetadata(
                    display_type=DisplayType.STRING,
                    created_at=mock_nowstr,
                    updated_at=mock_nowstr,
                ),
            ),
        }

        with pytest.raises(KeyError):
            store.update("key2", 2, display_type=DisplayType.INTEGER)

    def test_delete(self, monkeypatch, mock_nowstr, storage, store):
        store.delete("key1")

        assert storage.content == {
            URI("root2/key3"): Item(
                data="value",
                metadata=ItemMetadata(
                    display_type=DisplayType.STRING,
                    created_at=mock_nowstr,
                    updated_at=mock_nowstr,
                ),
            ),
        }

        with pytest.raises(KeyError):
            store.delete("key2")

    def test_iter(self, store):
        assert list(store) == ["key1"]

    def test_keys(self, store):
        assert list(store.keys()) == ["key1"]

    def test_items(self, store, mock_nowstr):
        assert list(store.items()) == [("key1", "value")]
        assert list(store.items(metadata=True)) == [
            (
                "key1",
                "value",
                {
                    "display_type": DisplayType.STRING,
                    "created_at": mock_nowstr,
                    "updated_at": mock_nowstr,
                },
            )
        ]
