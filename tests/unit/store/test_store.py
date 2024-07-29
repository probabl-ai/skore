import pytest
from mandr.item import DisplayType, Item, ItemMetadata
from mandr.storage import NonPersistentStorage
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
                "root": {
                    "key": Item(
                        data="value",
                        metadata=ItemMetadata(
                            display_type=DisplayType.STRING,
                            created_at=mock_nowstr,
                            updated_at=mock_nowstr,
                        ),
                    )
                }
            }
        )

    @pytest.fixture
    def store(self, storage):
        return Store("root", storage=storage)

    def test_eq(self, store):
        assert store == store

    def test_iter(self, store):
        assert list(store) == [("key", "value")]

    def test_insert(self, monkeypatch, mock_nowstr, storage, store):
        store.insert("key2", 2, display_type=DisplayType.INTEGER)

        assert storage.content == {
            "root": {
                "key": Item(
                    data="value",
                    metadata=ItemMetadata(
                        display_type=DisplayType.STRING,
                        created_at=mock_nowstr,
                        updated_at=mock_nowstr,
                    ),
                ),
                "key2": Item(
                    data=2,
                    metadata=ItemMetadata(
                        display_type=DisplayType.INTEGER,
                        created_at=mock_nowstr,
                        updated_at=mock_nowstr,
                    ),
                ),
            }
        }

    def test_read(self, store):
        assert store.read("key") == "value"

    def test_update(self, monkeypatch, mock_nowstr, storage, store):
        store.update("key", 2, display_type=DisplayType.INTEGER)

        assert storage.content == {
            "root": {
                "key": Item(
                    data=2,
                    metadata=ItemMetadata(
                        display_type=DisplayType.INTEGER,
                        created_at=mock_nowstr,
                        updated_at=mock_nowstr,
                    ),
                ),
            }
        }

    def test_delete(self, monkeypatch, storage, store):
        store.delete("key")

        assert not storage.content["root"]
