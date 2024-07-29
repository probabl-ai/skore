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
                    "#key": Item(
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

    def test_insert(self, monkeypatch, mock_nowstr, storage):
        store = Store("root", storage=storage)
        store.insert("key2", 2, display_type=DisplayType.INTEGER)

        assert storage.content == {
            "root": {
                "#key": Item(
                    data="value",
                    metadata=ItemMetadata(
                        display_type=DisplayType.STRING,
                        created_at=mock_nowstr,
                        updated_at=mock_nowstr,
                    ),
                ),
                "#key2": Item(
                    data=2,
                    metadata=ItemMetadata(
                        display_type=DisplayType.INTEGER,
                        created_at=mock_nowstr,
                        updated_at=mock_nowstr,
                    ),
                ),
            }
        }

    def test_read(self, storage):
        store = Store("root", storage=storage)

        assert store.read("key") == "value"

    def test_update(self, monkeypatch, mock_nowstr, storage):
        store = Store("root", storage=storage)
        store.update("key", 2, display_type=DisplayType.INTEGER)

        assert storage.content == {
            "root": {
                "#key": Item(
                    data=2,
                    metadata=ItemMetadata(
                        display_type=DisplayType.INTEGER,
                        created_at=mock_nowstr,
                        updated_at=mock_nowstr,
                    ),
                ),
            }
        }

    def test_delete(self, monkeypatch, storage):
        store = Store("root", storage=storage)
        store.delete("key")

        assert not storage.content["root"]

    def test_todict(self, storage):
        store = Store("root", storage=storage)

        assert store.todict() == {"key": "value"}

    def test_tolist(self, storage):
        store = Store("root", storage=storage)

        assert store.tolist() == [("key", "value")]
