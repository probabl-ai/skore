from datetime import UTC, datetime

import pytest
from skore.item import ItemRepository, MediaItem


class TestItemRepository:
    def test_get_item(self):
        now = datetime.now(tz=UTC).isoformat()
        item = MediaItem(
            media_bytes=b"media",
            media_encoding="utf-8",
            media_type="application/octet-stream",
            created_at=now,
            updated_at=now,
        )

        storage = {
            "key": {
                "item_class_name": "MediaItem",
                "item": vars(item),
            }
        }

        repository = ItemRepository(storage)

        assert vars(repository.get_item("key")) == vars(item)

        with pytest.raises(KeyError):
            repository.get_item("key2")

    def test_put_item(self):
        now = datetime.now(tz=UTC).isoformat()
        item = MediaItem(
            media_bytes=b"media",
            media_encoding="utf-8",
            media_type="application/octet-stream",
            created_at=now,
            updated_at=now,
        )

        storage = {}

        repository = ItemRepository(storage)

        repository.put_item("key", item)

        assert storage == {
            "key": {
                "item_class_name": "MediaItem",
                "item": vars(item),
            }
        }

        now2 = datetime.now(tz=UTC).isoformat()
        item2 = MediaItem(
            media_bytes=b"media2",
            media_encoding="utf-8",
            media_type="application/octet-stream",
            created_at=now2,
            updated_at=now2,
        )

        repository.put_item("key", item2)

        assert storage == {
            "key": {
                "item_class_name": "MediaItem",
                "item": {
                    "media_bytes": b"media2",
                    "media_encoding": "utf-8",
                    "media_type": "application/octet-stream",
                    "created_at": now,
                    "updated_at": now2,
                },
            }
        }
