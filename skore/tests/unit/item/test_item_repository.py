from datetime import datetime, timezone

import pytest
from skore.item import ItemRepository, MediaItem


class TestItemRepository:
    def test_get_item(self):
        now = datetime.now(tz=timezone.utc).isoformat()
        item_representation = dict(
            media_bytes=b"media",
            media_encoding="utf-8",
            media_type="application/octet-stream",
            created_at=now,
            updated_at=now,
        )

        storage = {
            "key": {
                "item_class_name": "MediaItem",
                "item": item_representation,
            }
        }

        repository = ItemRepository(storage)
        item = repository.get_item("key")

        assert item.media_bytes == item_representation["media_bytes"]
        assert item.media_encoding == item_representation["media_encoding"]
        assert item.media_type == item_representation["media_type"]
        assert item.created_at == item_representation["created_at"]
        assert item.updated_at == item_representation["updated_at"]

        with pytest.raises(KeyError):
            repository.get_item("key2")

    def test_put_item(self):
        now = datetime.now(tz=timezone.utc).isoformat()
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
                "item": {
                    "media_bytes": b"media",
                    "media_encoding": "utf-8",
                    "media_type": "application/octet-stream",
                    "created_at": now,
                    "updated_at": now,
                },
            }
        }

        now2 = datetime.now(tz=timezone.utc).isoformat()
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
