from datetime import datetime, timezone

import pytest
from skore.persistence.item import MediaItem
from skore.persistence.repository import ItemRepository


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
            "key": [
                {
                    "item_class_name": "MediaItem",
                    "item": item_representation,
                }
            ]
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
            "key": [
                {
                    "item_class_name": "MediaItem",
                    "item": {
                        "media_bytes": b"media",
                        "media_encoding": "utf-8",
                        "media_type": "application/octet-stream",
                        "created_at": now,
                        "updated_at": now,
                    },
                }
            ]
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
            "key": [
                {
                    "item_class_name": "MediaItem",
                    "item": {
                        "media_bytes": b"media",
                        "media_encoding": "utf-8",
                        "media_type": "application/octet-stream",
                        "created_at": now,
                        "updated_at": now,
                    },
                },
                {
                    "item_class_name": "MediaItem",
                    "item": {
                        "media_bytes": b"media2",
                        "media_encoding": "utf-8",
                        "media_type": "application/octet-stream",
                        "created_at": now,
                        "updated_at": now2,
                    },
                },
            ]
        }

    def test_get_item_versions(self):
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

        now2 = datetime.now(tz=timezone.utc).isoformat()
        item2 = MediaItem(
            media_bytes=b"media2",
            media_encoding="utf-8",
            media_type="application/octet-stream",
            created_at=now2,
            updated_at=now2,
        )

        repository.put_item("key", item2)

        items = repository.get_item_versions("key")

        assert len(items) == 2
        assert items[0].media_bytes == b"media"
        assert items[0].media_encoding == "utf-8"
        assert items[0].media_type == "application/octet-stream"
        assert items[0].created_at == now
        assert items[0].updated_at == now
        assert items[1].media_bytes == b"media2"
        assert items[1].media_encoding == "utf-8"
        assert items[1].media_type == "application/octet-stream"
        assert items[1].created_at == now
        assert items[1].updated_at == now2
