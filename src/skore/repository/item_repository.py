from datetime import UTC, datetime


class ItemRepository:
    ITEM_CLASS_NAME_TO_ITEM_CLASS = {
        "PrimitiveItem": PrimitiveItem,
        "PandasDataFrameItem": PandasDataFrameItem,
        "SklearnBaseEstimatorItem": SklearnBaseEstimatorItem,
    }

    def __init__(self, storage: Storage):
        self.storage = storage

    def get_item(self, key) -> Item:
        value = self.storage.getitem(key)
        item_class_name = value["item_class_name"]
        item_class = ItemRepository.ITEM_CLASS_NAME_TO_ITEM_CLASS[item_class_name]
        item = value["item"]

        return item_class(**item)

    def put_item(self, key, item: Item) -> None:
        now = datetime.now(tz=UTC).isoformat()

        self.storage.setitem(
            key,
            {
                "item_class_name": item.__class__.__name__,
                "item": vars(item),
                "created_at": now,
                "updated_at": now,
            },
        )


# to_json(encoder)
# from_json(decoder)
