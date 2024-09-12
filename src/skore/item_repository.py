import typing
from datetime import UTC, datetime
from functools import singledispatchmethod

Item = PrimitiveItem | SklearnBaseEstimatorItem
Metadata = dict[str, str]
Primitive = typing.Union[
    str,
    int,
    float,
    bytes,
    list[Primitive],
    tuple[Primitive],
    dict[str | int | float, Primitive],
]


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
        self.storage.setitem(
            key,
            {
                "item_class_name": item.__class__.__name__,
                "item": vars(item),
            },
        )


class PrimitiveItem:
    def __init__(self, primitive: Primitive, metadata: Metadata):
        self.primitive = primitive
        self.metadata = metadata

    @classmethod
    def factory(cls, primitive: Primitive) -> PrimitiveItem:
        now = datetime.now(tz=UTC).isoformat()
        metadata = {
            "created_at": now,
            "updated_at": now,
        }

        return cls(primitive=primitive, metadata=metadata)


class PandasDataFrameItem:
    def __init__(self, dataframe_dict: dict, metadata: Metadata):
        self.dataframe_dict = dataframe_dict
        self.metadata = metadata

    @classmethod
    def factory(cls, dataframe: pandas.DataFrame) -> PandasDataFrameItem:
        now = datetime.now(tz=UTC).isoformat()
        dataframe_dict = dataframe.to_dict(orient="split")
        metadata = {
            "created_at": now,
            "updated_at": now,
        }

        return cls(dataframe_dict=dataframe_dict, metadata=metadata)


class NumpyArrayItem:
    def __init__(self, array_list: list, metadata: Metadata):
        self.array_list = array_list
        self.metadata = metadata

    @classmethod
    def factory(cls, array: numpy.ndarray) -> NumpyArrayItem:
        now = datetime.now(tz=UTC).isoformat()
        array_list = array.tolist()
        metadata = {
            "created_at": now,
            "updated_at": now,
        }

        return cls(array_list=array_list, metadata=metadata)


class SklearnBaseEstimatorItem:
    def __init__(self, estimator_skops, estimator_html_repr, metadata: Metadata):
        self.estimator_skops = estimator_skops
        self.estimator_html_repr = estimator_html_repr
        self.metadata = metadata

    @classmethod
    def factory(cls, estimator: sklearn.base.BaseEstimator) -> SklearnBaseEstimatorItem:
        now = datetime.now(tz=UTC).isoformat()
        estimator_skops = skops.io.dumps(estimator)
        estimator_html_repr = sklearn.utils.estimator_html_repr(estimator)
        now = datetime.now(tz=UTC).isoformat()
        metadata = {
            "created_at": now,
            "updated_at": now,
        }

        return cls(
            estimator_skops=estimator_skops,
            estimator_html_repr=estimator_html_repr,
            metadata=metadata,
        )


Media = typing.Union[
    altair.vegalite.v5.schema.core.TopLevelSpec,
    matplotlib.figure.Figure,
    PIL.Image.Image,
    str,
    bytes,
]


class MediaItem:
    def __init__(
        self,
        media_bytes: bytes,
        media_encoding: str,
        media_type: str,
        metadata: Metadata,
    ):
        self.media_bytes = media_bytes
        self.media_encoding = media_encoding
        self.media_type = media_type
        self.metadata = metadata

    @staticmethod
    def __metadata():
        now = datetime.now(tz=UTC).isoformat()
        metadata = {
            "created_at": now,
            "updated_at": now,
        }

        return metadata

    @singledispatchmethod
    @classmethod
    def factory(cls, media):
        raise NotImplementedError

    @factory.register(bytes)
    @classmethod
    def factory_bytes(
        cls,
        media: bytes,
        media_encoding: str = "utf-8",
        media_type: str = "application/octet-stream",
    ) -> MediaItem:
        return cls(
            media_bytes=media,
            media_encoding=media_type,
            media_type=media_type,
            metadata=MediaItem.__metadata(),
        )

    @factory.register(str)
    @classmethod
    def factory_str(cls, media: str, media_type: str = "text/html") -> MediaItem:
        media_bytes = media.encode("utf-8")

        return cls(
            media_bytes=media_bytes,
            media_encoding="utf-8",
            media_type=media_type,
            metadata=MediaItem.__metadata(),
        )

    @factory.register(altair.vegalite.v5.schema.core.TopLevelSpec)
    @classmethod
    def factory_altair(
        cls, media: altair.vegalite.v5.schema.core.TopLevelSpec
    ) -> MediaItem:
        media_bytes = media.to_json().encode("utf-8")

        return cls(
            media_bytes=media_bytes,
            media_encoding="utf-8",
            media_type="application/vnd.vega.v5+json",
            metadata=MediaItem.__metadata(),
        )

    @factory.register(matplotlib.figure.Figure)
    @classmethod
    def factory_matplotlib(cls, media: matplotlib.figure.Figure) -> MediaItem:
        with BytesIO() as stream:
            media.savefig(stream, format="svg")
            media_bytes = stream.get_value()

            return cls(
                media_bytes=media_bytes,
                media_encoding="utf-8",
                media_type="image/svg+xml",
                metadata=MediaItem.__metadata(),
            )

    @factory.register(PIL.Image.Image)
    @classmethod
    def factory_pil(cls, media: PIL.Image.Image) -> MediaItem:
        with BytesIO() as stream:
            media.save(stream, format="jpeg")
            media_bytes = stream.get_value()

            return cls(
                media_bytes=media_bytes,
                media_encoding="utf-8",
                media_type="image/jpeg",
                metadata=MediaItem.__metadata(),
            )


# to_json(encoder)
# from_json(decoder)
