import typing
from datetime import UTC, datetime
from functools import cached_property, singledispatchmethod

from altair.vegalite.v5.schema.core import TopLevelSpec as Altair
from matplotlib.figure import Figure as Matplotlib
from PIL.Image import Image as Pillow

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


class PrimitiveItem:
    def __init__(self, primitive: Primitive, /):
        self.primitive = primitive

    @classmethod
    def factory(cls, primitive: Primitive) -> PrimitiveItem:
        return cls(primitive=primitive)


class PandasDataFrameItem:
    def __init__(self, dataframe_dict: dict, /):
        self.dataframe_dict = dataframe_dict

    @cached_property
    def dataframe(self) -> pandas.DataFrame:
        return pandas.DataFrame.from_dict(self.dataframe_dict, orient="split")

    @property
    def __dict__(self):
        return {"dataframe_dict": self.dataframe_dict}

    @classmethod
    def factory(cls, dataframe: pandas.DataFrame) -> PandasDataFrameItem:
        instance = cls(dataframe.to_dict(orient="split"))

        # add dataframe as cached property
        instance.dataframe = dataframe

        return instance


class NumpyArrayItem:
    def __init__(self, array_list: list, /):
        self.array_list = array_list

    @cached_property
    def array(self) -> numpy.ndarray:
        return numpy.asarray(self.array_list)

    @property
    def __dict__(self):
        return {"array_list": self.array_list}

    @classmethod
    def factory(cls, array: numpy.ndarray) -> NumpyArrayItem:
        instance = cls(array.tolist())

        # add array as cached property
        instance.array = array

        return instance


class SklearnBaseEstimatorItem:
    def __init__(self, estimator_skops, estimator_html_repr):
        self.estimator_skops = estimator_skops
        self.estimator_html_repr = estimator_html_repr

    @cached_property
    def estimator(self) -> sklearn.base.BaseEstimator:
        return sklearn.io.loads(self.estimator_skops)

    @property
    def __dict__(self):
        return {
            "estimator_skops": self.estimator_skops,
            "estimator_html_repr": self.estimator_html_repr,
        }

    @classmethod
    def factory(cls, estimator: sklearn.base.BaseEstimator) -> SklearnBaseEstimatorItem:
        instance = cls(
            skops.io.dumps(estimator),
            sklearn.utils.estimator_html_repr(estimator)
        )

        # add estimator as cached property
        instance.estimator = estimator

        return instance


Media = typing.Union[
    Altair,
    Matplotlib,
    Pillow,
    str,
    bytes,
]


class MediaItem:
    def __init__(
        self,
        media_bytes: bytes,
        media_encoding: str,
        media_type: str,
    ):
        self.media_bytes = media_bytes
        self.media_encoding = media_encoding
        self.media_type = media_type


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
        )

    @factory.register(str)
    @classmethod
    def factory_str(cls, media: str, media_type: str = "text/html") -> MediaItem:
        media_bytes = media.encode("utf-8")

        return cls(
            media_bytes=media_bytes,
            media_encoding="utf-8",
            media_type=media_type,
        )

    @factory.register(Altair)
    @classmethod
    def factory_altair(cls, media: Altair) -> MediaItem:
        media_bytes = media.to_json().encode("utf-8")

        return cls(
            media_bytes=media_bytes,
            media_encoding="utf-8",
            media_type="application/vnd.vega.v5+json",
        )

    @factory.register(Matplotlib)
    @classmethod
    def factory_matplotlib(cls, media: Matplotlib) -> MediaItem:
        with BytesIO() as stream:
            media.savefig(stream, format="svg")
            media_bytes = stream.get_value()

            return cls(
                media_bytes=media_bytes,
                media_encoding="utf-8",
                media_type="image/svg+xml",
            )

    @factory.register(Pillow)
    @classmethod
    def factory_pillow(cls, media: Pillow) -> MediaItem:
        with BytesIO() as stream:
            media.save(stream, format="jpeg")
            media_bytes = stream.get_value()

            return cls(
                media_bytes=media_bytes,
                media_encoding="utf-8",
                media_type="image/jpeg",
            )


# to_json(encoder)
# from_json(decoder)
