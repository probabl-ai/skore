"""Define a Project."""

import base64
import json
from dataclasses import dataclass
from enum import StrEnum, auto
from io import StringIO
from pathlib import Path
from typing import Any, List

from skore.storage import Storage
from skore.storage.filesystem import FileSystem


class ItemType(StrEnum):
    """Type of Item."""

    JSON = auto()
    PANDAS_DATAFRAME = auto()
    NUMPY_ARRAY = auto()
    SKLEARN_BASE_ESTIMATOR = auto()
    ALTAIR_CHART = auto()
    MEDIA = auto()


# FIXME: attensssion parlezen à thomas experience déguelasse
# s.put_item(
#     "HTML",
#     Item(
#         serialized=markup,
#         raw=markup,
#         media_type="text/html",
#         item_type=ItemType.MEDIA,
#     ),
# )
@dataclass(frozen=True)
class Item:
    """A value that is stored in a Project."""

    serialized: str
    raw: Any
    item_type: ItemType
    media_type: str | None = None


def serialize(o: Any) -> Item:
    """Transform an object into an Item."""
    try:
        serialized = json.dumps(o)
        return Item(raw=o, item_type=ItemType.JSON, serialized=serialized)
    except TypeError:
        import altair
        import matplotlib.figure
        import numpy
        import pandas
        import sklearn
        import skops.io

        if isinstance(o, pandas.DataFrame):
            return Item(
                raw=o,
                item_type=ItemType.PANDAS_DATAFRAME,
                serialized=o.to_json(orient="split"),
            )
        if isinstance(o, numpy.ndarray):
            return Item(
                raw=o,
                item_type=ItemType.NUMPY_ARRAY,
                serialized=json.dumps(o.tolist()),
            )
        if isinstance(o, sklearn.base.BaseEstimator):
            sk_dump = skops.io.dumps(o)
            serialized_model = base64.b64encode(sk_dump).decode("ascii")
            html_representation = sklearn.utils.estimator_html_repr(o)
            return Item(
                raw=o,
                item_type=ItemType.SKLEARN_BASE_ESTIMATOR,
                serialized=json.dumps(
                    {"skops": serialized_model, "html": html_representation}
                ),
                media_type="text/html",
            )
        if isinstance(o, altair.vegalite.v5.schema.core.TopLevelSpec):
            return Item(
                raw=o,
                item_type=ItemType.ALTAIR_CHART,
                serialized=json.dumps(o.to_dict()),
                media_type=None,
            )
        if isinstance(o, matplotlib.figure.Figure):
            output = StringIO()
            o.savefig(output, format="svg")
            return Item(
                raw=o,
                item_type=ItemType.MEDIA,
                serialized=output.getvalue(),
                media_type="image/svg+xml",
            )

    raise NotImplementedError(f"Type {o.__class__.__name__} is not supported yet.")


def deserialize(
    item_type: ItemType,
    media_type: str | None,
    serialized: str,
) -> Item:
    """Transform a serialized Item back to an object based on `item_type`."""
    match item_type:
        case ItemType.JSON:
            raw = json.loads(serialized)
        case ItemType.PANDAS_DATAFRAME:
            import pandas

            raw = pandas.read_json(serialized, orient="split")
        case ItemType.NUMPY_ARRAY:
            import numpy

            raw = numpy.array(json.loads(serialized))
        case ItemType.SKLEARN_BASE_ESTIMATOR:
            import skops.io

            o = json.loads(serialized)
            unserialized = base64.b64decode(o["skops"])
            raw = skops.io.loads(unserialized)
        case ItemType.ALTAIR_CHART:
            import altair

            raw = altair.Chart.from_dict(json.loads(serialized))
        case _:
            raw = None

    return Item(
        raw=raw,
        item_type=item_type,
        serialized=serialized,
        media_type=media_type,
    )


class Project:
    """A project is a collection of items that are stored in a storage."""

    def __init__(self, storage: Storage):
        self.storage = storage

    def put(self, key: str, value: Any):
        """Add a value to the Project."""
        i = serialize(value)
        self.put_item(key, i)

    def put_item(self, key: str, item: Item):
        """Add an Item to the Project."""
        self.storage.setitem(
            key,
            {
                "item_type": str(item.item_type),
                "media_type": item.media_type,
                "serialized": item.serialized,
            },
        )

    def get(self, key: str) -> Any:
        """Get the value corresponding to `key` from the Project."""
        return self.get_item(key).raw

    def get_item(self, key: str) -> Item:
        """Add the Item corresponding to `key` from the Project."""
        item = self.storage.getitem(key)

        return deserialize(**item)

    def list_keys(self) -> List[str]:
        """List all keys in the Project."""
        return list(self.storage.keys())

    def delete_item(self, key: str):
        """Delete an item from the Project."""
        self.storage.delitem(key)


def load(directory: str | Path) -> Project:
    """Load a Project given a project name or path."""
    storage = FileSystem(directory=Path(directory))
    project = Project(storage=storage)

    return project
