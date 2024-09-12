"""Define a Project."""

import base64
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum, auto
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any, TypedDict

from skore.persistence.core import AbstractStorage
from skore.persistence.disk import DirectoryDoesNotExist, DiskCacheStorage


class ItemType(StrEnum):
    """Type of Item."""

    JSON = auto()
    PANDAS_DATAFRAME = auto()
    NUMPY_ARRAY = auto()
    SKLEARN_BASE_ESTIMATOR = auto()
    MEDIA = auto()


@dataclass(frozen=True)
class Item:
    """A value that is stored in a Project."""

    serialized: str
    raw: Any
    item_type: ItemType
    media_type: str | None = None
    updated_at: datetime | None = None
    created_at: datetime | None = None


def object_to_item(o: Any) -> Item:
    """Transform an object into an Item."""
    try:
        serialized = json.dumps(o)
        return Item(raw=o, item_type=ItemType.JSON, serialized=serialized)
    except TypeError:
        import altair
        import matplotlib.figure
        import numpy
        import pandas
        import PIL.Image
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
                item_type=ItemType.MEDIA,
                serialized=json.dumps(o.to_dict()),
                media_type="application/vnd.vega.v5+json",
            )
        if isinstance(o, matplotlib.figure.Figure):
            with StringIO() as output:
                o.savefig(output, format="svg")
                return Item(
                    raw=o,
                    item_type=ItemType.MEDIA,
                    serialized=output.getvalue(),
                    media_type="image/svg+xml",
                )
        if isinstance(o, PIL.Image.Image):
            with BytesIO() as output:
                o.save(output, format="jpeg")
                return Item(
                    raw=o,
                    item_type=ItemType.MEDIA,
                    serialized=base64.b64encode(output.getvalue()).decode("ascii"),
                    media_type="image/jpeg",
                )

    raise NotImplementedError(f"Type {o.__class__.__name__} is not supported yet.")


class PersistedItem(TypedDict):
    """Item in persisted form."""

    item_type: ItemType
    media_type: str | None
    serialized: str
    created_at: str  # ISO 8601 datetime
    updated_at: str  # ISO 8601 datetime


def persist(item: Item, updated_at: datetime, created_at: datetime) -> PersistedItem:
    """Transform an Item to a PersistedItem."""
    return {
        "item_type": str(item.item_type),
        "media_type": item.media_type,
        "serialized": item.serialized,
        "created_at": created_at.isoformat(),
        "updated_at": updated_at.isoformat(),
    }


def unpersist(persisted_item: PersistedItem) -> Item:
    """Transform a persisted Item back to an object based on `item_type`."""
    match persisted_item["item_type"]:
        case ItemType.JSON:
            raw = json.loads(persisted_item["serialized"])
        case ItemType.PANDAS_DATAFRAME:
            import io

            import pandas

            raw = pandas.read_json(
                io.StringIO(persisted_item["serialized"]), orient="split"
            )
        case ItemType.NUMPY_ARRAY:
            import numpy

            raw = numpy.array(json.loads(persisted_item["serialized"]))
        case ItemType.SKLEARN_BASE_ESTIMATOR:
            import skops.io

            o = json.loads(persisted_item["serialized"])
            unserialized = base64.b64decode(o["skops"])
            raw = skops.io.loads(unserialized)
        case _:
            raw = None

    return Item(
        raw=raw,
        item_type=persisted_item["item_type"],
        serialized=persisted_item["serialized"],
        media_type=persisted_item["media_type"],
        created_at=datetime.fromisoformat(persisted_item["created_at"]),
        updated_at=datetime.fromisoformat(persisted_item["updated_at"]),
    )


class Project:
    """A project is a collection of items that are stored in a storage."""

    def __init__(self, storage: AbstractStorage):
        self.storage = storage

    def put(self, key: str, value: Any):
        """Add a value to the Project."""
        item = object_to_item(value)
        self.put_item(key, item)

    def put_item(self, key: str, item: Item):
        """Add an Item to the Project."""
        internal_key = self.__get_internal_item_key(key)

        updated_at = datetime.now(tz=timezone.utc)
        if item.created_at is None:
            created_at = updated_at

        self.storage[internal_key] = persist(item, updated_at, created_at)

    def get(self, key: str) -> Any:
        """Get the value corresponding to `key` from the Project."""
        return self.get_item(key).raw

    def get_item(self, key: str) -> Item:
        """Add the Item corresponding to `key` from the Project."""
        internal_key = self.__get_internal_item_key(key)
        persisted_item: PersistedItem = self.storage[internal_key]

        return unpersist(persisted_item)

    def list_keys(self) -> list[str]:
        """List all keys in the Project."""
        item_keys = [
            key
            for key in self.storage
            if key.startswith(self.__INTERNAL_ITEM_KEY_PREFIX)
        ]
        unprefixed_keys = map(
            lambda key: key.removeprefix(self.__INTERNAL_ITEM_KEY_PREFIX), item_keys
        )
        return list(unprefixed_keys)

    def delete_item(self, key: str):
        """Delete an item from the Project."""
        internal_key = self.__get_internal_item_key(key)
        del self.storage[internal_key]

    __INTERNAL_ITEM_KEY_PREFIX = "item_"

    def __get_internal_item_key(self, key: str) -> str:
        """Get the internal key for an item."""
        return f"{self.__INTERNAL_ITEM_KEY_PREFIX}{key}"


class ProjectDoesNotExist(Exception):
    """Project does not exist."""


def load(project_name: str | Path) -> Project:
    """Load an existing Project given a project name or path."""
    # Transform a project name to a directory path:
    # - Resolve relative path to current working directory,
    # - Check that the file ends with the ".skore" extension,
    #    - If not provided, it will be automatically appended,
    # - If project name is an absolute path, we keep that path.

    path = Path(project_name).resolve()

    if path.suffix != ".skore":
        path = path.parent / (path.name + ".skore")

    try:
        storage = DiskCacheStorage(directory=Path(path))
        project = Project(storage=storage)
    except DirectoryDoesNotExist as e:
        raise ProjectDoesNotExist(
            f"Project '{path}' does not exist. Did you create it?"
        ) from e

    return project
