from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING

from .item import Item, ItemTypeError

if TYPE_CHECKING:
    import polars


class PolarsDataFrameItem(Item):
    def __init__(self, dataframe_json_str: str):
        self.dataframe_json_str = dataframe_json_str

    @cached_property
    def __raw__(self) -> polars.DataFrame:
        """
        The polars DataFrame from the persistence.

        Its content can differ from the original dataframe because it has been
        serialized using polars' `to_json` function and not pickled, in order to be
        environment-independent.
        """
        import io

        import polars

        with io.StringIO(self.dataframe_json_str) as df_stream:
            return polars.read_json(df_stream)

    @property
    def __representation__(self) -> dict:
        return {
            "representation": {
                "media_type": "application/vnd.dataframe",
                "value": self.__raw__.to_pandas().fillna("NaN").to_dict(orient="tight"),
            }
        }

    @classmethod
    def factory(cls, dataframe: polars.DataFrame, /) -> PolarsDataFrameItem:
        import polars

        if not isinstance(dataframe, polars.DataFrame):
            raise ItemTypeError(f"Type '{dataframe.__class__}' is not supported.")

        instance = cls(dataframe.write_json())
        instance.__raw__ = dataframe

        return instance
