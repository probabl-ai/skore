from __future__ import annotations

from functools import cached_property

import pandas


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
