"""Implement a type-inference algorithm.

This aims to simplify the insertion of data into an `InfoMander`, by avoiding the need
for the user to explicitly specify a type when adding a value.
"""

import datetime
import pathlib
from enum import StrEnum, auto
from typing import Any

import altair
import matplotlib.figure
import numpy
import pandas
import polars


class DashboardType(StrEnum):
    """Type used to customize the visualization of objects stored in an InfoMander."""

    ANY = auto()
    ARRAY = auto()
    BOOLEAN = auto()
    DATAFRAME = auto()
    DATE = auto()
    DATETIME = auto()
    FILE = auto()
    HTML = auto()
    IMAGE = auto()
    INTEGER = auto()
    MARKDOWN = auto()
    MATPLOTLIB = auto()
    NUMBER = auto()
    NUMPY_ARRAY = auto()
    STRING = auto()
    VEGA = auto()

    @staticmethod
    def infer(x: Any) -> "DashboardType":
        """Infer the type of `x`.

        Notes
        -----
        If no match can be found the output is `DashboardType.ANY`.
        Strings are interpreted as Markdown by default.
        In general it is difficult to detect HTML or an image when given only a string,
        so for now we never infer these two types.

        Examples
        --------
        >>> DashboardType.infer(3)
        <DashboardType.INTEGER: 'integer'>

        >>> DashboardType.infer(None)
        <DashboardType.ANY: 'any'>

        >>> DashboardType.infer((1, "b"))
        <DashboardType.ANY: 'any'>

        >>> DashboardType.infer("hello")
        <DashboardType.MARKDOWN: 'markdown'>
        """
        TYPE_TO_DASHBOARD_TYPE = {
            list: DashboardType.ARRAY,
            bool: DashboardType.BOOLEAN,
            pandas.DataFrame: DashboardType.DATAFRAME,
            polars.DataFrame: DashboardType.DATAFRAME,
            datetime.date: DashboardType.DATE,
            datetime.datetime: DashboardType.DATETIME,
            int: DashboardType.INTEGER,
            str: DashboardType.MARKDOWN,
            matplotlib.figure.Figure: DashboardType.MATPLOTLIB,
            float: DashboardType.NUMBER,
            numpy.ndarray: DashboardType.NUMPY_ARRAY,
            altair.vegalite.v5.api.Chart: DashboardType.VEGA,
        }

        # `Paths` are `PosixPath` or `WindowsPath` when instantiated
        if isinstance(x, pathlib.Path):
            return DashboardType.FILE

        # Exact match
        return TYPE_TO_DASHBOARD_TYPE.get(type(x), DashboardType.ANY)
