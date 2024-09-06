"""Implement a type-inference algorithm.

This aims to simplify the insertion of data into an `Store`, by avoiding the need
for the user to explicitly specify a type when adding a value.
"""

from __future__ import annotations

import datetime
import pathlib
from enum import StrEnum, auto
from typing import Any

import altair.vegalite.v5.schema.core
import matplotlib.figure
import numpy
import pandas
import polars
from sklearn.base import BaseEstimator


class DisplayType(StrEnum):
    """Type used to customize the visualization of objects stored in a `Store`."""

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
    MATPLOTLIB_FIGURE = auto()
    NUMBER = auto()
    NUMPY_ARRAY = auto()
    STRING = auto()
    VEGA = auto()
    SKLEARN_MODEL = auto()

    @staticmethod
    def infer(x: Any) -> DisplayType:
        """Infer the type of `x`.

        Notes
        -----
        If no match can be found the output is `DisplayType.ANY`.
        Strings are interpreted as Markdown by default.
        In general it is difficult to detect HTML or an image when given only a string,
        so for now we never infer these two types.

        Examples
        --------
        >>> DisplayType.infer(3)
        <DisplayType.INTEGER: 'integer'>

        >>> DisplayType.infer(None)
        <DisplayType.ANY: 'any'>

        >>> DisplayType.infer((1, "b"))
        <DisplayType.ANY: 'any'>

        >>> DisplayType.infer("hello")
        <DisplayType.MARKDOWN: 'markdown'>
        """
        TYPE_TO_DISPLAY_TYPE = {
            list: DisplayType.ARRAY,
            bool: DisplayType.BOOLEAN,
            pandas.DataFrame: DisplayType.DATAFRAME,
            polars.DataFrame: DisplayType.DATAFRAME,
            datetime.date: DisplayType.DATE,
            datetime.datetime: DisplayType.DATETIME,
            int: DisplayType.INTEGER,
            str: DisplayType.MARKDOWN,
            matplotlib.figure.Figure: DisplayType.MATPLOTLIB_FIGURE,
            float: DisplayType.NUMBER,
            numpy.ndarray: DisplayType.NUMPY_ARRAY,
        }

        if isinstance(x, altair.vegalite.v5.schema.core.TopLevelSpec):
            return DisplayType.VEGA

        # `Paths` are `PosixPath` or `WindowsPath` when instantiated
        if isinstance(x, pathlib.Path):
            return DisplayType.FILE

        if isinstance(x, BaseEstimator):
            return DisplayType.SKLEARN_MODEL

        # Exact match
        return TYPE_TO_DISPLAY_TYPE.get(type(x), DisplayType.ANY)
