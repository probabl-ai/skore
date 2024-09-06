"""Schema to define endpoint requirements in the API."""

import typing

import pydantic
import typing_extensions

from skore.api.schema.any import Any
from skore.api.schema.array import Array
from skore.api.schema.boolean import Boolean
from skore.api.schema.dataframe import DataFrame
from skore.api.schema.date import Date
from skore.api.schema.datetime import Datetime
from skore.api.schema.file import File
from skore.api.schema.html import HTML
from skore.api.schema.integer import Integer
from skore.api.schema.markdown import Markdown
from skore.api.schema.matplotlib_figure import MatplotlibFigure
from skore.api.schema.number import Number
from skore.api.schema.numpy_array import NumpyArray
from skore.api.schema.sklearn_model import SKLearnModel
from skore.api.schema.string import String
from skore.api.schema.vega import Vega
from skore.store.layout import Layout

__all__ = [
    "Any",
    "Array",
    "Boolean",
    "DataFrame",
    "Date",
    "Datetime",
    "File",
    "HTML",
    "Integer",
    "Markdown",
    "MatplotlibFigure",
    "Number",
    "NumpyArray",
    "Store",
    "String",
    "SKLearnModel",
    "Vega",
]


class Store(pydantic.BaseModel):
    """Highest schema to transfer key-value pairs from store to dashboard.

    Examples
    --------
    >>> Store(uri="/root", payload={"key": {"type": "integer", "data": 0}})
    Store(...)
    """

    __NAME__ = "schema:dashboard:v0"

    model_config = pydantic.ConfigDict(strict=True)

    version: typing.Literal[__NAME__] = pydantic.Field(__NAME__, alias="schema")
    uri: str
    payload: dict[
        str,
        typing_extensions.Annotated[
            typing.Union[
                Any,
                Array,
                Boolean,
                DataFrame,
                Date,
                Datetime,
                File,
                HTML,
                Integer,
                Markdown,
                MatplotlibFigure,
                Number,
                NumpyArray,
                String,
                SKLearnModel,
                Vega,
            ],
            pydantic.Field(discriminator="type"),
        ],
    ]
    layout: Layout = pydantic.Field(default=[])
